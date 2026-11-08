"""Exact upstream ranks, metric definitions and recoverable query blocks."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ape import APEScore
from ape.propsearch.benchmark import _multi_vector_ranks, _rank

from ape_retrieval.encode import EncodingStore, atomic_json, encode_split
from ape_retrieval.evaluate import EvaluationRun, EvaluationStopped, evaluate_blocks, load_caches, metric_aggregates
from ape_retrieval.model import configure_reference
from test_encoding import FixtureEncoder, fixture_inputs


def scoring_fixture():
    e1, e2, e3 = torch.eye(3)
    candidates = {"a": torch.stack([e1, e2]), "partial": e1[None],
                  "duplicate": torch.stack([e2, e1]), "other": torch.stack([e1, e3])}
    queries = {"q0": torch.stack([e1, e2]), "q1": e1[None],
               "q2": e2[None], "q3": torch.stack([e2, e1])}
    rows = [{"id": qid, "gold_position": gold, "source_n_atoms": complexity}
            for qid, gold, complexity in zip(queries, [1, 0, 3, 0], [1, 2, 3, 6])]
    return {"candidates": candidates, "queries": queries}, {
        "candidates": [{"id": key} for key in candidates], "queries": rows}


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = configure_reference(json.loads(Path("configs/reference.json").read_text()),
                                         os.environ.get("APE_EVALUATION_TEST_DEVICE", "cpu"))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.caches, self.inputs = scoring_fixture()
        config = json.loads(Path("configs/propsearch-evaluation.json").read_text())
        self.identity = {"evaluation": {**config, "query_chunk_size": 2, "checkpoint_queries": 2}}

    def tearDown(self):
        self.temporary.cleanup()

    def new_run(self, name):
        return EvaluationRun(self.root / name, self.identity, self.inputs["queries"], 4)

    def test_exact_upstream_ranks_match_pairwise_oracle(self):
        query_ids = list(self.caches["queries"])
        candidate_ids = list(self.caches["candidates"])
        scores = torch.tensor([[APEScore(self.caches["queries"][q], self.caches["candidates"][c]).h_min_to_p()
                                for c in candidate_ids] for q in query_ids])
        expected = _rank(scores, torch.tensor([1, 0, 3, 0])).numpy()
        np.testing.assert_array_equal(expected, [3, 1, 3, 1])
        for block_size, chunk_size in [(2, 2), (4, 1), (8, 8)]:
            with self.subTest(checkpoint_queries=block_size, query_chunk_size=chunk_size):
                self.identity["evaluation"].update(checkpoint_queries=block_size, query_chunk_size=chunk_size)
                run = self.new_run(f"oracle-{block_size}-{chunk_size}")
                evaluate_blocks(self.caches, self.inputs, run, self.device)
                np.testing.assert_array_equal(run.verify(), expected)
        self.assertEqual(APEScore(self.caches["queries"]["q0"], self.caches["candidates"]["partial"]).h_min_to_p(), 0.)
        self.assertEqual(APEScore(self.caches["candidates"]["partial"], self.caches["queries"]["q0"]).h_min_to_p(), 1.)

    def test_distinct_aggregates_units_and_missing_bucket(self):
        ranks = np.array([100, 1, 1, 2, 4, 5, 10], dtype=np.int64)
        complexities = np.array([1, 2, 2, 3, 4, 5, 16])
        metrics = metric_aggregates(ranks, complexities)
        self.assertAlmostEqual(metrics["overall_query_mean"]["metrics"]["MRR"], np.mean(1. / ranks))
        macro = metrics["paper_table4_macro_2_3_4_5_6plus"]["metrics"]
        self.assertAlmostEqual(macro["MRR"], (1 + .5 + .25 + .2 + .1) / 5)
        self.assertEqual(macro["R@1"], 20.)
        self.assertAlmostEqual(metrics["overall_query_mean"]["metrics"]["R@1"], 200 / 7)
        self.assertEqual(metrics["by_source_n_atoms"]["1"]["metrics"]["MRR"], .01)
        with self.assertRaisesRegex(ValueError, "missing buckets"):
            metric_aggregates(ranks[:-1], complexities[:-1])

    def test_interrupted_block_resumes_without_partial_ranks(self):
        uninterrupted = self.new_run("full")
        evaluate_blocks(self.caches, self.inputs, uninterrupted, self.device)
        partial = self.new_run("partial")
        calls = []

        def interrupted(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise EvaluationStopped("injected stop inside upstream scoring")
            return _multi_vector_ranks(*args, **kwargs)

        with patch("ape_retrieval.evaluate._multi_vector_ranks", side_effect=interrupted):
            with self.assertRaises(EvaluationStopped):
                evaluate_blocks(self.caches, self.inputs, partial, self.device)
        self.assertEqual(partial.progress["processed_queries"], 2)
        resumed = self.new_run("partial")
        self.assertEqual(partial.manifest["deadline_epoch"], resumed.manifest["deadline_epoch"])
        evaluate_blocks(self.caches, self.inputs, resumed, self.device)
        np.testing.assert_array_equal(resumed.verify(), uninterrupted.verify())

    def test_orphan_corruption_and_resume_identity(self):
        run = self.new_run("recovery")
        run.append(np.array([3, 1], dtype=np.int64))
        previous = json.loads(json.dumps(run.progress))
        run.append(np.array([3, 1], dtype=np.int64))
        atomic_json(run.directory / "progress.json", previous)
        resumed = self.new_run("recovery")
        np.testing.assert_array_equal(resumed.verify(), [3, 1, 0, 0])
        resumed.append(np.array([3, 1], dtype=np.int64))
        np.testing.assert_array_equal(resumed.verify(), [3, 1, 3, 1])
        with self.assertRaisesRegex(ValueError, "identity changed"):
            EvaluationRun(resumed.directory, {"evaluation": {**self.identity["evaluation"], "query_chunk_size": 1}},
                          self.inputs["queries"], 4)
        path = resumed.directory / resumed.progress["chunks"][0]["file"]
        with path.open("r+b") as stream:
            stream.write(b"broken")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            resumed.verify()

    def test_complete_requires_full_coverage_and_final_artifacts(self):
        run = self.new_run("incomplete")
        run.append(np.array([3, 1], dtype=np.int64))
        run.checkpoint("complete")
        with self.assertRaisesRegex(ValueError, "full query coverage"):
            run.verify()
        run.append(np.array([3, 1], dtype=np.int64))
        run.checkpoint("complete")
        with self.assertRaisesRegex(ValueError, "final artifacts"):
            run.verify()

    def test_loader_retains_duplicate_candidate_positions(self):
        inputs = fixture_inputs()
        reference = json.loads(Path("configs/reference.json").read_text())
        store = EncodingStore(self.root / "loader", self.root / "data", {"reference": reference},
                              inputs, budget_seconds=60, disk_budget_bytes=8 * 2**30)
        for split, rows in inputs.items():
            encode_split(FixtureEncoder(), rows, split, store, reference, 4, lambda: False)
        caches = load_caches(store.directory, store.progress["chunks"])
        self.assertEqual(list(caches["candidates"]), [row["id"] for row in inputs["candidates"]])
        torch.testing.assert_close(caches["candidates"]["s0"], caches["candidates"]["d0"])
        self.assertNotEqual(caches["candidates"]["s0"].data_ptr(), caches["candidates"]["d0"].data_ptr())


if __name__ == "__main__":
    unittest.main()
