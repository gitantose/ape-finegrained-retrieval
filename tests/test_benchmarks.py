"""Independent metric examples, adapter joins and upstream score parity."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from ape import APEScore

from ape_retrieval.benchmarks.data import construct, json_lines
from ape_retrieval.benchmarks.encoding import segments, tokenizer_from_reference, encode, verify_shards
from ape_retrieval.benchmarks.retrieval import relevant_ranks, top_positions, load_vectors, retrieve
from ape_retrieval.benchmarks.scoring import block_scores, ExactScorer
from ape_retrieval.benchmarks.metrics import query_metrics, aggregate_metrics
from ape_retrieval.model import configure_reference
from ape_retrieval.benchmarks.runtime import run_context


class ContinuationTests(unittest.TestCase):
    def test_expired_budget_stays_expired_and_completion_mode_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"runs_dir": str(root), "storage_roots": [str(root)],
                      "limits": {"host_memory_bytes": 64 * 2**30, "cuda_allocated_bytes": 2 * 2**30,
                                 "storage_bytes": 2**30, "minimum_free_disk_bytes": 0}}
            budget = {"config": config, "started_epoch": 0, "deadline_epoch": 0}
            path = root / "budget.json"
            path.write_text(json.dumps(budget))
            with self.assertRaisesRegex(RuntimeError, "deadline expired"):
                with run_context(root / "run", config, 14400):
                    self.fail("An expired budget must not renew automatically")
            budget["deadline_epoch"] = None
            path.write_text(json.dumps(budget))
            with self.assertRaisesRegex(ValueError, "explicit recorded authorization"):
                with run_context(root / "run", config, None):
                    self.fail("Missing authorization must not remove the deadline")
            budget["continuation_mode"] = "until_complete"
            path.write_text(json.dumps(budget))
            with patch("torch.cuda.is_available", return_value=False):
                with run_context(root / "run", config, None) as should_stop:
                    self.assertFalse(should_stop())
            self.assertEqual(json.loads(path.read_text()), budget)


class MetricTests(unittest.TestCase):
    def test_single_multiple_missing_and_empty_retrieval(self):
        self.assertEqual(query_metrics([2])["MRR"], .5)
        self.assertEqual(query_metrics([1, 7])["Recall@5"], .5)
        self.assertEqual(query_metrics([1, 7])["Success@5"], 1)
        self.assertIsNone(query_metrics([]))
        self.assertEqual(query_metrics([None])["MRR"], 0)
        self.assertEqual(query_metrics([None])["Recall@20"], 0)
        report = aggregate_metrics([{"positive_ranks": []}, {"positive_ranks": [2]}])
        self.assertEqual(report["queries_without_qrels"], 1)
        self.assertEqual(report["metrics"]["MRR"], .5)
        with self.assertRaises(ValueError):
            query_metrics([1, 1])

    def test_stable_ties_and_duplicate_document_identity(self):
        scores = np.array([.7, 1., 1., 0.], dtype=np.float32)
        # Document IDs sort b,a,c,d: tied IDs a before c.
        priority = np.array([1, 0, 2, 3])
        self.assertEqual(relevant_ranks(scores, [1, 2, 0], priority), [1, 2, 3])
        np.testing.assert_array_equal(top_positions(scores, priority, 2), [1, 2])
        self.assertEqual(len(top_positions(np.array([]), np.array([]), 20)), 0)


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = configure_reference({"dtype": "float32", "tf32": False, "seed": 42, "cpu_threads": 4},
                                         os.environ.get("APE_SUITE_TEST_DEVICE", "cpu"))

    def test_all_rules_against_official_pairwise_with_long_documents(self):
        generator = np.random.default_rng(42)
        documents = [generator.normal(size=(n, 7)).astype(np.float32) for n in (1, 2, 5, 17)]
        documents.append(np.empty((0, 7), dtype=np.float32))
        for count in (0, 1, 2, 4):
            query = generator.normal(size=(count, 7)).astype(np.float32)
            actual = block_scores(query, documents, self.device)
            for index, doc in enumerate(documents):
                oracle = APEScore(query, doc)
                for name, method in [("HA", "hungarian"), ("MaxSim", "maxsim"), ("H2P", "h_min_to_p")]:
                    self.assertAlmostEqual(float(actual[name][index]), oracle.score(method), places=6)

    def test_bounded_scoring_and_exact_positive_ranks(self):
        documents = [np.eye(2, dtype=np.float32), np.array([[1., 0.]], dtype=np.float32), -np.eye(2, dtype=np.float32)]
        offsets = np.array([0, 2, 3, 5])
        query = np.eye(2, dtype=np.float32)
        scorer = ExactScorer(offsets, np.concatenate(documents), self.device, 2, 256 * 2**20)
        actual = {rule: value[0] for rule, value in scorer.score_batch([query], lambda: False).items()}
        np.testing.assert_array_equal(actual["H2P"], [1, 0, 0])
        np.testing.assert_array_equal(actual["HA"], [1, 1, 0])
        np.testing.assert_array_equal(actual["MaxSim"], [1, .75, 0])
        self.assertEqual(relevant_ranks(actual["HA"], [1], np.arange(3)), [2])
        self.assertIsNone(scorer.score_batch([query], lambda: True))

    def test_streaming_and_resident_batches_are_bitwise_equal(self):
        rng = np.random.default_rng(13)
        offsets = np.array([0, 1, 3, 8, 25])
        corpus = rng.normal(size=(25, 7)).astype(np.float32)
        queries = [rng.normal(size=(n, 7)).astype(np.float32) for n in (1, 2, 3, 5)]
        resident = ExactScorer(offsets, corpus, self.device, 8, 256 * 2**20)
        streaming = ExactScorer(offsets, corpus, self.device, 8, 0)
        expected = resident.score_batch(queries, lambda: False)
        actual = streaming.score_batch(queries, lambda: False)
        for rule in expected:
            np.testing.assert_array_equal(actual[rule], expected[rule])
            for index, query in enumerate(queries):
                single = resident.score_batch([query], lambda: False)
                np.testing.assert_array_equal(single[rule][0], expected[rule][index])

    def test_two_slot_optimal_assignment_with_collisions_ties_and_negatives(self):
        from ape_retrieval.benchmarks.scoring import reduce_scores
        from scipy.optimize import linear_sum_assignment
        rng = np.random.default_rng(7)
        for length in (1, 2, 7, 31):
            # Discrete values deliberately produce many ties and negative scores.
            sim = rng.integers(-8, 9, size=(2, 100 * length)).astype(np.float32) / 8
            offsets = np.arange(0, sim.shape[1] + 1, length)
            actual = reduce_scores(sim, offsets)["HA"]
            expected = []
            for start, stop in zip(offsets, offsets[1:]):
                matrix = sim[:, start:stop]
                row, col = linear_sum_assignment(-matrix)
                expected.append(matrix[row, col].mean())
            np.testing.assert_array_equal(actual, expected)


class AdapterTests(unittest.TestCase):
    def test_context_dedup_keeps_all_qrels_splits_and_invalid_offsets(self):
        row = {"id": "q", "context": "Rome is in Italy.", "question": "Where is Rome?",
               "answers": {"text": ["Italy"], "answer_start": [11]}}
        bad = {**row, "id": "bad", "answers": {"text": ["Italy"], "answer_start": [0]}}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            result = construct(directory, "squad", {}, {"train": [row, bad], "validation": [row]})
            corpus = list(json_lines(directory / "corpus.jsonl"))
            queries = list(json_lines(directory / "queries.jsonl"))
            self.assertEqual(len(corpus), 1)
            self.assertEqual(queries[0]["positive_ids"], queries[2]["positive_ids"])
            self.assertEqual(queries[1]["positive_ids"], [])
            self.assertEqual(result["counts"]["cross_split_shared_documents"], 1)
            self.assertEqual(result["counts"]["positive_mappings"], 2)
            self.assertEqual(len(list(json_lines(directory / "origins.jsonl"))), 3)

    def test_retired_benchmark_is_rejected_without_creating_data(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "retired"
            with self.assertRaisesRegex(ValueError, "Unsupported benchmark"):
                construct(directory, "unsupported", {}, {})
            self.assertFalse(directory.exists())
        config = json.loads(Path("configs/benchmarks.json").read_text())
        self.assertEqual(list(config["datasets"]), ["nq", "squad", "propsearch"])

    def test_propsearch_keeps_published_duplicate_positions(self):
        rows = {"candidates": [{"id": "s", "text": "Same"}, {"id": "d", "text": "Same"}],
                "queries": [{"id": "q", "text": "Same", "sentence_id": "s"}]}
        with tempfile.TemporaryDirectory() as temp:
            result = construct(Path(temp), "propsearch", {}, {"release": rows})
            self.assertEqual(result["counts"]["corpus_documents"], 2)
            self.assertNotIn("answers", next(json_lines(Path(temp) / "queries.jsonl")))


class SegmentationTests(unittest.TestCase):
    def test_original_offsets_and_no_overlong_segment(self):
        reference = json.loads(Path("configs/reference.json").read_text())
        tokenizer = tokenizer_from_reference(reference)
        short = "Rome is in Italy."
        self.assertEqual(segments(short, tokenizer, 128), [(0, len(short))])
        text = "London is a city. " * 60 + "astronomy " * 400
        spans = segments(text, tokenizer, 128)
        self.assertGreater(len(spans), 2)
        covered = np.zeros(len(text), dtype=bool)
        for start, stop in spans:
            self.assertLessEqual(len(tokenizer(text[start:stop])["input_ids"]), 128)
            covered[start:stop] = True
        self.assertTrue(all(covered[i] for i, char in enumerate(text) if not char.isspace()))


class FixtureEncoder:
    def tokenizer(self, texts, **kwargs):
        return {"length": [len(t) + 2 for t in texts]}

    def encode(self, texts, **kwargs):
        vectors, scores = [], []
        for text in texts:
            count = 1 + len(text) % 2
            array = np.zeros((count, 384), dtype=np.float32)
            array[:, len(text) % 384] = 1.
            vectors.append(array)
            scores.append(np.full(count, .9, dtype=np.float32))
        return vectors, scores


class EncodingRecoveryTests(unittest.TestCase):
    def test_ranking_resume_and_saved_metrics(self):
        reference = json.loads(Path("configs/reference.json").read_text())
        fixture = FixtureEncoder()
        fixture.tokenizer = tokenizer_from_reference(reference)
        fixture.requires_grad_ = lambda enabled: fixture
        config = {"device": "cpu", "segmentation": "punctuation-newline-offset-windows-v1",
                  "encoding_shard_documents": 2, "scoring_block_slots": 16384,
                  "ranking_checkpoint_queries": 1, "query_batch_size": 1, "ks": [1, 5, 10, 20]}
        rows = [{"id": str(i), "context": text, "question": "Where?", "answers": {"text": ["Rome"], "answer_start": [0]}}
                for i, text in enumerate(("Rome is in Italy.", "Rome is a city.", "Rome has a river."))]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            processed = root / "data"
            construct(processed, "squad", {}, {"test": rows})
            run = root / "run"
            with patch("ape_retrieval.benchmarks.encoding.load_model", return_value=fixture):
                encode(config, processed, reference, run / "encoding", lambda: False)
            def stop():
                progress = run / "retrieval/progress.json"
                return progress.exists() and json.loads(progress.read_text())["processed_queries"] >= 2
            partial = retrieve(config, processed, run, reference, stop)
            self.assertEqual(partial["status"], "stopped")
            self.assertEqual(partial["processed_queries"], 2)
            complete = retrieve(config, processed, run, reference, lambda: False)
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(complete["processed_queries"], 3)
            metrics = json.loads((run / "retrieval/metrics.json").read_text())
            self.assertAlmostEqual(metrics["by_split"]["test"]["HA"]["metrics"]["MRR"], (1 + .5 + 1/3) / 3)
            first = run / "retrieval" / complete["blocks"][0]["file"]
            first.write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "Invalid committed"):
                retrieve(config, processed, run, reference, lambda: False)

    def test_flat_vectors_recovery_identity_and_corruption(self):
        reference = json.loads(Path("configs/reference.json").read_text())
        fixture = FixtureEncoder()
        fixture.tokenizer = tokenizer_from_reference(reference)
        fixture.requires_grad_ = lambda enabled: fixture
        config = {"device": "cpu", "segmentation": "punctuation-newline-offset-windows-v1", "encoding_shard_documents": 2}
        rows = [{"id": str(i), "context": text, "question": "Where?", "answers": {"text": ["Rome"], "answer_start": [0]}}
                for i, text in enumerate(("Rome is in Italy.", "Rome is a city.", "Rome has a river."))]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            processed = root / "data"
            construct(processed, "squad", {}, {"test": rows})
            with patch("ape_retrieval.benchmarks.encoding.load_model", return_value=fixture):
                full = encode(config, processed, reference, root / "full", lambda: False)
                calls = []
                def stop():
                    calls.append(1)
                    return len(calls) >= 3
                stopped = encode(config, processed, reference, root / "resume", stop)
                self.assertEqual(stopped["status"], "stopped")
                # Uncommitted binary tail is recoverable without changing committed vectors.
                with (root / "resume/corpus.f32").open("ab") as stream:
                    stream.write(b"orphan")
                resumed = encode(config, processed, reference, root / "resume", lambda: False)
            self.assertEqual(resumed["status"], "complete")
            for split in ("corpus", "queries"):
                expected = load_vectors(root / "full", full, split)
                actual = load_vectors(root / "resume", resumed, split)
                self.assertEqual(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
                np.testing.assert_array_equal(actual[2], expected[2])
            with (root / "resume/corpus.f32").open("r+b") as stream:
                stream.write(b"bad!")
            with self.assertRaisesRegex(ValueError, "Invalid packed"):
                verify_shards(root / "resume", resumed)


if __name__ == "__main__":
    unittest.main()
