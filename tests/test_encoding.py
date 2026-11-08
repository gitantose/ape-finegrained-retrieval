"""Ordering, durable shards and interrupted/resumed encoding consistency."""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from ape_retrieval.encode import EncodingStore, atomic_json, encode_split
from ape_retrieval.propsearch import join_inputs


REFERENCE = json.loads(Path("configs/reference.json").read_text())


def fixture_inputs():
    sources = [{"id": "s0", "text": "Rome is the capital of Italy.", "n_atoms": 1},
               {"id": "s1", "text": "Rome was built in 753 BC and is now the capital of Italy.", "n_atoms": 2}]
    distractors = [{"id": "d0", "text": sources[0]["text"]},
                   {"id": "d1", "text": "A cat sleeps. " * 60},
                   {"id": "d2", "text": ""},
                   {"id": "d3", "text": "Paris is in France."},
                   {"id": "d4", "text": "Paris is in France."}]
    queries = [{"id": "q0", "atom": sources[0]["text"], "sentence_id": "s0"},
               {"id": "q1", "atom": "Rome was built in 753 BC.", "sentence_id": "s1"}]
    return join_inputs(sources, distractors, queries)


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


def flattened(store, split):
    values = {name: [] for name in ["embeddings", "activities", "ids", "token_lengths"]}
    for chunk in store.progress["chunks"]:
        if chunk["split"] == split:
            with np.load(store.directory / chunk["file"], allow_pickle=False) as arrays:
                for name in values:
                    values[name].append(arrays[name])
    return {name: np.concatenate(parts) for name, parts in values.items()}


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inputs = fixture_inputs()
        self.identity = {"reference": REFERENCE}

    def tearDown(self):
        self.temporary.cleanup()

    def store(self, name, identity=None):
        return EncodingStore(self.root / name, self.root / "data", identity or self.identity,
                             self.inputs, budget_seconds=14400, disk_budget_bytes=8 * 2**30)

    def encode_all(self, encoder, store):
        for split, rows in self.inputs.items():
            encode_split(encoder, rows, split, store, REFERENCE, 4, lambda: False)
        store.verify()

    def test_release_order_duplicates_and_qrels(self):
        candidates, queries = self.inputs["candidates"], self.inputs["queries"]
        self.assertEqual([r["id"] for r in candidates], ["s0", "s1", "d0", "d1", "d2", "d3", "d4"])
        self.assertEqual(candidates[0]["text"], candidates[2]["text"])
        self.assertEqual([r["gold_position"] for r in queries], [0, 1])
        self.assertEqual([r["sentence_id"] for r in queries], ["s0", "s1"])

    def check_resume_consistency(self, encoder):
        full, resumed = self.store("full"), self.store("resumed")
        self.encode_all(encoder, full)
        calls = []

        def stop_after_two_microbatches():
            calls.append(1)
            return len(calls) > 2

        encode_split(encoder, self.inputs["candidates"], "candidates", resumed,
                     REFERENCE, 512, stop_after_two_microbatches)
        self.assertEqual(resumed.progress["processed"]["candidates"], 4)
        deadline = resumed.manifest["deadline_epoch"]
        resumed = self.store("resumed")
        self.assertEqual(resumed.manifest["deadline_epoch"], deadline)
        resumed.verify()
        self.encode_all(encoder, resumed)
        for split in self.inputs:
            expected, actual = flattened(full, split), flattened(resumed, split)
            for name in expected:
                if expected[name].dtype.kind == "f":
                    np.testing.assert_allclose(actual[name], expected[name], atol=2e-5, rtol=2e-5)
                else:
                    np.testing.assert_array_equal(actual[name], expected[name])

    def test_interrupted_resumed_consistency(self):
        self.check_resume_consistency(FixtureEncoder())

    def test_model_failure_preserves_completed_microbatches(self):
        class FailingEncoder(FixtureEncoder):
            calls = 0

            def encode(self, texts, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("injected model failure")
                return super().encode(texts, **kwargs)

        store = self.store("failure")
        with self.assertRaisesRegex(RuntimeError, "injected model failure"):
            encode_split(FailingEncoder(), self.inputs["candidates"], "candidates", store,
                         REFERENCE, 512, lambda: False)
        self.assertEqual(store.progress["processed"]["candidates"], 2)
        recovered = self.store("failure")
        recovered.verify()
        self.encode_all(FixtureEncoder(), recovered)
        np.testing.assert_array_equal(flattened(recovered, "candidates")["ids"],
                                      [r["id"] for r in self.inputs["candidates"]])

    def test_deadline_does_not_reset_and_disk_cap_rejects_write(self):
        store = EncodingStore(self.root / "expired", self.root / "data", self.identity,
                              self.inputs, budget_seconds=0, disk_budget_bytes=8 * 2**30)
        deadline = store.manifest["deadline_epoch"]
        encode_split(FixtureEncoder(), self.inputs["candidates"], "candidates", store,
                     REFERENCE, 4, lambda: time.time() >= deadline)
        self.assertEqual(store.progress["processed"]["candidates"], 0)
        resumed = self.store("expired")
        self.assertEqual(resumed.manifest["deadline_epoch"], deadline)
        limited = EncodingStore(self.root / "limited", self.root / "data", self.identity,
                                self.inputs, budget_seconds=14400, disk_budget_bytes=1048576)
        with self.assertRaisesRegex(RuntimeError, "disk budget"):
            self.encode_all(FixtureEncoder(), limited)
        self.assertEqual(limited.progress["processed"]["candidates"], 0)
        self.assertEqual(list(limited.shards.glob("*.npz")), [])

    def test_corruption_identity_and_orphan_commit(self):
        store = self.store("recovery")
        old_progress = json.loads(json.dumps(store.progress))
        self.encode_all(FixtureEncoder(), store)
        # Simulate death after shard rename but before the progress commit.
        atomic_json(store.directory / "progress.json", old_progress)
        recovered = self.store("recovery")
        self.encode_all(FixtureEncoder(), recovered)
        self.assertEqual(recovered.progress["processed"], {"candidates": 7, "queries": 2})
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.store("recovery", {"reference": {**REFERENCE, "max_seq_len": 64}})
        path = recovered.directory / recovered.progress["chunks"][0]["file"]
        with path.open("r+b") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            recovered.verify()

    @unittest.skipUnless(os.environ.get("APE_ENCODING_CUDA_TEST") == "1", "explicit real CUDA fixture")
    def test_real_cuda_interrupted_resumed_consistency(self):
        from ape_retrieval.model import configure_reference, load_model
        model = load_model(REFERENCE, configure_reference(REFERENCE, "cuda"))
        self.check_resume_consistency(model)


if __name__ == "__main__":
    unittest.main()
