"""Frozen pooling, masking, exact ties, positional storage and recovery."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from ape.training.teacher import pool
from ape.propsearch.benchmark import _pooled_ranks

from ape_retrieval.bge import (MODES, budget, check_resources, encode_batch, encode_rows,
                               load_pretrained, memory_plan, validate_config)
from ape_retrieval.bge_evaluate import pooled_blocks
from ape_retrieval.encode import EncodingStore
from ape_retrieval.evaluate import EvaluationRun, EvaluationStopped
from ape_retrieval.model import configure_reference
from test_encoding import fixture_inputs, flattened


CONFIG = json.loads(Path('configs/bge.json').read_text())


class FakeTokenizer:
    def __init__(self):
        self.seen = []

    def __call__(self, texts, **kwargs):
        self.seen.extend(texts)
        return {"length": [len(t) + 2 for t in texts]}


def fake_batch(model, tokenizer, texts, device, config):
    vectors = np.zeros((len(texts), 384), dtype=np.float32)
    for i, text in enumerate(texts):
        vectors[i, len(text) % 384] = 1
    return {"mean": vectors, "cls": np.roll(vectors, 1, axis=1)}


class BGETests(unittest.TestCase):
    def test_upstream_pool_mask_and_normalization(self):
        hidden = torch.tensor([[[1., 2.], [3., 4.], [999., 999.]], [[2., 0.], [0., 2.], [2., 2.]]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        torch.testing.assert_close(pool(hidden, mask, 'cls'), hidden[:, 0])
        torch.testing.assert_close(pool(hidden, mask, 'mean'), torch.tensor([[2., 3.], [4/3, 4/3]]))
        for mode in MODES:
            torch.testing.assert_close(F.normalize(pool(hidden, mask, mode), dim=-1).norm(dim=-1), torch.ones(2))

    def test_frozen_config_plan_and_nonrenewing_budget(self):
        validate_config(CONFIG)
        for field, value in [('query_prefix', ''), ('tf32', True), ('batch_size', 4)]:
            with self.assertRaises(ValueError):
                validate_config({**CONFIG, field: value})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = budget(root, CONFIG)
            self.assertEqual(budget(root, CONFIG), original)
            with self.assertRaises(ValueError):
                budget(root, {**CONFIG, 'budget_seconds': 7201})
            with self.assertRaisesRegex(RuntimeError, 'host_peak'):
                check_resources(root, {**CONFIG, 'host_memory_budget_bytes': 1}, torch.device('cpu'))
        plan = memory_plan({'candidates': range(314483), 'queries': range(36337)}, CONFIG)
        self.assertLess(plan['scoring_tensor_estimate_bytes'], 2 * 2**30)
        self.assertEqual(plan['two_condition_raw_vector_bytes'], 1077719040)

    def test_prefix_order_duplicates_recovery_and_corruption(self):
        inputs = fixture_inputs()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {**CONFIG, 'shard_rows': 4}
            def stores(name):
                return {mode: EncodingStore(root / name / mode, root / 'data',
                    {'reference': CONFIG, 'pooling': mode}, inputs, 7200, 4 * 2**30) for mode in MODES}
            full, resumed = stores('full'), stores('resume')
            tokenizer = FakeTokenizer()
            with patch('ape_retrieval.bge.encode_batch', side_effect=fake_batch):
                encode_rows(None, tokenizer, inputs, full, config, torch.device('cpu'), lambda: False, lambda: None)
                count = []
                def stopping():
                    count.append(1)
                    return len(count) > 2
                encode_rows(None, FakeTokenizer(), inputs, resumed, config, torch.device('cpu'), stopping, lambda: None)
                self.assertEqual(resumed['mean'].progress['processed']['candidates'], 4)
                # Simulate a crash between the independent mean and CLS commits.
                prior = resumed['cls'].progress['chunks'].pop()
                resumed['cls'].progress['processed']['candidates'] = prior['start']
                resumed['cls'].checkpoint('stopped')
                resumed = stores('resume')
                encode_rows(None, FakeTokenizer(), inputs, resumed, config, torch.device('cpu'), lambda: False, lambda: None)
            self.assertEqual(tokenizer.seen[:len(inputs['candidates'])], [r['text'] for r in inputs['candidates']])
            self.assertEqual(tokenizer.seen[len(inputs['candidates']):], ['query: ' + r['text'] for r in inputs['queries']])
            for mode in MODES:
                resumed[mode].verify()
                for split in inputs:
                    a, b = flattened(full[mode], split), flattened(resumed[mode], split)
                    for key in a:
                        np.testing.assert_array_equal(a[key], b[key])
                a = flattened(full[mode], 'candidates')
                np.testing.assert_array_equal(a['embeddings'][0], a['embeddings'][2])
                self.assertEqual(a['ids'].tolist(), [r['id'] for r in inputs['candidates']])
            path = resumed['mean'].directory / resumed['mean'].progress['chunks'][0]['file']
            path.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                resumed['mean'].verify()

    def test_exact_ranks_ties_and_interrupted_evaluation(self):
        device = configure_reference(CONFIG, os.environ.get('BGE_TEST_DEVICE', 'cpu'))
        vectors = torch.eye(3)
        caches = {'candidates': {'a': vectors[0:1], 'dup': vectors[0:1], 'b': vectors[1:2]},
                  'queries': {'q0': vectors[0:1], 'q1': vectors[1:2], 'q2': vectors[2:3]}}
        inputs = {'candidates': [{'id': k} for k in caches['candidates']],
                  'queries': [{'id': f'q{i}', 'gold_position': gold, 'source_n_atoms': i+2}
                              for i, gold in enumerate([2, 2, 0])]}
        with tempfile.TemporaryDirectory() as temp:
            identity = {'evaluation': {**CONFIG, 'query_chunk_size': 2, 'checkpoint_queries': 2}}
            def new(name):
                return EvaluationRun(Path(temp) / name, identity, inputs['queries'], 3)
            full, resumed = new('full'), new('resume')
            pooled_blocks(caches, inputs, full, device, lambda: {})
            np.testing.assert_array_equal(full.verify(), [3, 1, 1])
            calls = []
            def interrupted(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise EvaluationStopped('fixture')
                return _pooled_ranks(*args, **kwargs)
            with patch('ape_retrieval.bge_evaluate._pooled_ranks', side_effect=interrupted):
                with self.assertRaises(EvaluationStopped):
                    pooled_blocks(caches, inputs, resumed, device, lambda: {})
            self.assertEqual(resumed.progress['processed_queries'], 2)
            resumed = new('resume')
            pooled_blocks(caches, inputs, resumed, device, lambda: {})
            np.testing.assert_array_equal(resumed.verify(), full.verify())

    @unittest.skipUnless(os.environ.get('BGE_REAL_TEST') == '1', 'opt-in real pretrained checkpoint')
    def test_real_cuda_shared_forward_pooling_and_truncation(self):
        device = configure_reference(CONFIG, 'cuda')
        model, tokenizer = load_pretrained(Path('runs/bge-frozen'), CONFIG, device)
        texts = ['Rome is the capital of Italy.', 'query: ' + 'word ' * 140]
        actual = encode_batch(model, tokenizer, texts, device, CONFIG)
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        self.assertEqual(tokens.input_ids.shape[1], 128)
        lengths = tokenizer(texts, truncation=False, return_length=True)['length']
        self.assertGreater(lengths[1], 128)
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            mask = tokens.attention_mask[..., None]
            expected = {'mean': (hidden * mask).sum(1) / mask.sum(1), 'cls': hidden[:, 0]}
            for mode in MODES:
                np.testing.assert_allclose(actual[mode], F.normalize(expected[mode], dim=-1).cpu().numpy(), atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(np.linalg.norm(actual[mode], axis=-1), 1., atol=1e-6)
        inputs = fixture_inputs()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def stores(name):
                return {mode: EncodingStore(root / name / mode, root / 'data',
                    {'reference': CONFIG, 'pooling': mode}, inputs, 7200, 4 * 2**30) for mode in MODES}
            full, resumed = stores('full'), stores('resume')
            encode_rows(model, tokenizer, inputs, full, CONFIG, device, lambda: False, lambda: None)
            calls = []
            def stopping():
                calls.append(1)
                return len(calls) > 2
            encode_rows(model, tokenizer, inputs, resumed, CONFIG, device, stopping, lambda: None)
            resumed = stores('resume')
            encode_rows(model, tokenizer, inputs, resumed, CONFIG, device, lambda: False, lambda: None)
            for mode in MODES:
                resumed[mode].verify()
                for split in inputs:
                    a, b = flattened(full[mode], split), flattened(resumed[mode], split)
                    for key in a:
                        np.testing.assert_array_equal(a[key], b[key])
        # The same tokenizer adds two prefix tokens at the overflow boundary.
        boundary = 'word ' * 125
        self.assertEqual(tokenizer(boundary, return_length=True)['length'][0], 127)
        self.assertEqual(tokenizer('query: ' + boundary, return_length=True)['length'][0], 129)
        print(json.dumps({'device': str(device), 'weight_revision': CONFIG['model_revision'],
                          'token_lengths': lengths, 'peak_allocated': torch.cuda.max_memory_allocated()}))


if __name__ == '__main__':
    unittest.main()
