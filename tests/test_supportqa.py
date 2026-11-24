"""Known-answer and recovery tests for the full SupportQA experiment."""

import json
from pathlib import Path
import tempfile
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ape_retrieval.benchmarks.data import json_lines
from ape_retrieval.benchmarks.dense import encode_index, encode_texts, open_vectors, rank_dataset
from ape_retrieval.benchmarks.encoding import tokenizer_from_reference
from ape_retrieval.supportqa import select, text_hash, write_rows
from ape_retrieval.supportqa.supportqa_annotations import compact_prompt_batch, normalize_response, restore_review_ids, QUALITY_AXES
from ape_retrieval.supportqa import supportqa_annotations
from ape_retrieval.supportqa.supportqa_evaluation import predict_ape, rank_support


class FullSelectionTests(unittest.TestCase):
    def test_global_precedence_cap_and_native_row_accounting(self):
        reference = 'configs/reference.json'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {'nq': ['Rome', 'Rome', 'Rome', 'Rome prior', 'Rome long'],
                     'squad': ['Rome', 'Rome squad']}
            config = {'reference': reference, 'datasets': {}}
            for name, split in [('nq', 'test'), ('squad', 'validation')]:
                path = root / f'{name}.json'
                write_rows(path, [{'context': text, 'question': f'Native {i}',
                                   'answers': {'text': ['Rome'], 'answer_start': [0]}}
                                  for i, text in enumerate(files[name])])
                config['datasets'][name] = {'raw_dir': str(root), 'revision': 'fixed', 'splits': {split: [path.name]}}

            def tokenizer(text):
                return {'input_ids': [0] * (129 if text.endswith('long') else 5)}

            output = root / 'out'
            manifest = select(config, output, tokenizer, quota=None, excluded_hashes={text_hash('Rome prior')})
            self.assertTrue(manifest['complete_quota'])
            self.assertEqual([q['query'] for q in json_lines(output / 'queries.jsonl')], ['Native 0', 'Native 1', 'Native 1'])
            rows = [r for r in json_lines(output / 'exclusions.jsonl') if r.get('query_id')]
            self.assertEqual({r['reason'] for r in rows}, {'question_cap', 'excluded_reference_context',
                             'context_over_128_tokens', 'already_selected_other_dataset'})
            self.assertEqual(len(rows), 4)


class SupportEvidenceTests(unittest.TestCase):
    def test_codex_response_id_membership_and_consistent_failure_axes(self):
        source = {'document_id': 'd1', 'document_sha256': 'doc', 'decomposition_version': 'raw',
                  'format_valid': True, 'eos_reached': True, 'token_cap_hit': False,
                  'propositions': [{'id': 'p001'}], 'previous_quality_review': None,
                  'queries': [{'query_id': 'q', 'query': 'Which city?', 'answers': ['Rome']}]}

        def result(answerability, mapping, supports):
            return {'contexts': [{'document_id': 'd1', 'quality_assessment': {k: 'Reviewed' for k in QUALITY_AXES},
                                  'quality_issues': [], 'quality_notes': '', 'queries': [{
                                      'query_id': 'q', 'document_answerability': answerability, 'mapping_status': mapping,
                                      'answerability_rationale': 'Source assessment', 'support_rationale': 'Set assessment',
                                      'minimality_notes': 'Alternative assessment', 'support_sets': supports}]}]}

        valid = normalize_response(result('answerable', 'supported', [['p001']]), [source])
        self.assertEqual(valid[0]['queries']['q']['native_answers'], ['Rome'])
        evidence, aliases = compact_prompt_batch([source])
        self.assertEqual(evidence[0]['document_id'], 'c01')
        coded = result('answerable', 'supported', [['p001']])
        coded['contexts'][0]['document_id'] = 'c01'
        coded['contexts'][0]['queries'][0]['query_id'] = 'q1'
        self.assertEqual(normalize_response(restore_review_ids(coded, aliases), [source]), valid)
        for invalid in [result('insufficient_document', 'decomposition_error', []),
                        result('answerable', 'supported', [['p999']]), {'contexts': []}]:
            with self.assertRaises(ValueError):
                normalize_response(invalid, [source])

    def test_ape_scores_document_slots_and_keeps_unmapped_required_members(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / 'diagnostics').mkdir()
            (output / 'encodings').mkdir()
            case = {'document_id': 'd:doc', 'decomposition_version': 'dec', 'support_sets': [['p1', 'p2'], ['p3']]}
            (output / 'diagnostics' / 'doc.json').write_text(json.dumps({
                'encoding_file': 'encodings/doc.json', 'decomposition_output_sha256': 'dec',
                'matching': {'status': 'provisional_teacher_space', 'reference_ids': ['p1', 'p2', 'p3'],
                             'pairs': [{'reference_id': 'p1', 'slot_id': 4}, {'reference_id': 'p3', 'slot_id': 2}]}}))
            (output / 'encodings' / 'doc.json').write_text(json.dumps({'vectors': [[1, 0], [0, 1]], 'slot_ids': [4, 2]}))
            question = 'original question'
            (output / 'encodings' / (text_hash(question) + '.json')).write_text(json.dumps({'vectors': [[1, 0], [-1, 0]]}))
            result = predict_ape(output, {'query': question}, case, [1, 5])
            self.assertEqual(result['ranked_ids'], [4, 2])
            alternatives = result['metrics']['alternatives']
            self.assertEqual(alternatives[0]['unmapped_support_ids'], ['p2'])
            self.assertEqual(alternatives[0]['metrics']['Recall@5'], .5)
            self.assertEqual(result['metrics']['best_sufficient_set_per_metric']['all_required_Success@5'], 1)
            self.assertEqual(rank_support([5, 2], np.asarray([1., 1.], dtype=np.float32)), [2, 5])


class ConcurrentReviewTests(unittest.TestCase):
    def test_quota_drains_returned_results_and_preserves_unfinished_batches(self):
        docs = {name: {} for name in ('one', 'two', 'three', 'four')}
        queries = {name: [{}] for name in docs}
        barrier = Barrier(3, timeout=5)

        def pending(_output, available, *_args):
            return [{'document_id': name, 'document_sha256': name, 'queries': {name: {}}}
                    for name in list(available)[:_args[-1]]]

        def review(_protocol, batch, *_args):
            barrier.wait()
            if batch[0]['document_id'] == 'one':
                raise RuntimeError('Codex usage limit. Try again at Nov 20th, 2026 2:58 AM.')
            return {'entries': batch, 'metadata': {'model': 'gpt-6.2-sol'}}

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            protocol = output / 'protocol.md'
            protocol.write_text('fixture')
            (output / 'status.json').write_text('{"phases":{"decompose":{"status":"COMPLETED"}}}')
            login = SimpleNamespace(returncode=0, stdout='Logged in using ChatGPT', stderr='')
            with patch.object(supportqa_annotations.subprocess, 'run', return_value=login), \
                    patch.object(supportqa_annotations, 'selected_rows', return_value=(docs, queries)), \
                    patch.object(supportqa_annotations, 'review_records', return_value=({}, {})), \
                    patch.object(supportqa_annotations, 'pending_batch', side_effect=pending), \
                    patch.object(supportqa_annotations, 'call_codex', side_effect=review) as cli:
                limits = {'host_memory_bytes': 6 * 2**30, 'storage_bytes': 6 * 2**30, 'minimum_free_disk_bytes': 0}
                supportqa_annotations.run_codex_reviews({'limits': limits}, output, protocol, batch_size=1, workers=3)
            self.assertEqual(cli.call_count, 3)
            state = json.loads((output / 'annotation-process.json').read_text())
            self.assertEqual((state['status'], state['quota_retry_time_displayed']), ('BLOCKED', 'Nov 20th, 2026 2:58 AM'))
            batches = [json.loads(path.read_text()) for path in (output / 'semantic-batches').glob('*.json')]
            self.assertEqual({entry['document_id'] for batch in batches for entry in batch['contexts']}, {'two', 'three'})


class DenseBaselineTests(unittest.TestCase):
    def test_pooling_cosine_dot_and_segmentation(self):
        ref = json.loads(Path('configs/reference.json').read_text())
        tokenizer = tokenizer_from_reference(ref)

        def model(**inputs):
            ids = inputs['input_ids'].float()
            return SimpleNamespace(last_hidden_state=torch.stack((ids, torch.ones_like(ids)), dim=-1))

        spec = {'score': 'cosine', 'pooling': 'cls_before_pooler', 'max_tokens': 8}
        texts = ['Rome', 'Paris is in France. Rome is in Italy. The cities are capitals.']
        values, offsets = encode_texts(model, tokenizer, texts, spec, 3, torch.device('cpu'))
        self.assertGreater(len(offsets[1]), 1)
        self.assertEqual(''.join(texts[1][a:b] for a, b in offsets[1]), texts[1])
        cls = np.asarray([tokenizer.cls_token_id, 1], dtype=np.float32)
        np.testing.assert_allclose(values, np.tile(cls / np.linalg.norm(cls), (2, 1)), atol=1e-6)
        spec.update(score='dot_product', pooling='attention_masked_mean')
        values, offsets = encode_texts(model, tokenizer, texts, spec, 3, torch.device('cpu'))
        expected = [np.mean([np.mean(tokenizer(text[a:b])['input_ids']) for a, b in spans])
                    for text, spans in zip(texts, offsets)]
        np.testing.assert_allclose(values[:, 0], expected, rtol=1e-6)
        np.testing.assert_allclose(values[:, 1], 1)

    def test_dense_encoding_resume_discards_only_uncommitted_tail(self):
        config = {'reference': 'configs/reference.json',
                  'baselines': {'bge': {'score': 'cosine', 'dimension': 384, 'max_tokens': 512}},
                  'retrieval': {'encoding_shard_rows': 2, 'encoding_batch_size': 2}}
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=384, max_position_embeddings=512))

        def fake_encode(_model, _tokenizer, texts, _spec, _batch_size, _device):
            vectors = np.zeros((len(texts), 384), dtype=np.float32)
            for index, text in enumerate(texts):
                vectors[index, ord(text)] = 1
            return vectors, [[(0, len(t))] for t in texts]

        with tempfile.TemporaryDirectory() as temp, patch('ape_retrieval.benchmarks.dense.configure_reference'), \
                patch('ape_retrieval.benchmarks.dense.load_encoder', return_value=(model, None)), \
                patch('ape_retrieval.benchmarks.dense.encode_texts', side_effect=fake_encode) as encode:
            root = Path(temp)
            (root / 'files.json').write_text('{}')
            directory = root / 'encoding'
            stops = iter([False, True, True])
            with self.assertRaises(InterruptedError):
                encode_index(config, directory, list('abc'), {}, 'bge', root, lambda force=False: next(stops))
            with (directory / 'vectors.f32').open('ab') as stream:
                stream.write(b'partial')
            encode.reset_mock()
            encode_index(config, directory, list('abc'), {}, 'bge', root, lambda force=False: False)
            self.assertEqual(encode.call_args.args[2], ['c'])
            vectors = open_vectors(directory)
            self.assertEqual(vectors.shape, (3, 384))
            np.testing.assert_array_equal(vectors.argmax(axis=1), [97, 98, 99])
            with self.assertRaisesRegex(ValueError, 'Resume identity changed'):
                encode_index(config, directory, list('abd'), {}, 'bge', root, lambda force=False: False)

    def test_exact_full_rank_ties_and_native_ineligibility(self):
        config = {'baselines': {'contriever': {'score': 'dot_product'}}, 'retrieval': {
            'ks': [1, 5, 10, 20], 'scoring_query_batch_size': 2, 'ranking_checkpoint_queries': 1}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'manifest.json').write_text('{}')
            write_rows(root / 'queries.jsonl', [{'query_id': 'q1', 'split': 'test', 'positive_ids': ['z']},
                                                 {'query_id': 'q2', 'split': 'test', 'positive_ids': []}])
            ids = {'corpus': {'z': 0, 'a': 0, 'b': 1}, 'queries': {'q1': 0, 'q2': 1}}
            scores = np.array([[1, 0], [0, 1]], dtype=np.float32)
            args = (config, root, root, 'nq', 'contriever', scores, ids, {}, torch.device('cpu'))
            rank_dataset(*args, lambda force=False: False)
            row = json.loads((root / 'nq' / 'metrics.json').read_text())['by_split']['test']['dot_product']
            self.assertEqual(row['eligible_queries'], 1)
            self.assertEqual(row['metrics']['MRR'], .5)
            self.assertEqual(row['metrics']['Recall@1'], 0)
            prediction = next(json_lines(root / 'nq' / 'queries-000000000.jsonl.gz'))
            self.assertEqual(prediction['aggregations']['dot_product']['top_corpus_indices'], [1, 0, 2])
            rank_dataset(*args, lambda force=False: self.fail('Complete rank cache should return without scoring'))
