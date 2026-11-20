"""SupportQA provenance, boundary failures and exact query-independent diagnostics."""
import json
from pathlib import Path
import tempfile
import unittest

from ape_retrieval.supportqa import parse_decomposition, select, text_hash, selection_order


class SupportQADataTests(unittest.TestCase):
    def test_whole_context_selection_and_original_questions(self):
        class Tokenizer:
            def __call__(self, text):
                return {'input_ids': [1] * (129 if text.startswith('LONG') else 128)}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {'reference': 'configs/reference.json', 'datasets': {}}
            for name in ('nq', 'squad'):
                raw = root / name
                raw.mkdir()
                rows = [{'context': f'Rome {name} {i}', 'question': f'original question {i}',
                         'answers': {'text': ['Rome'], 'answer_start': [0]}}
                        for i in range(27)]
                rows.append({'context': 'LONG Rome', 'question': 'kept?',
                             'answers': {'text': ['Rome'], 'answer_start': [5]}})
                path = raw / 'rows.json'
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                split = 'test' if name == 'nq' else 'validation'
                config['datasets'][name] = {'raw_dir': str(raw), 'revision': 'fixed', 'splits': {split: ['rows.json']}}
            result = select(config, root / 'out', Tokenizer(), quota=25)
            self.assertTrue(result['complete_quota'])
            self.assertEqual(result['datasets']['nq']['context_over_128_tokens'], 1)
            queries = [json.loads(s) for s in (root / 'out/queries.jsonl').read_text().splitlines()]
            self.assertEqual(len(queries), 50)
            self.assertTrue(all(q['query'].startswith('original question') and not q['query'].endswith('?') for q in queries))
            self.assertEqual(result, select(config, root / 'out', Tokenizer(), quota=25))
            docs = [json.loads(s) for s in (root / 'out/documents.jsonl').read_text().splitlines()]
            self.assertEqual(len({d['context_sha256'] for d in docs}), 50)
            expected = sorted([text_hash(f'Rome nq {i}') for i in range(27)], key=lambda h: selection_order('nq', h))[:25]
            self.assertEqual([d['context_sha256'] for d in docs[:25]], expected)
            (root / 'out/queries.jsonl').write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'corruption'):
                select(config, root / 'out', Tokenizer(), quota=25)

    def test_output_errors_are_not_repaired_or_accepted_as_complete(self):
        ok = parse_decomposition('["Rome is a city."]', True, False)
        self.assertTrue(ok['format_valid'])
        self.assertEqual(ok['semantic_completeness'], 'unreviewed')
        for raw, eos, cap in [('["partial', True, False), ('[]', True, False), ('["fact"]', False, True), ('["fact"]', False, False)]:
            result = parse_decomposition(raw, eos, cap)
            self.assertTrue(result['potentially_incomplete'])
        duplicate = parse_decomposition('["fact", "fact"]', True, False)
        self.assertEqual(len(duplicate['reference_propositions']), 2)

class SupportQADiagnosticTests(unittest.TestCase):
    def test_assignment_rectangular_ties_and_global_optimum(self):
        import numpy as np
        from ape_retrieval.supportqa.diagnostics import assignment
        self.assertEqual(assignment([[1, 1], [1, 1]]), [(0, 0), (1, 1)])
        self.assertEqual(assignment([[1], [1], [1]]), [(0, 0)])
        self.assertEqual(assignment([[0, 9], [8, 0], [0, 0]]), [(0, 1), (1, 0)])
        self.assertEqual(assignment(np.empty((0, 2))), [])
        from itertools import permutations, combinations
        rng = np.random.default_rng(42)
        for k, n in [(2, 3), (3, 2), (3, 3)]:
            for _ in range(8):
                matrix = rng.integers(-2, 3, (k, n)).astype(np.float32)
                candidates = []
                for rows in combinations(range(k), min(k, n)):
                    for cols in permutations(range(n), min(k, n)):
                        pairs = list(zip(rows, cols))
                        candidates.append((sum(float(matrix[r, c]) for r, c in pairs), pairs))
                best = max(v[0] for v in candidates)
                expected = min(v[1] for v in candidates if v[0] == best)
                self.assertEqual(assignment(matrix), expected)

    def test_unmatched_support_is_a_miss_and_alternatives_are_not_a_union(self):
        from ape_retrieval.supportqa.diagnostics import support_metrics
        result = support_metrics([['p1', 'p2']], {'p1': 4}, [4, 0], [1, 3, 5])
        metric = result['best_sufficient_set_per_metric']
        self.assertEqual(metric['Recall@1'], .5)
        self.assertEqual(metric['all_required_Success@5'], 0)
        self.assertIsNone(metric['completion_rank'])
        result = support_metrics([['p1'], ['p2']], {'p1': 4, 'p2': 0}, [4, 0])
        self.assertEqual(result['best_sufficient_set_per_metric']['completion_rank'], 1)
        self.assertEqual(result['best_sufficient_set_per_metric']['Recall@1'], 1)

class SupportQARecoveryTests(unittest.TestCase):
    def test_malformed_committed_output_is_preserved_and_corruption_rejected(self):
        from unittest.mock import patch
        from ape_retrieval.supportqa import decompose, write_rows
        spec = {'directory': 'unused', 'input_template': 'Title:\nSection:\nContent: {context}'}
        config = {'reference': 'configs/reference.json', 'propositionizer': spec}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);(root / 'decompositions').mkdir()
            digest = text_hash('Rome')
            write_rows(root / 'documents.jsonl', [{'context_sha256': digest, 'document': 'Rome'}])
            path = root / 'decompositions' / (digest + '.json')
            record = {'context_sha256': digest, 'spec': spec, 'input_text': spec['input_template'].format(context='Rome'),
                      'raw_output': '["broken', 'output_sha256': text_hash('["broken')}
            path.write_text(json.dumps(record))
            with patch('transformers.AutoTokenizer.from_pretrained'), patch('transformers.AutoModelForSeq2SeqLM.from_pretrained', side_effect=AssertionError('No regeneration')):
                decompose(config, root, lambda force=False: False)
                self.assertEqual(json.loads(path.read_text()), record)
                record['raw_output'] = 'changed'
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, 'hash changed'):
                    decompose(config, root, lambda force=False: False)

class NewSupportQATests(unittest.TestCase):
    def test_prior_hashes_are_excluded_and_policy_cannot_change_on_resume(self):
        from ape_retrieval.supportqa import write_rows
        class Tokenizer:
            def __call__(self, text): return {'input_ids':[1,2,3]}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config={'reference':'configs/reference.json','datasets':{}}
            for dataset,split in [('nq','test'),('squad','validation')]:
                directory=root/dataset;directory.mkdir()
                rows=[{'context':f'Rome {dataset} {i}','question':f'where {i}',
                       'answers':{'text':['Rome'],'answer_start':[0]}} for i in range(7)]
                write_rows(directory/'rows.json',rows)
                config['datasets'][dataset]={'raw_dir':str(directory),'revision':'fixed','splits':{split:['rows.json']}}
            old={text_hash(f'Rome {dataset} 0') for dataset in ('nq','squad')}
            first=select(config,root/'a',Tokenizer(),quota=3,excluded_hashes=old)
            second=select(config,root/'b',Tokenizer(),quota=3,excluded_hashes=old)
            self.assertEqual(first['files'],second['files'])
            docs=[json.loads(s) for s in (root/'a/documents.jsonl').read_text().splitlines()]
            self.assertTrue(old.isdisjoint(d['context_sha256'] for d in docs))
            with self.assertRaisesRegex(ValueError,'policy changed'):
                select(config,root/'a',Tokenizer(),quota=4,excluded_hashes=old)

    def test_decomposition_failure_is_separate_from_document_answerability(self):
        from ape_retrieval.supportqa.annotations import derive_support
        row = derive_support({'document_answerability': 'answerable',
            'mapping_status': 'decomposition_error', 'support_sets': []}, {'p001'})
        self.assertEqual(row['support_type'], 'decomposition_error')
        self.assertEqual(row['document_answerability'], 'answerable')
        self.assertIsNone(row['minimum_support_cardinality'])
        missing = derive_support({'document_answerability': 'insufficient_document',
            'mapping_status': 'not_applicable'}, {'p001'})
        self.assertEqual(missing['support_type'], 'insufficient_document')
        self.assertEqual(derive_support({}, set())['support_type'], 'unreviewed')

class SemanticSupportTests(unittest.TestCase):
    def test_proposition_units_and_alternative_cardinalities(self):
        from ape_retrieval.supportqa.annotations import derive_support
        proposal = {'document_answerability': 'answerable', 'mapping_status': 'supported',
            'support_sets': [['p007'], ['p002', 'p003']]}
        row = derive_support(proposal, {'p007', 'p002', 'p003'})
        self.assertEqual(row['support_cardinalities'], [1, 2])
        self.assertEqual(row['minimum_support_cardinality'], 1)
        self.assertEqual(row['support_type'], 'single_support')
        proposal['support_sets'] = [['p002', 'p003']]
        self.assertEqual(derive_support(proposal, {'p002', 'p003'})['support_type'], 'multi_support')

    def test_invalid_links_nonminimal_and_contradictory_sets_are_rejected(self):
        from ape_retrieval.supportqa.annotations import derive_support
        base = {'document_answerability': 'answerable', 'mapping_status': 'supported'}
        for sets in ([['missing']], [['p1', 'p1']], [['p1'], ['p1', 'p2']], []):
            with self.assertRaises(ValueError):
                derive_support({**base, 'support_sets': sets}, {'p1', 'p2'})
        with self.assertRaises(ValueError):
            derive_support({'document_answerability': 'insufficient_document',
                'mapping_status': 'supported', 'support_sets': [['p1']]}, {'p1'})

    def test_unknown_or_changed_cache_configuration_is_rejected(self):
        from ape_retrieval.supportqa import reuse_cached_outputs
        with tempfile.TemporaryDirectory() as temp:
            previous = Path(temp) / 'previous'; previous.mkdir()
            (previous / 'configuration.json').write_text(json.dumps({'reference': 'old'}))
            with self.assertRaisesRegex(ValueError, 'Incompatible prior cache'):
                reuse_cached_outputs({'reuse_from': str(previous), 'reference': 'new'}, Path(temp) / 'current')
