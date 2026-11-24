"""Fictional decisions for checking SupportQA review software only."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from ape_retrieval.encode import atomic_json
from ape_retrieval.supportqa.human_review import MODELS, new_draft, now, save_version, scored, source_sentences, state, validate_sets
from ape_retrieval.supportqa.review import Records, Review, evidence_lines, comparison_allowed
from ape_retrieval.supportqa.review_report import judgment_records, latest_confirmations, model_scores


EVIDENCE = {
    'question': 'Which town held the fictional event?',
    'answers': ['Alden'],
    'context': 'The event was held in Alden.\n\nThe venue was a library.',
    'sentences': source_sentences('The event was held in Alden.\n\nThe venue was a library.'),
    'propositions': {'p001': 'The event was in Alden.', 'p002': 'The event used a library.',
                     'p003': 'Alden hosted the event.', 'p004': 'The event was fictional.'},
}
IDS = ['fiction:one', 'fiction:two']


class HumanReviewTests(unittest.TestCase):
    def test_single_screen_entry_pending_and_resume(self):
        fox = {'question': 'what is the scientific name for a red fox',
               'answers': ['Vulpes vulpes'],
               'context': 'The red fox (Vulpes vulpes) is a mammal.',
               'sentences': source_sentences('The red fox (Vulpes vulpes) is a mammal.'),
               'propositions': {'p001': 'The red fox is a mammal.'}}
        ids = [*IDS, 'fiction:third', 'fiction:fox', 'fiction:objection', 'fiction:uncertain']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / 'diagnostic.json', {'files': {}})
            study = SimpleNamespace(evidence=lambda qid: fox if qid == 'fiction:fox' else EVIDENCE,
                                    load_rankings=lambda: None)
            store = Records(root, 'evaluation', 'fixture-primary')
            selection = {'query_ids': ids}
            review = Review(None, study, store, 'evaluation', selection, root)
            lines = '\n'.join(evidence_lines(review.evidence, review.record))
            self.assertIn('Dataset reference answer: Alden', lines)
            self.assertIn('Context\nThe event was held in Alden.\n\nThe venue was a library.', lines)
            review.submit(':spans')
            self.assertEqual(review.page, 'spans')
            self.assertIn('1. [0:28] The event was held in Alden.',
                          '\n'.join(evidence_lines(review.evidence, review.record, show_spans=True)))
            review.submit(':back')
            self.assertIn('2. [p002]', lines)
            self.assertNotIn('Silver outcome', lines)
            self.assertNotIn('Saved ranks', lines)
            with self.assertRaisesRegex(ValueError, 'full queue'):
                review.submit(':compare')
            for invalid in ('20', '2 /', '2 // 4', '2 / 2 4'):
                with self.assertRaises(ValueError):
                    review.submit(invalid)
                self.assertEqual(review.position, 0)
                self.assertEqual(state(review.record), 'unreviewed')
            for value, expected in (('2', [['p002']]), ('2 4', [['p002', 'p004']]),
                                    ('2 / 4', [['p002'], ['p004']])):
                qid = review.record['query_id']
                review.submit(value)
                self.assertEqual(store.load(qid)['versions'][0]['judgment']['support_sets'], expected)
            self.assertEqual(review.evidence['answers'], ['Vulpes vulpes'])
            review.submit('m')
            self.assertEqual(review.position, 4)
            saved = store.load('fiction:fox')['versions'][0]
            self.assertEqual(saved['status'], 'confirmed')
            self.assertEqual(saved['judgment']['source']['answerability'], 'answerable')
            self.assertEqual(saved['judgment']['defects'], ['missing'])
            self.assertEqual(saved['judgment']['support_sets'], [])
            self.assertEqual(saved['judgment']['explanation'], '')
            review.submit('r')
            self.assertEqual(review.position, 5)
            self.assertEqual(state(store.load('fiction:objection')), 'needs adjudication')
            self.assertFalse(comparison_allowed('evaluation', store.load('fiction:fox'), review.records))
            self.assertTrue(comparison_allowed('diagnostic', store.load('fiction:fox'), review.records))
            review = Review(None, study, store, 'evaluation', selection, root)
            self.assertEqual(review.position, 5)
            review.submit(':undo')
            self.assertEqual(review.position, 4)
            self.assertEqual(state(review.record), 'unreviewed')
            review.submit('r')
            review.submit('u')
            review.submit(':goto 4')
            review.submit(':compare')
            self.assertEqual(review.page, 'comparison')
            review.submit(':back')
            review.submit('d')
            versions = store.load('fiction:fox')['versions']
            self.assertFalse(versions[0]['comparison_seen'])
            self.assertTrue(versions[1]['comparison_seen'])
            self.assertEqual(versions[1]['judgment']['defects'], ['distorted'])
            records = judgment_records(study, root, 'evaluation', ids)
            blind = latest_confirmations(records, blinded=True)
            self.assertEqual(blind['fiction:fox']['defects'], ['missing'])
            self.assertNotIn('fiction:objection', blind)
            self.assertNotIn('fiction:uncertain', blind)
            second = Records(root, 'second', 'fixture-independent')
            self.assertEqual(state(second.load(IDS[0])), 'unreviewed')

    def test_legacy_record_remains_reportable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = Records(root, 'evaluation', 'fixture')
            record = store.load(IDS[0])
            draft = new_draft()
            stamp = now()
            draft['source'] = {'answerability': 'answerable', 'answer': 'Alden',
                               'spans': [{'sentence': 1, 'start': 22, 'end': 27}]}
            draft['support_sets'] = [['p001']]
            record.update(initial={'source': draft['source'], 'saved_at': stamp, 'reviewer': store.reviewer},
                          reference_seen=stamp, draft=draft)
            save_version(record, EVIDENCE, store.reviewer, 'confirmed')
            store.save(record)
            study = SimpleNamespace(evidence=lambda qid: EVIDENCE)
            records = judgment_records(study, root, 'evaluation', IDS)
            self.assertEqual(latest_confirmations(records, True)[IDS[0]]['support_sets'], [['p001']])

    def test_alternatives_and_unavailable_saved_predictions(self):
        match = {'status': 'provisional_teacher_space', 'slot_ids': [1, 2],
                 'reference_ids': ['p001', 'p002', 'p003'],
                 'pairs': [{'reference_id': 'p003', 'slot_id': 2}, {'reference_id': 'p002', 'slot_id': 1}]}
        study = SimpleNamespace(
            silver={IDS[0]: {'support_sets': [['p001']]}, IDS[1]: {'support_sets': []}},
            ranks={model: {
                IDS[0]: {'ranked_ids': [1, 2] if model == 'ape' else ['p003', 'p002', 'p001'],
                         'eligibility': 'eligible'},
                IDS[1]: {'eligibility': 'decomposition_error'},
            } for model in MODELS},
            matching=lambda qid: match,
            load_rankings=lambda: None,
        )
        sets = [['p001', 'p002'], ['p003']]
        validate_sets(sets, {'p001', 'p002', 'p003'})
        with self.assertRaisesRegex(ValueError, 'independently minimal'):
            validate_sets([['p001'], ['p001', 'p002']], {'p001', 'p002'})
        result, reason = scored(study, IDS[0], 'ape', sets)
        self.assertIsNone(reason)
        self.assertEqual(result['alternatives'][0]['ranks'], [None, 1])
        self.assertEqual(result['alternatives'][0]['metrics']['Recall@5'], .5)
        self.assertEqual(result['best_sufficient_set_per_metric']['all_required_Success@5'], 1)
        human, silver, _, missing = model_scores(study, {
            IDS[0]: {'support_sets': sets}, IDS[1]: {'support_sets': [['p001']]},
        })
        self.assertEqual(human['bge'][IDS[0]]['first_support_reciprocal_rank'], 1)
        self.assertEqual(silver['bge'][IDS[0]]['first_support_reciprocal_rank'], 1 / 3)
        for model in MODELS:
            self.assertNotIn(IDS[1], human[model])
            self.assertEqual(missing[model]['no saved ranking: decomposition_error'], 1)
        human, silver, changes, _ = model_scores(study, {IDS[0]: {'support_sets': [], 'defects': ['missing']}})
        for model in MODELS:
            self.assertNotIn(IDS[0], human[model])
            self.assertIn(IDS[0], silver[model])
            self.assertEqual(changes[model]['eligible', 'no sufficient set'], 1)
