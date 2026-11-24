"""Frozen review queues, evidence IDs and local human judgment records."""

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re

from ape_retrieval.benchmarks.data import json_lines
from ape_retrieval.encode import atomic_json
from ape_retrieval.propsearch import sha256
from ape_retrieval.supportqa.diagnostics import support_metrics
from ape_retrieval.supportqa.supportqa_annotations import read_decomposition


DATA = Path('data/supportqa')
RUN = Path('runs/supportqa')
BGE = Path('runs/bge-small')
ROOT = Path('runs/supportqa-human-v1')
MODELS = ('ape', 'bge', 'simcse', 'contriever')
RANK_FILES = {model: (BGE if model == 'bge' else RUN) / 'support' / f'{model}-predictions.jsonl.gz'
              for model in MODELS}
KS = (1, 5, 10, 20)
SEEDS = {'evaluation': 20261120, 'diagnostic': 20261121, 'second': 20261122}
ANSWERABILITY = ('answerable', 'insufficient', 'ambiguous', 'uncertain')
DEFECTS = ('missing', 'distorted')
CAUSES = ('decomposition', 'label', 'matching', 'ranking', 'suspected representation', 'unresolved')
TERMINAL = ('confirmed', 'needs adjudication')
INLINE_CODES = {'m': ('answerable', ['missing']), 'd': ('answerable', ['distorted']),
                'i': ('insufficient', []), 'a': ('ambiguous', []),
                'r': ('uncertain', []), 'u': ('uncertain', [])}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def read(path):
    return json.loads(path.read_text())


def index(path, key='query_id'):
    result = {}
    for row in json_lines(path):
        if row[key] in result:
            raise ValueError(f'Duplicate {key} in {path}: {row[key]}')
        result[row[key]] = row
    return result


def verify_hashes(files):
    for name, expected in files.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f'Frozen review artifact changed: {name}')


def freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen queue differs from the reconstructed queue: {path}')
    else:
        atomic_json(path, value)


class Study:
    def __init__(self):
        self.docs = index(DATA / 'documents.jsonl', 'document_id')
        self.queries = index(DATA / 'queries.jsonl')
        self.silver = index(DATA / 'annotations.jsonl')
        self.ranks = {}
        if self.queries.keys() != self.silver.keys():
            raise ValueError('Query and silver outcome IDs differ')
        for qid, query in self.queries.items():
            case = self.silver[qid]
            doc = self.docs[query['document_id']]
            if (case['document_id'] != query['document_id']
                    or case['dataset'] != query['source']['dataset'] or doc['dataset'] != case['dataset']):
                raise ValueError(f'Source IDs differ for {qid}')

    def paths(self, qid):
        doc = self.docs[self.queries[qid]['document_id']]
        digest = doc['context_sha256']
        return DATA / 'decompositions' / f'{digest}.json', RUN / 'diagnostics' / f'{digest}.json'

    def evidence(self, qid):
        query = self.queries[qid]
        doc = self.docs[query['document_id']]
        generated = read_decomposition(RUN, doc)
        if generated is None or generated['output_sha256'] != self.silver[qid]['decomposition_version']:
            raise ValueError(f'Frozen proposition join failed for {qid}')
        # Whitelist the fields that the blinded screens are allowed to receive.
        return {'question': query['query'], 'answers': query['answers'], 'context': doc['document'],
                'sentences': source_sentences(doc['document']),
                'propositions': {p['id']: p['text'] for p in generated['reference_propositions']}}

    def matching(self, qid):
        _, path = self.paths(qid)
        if not path.exists():
            return {}
        record = read(path)
        case = self.silver[qid]
        if (record['document_id'] != case['document_id']
                or record['decomposition_output_sha256'] != case['decomposition_version']):
            raise ValueError(f'Frozen alignment join failed for {qid}')
        return {**record['matching'], 'slot_ids': record['ape']['slot_ids']}

    def load_rankings(self):
        if self.ranks:
            return
        old = read(RUN / 'support/metrics.json')['predictions']
        bge = read(BGE / 'support/metrics.json')['predictions']['bge']
        for model, path in RANK_FILES.items():
            receipt = bge if model == 'bge' else old[model]
            if Path(receipt['file']) != path or sha256(path) != receipt['sha256']:
                raise ValueError(f'Frozen {model} prediction hash or path differs')
            self.ranks[model] = index(path)
            if self.ranks[model].keys() != self.queries.keys():
                raise ValueError(f'Frozen {model} prediction IDs differ')


def source_sentences(text):
    """Keep offsets in original Unicode code points, using punctuation/newlines."""
    boundaries = [0, *[m.end() for m in re.finditer(r'(?<=[.!?])\s+|\n+', text)], len(text)]
    result = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        raw = text[left:right]
        part = raw.strip()
        if part:
            start = left + len(raw) - len(raw.lstrip())
            result.append({'id': len(result) + 1, 'start': start, 'end': start + len(part), 'text': part})
    return result


def provenance(study, ids):
    result = {}
    for qid in ids:
        query = study.queries[qid]
        doc = study.docs[query['document_id']]
        result[qid] = {'document_id': doc['document_id'], 'context_sha256': doc['context_sha256'],
                       'source_document_group': query['source'].get('document_group'),
                       'stratum': [study.silver[qid]['dataset'], study.silver[qid]['support_type']]}
    return result


def evaluation_sample(study):
    strata = defaultdict(list)
    for qid, case in sorted(study.silver.items()):
        strata[(case['dataset'], case['support_type'])].append(qid)
    total = len(study.queries)
    if total != 10142:
        raise ValueError('Expected all 10,142 frozen SupportQA outcomes')
    seats = {key: 200 * len(ids) // total for key, ids in strata.items()}
    remaining = 200 - sum(seats.values())
    order = sorted(strata, key=lambda key: (-(200 * len(strata[key]) % total), key))
    for key in order[:remaining]:
        seats[key] += 1
    rng = random.Random(SEEDS['evaluation'])
    selected = [qid for key in sorted(strata) for qid in rng.sample(strata[key], seats[key])]
    rng.shuffle(selected)
    return {
        'schema': 1, 'queue': 'evaluation', 'seed': SEEDS['evaluation'], 'query_ids': selected,
        'design': 'stratified simple random sampling without replacement; largest-remainder proportional allocation',
        'strata': [{'source': s, 'outcome': o, 'population': len(strata[s, o]), 'sample': seats[s, o],
                    'probability': seats[s, o] / len(strata[s, o]), 'weight': len(strata[s, o]) / seats[s, o]}
                   for s, o in sorted(strata)],
        'cases': provenance(study, selected),
        'files': {str(path): sha256(path) for path in (DATA / 'documents.jsonl', DATA / 'queries.jsonl',
                  DATA / 'annotations.jsonl', RUN / 'selection.json',
                  *{study.paths(qid)[0] for qid in selected})},
    }


def diagnostic_sample(study, evaluation):
    excluded = {row['context_sha256'] for row in evaluation['cases'].values()}
    rng = random.Random(SEEDS['diagnostic'])
    pools = defaultdict(lambda: defaultdict(list))
    for qid, case in sorted(study.silver.items()):
        if case['support_type'] not in ('single_support', 'multi_support'):
            continue
        if study.docs[case['document_id']]['context_sha256'] in excluded:
            continue
        ape, bge = (study.ranks[model][qid]['metrics'] for model in ('ape', 'bge'))
        if not ape or not bge:
            continue
        a, b = (bool(m['best_sufficient_set_per_metric']['all_required_Success@5']) for m in (ape, bge))
        early = bool(ape['best_sufficient_set_per_metric']['all_required_Success@1'])
        category = {(False, True): 'BGE only at k=5', (True, False): 'APE only at k=5',
                    (False, False): 'both miss at k=5'}.get((a, b), 'both succeed at k=5')
        if a:
            category += ' / APE succeeds at k=1' if early else ' / APE misses at k=1'
        pools[(case['dataset'], case['support_type'])][category].append(qid)
    selected, reasons = [], {}
    for key in sorted(pools):
        categories = pools[key]
        for ids in categories.values():
            rng.shuffle(ids)
        n = 0
        while n < 10:
            previous = n
            for category in sorted(categories):
                ids = categories[category]
                while ids and study.docs[study.silver[ids[-1]]['document_id']]['context_sha256'] in excluded:
                    ids.pop()
                if ids:
                    qid = ids.pop()
                    selected.append(qid)
                    reasons[qid] = category
                    excluded.add(study.docs[study.silver[qid]['document_id']]['context_sha256'])
                    n += 1
                if n == 10:
                    break
            if n == previous:
                raise ValueError(f'Insufficient disjoint diagnostic cases for {key}')
    if len(selected) != 40:
        raise ValueError('Invalid diagnostic queue size')
    rng.shuffle(selected)
    return {'schema': 1, 'queue': 'diagnostic', 'seed': SEEDS['diagnostic'], 'query_ids': selected,
            'design': 'targeted round-robin over frozen k=1/k=5 outcome categories',
            'reasons': reasons, 'cases': provenance(study, selected)}


def prepare(study, root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    evaluation = evaluation_sample(study)
    freeze(root / 'evaluation.json', evaluation)
    second = sorted(random.Random(SEEDS['second']).sample(evaluation['query_ids'], 40))
    random.Random(SEEDS['second']).shuffle(second)
    freeze(root / 'second.json', {'schema': 1, 'queue': 'second', 'seed': SEEDS['second'],
           'design': 'simple random subset of the frozen evaluation sample', 'query_ids': second,
           'evaluation_sha256': sha256(root / 'evaluation.json')})
    study.load_rankings()
    diagnostic = diagnostic_sample(study, evaluation)
    all_ids = evaluation['query_ids'] + diagnostic['query_ids']
    artifacts = {*RANK_FILES.values(), RUN / 'support/metrics.json', BGE / 'support/metrics.json'}
    for qid in all_ids:
        study.evidence(qid)
        study.matching(qid)
        artifacts.update(p for p in study.paths(qid) if p.exists())
    diagnostic.update(evaluation_sha256=sha256(root / 'evaluation.json'),
                      files={str(path): sha256(path) for path in artifacts})
    freeze(root / 'diagnostic.json', diagnostic)
    print(f'Frozen queues in {root}. Evaluation 200, diagnostic 40, independent second 40.')


def open_queues(root, queue):
    evaluation = read(root / 'evaluation.json')
    verify_hashes(evaluation['files'])
    selected = evaluation if queue == 'evaluation' else read(root / f'{queue}.json')
    if queue != 'evaluation' and selected['evaluation_sha256'] != sha256(root / 'evaluation.json'):
        raise ValueError('Queue does not match the frozen evaluation sample')
    if queue == 'diagnostic':
        verify_hashes(selected['files'])
    return selected


def support_class(judgment):
    sets = judgment['support_sets']
    return 'no_sufficient_set' if not sets else ('single_support' if min(map(len, sets)) == 1 else 'multi_support')


def validate_source(source, evidence, confirming=False, complete=True):
    if source['answerability'] not in ANSWERABILITY and (complete or source['answerability'] != ''):
        raise ValueError('Choose an answerability judgment (1–4).')
    sentences = {s['id']: s for s in evidence['sentences']}
    used = set()
    for s in source['spans']:
        sentence = sentences.get(s['sentence'])
        if (sentence is None or s['sentence'] in used
                or not sentence['start'] <= s['start'] < s['end'] <= sentence['end']):
            raise ValueError('Evidence must use a unique sentence ID and offsets within that sentence.')
        used.add(s['sentence'])
    if complete and source['answerability'] == 'answerable' and (not used or not source['answer'].strip()):
        raise ValueError('An answerable judgment requires evidence and a short answer.')
    if confirming and source['answerability'] == 'uncertain':
        raise ValueError('Use J to save an uncertain judgment for adjudication.')


def validate_sets(sets, known):
    unique = [frozenset(group) for group in sets]
    if any(not group or len(group) != len(keys) or not keys <= known
           for group, keys in zip(sets, unique, strict=True)):
        raise ValueError('Each alternative needs distinct proposition IDs from this context.')
    if len(set(unique)) != len(unique) or any(a < b for a in unique for b in unique):
        raise ValueError('Alternatives must be distinct and independently minimal.')


def blank(qid):
    return {'query_id': qid, 'initial': None, 'reference_seen': None, 'comparison_seen': None,
            'versions': [], 'draft': None, 'diagnoses': [], 'skipped': False}


def new_draft():
    return {'source': {'answerability': '', 'spans': [], 'answer': ''},
            'reference_note': '', 'source_revision_reason': '', 'support_sets': [],
            'defects': [], 'explanation': '', 'revision_reason': ''}


def inline_judgment(value, evidence, draft):
    judgment = deepcopy(draft)
    sets = []
    if value in INLINE_CODES:
        answerability, defects = INLINE_CODES[value]
    else:
        if not re.fullmatch(r'[0-9]+(?:\s+[0-9]+)*(?:\s*/\s*[0-9]+(?:\s+[0-9]+)*)*', value):
            raise ValueError('Use proposition numbers, alternatives separated by /, or m/d/i/a/r/u.')
        ids = list(evidence['propositions'])
        for group in value.split('/'):
            numbers = list(map(int, group.split()))
            if not all(1 <= n <= len(ids) for n in numbers):
                raise ValueError('Use proposition numbers shown on this screen.')
            sets.append([ids[n - 1] for n in numbers])
        answerability, defects = 'answerable', []
    judgment.update(entry_mode='inline', support_sets=sets, defects=defects,
                    pending_reason={'r': 'reference discrepancy', 'u': 'uncertain'}.get(value))
    judgment['source'].update(answerability=answerability,
                              answer=' | '.join(evidence['answers']) if answerability == 'answerable' else '')
    return judgment, 'needs adjudication' if judgment['pending_reason'] else 'confirmed'


def validate_decision(record, judgment, evidence, confirming):
    if judgment.get('entry_mode') == 'inline':
        source = judgment['source']
        validate_source(source, evidence, confirming, complete=False)
        sets, defects = judgment['support_sets'], judgment['defects']
        validate_sets(sets, evidence['propositions'].keys())
        if source['answerability'] not in ANSWERABILITY:
            raise ValueError('Missing source judgment.')
        pending = judgment['pending_reason']
        if source['answerability'] == 'answerable':
            if source['answer'] != ' | '.join(evidence['answers']) or not source['answer']:
                raise ValueError('Saved answer must match the visible dataset reference.')
            if defects not in ([], ['missing'], ['distorted']) or bool(sets) == bool(defects):
                raise ValueError('Select sufficient propositions or enter m or d.')
        elif sets or defects or source['answer']:
            raise ValueError('Only answerable cases may have proposition support or coverage flags.')
        if (confirming and (pending or source['answerability'] == 'uncertain')
                or not confirming and (pending not in ('reference discrepancy', 'uncertain')
                                       or source['answerability'] != 'uncertain')):
            raise ValueError('Reference objections and uncertain cases must stay pending.')
        return
    initial = record['initial']
    if not initial or not record['reference_seen']:
        raise ValueError('Save the initial source judgment and open the native reference first.')
    validate_source(initial['source'], evidence)
    validate_source(judgment['source'], evidence, confirming)
    if judgment['source'] != initial['source'] and not judgment['source_revision_reason'].strip():
        raise ValueError('Record a reason for revising the initial source judgment.')
    sets = judgment['support_sets']
    validate_sets(sets, evidence['propositions'].keys())
    if sets and judgment['source']['answerability'] != 'answerable':
        raise ValueError('Sufficient proposition sets require an answerable source judgment.')
    if set(judgment['defects']) - set(DEFECTS):
        raise ValueError('Unknown proposition coverage flag.')
    if (judgment['defects'] or not sets) and not judgment['explanation'].strip():
        raise ValueError('Explain the coverage concern or why there is no sufficient set.')
    if record['versions'] and not judgment['revision_reason'].strip():
        raise ValueError('Enter a reason for revising an earlier decision.')


def current_version(record):
    return next((version for version in reversed(record['versions']) if not version.get('undone_at')), None)


def state(record):
    if record['draft'] is not None:
        return 'draft'
    version = current_version(record)
    return version['status'] if version else 'unreviewed'


def save_version(record, evidence, reviewer, status):
    judgment = record['draft']
    validate_decision(record, judgment, evidence, status == 'confirmed')
    version = {'version': len(record['versions']) + 1, 'status': status, 'reviewer': reviewer,
               'saved_at': now(), 'comparison_seen': bool(record['comparison_seen']),
               'support_class': support_class(judgment), 'judgment': deepcopy(judgment)}
    record['versions'].append(version)
    record['draft'] = None
    record['skipped'] = False


def scored(study, qid, model, sets, matching=None):
    if not sets:
        return None, 'no sufficient set'
    row = study.ranks[model].get(qid, {})
    ranked = row.get('ranked_ids')
    if not ranked:
        return None, 'no saved ranking: ' + row.get('eligibility', 'missing prediction')
    if len(ranked) != len(set(ranked)):
        raise ValueError(f'Duplicate ranking IDs for {qid} / {model}')
    if model == 'ape':
        if matching is None:
            matching = study.matching(qid)
        if matching.get('status') != 'provisional_teacher_space':
            return None, 'no usable saved assignment: ' + matching.get('reason', 'missing assignment')
        if (set(ranked) != set(matching['slot_ids'])
                or not set().union(*map(set, sets)) <= set(matching['reference_ids'])):
            raise ValueError(f'APE ranking or reference join failed for {qid}')
        mapping = {p['reference_id']: p['slot_id'] for p in matching['pairs']}
    else:
        mapping = {p: p for p in ranked}
        if not set().union(*map(set, sets)) <= mapping.keys():
            raise ValueError(f'Proposition IDs missing in {model} ranking for {qid}')
    return support_metrics(sets, mapping, ranked, KS), None
