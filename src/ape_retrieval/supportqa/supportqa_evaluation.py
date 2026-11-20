"""Within-context support rankings and alternative-set metrics for SupportQA."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ape_retrieval.benchmarks.data import json_lines
from ape_retrieval.benchmarks.dense import (benchmark_text_index, checkpoint, encode_index,
    open_vectors, source_paths, text_index_hash, write_rank_block)
from ape_retrieval.encode import atomic_json
from ape_retrieval.supportqa import dataset_path, text_hash
from ape_retrieval.supportqa.diagnostics import support_metrics
from ape_retrieval.supportqa.supportqa_annotations import read_decomposition, selected_rows
from ape_retrieval.propsearch import sha256


CLASSES = ('single_support', 'multi_support')
MODELS = ('ape', 'simcse', 'contriever')


def read(path):
    return json.loads(path.read_text())


def support_sources(config, output, name):
    paths, sources = source_paths(config)
    benchmark_positions, _ = benchmark_text_index(paths)
    encoding = output / 'retrieval' / name / 'encoding'
    manifest = read(encoding / 'manifest.json')
    model_dir = checkpoint(output, name, config['baselines'][name])
    if (manifest['identity'] != sources or manifest['text_index_sha256'] != text_index_hash(benchmark_positions)
            or manifest['checkpoint_files_sha256'] != sha256(model_dir / 'files.json')
            or manifest['model'] != config['baselines'][name]):
        raise ValueError('Support query requires compatible document-benchmark vectors')
    documents, groups = selected_rows(output)
    propositions, new_texts = {}, {}
    generation = hashlib.sha256()
    for doc_id, document in documents.items():
        value = read_decomposition(output, document)
        if value is None:
            raise InterruptedError('Waiting for complete question-blind generation')
        generation.update(bytes.fromhex(sha256(output / 'decompositions' / (document['context_sha256'] + '.json'))))
        propositions[doc_id] = value['reference_propositions']
        if not value['potentially_incomplete']:
            for p in value['reference_propositions']:
                if p['text'] not in benchmark_positions:
                    new_texts.setdefault(p['text'], len(new_texts))
    if any(q['query'] not in benchmark_positions for rows in groups.values() for q in rows):
        raise ValueError('Selected native query missing from the document benchmark embedding index')
    identity = {'benchmark_manifest_sha256': sha256(encoding / 'manifest.json'),
                'benchmark_progress_sha256': sha256(encoding / 'progress.json'),
                'supportqa_selection_sha256': sha256(output / 'selection.json'),
                'decompositions_sha256': generation.hexdigest()}
    return model_dir, encoding, identity, benchmark_positions, new_texts, propositions


def encode_support(config, output, name, should_stop):
    model_dir, _, identity, _, new_texts, _ = support_sources(config, output, name)
    encode_index(config, output / 'support' / name / 'encoding', list(new_texts), identity,
                 name, model_dir, should_stop)


def rank_support(ids, scores):
    if len(ids) != len(scores) or len(ids) != len(set(ids)) or not np.isfinite(scores).all():
        raise ValueError('Invalid within-context scoring candidates')
    return [ids[i] for i in np.lexsort((ids, -scores))]


def score_alternatives(alternatives, mapping, ranked, ks):
    result = support_metrics(alternatives, mapping, ranked, ks)
    for alternative in result['alternatives']:
        alternative['cardinality'] = len(alternative['support_ids'])
        alternative['unmapped_support_ids'] = [p for p in alternative['support_ids'] if p not in mapping]
    return result


def predict_ape(output, query, case, ks):
    digest = case['document_id'][2:]
    diagnostic = output / 'diagnostics' / (digest + '.json')
    qpath = output / 'encodings' / (text_hash(query['query']) + '.json')
    if not diagnostic.exists() or not qpath.exists():
        return {'eligibility': 'missing_ape_encoding', 'metrics': None}
    record, qvector = read(diagnostic), read(qpath)
    if record['decomposition_output_sha256'] != case['decomposition_version']:
        raise ValueError('APE teacher proxy uses a different decomposition revision')
    if record['encoding_file'] != f'encodings/{digest}.json':
        raise ValueError('APE teacher proxy uses a different context encoding')
    dvector = read(output / record['encoding_file'])
    query_vectors = np.asarray(qvector['vectors'], dtype=np.float32)
    context_vectors = np.asarray(dvector['vectors'], dtype=np.float32)
    slot_ids = dvector['slot_ids']
    scores = (query_vectors @ context_vectors.T).max(axis=0)
    ranked = rank_support(slot_ids, scores)
    result = {'active_slots': len(slot_ids), 'ranked_ids': ranked, 'metrics': None}
    matching = record['matching']
    if matching['status'] != 'provisional_teacher_space':
        return {**result, 'eligibility': matching['reason']}
    known_ids = set(matching['reference_ids'])
    if not all(set(s) <= known_ids for s in case['support_sets']):
        raise ValueError('Support references absent from APE teacher-space input')
    mapping = {p['reference_id']: p['slot_id'] for p in matching['pairs']}
    return {**result, 'eligibility': 'eligible', 'candidate_count': len(ranked),
            'metrics': score_alternatives(case['support_sets'], mapping, ranked, ks)}


def baseline_accessor(config, output, name):
    model_dir, encoding, identity, positions, new_texts, propositions = support_sources(config, output, name)
    support_dir = output / 'support' / name / 'encoding'
    manifest = read(support_dir / 'manifest.json')
    if (manifest['identity'] != identity or manifest['text_index_sha256'] != text_index_hash(new_texts)
            or manifest['checkpoint_files_sha256'] != sha256(model_dir / 'files.json')
            or manifest['model'] != config['baselines'][name]):
        raise ValueError('Support proposition cache identity changed')
    benchmark_vectors = open_vectors(encoding)
    support_vectors = open_vectors(support_dir)

    def predict(query, case, ks):
        props = propositions[case['document_id']]
        ids = [p['id'] for p in props]
        vectors = np.stack([benchmark_vectors[positions[p['text']]] if p['text'] in positions
                            else support_vectors[new_texts[p['text']]] for p in props])
        ranked = rank_support(ids, vectors @ benchmark_vectors[positions[query['query']]])
        return {'eligibility': 'eligible', 'candidate_count': len(ranked), 'ranked_ids': ranked,
                'metrics': score_alternatives(case['support_sets'], {p: p for p in ids}, ranked, ks)}

    return predict


def summarize_group(records, model, ks, common=False):
    eligible = [r[model] for r in records if r[model]['metrics']
                and (not common or all(r[m]['metrics'] for m in MODELS))]
    values = [r['metrics']['best_sufficient_set_per_metric'] for r in eligible]
    keys = ['first_support_reciprocal_rank', *(f'Recall@{k}' for k in ks),
            *(f'all_required_Success@{k}' for k in ks)]
    eligible_count = len(eligible)
    means = {('MRR' if key == keys[0] else key): sum(v[key] for v in values) / eligible_count
             for key in keys} if values else None
    exclusions = Counter(r[model]['eligibility'] for r in records if not r[model]['metrics'])
    if common:
        excluded_by_other_models = len(records) - sum(exclusions.values()) - eligible_count
        if excluded_by_other_models:
            exclusions['ineligible_for_another_model'] = excluded_by_other_models
    return {'labeled_queries': len(records), 'metric_queries': eligible_count, 'metrics': means,
            'exclusions': dict(exclusions),
            'queries_with_more_candidates_than_k': {str(k): sum(r['candidate_count'] > k for r in eligible) for k in ks},
            'alternative_support_sets': sum(len(r['metrics']['alternatives']) for r in eligible),
            'unmapped_support_occurrences_across_alternatives': sum(len(a['unmapped_support_ids'])
                for r in eligible for a in r['metrics']['alternatives']),
            'queries_with_unmapped_support_in_any_alternative': sum(any(a['unmapped_support_ids']
                for a in r['metrics']['alternatives']) for r in eligible)}


def evaluate_support(config, output, should_stop):
    annotation = read(output / 'annotation-summary.json')
    if annotation['status'] != 'COMPLETE':
        raise InterruptedError('Waiting for complete Codex semantic annotation')
    rows_path = dataset_path(output) / 'annotations.jsonl'
    if sha256(rows_path) != annotation['queries_sha256']:
        raise ValueError('Support annotation query hash changed')
    documents, by_doc = selected_rows(output)
    queries = {q['query_id']: q for rows in by_doc.values() for q in rows}
    annotation_rows = list(json_lines(rows_path))
    cases = {row['query_id']: row for row in annotation_rows}
    selected_count = sum(len(rows) for rows in by_doc.values())
    if len(annotation_rows) != len(cases) or len(queries) != selected_count or cases.keys() != queries.keys():
        raise ValueError('Support annotation coverage differs from the frozen query population')
    ks = config['retrieval']['ks']
    directory = output / 'support'
    predictions = {qid: {} for qid in cases}
    source_paths_by_model = {}
    for name in MODELS:
        if should_stop():
            raise InterruptedError('Stopped before SupportQA model evaluation')
        if name == 'ape':
            def predict(query, case, cutoffs):
                return predict_ape(output, query, case, cutoffs)
        else:
            predict = baseline_accessor(config, output, name)
        for qid, case in cases.items():
            if case['document_id'] != queries[qid]['document_id']:
                raise ValueError('Annotation context does not match original query')
            if case['support_type'] in CLASSES:
                result = predict(queries[qid], case, ks)
            else:
                result = {'eligibility': case['support_type'], 'metrics': None}
            predictions[qid][name] = result
        path = directory / f'{name}-predictions.jsonl.gz'
        write_rank_block(path, ({'query_id': qid, **by_model[name]} for qid, by_model in predictions.items()))
        source_paths_by_model[name] = {'file': str(path), 'sha256': sha256(path)}
        del predict
        should_stop(True)
    groups = defaultdict(list)
    for qid, case in cases.items():
        group = case['support_type']
        if group in CLASSES:
            groups[case['dataset'], group].append(predictions[qid])
            groups['combined', group].append(predictions[qid])
    active_slots = defaultdict(Counter)
    for doc in documents.values():
        diagnostic = output / 'diagnostics' / (doc['context_sha256'] + '.json')
        if diagnostic.exists():
            count = read(diagnostic)['ape']['n_active_slots']
            active_slots[doc['dataset']][count] += 1
            active_slots['combined'][count] += 1
    all_groups = [(dataset, label) for dataset in ('nq', 'squad', 'combined') for label in CLASSES]
    atomic_json(directory / 'metrics.json', {'task': 'within-context proposition support',
        'input_annotations_sha256': annotation['queries_sha256'],
        'predictions': source_paths_by_model, 'active_slots_per_context': active_slots,
        'method_eligibility': {ds: {m: dict(Counter(predictions[qid][m]['eligibility'] for qid, case in cases.items()
                                         if case['dataset'] == ds)) for m in MODELS} for ds in ('nq', 'squad')},
        'groups': {f'{ds}/{label}': {m: summarize_group(groups[ds, label], m, ks) for m in MODELS}
                   for ds, label in all_groups},
        'common_eligible_groups': {f'{ds}/{label}': {m: summarize_group(groups[ds, label], m, ks, common=True) for m in MODELS}
                                   for ds, label in all_groups}})
