"""Run the standalone BGE comparison against the frozen benchmark populations."""

import argparse
from collections import Counter, defaultdict
import fcntl
from itertools import chain
import json
import math
from pathlib import Path
import shutil
import time

from ape_retrieval.benchmarks.dense import (benchmark_text_index, checkpoint, commit_manifest,
                                           open_vectors, run_dense, source_paths)
from ape_retrieval.benchmarks.data import json_lines
from ape_retrieval.benchmarks.metrics import aggregate_splits
from ape_retrieval.benchmarks.runtime import run_context
from ape_retrieval.encode import atomic_json
from ape_retrieval.propsearch import sha256
from ape_retrieval.supportqa.supportqa_annotations import read_decomposition
from ape_retrieval.supportqa.supportqa_evaluation import (CLASSES, MODELS, encode_support,
                                                          rank_frozen_support, read, support_cases)


CONFIG = Path('configs/bge.json')
MODEL = 'bge'
PHASES = ('encode', 'rank', 'encode-support', 'rank-support')


def check_checkpoint(config, output):
    spec = config['baselines'][MODEL]
    reference = read(Path(config['reference']))
    support_config = read(Path(config['supportqa_config']))
    old_config = read(Path(config['supportqa_run']) / 'configuration.json')
    teacher = support_config['teacher']
    model_pin = spec['model_id'], spec['revision']
    if (model_pin != (reference['backbone_id'], reference['backbone_revision'])
            or model_pin != (teacher['model_id'], teacher['revision'])
            or spec['weights_sha256'] != teacher['model_sha256']
            or teacher != old_config['teacher']
            or teacher['pooling'] != 'attention_masked_mean' or not teacher['normalize']):
        raise ValueError('BGE does not match the frozen APE backbone and SupportQA teacher')
    if config['retrieval']['ks'] != old_config['retrieval']['ks']:
        raise ValueError('BGE metric cutoffs differ from the frozen comparison')
    directory = checkpoint(output, MODEL, spec)
    architecture = read(directory / 'config.json')
    if (architecture['hidden_size'] != spec['dimension']
            or architecture['max_position_embeddings'] != spec['max_tokens']):
        raise ValueError('BGE dimensions differ from the local checkpoint')


def prior_results(frozen):
    suite = Path('runs/benchmark-suite')
    paths = [frozen / filename for filename in (
        'results.json', 'report.md', 'verification.json', 'support/metrics.json',
        'annotations.jsonl', 'annotation-summary.json', 'selection.json',
        'documents.jsonl', 'queries.jsonl', 'configuration.json')]
    for directory in (frozen / 'retrieval', suite):
        for pattern in ('queries-*.jsonl.gz', 'metrics.json'):
            paths.extend(directory.rglob(pattern))
    paths.extend((suite / 'reports').iterdir())
    paths.extend((frozen / 'support').glob('*-predictions.jsonl.gz'))
    paths.extend((frozen / 'diagnostics').glob('*.json'))
    return {str(path): sha256(path) for path in sorted(set(paths)) if path.is_file()}


def preflight(config, output):
    check_checkpoint(config, output)
    paths, sources = source_paths(config)
    positions, ids = benchmark_text_index(paths)
    frozen = Path(config['supportqa_run'])
    documents, queries, _, annotation_hash = support_cases(frozen)
    proposition_texts = set()
    for document in documents.values():
        record = read_decomposition(frozen, document)
        if record is None:
            raise ValueError('A frozen SupportQA decomposition is missing')
        if not record['potentially_incomplete']:
            proposition_texts.update(p['text'] for p in record['reference_propositions']
                                     if p['text'] not in positions)
    spec = config['baselines'][MODEL]
    populations = {dataset: {part: len(rows) for part, rows in parts.items()}
                   for dataset, parts in ids.items()}
    vector_bytes = (len(positions) + len(proposition_texts)) * spec['dimension'] * 4
    corpus_bytes = max(data['corpus'] for data in populations.values()) * spec['dimension'] * 4
    score_batch_bytes = max(data['corpus'] for data in populations.values()) * config['retrieval']['scoring_query_batch_size'] * 4
    snapshot = {'benchmark_manifests': sources, 'populations': populations,
                'unique_benchmark_texts': len(positions), 'new_support_texts': len(proposition_texts),
                'supportqa_documents': len(documents), 'supportqa_queries': len(queries),
                'annotations_sha256': annotation_hash,
                'fp32_vector_bytes': vector_bytes, 'max_corpus_tensor_bytes': corpus_bytes,
                'max_score_batch_bytes': score_batch_bytes,
                'config_sha256': sha256(CONFIG), 'uv_lock_sha256': sha256(Path('uv.lock')),
                'previous_result_hashes': prior_results(frozen)}
    reserve = config['limits']['minimum_free_disk_bytes']
    if (vector_bytes >= config['limits']['storage_bytes']
            or not (output / 'preflight.json').exists() and shutil.disk_usage(output).free < vector_bytes + reserve):
        raise RuntimeError('BGE vectors and required free reserve will not fit')
    commit_manifest(output / 'preflight.json', snapshot)
    print(json.dumps({key: val for key, val in snapshot.items() if key != 'previous_result_hashes'}), flush=True)


def run_phase(config, output, phase):
    directory = output / 'phases' / phase
    receipt = directory / 'completion.json'
    if receipt.exists():
        print(json.dumps({'phase': phase, 'status': 'ALREADY_COMPLETED'}), flush=True)
        return
    started = time.time()
    print(json.dumps({'phase': phase, 'status': 'RUNNING'}), flush=True)
    with run_context(directory, config, None, until_complete=True) as guard:
        if phase in ('encode', 'rank'):
            run_dense(config, output, phase, MODEL, guard)
        elif phase == 'encode-support':
            encode_support(config, output, MODEL, guard)
        else:
            rank_frozen_support(config, output, MODEL, guard)
    atomic_json(receipt, {'phase': phase, 'elapsed_seconds': round(time.time() - started, 3)})
    print(json.dumps({'phase': phase, 'status': 'COMPLETED'}), flush=True)


def document_ranks(folder, source, ks):
    state = read(folder / 'progress.json')
    if state['status'] != 'COMPLETED' or sha256(folder / 'metrics.json') != state['metrics_sha256']:
        raise ValueError(f'Unverified ranking completion: {folder}')
    ids = {row['document_id']: i for i, row in enumerate(json_lines(source / 'corpus.jsonl'))}
    cursor = 0
    for block in state['blocks']:
        if block['start'] != cursor or sha256(folder / block['file']) != block['sha256']:
            raise ValueError(f'Invalid committed rank block: {folder}')
        cursor = block['stop']
    predictions = chain.from_iterable(json_lines(folder / block['file']) for block in state['blocks'])
    count = 0
    for query, pred in zip(json_lines(source / 'queries.jsonl'), predictions, strict=True):
        if (query['query_id'], query['split']) != (pred['query_id'], pred['split']):
            raise ValueError(f'Ranked query differs from the frozen query: {query["query_id"]}')
        ranks = pred['aggregations']['cosine']['positive_ranks']
        top = pred['aggregations']['cosine']['top_corpus_indices']
        positives = query['positive_ids']
        expected_top = min(max(ks), len(ids)) if positives else 0
        if (len(ranks) != len(positives) or len(top) != expected_top or len(set(top)) != len(top)
                or any(not 0 <= index < len(ids) for index in top)
                or any(not isinstance(rank, int) or not 1 <= rank <= len(ids) for rank in ranks)):
            raise ValueError(f'Invalid saved document ranks: {query["query_id"]}')
        for positive, rank in zip(positives, ranks, strict=True):
            position = ids[positive]
            if (rank <= len(top) and top[rank - 1] != position) or (rank > len(top) and position in top):
                raise ValueError(f'Positive rank disagrees with saved top candidates: {query["query_id"]}')
        count += 1
        yield pred
    if count != cursor or count != state['completed']:
        raise ValueError(f'Invalid ranking coverage: {folder}')


def support_metrics_from_ranks(config, output, ks):
    frozen = Path(config['supportqa_run'])
    documents, _, cases, annotation_hash = support_cases(frozen)
    saved = read(output / 'support' / 'metrics.json')
    predictions = {}
    for model in (*MODELS, MODEL):
        receipt = saved['predictions'][model]
        path = Path(receipt['file'])
        rows = list(json_lines(path))
        by_query = {row['query_id']: row for row in rows}
        if sha256(path) != receipt['sha256'] or len(rows) != len(cases) or by_query.keys() != cases.keys():
            raise ValueError(f'Invalid SupportQA prediction receipt for {model}')
        predictions[model] = by_query
    if annotation_hash != saved['input_annotations_sha256']:
        raise ValueError('BGE scoring used different SupportQA annotations')
    prior_common = set.intersection(*({qid for qid, row in predictions[model].items()
                                       if row['metrics'] is not None} for model in MODELS))
    common = prior_common & {qid for qid, row in predictions[MODEL].items() if row['metrics'] is not None}
    if len(prior_common) != saved['prior_common_eligible_queries'] or sorted(common) != saved['common_eligible_query_ids']:
        raise ValueError('Four-model SupportQA intersection differs from the saved result')
    if Counter(predictions[MODEL][qid]['eligibility'] for qid in prior_common - common) != saved['additional_common_exclusions']:
        raise ValueError('Additional common eligibility exclusions differ from the saved result')
    for source in ('nq', 'squad'):
        counts = Counter(row['eligibility'] for qid, row in predictions[MODEL].items() if cases[qid]['dataset'] == source)
        if counts != saved['method_eligibility'][source]:
            raise ValueError(f'SupportQA eligibility counts differ for {source}')

    reference_ids = {}
    groups, common_groups = defaultdict(list), defaultdict(list)
    for qid, row in predictions[MODEL].items():
        case = cases[qid]
        if case['support_type'] not in CLASSES:
            if row['eligibility'] != case['support_type'] or row['metrics'] is not None:
                raise ValueError(f'SupportQA exclusion changed: {qid}')
            continue
        if row['metrics'] is None:
            if row['eligibility'] == 'eligible':
                raise ValueError(f'Missing metrics for eligible SupportQA question: {qid}')
            continue
        doc_id = case['document_id']
        if doc_id not in reference_ids:
            decomposition = read_decomposition(frozen, documents[doc_id])
            reference_ids[doc_id] = {p['id'] for p in decomposition['reference_propositions']}
        ranked = {p: i for i, p in enumerate(row['ranked_ids'], 1)}
        if (ranked.keys() != reference_ids[doc_id] or len(ranked) != len(row['ranked_ids'])
                or len(ranked) != row['candidate_count'] or row['eligibility'] != 'eligible'):
            raise ValueError(f'Invalid direct proposition ranking: {qid}')
        ranks = [[ranked[p] for p in support] for support in case['support_sets']]
        alternatives = row['metrics']['alternatives']
        if (case['support_sets'] != [a['support_ids'] for a in alternatives]
                or ranks != [a['ranks'] for a in alternatives]):
            raise ValueError(f'Ranked silver alternatives changed: {qid}')
        metrics = {'MRR': 1 / min(min(items) for items in ranks)}
        for k in ks:
            metrics[f'Recall@{k}'] = max(sum(r <= k for r in items) / len(items) for items in ranks)
            metrics[f'all_required_Success@{k}'] = float(any(max(items) <= k for items in ranks))
        for source in (case['dataset'], 'combined'):
            key = f'{source}/{case["support_type"]}'
            groups[key].append(metrics)
            if qid in common:
                common_groups[key].append(metrics)
    metric_keys = {'MRR', *(f'Recall@{k}' for k in ks), *(f'all_required_Success@{k}' for k in ks)}
    group_keys = {f'{source}/{label}' for source in ('nq', 'squad', 'combined') for label in CLASSES}
    for field, verified in [('groups', groups), ('common_eligible_groups', common_groups)]:
        if saved[field].keys() != group_keys:
            raise ValueError(f'Missing SupportQA groups in {field}')
        for group, summaries in saved[field].items():
            values = verified[group]
            summary = summaries if field == 'groups' else summaries[MODEL]
            if len(values) != summary['metric_queries']:
                raise ValueError(f'Unexpected SupportQA metric denominator: {group}')
            if (summary['metrics'] or {}).keys() != (metric_keys if values else set()):
                raise ValueError(f'Incomplete SupportQA metric row: {group}')
            for metric, score in (summary['metrics'] or {}).items():
                expected = math.fsum(value[metric] for value in values) / len(values)
                if not math.isclose(score, expected, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f'SupportQA rank metric mismatch: {group}, {metric}')
    return saved


def verify(config, output, preflight_info):
    previous = prior_results(Path(config['supportqa_run']))
    if previous != preflight_info['previous_result_hashes']:
        raise ValueError('Previous result or alignment hashes changed since preflight')
    paths, manifests = source_paths(config)
    if manifests != preflight_info['benchmark_manifests']:
        raise ValueError('Processed benchmark inputs differ from preflight')
    for folder, text_count in ((output / 'retrieval' / MODEL / 'encoding', 'unique_benchmark_texts'),
                               (output / 'support' / MODEL / 'encoding', 'new_support_texts')):
        if open_vectors(folder).shape != (preflight_info[text_count], config['baselines'][MODEL]['dimension']):
            raise ValueError(f'Unexpected BGE embedding dimensions: {folder}')
    ks = config['retrieval']['ks']
    document_results = {}
    for dataset, source in paths.items():
        folder = output / 'retrieval' / MODEL / dataset
        metrics = aggregate_splits(document_ranks(folder, source, ks), ('cosine',), ks)
        if metrics != read(folder / 'metrics.json')['by_split']:
            raise ValueError(f'Document rank metrics differ for {dataset}')
        document_results[dataset] = metrics
    support = support_metrics_from_ranks(config, output, ks)
    results = {'status': 'VERIFIED', 'document_retrieval': document_results, 'within_context_support': support}
    atomic_json(output / 'results.json', results)
    receipt = {'status': 'VERIFIED', 'previous_result_files_unchanged': len(previous),
               'common_eligible_queries': len(support['common_eligible_query_ids']),
               'results_sha256': sha256(output / 'results.json')}
    atomic_json(output / 'verification.json', receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'verify', 'all', *PHASES))
    args = parser.parse_args()
    config = read(CONFIG)
    output = Path(config['runs_dir'])
    output.mkdir(parents=True, exist_ok=True)
    with Path('runs/benchmark-suite/suite.lock').open('r') as suite_lock:
        fcntl.flock(suite_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'preflight':
            preflight(config, output)
            return
        preflight_path = output / 'preflight.json'
        preflight_info = read(preflight_path) if preflight_path.exists() else {}
        if (preflight_info.get('config_sha256') != sha256(CONFIG)
                or preflight_info.get('uv_lock_sha256') != sha256(Path('uv.lock'))):
            raise ValueError('Run the BGE preflight for this configuration first')
        check_checkpoint(config, output)
        if args.phase == 'verify':
            verify(config, output, preflight_info)
            return
        for phase in PHASES if args.phase == 'all' else (args.phase,):
            run_phase(config, output, phase)


if __name__ == '__main__':
    main()
