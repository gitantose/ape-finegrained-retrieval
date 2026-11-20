"""Assemble SupportQA and preserved benchmark evidence without rescoring APE."""

import json
from pathlib import Path
import shutil

from ape_retrieval.benchmarks.dense import DATASETS
from ape_retrieval.benchmarks.runtime import exclusive_disk_bytes
from ape_retrieval.encode import atomic_json
from ape_retrieval.propsearch import sha256


MODELS = ('simcse', 'contriever')
APE_NAMES = {'HA': 'HA', 'MaxSim': 'symmetric MaxSim', 'H2P': 'query-as-hypothesis H2P'}


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def document_results(config, output):
    suite_dir = Path(read(Path(config['retrieval']['config']))['runs_dir'])
    historical = read(suite_dir / 'reports' / 'results.json')['benchmarks']
    expected = read(Path(config['reuse_from']) / 'closure.json')['benchmark_integrity']
    comparisons = {}
    for dataset in DATASETS:
        suite = historical[dataset]
        path = Path(suite['run_dir']) / 'retrieval' / 'metrics.json'
        if sha256(path) != expected[dataset]['metrics_sha256']:
            raise ValueError(f'Preserved APE result changed: {dataset}')
        ape = read(path)['by_split']
        results = {'ape': {'status': 'COMPLETED', 'metrics_source': str(path), 'by_split': ape}}
        for model in MODELS:
            folder = output / 'retrieval' / model / dataset
            receipt = read(folder / 'progress.json')
            path = folder / 'metrics.json'
            if receipt.get('status') == 'COMPLETED':
                if sha256(path) != receipt['metrics_sha256']:
                    raise ValueError(f'Baseline result changed: {dataset}/{model}')
                metrics = read(path)['by_split']
                if metrics.keys() != ape.keys():
                    raise ValueError(f'Baseline query splits differ: {dataset}/{model}')
                for split, rules in metrics.items():
                    expected_row, baseline = next(iter(ape[split].values())), next(iter(rules.values()))
                    if any(expected_row[k] != baseline[k] for k in ('queries', 'eligible_queries', 'queries_without_qrels')):
                        raise ValueError(f'Baseline query denominator differs: {dataset}/{model}/{split}')
                results[model] = {'status': 'COMPLETED', 'metrics_source': str(path), 'by_split': metrics}
            else:
                results[model] = {'status': receipt.get('status', 'PENDING'), 'by_split': None,
                                  'ranked_queries': receipt.get('completed', 0)}
        comparisons[dataset] = {'main_split': suite['main_split'], 'populations': expected[dataset]['processed'],
                                'models': results}
    return comparisons


def markdown_table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
            *['| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows], '']


def display_metric(metrics, key):
    return f'{metrics[key]:.6f}' if metrics else '—'


def report(config, output):
    documents = document_results(config, output)
    selection = read(output / 'selection.json')
    annotations = read(output / 'annotation-summary.json')
    support = read(output / 'support' / 'metrics.json')
    if support and support['input_annotations_sha256'] != annotations.get('queries_sha256'):
        support = {}
    state = read(output / 'status.json')
    annotation_process = read(output / 'annotation-process.json')
    status, reason = state.get('status', 'PENDING'), state.get('reason')
    if (status == 'BLOCKED' and reason == 'Waiting for complete Codex semantic annotation'
            and annotation_process.get('status') in ('RUNNING', 'DRAINING')):
        status, reason = 'RUNNING', None
    feasibility = read(output / 'feasibility.json')
    storage_roots = [Path(path) for path in config.get('storage_roots', [output])]
    storage = {'exclusive_allocated_bytes': exclusive_disk_bytes(*storage_roots),
               'remaining_free_disk_bytes': shutil.disk_usage(output).free,
               'reclaimed_bytes': feasibility.get('disk', {}).get('measured_total_bytes_reclaimed')}
    results = {'status': status, 'reason': reason,
               'source_selection': selection.get('datasets'), 'document_retrieval': documents,
               'support_annotations': annotations, 'within_context_support': support or None,
               'phases': state.get('phases', {}), 'annotation_process': annotation_process,
               'checkpoint_preflight': read(output / 'checkpoint-preflight.json'), 'feasibility': feasibility,
               'baseline_inference_preflight': read(output / 'baseline-inference-preflight.json'),
               'storage': storage, 'limits': config['limits'], 'checkpoints': config['baselines']}
    atomic_json(output / 'results.json', results)
    title = 'STATUS' if results['status'] == 'RUNNING' else 'FINAL STATUS'
    lines = ['# SupportQA and frozen retrieval comparisons', '', f'**{title} {results["status"]}**', '']
    if results['reason']:
        lines += [results['reason'] + '.', '']
    active = state.get('current_phase')
    if active:
        lines += [f'Current phase is `{active}`. Phase receipts and live progress are linked below.', '']
    work = [(phase, entry['status'], entry.get('reason') or '—') for phase, entry in results['phases'].items()]
    if work:
        lines += markdown_table(('Phase', 'Status', 'Reason'), work)
    lines += ['## SupportQA coverage', '']
    coverage = []
    cases = ('single_support', 'multi_support', 'insufficient_document', 'decomposition_error', 'ambiguous', 'unreviewed')
    for source, selected in (results['source_selection'] or {}).items():
        summary = annotations.get('by_dataset', {}).get(source, {})
        coverage.append((source, selected['selected_contexts'], selected['selected_questions'],
                         *(summary.get('classes', {}).get(label, 0) for label in cases)))
    lines += markdown_table(('Source', 'Contexts', 'Questions', 'Single', 'Multi', 'Insufficient document',
                             'Decomposition error', 'Ambiguous', 'Unreviewed'), coverage)
    exclusions = [(source, key, count) for source, selected in (results['source_selection'] or {}).items()
                  for key, count in selected.items()
                  if key.startswith('questions_excluded') or key in {
                      'invalid_questions', 'excluded_reference_context', 'empty_context', 'context_over_128_tokens',
                      'no_eligible_existing_question', 'already_selected_other_dataset'}]
    if exclusions:
        lines += ['### Eligibility exclusions', ''] + markdown_table(('Source', 'Reason', 'Count'), exclusions)
    for source, summary in annotations.get('by_dataset', {}).items():
        common_flags = sorted(summary['quality_flags'].items(), key=lambda pair: (-pair[1], pair[0]))[:8]
        lines += [f'**{source}** generation status ' + ', '.join(f'{v} {k}' for k, v in summary['generation'].items()) + '.', '',
                  'Most frequent context quality flags ' + (', '.join(f'{v} {k}' for k, v in common_flags) or 'none recorded') + '.', '']
    lines += ['The original native text, offsets and eligibility dispositions are in the linked records. Quality flags may overlap. '
              'Codex reviews are silver proposals and human acceptance is pending.', '',
              '## Document retrieval', '',
              'These are separate full candidate corpora. Main-split results are shown below. Results for every evaluated '
              'native split, full first-positive MRR, set recall and query eligibility are in the machine-readable table.', '']
    metrics = ['MRR', *(f'Recall@{k}' for k in config['retrieval']['ks'])]
    rows = []
    for dataset, table in documents.items():
        split = table['main_split']
        for model, item in table['models'].items():
            if item['by_split'] is None:
                rows.append((dataset, split, model, item['status'], '—', *['—'] * len(metrics)))
                continue
            for rule, record in item['by_split'][split].items():
                count = f'{record["eligible_queries"]}/{record["queries"]}'
                rows.append((dataset, split, model, APE_NAMES.get(rule, rule), count,
                             *(display_metric(record['metrics'], m) for m in metrics)))
    lines += markdown_table(('Benchmark', 'Main split', 'Model', 'Scoring', 'Eligible/total', *metrics), rows)
    lines += ['## Within-context support', '']
    if support:
        success = [f'all_required_Success@{k}' for k in config['retrieval']['ks']]
        for title, key in [('Model-eligible queries', 'groups'), ('Common eligible queries', 'common_eligible_groups')]:
            support_rows = []
            for group, models in support[key].items():
                for model, item in models.items():
                    support_rows.append((group, model, item['metric_queries'],
                        item['labeled_queries'] - item['metric_queries'],
                        *(display_metric(item['metrics'], m) for m in metrics + success),
                        item['unmapped_support_occurrences_across_alternatives']))
            lines += [f'### {title}', '']
            if key == 'common_eligible_groups':
                lines += ['Each row uses queries that all three models could score within that source and support class.', '']
            lines += markdown_table(('Source / class', 'Model', 'Scored', 'Excluded from class', *metrics,
                *(s.replace('all_required_', 'All ') for s in success), 'Unmapped support occurrences'), support_rows)
        model_exclusions = [(source, model, reason, count)
            for source, models in support['method_eligibility'].items()
            for model, outcomes in models.items() for reason, count in outcomes.items() if reason != 'eligible']
        lines += ['### Scoring exclusions by outcome', '']
        lines += markdown_table(('Source', 'Model', 'Outcome', 'Queries'), model_exclusions)
        for source, slots in support['active_slots_per_context'].items():
            lines += [f'APE active slots per {source} context, shown as slots → contexts. '
                      + ', '.join(f'{k} → {v}' for k, v in sorted(slots.items(), key=lambda item: int(item[0]))) + '.', '']
        saturated = [f'{group}/{model} at k={k}' for group, models in support['groups'].items()
            for model, result in models.items() for k, count in result['queries_with_more_candidates_than_k'].items()
            if result['metric_queries'] and not count]
        if saturated:
            lines += ['The following cutoffs admit every ranked candidate for every eligible query in the row. '
                      + ', '.join(saturated) + '.', '']
    else:
        lines += ['SupportQA metrics become available after complete coverage and final scoring.', '']
    lines += ['Each model is given the original question and supplied correct context. APE ranks context slots and uses a '
        'query-independent teacher-space assignment as a provisional proxy. SimCSE and Contriever score generated '
        'proposition text directly. The different candidate representations and teacher assignment limit comparability.', '',
        'Each sufficient set contributes its own metrics, with the best valid alternative per metric and then a mean '
        'over eligible queries. The combined source row weights queries equally. Unmapped APE IDs stay in the support '
        'denominator as misses. First-positive MRR can be high even when full multi-support completion fails. '
        'Active-slot agreement with generated string counts cannot establish atom-counting accuracy.', '',
        '## Execution and artifacts', '',
        f'Allocated new storage is {storage["exclusive_allocated_bytes"]:,} bytes. '
        f'Free space is {storage["remaining_free_disk_bytes"]:,} bytes. '
        f'Disposable file reclamation was {storage["reclaimed_bytes"] or 0:,} bytes.', '']
    resource_rows = []
    for phase, value in results['phases'].items():
        for number, attempt in enumerate(value['attempts'], 1):
            usage = attempt.get('resources', {})
            resource_rows.append((phase, number, attempt.get('elapsed_seconds', '—'),
                                  usage.get('host_peak_rss_bytes', '—'), usage.get('cuda_peak_allocated_bytes', '—')))
    if resource_rows:
        lines += markdown_table(('Phase', 'Attempt', 'Elapsed s', 'Host peak bytes', 'CUDA allocated peak bytes'), resource_rows)
    annotation_process = results['annotation_process']
    if annotation_process:
        lines += [f'Codex review process status is {annotation_process["status"]}. It has committed '
                  f'{annotation_process["reviewed_queries"]:,} of {annotation_process["query_total"]:,} query reviews '
                  'including imported reference-v3 reviews.', '']
        if reason := annotation_process.get('reason'):
            lines += [reason + '.', '']
    if preflight := results['checkpoint_preflight']:
        lines += [f'Checkpoint preflight status is {preflight["status"]}.', '']
    if results['baseline_inference_preflight']:
        lines += ['Both official checkpoints passed a two-input CPU inference check. The frozen CUDA comparisons '
                  'have their own execution receipts.', '']
    links = ['results.json', 'status.json', 'progress.log', 'errors.log', 'feasibility.json', 'selection.json',
             'exclusions.jsonl', 'context-reviews.jsonl', 'annotations.jsonl', 'annotation-summary.json',
             'annotation-process.json', 'checkpoint-preflight.json', 'baseline-inference-preflight.json', 'cache-reuse.json']
    lines += [' · '.join(f'[{name}]({name})' for name in links if (output / name).exists()), '']
    path = output / 'report.md'
    temp = path.with_suffix('.md.tmp')
    temp.write_text('\n'.join(lines))
    temp.replace(path)
    return results
