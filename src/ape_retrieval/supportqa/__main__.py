"""Prepare and run the full held-out SupportQA experiment."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from ape_retrieval.benchmarks.encoding import tokenizer_from_reference
from ape_retrieval.benchmarks.runtime import run_context
from ape_retrieval.encode import atomic_json
from ape_retrieval.supportqa import dataset_path, decompose, reuse_cached_outputs, select
from ape_retrieval.propsearch import sha256


CONFIG = Path('configs/supportqa.json')
PROTOCOL = Path('data/supportqa/annotation-protocol.md')
COMPUTE_PHASES = ('prepare', 'decompose', 'acquire', 'encode-simcse', 'rank-simcse',
                  'encode-contriever', 'rank-contriever', 'align', 'support-simcse', 'support-contriever')


def read(path):
    return json.loads(path.read_text())


def prepare_selection(config, output):
    dataset_directory = Path(config['dataset_dir']).resolve()
    dataset_directory.mkdir(parents=True, exist_ok=True)
    attached = output / 'dataset'
    if not attached.is_symlink():
        attached.symlink_to(os.path.relpath(dataset_directory, output.resolve()), target_is_directory=True)
    if dataset_path(output) != dataset_directory:
        raise ValueError('SupportQA dataset directory changed')
    decompositions = dataset_directory / 'decompositions'
    if not decompositions.is_symlink():
        decompositions.symlink_to(os.path.relpath((output / 'decompositions').resolve(), dataset_directory),
                                  target_is_directory=True)
    if decompositions.resolve() != (output / 'decompositions').resolve():
        raise ValueError('SupportQA decomposition directory changed')
    suite = read(Path(config['retrieval']['config']))
    reference = read(Path(config['reference']))
    previous = Path(config['reuse_from'])
    excluded = read(Path(config['selection']['exclude_contexts_file']))
    result = select(suite, output, tokenizer_from_reference(reference),
                    config['selection']['contexts_per_dataset'],
                    {c['context_sha256'] for c in excluded['contexts']})
    if not result['complete_quota']:
        raise ValueError('Full eligible population was not selected')
    for path, digest in result['source_files'].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f'Native source changed: {path}')
    closure = read(previous / 'closure.json')
    for name, digest in closure['artifacts'].items():
        if sha256(previous / name) != digest:
            raise ValueError(f'SupportQA closure artifact changed: {name}')
    reuse = reuse_cached_outputs(config, output)
    print(json.dumps({'selection': result['datasets'], 'reused_contexts': reuse['reused_contexts'],
                      'new_contexts': reuse['new_contexts']}), flush=True)


def validate_generation_inputs(config, output):
    selection = read(output / 'selection.json')
    if selection['quota_per_dataset'] is not None or not selection['complete_quota']:
        raise ValueError('Generation requires the frozen, full eligible selection')
    for filename, digest in selection['files'].items():
        if sha256(dataset_path(output) / filename) != digest:
            raise ValueError(f'Selection artifact changed: {filename}')
    for filename, digest in read(output / 'model-files.json').items():
        if sha256(Path(config['propositionizer']['directory']) / filename) != digest:
            raise ValueError(f'Propositionizer file changed: {filename}')


def run_phase(config, output, phase):
    output.mkdir(parents=True, exist_ok=True)
    with Path('runs/benchmark-suite/suite.lock').open('r') as suite_lock:
        fcntl.flock(suite_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / 'status.json'
        state = read(state_path) if state_path.exists() else {'phases': {}}
        manifest_path = output / 'launch.json'
        snapshot = output / 'configuration.json'
        if snapshot.exists() and read(snapshot) != config:
            raise ValueError('Frozen SupportQA configuration changed')
        if not snapshot.exists():
            atomic_json(snapshot, config)
        manifest = {
            'configuration_sha256': sha256(snapshot), 'annotation_protocol_sha256': sha256(PROTOCOL),
            'reference_sha256': sha256(Path(config['reference'])),
            'dependency_lock_sha256': sha256(Path('uv.lock')),
            'retrieval_configuration_sha256': sha256(Path(config['retrieval']['config'])),
            'authorization': config['authorization'], 'deadline_epoch': None,
            'created_epoch': time.time(), 'working_directory': str(Path.cwd()),
        }
        if manifest_path.exists():
            saved = read(manifest_path)
            if any(saved[key] != value for key, value in manifest.items() if key != 'created_epoch'):
                raise ValueError('SupportQA launch identity changed')
        else:
            atomic_json(manifest_path, manifest)
        started = time.time()
        entry = state['phases'].setdefault(phase, {'attempts': []})
        attempt = {'pid': os.getpid(), 'process_group': os.getpgrp(), 'started_epoch': started,
                   'command': [sys.executable, '-m', 'ape_retrieval.supportqa', phase]}
        entry['attempts'].append(attempt)
        state.update(status='RUNNING', current_phase=phase, reason=None)
        entry['status'] = 'RUNNING'
        atomic_json(state_path, state)
        print(json.dumps({'phase': phase, 'status': 'RUNNING', 'pid': os.getpid()}), flush=True)
        try:
            with run_context(output, config, None, until_complete=True) as guard:
                if phase == 'decompose':
                    validate_generation_inputs(config, output)
                execute_phase(config, output, phase, guard)
        except InterruptedError as exc:
            entry.update(status='BLOCKED', reason=str(exc))
            raise
        except BaseException as exc:
            entry.update(status='FAILED', reason=str(exc))
            raise
        else:
            entry.update(status='COMPLETED', reason=None)
        finally:
            usage_path = output / 'resources.json'
            usage = read(usage_path) if usage_path.exists() else {}
            if usage.get('updated_epoch', 0) >= started:
                attempt['resources'] = usage
            attempt.update(status=entry['status'], elapsed_seconds=round(time.time() - started, 3))
            finished = phase == 'finalize' and entry['status'] == 'COMPLETED'
            experiment_status = entry['status'] if finished or entry['status'] != 'COMPLETED' else 'BLOCKED'
            state.update(status=experiment_status, reason=None if finished else entry['reason'] or
                         'Waiting for remaining experiment phases', current_phase=None)
            atomic_json(state_path, state)
            if phase == 'finalize':
                from ape_retrieval.supportqa.supportqa_report import report
                try:
                    report(config, output)
                except Exception as exc:
                    entry.update(status='FAILED', reason=f'SupportQA report export failed: {exc}')
                    attempt['status'] = 'FAILED'
                    state.update(status='FAILED', reason=entry['reason'])
                    atomic_json(state_path, state)
                    raise
            print(json.dumps({'phase': phase, 'status': entry['status'], **attempt}), flush=True)
            if phase == 'finalize':
                print(f'FINAL STATUS {experiment_status} • {output / "report.md"}', flush=True)
                if finished:
                    print('Completed APE, SimCSE and Contriever on NQ, SQuAD, PROPSEARCH and within-context SupportQA.', flush=True)
                print(f'Results {output / "results.json"} • reviews {output / "annotations.jsonl"}', flush=True)
                print(f'cat {output / "report.md"}', flush=True)


def execute_phase(config, output, phase, guard):
    if phase == 'prepare':
        prepare_selection(config, output)
    elif phase == 'decompose':
        decompose(config, output, guard)
    elif phase == 'acquire':
        from ape_retrieval.benchmarks.dense import acquire
        acquire(config, output, guard)
    elif phase == 'align':
        from ape_retrieval.supportqa.diagnostics import diagnose
        diagnose(config, output, guard)
    elif phase == 'finalize':
        from ape_retrieval.supportqa.supportqa_annotations import materialize
        from ape_retrieval.supportqa.supportqa_evaluation import evaluate_support
        from ape_retrieval.supportqa.supportqa_report import document_results
        with (output / 'annotation.lock').open('a+') as annotation_lock:
            while True:
                try:
                    fcntl.flock(annotation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if guard():
                        raise InterruptedError('Stopped while waiting for semantic review')
                    time.sleep(10)
            materialize(config, output, sha256(PROTOCOL))
        completed = read(output / 'status.json')['phases']
        missing = [p for p in COMPUTE_PHASES if completed.get(p, {}).get('status') != 'COMPLETED']
        if missing:
            raise InterruptedError('Pending experiment phases ' + ', '.join(missing))
        if any(model['status'] != 'COMPLETED' for result in document_results(config, output).values()
               for model in result['models'].values()):
            raise ValueError('Document ranking metrics are incomplete')
        evaluate_support(config, output, guard)
    else:
        action, model = phase.split('-')
        if action == 'support':
            from ape_retrieval.supportqa.supportqa_evaluation import encode_support
            encode_support(config, output, model, guard)
        else:
            from ape_retrieval.benchmarks.dense import run_dense
            run_dense(config, output, action, model, guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=(*COMPUTE_PHASES, 'finalize', 'report', 'annotate'))
    parser.add_argument('--once', action='store_true', help='Process one semantic review batch')
    parser.add_argument('--batch-size', type=int, default=14, help='Maximum contexts per Codex review batch')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=1, help='Maximum concurrent Codex calls')
    args = parser.parse_args()
    config = read(CONFIG)
    output = Path(config['runs_dir'])
    status = output / 'status.json'
    if status.exists() and read(status)['status'] == 'COMPLETED':
        print((output / 'report.md').read_text(), end='')
        return
    if args.phase == 'annotate':
        if args.batch_size <= 0:
            parser.error('annotation batch size must be positive')
        from ape_retrieval.supportqa.supportqa_annotations import run_codex_reviews
        run_codex_reviews(config, output, PROTOCOL, once=args.once, batch_size=args.batch_size, workers=args.workers)
    elif args.phase == 'report':
        from ape_retrieval.supportqa.supportqa_annotations import materialize
        from ape_retrieval.supportqa.supportqa_report import report
        summary = output / 'annotation-summary.json'
        if not summary.exists() or read(summary)['status'] != 'COMPLETE':
            materialize(config, output, sha256(PROTOCOL))
        report(config, output)
    else:
        run_phase(config, output, args.phase)


if __name__ == '__main__':
    main()
