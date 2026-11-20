"""Small, provenance-preserving NQ/SQuAD correct-context diagnostic reference."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from ape_retrieval.benchmarks.data import answer_audit, json_lines, source_rows
from ape_retrieval.encode import atomic_json, sync_directory
from ape_retrieval.propsearch import sha256


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def selection_order(dataset, context_hash, seed=42):
    return text_hash(f'{seed}|{dataset}|{context_hash}'), context_hash


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def dataset_path(output):
    """Resolve the external dataset attached to a run, or its local fixture."""
    attached = output / 'dataset'
    return attached.resolve(strict=True) if attached.is_symlink() else output


def select(config, output, tokenizer, quota=100, excluded_hashes=()):
    """Freeze whole contexts before any generator/APE performance is observed."""
    output.mkdir(parents=True, exist_ok=True)
    dataset_directory = dataset_path(output)
    if (output / 'selection.json').exists():
        saved = json.loads((output / 'selection.json').read_text())
        if saved['quota_per_dataset'] != quota or saved['excluded_previous_context_hashes'] != sorted(excluded_hashes):
            raise ValueError('Frozen selection policy changed')
        for name, digest in saved['files'].items():
            if sha256(dataset_directory / name) != digest:
                raise ValueError('Frozen selection corruption')
        return saved
    documents, queries, excluded, summaries = [], [], [], {}
    selected_hashes = set()
    for dataset, split in [('nq', 'test'), ('squad', 'validation')]:
        spec = config['datasets'][dataset]
        contexts = {}
        counts = Counter()
        paths = spec['splits'][split]
        for filename in paths:
            for index, row in enumerate(source_rows(Path(spec['raw_dir']), [filename])):
                counts['native_rows'] += 1
                text = row['context']
                digest = text_hash(text)
                native = row.get('id', index)
                origin = {'dataset': dataset, 'revision': spec['revision'], 'split': split,
                          'source_file': filename, 'original_query_id_or_row': native,
                          'document_group': row.get('title') or None,
                          'group_provenance': 'native_title_only' if row.get('title') else 'unavailable'}
                context = contexts.setdefault(digest, {'document_id': 'd:' + digest, 'document': text,
                    'context_sha256': digest, 'dataset': dataset, 'source_occurrences': [], 'questions': []})
                context['source_occurrences'].append(origin)
                audit = answer_audit(row)
                question = {'query_id': f'{dataset}:{split}:{native}', 'query': row['question'],
                    'answers': row['answers']['text'], 'answer_offsets': row['answers'],
                    'answer_span_audit': audit, 'document_id': context['document_id'], 'source': origin}
                if not row['question'].strip() or not any(a['valid'] for a in audit):
                    excluded.append({'dataset': dataset, 'query_id': question['query_id'],
                                     'reason': 'empty_question_or_no_valid_native_answer_offset'})
                    counts['invalid_questions'] += 1
                else:
                    context['questions'].append(question)
        eligible = []
        for digest, context in contexts.items():
            length = len(tokenizer(context['document'])['input_ids'])
            context['ape_tokens'] = length
            reason = ('excluded_reference_context' if digest in excluded_hashes else
                      'empty_context' if not context['document'].strip() else
                      'context_over_128_tokens' if length > 128 else
                      'no_eligible_existing_question' if not context['questions'] else
                      'already_selected_other_dataset' if digest in selected_hashes else None)
            if reason:
                excluded.append({'dataset': dataset, 'context_sha256': digest, 'ape_tokens': length, 'reason': reason})
                counts[reason] += 1
                if quota is None:
                    for question in context['questions']:
                        excluded.append({'dataset': dataset, 'query_id': question['query_id'],
                                         'context_sha256': digest, 'reason': reason, 'source': question['source']})
                        counts['questions_excluded_' + reason] += 1
            else:
                eligible.append(context)
        counts['distinct_contexts'] = len(contexts)
        counts['eligible_contexts'] = len(eligible)
        chosen = sorted(eligible, key=lambda c: selection_order(dataset, c['context_sha256']))[:quota]
        counts['selected_contexts'] = len(chosen)
        for context in chosen:
            ordered = sorted(context.pop('questions'), key=lambda q: q['source']['original_query_id_or_row'])
            questions = ordered[:2]
            if quota is None:
                for question in ordered[2:]:
                    excluded.append({'dataset': dataset, 'query_id': question['query_id'],
                                     'context_sha256': context['context_sha256'],
                                     'reason': 'question_cap', 'source': question['source']})
                    counts['questions_excluded_question_cap'] += 1
            for q in questions:
                q.update({'support_proposition_ids': [], 'support_proposition_sets': [],
                          'support_type': 'ambiguous', 'annotation_status': 'automatic_draft',
                          'support_label_status': 'pending_decomposition_and_review', 'review_notes': []})
            queries.extend(questions)
            documents.append(context)
            selected_hashes.add(context['context_sha256'])
        counts['selected_questions'] = sum(q['source']['dataset'] == dataset for q in queries)
        summaries[dataset] = dict(counts)
    # Preserve all known split occurrences for selected contexts, including overlap.
    for dataset in ('nq', 'squad'):
        lookup = {d['context_sha256']: d for d in documents if d['dataset'] == dataset}
        spec = config['datasets'][dataset]
        for split, filenames in spec['splits'].items():
            for filename in filenames:
                for index, row in enumerate(source_rows(Path(spec['raw_dir']), [filename])):
                    digest = text_hash(row['context'])
                    if digest in lookup:
                        occurrence = {'dataset': dataset, 'revision': spec['revision'], 'split': split,
                                      'source_file': filename, 'original_query_id_or_row': row.get('id', index),
                                      'document_group': row.get('title') or None,
                                      'group_provenance': 'native_title_only' if row.get('title') else 'unavailable'}
                        if occurrence not in lookup[digest]['source_occurrences']:
                            lookup[digest]['source_occurrences'].append(occurrence)
    write_rows(dataset_directory / 'documents.jsonl', documents)
    write_rows(dataset_directory / 'queries.jsonl', queries)
    write_rows(dataset_directory / 'exclusions.jsonl', excluded)
    record = {'format_version': 1, 'seed': 42, 'selection': 'SHA256(42|dataset|context_hash), then context_hash; NQ first',
              'context_tokens': 'whole context, pinned APE tokenizer including specials <=128',
              'question_eligibility': 'nonempty question and at least one exact native answer offset; order by native ID/numeric NQ row, first two',
              'datasets': summaries, 'quota_per_dataset': quota, 'excluded_previous_context_hashes': sorted(excluded_hashes),
              'complete_quota': all(v['selected_contexts'] == (v['eligible_contexts'] if quota is None else quota)
                                    for v in summaries.values()),
              'source_files': {str(Path(spec['raw_dir']) / f): sha256(Path(spec['raw_dir']) / f)
                  for name, spec in config['datasets'].items() if name in ('nq', 'squad')
                  for filenames in spec['splits'].values() for f in filenames},
              'reference': json.loads(Path(config['reference']).read_text()),
              'source_sha256': sha256(Path(__file__)),
              'files': {name: sha256(dataset_directory / name) for name in ('documents.jsonl', 'queries.jsonl', 'exclusions.jsonl')},
              'performance_used_for_selection': False, 'source_document_disjointness_claimed': False}
    atomic_json(output / 'selection.json', record)
    return record


def reuse_cached_outputs(config, output):
    """Link immutable compatible reference outputs."""
    previous = Path(config['reuse_from'])
    prior = json.loads((previous / 'configuration.json').read_text())
    for key in ('reference', 'teacher', 'diagnostics'):
        if prior[key] != config[key]:
            raise ValueError(f'Incompatible prior cache: {key}')
    if generation_spec(prior['propositionizer']) != generation_spec(config['propositionizer']):
        raise ValueError('Incompatible prior cache: propositionizer')
    provenance = json.loads((previous / 'diagnostic-provenance.json').read_text())
    if provenance['reference'] != json.loads(Path(config['reference']).read_text()):
        raise ValueError('Reference contents changed')
    if provenance['uv_lock_sha256'] != sha256(Path('uv.lock')):
        raise ValueError('Dependency identity changed')
    dataset_directory = dataset_path(output)
    documents = list(json_lines(dataset_directory / 'documents.jsonl'))
    old = {d['context_sha256']: d for d in json_lines(previous / 'documents.jsonl')}
    current = {d['context_sha256']: d for d in documents}
    if not old.keys() <= current.keys():
        raise ValueError('Extended selection lost previous contexts')
    prior_selection = json.loads((previous / 'selection.json').read_text())
    if sha256(previous / 'documents.jsonl') != prior_selection['files']['documents.jsonl']:
        raise ValueError('Prior selection has changed')
    linked = {}
    def link(source, target):
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            if sha256(target) != sha256(source):
                raise ValueError('Conflicting committed cache')
        else:
            target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))
        linked[str(target.relative_to(output))] = {'source': str(source), 'sha256': sha256(source)}
    for digest, doc in old.items():
        if doc['document'] != current[digest]['document']:
            raise ValueError('Context changed')
        source = previous / 'decompositions' / (digest + '.json')
        dec = json.loads(source.read_text())
        if (generation_spec(dec['spec']) != generation_spec(config['propositionizer']) or dec['context_sha256'] != digest
                or dec['input_text'] != config['propositionizer']['input_template'].format(context=doc['document'])
                or dec['output_sha256'] != text_hash(dec['raw_output'])):
            raise ValueError('Prior decomposition provenance mismatch')
        diagnostic = previous / 'diagnostics' / source.name
        if json.loads(diagnostic.read_text())['decomposition_output_sha256'] != dec['output_sha256']:
            raise ValueError('Prior matching decomposition mismatch')
        link(source, output / 'decompositions' / source.name)
        link(diagnostic, output / 'diagnostics' / source.name)
    texts = {d['document'] for d in documents} | {q['query'] for q in json_lines(dataset_directory / 'queries.jsonl')}
    for text in texts:
        source = previous / 'encodings' / (text_hash(text) + '.json')
        if source.exists():
            enc = json.loads(source.read_text())
            if enc['text_sha256'] != text_hash(text) or enc['reference'] != provenance['reference']:
                raise ValueError('Prior encoding provenance mismatch')
            link(source, output / 'encodings' / source.name)
    model_hashes = json.loads((previous / 'model-files.json').read_text())
    for name, digest in model_hashes.items():
        if sha256(Path(config['propositionizer']['directory']) / name) != digest:
            raise ValueError('Reused model hash mismatch')
    atomic_json(output / 'model-files.json', model_hashes)
    record = {'previous_run': str(previous), 'previous_config_sha256': sha256(previous / 'configuration.json'),
              'previous_budget_sha256': sha256(previous / 'budget.json'),
              'reused_contexts': len(old), 'new_contexts': len(current) - len(old),
              'method': 'relative symbolic links to immutable outputs; no data/model duplication', 'files': linked}
    atomic_json(output / 'cache-reuse.json', record)
    return record


def generation_spec(spec):
    return {key: value for key, value in spec.items() if key != 'directory'}


def parse_decomposition(raw, eos_reached, cap_hit):
    """Preserve failures: valid JSON is not semantic completeness."""
    try:
        value = json.loads(raw)
        valid = isinstance(value, list) and bool(value) and all(isinstance(v, str) and v.strip() for v in value)
    except json.JSONDecodeError:
        value, valid = [], False
    return {'reference_propositions': [{'id': f'p{i+1:03d}', 'text': text, 'supporting_source_spans': []}
                                      for i, text in enumerate(value)] if valid else [],
            'format_valid': valid, 'eos_reached': eos_reached, 'token_cap_hit': cap_hit,
            'potentially_incomplete': not valid or not eos_reached or cap_hit,
            'semantic_completeness': 'unreviewed'}


def decompose(config, output, should_stop):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, StoppingCriteria
    from ape_retrieval.model import configure_reference
    spec = config['propositionizer']
    reference = json.loads(Path(config['reference']).read_text())
    configure_reference(reference, 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(spec['directory'], local_files_only=True)
    documents = list(json_lines(dataset_path(output) / 'documents.jsonl'))
    destination = output / 'decompositions'
    destination.mkdir(exist_ok=True)
    for doc in documents:
        path = destination / (doc['context_sha256'] + '.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved['context_sha256'] != doc['context_sha256'] or generation_spec(saved['spec']) != generation_spec(spec)
                    or saved['input_text'] != spec['input_template'].format(context=doc['document'])
                    or saved['output_sha256'] != text_hash(saved['raw_output'])):
                raise ValueError('Decomposition identity or output hash changed')
    pending = [d for d in documents if not (destination / (d['context_sha256'] + '.json')).exists()]
    if not pending:
        return
    model = AutoModelForSeq2SeqLM.from_pretrained(spec['directory'], local_files_only=True,
                                                 dtype=torch.float32).eval().requires_grad_(False)
    class BudgetStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return should_stop()
    should_stop(True)
    started = time.monotonic()
    last_log = 0.0
    for completed, doc in enumerate(pending, 1):
        path = destination / (doc['context_sha256'] + '.json')
        if should_stop():
            raise InterruptedError('Generation stopped; committed contexts are recoverable')
        text = spec['input_template'].format(context=doc['document'])
        inputs = tokenizer(text, return_tensors='pt', truncation=False)
        if inputs['input_ids'].shape[1] > spec['max_input_tokens']:
            raise ValueError('Propositionizer input overflow; no truncation or substitute')
        start = time.time()
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=spec['max_new_tokens'],
                do_sample=False, num_beams=1, stopping_criteria=[BudgetStop()])[0].cpu()
        generated = tokens[1:].tolist()  # T5 decoder-start token is not a new token.
        eos = tokenizer.eos_token_id in generated
        if not eos and should_stop():
            raise InterruptedError('Generation stopped before EOS; interrupted output was not committed')
        cap = len(generated) >= spec['max_new_tokens']
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        record = {'document_id': doc['document_id'], 'context_sha256': doc['context_sha256'], 'spec': spec,
                  'input_text': text, 'input_tokens': inputs['input_ids'][0].tolist(),
                  'output_token_ids': tokens.tolist(), 'raw_output': raw, 'generated_tokens': len(generated),
                  'seconds': time.time() - start, 'annotation_status': 'automatic_draft',
                  **parse_decomposition(raw, eos, cap)}
        record['reference_n_atoms'] = len(record['reference_propositions'])
        record['output_sha256'] = text_hash(raw)
        atomic_json(path, record)
        now = time.monotonic()
        if now - last_log >= 45 or completed == len(pending) or record['potentially_incomplete']:
            rate = completed / (now - started)
            print(json.dumps({'decomposed_contexts': len(documents) - len(pending) + completed,
                              'total': len(documents), 'contexts_per_hour': round(3600 * rate, 1),
                              'eta_seconds': round((len(pending) - completed) / rate),
                              'context_sha256': doc['context_sha256'], 'eos': eos, 'cap_hit': cap,
                              'format_valid': record['format_valid']}), flush=True)
            last_log = now
        should_stop()
