"""Query-independent teacher-space matching and provisional reference diagnostics."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from ape_retrieval.benchmarks.data import json_lines
from ape_retrieval.encode import atomic_json
from ape_retrieval.model import configure_reference, load_model
from ape_retrieval.propsearch import sha256
from ape_retrieval.supportqa import dataset_path


def assignment(similarity):
    """Use SciPy optimal assignment, with exact lexicographic optimum tie resolution.

    A constrained sequence of library solves fixes the first feasible optimal
    pair; no custom Hungarian solver, epsilon or query-dependent positives.
    """
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or not np.isfinite(similarity).all():
        raise ValueError('Expected finite reference-by-slot cosine matrix')
    nrows, ncols = similarity.shape
    size = min(nrows, ncols)
    if not size:
        return []
    rows, cols = linear_sum_assignment(similarity, maximize=True)
    target = math.fsum(similarity[rows, cols].tolist())
    fixed, available = [], list(range(ncols))
    start = 0
    while len(fixed) < size:
        remaining = size - len(fixed) - 1
        found = False
        for row in range(start, nrows):
            for col in available:
                later_rows = list(range(row + 1, nrows))
                later_cols = [c for c in available if c != col]
                if min(len(later_rows), len(later_cols)) < remaining:
                    continue
                rest = []
                if remaining:
                    sub = similarity[np.ix_(later_rows, later_cols)]
                    ri, ci = linear_sum_assignment(sub, maximize=True)
                    rest = sub[ri, ci].tolist()
                total = math.fsum([similarity[r, c] for r, c in fixed] + [similarity[row, col]] + rest)
                if total == target:
                    fixed.append((row, col))
                    available.remove(col)
                    start = row + 1
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError('Exact assignment tie resolution failed; no approximate fallback')
    return fixed


def support_metrics(sufficient_sets, mapping, ranked_slots, ks=(1, 3, 5)):
    positions = {slot: i + 1 for i, slot in enumerate(ranked_slots)}
    alternatives = []
    for support in sufficient_sets:
        if not support or len(set(support)) != len(support):
            raise ValueError('Support sets must be nonempty unique proposition IDs')
        ranks = [positions[mapping[p]] if p in mapping else None for p in support]
        present = [r for r in ranks if r is not None]
        metrics = {'mapped_support_coverage': len(present) / len(support),
                   'first_support_reciprocal_rank': 1 / min(present) if present else 0,
                   'completion_rank': max(present) if len(present) == len(support) else None}
        for k in ks:
            metrics[f'Recall@{k}'] = sum(r is not None and r <= k for r in ranks) / len(support)
            metrics[f'all_required_Success@{k}'] = float(all(r is not None and r <= k for r in ranks))
        alternatives.append({'support_ids': support, 'ranks': ranks, 'metrics': metrics})
    if not alternatives:
        return None
    best = {key: max(a['metrics'][key] for a in alternatives)
            for key in alternatives[0]['metrics'] if key != 'completion_rank'}
    complete = [a['metrics']['completion_rank'] for a in alternatives if a['metrics']['completion_rank'] is not None]
    best['completion_rank'] = min(complete) if complete else None
    return {'alternatives': alternatives, 'best_sufficient_set_per_metric': best}


def encode_ape(model, texts, reference, output, should_stop):
    """One upstream batched forward per unique text; retain original slot IDs.

    Public encode omits IDs, while a second activity_scores call repeats inference.
    The pinned upstream batch iterator already returns all slots and activities.
    """
    from ape_retrieval.supportqa import text_hash
    directory = output / 'encodings'
    directory.mkdir(exist_ok=True)
    cache, pending = {}, []
    for text in dict.fromkeys(texts):
        digest = text_hash(text)
        path = directory / (digest + '.json')
        if path.exists():
            record = json.loads(path.read_text())
            if record['reference'] != reference or record['text_sha256'] != digest:
                raise ValueError('APE checkpoint identity changed')
            vectors = np.asarray(record['vectors'], dtype=np.float32)
            if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5):
                raise ValueError('Invalid cached APE vectors')
            cache[text] = record
        else:
            if len(model.tokenizer(text)['input_ids']) > reference['max_seq_len']:
                raise ValueError('APE input overflow; no hidden truncation')
            pending.append(text)
    if pending:
        with torch.inference_mode():
            for indices, vectors, activities in model._batches(pending, reference['batch_size'], reference['max_seq_len'], False):
                if should_stop():
                    raise RuntimeError('SupportQA budget stop during APE encoding')
                vectors = torch.nn.functional.normalize(vectors, dim=-1).cpu().numpy()
                activities = activities.cpu().numpy()
                for index, values, probs in zip(indices, vectors, activities):
                    text = pending[index]
                    active = np.flatnonzero(probs > reference['activity_threshold']).tolist()
                    fallback = not active
                    if fallback:
                        active = [int(probs.argmax())]
                    record = {'text_sha256': text_hash(text), 'reference': reference, 'vectors': values[active].tolist(),
                              'slot_ids': active, 'activities': probs.tolist(), 'selected_activities': probs[active].tolist(),
                              'strongest_slot_fallback': fallback, 'n_active_slots': len(active)}
                    atomic_json(directory / (text_hash(text) + '.json'), record)
                    cache[text] = record
    return cache


def diagnose(config, output, should_stop):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    from ape.evaluation.encoders import PooledEncoder
    from ape_retrieval.supportqa import text_hash
    reference = json.loads(Path(config['reference']).read_text())
    device = configure_reference(reference, config['diagnostics']['ape_device'])
    destination = output / 'diagnostics'
    destination.mkdir(exist_ok=True)
    documents = list(json_lines(dataset_path(output) / 'documents.jsonl'))
    queries = list(json_lines(dataset_path(output) / 'queries.jsonl'))
    texts = list(dict.fromkeys([d['document'] for d in documents] + [q['query'] for q in queries]))
    pending = any(not (output / 'encodings' / (text_hash(t) + '.json')).exists() for t in texts)
    model = load_model(reference, device).requires_grad_(False) if pending else None
    encoded = encode_ape(model, texts, reference, output, should_stop)
    del model
    torch.cuda.empty_cache()
    decompositions = {d['document_id']: json.loads((output / 'decompositions' / (d['context_sha256'] + '.json')).read_text()) for d in documents}
    contexts = {}
    for doc in documents:
        path = destination / (doc['context_sha256'] + '.json')
        if path.exists():
            record = json.loads(path.read_text())
            if record['decomposition_output_sha256'] != decompositions[doc['document_id']]['output_sha256']:
                raise ValueError('Reference decomposition changed')
            contexts[doc['document_id']] = record
    needed = [d for d in documents if d['document_id'] not in contexts]
    if needed:
        teacher_spec = config['teacher']
        teacher_dir = Path(snapshot_download(teacher_spec['model_id'], revision=teacher_spec['revision'],
            cache_dir=Path('.cache/huggingface').resolve(), local_files_only=True,
            allow_patterns=['model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
                            'special_tokens_map.json', 'vocab.txt']))
        if sha256(teacher_dir / 'model.safetensors') != teacher_spec['model_sha256']:
            raise ValueError('Frozen original teacher weight mismatch')
        teacher = AutoModel.from_pretrained(teacher_dir, local_files_only=True).to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(teacher_dir, local_files_only=True)
        pooled = PooledEncoder('supportqa_frozen_teacher_mean', teacher, tokenizer, 'mean', reference['batch_size'], reference['max_seq_len'])
        valid, overflow = {}, {}
        for doc in needed:
            dec = decompositions[doc['document_id']]
            props = sorted(dec['reference_propositions'], key=lambda p: p['id'])
            overflow[doc['document_id']] = [p['id'] for p in props if len(tokenizer(p['text'])['input_ids']) > reference['max_seq_len']]
            if props and not dec['potentially_incomplete'] and not overflow[doc['document_id']]:
                valid[doc['document_id']] = props
        # Upstream deduplicates all reference strings once across the complete reference.
        if should_stop(True):
            raise RuntimeError('SupportQA budget stop before teacher encoding')
        targets = pooled.encode([p['text'] for props in valid.values() for p in props])
        for doc in needed:
            if should_stop():
                raise RuntimeError('SupportQA budget stop during reference matching')
            ape = encoded[doc['document']]
            decomp = decompositions[doc['document_id']]
            record = {'document_id': doc['document_id'], 'dataset': doc['dataset'],
                      'encoding_file': 'encodings/' + text_hash(doc['document']) + '.json',
                      'ape': {k: v for k, v in ape.items() if k != 'vectors'},
                      'decomposition_output_sha256': decomp['output_sha256']}
            props = valid.get(doc['document_id'])
            if props is None:
                record['matching'] = {'status': 'ineligible', 'reason': 'incomplete_or_malformed_output_or_reference_overflow', 'overflow_reference_ids': overflow[doc['document_id']]}
            else:
                target_vectors = torch.cat([targets[p['text']] for p in props])
                slots = torch.tensor(ape['vectors'], dtype=torch.float32)
                cosine = (target_vectors @ slots.T).numpy()
                pairs = assignment(cosine)
                record['matching'] = {'status': 'provisional_teacher_space', 'query_independent': True,
                    'reference_ids': [p['id'] for p in props], 'cosine': cosine.tolist(),
                    'pairs': [{'reference_id': props[i]['id'], 'slot_id': ape['slot_ids'][j], 'cosine': float(cosine[i,j])} for i,j in pairs],
                    'unmatched_reference_ids': [p['id'] for i,p in enumerate(props) if i not in {r for r,_ in pairs}],
                    'unmatched_slot_ids': [slot for j,slot in enumerate(ape['slot_ids']) if j not in {c for _,c in pairs}]}
            record['generated_proposition_count'] = decomp['reference_n_atoms'] if decomp['format_valid'] else None
            record['human_counting_eligible'] = False
            contexts[doc['document_id']] = record
            atomic_json(destination / (doc['context_sha256'] + '.json'), record)
    atomic_json(output / 'diagnostic-provenance.json', {'config': config, 'reference': reference,
        'source_sha256': sha256(Path(__file__)), 'uv_lock_sha256': sha256(Path('uv.lock')),
        'selection_sha256': sha256(output / 'selection.json'), 'human_gold_metrics': False,
        'unique_texts_encoded': len(encoded), 'upstream_inference': 'pinned APE._batches, one forward per unique text; configured batch size',
        'teacher_reference_deduplication': 'global exact-string deduplication by upstream PooledEncoder'})
    should_stop(True)
    print(json.dumps({'stage':'diagnostics_complete','contexts':len(contexts),'queries':len(queries),'unique_texts':len(encoded)}),flush=True)
