"""Exact fixed-vector baselines for the frozen document benchmarks."""

import gzip
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import time

from ape.training.teacher import pool
from huggingface_hub import snapshot_download
import numpy as np
import torch
from transformers import AutoTokenizer, BertModel

from ape_retrieval.benchmarks.data import json_lines, processed_location, write_line
from ape_retrieval.benchmarks.encoding import segments
from ape_retrieval.benchmarks.metrics import aggregate_splits
from ape_retrieval.benchmarks.retrieval import relevant_ranks, top_positions
from ape_retrieval.encode import atomic_json, sync_directory
from ape_retrieval.model import configure_reference
from ape_retrieval.propsearch import sha256


DATASETS = ('nq', 'squad', 'propsearch')
DIMENSION = 768
ROW_BYTES = DIMENSION * np.dtype(np.float32).itemsize
MODEL_FILES = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json',
               'special_tokens_map.json', 'vocab.txt', 'README.md']


def acquire(config, output, should_stop):
    for name, spec in config['baselines'].items():
        if should_stop():
            raise InterruptedError('Stopped before checkpoint acquisition')
        directory = output / 'models' / name
        if (directory / 'files.json').exists():
            checkpoint(output, name, spec)
            print(json.dumps({'stage': 'checkpoint', 'model': name, 'status': 'verified_local'}), flush=True)
            continue
        snapshot_download(spec['model_id'], revision=spec['revision'], local_dir=directory,
                          allow_patterns=MODEL_FILES, max_workers=1)
        receipt = directory / 'files.json'
        files = {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file() and p != receipt}
        if files.get('pytorch_model.bin') != spec['weights_sha256']:
            raise ValueError(f'Official checkpoint hash mismatch for {name}')
        atomic_json(receipt, files)
        print(json.dumps({'stage': 'checkpoint', 'model': name, 'status': 'verified_download'}), flush=True)
        should_stop(True)


def checkpoint(output, name, spec):
    directory = output / 'models' / name
    files = json.loads((directory / 'files.json').read_text())
    if files.get('pytorch_model.bin') != spec['weights_sha256']:
        raise ValueError(f'Checkpoint receipt differs from the pinned revision for {name}')
    for filename, digest in files.items():
        if sha256(directory / filename) != digest:
            raise ValueError(f'Model or tokenizer file changed for {name}: {filename}')
    return directory


def load_encoder(directory, device):
    model, info = BertModel.from_pretrained(directory, add_pooling_layer=False, dtype=torch.float32,
                                          local_files_only=True, output_loading_info=True)
    if (info.get('missing_keys') or info.get('mismatched_keys')
            or any(not k.startswith('pooler.') for k in info.get('unexpected_keys', ()))):
        raise ValueError(f'Incomplete or incompatible backbone weights: {info}')
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    return model.to(device).eval().requires_grad_(False), tokenizer


@torch.inference_mode()
def encode_texts(model, tokenizer, texts, spec, batch_size, device):
    """Pool with upstream APE's generic BERT pooling, then aggregate segments."""
    offsets = [segments(text, tokenizer, spec['max_tokens']) for text in texts]
    encoded_texts = [text[start:stop] for text, spans in zip(texts, offsets) for start, stop in spans]
    pieces = []
    cosine = spec['score'] == 'cosine'
    pooling = 'cls' if spec['pooling'] == 'cls_before_pooler' else 'mean'
    # Sorting within a shard reduces padding and preserves the original mapping.
    indices = sorted(range(len(encoded_texts)), key=lambda i: len(encoded_texts[i]))
    for cursor in range(0, len(indices), batch_size):
        batch = indices[cursor:cursor + batch_size]
        inputs = tokenizer([encoded_texts[i] for i in batch], padding=True,
                           truncation=False, return_tensors='pt').to(device)
        if inputs.input_ids.shape[1] > spec['max_tokens']:
            raise ValueError('A baseline segment exceeds the frozen token limit')
        encoded = pool(model(**inputs).last_hidden_state, inputs.attention_mask, mode=pooling)
        pieces.append(torch.nn.functional.normalize(encoded, dim=-1) if cosine else encoded)
    ordered = torch.cat(pieces)[np.argsort(indices)]
    if len(encoded_texts) != len(texts):
        combined = []
        start = 0
        for spans in offsets:
            stop = start + len(spans)
            value = ordered[start:stop]
            combined.append(value[0] if len(spans) == 1 else value.mean(dim=0))
            start = stop
        ordered = torch.stack(combined)
        if cosine:
            multi = torch.tensor([len(parts) > 1 for parts in offsets], device=device)
            ordered[multi] = torch.nn.functional.normalize(ordered[multi], dim=-1)
    return ordered.cpu().numpy(), offsets


def source_paths(config):
    """Open the established benchmark caches read-only and verify their inputs."""
    suite = json.loads(Path(config['retrieval']['config']).read_text())
    paths, sources = {}, {}
    for dataset in DATASETS:
        path, identity = processed_location(suite, dataset)
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest['source'] != identity:
            raise ValueError(f'Processed dataset identity changed for {dataset}')
        if any(sha256(path / name) != digest for name, digest in manifest['files'].items()):
            raise ValueError(f'Processed dataset files changed for {dataset}')
        paths[dataset] = path
        sources[dataset] = {'path': str(path), 'manifest_sha256': sha256(path / 'manifest.json')}
    return paths, sources


def benchmark_text_index(paths):
    positions, ids = {}, {}
    for dataset, path in paths.items():
        ids[dataset] = {}
        for part, id_key, text_key in [('corpus', 'document_id', 'text'), ('queries', 'query_id', 'query')]:
            ids[dataset][part] = {row[id_key]: positions.setdefault(row[text_key], len(positions))
                                  for row in json_lines(path / f'{part}.jsonl')}
    return positions, ids


def text_index_hash(texts):
    digest = hashlib.sha256()
    for text in texts:
        digest.update(hashlib.sha256(text.encode('utf-8')).digest())
    return digest.hexdigest()


def commit_manifest(path, identity):
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(f'Resume identity changed: {path}')
    else:
        atomic_json(path, identity)


def validate_vector_blocks(directory, state, total):
    cursor = 0
    path = directory / 'vectors.f32'
    if state['blocks']:
        with path.open('rb') as stream:
            for block in state['blocks']:
                rows = block['stop'] - cursor
                if block['start'] != cursor or rows <= 0 or block['stop'] > total:
                    raise ValueError('Invalid vector checkpoint interval')
                data = stream.read(rows * ROW_BYTES)
                if len(data) != rows * ROW_BYTES or hashlib.sha256(data).hexdigest() != block['sha256']:
                    raise ValueError('Corrupt committed FP32 vectors')
                if block.get('segment_file') and sha256(directory / block['segment_file']) != block['segment_sha256']:
                    raise ValueError('Corrupt committed segment offsets')
                cursor = block['stop']
    if cursor != state['completed'] or (state['status'] == 'COMPLETED' and cursor != total):
        raise ValueError('Vector checkpoint row count changed')
    if state['status'] == 'COMPLETED' and path.stat().st_size != total * ROW_BYTES:
        raise ValueError('Completed vector storage size changed')
    return cursor


def encode_index(config, directory, texts, identity, name, model_dir, should_stop):
    spec = config['baselines'][name]
    settings = config['retrieval']
    directory.mkdir(parents=True, exist_ok=True)
    commit_manifest(directory / 'manifest.json', {'model': spec, 'rows': len(texts), 'identity': identity,
        'text_index_sha256': text_index_hash(texts), 'encoding_batch_size': settings['encoding_batch_size'],
        'encoding_shard_rows': settings['encoding_shard_rows'], 'source_sha256': sha256(Path(__file__)),
        'source_offset_segmenter_sha256': sha256(Path(__file__).with_name('encoding.py')),
        'checkpoint_files_sha256': sha256(model_dir / 'files.json'), 'uv_lock_sha256': sha256(Path('uv.lock')),
        'precision': 'FP32; TF32 off', 'segmentation': 'source-offset windows; equal segment weights'})
    state_path = directory / 'progress.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'status': 'RUNNING', 'completed': 0, 'blocks': []}
    resumed = validate_vector_blocks(directory, state, len(texts))
    if state['status'] == 'COMPLETED':
        return
    device = configure_reference(json.loads(Path(config['reference']).read_text()), 'cuda')
    if not texts:
        (directory / 'vectors.f32').touch()
        state['status'] = 'COMPLETED'
        atomic_json(state_path, state)
        return
    model, tokenizer = load_encoder(model_dir, device)
    started = time.monotonic()
    for start in range(resumed, len(texts), settings['encoding_shard_rows']):
        if should_stop():
            raise InterruptedError('Stopped before the next baseline encoding shard')
        batch = texts[start:start + settings['encoding_shard_rows']]
        values, offsets = encode_texts(model, tokenizer, batch, spec, settings['encoding_batch_size'], device)
        if values.shape != (len(batch), DIMENSION) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('Invalid baseline vector shard')
        if spec['score'] == 'cosine' and not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-5):
            raise ValueError('Cosine baseline emitted non-unit vectors')
        data, stop = values.tobytes(), start + len(values)
        path = directory / 'vectors.f32'
        with path.open('r+b' if path.exists() else 'w+b') as stream:
            # Only an uncommitted tail can be present after interruption.
            stream.truncate(start * ROW_BYTES)
            stream.seek(0, os.SEEK_END)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        block = {'start': start, 'stop': stop, 'sha256': hashlib.sha256(data).hexdigest()}
        segmented = {str(i): spans for i, spans in enumerate(offsets, start) if len(spans) > 1}
        if segmented:
            path = directory / f'segments-{start:09d}.json'
            atomic_json(path, segmented)
            block.update(segment_file=path.name, segment_sha256=sha256(path))
        state['blocks'].append(block)
        state.update(completed=stop, status='COMPLETED' if stop == len(texts) else 'RUNNING')
        atomic_json(state_path, state)
        report_progress(name, 'dense_encoding', stop, len(texts), resumed, started)
        should_stop(True)


def report_progress(name, stage, completed, total, resumed, started):
    rate = (completed - resumed) / max(time.monotonic() - started, 1e-3)
    print(json.dumps({'model': name, 'stage': stage, 'completed': completed, 'total': total,
                      'per_second': round(rate, 2), 'eta_seconds': round((total - completed) / rate) if rate else None}), flush=True)


def open_vectors(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    progress = json.loads((directory / 'progress.json').read_text())
    total = manifest['rows']
    if progress['status'] != 'COMPLETED':
        raise ValueError('Baseline embeddings are incomplete')
    validate_vector_blocks(directory, progress, total)
    return np.memmap(directory / 'vectors.f32', mode='r', dtype=np.float32, shape=(total, DIMENSION)) if total else np.empty((0, DIMENSION), dtype=np.float32)


def write_rank_block(path, rows):
    temporary = path.with_suffix('.tmp')
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as stream:
        for row in rows:
            write_line(stream, row)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def rank_dataset(config, directory, source, dataset, name, vectors, ids, embedding_identity, device, should_stop):
    settings = config['retrieval']
    rule = config['baselines'][name]['score']
    folder = directory / dataset
    folder.mkdir(parents=True, exist_ok=True)
    commit_manifest(folder / 'manifest.json', {'encoding': embedding_identity,
        'processed_manifest_sha256': sha256(source / 'manifest.json'), 'rule': rule, 'ks': settings['ks'],
        'query_batch_size': settings['scoring_query_batch_size'],
        'dense_ranking_source_sha256': sha256(Path(__file__)),
        'ranking_source_sha256': sha256(Path(__file__).with_name('retrieval.py')),
        'metrics_source_sha256': sha256(Path(__file__).with_name('metrics.py')),
        'tie_rule': 'descending FP32 score then ascending canonical document ID',
        'top_corpus_indices': 'zero-based positions in the immutable processed corpus.jsonl'})
    state_path = folder / 'progress.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'completed': 0, 'blocks': [], 'status': 'RUNNING'}
    start = 0
    for block in state['blocks']:
        if (block['start'] != start or block['stop'] <= start or block['stop'] > len(ids['queries'])
                or sha256(folder / block['file']) != block['sha256']):
            raise ValueError('Invalid committed ranking block')
        start = block['stop']
    if state['completed'] != start:
        raise ValueError('Ranking cursor mismatch')
    if state['status'] == 'COMPLETED':
        if start != len(ids['queries']) or sha256(folder / 'metrics.json') != state['metrics_sha256']:
            raise ValueError('Completed ranking no longer matches its receipt')
        return
    doc_ids = list(ids['corpus'])
    doc_positions = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    tie_priority = np.empty(len(doc_ids), dtype=np.int64)
    tie_priority[np.argsort(doc_ids)] = np.arange(len(doc_ids))
    corpus = torch.from_numpy(vectors[list(ids['corpus'].values())]).to(device)
    queries = islice(json_lines(source / 'queries.jsonl'), start, None)
    resumed, started = start, time.monotonic()
    batch_size = settings['scoring_query_batch_size']
    while block_rows := list(islice(queries, settings['ranking_checkpoint_queries'])):
        if should_stop():
            raise InterruptedError('Stopped before the next ranking block')
        predictions = []
        for offset in range(0, len(block_rows), batch_size):
            rows = block_rows[offset:offset + batch_size]
            eligible = [row for row in rows if row['positive_ids']]
            query_values = vectors[[ids['queries'][row['query_id']] for row in eligible]]
            scores = iter((torch.from_numpy(query_values).to(device) @ corpus.T).cpu().numpy())
            for row in rows:
                result = {'positive_ranks': [], 'top_corpus_indices': []}
                if row['positive_ids']:
                    values = next(scores)
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite baseline retrieval scores')
                    positives = [doc_positions[p] for p in row['positive_ids']]
                    result = {'positive_ranks': relevant_ranks(values, positives, tie_priority),
                              'top_corpus_indices': top_positions(values, tie_priority, max(settings['ks'])).tolist()}
                predictions.append({'query_id': row['query_id'], 'split': row['split'],
                                    'aggregations': {rule: result}})
        stop = start + len(predictions)
        path = folder / f'queries-{start:09d}.jsonl.gz'
        write_rank_block(path, predictions)
        state['blocks'].append({'start': start, 'stop': stop, 'file': path.name, 'sha256': sha256(path)})
        state['completed'] = start = stop
        atomic_json(state_path, state)
        report_progress(name, f'{dataset}_ranking', stop, len(ids['queries']), resumed, started)
        should_stop(True)
    if start != len(ids['queries']):
        raise ValueError('Ranked coverage differs from the processed query population')
    predictions = (row for block in state['blocks'] for row in json_lines(folder / block['file']))
    metrics_path = folder / 'metrics.json'
    atomic_json(metrics_path, {'by_split': aggregate_splits(predictions, (rule,), settings['ks'])})
    state.update(status='COMPLETED', metrics_sha256=sha256(metrics_path))
    atomic_json(state_path, state)


def run_dense(config, output, action, name, should_stop):
    paths, sources = source_paths(config)
    text_positions, ids = benchmark_text_index(paths)
    model_dir = checkpoint(output, name, config['baselines'][name])
    output = output / 'retrieval' / name
    encoding = output / 'encoding'
    if action == 'encode':
        encode_index(config, encoding, list(text_positions), sources, name, model_dir, should_stop)
        return
    manifest = json.loads((encoding / 'manifest.json').read_text())
    if (manifest['text_index_sha256'] != text_index_hash(text_positions)
            or manifest['identity'] != sources or manifest['model'] != config['baselines'][name]
            or manifest['checkpoint_files_sha256'] != sha256(model_dir / 'files.json')):
        raise ValueError('Benchmark text/vector identity changed')
    vectors = open_vectors(encoding)
    device = configure_reference(json.loads(Path(config['reference']).read_text()), 'cuda')
    identity = {f: sha256(encoding / f) for f in ('manifest.json', 'progress.json')}
    for dataset in DATASETS:
        rank_dataset(config, output, paths[dataset], dataset, name, vectors, ids[dataset], identity, device, should_stop)
