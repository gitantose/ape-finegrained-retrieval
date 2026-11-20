"""Bounded, recoverable APE inference with source-document segment offsets."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from transformers import AutoTokenizer

from ape_retrieval.benchmarks.data import json_lines, prepare
from ape_retrieval.encode import atomic_json
from ape_retrieval.model import configure_reference, load_model
from ape_retrieval.propsearch import sha256


def segments(text, tokenizer, limit):
    """Preserve short documents; split long ones at punctuation/newlines.

    Overlong sentences are subdivided on tokenizer offsets, never discarded.
    Offsets refer to the original text. Punctuation segmentation is a declared
    deterministic heuristic, not a linguistically validated sentence parser.
    """
    if len(tokenizer(text)["input_ids"]) <= limit:
        return [(0, len(text))]
    boundaries = [0, *[m.end() for m in re.finditer(r"(?<=[.!?])\s+|\n+", text)], len(text)]
    result = []
    for start, stop in zip(boundaries, boundaries[1:]):
        if not text[start:stop].strip():
            continue
        encoded = tokenizer(text[start:stop], add_special_tokens=False, return_offsets_mapping=True)
        ends = np.asarray([end + start for _, end in encoded["offset_mapping"]])
        cursor = start
        while cursor < stop:
            first = int(np.searchsorted(ends, cursor, side="right"))
            last = min(first + limit - tokenizer.num_special_tokens_to_add(), len(ends)) - 1
            end = stop if last == len(ends) - 1 else int(ends[last])
            while len(tokenizer(text[cursor:end])["input_ids"]) > limit:
                last -= 1
                end = int(ends[last])
            if end <= cursor:
                raise ValueError("Tokenizer subdivision made no progress")
            result.append((cursor, end))
            cursor = end
    if not result:
        raise ValueError("Document has no encodable nonempty segment")
    return result


def tokenizer_from_reference(reference):
    # Loading tokenizer metadata alone is sufficient for a complete shape audit.
    from huggingface_hub import snapshot_download
    path = snapshot_download(reference["backbone_id"], revision=reference["backbone_revision"],
                             cache_dir=Path(".cache/huggingface").resolve(), local_files_only=True,
                             allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                                             "special_tokens_map.json", "vocab.txt", "README.md"])
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def identity(config, processed, reference):
    value = {"format_version": 1, "processed_manifest_sha256": sha256(processed / "manifest.json"),
             "reference": reference, "segmentation": config["segmentation"],
             "source_sha256": {name: sha256(Path(__file__) if name == "encoding.py" else Path(__file__).parent.parent / name)
                               for name in ("encoding.py", "model.py", "encode.py")},
             "lock_sha256": sha256(Path("uv.lock"))}
    compatibility = Path(config["runs_dir"]) / "layout-compatibility.json" if "runs_dir" in config else None
    if compatibility and compatibility.exists():
        receipt = json.loads(compatibility.read_text())
        record = receipt['completed_encodings'].get(processed.parent.name)
        if record and receipt['configuration'] == config and receipt['current_source_sha256'] == value['source_sha256']:
            previous = record['identity']
            if all(previous.get(k) == v for k, v in value.items() if k != 'source_sha256'):
                key = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
                directory = Path(config['runs_dir']) / processed.parent.name / key / 'encoding'
                if (sha256(directory / 'manifest.json') != record['manifest_sha256']
                        or sha256(directory / 'progress.json') != record['progress_sha256']):
                    raise ValueError('Preserved encoding receipt changed')
                return previous, key
    key = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value, key


def plan(config, processed, reference):
    tokenizer = tokenizer_from_reference(reference)
    stats = {"documents": 0, "segments": 0, "max_segments_per_document": 0,
             "queries": 0, "eligible_queries": 0, "overlong_queries": 0}
    for row in json_lines(processed / "corpus.jsonl"):
        parts = segments(row["text"], tokenizer, reference["max_seq_len"])
        stats["documents"] += 1
        stats["segments"] += len(parts)
        stats["max_segments_per_document"] = max(stats["max_segments_per_document"], len(parts))
    for row in json_lines(processed / "queries.jsonl"):
        stats["queries"] += 1
        stats["eligible_queries"] += bool(row["positive_ids"])
        stats["overlong_queries"] += len(tokenizer(row["query"])["input_ids"]) > reference["max_seq_len"]
    stats["query_document_pairs"] = stats["eligible_queries"] * stats["documents"]
    stats["embedding_bytes_lower_bound"] = (stats["segments"] + stats["queries"]) * 384 * 4
    stats["embedding_bytes_upper_bound"] = stats["embedding_bytes_lower_bound"] * 8
    stats["note"] = "FP32 values only; excludes offsets, IDs, activity, manifests and temporary shards; no runtime forecast"
    return stats


def verify_shards(directory, progress):
    for split, chunks in progress["shards"].items():
        position = 0
        for chunk in chunks:
            if chunk["start"] != position or chunk["stop"] <= position:
                raise ValueError("Noncontiguous embedding shards")
            path = directory / chunk["file"]
            if sha256(path) != chunk["sha256"]:
                raise ValueError("Embedding shard corruption")
            with np.load(path, allow_pickle=False) as arrays:
                offsets = arrays["offsets"]
                vectors = np.memmap(directory / f"{split}.f32", mode="r", dtype=np.float32,
                                    offset=chunk["vector_start"] * 384 * 4,
                                    shape=(chunk["vectors"], 384))
                if (hashlib.sha256(vectors.tobytes()).hexdigest() != chunk["vectors_sha256"]
                        or not np.isfinite(vectors).all() or offsets[0] != 0
                        or offsets[-1] != len(vectors) or (np.diff(offsets) < 1).any()
                        or len(offsets) != chunk["stop"] - position + 1
                        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5)):
                    raise ValueError("Invalid packed document/query embeddings")
            position = chunk["stop"]
        if position != progress["processed"][split]:
            raise ValueError("Embedding progress differs from committed shards")


def encode(config, processed, reference, directory, should_stop):
    value, _ = identity(config, processed, reference)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != value:
            raise ValueError("Embedding cache identity changed")
    else:
        atomic_json(manifest, value)
    progress_path = directory / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "status": "prepared", "processed": {"corpus": 0, "queries": 0},
        "shards": {"corpus": [], "queries": []}, "query_truncations": []}
    verify_shards(directory, progress)
    if progress["status"] == "complete":
        return progress
    device = configure_reference(reference, config["device"])
    model = load_model(reference, device).requires_grad_(False)
    for split in ("corpus", "queries"):
        pending, positions, provenance = [], [], []
        start = progress["processed"][split]

        def commit():
            nonlocal pending, positions, provenance, start
            if not pending:
                return
            path = directory / f"{split}-{start:09d}.npz"
            temporary = path.with_suffix(".tmp")
            values = np.concatenate(pending).astype(np.float32, copy=False)
            vector_start = sum(c["vectors"] for c in progress["shards"][split])
            vector_path = directory / f"{split}.f32"
            # Remove only an uncommitted tail in this new cache after a crash.
            import os
            with vector_path.open("r+b" if vector_path.exists() else "w+b") as stream:
                stream.truncate(vector_start * 384 * 4)
                stream.seek(0, 2)
                stream.write(values.tobytes())
                stream.flush()
                os.fsync(stream.fileno())
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, offsets=np.asarray([0, *np.cumsum([len(v) for v in pending])], dtype=np.int64),
                         ids=np.asarray(positions), segments_json=np.asarray([json.dumps(p) for p in provenance]))
                stream.flush()
                import os
                os.fsync(stream.fileno())
            temporary.replace(path)
            progress["processed"][split] += len(pending)
            progress["shards"][split].append({"start": start, "stop": progress["processed"][split],
                                              "file": path.name, "sha256": sha256(path),
                                              "vector_start": vector_start, "vectors": len(values),
                                              "vectors_sha256": hashlib.sha256(values.tobytes()).hexdigest()})
            progress["status"] = "running"
            atomic_json(progress_path, progress)
            print(json.dumps({"encoding": progress["processed"]}), flush=True)
            start = progress["processed"][split]
            pending, positions, provenance = [], [], []

        try:
            for index, row in enumerate(json_lines(processed / f"{split}.jsonl")):
                if index < start:
                    continue
                if should_stop():
                    break
                text = row["text"] if split == "corpus" else row["query"]
                spans = segments(text, model.tokenizer, reference["max_seq_len"]) if split == "corpus" else [(0, len(text))]
                if split == "queries":
                    length = len(model.tokenizer(text)["input_ids"])
                vectors, activities = [], []
                for batch_start in range(0, len(spans), reference["batch_size"]):
                    texts = [text[a:b] for a, b in spans[batch_start:batch_start + reference["batch_size"]]]
                    with torch.inference_mode():
                        batch_vectors, batch_activities = model.encode(texts, batch_size=reference["batch_size"],
                            max_seq_len=reference["max_seq_len"], activity_threshold=reference["activity_threshold"],
                            normalize_embeddings=True, convert_to_numpy=True, return_scores=True)
                    vectors.extend(batch_vectors)
                    activities.extend(batch_activities)
                pending.append(np.concatenate(vectors))
                positions.append(row["document_id"] if split == "corpus" else row["query_id"])
                provenance.append([{"start": a, "stop": b, "active_slots": len(v),
                                    "selected_activities": activity.tolist(),
                                    "strongest_slot_fallback": bool(not (activity > reference["activity_threshold"]).any())}
                                   for (a, b), v, activity in zip(spans, vectors, activities)])
                if split == "queries" and length > reference["max_seq_len"]:
                    progress["query_truncations"].append({"query_id": row["query_id"], "token_length": length})
                if len(pending) >= config["encoding_shard_documents"]:
                    commit()
        finally:
            commit()
        if should_stop():
            progress["status"] = "stopped"
            atomic_json(progress_path, progress)
            return progress
    verify_shards(directory, progress)
    progress["status"] = "complete"
    atomic_json(progress_path, progress)
    return progress


def main():
    from ape_retrieval.benchmarks.runtime import run_context
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "encode"))
    parser.add_argument("benchmark", choices=("nq", "squad", "propsearch"))
    parser.add_argument("--config", type=Path, default=Path("configs/benchmarks.json"))
    parser.add_argument("--budget-seconds", type=float)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    reference = json.loads(Path(config["reference"]).read_text())
    processed = prepare(config, args.benchmark)
    value, key = identity(config, processed, reference)
    directory = Path(config["runs_dir"]) / args.benchmark / key
    if args.action == "plan":
        result = plan(config, processed, reference)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "plan.json", result)
        print(json.dumps({"run_dir": str(directory), **result}), flush=True)
    else:
        with run_context(directory, config, args.budget_seconds) as should_stop:
            progress = encode(config, processed, reference, directory / "encoding", should_stop)
            if progress["status"] != "complete":
                raise SystemExit(2)


if __name__ == "__main__":
    main()
