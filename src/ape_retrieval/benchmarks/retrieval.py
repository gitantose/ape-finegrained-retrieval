"""Exact bounded multi-vector scoring, deterministic ranks and durable results."""

import argparse
import json
import gzip
import os
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from ape_retrieval.benchmarks.data import json_lines, prepare, write_line
from ape_retrieval.benchmarks.encoding import identity, verify_shards
from ape_retrieval.benchmarks.runtime import run_context
from ape_retrieval.encode import atomic_json
from ape_retrieval.benchmarks.metrics import aggregate_splits
from ape_retrieval.model import configure_reference
from ape_retrieval.propsearch import sha256


from ape_retrieval.benchmarks.scoring import RULES, ExactScorer


def relevant_ranks(scores, positives, tie_priority):
    """A total ordering: descending raw FP32 score, then ascending stable ID."""
    return [int(1 + np.count_nonzero(scores > scores[p])
                + np.count_nonzero((scores == scores[p]) & (tie_priority < tie_priority[p])))
            for p in positives]


def top_positions(scores, tie_priority, k):
    if not len(scores):
        return np.empty(0, dtype=np.int64)
    k = min(k, len(scores))
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    candidates = np.flatnonzero(scores >= threshold)
    order = np.lexsort((tie_priority[candidates], -scores[candidates]))
    return candidates[order[:k]]


def load_vectors(directory, progress, split):
    ids, offsets = [], [0]
    for chunk in progress["shards"][split]:
        with np.load(directory / chunk["file"], allow_pickle=False) as arrays:
            ids.extend(arrays["ids"].tolist())
            offsets.extend((arrays["offsets"][1:] + chunk["vector_start"]).tolist())
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate embedding IDs")
    vectors = np.memmap(directory / f"{split}.f32", mode="r", dtype=np.float32, shape=(offsets[-1], 384))
    return ids, np.asarray(offsets, dtype=np.int64), vectors


def retrieval_identity(encoding, config):
    return {"encoding_manifest": sha256(encoding / "manifest.json"),
              "encoding_progress": sha256(encoding / "progress.json"),
              "source_sha256": sha256(Path(__file__)),
              "scoring_source_sha256": sha256(Path(__file__).with_name("scoring.py")),
              "metrics_source_sha256": sha256(Path(__file__).with_name("metrics.py")),
              "config": config, "tie_rule": "descending raw score, ascending document ID"}


def retrieve(config, processed, directory, reference, should_stop):
    encoding = directory / "encoding"
    progress = json.loads((encoding / "progress.json").read_text())
    if progress["status"] != "complete":
        raise ValueError("Embedding cache is incomplete")
    verify_shards(encoding, progress)
    doc_ids, doc_offsets, corpus = load_vectors(encoding, progress, "corpus")
    query_ids, query_offsets, queries = load_vectors(encoding, progress, "queries")
    doc_positions = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    priority = np.empty(len(doc_ids), dtype=np.int64)
    priority[np.argsort(doc_ids)] = np.arange(len(doc_ids))
    output = directory / "retrieval"
    output.mkdir(exist_ok=True)
    source = retrieval_identity(encoding, config)
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != source:
        raise ValueError("Retrieval identity changed")
    if not manifest.exists():
        atomic_json(manifest, source)
    progress_path = output / "progress.json"
    state = json.loads(progress_path.read_text()) if progress_path.exists() else {"status": "prepared", "processed_queries": 0, "blocks": []}
    cursor = 0
    for block in state["blocks"]:
        if block["start"] != cursor or sha256(output / block["file"]) != block["sha256"]:
            raise ValueError("Invalid committed ranking block")
        cursor = block["stop"]
    if cursor != state["processed_queries"]:
        raise ValueError("Ranking coverage mismatch")
    if state["status"] == "complete":
        if cursor != len(query_ids) or sha256(output / "metrics.json") != state["metrics_sha256"]:
            raise ValueError("Invalid complete retrieval artifacts")
        return state
    device = configure_reference(reference, config["device"])
    pending, start = [], cursor
    last_log, started = time.monotonic(), time.monotonic()

    def commit():
        nonlocal pending, start, last_log
        if not pending:
            return
        path = output / f"queries-{start:09d}.jsonl.gz"
        temporary = path.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as stream:
            for row in pending:
                write_line(stream, row)
            stream.flush()
        with temporary.open("rb") as saved:
            os.fsync(saved.fileno())
        temporary.replace(path)
        stop = start + len(pending)
        state["blocks"].append({"start": start, "stop": stop, "file": path.name, "sha256": sha256(path)})
        state.update(status="running", processed_queries=stop)
        atomic_json(progress_path, state)
        if time.monotonic() - last_log >= 30 or stop == len(query_ids):
            elapsed = time.monotonic() - started
            print(json.dumps({"ranked_queries": stop, "total_queries": len(query_ids),
                              "percent": round(100 * stop / len(query_ids), 1),
                              "queries_per_second": round((stop - cursor) / max(elapsed, .001), 2)}), flush=True)
            last_log = time.monotonic()
        start, pending = stop, []

    scorer = ExactScorer(doc_offsets, corpus, device, config["scoring_block_slots"],
                         config.get("limits", {}).get("cuda_allocated_bytes", 2 * 2**30))
    print(json.dumps({"scorer": scorer.mode, "query_batch_size": config.get("query_batch_size", 32)}), flush=True)
    rows = islice(json_lines(processed / "queries.jsonl"), cursor, None)
    index = cursor
    try:
        while batch := list(islice(rows, config.get("query_batch_size", 32))):
            if should_stop():
                break
            eligible = [i for i, row in enumerate(batch) if row["positive_ids"]]
            for i, row in enumerate(batch):
                if row["query_id"] != query_ids[index + i]:
                    raise ValueError("Query/embedding identity mismatch")
                if any(p not in doc_positions for p in row["positive_ids"]):
                    raise ValueError("Positive document absent from corpus")
            query_vectors = [queries[query_offsets[index + i]:query_offsets[index + i + 1]] for i in eligible]
            batch_scores = scorer.score_batch(query_vectors, should_stop) if eligible else {}
            if batch_scores is None:
                break
            positions = {row_index: j for j, row_index in enumerate(eligible)}
            for i, row in enumerate(batch):
                result = {"query_id": row["query_id"], "split": row["split"],
                          "positive_ids": row["positive_ids"], "qrel_provenance": row["qrel_provenance"],
                          "ineligible_reason": row["ineligible_reason"], "aggregations": {}}
                if i in positions:
                    positives = [doc_positions[p] for p in row["positive_ids"]]
                    for rule in RULES:
                        values = batch_scores[rule][positions[i]]
                        top = top_positions(values, priority, max(config["ks"]))
                        result["aggregations"][rule] = {
                            "positive_ranks": relevant_ranks(values, positives, priority),
                            "positive_scores": values[positives].tolist(),
                            "top_ids": [doc_ids[p] for p in top], "top_scores": values[top].tolist()}
                else:
                    result["aggregations"] = {rule: {"positive_ranks": [], "top_ids": [], "top_scores": []} for rule in RULES}
                pending.append(result)
                if len(pending) >= config["ranking_checkpoint_queries"]:
                    commit()
            index += len(batch)
    finally:
        commit()
    if state["processed_queries"] == len(query_ids):
        records = (row for block in state["blocks"] for row in json_lines(output / block["file"]))
        metrics = aggregate_splits(records, RULES, config["ks"])
        atomic_json(output / "metrics.json", {"by_split": metrics, "retrieval": "exact full corpus",
                                              "MRR": "full first relevant rank", "metric_source_sha256": sha256(Path(__file__).with_name("metrics.py"))})
        state.update(status="complete", metrics_sha256=sha256(output / "metrics.json"))
    else:
        state["status"] = "stopped"
    atomic_json(progress_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("nq", "squad", "propsearch"))
    parser.add_argument("--config", type=Path, default=Path("configs/benchmarks.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    reference = json.loads(Path(config["reference"]).read_text())
    processed = prepare(config, args.benchmark)
    _, key = identity(config, processed, reference)
    directory = Path(config["runs_dir"]) / args.benchmark / key
    with run_context(directory, config, None) as should_stop:
        state = retrieve(config, processed, directory, reference, should_stop)
        if state["status"] != "complete":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
