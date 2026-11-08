"""Frozen pretrained BGE: shared forward pass, two upstream pooling conditions."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from ape.training.teacher import pool
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModel, AutoTokenizer

from ape_retrieval.encode import EncodingStore, atomic_json, disk_bytes, run_identity, write_summary
from ape_retrieval.model import configure_reference
from ape_retrieval.propsearch import load_inputs, sha256


MODES = ("mean", "cls")


def validate_config(config):
    if (config["pooling"] != list(MODES) or config["query_prefix"] != "query: "
            or config["batch_size"] != 2 or config["max_seq_len"] != 128
            or not config["normalize_embeddings"] or config["device"] != "cuda"
            or config["dtype"] != "float32" or config["tf32"] is not False
            or config["model_revision"] != "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
            or config["model_id"] != "BAAI/bge-small-en-v1.5"
            or config["rank_rule"] != "1 + count(score > gold_score)"):
        raise ValueError("Frozen BGE protocol changed")


def budget(root, config):
    path = root / "budget.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["config"] != config:
            raise ValueError("Shared budget/configuration changed")
        return saved
    now = time.time()
    saved = {"config": config, "started_epoch": now,
             "deadline_epoch": now + config["budget_seconds"]}
    atomic_json(path, saved)
    return saved


def memory_plan(inputs, config):
    n, q = (len(inputs[k]) for k in ("candidates", "queries"))
    corpus = n * 384 * 4
    # Upstream retains old/new score chunks briefly, plus comparison mask.
    gpu = corpus + config["checkpoint_queries"] * 384 * 4 + 2 * config["query_chunk_size"] * n * 5
    disk = 2 * (n + q) * (384 * 4 + 256) + 256 * 2**20
    plan = {"candidate_positions": n, "queries": q, "dimension": 384,
            "two_condition_raw_vector_bytes": 2 * (n + q) * 384 * 4,
            "scoring_tensor_estimate_bytes": gpu,
            "host_array_estimate_bytes": 2 * (n + q) * 384 * 4,
            "additional_disk_estimate_bytes": disk,
            "caveat": "Host estimate excludes Python/library/ID objects; actual RSS is guarded. CUDA estimate excludes driver/context."}
    if gpu > config["cuda_allocated_budget_bytes"] or disk > config["additional_disk_budget_bytes"]:
        raise ValueError("Whole-task shape exceeds budget")
    return plan


def acquire(root, config):
    budget(root, config)
    info = HfApi().model_info(config["model_id"], revision=config["model_revision"], files_metadata=True)
    entry = next(f for f in info.siblings if f.rfilename == "model.safetensors")
    expected = entry.lfs.sha256
    path = Path(snapshot_download(config["model_id"], revision=config["model_revision"],
        allow_patterns=["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "vocab.txt", "README.md"],
        cache_dir=Path(".cache/huggingface").resolve()))
    actual = sha256(path / "model.safetensors")
    if actual != expected:
        raise ValueError("Pretrained weight hash differs from pinned Hub LFS identity")
    receipt = {"model_id": config["model_id"], "revision": info.sha,
               "weight_sha256": actual, "weight_bytes": entry.size,
               "files": {p.name: sha256(p) for p in path.iterdir() if p.is_file()}}
    atomic_json(root / "acquisition.json", receipt)
    print(json.dumps(receipt, indent=2))


def load_pretrained(root, config, device):
    receipt = json.loads((root / "acquisition.json").read_text())
    path = Path(snapshot_download(config["model_id"], revision=config["model_revision"],
                                 allow_patterns=list(receipt["files"]),
                                 cache_dir=Path(".cache/huggingface").resolve(), local_files_only=True))
    if receipt["revision"] != config["model_revision"]:
        raise ValueError("Wrong pretrained revision")
    for name, digest in receipt["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Pretrained file changed: {name}")
    model = AutoModel.from_pretrained(path, local_files_only=True, dtype=torch.float32).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer


def encode_batch(model, tokenizer, texts, device, config):
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=config["max_seq_len"],
                       return_tensors="pt").to(device)
    with torch.inference_mode():
        hidden = model(**tokens).last_hidden_state
        return {mode: F.normalize(pool(hidden, tokens["attention_mask"], mode=mode), dim=-1).cpu().numpy()
                for mode in MODES}


def check_resources(root, config, device):
    values = {"host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
    for key, limit in [("host_peak_rss_bytes", "host_memory_budget_bytes"),
                       ("cuda_peak_allocated_bytes", "cuda_allocated_budget_bytes")]:
        if values[key] > config[limit]:
            raise RuntimeError(f"{key} exceeded approved budget; no fallback")
    weight_bytes = json.loads((root / "acquisition.json").read_text())["weight_bytes"] if (root / "acquisition.json").exists() else 0
    values["additional_disk_bytes"] = disk_bytes(root) + weight_bytes
    if values["additional_disk_bytes"] + 8 * 2**20 > config["additional_disk_budget_bytes"]:
        raise RuntimeError("Shared additional disk budget exhausted")
    return values


@contextmanager
def task_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "pid").write_text(f"{os.getpid()}\n")
        yield


def encode_rows(model, tokenizer, inputs, stores, config, device, should_stop, resources):
    """Commit each pooling independently; a crash between commits recomputes only its missing tail."""
    for split, rows in inputs.items():
        starts = {mode: stores[mode].progress["processed"][split] for mode in MODES}
        pending = {mode: ([], [], []) for mode in MODES}

        def flush(mode):
            embeddings, activities, lengths = pending[mode]
            if embeddings:
                stores[mode].append(split, starts[mode], embeddings, activities, lengths)
                starts[mode] = stores[mode].progress["processed"][split]
                pending[mode] = ([], [], [])
                resources()

        try:
            for position in range(min(starts.values()), len(rows), config["batch_size"]):
                if should_stop():
                    return
                pair = rows[position:position + config["batch_size"]]
                texts = [(config["query_prefix"] if split == "queries" else "") + row["text"] for row in pair]
                lengths = tokenizer(texts, truncation=False, return_length=True)["length"]
                outputs = encode_batch(model, tokenizer, texts, device, config)
                for mode in MODES:
                    for i in range(len(pair)):
                        if position + i >= starts[mode]:
                            pending[mode][0].append(outputs[mode][i:i + 1])
                            # Existing packed store schema: one vector, unit placeholder activity.
                            pending[mode][1].append(np.ones(1, dtype=np.float32))
                            pending[mode][2].append(lengths[i])
                    if len(pending[mode][0]) >= config["shard_rows"]:
                        flush(mode)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > config["host_memory_budget_bytes"]:
                    raise RuntimeError("Host memory budget exhausted")
                if device.type == "cuda" and torch.cuda.max_memory_allocated(device) > config["cuda_allocated_budget_bytes"]:
                    raise RuntimeError("CUDA allocated memory budget exhausted")
        finally:
            for mode in MODES:
                flush(mode)


def verify_encoding(root, stores, dataset):
    for store in stores.values():
        store.verify()
        if any(c["slots"] != c["stop"] - c["start"] for c in store.progress["chunks"]):
            raise ValueError("Pooled artifacts must have exactly one vector per position")
        if store.progress["status"] == "complete" and store.progress["processed"] != {k: len(v) for k, v in store.inputs.items()}:
            raise ValueError("Incomplete encoding labelled complete")
        write_summary(store, dataset)
    # Compare prefixed lengths with the completed original tokenizer audit, without re-encoding.
    original = Path("runs/propsearch-original")
    progress = json.loads((original / "progress.json").read_text())
    raw_lengths = []
    for chunk in progress["chunks"]:
        if chunk["split"] == "queries":
            with np.load(original / chunk["file"], allow_pickle=False) as a:
                raw_lengths.extend(a["token_lengths"].tolist())
    affected = []
    store = stores["mean"]
    for chunk in store.progress["chunks"]:
        if chunk["split"] == "queries":
            with np.load(store.directory / chunk["file"], allow_pickle=False) as a:
                for i, length in enumerate(a["token_lengths"]):
                    pos = chunk["start"] + i
                    if length > 128:
                        affected.append({"position": pos, "id": str(a["ids"][i]),
                                         "unprefixed_tokens": raw_lengths[pos], "prefixed_tokens": int(length),
                                         "prefix_induced_overflow": raw_lengths[pos] <= 128})
    atomic_json(root / "query-truncation.json", {"prefix": "query: ", "affected_inputs": affected,
        "truncated_queries": len(affected), "prefix_induced_overflow": sum(a["prefix_induced_overflow"] for a in affected),
        "original_progress_sha256": sha256(original / "progress.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["acquire", "encode", "verify", "plan"])
    parser.add_argument("--run-dir", type=Path, default=Path("runs/bge-frozen"))
    parser.add_argument("--config", type=Path, default=Path("configs/bge.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    root = args.run_dir
    with task_lock(root):
        if args.action == "acquire":
            acquire(root, config)
            return
        dataset = json.loads(Path("configs/propsearch.json").read_text())
        inputs = load_inputs(Path("data/propsearch"), dataset)
        plan = memory_plan(inputs, config)
        if args.action == "plan":
            print(json.dumps(plan, indent=2))
            return
        if args.action == "verify":
            stores = {mode: EncodingStore(root / mode / "encoding", Path("data/propsearch"),
                        json.loads((root / mode / "encoding/manifest.json").read_text())["identity"],
                        inputs, 0, config["additional_disk_budget_bytes"]) for mode in MODES}
            verify_encoding(root, stores, dataset)
            print("Both encoding conditions verified")
            return
        shared = budget(root, config)
        if time.time() >= shared["deadline_epoch"]:
            raise RuntimeError("Shared two-hour budget expired; continuation approval required")
        device = configure_reference(config, config["device"])
        identity = run_identity(config, dataset, inputs)
        identity.update(bge=config, acquisition_sha256=sha256(root / "acquisition.json"), memory_plan=plan)
        stores = {mode: EncodingStore(root / mode / "encoding", Path("data/propsearch"),
                    {**identity, "pooling": mode}, inputs, shared["deadline_epoch"] - time.time(),
                    config["additional_disk_budget_bytes"]) for mode in MODES}
        for store in stores.values():
            store.verify()
        if all(s.progress["status"] == "complete" for s in stores.values()):
            print("Already encoded; verified")
            return
        stopping = []
        for number in (signal.SIGINT, signal.SIGTERM):
            signal.signal(number, lambda n, frame: stopping.append(n))
        should_stop = lambda: bool(stopping or time.time() >= shared["deadline_epoch"])
        def resources():
            values = check_resources(root, config, device)
            atomic_json(root / "encoding-resources.json", {**values, "updated_epoch": time.time(),
                        "elapsed_shared_wall_seconds": time.time() - shared["started_epoch"]})
        try:
            model, tokenizer = load_pretrained(root, config, device)
            resources()
            encode_rows(model, tokenizer, inputs, stores, config, device, should_stop, resources)
            for store in stores.values():
                complete = store.progress["processed"] == {k: len(v) for k, v in inputs.items()}
                store.checkpoint("complete" if complete else "stopped", None if complete else "requested_stop_or_deadline")
            verify_encoding(root, stores, dataset)
            resources()
            if stopping or any(s.progress["status"] != "complete" for s in stores.values()):
                raise SystemExit(2)
        except Exception:
            for store in stores.values():
                store.checkpoint("failed", traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
