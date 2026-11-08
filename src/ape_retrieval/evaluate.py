"""Exact original PROPSEARCH ranks using the pinned upstream in-memory scorer."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import time
import traceback

import numpy as np
import torch
from ape.propsearch.benchmark import _multi_vector_ranks, _retrieval_metrics

from ape_retrieval.encode import EncodingStore, atomic_json, disk_bytes, sync_directory
from ape_retrieval.model import configure_reference
from ape_retrieval.propsearch import load_inputs, sha256


def metric_aggregates(ranks: np.ndarray, complexities: np.ndarray) -> dict:
    if (ranks.ndim != 1 or ranks.dtype != np.int64 or not len(ranks)
            or len(ranks) != len(complexities) or (ranks < 1).any()
            or (complexities < 1).any()):
        raise ValueError("Invalid ranks or source complexity labels")
    strata = {}
    for label in ["1", "2", "3", "4", "5", "6+"]:
        mask = complexities >= 6 if label == "6+" else complexities == int(label)
        strata[label] = {"n_queries": int(mask.sum()),
                         "metrics": _retrieval_metrics(ranks[mask]) if mask.any() else None}
    macro_labels = ["2", "3", "4", "5", "6+"]
    missing = [label for label in macro_labels if not strata[label]["n_queries"]]
    if missing:
        raise ValueError(f"Cannot report paper macro with missing buckets: {missing}")
    macro = {key: float(np.mean([strata[label]["metrics"][key] for label in macro_labels]))
             for key in ["MRR", "R@1", "R@5", "R@10"]}
    return {"units": {"MRR": "fraction", "R@1": "percent", "R@5": "percent", "R@10": "percent"},
            "overall_query_mean": {"n_queries": len(ranks), "metrics": _retrieval_metrics(ranks)},
            "by_source_n_atoms": strata,
            "paper_table4_macro_2_3_4_5_6plus": {"buckets": macro_labels, "metrics": macro}}


def load_caches(directory: Path, chunks: list) -> dict:
    caches = {"candidates": {}, "queries": {}}
    for chunk in chunks:
        with np.load(directory / chunk["file"], allow_pickle=False) as arrays:
            vectors = torch.from_numpy(arrays["embeddings"])
            offsets = arrays["offsets"]
            for i, identifier in enumerate(arrays["ids"].tolist()):
                caches[chunk["split"]][identifier] = vectors[offsets[i]:offsets[i + 1]]
    return caches


def memory_plan(directory: Path, progress: dict, config: dict) -> dict:
    candidate_slots = sum(c["slots"] for c in progress["chunks"] if c["split"] == "candidates")
    query_slots = sum(c["slots"] for c in progress["chunks"] if c["split"] == "queries")
    max_query_slots = 0
    for chunk in progress["chunks"]:
        if chunk["split"] == "queries":
            with np.load(directory / chunk["file"], allow_pickle=False) as arrays:
                max_query_slots = max(max_query_slots, int(np.diff(arrays["offsets"]).max()))
    n_candidates = progress["processed"]["candidates"]
    batch = config["query_chunk_size"]
    query_slot_bound = batch * max_query_slots
    sim_bytes = query_slot_bound * candidate_slots * 4
    grouped_bytes = query_slot_bound * n_candidates * 4
    rank_score_bytes = batch * n_candidates * 4
    corpus_bytes = candidate_slots * 384 * 4
    # Allow old/new chunk temporaries to coexist inside the upstream Python loop.
    tensor_bound = corpus_bytes + candidate_slots * 8 + 2 * (sim_bytes + grouped_bytes + rank_score_bytes)
    return {"candidate_slots": candidate_slots, "query_slots": query_slots,
            "max_query_slots": max_query_slots, "candidate_embedding_bytes": corpus_bytes,
            "query_embedding_bytes": query_slots * 384 * 4,
            "worst_chunk_similarity_bytes": sim_bytes,
            "upstream_live_tensor_estimate_bytes": tensor_bound,
            "host_embedding_arrays_and_upstream_normalize_cat_bytes": 3 * corpus_bytes + query_slots * 384 * 4,
            "caveat": "Tensor estimate excludes allocator/library/driver overhead and host Python objects"}


def atomic_npz(path: Path, arrays: dict, disk_budget: int):
    if disk_bytes(path.parent) + sum(a.nbytes for a in arrays.values()) + 1048576 > disk_budget:
        raise RuntimeError("Evaluation disk budget would be exceeded")
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


@contextmanager
def defer_stop_signals():
    # A deadline/signal may interrupt upstream scoring, but never a progress commit.
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGALRM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


class EvaluationRun:
    """Checkpoint only completed upstream query blocks; retain the original deadline."""

    def __init__(self, directory: Path, identity: dict, queries: list, n_candidates: int):
        self.directory, self.queries, self.n_candidates = directory, queries, n_candidates
        self.config = identity["evaluation"]
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path, progress_path = directory / "manifest.json", directory / "progress.json"
        if not manifest_path.exists() and progress_path.exists():
            raise ValueError("Evaluation progress has no immutable manifest")
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            if self.manifest["identity"] != identity:
                raise ValueError("Evaluation resume identity changed")
        else:
            self.manifest = {"identity": identity, "created_utc": datetime.now(timezone.utc).isoformat(),
                             "started_epoch": time.time(),
                             "deadline_epoch": time.time() + self.config["budget_seconds"]}
            atomic_json(manifest_path, self.manifest)
        if progress_path.exists():
            self.progress = json.loads(progress_path.read_text())
        else:
            self.progress = {"status": "prepared", "processed_queries": 0, "chunks": []}
            self.checkpoint("prepared")

    def checkpoint(self, status: str, reason=None):
        with defer_stop_signals():
            self.progress.update(status=status, reason=reason,
                                 updated_utc=datetime.now(timezone.utc).isoformat(),
                                 elapsed_wall_seconds=time.time() - self.manifest["started_epoch"],
                                 host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            atomic_json(self.directory / "progress.json", self.progress)

    def verify(self) -> np.ndarray:
        ranks = np.zeros(len(self.queries), dtype=np.int64)
        position = 0
        for chunk in self.progress["chunks"]:
            start, stop = chunk["start"], chunk["stop"]
            if start != position or not start < stop <= len(ranks):
                raise ValueError("Invalid evaluation coverage")
            path = self.directory / chunk["file"]
            if sha256(path) != chunk["sha256"]:
                raise ValueError("Rank artifact SHA-256 mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                self.validate(arrays, start, stop)
                ranks[start:stop] = arrays["ranks"]
            position = stop
        if position != self.progress["processed_queries"]:
            raise ValueError("Rank artifacts disagree with progress")
        if self.progress["status"] == "complete":
            if position != len(self.queries):
                raise ValueError("Complete status lacks full query coverage")
            if set(self.progress.get("final_artifacts", {})) != {"predictions.npz", "metrics.json"}:
                raise ValueError("Complete status lacks final artifacts")
            for filename, digest in self.progress["final_artifacts"].items():
                if sha256(self.directory / filename) != digest:
                    raise ValueError(f"Final artifact SHA-256 mismatch: {filename}")
        return ranks

    def validate(self, arrays, start: int, stop: int):
        ranks = arrays["ranks"]
        if (stop <= start or ranks.dtype != np.int64 or ranks.shape != (stop - start,)
                or not ((ranks >= 1) & (ranks <= self.n_candidates)).all()):
            raise ValueError("Invalid exact ranks")
        for key, field in [("query_ids", "id"), ("gold_positions", "gold_position"),
                           ("source_n_atoms", "source_n_atoms")]:
            np.testing.assert_array_equal(arrays[key], [row[field] for row in self.queries[start:stop]])

    def arrays(self, ranks: np.ndarray, start: int, stop: int) -> dict:
        rows = self.queries[start:stop]
        return {"ranks": ranks, "query_ids": np.asarray([r["id"] for r in rows], dtype=np.str_),
                "gold_positions": np.asarray([r["gold_position"] for r in rows], dtype=np.int64),
                "source_n_atoms": np.asarray([r["source_n_atoms"] for r in rows], dtype=np.int64)}

    def append(self, ranks: np.ndarray):
        start = self.progress["processed_queries"]
        stop = start + len(ranks)
        arrays = self.arrays(ranks, start, stop)
        self.validate(arrays, start, stop)
        path = self.directory / f"ranks-{start:09d}.npz"
        with defer_stop_signals():
            atomic_npz(path, arrays, self.config["additional_disk_budget_bytes"])
            self.progress["chunks"].append({"start": start, "stop": stop,
                                            "file": path.name, "sha256": sha256(path)})
            self.progress["processed_queries"] = stop
            self.checkpoint("running")
        print(json.dumps({k: self.progress[k] for k in ["status", "processed_queries", "elapsed_wall_seconds"]}), flush=True)


class EvaluationStopped(Exception):
    pass


def evaluate_blocks(caches: dict, inputs: dict, run: EvaluationRun, device: torch.device):
    candidate_ids = [row["id"] for row in inputs["candidates"]]
    query_ids = [row["id"] for row in inputs["queries"]]
    block_size = run.config["checkpoint_queries"]
    for start in range(run.progress["processed_queries"], len(query_ids), block_size):
        rows = inputs["queries"][start:start + block_size]
        gold = torch.tensor([row["gold_position"] for row in rows], dtype=torch.long, device=device)
        with torch.inference_mode():
            ranks = _multi_vector_ranks(
                caches["queries"], query_ids[start:start + block_size],
                caches["candidates"], candidate_ids, gold, device,
                chunk_size=run.config["query_chunk_size"], all_rules=False,
            )["h_min_to_p"]
        run.progress["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        run.progress["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        run.append(ranks)
        if run.progress["host_peak_rss_bytes"] > run.config["host_memory_budget_bytes"]:
            raise RuntimeError("Host memory budget exceeded; no fallback")
        if run.progress["cuda_peak_allocated_bytes"] > run.config["cuda_allocated_budget_bytes"]:
            raise RuntimeError("CUDA allocated memory budget exceeded; no fallback")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding-dir", type=Path, default=Path("runs/propsearch-original"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/propsearch"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/propsearch-h2p"))
    parser.add_argument("--config", type=Path, default=Path("configs/propsearch-evaluation.json"))
    parser.add_argument("--plan-only", action="store_true", help="Inspect shapes/budgets without evaluating queries")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (config["device"] != "cuda" or config["rule"] != "h_min_to_p"
            or config["rank_rule"] != "1 + count(score > gold_score)"
            or config["checkpoint_queries"] <= 0 or config["query_chunk_size"] <= 0
            or config["checkpoint_queries"] % config["query_chunk_size"]):
        raise ValueError("Invalid explicit evaluation protocol")
    encoded_manifest = json.loads((args.encoding_dir / "manifest.json").read_text())
    encoded_progress = json.loads((args.encoding_dir / "progress.json").read_text())
    original = encoded_manifest["identity"]
    inputs = load_inputs(args.data_dir, original["propsearch"])
    if (encoded_progress["status"] != "complete"
            or encoded_progress["processed"] != {k: len(v) for k, v in inputs.items()}):
        raise ValueError("Encoding is incomplete")
    plan = memory_plan(args.encoding_dir, encoded_progress, config)
    if plan["upstream_live_tensor_estimate_bytes"] > config["cuda_allocated_budget_bytes"]:
        raise ValueError("Planned upstream tensor footprint exceeds the proposed CUDA budget")
    if plan["host_embedding_arrays_and_upstream_normalize_cat_bytes"] > config["host_memory_budget_bytes"]:
        raise ValueError("Planned host arrays alone exceed the proposed host budget")
    if args.plan_only:
        print(json.dumps({"plan": plan, "proposed_budgets": config}, indent=2))
        return
    # Full execution is separate from --plan-only and requires its run/budget approval.
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.encoding_dir / "lock").open("r") as encoding_lock, (args.run_dir / "lock").open("a+") as run_lock:
        fcntl.flock(encoding_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = configure_reference(original["reference"], config["device"])
        identity = {"evaluation": config, "reference": original["reference"], "memory_plan": plan,
                    "encoding_manifest_sha256": sha256(args.encoding_dir / "manifest.json"),
                    "encoding_progress_sha256": sha256(args.encoding_dir / "progress.json"),
                    "python": platform.python_version(),
                    "versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                    "source_sha256": {str(p): sha256(p) for p in sorted(Path("src/ape_retrieval").glob("*.py"))},
                    "uv_lock_sha256": sha256(Path("uv.lock")), "gpu": torch.cuda.get_device_name(),
                    "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
                    "torch_cuda_runtime": torch.version.cuda,
                    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "tf32_cudnn": torch.backends.cudnn.allow_tf32}
        run = EvaluationRun(args.run_dir, identity, inputs["queries"], len(inputs["candidates"]))
        run.verify()
        if run.progress["status"] == "complete":
            print("Evaluation already complete; outputs verified")
            return
        remaining = run.manifest["deadline_epoch"] - time.time()
        if remaining <= 0:
            raise RuntimeError("Evaluation budget expired; continuation approval required")

        def stop(number, frame):
            raise EvaluationStopped("deadline" if number == signal.SIGALRM else "requested_stop")

        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            signal.signal(number, stop)
        (args.run_dir / "pid").write_text(str(os.getpid()) + "\n")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            store = EncodingStore(args.encoding_dir, args.data_dir, original, inputs,
                                  original["propsearch"]["budget_seconds"], original["propsearch"]["disk_budget_bytes"])
            store.verify()
            run.checkpoint("loading")
            caches = load_caches(args.encoding_dir, encoded_progress["chunks"])
            evaluate_blocks(caches, inputs, run, device)
            ranks = run.verify()
            metrics = metric_aggregates(ranks, np.asarray([row["source_n_atoms"] for row in inputs["queries"]]))
            arrays = run.arrays(ranks, 0, len(ranks))
            arrays["gold_candidate_ids"] = np.asarray([row["sentence_id"] for row in inputs["queries"]], dtype=np.str_)
            with defer_stop_signals():
                atomic_npz(args.run_dir / "predictions.npz", arrays, config["additional_disk_budget_bytes"])
                atomic_json(args.run_dir / "metrics.json", metrics)
                run.progress["final_artifacts"] = {name: sha256(args.run_dir / name) for name in ["predictions.npz", "metrics.json"]}
                signal.setitimer(signal.ITIMER_REAL, 0)
                run.checkpoint("complete")
        except EvaluationStopped as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if run.progress["status"] != "complete":
                run.checkpoint("stopped", str(error))
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            run.checkpoint("failed", traceback.format_exc())
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
