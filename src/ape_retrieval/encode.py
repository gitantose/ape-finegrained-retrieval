"""Recoverable original PROPSEARCH encoding; no retrieval evaluation."""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time
import traceback

import numpy as np
import torch

from ape_retrieval.model import configure_reference, load_model
from ape_retrieval.propsearch import load_inputs, sha256


def atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def sync_directory(directory: Path):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def disk_bytes(*directories: Path) -> int:
    return sum(p.stat().st_size for directory in directories
               for p in directory.rglob("*") if p.is_file())


def validate_arrays(arrays, rows: list, start: int, stop: int):
    offsets, vectors = arrays["offsets"], arrays["embeddings"]
    counts = np.diff(offsets)
    if (offsets.dtype != np.int64 or offsets.shape != (stop - start + 1,)
            or offsets[0] != 0 or offsets[-1] != len(vectors)
            or not ((counts >= 1) & (counts <= 8)).all()):
        raise ValueError("Invalid slot offsets")
    if vectors.dtype != np.float32 or vectors.shape != (int(offsets[-1]), 384):
        raise ValueError("Invalid FP32 embedding shape")
    if (not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5, rtol=1e-5)):
        raise ValueError("Nonfinite or unnormalized embeddings")
    activities = arrays["activities"]
    if (activities.dtype != np.float32 or activities.shape != (len(vectors),)
            or not np.isfinite(activities).all()
            or not ((activities >= 0) & (activities <= 1)).all()):
        raise ValueError("Invalid slot activities")
    np.testing.assert_array_equal(arrays["ids"], [r["id"] for r in rows[start:stop]])
    lengths = arrays["token_lengths"]
    if lengths.dtype != np.int32 or lengths.shape != (stop - start,) or (lengths < 2).any():
        raise ValueError("Invalid untruncated token lengths")
    if "gold_positions" in arrays:
        np.testing.assert_array_equal(arrays["gold_positions"],
                                      [r["gold_position"] for r in rows[start:stop]])
        np.testing.assert_array_equal(arrays["source_n_atoms"],
                                      [r["source_n_atoms"] for r in rows[start:stop]])


class EncodingStore:
    """Atomic packed NumPy shards and a small authoritative progress manifest."""

    def __init__(self, directory: Path, data_dir: Path, identity: dict, inputs: dict,
                 budget_seconds: float, disk_budget_bytes: int):
        self.directory, self.data_dir, self.inputs = directory, data_dir, inputs
        self.disk_budget_bytes = disk_budget_bytes
        directory.mkdir(parents=True, exist_ok=True)
        self.shards = directory / "shards"
        self.shards.mkdir(exist_ok=True)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists() and (directory / "progress.json").exists():
            raise ValueError("Progress exists without its immutable run manifest")
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            if self.manifest["identity"] != identity:
                raise ValueError("Resume identity changed (data, code, environment or configuration)")
        else:
            now = time.time()
            self.manifest = {"format_version": 1, "identity": identity,
                             "created_utc": datetime.now(timezone.utc).isoformat(),
                             "started_epoch": now, "deadline_epoch": now + budget_seconds}
            atomic_json(manifest_path, self.manifest)
        progress_path = directory / "progress.json"
        if progress_path.exists():
            self.progress = json.loads(progress_path.read_text())
        else:
            self.progress = {"status": "prepared", "chunks": [],
                             "processed": {name: 0 for name in inputs}}
            self.checkpoint("prepared")

    def checkpoint(self, status: str, reason: str | None = None):
        self.progress.update(status=status, reason=reason,
                             updated_utc=datetime.now(timezone.utc).isoformat(),
                             elapsed_wall_seconds=time.time() - self.manifest["started_epoch"],
                             deadline_epoch=self.manifest["deadline_epoch"])
        atomic_json(self.directory / "progress.json", self.progress)

    def verify(self):
        positions = {name: 0 for name in self.inputs}
        for chunk in self.progress["chunks"]:
            split, start, stop = chunk["split"], chunk["start"], chunk["stop"]
            if start != positions[split] or not start < stop <= len(self.inputs[split]):
                raise ValueError("Noncontiguous/overlapping shard coverage")
            path = self.directory / chunk["file"]
            if sha256(path) != chunk["sha256"]:
                raise ValueError(f"Shard SHA-256 mismatch: {path}")
            with np.load(path, allow_pickle=False) as arrays:
                validate_arrays(arrays, self.inputs[split], start, stop)
                if len(arrays["embeddings"]) != chunk["slots"]:
                    raise ValueError("Slot count disagrees with progress")
            positions[split] = stop
        if positions != self.progress["processed"]:
            raise ValueError("Progress disagrees with committed shards")
        return positions

    def append(self, split: str, start: int, embeddings: list, activities: list, lengths: list):
        stop = start + len(embeddings)
        if start != self.progress["processed"][split]:
            raise ValueError("Attempted out-of-order shard commit")
        rows = self.inputs[split]
        arrays = {"offsets": np.asarray([0, *np.cumsum([len(e) for e in embeddings])], dtype=np.int64),
                  "embeddings": np.concatenate(embeddings), "activities": np.concatenate(activities),
                  "ids": np.asarray([r["id"] for r in rows[start:stop]], dtype=np.str_),
                  "token_lengths": np.asarray(lengths, dtype=np.int32)}
        if split == "queries":
            arrays["gold_positions"] = np.asarray([r["gold_position"] for r in rows[start:stop]], dtype=np.int64)
            arrays["source_n_atoms"] = np.asarray([r["source_n_atoms"] for r in rows[start:stop]], dtype=np.int32)
        validate_arrays(arrays, rows, start, stop)
        payload_bytes = sum(a.nbytes for a in arrays.values()) + 16384
        if disk_bytes(self.data_dir, self.directory) + payload_bytes + 1048576 > self.disk_budget_bytes:
            raise RuntimeError("Additional disk budget would be exceeded; refusing shard write")
        path = self.shards / f"{split}-{start:09d}.npz"
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(self.shards)
        chunk = {"split": split, "start": start, "stop": stop,
                 "file": str(path.relative_to(self.directory)), "sha256": sha256(path),
                 "slots": len(arrays["embeddings"]),
                 "truncated_inputs": [{"position": start + i, "id": rows[start + i]["id"],
                                       "token_length": int(length)}
                                      for i, length in enumerate(lengths)
                                      if length > self.manifest["identity"]["reference"]["max_seq_len"]]}
        self.progress["chunks"].append(chunk)
        self.progress["processed"][split] = stop
        self.checkpoint("running")
        print(json.dumps({"processed": self.progress["processed"],
                          "elapsed_seconds": round(self.progress["elapsed_wall_seconds"], 1)}), flush=True)


def encode_split(model, rows: list, split: str, store: EncodingStore, reference: dict,
                 shard_rows: int, should_stop):
    start = store.progress["processed"][split]
    embeddings, activities, lengths = [], [], []

    def flush():
        nonlocal start, embeddings, activities, lengths
        if embeddings:
            pending = embeddings, activities, lengths
            embeddings, activities, lengths = [], [], []
            store.append(split, start, *pending)
            start = store.progress["processed"][split]

    try:
        for position in range(start, len(rows), reference["batch_size"]):
            if should_stop():
                break
            texts = [row["text"] for row in rows[position:position + reference["batch_size"]]]
            tokenized = model.tokenizer(texts, truncation=False, return_length=True)
            batch_embeddings, batch_activities = model.encode(
                texts, batch_size=reference["batch_size"],
                activity_threshold=reference["activity_threshold"],
                max_seq_len=reference["max_seq_len"],
                normalize_embeddings=reference["normalize_embeddings"],
                return_scores=True, convert_to_numpy=True,
            )
            embeddings.extend(batch_embeddings)
            activities.extend(batch_activities)
            lengths.extend(tokenized["length"])
            if len(embeddings) >= shard_rows:
                flush()
    finally:
        # Completed microbatches survive a signal, deadline or later model failure.
        flush()


def run_identity(reference: dict, config: dict, inputs: dict) -> dict:
    return {
        "reference": reference, "propsearch": config,
        "populations": {name: len(rows) for name, rows in inputs.items()},
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "source_sha256": {str(p): sha256(p) for p in sorted(Path("src/ape_retrieval").glob("*.py"))},
        "uv_lock_sha256": sha256(Path("uv.lock")),
        "gpu": torch.cuda.get_device_name(),
        "cuda_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_cuda_runtime": torch.version.cuda,
        "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                           "--format=csv,noheader"], text=True).strip(),
        "device": config["device"],
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def write_summary(store: EncodingStore, config: dict):
    affected = [{"split": chunk["split"], **row}
                for chunk in store.progress["chunks"] for row in chunk["truncated_inputs"]]
    n_sources = config["expected_rows"]["source_sentences"]
    truncated_sources = {row["position"] for row in affected
                         if row["split"] == "candidates" and row["position"] < n_sources}
    summary = {"status": store.progress["status"], "processed": store.progress["processed"],
               "active_slots": sum(chunk["slots"] for chunk in store.progress["chunks"]),
               "truncated_sources": len(truncated_sources),
               "truncated_distractors": sum(row["split"] == "candidates" and row["position"] >= n_sources
                                            for row in affected),
               "truncated_queries": sum(row["split"] == "queries" for row in affected),
               "queries_with_encoded_truncated_gold": sum(row["gold_position"] in truncated_sources
                                                          for row in store.inputs["queries"]),
               "affected_inputs": affected,
               "additional_disk_bytes": disk_bytes(store.data_dir, store.directory)}
    atomic_json(store.directory / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/propsearch-original"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/propsearch"))
    parser.add_argument("--config", type=Path, default=Path("configs/propsearch.json"))
    parser.add_argument("--reference", type=Path, default=Path("configs/reference.json"))
    parser.add_argument("--verify-only", action="store_true", help="Read and verify all committed shards without CUDA")
    args = parser.parse_args()
    config, reference = json.loads(args.config.read_text()), json.loads(args.reference.read_text())
    inputs = load_inputs(args.data_dir, config)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.verify_only:
            identity = json.loads((args.run_dir / "manifest.json").read_text())["identity"]
            if identity["reference"] != reference or identity["propsearch"] != config:
                raise ValueError("Verification configuration differs from the saved run")
        else:
            if config["device"] != "cuda" or reference["batch_size"] != 2:
                raise ValueError("This authorized run requires CUDA batch size 2")
            if config["shard_rows"] <= 0 or config["shard_rows"] % reference["batch_size"]:
                raise ValueError("Shard rows must be a positive multiple of batch size")
            if not reference["normalize_embeddings"] or reference["max_seq_len"] != 128 or reference["activity_threshold"] != .6:
                raise ValueError("Original reproduction policy changed")
            device = configure_reference(reference, config["device"])
            identity = run_identity(reference, config, inputs)
        store = EncodingStore(args.run_dir, args.data_dir, identity, inputs,
                              config["budget_seconds"], config["disk_budget_bytes"])
        # Reserve enough for maximum packed values, IDs, metadata and one temporary shard.
        max_id_length = max(len(row["id"]) for rows in inputs.values() for row in rows)
        upper_bound = sum(map(len, inputs.values())) * (8 * 384 * 4 + 8 * 4 + 32 + 4 * max_id_length)
        upper_bound += disk_bytes(args.data_dir) + config["shard_rows"] * 8 * 384 * 4 + 16 * 1048576
        if upper_bound > config["disk_budget_bytes"]:
            raise RuntimeError(f"Worst-case storage {upper_bound} exceeds authorized budget")
        store.verify()
        if args.verify_only:
            print(json.dumps({"verified": store.progress["processed"],
                              "status": store.progress["status"]}))
            return
        if store.progress["status"] == "complete":
            print("Already complete; all shards verified", flush=True)
            return
        if time.time() >= store.manifest["deadline_epoch"]:
            raise RuntimeError("Original four-hour deadline expired; continuation budget required")
        stop_at = time.monotonic() + store.manifest["deadline_epoch"] - time.time()
        (args.run_dir / "pid").write_text(str(os.getpid()) + "\n")
        stopping = []
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda number, frame: stopping.append(number))

        def should_stop():
            return bool(stopping or time.monotonic() >= stop_at)

        try:
            store.checkpoint("loading")
            model = load_model(reference, device)
            for split, rows in inputs.items():
                encode_split(model, rows, split, store, reference, config["shard_rows"], should_stop)
                if should_stop():
                    break
            store.progress["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            complete = all(store.progress["processed"][name] == len(rows) for name, rows in inputs.items())
            if complete:
                store.verify()
                store.checkpoint("complete")
            else:
                reason = "deadline" if time.monotonic() >= stop_at else "requested_stop"
                store.checkpoint("stopped", reason)
            write_summary(store, config)
            print(json.dumps({"status": store.progress["status"], "reason": store.progress["reason"],
                              "processed": store.progress["processed"]}), flush=True)
        except BaseException:
            store.checkpoint("failed", traceback.format_exc())
            write_summary(store, config)
            raise


if __name__ == "__main__":
    main()
