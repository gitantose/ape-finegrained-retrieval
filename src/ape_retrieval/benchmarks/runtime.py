"""One explicit, persistent execution allowance for the sequential suite."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import stat
import time

import torch

from ape_retrieval.encode import atomic_json, disk_bytes


def exclusive_disk_bytes(*roots):
    total = 0
    for root in roots:
        for entry in root.rglob('*'):
            try:
                info = entry.lstat()
            except FileNotFoundError:
                # A concurrent atomic publication can rename a listed temporary file.
                continue
            if not stat.S_ISLNK(info.st_mode):
                total += info.st_blocks * 512
    return total


@contextmanager
def run_context(directory, config, budget_seconds, *, until_complete=False):
    root = Path(config["runs_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    with (root / "suite.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget_path = root / "budget.json"
        if budget_path.exists():
            budget = json.loads(budget_path.read_text())
            if budget["config"] != config:
                raise ValueError("Shared suite configuration changed")
        else:
            if not until_complete and (budget_seconds is None or budget_seconds <= 0):
                raise ValueError("First execution requires an explicitly approved --budget-seconds")
            now = time.time()
            budget = {"config": config, "started_epoch": now,
                      "deadline_epoch": None if until_complete else now + budget_seconds,
                      "started_utc": datetime.now(timezone.utc).isoformat(),
                      "python": platform.python_version(), "platform": platform.platform(),
                      "versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
            if until_complete:
                budget["continuation_mode"] = "until_complete"
            atomic_json(budget_path, budget)
        deadline = budget["deadline_epoch"]
        if deadline is None and budget.get("continuation_mode") != "until_complete":
            raise ValueError("An open-ended continuation requires an explicit recorded authorization")
        if deadline is not None and time.time() >= deadline:
            raise RuntimeError("Shared execution deadline expired; continuation budget needs approval")
        stopping = []
        previous = {}
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, lambda n, frame: stopping.append(n))
        (root / "pid").write_text(str(os.getpid()) + "\n")
        last_check = 0
        resources = {}

        def should_stop(force=False):
            nonlocal last_check, resources
            now = time.time()
            if force or now - last_check >= 30:
                last_check = now
                roots = [Path(p) for p in config["storage_roots"]]
                storage = exclusive_disk_bytes(*roots) if config.get("exclusive_allocated_storage") else disk_bytes(*roots)
                resources = {"host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                             "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                             "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                             "suite_disk_bytes": storage,
                             "free_disk_bytes": shutil.disk_usage(root).free,
                             "updated_epoch": now, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
                atomic_json(directory / "resources.json", resources)
                for field, limit in [("host_peak_rss_bytes", "host_memory_bytes"),
                                     ("cuda_peak_allocated_bytes", "cuda_allocated_bytes"),
                                     ("suite_disk_bytes", "storage_bytes")]:
                    if resources[field] > config["limits"][limit]:
                        raise RuntimeError(f"{field} exceeded the explicit allowance; no fallback")
                if resources["free_disk_bytes"] < config["limits"]["minimum_free_disk_bytes"]:
                    raise RuntimeError("Disk reserve reached; no deletion or silent fallback")
            return bool(stopping or (deadline is not None and now >= deadline))

        try:
            should_stop(True)
            yield should_stop
        finally:
            if until_complete:
                (root / "pid").unlink(missing_ok=True)
            for number, handler in previous.items():
                signal.signal(number, handler)
            should_stop(True)
