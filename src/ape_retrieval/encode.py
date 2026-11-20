"""Atomic artifact writes and local storage accounting."""

import json
import os
from pathlib import Path


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
