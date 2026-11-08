"""Read the pinned release without losing candidate positions, IDs or qrels."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_inputs(directory: Path, config: dict) -> dict:
    tables = {}
    for filename, expected_hash in config["files"].items():
        path = directory / filename
        if sha256(path) != expected_hash:
            raise ValueError(f"Dataset SHA-256 mismatch: {filename}")
        name = filename.split("/")[0]
        rows = pq.read_table(path).to_pylist()
        if len(rows) != config["expected_rows"][name]:
            raise ValueError(f"Unexpected population for {name}: {len(rows)}")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate release IDs in {name}")
        tables[name] = rows
    return join_inputs(tables["source_sentences"], tables["corpus"], tables["queries"])


def join_inputs(sources: list, distractors: list, queries: list) -> dict:
    candidates = sources + distractors
    if len({row["id"] for row in candidates}) != len(candidates):
        raise ValueError("Candidate IDs overlap across source/distractor tables")
    source_positions = {row["id"]: i for i, row in enumerate(sources)}
    query_rows = []
    for row in queries:
        gold = source_positions[row["sentence_id"]]
        query_rows.append({**row, "text": row["atom"], "gold_position": gold,
                           "source_n_atoms": sources[gold]["n_atoms"]})
    return {"candidates": candidates, "queries": query_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/propsearch.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/propsearch"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.download:
        # Download directly to the data directory, without an Arrow conversion cache.
        for filename in [*config["files"], "README.md"]:
            hf_hub_download(config["dataset_id"], filename, repo_type="dataset",
                            revision=config["dataset_revision"], local_dir=args.data_dir)
    inputs = load_inputs(args.data_dir, config)
    n_sources = config["expected_rows"]["source_sentences"]
    distractor_texts = Counter(r["text"] for r in inputs["candidates"][n_sources:])
    audit = {"dataset_id": config["dataset_id"], "dataset_revision": config["dataset_revision"],
             "rows": {k: len(v) for k, v in inputs.items()}, "parquet_sha256": config["files"],
             "distractor_duplicate_text_groups": sum(n > 1 for n in distractor_texts.values()),
             "source_n_atoms_histogram": dict(sorted(Counter(
                 r["n_atoms"] for r in inputs["candidates"][:n_sources]).items())),
             "card_sha256": sha256(args.data_dir / "README.md")}
    (args.data_dir / "acquisition.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
