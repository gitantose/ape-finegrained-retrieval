"""Pinned source acquisition and deterministic benchmark-specific qrels/corpora."""

import argparse
from collections import Counter, defaultdict
import hashlib
import gzip
import json
from pathlib import Path

from huggingface_hub import snapshot_download
import pyarrow.parquet as pq

from ape_retrieval.encode import atomic_json
from ape_retrieval.propsearch import load_inputs, sha256


def json_lines(path):
    with (gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else Path(path).open()) as stream:
        for line in stream:
            yield json.loads(line)


def write_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def source_rows(directory, paths):
    for filename in paths:
        path = directory / filename
        if path.suffix == ".parquet":
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                yield from batch.to_pylist()
        else:
            # The inspected NQ release uses JSON Lines despite its .json suffix.
            yield from json_lines(path)


def answer_audit(row):
    answers = row["answers"]
    texts, starts = answers["text"], answers["answer_start"]
    if len(texts) != len(starts):
        raise ValueError("Unaligned answer texts and offsets")
    return [{"text": text, "start": start,
             "valid": bool(text) and isinstance(start, int) and start >= 0
                      and row["context"][start:start + len(text)] == text}
            for text, start in zip(texts, starts)]


def construct(directory, name, spec, row_splits):
    """Stream text once; exact text hashes preserve all alias-to-canonical joins.

    PROPSEARCH keeps released candidate multiplicity; QA corpora deduplicate
    byte-identical contexts/documents, with every original occurrence recorded.
    """
    if name not in ("nq", "squad", "propsearch"):
        raise ValueError(f"Unsupported benchmark: {name}")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise ValueError("Refusing to overwrite a committed processed dataset")
    seen_documents, seen_queries = set(), set()
    memberships = defaultdict(set)
    counts, split_counts, reasons = Counter(), defaultdict(Counter), Counter()
    with (directory / "corpus.jsonl").open("w") as corpus, \
         (directory / "queries.jsonl").open("w") as queries, \
         (directory / "origins.jsonl").open("w") as origins:

        def document(text, native_id, split, preserve_id=False):
            if not isinstance(text, str):
                raise ValueError("Corpus text must be a string")
            if not text.strip() and not preserve_id:
                counts["empty_documents"] += 1
                return None
            doc_id = str(native_id) if preserve_id else "d:" + hashlib.sha256(text.encode()).hexdigest()
            if doc_id not in seen_documents:
                write_line(corpus, {"document_id": doc_id, "text": text, "source": name})
                seen_documents.add(doc_id)
                counts["corpus_documents"] += 1
                counts["corpus_text_utf8_bytes"] += len(text.encode())
            else:
                counts["duplicate_document_occurrences"] += 1
            memberships[doc_id].add(split)
            write_line(origins, {"document_id": doc_id, "original_id": str(native_id), "split": split})
            return doc_id

        if name == "propsearch":
            inputs = row_splits["release"]
            for row in inputs["candidates"]:
                document(row["text"], row["id"], "train", preserve_id=True)
            row_splits = {"train": inputs["queries"]}

        for split, rows in row_splits.items():
            for index, row in enumerate(rows):
                counts["queries"] += 1
                split_counts[split]["queries"] += 1
                native = row.get("question_id", row.get("id", str(index)))
                query_id = split + ":" + str(native)
                if query_id in seen_queries:
                    raise ValueError(f"Duplicate query ID: {query_id}")
                seen_queries.add(query_id)
                positives, answers, audit, reason = [], None, None, None
                if name in ("nq", "squad"):
                    doc_id = document(row["context"], row.get("title", native), split)
                    audit = answer_audit(row)
                    answers = row["answers"]["text"]
                    if doc_id is not None and any(a["valid"] for a in audit):
                        positives = [doc_id]
                    else:
                        reason = "empty_context" if doc_id is None else "no_valid_annotated_answer_span"
                    counts["invalid_answer_spans"] += sum(not a["valid"] for a in audit)
                    query = row["question"]
                    provenance = "annotated_context_with_validated_answer_offset"
                else:
                    query = row["text"]
                    positives = [row["sentence_id"]]
                    if positives[0] not in seen_documents:
                        raise ValueError("Released PROPSEARCH gold is absent from corpus")
                    provenance = "released_proposition_to_source_sentence"
                positives = sorted(set(positives))
                if not isinstance(query, str) or not query.strip():
                    reason, positives = "empty_query", []
                result = {"query_id": query_id, "original_id": str(native), "query": query,
                          "split": split, "positive_ids": positives, "qrel_provenance": provenance,
                          "ineligible_reason": reason}
                if answers is not None:
                    result["answers"] = answers
                if audit is not None:
                    result["answer_span_audit"] = audit
                write_line(queries, result)
                counts["positive_mappings"] += len(positives)
                split_counts[split]["positive_mappings"] += len(positives)
                split_counts[split]["eligible_queries"] += bool(positives)
                if reason:
                    reasons[reason] += 1
    counts["cross_split_shared_documents"] = sum(len(splits) > 1 for splits in memberships.values())
    record = {"format_version": 1, "benchmark": name, "source": spec,
              "counts": dict(counts), "splits": {k: dict(v) for k, v in split_counts.items()},
              "ineligible_reasons": dict(reasons),
              "corpus_scope": "all released splits; every query searches the entire benchmark corpus",
              "deduplication": "none; retain release multiplicity" if name == "propsearch" else "exact original text only",
              "qrels_exhaustive": False,
              "files": {p.name: sha256(p) for p in directory.glob("*.jsonl")}}
    atomic_json(directory / "manifest.json", record)
    return record


def processed_location(config, name):
    spec = config["datasets"][name]
    if name == "propsearch":
        dataset_config = json.loads(Path(spec["config"]).read_text())
        files = {p: sha256(Path(spec["raw_dir"]) / p) for p in dataset_config["files"]}
    else:
        files = {p: sha256(Path(spec["raw_dir"]) / p) for paths in spec["splits"].values() for p in paths}
    identity = {"adapter_version": 1, "spec": spec, "files": files,
                "adapter_sha256": sha256(Path(__file__))}
    if name == "propsearch":
        identity["release_loader_sha256"] = sha256(Path(__file__).parent.parent / "propsearch.py")
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = Path(config["processed_dir"]) / name / key
    return output, identity


def prepare(config, name):
    output, identity = processed_location(config, name)
    if (output / "manifest.json").exists():
        saved = json.loads((output / "manifest.json").read_text())
        for filename, digest in saved["files"].items():
            if sha256(output / filename) != digest:
                raise ValueError("Processed dataset corruption")
    else:
        spec = config["datasets"][name]
        if name == "propsearch":
            inputs = load_inputs(Path(spec["raw_dir"]), json.loads(Path(spec["config"]).read_text()))
            rows = {"release": inputs}
        else:
            rows = {split: source_rows(Path(spec["raw_dir"]), paths) for split, paths in spec["splits"].items()}
        saved = construct(output, name, identity, rows)
    print(json.dumps({"benchmark": name, "processed_dir": str(output), **saved["counts"]}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("acquire", "prepare"))
    parser.add_argument("benchmark", choices=("nq", "squad", "propsearch", "all"))
    parser.add_argument("--config", type=Path, default=Path("configs/benchmarks.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for name in config["datasets"] if args.benchmark == "all" else [args.benchmark]:
        spec = config["datasets"][name]
        if args.action == "acquire":
            if name != "propsearch":
                snapshot_download(spec["dataset_id"], repo_type="dataset", revision=spec["revision"],
                                  allow_patterns=["README.md", *[p for paths in spec["splits"].values() for p in paths]],
                                  local_dir=spec["raw_dir"], max_workers=2)
        else:
            prepare(config, name)


if __name__ == "__main__":
    main()
