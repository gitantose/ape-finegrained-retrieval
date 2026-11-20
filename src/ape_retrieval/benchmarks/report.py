"""Export actual completed metrics and explicit missing-status tables."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from ape_retrieval.benchmarks.data import processed_location, json_lines
from ape_retrieval.benchmarks.encoding import identity
from ape_retrieval.benchmarks.retrieval import RULES
from ape_retrieval.encode import atomic_json, disk_bytes
from ape_retrieval.propsearch import sha256
from ape_retrieval.benchmarks.metrics import aggregate_splits


def report(config):
    reference = json.loads(Path(config["reference"]).read_text())
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(), "model": reference,
              "config": config, "benchmarks": {},
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
              "lock_sha256": sha256(Path("uv.lock"))}
    budget = Path(config["runs_dir"]) / "budget.json"
    result["execution_budget_and_environment"] = json.loads(budget.read_text()) if budget.exists() else None
    csv_rows = []
    for name, spec in config["datasets"].items():
        try:
            processed, _ = processed_location(config, name)
            if not (processed / "manifest.json").exists():
                raise FileNotFoundError("Processed dataset not committed yet")
        except (FileNotFoundError, ValueError) as error:
            result["benchmarks"][name] = {"status": "not_prepared", "reason": str(error)}
            continue
        _, key = identity(config, processed, reference)
        run = Path(config["runs_dir"]) / name / key
        record = {"status": "not_evaluated", "processed_dir": str(processed), "run_dir": str(run),
                  "dataset": json.loads((processed / "manifest.json").read_text()),
                  "raw_storage_bytes": disk_bytes(Path(spec["raw_dir"])),
                  "processed_storage_bytes": disk_bytes(processed),
                  "embedding_storage_bytes": disk_bytes(run / "encoding"),
                  "main_split": spec["main_split"]}
        encoding_progress = run / "encoding/progress.json"
        if encoding_progress.exists():
            encoded = json.loads(encoding_progress.read_text())
            record["status"] = "encoding_" + encoded["status"]
            record["encoded_inputs"] = encoded["processed"]
            record["query_truncations"] = encoded["query_truncations"]
        resources = run / "resources.json"
        if resources.exists():
            record["resources"] = json.loads(resources.read_text())
        progress = run / "retrieval/progress.json"
        if progress.exists():
            state = json.loads(progress.read_text())
            record["status"] = state["status"]
            record["processed_queries"] = state["processed_queries"]
            for block in state["blocks"]:
                if sha256(run / "retrieval" / block["file"]) != block["sha256"]:
                    raise ValueError("Cannot report corrupted ranking artifacts")
            if state["status"] == "complete":
                if sha256(run / "retrieval/metrics.json") != state["metrics_sha256"]:
                    raise ValueError("Final metrics integrity failure")
                record["results"] = json.loads((run / "retrieval/metrics.json").read_text())
                records = (row for block in state["blocks"] for row in json_lines(run / "retrieval" / block["file"]))
                if aggregate_splits(records, RULES, config["ks"]) != record["results"]["by_split"]:
                    raise ValueError("Metrics do not recompute from committed relevant ranks")
                for split, rules in record["results"]["by_split"].items():
                    for rule, aggregate in rules.items():
                        if aggregate["metrics"] is None:
                            continue
                        for metric, value in aggregate["metrics"].items():
                            csv_rows.append({"benchmark": name, "split": split, "model": "APE Baseline",
                                "checkpoint": reference["model_revision"], "aggregation": rule,
                                "metric": metric, "value": value, "units": "fraction",
                                "eligible_queries": aggregate["eligible_queries"], "run_dir": str(run)})
        result["benchmarks"][name] = record
    output = Path(config["runs_dir"]) / "reports"
    saved = output / 'results.json'
    compatibility = output.parent / 'layout-compatibility.json'
    if compatibility.exists() and saved.exists():
        historical = json.loads(saved.read_text())
        if historical['config'] != config or any(
                row['status'] != 'complete' or row['run_dir'] != historical['benchmarks'][name]['run_dir']
                or row['results'] != historical['benchmarks'][name]['results']
                for name, row in result['benchmarks'].items()):
            raise ValueError('Saved benchmark report no longer matches the verified runs')
        print(json.dumps({name: 'verified' for name in result['benchmarks']}), flush=True)
        return historical
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "results.json", result)
    fields = ("benchmark", "split", "model", "checkpoint", "aggregation", "metric", "value", "units", "eligible_queries", "run_dir")
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = ["# Pretrained APE retrieval results", "", "All values are fractions. An em dash marks pending results.",
             "Each column uses the configured main split and its own complete corpus.", ""]
    for metric in ("Recall@5", "Recall@20", "MRR"):
        lines += ["## " + metric, "", "| Model | Aggregation | NQ | SQuAD | PROPSEARCH |", "|---|---|---|---|---|"]
        for rule in RULES:
            values = []
            for name in config["datasets"]:
                record = result["benchmarks"][name]
                aggregate = record.get("results", {}).get("by_split", {}).get(record.get("main_split"), {})
                measured = aggregate.get(rule, {}).get("metrics")
                if measured:
                    best = max(v["metrics"][metric] for v in aggregate.values() if v["metrics"])
                    value = f"{measured[metric]:.6f}"
                    values.append("**" + value + "**" if measured[metric] == best else value)
                else:
                    values.append("—")
            lines.append("| " + " | ".join(["APE", rule, *values]) + " |")
        lines.append("")
    lines += ["## Completion status", ""]
    lines.extend(f"- {name}: {record['status']}." for name, record in result["benchmarks"].items())
    (output / "tables.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({name: row["status"] for name, row in result["benchmarks"].items()}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmarks.json"))
    args = parser.parse_args()
    report(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
