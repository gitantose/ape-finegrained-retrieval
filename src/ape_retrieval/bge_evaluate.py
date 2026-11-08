"""Exact upstream pooled ranks and matched-query comparison with completed APE."""

import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
from ape.propsearch.benchmark import _pooled_ranks

from ape_retrieval.bge import MODES, budget, check_resources, memory_plan, task_lock, validate_config
from ape_retrieval.encode import EncodingStore, atomic_json
from ape_retrieval.evaluate import (EvaluationRun, EvaluationStopped, atomic_npz,
                                    defer_stop_signals, load_caches, metric_aggregates)
from ape_retrieval.model import configure_reference
from ape_retrieval.propsearch import load_inputs, sha256


def pooled_blocks(caches, inputs, run, device, resources):
    candidate_ids = [r["id"] for r in inputs["candidates"]]
    queries = inputs["queries"]
    for start in range(run.progress["processed_queries"], len(queries), run.config["checkpoint_queries"]):
        rows = queries[start:start + run.config["checkpoint_queries"]]
        gold = torch.tensor([r["gold_position"] for r in rows], device=device)
        with torch.inference_mode():
            ranks = _pooled_ranks(caches["queries"], [r["id"] for r in rows],
                                  caches["candidates"], candidate_ids, gold, device,
                                  chunk_size=run.config["query_chunk_size"])
        run.progress.update(resources())
        run.append(ranks)


def verify_results(root, inputs, config):
    ape_dir = Path("runs/propsearch-h2p")
    ape_progress = json.loads((ape_dir / "progress.json").read_text())
    for name, digest in ape_progress["final_artifacts"].items():
        if sha256(ape_dir / name) != digest:
            raise ValueError("APE final artifact changed")
    with np.load(ape_dir / "predictions.npz", allow_pickle=False) as a:
        ape_arrays = {key: a[key] for key in a.files}
    complexities = np.asarray([r["source_n_atoms"] for r in inputs["queries"]])
    ape_metrics = metric_aggregates(ape_arrays["ranks"], complexities)
    if ape_metrics != json.loads((ape_dir / "metrics.json").read_text()):
        raise ValueError("APE metric recomputation differs")
    comparisons = {"ape_metrics": ape_metrics, "conditions": {}}
    for mode in MODES:
        directory = root / mode / "evaluation"
        identity = json.loads((directory / "manifest.json").read_text())["identity"]
        run = EvaluationRun(directory, identity, inputs["queries"], len(inputs["candidates"]))
        ranks = run.verify()
        if run.progress["status"] != "complete":
            raise ValueError(f"{mode} evaluation incomplete")
        with np.load(directory / "predictions.npz", allow_pickle=False) as a:
            run.validate(a, 0, len(ranks))
            np.testing.assert_array_equal(a["ranks"], ranks)
            for key in ("query_ids", "gold_positions", "gold_candidate_ids", "source_n_atoms"):
                np.testing.assert_array_equal(a[key], ape_arrays[key])
        metrics = metric_aggregates(ranks, complexities)
        if metrics != json.loads((directory / "metrics.json").read_text()):
            raise ValueError("Metric recomputation differs")
        # Independent reciprocal-rank and recall calculation, per aggregate.
        masks = {"overall_query_mean": np.ones(len(ranks), dtype=bool)}
        masks.update({label: complexities >= 6 if label == "6+" else complexities == int(label)
                      for label in ("1", "2", "3", "4", "5", "6+")})
        independent = {}
        for label, mask in masks.items():
            values = ranks[mask]
            result = {"MRR": float(np.mean(1. / values)),
                      **{f"R@{k}": float(np.count_nonzero(values <= k) / len(values) * 100) for k in (1, 5, 10)}}
            reference = metrics["overall_query_mean"]["metrics"] if label == "overall_query_mean" else metrics["by_source_n_atoms"][label]["metrics"]
            np.testing.assert_allclose([result[k] for k in result], [reference[k] for k in result], rtol=0, atol=3e-14)
            independent[label] = result
        independent_macro = {key: float(np.mean([independent[label][key] for label in ("2", "3", "4", "5", "6+")]))
                             for key in ("MRR", "R@1", "R@5", "R@10")}
        reference_macro = metrics["paper_table4_macro_2_3_4_5_6plus"]["metrics"]
        np.testing.assert_allclose([independent_macro[k] for k in independent_macro],
                                   [reference_macro[k] for k in independent_macro], rtol=0, atol=3e-14)
        independent["paper_macro"] = independent_macro
        delta = {}
        for label in ["overall_query_mean", "paper_table4_macro_2_3_4_5_6plus"]:
            delta[label] = {key: metrics[label]["metrics"][key] - ape_metrics[label]["metrics"][key]
                            for key in metrics[label]["metrics"]}
        delta["by_source_n_atoms"] = {label: {key: metrics["by_source_n_atoms"][label]["metrics"][key] -
            ape_metrics["by_source_n_atoms"][label]["metrics"][key] for key in ("MRR", "R@1", "R@5", "R@10")}
            for label in ("1", "2", "3", "4", "5", "6+")}
        comparisons["conditions"][mode] = {"metrics": metrics, "minus_ape": delta,
            "matched_queries": len(ranks), "better_rank_than_ape": int((ranks < ape_arrays["ranks"]).sum()),
            "same_rank_as_ape": int((ranks == ape_arrays["ranks"]).sum()),
            "worse_rank_than_ape": int((ranks > ape_arrays["ranks"]).sum()),
            "independent_recomputation": independent,
            "final_artifacts": run.progress["final_artifacts"],
            "resources_and_progress": run.progress}
    comparisons["verified_epoch"] = time.time()
    comparisons["shared_wall_seconds_to_verification"] = time.time() - budget(root, config)["started_epoch"]
    atomic_json(root / "comparison.json", comparisons)
    print(json.dumps({mode: comparisons["conditions"][mode]["metrics"]["overall_query_mean"] for mode in MODES}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=[*MODES, "verify"])
    parser.add_argument("--run-dir", type=Path, default=Path("runs/bge-frozen"))
    parser.add_argument("--config", type=Path, default=Path("configs/bge.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    root = args.run_dir
    with task_lock(root):
        dataset = json.loads(Path("configs/propsearch.json").read_text())
        inputs = load_inputs(Path("data/propsearch"), dataset)
        if args.condition == "verify":
            verify_results(root, inputs, config)
            return
        shared = budget(root, config)
        if time.time() >= shared["deadline_epoch"]:
            raise RuntimeError("Shared two-hour budget expired; continuation approval required")
        directory = root / args.condition / "encoding"
        manifest = json.loads((directory / "manifest.json").read_text())
        progress = json.loads((directory / "progress.json").read_text())
        if progress["status"] != "complete":
            raise ValueError("Encoding incomplete")
        store = EncodingStore(directory, Path("data/propsearch"), manifest["identity"], inputs, 0,
                              config["additional_disk_budget_bytes"])
        store.verify()
        if progress["processed"] != {k: len(v) for k, v in inputs.items()}:
            raise ValueError("Missing encoded inputs")
        device = configure_reference(config, config["device"])
        evaluation = {**config, "budget_seconds": config["budget_seconds"]}
        identity = {"evaluation": evaluation, "pooling": args.condition, "memory_plan": memory_plan(inputs, config),
                    "encoding_manifest_sha256": sha256(directory / "manifest.json"),
                    "encoding_progress_sha256": sha256(directory / "progress.json"),
                    "source_sha256": {str(p): sha256(p) for p in sorted(Path("src/ape_retrieval").glob("*.py"))},
                    "shared_budget_sha256": sha256(root / "budget.json"),
                    "reference_environment": manifest["identity"]}
        run = EvaluationRun(root / args.condition / "evaluation", identity, inputs["queries"], len(inputs["candidates"]))
        run.verify()
        if run.progress["status"] == "complete":
            print("Already evaluated; verified")
            return
        def stop(number, frame):
            raise EvaluationStopped("deadline" if number == signal.SIGALRM else "requested_stop")
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
            signal.signal(number, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, shared["deadline_epoch"] - time.time()))
        try:
            caches = load_caches(directory, progress["chunks"])
            pooled_blocks(caches, inputs, run, device, lambda: check_resources(root, config, device))
            ranks = run.verify()
            metrics = metric_aggregates(ranks, np.asarray([r["source_n_atoms"] for r in inputs["queries"]]))
            arrays = run.arrays(ranks, 0, len(ranks))
            arrays["gold_candidate_ids"] = np.asarray([r["sentence_id"] for r in inputs["queries"]], dtype=np.str_)
            with defer_stop_signals():
                atomic_npz(run.directory / "predictions.npz", arrays, config["additional_disk_budget_bytes"])
                atomic_json(run.directory / "metrics.json", metrics)
                run.progress["final_artifacts"] = {name: sha256(run.directory / name) for name in ("predictions.npz", "metrics.json")}
                run.progress.update(check_resources(root, config, device))
                signal.setitimer(signal.ITIMER_REAL, 0)
                run.checkpoint("complete")
        except EvaluationStopped as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            run.checkpoint("stopped", str(error))
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            run.checkpoint("failed", traceback.format_exc())
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if run.progress["status"] != "complete":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
