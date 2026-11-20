"""Metrics from exact relevant ranks, independent of scoring and retrieval."""

import numpy as np
from collections import defaultdict


def query_metrics(positive_ranks, ks=(1, 5, 10, 20)):
    """Missing judgments are ineligible (None), not fabricated zero scores.

    A known relevant item absent from a truncated retrieval result has rank None.
    Such results require an explicit MRR cutoff in the caller's metadata.
    """
    if not positive_ranks:
        return None
    ranks = np.asarray([np.inf if r is None else r for r in positive_ranks], dtype=float)
    if np.isnan(ranks).any() or (ranks < 1).any() or (ranks[np.isfinite(ranks)] % 1).any():
        raise ValueError("Ranks must be positive integers or missing")
    if len(set(ranks[np.isfinite(ranks)].tolist())) != np.isfinite(ranks).sum():
        raise ValueError("Distinct positive documents must have distinct deterministic ranks")
    return {"MRR": float(1 / ranks.min()),
            **{f"Recall@{k}": float(np.mean(ranks <= k)) for k in ks},
            **{f"Success@{k}": float(np.any(ranks <= k)) for k in ks}}


def aggregate_metrics(rows, ks=(1, 5, 10, 20)):
    values = [query_metrics(row["positive_ranks"], ks) for row in rows]
    eligible = [v for v in values if v is not None]
    return {"queries": len(values), "eligible_queries": len(eligible),
            "queries_without_qrels": len(values) - len(eligible),
            "units": "fraction", "MRR_cutoff": None,
            "metrics": {key: float(np.mean([v[key] for v in eligible]))
                        for key in eligible[0]} if eligible else None}


def aggregate_splits(records, rules, ks=(1, 5, 10, 20)):
    """Keep only relevant ranks while streaming larger prediction records."""
    ranks = defaultdict(lambda: {rule: [] for rule in rules})
    for row in records:
        for rule in rules:
            ranks[row["split"]][rule].append({"positive_ranks": row["aggregations"][rule]["positive_ranks"]})
    return {split: {rule: aggregate_metrics(values, ks) for rule, values in by_rule.items()}
            for split, by_rule in ranks.items()}
