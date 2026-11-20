"""Exact APE reductions with a reusable corpus tensor and bounded query batches."""

import mmap

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

RULES = ("HA", "MaxSim", "H2P")


def reduce_scores(sim, offsets):
    """Same FP32 reductions as the original run; no changed matmul geometry.

    For two query slots, the optimal assignment needs only each row's two best
    columns: distinct best columns win directly; a collision replaces one best
    with its second best. This is exactly the rectangular Hungarian objective,
    using NumPy reductions instead of thousands of tiny solver calls. Larger
    assignments continue to use SciPy, as in official APEScore.
    """
    lengths = np.diff(offsets)
    best = np.maximum.reduceat(sim, offsets[:-1], axis=1)
    result = {"H2P": best.min(axis=0),
              "MaxSim": .5 * (best.mean(axis=0)
                    + np.add.reduceat(sim.max(axis=0), offsets[:-1]) / lengths.astype(np.float32))}
    if len(sim) == 1:
        result["HA"] = result["H2P"].copy()
    elif len(sim) == 2:
        groups = np.repeat(np.arange(len(lengths)), lengths)
        positions = np.arange(sim.shape[1])
        argmax = np.minimum.reduceat(np.where(sim == best[:, groups], positions, sim.shape[1]), offsets[:-1], axis=1)
        other = sim.copy()
        np.put_along_axis(other, argmax, -np.inf, axis=1)
        second = np.maximum.reduceat(other, offsets[:-1], axis=1)
        optimum = np.where(argmax[0] != argmax[1], best[0] + best[1],
                           np.maximum(best[0] + second[1], second[0] + best[1]))
        result["HA"] = optimum * np.float32(.5)
        result["HA"][lengths == 1] = best[:, lengths == 1].max(axis=0)
    else:
        result["HA"] = best.max(axis=0)
        for i in np.flatnonzero(lengths > 1):
            matrix = sim[:, offsets[i]:offsets[i + 1]]
            row, col = linear_sum_assignment(-matrix)
            result["HA"][i] = matrix[row, col].mean()
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("Nonfinite APE scores")
    return result


def block_scores(query, documents, device):
    result = {rule: np.zeros(len(documents), dtype=np.float32) for rule in RULES}
    active = [i for i, doc in enumerate(documents) if len(doc)]
    if not len(query) or not active:
        return result
    offsets = np.r_[0, np.cumsum([len(documents[i]) for i in active])]
    with torch.inference_mode():
        q = F.normalize(torch.as_tensor(np.array(query, copy=True), device=device), dim=-1)
        d = F.normalize(torch.as_tensor(np.concatenate([documents[i] for i in active]), device=device), dim=-1)
        values = reduce_scores((q @ d.T).cpu().numpy(), offsets)
    for rule in RULES:
        result[rule][active] = values[rule]
    return result


class ExactScorer:
    """Keep a fitting corpus on the device; otherwise reuse each block per batch.

    Query matmuls retain the previous dimensions and document boundaries so
    already saved FP32 scores/ranks can be validated and reused. Batching changes
    loop order, not the arithmetic of an individual query-document pair.
    """

    def __init__(self, offsets, corpus, device, slot_budget, memory_budget):
        self.offsets, self.corpus, self.device = offsets, corpus, device
        self.blocks = []
        start = 0
        while start < len(offsets) - 1:
            stop = max(start + 1, int(np.searchsorted(offsets, offsets[start] + slot_budget, side="right") - 1))
            stop = min(stop, len(offsets) - 1)
            self.blocks.append((start, stop, int(offsets[start]), int(offsets[stop])))
            start = stop
        self.resident = None
        self.mode = "streamed_blocks"
        if corpus.nbytes + 128 * 2**20 <= memory_budget:
            self.resident = torch.empty(corpus.shape, dtype=torch.float32, device=device)
            for _, _, a, b in self.blocks:
                self.resident[a:b].copy_(self.normalized_block(a, b))
            self.mode = "resident_corpus"
            self.release_pages()

    def normalized_block(self, start, stop):
        values = torch.as_tensor(np.array(self.corpus[start:stop], copy=True), device=self.device)
        return F.normalize(values, dim=-1)

    def release_pages(self):
        mapping = getattr(self.corpus, "_mmap", None)
        if mapping is not None:
            mapping.madvise(mmap.MADV_DONTNEED)

    def score_batch(self, queries, should_stop):
        count = len(self.offsets) - 1
        scores = {rule: np.zeros((len(queries), count), dtype=np.float32) for rule in RULES}
        normalized = [F.normalize(torch.as_tensor(np.array(q, copy=True), device=self.device), dim=-1) for q in queries]
        with torch.inference_mode():
            for start, stop, a, b in self.blocks:
                if should_stop():
                    return None
                documents = self.resident[a:b] if self.resident is not None else self.normalized_block(a, b)
                offsets = self.offsets[start:stop + 1] - a
                for index, query in enumerate(normalized):
                    if should_stop():
                        return None
                    if not len(query):
                        continue
                    block = reduce_scores((query @ documents.T).cpu().numpy(), offsets)
                    for rule in RULES:
                        scores[rule][index, start:stop] = block[rule]
                if self.resident is None:
                    self.release_pages()
        return scores
