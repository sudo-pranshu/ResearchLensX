"""Rank fusion: Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009)
and min-max-normalised linear score fusion."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .base import Retriever


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = 60, weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """rankings: one ranked list of doc ids per system (best first; ranks start at 1).
    score(d) = sum_r w_r / (k + rank_r(d)). Documents absent from a list contribute 0."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += w / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class RRFRetriever(Retriever):
    """Hybrid retriever: fuse the top-`depth` of each component retriever with RRF."""

    def __init__(self, retrievers: Sequence[Retriever], k: int = 60, depth: int = 100,
                 weights: Sequence[float] | None = None, name: str | None = None, fit_components: bool = True):
        self.retrievers, self.k, self.depth, self.weights = list(retrievers), k, depth, weights
        self.fit_components = fit_components
        self.name = name or "rrf(" + "+".join(r.name for r in self.retrievers) + ")"

    def fit(self, docs):
        if self.fit_components:
            for r in self.retrievers:
                r.fit(docs)
        self.n_docs = len(docs)
        return self

    def score(self, query):
        raise NotImplementedError("RRF produces a fused ranking, not a full score vector; use search().")

    def search(self, query, k=10, candidates=None):
        rankings = [[d for d, _ in r.search(query, self.depth, candidates)] for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, self.k, self.weights)[:k]


class LinearFusionRetriever(Retriever):
    """alpha * minmax(lexical) + (1 - alpha) * minmax(dense), normalised over the candidate set."""

    def __init__(self, lexical: Retriever, dense: Retriever, alpha: float = 0.5, fit_components: bool = True):
        self.lex, self.dense, self.alpha, self.fit_components = lexical, dense, alpha, fit_components
        self.name = f"linear(a={alpha})"

    def fit(self, docs):
        if self.fit_components:
            self.lex.fit(docs)
            self.dense.fit(docs)
        self.n_docs = len(docs)
        return self

    def _norm(self, s, idx):
        sub = s[idx]
        lo, hi = sub.min(), sub.max()
        out = np.zeros_like(s, dtype=float)
        out[idx] = (sub - lo) / (hi - lo) if hi > lo else 0.0
        return out

    def score(self, query):  # over all docs
        idx = np.arange(self.n_docs)
        return self.alpha * self._norm(self.lex.score(query), idx) + (1 - self.alpha) * self._norm(self.dense.score(query), idx)

    def search(self, query, k=10, candidates=None):
        idx = np.arange(self.n_docs) if candidates is None else np.asarray(candidates)
        s = self.alpha * self._norm(self.lex.score(query), idx) + (1 - self.alpha) * self._norm(self.dense.score(query), idx)
        sub = s[idx]
        order = np.lexsort((idx, -sub))[:k]
        return [(int(idx[i]), float(sub[i])) for i in order]
