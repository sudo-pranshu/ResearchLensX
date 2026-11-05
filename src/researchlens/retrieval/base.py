"""Common retriever interface.

Every retriever is fitted once on the full corpus and scores *all* documents for a
query. `search` optionally restricts to a candidate subset, which is how the
QASPER within-paper setting is run without refitting per paper.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Retriever(ABC):
    name: str = "retriever"
    n_docs: int = 0

    @abstractmethod
    def fit(self, docs: Sequence[str]) -> "Retriever": ...

    @abstractmethod
    def score(self, query: str) -> np.ndarray:
        """Return a score for every document in the fitted corpus (higher = better)."""

    def search(self, query: str, k: int = 10, candidates: np.ndarray | None = None) -> list[tuple[int, float]]:
        scores = self.score(query)
        idx = np.arange(self.n_docs) if candidates is None else np.asarray(candidates)
        sub = scores[idx]
        k = min(k, len(idx))
        if k == 0:
            return []
        top = np.argpartition(-sub, k - 1)[:k]
        # Stable tie-break on document index so runs are deterministic.
        top = top[np.lexsort((idx[top], -sub[top]))]
        return [(int(idx[i]), float(sub[i])) for i in top]
