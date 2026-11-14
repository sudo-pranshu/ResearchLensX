"""Second-stage reranking with a cross-encoder.

First stage (BM25 / dense / hybrid) returns `depth` candidates; the cross-encoder scores every
(query, passage) pair jointly and the top `k` by cross-encoder score are returned. Same `Retriever`
interface as RRF, so it plugs into `ChunkIndex`, the registry (`rerank:<spec>`) and experiments, and
works inside a candidate subset (QASPER within-paper) because the first stage is searched with the
same `candidates`.

The default model is a small MS MARCO cross-encoder (~90 MB, CPU-friendly). It was trained on web
search, not scientific text; whether it helps here is an empirical question (EXP-H1-007).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import Retriever

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderScorer:
    """Lazily loaded sentence-transformers CrossEncoder; `release()` frees it (one model at a time)."""

    def __init__(self, model: str = DEFAULT_RERANKER, batch_size: int = 16, max_length: int = 512):
        self.model_name, self.batch_size, self.max_length = model, batch_size, max_length
        self._model = None
        self._cache: dict[tuple[str, str], float] = {}     # (query, text) -> score; shared across systems
        self.n_pairs_scored = 0

    def _load(self):
        if self._model is None:
            import researchlens._runtime  # noqa: F401  (macOS OpenMP workaround, must precede torch)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def scores(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        todo = [t for t in dict.fromkeys(texts) if (query, t) not in self._cache]
        if todo:
            out = self._load().predict([(query, t) for t in todo], batch_size=self.batch_size, show_progress_bar=False)
            for t, s in zip(todo, np.asarray(out, dtype=float).reshape(-1)):
                self._cache[(query, t)] = float(s)
            self.n_pairs_scored += len(todo)
        return np.array([self._cache[(query, t)] for t in texts])

    def release(self) -> None:
        self._model = None


class RerankRetriever(Retriever):
    def __init__(self, first_stage: Retriever, scorer, depth: int = 50, name: str | None = None,
                 fit_first_stage: bool = True):
        self.first, self.scorer, self.depth = first_stage, scorer, depth
        self.fit_first_stage = fit_first_stage
        self.name = name or f"rerank({first_stage.name})"
        self.texts: list[str] = []

    def fit(self, docs):
        if self.fit_first_stage:
            self.first.fit(docs)
        self.texts = list(docs)
        self.n_docs = len(docs)
        return self

    def score(self, query):
        raise NotImplementedError("Reranking produces a top-k ordering, not a full score vector; use search().")

    def search(self, query, k=10, candidates=None):
        cand = self.first.search(query, max(self.depth, k), candidates)
        if not cand:
            return []
        ce = self.scorer.scores(query, [self.texts[i] for i, _ in cand])
        order = sorted(range(len(cand)), key=lambda j: (-ce[j], cand[j][0]))   # deterministic tie-break
        return [(cand[j][0], float(ce[j])) for j in order[:k]]
