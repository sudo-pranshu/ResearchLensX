"""ChunkIndex: adapts any low-level Retriever (index-based) to the PassageRetriever contract."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

import numpy as np

from researchlens.contracts import Chunk, RetrievedPassage

from .base import Retriever
from .postprocess import suppress_redundant


class ChunkIndex:
    def __init__(self, retriever: Retriever, chunks: list[Chunk], titles: dict[str, str] | None = None,
                 name: str | None = None, prefit: bool = False, dedup_threshold: float | None = None):
        self.retriever, self.chunks = retriever, list(chunks)
        self.dedup_threshold = dedup_threshold          # None = no redundancy suppression
        self.name = name or retriever.name
        titles = titles or {}
        self._by_doc: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.chunks):
            self._by_doc[c.doc_id].append(i)
        if not prefit:
            retriever.fit([c.index_text(titles.get(c.doc_id, "")) for c in self.chunks])

    @property
    def doc_ids(self) -> list[str]:
        return list(self._by_doc)

    def search(self, query: str, k: int = 10, doc_ids: Sequence[str] | None = None,
               where: Callable[[Chunk], bool] | None = None) -> list[RetrievedPassage]:
        """`doc_ids` restricts to documents; `where` is a chunk predicate (e.g. `section_filter`). Both are
        applied BEFORE ranking, so k results are returned whenever enough chunks qualify. With
        `dedup_threshold`, extra candidates are fetched and redundant passages suppressed."""
        cands = None
        if doc_ids is not None or where is not None:
            pool = range(len(self.chunks)) if doc_ids is None else [i for d in doc_ids for i in self._by_doc.get(d, [])]
            if where is not None:
                pool = [i for i in pool if where(self.chunks[i])]
            cands = np.array(sorted(pool), dtype=int)
            if len(cands) == 0:
                return []
        fetch = k if self.dedup_threshold is None else max(3 * k, k + 10)
        hits = self.retriever.search(query, fetch, cands)
        out = [RetrievedPassage(chunk=self.chunks[i], score=s, rank=r, retriever=self.name)
               for r, (i, s) in enumerate(hits, start=1)]
        if self.dedup_threshold is not None:
            out = suppress_redundant(out, self.dedup_threshold, k=k)
        return out
