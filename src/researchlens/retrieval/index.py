"""ChunkIndex: adapts any low-level Retriever (index-based) to the PassageRetriever contract."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from researchlens.contracts import Chunk, RetrievedPassage

from .base import Retriever


class ChunkIndex:
    def __init__(self, retriever: Retriever, chunks: list[Chunk], titles: dict[str, str] | None = None,
                 name: str | None = None, prefit: bool = False):
        self.retriever, self.chunks = retriever, list(chunks)
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

    def search(self, query: str, k: int = 10, doc_ids: Sequence[str] | None = None) -> list[RetrievedPassage]:
        cands = None
        if doc_ids is not None:
            cands = np.array(sorted(i for d in doc_ids for i in self._by_doc.get(d, [])), dtype=int)
            if len(cands) == 0:
                return []
        hits = self.retriever.search(query, k, cands)
        return [RetrievedPassage(chunk=self.chunks[i], score=s, rank=r, retriever=self.name)
                for r, (i, s) in enumerate(hits, start=1)]
