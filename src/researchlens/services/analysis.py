"""Analysis service: knowledge graph, research gaps and literature review over a selection of papers."""
from __future__ import annotations

from typing import Sequence

from researchlens.models import ModelManager
from researchlens.store.corpus import CorpusStore

from .search import SearchService
from .types import Settings


class AnalysisService:
    def __init__(self, store: CorpusStore, search: SearchService, models: ModelManager):
        self.store, self.search, self.models = store, search, models

    def graph(self, s: Settings, doc_ids: Sequence[str] | None = None, link_with_nli: bool = False, min_mentions: int = 2):
        """Knowledge graph of the selected papers. `link_with_nli` adds verified supports/contradicts edges between a paper's
        findings and other papers (needs the real NLI model; a mock never creates edges)."""
        docs = self.store.documents(doc_ids)
        if not docs:
            raise ValueError("The library is empty: upload at least one PDF first.")
        return self.models.run(self._graph, s, doc_ids, docs, link_with_nli, min_mentions)

    def _graph(self, s, doc_ids, docs, link_with_nli, min_mentions):
        from researchlens.knowledge_graph.builder import GraphBuilder
        key = ("graph", tuple(d.doc_id for d in docs), link_with_nli, min_mentions)
        cache = self.search._engines
        if key not in cache:
            builder = GraphBuilder(min_mentions=min_mentions)
            if link_with_nli:
                qa, _ = self.search.engine(s, doc_ids)
                cache[key] = (builder.build(docs, verifier=self.search.verifier(False), index=qa.index), None)
            else:
                cache[key] = (builder.build(docs), None)
        return cache[key][0]

    def gaps(self, s: Settings, doc_ids: Sequence[str] | None = None):
        from researchlens.analysis.gaps import find_gaps
        docs = self.store.documents(doc_ids)
        kg = self.graph(s, doc_ids)
        return find_gaps(docs, kg), kg

    def review(self, s: Settings, doc_ids: Sequence[str] | None = None, link_with_nli: bool = False):
        from researchlens.analysis.compare import conflicting_pairs
        from researchlens.analysis.gaps import find_gaps
        from researchlens.analysis.review import build_review
        docs = self.store.documents(doc_ids)
        kg = self.graph(s, doc_ids, link_with_nli=link_with_nli)
        return build_review(docs, kg, find_gaps(docs, kg), conflicting_pairs(kg))
