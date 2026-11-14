"""Workspace: the single object the app (and any API) talks to.

Wraps the corpus store, retrieval index, multi-document QA and the NLI verifier, and reports honestly which
components are real, which fell back, and which are mocks. No Streamlit imports: fully testable offline.

Fallback policy (shown in the UI via `Workspace.status()`):
  * cross-encoder reranker unavailable -> BM25-only ranking
  * NLI verifier unavailable           -> MockVerifier (word overlap) with is_mock=True, UI shows a MOCK badge
  * answer generator                   -> ExtractiveGenerator (CPU, no LLM, copies sentences from cited passages)
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from researchlens.contracts import Claim, Document, RetrievedPassage, VerificationResult
from researchlens.store.corpus import CorpusStore, IngestReport

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_TEMPERATURE = 2.24      # fitted on SciFact dev (EXP-H2-003); calibrates reported confidence only


@dataclass
class ComponentStatus:
    name: str
    state: str                  # "real" | "fallback" | "mock" | "off"
    detail: str = ""


@dataclass
class Settings:
    retriever: str = "bm25"            # bm25 | tfidf
    rerank: bool = False
    dedup: float | None = 0.85
    k: int = 6
    max_per_doc: int = 3
    verify: bool = False
    exclude_sections: tuple[str, ...] = ("references",)


class Workspace:
    def __init__(self, db_path: str | Path = ":memory:"):
        self.store = CorpusStore(db_path)
        self._engines: dict = {}
        self._verifier = None
        self._verifier_note = ""
        self._fallbacks: list[ComponentStatus] = []

    # ------------------------------------------------------------------ library
    def upload(self, files: Sequence[tuple[str, bytes]]) -> IngestReport:
        """files: (filename, bytes). Each is parsed; duplicates and failures are reported, never silently dropped."""
        rep = IngestReport()
        with tempfile.TemporaryDirectory() as td:
            for name, data in files:
                path = Path(td) / (Path(name).name or "upload.pdf")
                path.write_bytes(data)
                sub = self.store.add_pdfs([path])
                rep.added += sub.added
                rep.duplicates += [name] * len(sub.duplicates)
                rep.failed += [(name, e) for _, e in sub.failed]
        if rep.added:
            self._engines.clear()
        return rep

    def add_documents(self, docs: Sequence[Document]) -> int:
        n = sum(self.store.add_document(d) for d in docs)
        if n:
            self._engines.clear()
        return n

    def remove(self, doc_id: str) -> bool:
        ok = self.store.delete(doc_id)
        self._engines.clear()
        return ok

    def library(self):
        return self.store.list_documents()

    # ------------------------------------------------------------------ components
    def verifier(self):
        """Real NLI verifier if the model can be loaded, else the labelled mock."""
        if self._verifier is None:
            try:
                from researchlens.verification.nli import HFNLIScorer, NLIVerifier
                scorer = HFNLIScorer(NLI_MODEL, batch_size=8)
                self._verifier = NLIVerifier(scorer, granularity="sentence", combine=True, temperature=NLI_TEMPERATURE)
                self._verifier_note = f"{NLI_MODEL}, calibrated T={NLI_TEMPERATURE}"
            except Exception as e:           # no torch / no model files / no network
                from researchlens.mocks import MockVerifier
                self._verifier = MockVerifier()
                self._verifier_note = f"NLI model unavailable ({type(e).__name__}); using MOCK word-overlap verifier"
        return self._verifier

    def engine(self, s: Settings, doc_ids: Sequence[str] | None = None):
        from researchlens.generation.extractive import ExtractiveGenerator
        from researchlens.pipeline import build_default_pipeline
        from researchlens.qa.engine import MultiDocQA

        docs = self.store.documents(doc_ids)
        if not docs:
            raise ValueError("The library is empty: upload at least one PDF first.")
        key = (tuple(d.doc_id for d in docs), s.retriever, s.rerank, s.dedup)
        if key not in self._engines:
            rerank, note = s.rerank, None
            try:
                pipe = build_default_pipeline(docs, retriever=s.retriever, rerank=rerank, dedup_threshold=s.dedup)
                if rerank:                                     # force the model load now so a failure falls back here
                    pipe.retriever.search("probe", k=1)
            except Exception as e:
                note = f"reranker unavailable ({type(e).__name__}); fell back to {s.retriever} only"
                pipe = build_default_pipeline(docs, retriever=s.retriever, rerank=False, dedup_threshold=s.dedup)
            self._engines[key] = (MultiDocQA(pipe.retriever, ExtractiveGenerator(max_sentences=3), docs, None), note)
        return self._engines[key]

    # ------------------------------------------------------------------ operations
    def _where(self, s: Settings):
        from researchlens.retrieval.postprocess import section_filter
        return section_filter(exclude=s.exclude_sections) if s.exclude_sections else None

    def search(self, query: str, s: Settings, doc_ids: Sequence[str] | None = None) -> tuple[list[RetrievedPassage], dict]:
        qa, note = self.engine(s, doc_ids)
        t = time.perf_counter()
        passages = qa.index.search(query, k=s.k, doc_ids=doc_ids, where=self._where(s))
        return passages, {"latency_ms": (time.perf_counter() - t) * 1e3, "note": note, "retriever": qa.index.name}

    def ask(self, question: str, s: Settings, doc_ids: Sequence[str] | None = None):
        qa, note = self.engine(s, doc_ids)
        qa.verifier = self.verifier() if s.verify else None
        res = qa.answer(question, k=s.k, max_per_doc=s.max_per_doc, doc_ids=doc_ids, where=self._where(s), verify=s.verify)
        return res, note

    def verify_claim(self, claim: str, s: Settings, doc_ids: Sequence[str] | None = None) -> tuple[VerificationResult, list[RetrievedPassage]]:
        passages, _ = self.search(claim, s, doc_ids)
        v = self.verifier().verify(Claim(claim_id="user", text=claim, source="user"), passages)
        return v, passages

    def graph(self, s: Settings, doc_ids: Sequence[str] | None = None, link_with_nli: bool = False, min_mentions: int = 2):
        """Knowledge graph of the selected papers. `link_with_nli` adds verified supports/contradicts edges between
        a paper's findings and other papers (needs the real NLI model; a mock never creates edges)."""
        from researchlens.knowledge_graph.builder import GraphBuilder
        docs = self.store.documents(doc_ids)
        if not docs:
            raise ValueError("The library is empty: upload at least one PDF first.")
        key = ("graph", tuple(d.doc_id for d in docs), link_with_nli, min_mentions)
        if key not in self._engines:
            builder = GraphBuilder(min_mentions=min_mentions)
            if link_with_nli:
                qa, _ = self.engine(s, doc_ids)
                self._engines[key] = (builder.build(docs, verifier=self.verifier(), index=qa.index), None)
            else:
                self._engines[key] = (builder.build(docs), None)
        return self._engines[key][0]

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

    def status(self, s: Settings | None = None) -> list[ComponentStatus]:
        s = s or Settings()
        out = [ComponentStatus("Ingestion", "real", "PyMuPDF (text PDFs only; no OCR)"),
               ComponentStatus("Retrieval", "real", f"{s.retriever.upper()} (lexical, from scratch)"),
               ComponentStatus("Answer generator", "real", "Extractive: sentences copied from cited passages, no LLM")]
        out.append(ComponentStatus("Reranker", "real" if s.rerank else "off",
                                   "cross-encoder ms-marco-MiniLM-L-6-v2" if s.rerank else "disabled"))
        if self._verifier is None:
            out.append(ComponentStatus("Verifier", "off" if not s.verify else "real", "loads on first use"))
        else:
            mock = getattr(self._verifier, "is_mock", False)
            out.append(ComponentStatus("Verifier", "mock" if mock else "real", self._verifier_note))
        return out
