"""Search service: the library index (one per library snapshot), retrieval, QA, claim verification and the verifier policy.

Every model-touching call runs on the ModelManager's inference thread. A scope (selected papers) is a candidate filter at
query time, so scoped and unscoped results come from the same index generation.
"""
from __future__ import annotations

import threading
import time
from typing import Sequence

from researchlens.contracts import Claim, RetrievedPassage, VerificationResult
from researchlens.models import MB, ModelManager
from researchlens.retrieval.chunk_embeddings import ChunkEmbeddingCache
from researchlens.store.corpus import CorpusStore

from .types import NLI_MODEL, NLI_TEMPERATURE, RETRIEVERS, ComponentStatus, Settings


class SearchService:
    def __init__(self, store: CorpusStore, models: ModelManager, cache: ChunkEmbeddingCache):
        self.store, self.models, self.cache = store, models, cache
        self._engines: dict = {}
        self._engine_lock = threading.RLock()
        self._verifiers: dict[bool, object] = {}
        self._verifier_notes: dict[bool, str] = {}
        self._fallbacks: dict[str, ComponentStatus] = {}

    def invalidate(self) -> None:
        self._engines.clear()

    # ------------------------------------------------------------------ verifier policy
    def verifier(self, combine: bool = False):
        """Real NLI verifier if the model can be loaded, else the labelled mock.
        Default = the validated configuration: passage-level premises (EXP-H2-001, ADR-010) with the temperature fitted on exactly
        those probabilities (EXP-H2-003). `combine=True` is the experimental sentence-level multi-passage mode: uncalibrated (T=1)."""
        if combine not in self._verifiers:
            try:
                from researchlens.verification.nli import HFNLIScorer, NLIVerifier
                scorer = self.models.get(f"nli:{NLI_MODEL}", lambda: HFNLIScorer(NLI_MODEL, batch_size=8), 700 * MB)
                if combine:
                    v = NLIVerifier(scorer, granularity="sentence", combine=True)
                    note = f"{NLI_MODEL}, EXPERIMENTAL sentence-level combined evidence, uncalibrated"
                else:
                    v = NLIVerifier(scorer, granularity="passage", temperature=NLI_TEMPERATURE)
                    note = f"{NLI_MODEL}, passage-level, calibrated T={NLI_TEMPERATURE}"
            except Exception as e:           # no torch / no model files / no network
                from researchlens.mocks import MockVerifier
                v, note = MockVerifier(), f"NLI model unavailable ({type(e).__name__}); using MOCK word-overlap verifier"
            self._verifiers[combine], self._verifier_notes[combine] = v, note
        return self._verifiers[combine]

    # ------------------------------------------------------------------ index generations
    def engine(self, s: Settings, doc_ids: Sequence[str] | None = None, include_statuses: Sequence[str] = ("ready",)):
        docs = [d for st in include_statuses for d in self.store.documents(status=st)]   # ONE index per library; a scope is a filter
        docs.sort(key=lambda d: d.doc_id)                  # canonical order so the generation key is stable
        if not docs:
            raise ValueError("The library is empty: upload at least one PDF first.")
        if doc_ids is not None and not ({d.doc_id for d in docs} & set(doc_ids)):
            raise ValueError("None of the selected papers is ready for search.")
        key = (tuple(d.doc_id for d in docs), s.retriever, s.rerank, s.dedup)
        with self._engine_lock:
            return self._engine_locked(key, docs, s)

    def _engine_locked(self, key, docs, s: Settings):
        from researchlens.generation.extractive import ExtractiveGenerator
        from researchlens.pipeline import build_default_pipeline
        from researchlens.qa.engine import MultiDocQA
        # keep only engines of the current library snapshot (plus scoped graph caches) and bound the cache
        self._engines = {k: v for k, v in self._engines.items() if k[0] == "graph" or k[0] == key[0]}
        while len(self._engines) >= 16:
            self._engines.pop(next(iter(self._engines)))
        if key not in self._engines:
            retr, rerank, notes = s.retriever, s.rerank, []
            pipe = None
            while pipe is None:
                try:
                    pipe = build_default_pipeline(docs, retriever=retr, rerank=rerank, dedup_threshold=s.dedup, models=self.models,
                                                  cache=self.cache)
                    pipe.retriever.search("probe", k=1)       # force model loads now so any failure falls back here
                except Exception as e:
                    pipe = None
                    if rerank:
                        notes.append(f"reranker unavailable ({type(e).__name__}); fell back to {retr} only")
                        self._fallbacks["Reranker"] = ComponentStatus("Reranker", "fallback", notes[-1])
                        rerank = False
                    elif retr != "bm25":
                        notes.append(f"{retr} unavailable ({type(e).__name__}); fell back to BM25")
                        self._fallbacks["Retrieval"] = ComponentStatus("Retrieval", "fallback", notes[-1])
                        retr = "bm25"
                    else:
                        raise
            if rerank == s.rerank:
                self._fallbacks.pop("Reranker", None)          # the requested configuration works again: forget the old fallback
            if retr == s.retriever:
                self._fallbacks.pop("Retrieval", None)
            self._engines[key] = (MultiDocQA(pipe.retriever, ExtractiveGenerator(max_sentences=3), docs, None), "; ".join(notes) or None)
        return self._engines[key]

    def rebuild_index(self) -> None:
        """Build the index generation that includes the documents waiting in `indexing` (called by the job runner); they are flipped
        to `ready` by the runner only after this returns, so a document is searchable only once it is part of a generation."""
        self.invalidate()
        self.engine(Settings(), include_statuses=("ready", "indexing"))

    # ------------------------------------------------------------------ operations (each runs on the inference thread)
    @staticmethod
    def _where(s: Settings):
        from researchlens.retrieval.postprocess import section_filter
        return section_filter(exclude=s.exclude_sections) if s.exclude_sections else None

    def search(self, query: str, s: Settings, doc_ids: Sequence[str] | None = None) -> tuple[list[RetrievedPassage], dict]:
        return self.models.run(self._search, query, s, doc_ids)

    def _search(self, query, s, doc_ids):
        qa, note = self.engine(s, doc_ids)
        t = time.perf_counter()
        passages = qa.index.search(query, k=s.k, doc_ids=doc_ids, where=self._where(s))
        return passages, {"latency_ms": (time.perf_counter() - t) * 1e3, "note": note, "retriever": qa.index.name}

    def ask(self, question: str, s: Settings, doc_ids: Sequence[str] | None = None):
        return self.models.run(self._ask, question, s, doc_ids)

    def _ask(self, question, s, doc_ids):
        qa, note = self.engine(s, doc_ids)
        verifier = self.verifier(s.combine_evidence) if s.verify else None        # passed per call: the shared engine is never mutated
        res = qa.answer(question, k=s.k, max_per_doc=s.max_per_doc, doc_ids=doc_ids, where=self._where(s), verify=s.verify,
                        verifier=verifier)
        return res, note

    def verify_claim(self, claim: str, s: Settings, doc_ids: Sequence[str] | None = None) -> tuple[VerificationResult, list[RetrievedPassage]]:
        return self.models.run(self._verify_claim, claim, s, doc_ids)

    def _verify_claim(self, claim, s, doc_ids):
        passages, _ = self._search(claim, s, doc_ids)
        v = self.verifier(s.combine_evidence).verify(Claim(claim_id="user", text=claim, source="user"), passages)
        return v, passages

    # ------------------------------------------------------------------ status
    def status(self, s: Settings | None = None) -> list[ComponentStatus]:
        s = s or Settings()
        fb = dict(self._fallbacks)
        out = [ComponentStatus("Ingestion", "real", "PyMuPDF (text PDFs only; no OCR)")]
        out.append(fb.get("Retrieval") or ComponentStatus("Retrieval", "real", RETRIEVERS.get(s.retriever, s.retriever)))
        out.append(ComponentStatus("Answer generator", "real", "Extractive: sentences copied from cited passages, no LLM"))
        out.append(fb.get("Reranker") or ComponentStatus("Reranker", "real" if s.rerank else "off",
                                                          "cross-encoder ms-marco-MiniLM-L-6-v2" if s.rerank else "disabled"))
        v = self._verifiers.get(s.combine_evidence)
        if v is None:
            out.append(ComponentStatus("Verifier", "off" if not s.verify else "real", "loads on first use"))
        else:
            out.append(ComponentStatus("Verifier", "mock" if getattr(v, "is_mock", False) else "real", self._verifier_notes[s.combine_evidence]))
        return out
