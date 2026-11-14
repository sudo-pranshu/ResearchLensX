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
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from researchlens.contracts import Claim, Document, RetrievedPassage, VerificationResult
from researchlens.models import MB, ModelManager
from researchlens.settings import AppConfig
from researchlens.storage.documents import DocumentStorage, LocalDocumentStorage
from researchlens.store.corpus import CorpusStore, FileInfo, IngestReport

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_TEMPERATURE = 2.24      # fitted on SciFact dev PASSAGE-level probabilities (EXP-H2-003); valid for granularity="passage" only
RETRIEVERS = {"bm25": "BM25 (lexical)", "tfidf": "TF-IDF (lexical)", "e5": "e5-base-v2 (dense)", "rrf:bm25+e5": "Hybrid: RRF(BM25 + e5)"}


@dataclass
class ComponentStatus:
    name: str
    state: str                  # "real" | "fallback" | "mock" | "off"
    detail: str = ""


@dataclass
class Settings:
    retriever: str = "bm25"            # bm25 | tfidf | e5 | rrf:bm25+e5 (dense/hybrid need the dense extra; fall back to BM25 with a note)
    rerank: bool = False
    dedup: float | None = 0.85
    k: int = 6
    max_per_doc: int = 3
    verify: bool = False
    combine_evidence: bool = False      # experimental: multi-passage support; sentence-level, uncalibrated, never evaluated on real data
    exclude_sections: tuple[str, ...] = ("references",)


class Workspace:
    def __init__(self, db_path: str | Path = ":memory:", storage: DocumentStorage | None = None, config: AppConfig | None = None,
                 workspace: str = "local", models: ModelManager | None = None):
        self.workspace = workspace
        self.models = models or ModelManager()
        self._engine_lock = threading.RLock()
        self.config = config
        self.max_upload_bytes = config.max_upload_bytes if config else 50 * 1024 * 1024
        self.max_pages = config.max_pages if config else 400
        self.store = CorpusStore(db_path, workspace=workspace)
        if storage is None:
            import tempfile
            root = Path(tempfile.mkdtemp(prefix="rlx-files-")) if str(db_path) == ":memory:" else Path(db_path).parent / "library"
            storage = LocalDocumentStorage(root)
        self.storage = storage
        from researchlens.retrieval.embedding_cache import EmbeddingCache
        self.cache = EmbeddingCache(config.embedding_dir if config else None)
        self._engines: dict = {}
        self._verifiers: dict[bool, object] = {}
        self._verifier_notes: dict[bool, str] = {}
        self._last_combine = False
        self._fallbacks: dict[str, ComponentStatus] = {}

    @classmethod
    def from_config(cls, config: AppConfig) -> "Workspace":
        return cls(config.db_path, storage=LocalDocumentStorage(config.files_dir), config=config, workspace=config.workspace)

    # ------------------------------------------------------------------ library
    def upload(self, files: Sequence[tuple[str, bytes]]) -> IngestReport:
        """files: (filename, bytes). Each is validated and parsed; the original is stored. Duplicates and failures are
        reported with a reason, never silently dropped. Identity is the document id (arXiv id or content hash)."""
        from researchlens.ingestion.pymupdf_ingestor import PyMuPDFIngestor
        rep = IngestReport()
        with tempfile.TemporaryDirectory() as td:
            for name, data in files:
                label = Path(name).name or "upload.pdf"
                if len(data) > self.max_upload_bytes:
                    rep.failed.append((name, f"file is {len(data) / 1e6:.1f} MB; the limit is {self.max_upload_bytes // (1024 * 1024)} MB")); continue
                if not data.lstrip()[:5] == b"%PDF-":
                    rep.failed.append((name, "not a PDF file (missing %PDF header)")); continue
                path = Path(td) / "upload.pdf"
                path.write_bytes(data)
                try:
                    doc = PyMuPDFIngestor().ingest(str(path), max_pages=self.max_pages)
                except Exception as e:                                     # parse failures are reported with their reason
                    rep.failed.append((name, f"{type(e).__name__}: {str(e).replace('upload.pdf', label)}")); continue
                doc.source_path = label
                if self.store.add_document(doc, status="ready"):
                    try:
                        sf = self.storage.put(self.workspace, doc.doc_id, data)
                        self.store.set_file(doc.doc_id, FileInfo(sf.sha256, sf.key, sf.bytes, doc.n_pages))
                    except Exception as e:                                 # never leave a row without its original
                        self.store.delete(doc.doc_id)
                        rep.failed.append((name, f"could not store the original file: {e}")); continue
                    rep.added.append(doc.doc_id)
                else:
                    rep.duplicates.append(name)
        if rep.added:
            self._engines.clear()
        return rep

    def add_documents(self, docs: Sequence[Document]) -> int:
        n = sum(self.store.add_document(d) for d in docs)
        if n:
            self._engines.clear()
        return n

    def remove(self, doc_id: str) -> bool:
        """Delete the document's row and its stored original. (Embeddings are content-addressed and swept separately.)"""
        ok = self.store.delete(doc_id)
        if ok:
            self.storage.delete(self.workspace, doc_id)
        self._engines.clear()
        return ok

    def library(self):
        return self.store.list_documents()

    # ------------------------------------------------------------------ components
    def verifier(self, combine: bool = False):
        """Real NLI verifier if the model can be loaded, else the labelled mock.
        Default = the validated configuration: passage-level premises (EXP-H2-001, ADR-010) with the temperature fitted on exactly
        those probabilities (EXP-H2-003). `combine=True` is the experimental sentence-level multi-passage mode: uncalibrated (T=1)."""
        self._last_combine = combine
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

    def engine(self, s: Settings, doc_ids: Sequence[str] | None = None):
        from researchlens.generation.extractive import ExtractiveGenerator
        from researchlens.pipeline import build_default_pipeline
        from researchlens.qa.engine import MultiDocQA

        docs = self.store.documents(doc_ids)
        if not docs:
            raise ValueError("The library is empty: upload at least one PDF first.")
        key = (tuple(d.doc_id for d in docs), s.retriever, s.rerank, s.dedup)
        with self._engine_lock:
            return self._engine_locked(key, docs, s)

    def _engine_locked(self, key, docs, s: Settings):
        from researchlens.generation.extractive import ExtractiveGenerator
        from researchlens.pipeline import build_default_pipeline
        from researchlens.qa.engine import MultiDocQA
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

    # ------------------------------------------------------------------ operations
    def _where(self, s: Settings):
        from researchlens.retrieval.postprocess import section_filter
        return section_filter(exclude=s.exclude_sections) if s.exclude_sections else None

    def search(self, query: str, s: Settings, doc_ids: Sequence[str] | None = None) -> tuple[list[RetrievedPassage], dict]:
        return self.models.run(self._search, query, s, doc_ids)         # every model-touching call runs on the inference thread

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

    def graph(self, s: Settings, doc_ids: Sequence[str] | None = None, link_with_nli: bool = False, min_mentions: int = 2):
        """Knowledge graph of the selected papers. `link_with_nli` adds verified supports/contradicts edges between
        a paper's findings and other papers (needs the real NLI model; a mock never creates edges)."""
        from researchlens.knowledge_graph.builder import GraphBuilder
        docs = self.store.documents(doc_ids)
        if not docs:
            raise ValueError("The library is empty: upload at least one PDF first.")
        return self.models.run(self._graph, s, doc_ids, docs, link_with_nli, min_mentions)

    def _graph(self, s, doc_ids, docs, link_with_nli, min_mentions):
        from researchlens.knowledge_graph.builder import GraphBuilder
        key = ("graph", tuple(d.doc_id for d in docs), link_with_nli, min_mentions)
        if key not in self._engines:
            builder = GraphBuilder(min_mentions=min_mentions)
            if link_with_nli:
                qa, _ = self.engine(s, doc_ids)
                self._engines[key] = (builder.build(docs, verifier=self.verifier(False), index=qa.index), None)
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
