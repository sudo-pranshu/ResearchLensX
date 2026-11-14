"""Workspace: the single object the app (and the API) talks to.

A thin facade that composes the application services over the research engine:
  * LibraryService  (services/library.py)   uploads via the job queue, documents, storage hygiene
  * SearchService   (services/search.py)    library index, search, QA, claim verification, verifier policy, component status
  * AnalysisService (services/analysis.py)  knowledge graph, gaps, literature review
plus the infrastructure they share: CorpusStore (metadata), DocumentStorage (originals), ChunkEmbeddingCache, ModelManager and the
job runner. It honestly reports which components are real, which fell back and which are mocks. No UI imports: fully testable offline.

Fallback policy (shown via `Workspace.status()`):
  * cross-encoder reranker unavailable -> BM25-only ranking
  * dense / hybrid retrieval unavailable -> BM25
  * NLI verifier unavailable           -> MockVerifier (word overlap) with is_mock=True, UI shows a MOCK badge
  * answer generator                   -> ExtractiveGenerator (CPU, no LLM, copies sentences from cited passages)
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

from researchlens.contracts import Document, RetrievedPassage, VerificationResult
from researchlens.models import ModelManager
from researchlens.retrieval.chunk_embeddings import ChunkEmbeddingCache
from researchlens.services import (NLI_MODEL, NLI_TEMPERATURE, RETRIEVERS, AnalysisService, ComponentStatus, LibraryService,
                                   SearchService, Settings)
from researchlens.settings import AppConfig
from researchlens.storage.documents import DocumentStorage, LocalDocumentStorage
from researchlens.store.corpus import CorpusStore, IngestReport

__all__ = ["Workspace", "Settings", "ComponentStatus", "NLI_MODEL", "NLI_TEMPERATURE", "RETRIEVERS"]


class Workspace:
    def __init__(self, db_path: str | Path = ":memory:", storage: DocumentStorage | None = None, config: AppConfig | None = None,
                 workspace: str = "local", models: ModelManager | None = None):
        from researchlens.jobs import LocalJobRunner
        self.workspace = workspace
        self.config = config
        self.models = models or ModelManager()
        self.max_upload_bytes = config.max_upload_bytes if config else 50 * 1024 * 1024
        self.max_pages = config.max_pages if config else 400
        self.store = CorpusStore(db_path, workspace=workspace)
        if storage is None:
            root = Path(tempfile.mkdtemp(prefix="rlx-files-")) if str(db_path) == ":memory:" else Path(db_path).parent / "library"
            storage = LocalDocumentStorage(root)
        self.storage = storage
        if config is not None:
            emb_path = config.embedding_dir / "chunks.sqlite"
        else:
            emb_path = ":memory:" if str(db_path) == ":memory:" else Path(db_path).parent / "embeddings.sqlite"
        self.cache = ChunkEmbeddingCache(emb_path)
        self.search_service = SearchService(self.store, self.models, self.cache)
        self.jobs = LocalJobRunner(self.store, self.storage, self.search_service.rebuild_index, max_pages=self.max_pages,
                                   max_upload_bytes=self.max_upload_bytes)
        self.library_service = LibraryService(self.store, self.storage, self.cache, self.jobs, self.search_service.invalidate)
        self.analysis_service = AnalysisService(self.store, self.search_service, self.models)

    @classmethod
    def from_config(cls, config: AppConfig) -> "Workspace":
        return cls(config.db_path, storage=LocalDocumentStorage(config.files_dir), config=config, workspace=config.workspace)

    @property
    def _engines(self) -> dict:                      # exposed for tests and diagnostics
        return self.search_service._engines

    # ---- library
    def submit_uploads(self, files: Sequence[tuple[str, bytes]]):
        return self.library_service.submit_uploads(files)

    def upload(self, files: Sequence[tuple[str, bytes]]) -> IngestReport:
        return self.library_service.upload(files)

    def add_documents(self, docs: Sequence[Document]) -> int:
        return self.library_service.add_documents(docs)

    def remove(self, doc_id: str) -> bool:
        return self.library_service.remove(doc_id)

    def gc_embeddings(self) -> int:
        return self.library_service.gc_embeddings()

    def library(self):
        return self.library_service.library()

    def rebuild_index(self) -> None:
        self.search_service.rebuild_index()

    # ---- search, QA, verification
    def verifier(self, combine: bool = False):
        return self.search_service.verifier(combine)

    def engine(self, s: Settings, doc_ids: Sequence[str] | None = None, include_statuses: Sequence[str] = ("ready",)):
        return self.search_service.engine(s, doc_ids, include_statuses)

    def search(self, query: str, s: Settings, doc_ids: Sequence[str] | None = None) -> tuple[list[RetrievedPassage], dict]:
        return self.search_service.search(query, s, doc_ids)

    def ask(self, question: str, s: Settings, doc_ids: Sequence[str] | None = None):
        return self.search_service.ask(question, s, doc_ids)

    def verify_claim(self, claim: str, s: Settings, doc_ids: Sequence[str] | None = None) -> tuple[VerificationResult, list[RetrievedPassage]]:
        return self.search_service.verify_claim(claim, s, doc_ids)

    # ---- analysis
    def graph(self, s: Settings, doc_ids: Sequence[str] | None = None, link_with_nli: bool = False, min_mentions: int = 2):
        return self.analysis_service.graph(s, doc_ids, link_with_nli, min_mentions)

    def gaps(self, s: Settings, doc_ids: Sequence[str] | None = None):
        return self.analysis_service.gaps(s, doc_ids)

    def review(self, s: Settings, doc_ids: Sequence[str] | None = None, link_with_nli: bool = False):
        return self.analysis_service.review(s, doc_ids, link_with_nli)

    # ---- status
    def status(self, s: Settings | None = None) -> list[ComponentStatus]:
        return self.search_service.status(s)
