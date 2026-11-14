"""Library service: uploads (via the job queue), document management and storage hygiene."""
from __future__ import annotations

from typing import Callable, Sequence

from researchlens.contracts import Document
from researchlens.retrieval.chunk_embeddings import ChunkEmbeddingCache
from researchlens.storage.documents import DocumentStorage
from researchlens.store.corpus import CorpusStore, IngestReport


class LibraryService:
    def __init__(self, store: CorpusStore, storage: DocumentStorage, cache: ChunkEmbeddingCache, jobs, on_change: Callable[[], None]):
        self.store, self.storage, self.cache, self.jobs, self._on_change = store, storage, cache, jobs, on_change
        self.workspace = store.workspace

    def submit_uploads(self, files: Sequence[tuple[str, bytes]]):
        """Validate, store and enqueue each file; returns immediately with one job per file (state queued, or failed with a reason)."""
        return [self.jobs.submit_ingest(name, data) for name, data in files]

    def upload(self, files: Sequence[tuple[str, bytes]]) -> IngestReport:
        """Synchronous convenience over the job queue: waits until every file is processed and reports added documents,
        duplicates and failures with reasons. The UI/API use `submit_uploads` and poll `jobs` instead."""
        rep = IngestReport()
        submitted = list(zip([n for n, _ in files], self.submit_uploads(files)))
        if any(j.state == "queued" for _, j in submitted):
            self.jobs.wait_idle()
        for name, j in submitted:
            j = self.jobs.get(j.id) or j
            if j.state == "ready" and j.outcome == "duplicate":
                rep.duplicates.append(name)
            elif j.state == "ready":
                rep.added.append(j.doc_id)
            else:
                rep.failed.append((name, j.error or f"job ended in state {j.state}"))
        return rep

    def add_documents(self, docs: Sequence[Document]) -> int:
        n = sum(self.store.add_document(d) for d in docs)
        if n:
            self._on_change()
        return n

    def remove(self, doc_id: str) -> bool:
        """Delete the document's row and its stored original, then sweep embeddings nothing needs any more."""
        ok = self.store.delete(doc_id)
        if ok:
            self.storage.delete(self.workspace, doc_id)
        self._on_change()
        if ok:
            self.gc_embeddings()
        return ok

    def gc_embeddings(self) -> int:
        """Content-addressed embedding rows are shared between documents, so deleting a document never deletes them; this sweeps."""
        from researchlens.processing.chunking import chunk_document
        texts = [c.index_text(d.title) for d in self.store.documents() for c in chunk_document(d)]
        return self.cache.gc([(k, t) for k in self.cache.encoder_keys() for t in texts])

    def library(self):
        return self.store.list_documents()
