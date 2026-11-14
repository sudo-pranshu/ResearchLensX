"""Local job runner for document ingestion.

State machine (one for jobs and documents): queued -> parsing -> indexing -> ready, or failed / cancelled.
  * `submit` stores the original BEFORE enqueueing (so a crash or restart can re-queue it), under a provisional key derived from
    its hash; it is moved to the document id once parsing reveals it.
  * One worker thread processes the queue serially. Parsing runs in a spawned subprocess (see parse_worker.py) with a time and
    memory limit; only this process writes to SQLite.
  * Indexing is a library-wide step: after the queue drains, ONE rebuild makes every `indexing` document searchable and marks it
    `ready`. A document is `ready` only once it is part of the current index generation.
  * Restart recovery: jobs that were `parsing` return to `queued`; documents left in `indexing` are rebuilt.
  * Idempotent: submitting identical bytes again returns the existing job.
The same interface (`JobRunner`) is what a distributed queue would implement later (docs/deployment-readiness).
"""
from __future__ import annotations

import hashlib
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Protocol

from researchlens.contracts import Document
from researchlens.parse_worker import parse_pdf
from researchlens.storage.documents import DocumentStorage
from researchlens.store.corpus import CorpusStore, FileInfo, JobInfo


class JobRunner(Protocol):
    def submit_ingest(self, filename: str, data: bytes) -> JobInfo: ...
    def get(self, job_id: str) -> JobInfo | None: ...
    def list(self, states=None, limit: int = 200) -> list[JobInfo]: ...
    def cancel(self, job_id: str) -> bool: ...
    def retry(self, job_id: str) -> JobInfo | None: ...


def pending_key(sha256: str) -> str:
    return f"pending-{sha256[:24]}"


class LocalJobRunner:
    def __init__(self, store: CorpusStore, storage: DocumentStorage, rebuild: Callable[[], None], *, max_pages: int | None = 400,
                 max_upload_bytes: int = 50 * 1024 * 1024, parse_timeout_s: float = 60.0, max_rss_mb: float = 1500.0,
                 autostart: bool = True):
        self.store, self.storage, self.rebuild = store, storage, rebuild
        self.workspace = store.workspace
        self.max_pages, self.max_upload_bytes = max_pages, max_upload_bytes
        self.parse_timeout_s, self.max_rss_mb = parse_timeout_s, max_rss_mb
        self._cv = threading.Condition()
        self._stop = False
        self._busy = False
        self._cancel: set[str] = set()
        self._thread: threading.Thread | None = None
        self.recovered = store.requeue_interrupted()
        if autostart:
            self.start()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="rlx-jobs", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout)

    def wait_idle(self, timeout: float = 120.0) -> bool:
        """Block until the queue is empty and the worker is idle (used by the synchronous upload and by tests)."""
        end = time.monotonic() + timeout
        with self._cv:
            while self._busy or self._has_work():
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(min(left, 0.2))
        return True

    def _has_work(self) -> bool:
        return bool(self.store.list_jobs(["queued", "parsing"], limit=1)) or bool(self.store.list_documents(status="indexing"))

    # ------------------------------------------------------------------ public API
    def submit_ingest(self, filename: str, data: bytes) -> JobInfo:
        label = Path(filename).name or "upload.pdf"
        if len(data) > self.max_upload_bytes:
            return self.store.create_job("ingest", label, state="failed",
                                         error=f"file is {len(data) / 1e6:.1f} MB; the limit is {self.max_upload_bytes // (1024 * 1024)} MB")
        if data.lstrip()[:5] != b"%PDF-":
            return self.store.create_job("ingest", label, state="failed", error="not a PDF file (missing %PDF header)")
        sha = hashlib.sha256(data).hexdigest()
        known = self.store.find_by_sha256(sha)
        if known:                                                   # identical bytes already in the library: no storing, no parsing
            job = self.store.create_job("ingest", label, state="ready")
            self.store.update_job(job.id, outcome="duplicate", doc_id=known, progress=1.0)
            return self.store.get_job(job.id)
        key = pending_key(sha)
        self.storage.put(self.workspace, key, data)               # original is durable before the job exists
        job = self.store.create_job("ingest", label, pending_key=key, idempotency_key=sha)
        with self._cv:
            self._cv.notify_all()
        return job

    def get(self, job_id: str) -> JobInfo | None:
        return self.store.get_job(job_id)

    def list(self, states=None, limit: int = 200) -> list[JobInfo]:
        return self.store.list_jobs(states, limit)

    def cancel(self, job_id: str) -> bool:
        job = self.store.get_job(job_id)
        if not job or job.state in ("ready", "failed", "cancelled"):
            return False
        if job.state == "queued":
            self._finish(job, "cancelled", error="cancelled by the user")
            return True
        if job.state == "parsing":
            self._cancel.add(job_id)                               # the parse loop stops the child process
            return True
        return False                                                # indexing is a short library-wide step and is not interruptible

    def retry(self, job_id: str) -> JobInfo | None:
        job = self.store.get_job(job_id)
        if not job or job.state not in ("failed", "cancelled") or not job.pending_key or not self.storage.exists(self.workspace, job.pending_key):
            return None                                             # validation failures never stored an original: re-upload instead
        self.store.update_job(job_id, state="queued", error=None, stage=None, progress=0, outcome=None, finished_at=None,
                              attempts=job.attempts)
        with self._cv:
            self._cv.notify_all()
        return self.store.get_job(job_id)

    # ------------------------------------------------------------------ worker
    def _loop(self) -> None:
        while True:
            with self._cv:
                if self._stop:
                    return
                self._busy = False
                self._cv.notify_all()
            job = self._next_job()
            if job is None:
                if self.store.list_documents(status="indexing"):
                    self._run_rebuild()
                    continue
                with self._cv:
                    if self._stop:
                        return
                    self._cv.wait(0.25)
                continue
            with self._cv:
                self._busy = True
            try:
                self._run_ingest(job)
            except Exception as e:                                  # never let one job kill the worker
                self._finish(self.store.get_job(job.id) or job, "failed", error=f"{type(e).__name__}: {e}")

    def _next_job(self) -> JobInfo | None:
        queued = self.store.list_jobs(["queued"], limit=1000)
        return queued[-1] if queued else None                       # list is newest first: take the oldest

    def _finish(self, job: JobInfo, state: str, error: str | None = None, outcome: str | None = None, doc_id: str | None = None) -> None:
        self.store.update_job(job.id, state=state, error=error, outcome=outcome, finished_at=time.time(),
                              **({"doc_id": doc_id} if doc_id else {}), progress=1.0 if state == "ready" else job.progress)
        self._cancel.discard(job.id)

    def _run_ingest(self, job: JobInfo) -> None:
        self.store.update_job(job.id, state="parsing", stage="parsing the PDF", progress=0.1, started_at=time.time(),
                              attempts=job.attempts + 1)
        if not job.pending_key or not self.storage.exists(self.workspace, job.pending_key):
            return self._finish(job, "failed", error="the uploaded file is no longer stored; please upload it again")
        with tempfile.TemporaryDirectory(prefix="rlx-parse-") as td:
            path = Path(td) / "upload.pdf"
            with self.storage.open_read(self.workspace, job.pending_key) as f:
                data = f.read()
            path.write_bytes(data)
            res = parse_pdf(str(path), self.max_pages, self.parse_timeout_s, self.max_rss_mb,
                            should_cancel=lambda: job.id in self._cancel)
        label = job.filename or "upload.pdf"
        if not res.ok:
            if res.error == "cancelled":
                return self._finish(job, "cancelled", error="cancelled by the user")
            return self._finish(job, "failed", error=(res.error or "parsing failed").replace("upload.pdf", label))
        doc = Document.model_validate_json(res.document_json)
        doc.source_path = label
        self.store.update_job(job.id, stage="storing", progress=0.6)
        if self.store.add_document(doc, status="indexing"):
            sf = self.storage.put(self.workspace, doc.doc_id, data)
            self.store.set_file(doc.doc_id, FileInfo(sf.sha256, sf.key, sf.bytes, doc.n_pages))
            self.store.update_job(job.id, state="indexing", stage="waiting for the index", progress=0.8, doc_id=doc.doc_id, outcome="added")
        else:
            self.storage.delete(self.workspace, job.pending_key)
            self._finish(job, "ready", outcome="duplicate", doc_id=doc.doc_id)
            return
        self.storage.delete(self.workspace, job.pending_key)

    def _run_rebuild(self) -> None:
        """One library-wide rebuild for every document waiting in `indexing`, then flip them (and their jobs) to ready."""
        waiting = self.store.list_documents(status="indexing")
        try:
            self.rebuild()
        except Exception as e:
            for d in waiting:
                self.store.set_status(d.doc_id, "failed", f"indexing failed: {type(e).__name__}: {e}")
            for j in self.store.list_jobs(["indexing"]):
                self._finish(j, "failed", error=f"indexing failed: {type(e).__name__}: {e}")
            return
        for d in waiting:
            self.store.set_status(d.doc_id, "ready")
        for j in self.store.list_jobs(["indexing"]):
            self._finish(j, "ready", outcome=j.outcome or "added", doc_id=j.doc_id)
