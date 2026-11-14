"""SQLite corpus store: the persistent library behind the app's "upload papers" feature.

Stores parsed `Document` objects (as JSON) with queryable metadata. Chunks are NOT stored: they are
derived deterministically by `chunk_document`, so changing chunking never needs a migration.

  * add_pdf / add_pdfs   ingest with PyMuPDF; duplicates (same doc_id) are skipped, failures are reported
  * add_document         store an already parsed Document (QASPER loader, fixtures)
  * documents(...)       load documents, optionally filtered by metadata (ids, year range, title text)
  * delete / list_documents / count
  * workspace scoping: every row carries a workspace id (default "local"); a store only ever sees its own workspace
  * status ("queued" / "parsing" / "indexing" / "ready" / "failed" / "cancelled") and error text per document; original file metadata

The schema is versioned (`store/migrations.py`); opening an older library file upgrades it in place.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from researchlens.contracts import SCHEMA_VERSION, Document
from researchlens.store.migrations import migrate, user_version

STATUSES = ("queued", "parsing", "indexing", "ready", "failed", "cancelled")


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)                # doc_ids
    duplicates: list[str] = field(default_factory=list)           # paths whose doc_id already existed
    failed: list[tuple[str, str]] = field(default_factory=list)   # (path, error message)


@dataclass
class FileInfo:
    """Metadata of the stored original file (see storage/documents.py)."""
    sha256: str | None = None
    pdf_path: str | None = None
    bytes: int | None = None
    pages: int | None = None


@dataclass
class DocumentInfo:
    doc_id: str
    title: str
    year: int | None
    source: str
    source_path: str | None
    n_sections: int
    n_references: int
    added_at: float
    status: str = "ready"
    error: str | None = None
    sha256: str | None = None
    pdf_path: str | None = None
    bytes: int | None = None
    pages: int | None = None


class CorpusStore:
    def __init__(self, path: str | Path = ":memory:", workspace: str = "local"):
        self.path = str(path)
        self.workspace = workspace
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # Streamlit runs each rerun on a different thread: allow cross-thread use, serialise access with a lock.
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._lock = threading.RLock()
        self._db.execute("PRAGMA journal_mode=WAL")          # readers never block the single writer
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA foreign_keys=ON")
        migrate(self._db)

    def schema_version(self) -> int:
        with self._lock:
            return user_version(self._db)

    def integrity_check(self) -> str:
        """'ok' or a description of the corruption found."""
        with self._lock:
            return "; ".join(r[0] for r in self._db.execute("PRAGMA integrity_check").fetchall())

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "CorpusStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ write
    def add_document(self, doc: Document, status: str = "ready", file: "FileInfo | None" = None) -> bool:
        """Insert a document; returns False (and changes nothing) if its doc_id is already stored in this workspace."""
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        f = file or FileInfo()
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO documents (workspace, doc_id, title, year, source, source_path, n_sections, n_references, "
                "schema_version, added_at, json, status, sha256, pdf_path, bytes, pages) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.workspace, doc.doc_id, doc.title, doc.year, doc.source, doc.source_path, len(doc.sections),
                 len(doc.references), doc.schema_version, time.time(), doc.model_dump_json(), status,
                 f.sha256, f.pdf_path, f.bytes, f.pages))
            self._db.commit()
            return cur.rowcount == 1

    def set_status(self, doc_id: str, status: str, error: str | None = None) -> bool:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self._lock:
            cur = self._db.execute("UPDATE documents SET status = ?, error = ? WHERE workspace = ? AND doc_id = ?",
                                   (status, error, self.workspace, doc_id))
            self._db.commit()
            return cur.rowcount == 1

    def set_file(self, doc_id: str, file: "FileInfo") -> bool:
        with self._lock:
            cur = self._db.execute("UPDATE documents SET sha256 = ?, pdf_path = ?, bytes = ?, pages = ? WHERE workspace = ? AND doc_id = ?",
                                   (file.sha256, file.pdf_path, file.bytes, file.pages, self.workspace, doc_id))
            self._db.commit()
            return cur.rowcount == 1

    def add_pdf(self, pdf_path: str | Path) -> tuple[str, bool]:
        from researchlens.ingestion.pymupdf_ingestor import PyMuPDFIngestor
        doc = PyMuPDFIngestor().ingest(str(pdf_path))
        return doc.doc_id, self.add_document(doc)

    def add_pdfs(self, pdf_paths: Iterable[str | Path]) -> IngestReport:
        """Ingest many PDFs; one bad file never aborts the batch."""
        rep = IngestReport()
        for p in pdf_paths:
            try:
                doc_id, is_new = self.add_pdf(p)
            except Exception as e:                       # report, do not hide
                rep.failed.append((str(p), f"{type(e).__name__}: {e}"))
                continue
            (rep.added if is_new else rep.duplicates).append(doc_id if is_new else str(p))
        return rep

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM documents WHERE workspace = ? AND doc_id = ?", (self.workspace, doc_id))
            self._db.commit()
            return cur.rowcount == 1

    # ------------------------------------------------------------------ read
    def count(self, status: str | None = None) -> int:
        with self._lock:
            if status is None:
                return self._db.execute("SELECT COUNT(*) FROM documents WHERE workspace = ?", (self.workspace,)).fetchone()[0]
            return self._db.execute("SELECT COUNT(*) FROM documents WHERE workspace = ? AND status = ?",
                                    (self.workspace, status)).fetchone()[0]

    def _where(self, doc_ids, year_min, year_max, title_contains, status=None):
        clauses, args = ["workspace = ?"], [self.workspace]
        if doc_ids is not None:
            clauses.append(f"doc_id IN ({','.join('?' * len(doc_ids))})" if doc_ids else "0")
            args += list(doc_ids)
        if year_min is not None:
            clauses.append("year >= ?"); args.append(year_min)         # documents with unknown year never match a range
        if year_max is not None:
            clauses.append("year <= ?"); args.append(year_max)
        if title_contains:
            clauses.append("title LIKE ? ESCAPE '\\'")
            args.append("%" + title_contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        if status is not None:
            clauses.append("status = ?"); args.append(status)
        return " WHERE " + " AND ".join(clauses), args

    _INFO_COLS = ("doc_id, title, year, source, source_path, n_sections, n_references, added_at, "
                  "status, error, sha256, pdf_path, bytes, pages")

    def list_documents(self, doc_ids: Sequence[str] | None = None, year_min: int | None = None,
                       year_max: int | None = None, title_contains: str | None = None,
                       status: str | None = None) -> list[DocumentInfo]:
        """Metadata for every document in the workspace (any status unless `status` is given)."""
        where, args = self._where(doc_ids, year_min, year_max, title_contains, status)
        with self._lock:
            rows = self._db.execute(f"SELECT {self._INFO_COLS} FROM documents{where} ORDER BY added_at, doc_id", args).fetchall()
        return [DocumentInfo(*r) for r in rows]

    def documents(self, doc_ids: Sequence[str] | None = None, year_min: int | None = None,
                  year_max: int | None = None, title_contains: str | None = None,
                  status: str | None = "ready") -> list[Document]:
        """Parsed documents. Only documents whose status is "ready" take part in search by default."""
        where, args = self._where(doc_ids, year_min, year_max, title_contains, status)
        with self._lock:
            rows = self._db.execute(f"SELECT json, schema_version FROM documents{where} ORDER BY added_at, doc_id", args).fetchall()
        out = []
        for js, ver in rows:
            if ver.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
                raise ValueError(f"stored document has incompatible schema {ver} (current {SCHEMA_VERSION})")
            out.append(Document.model_validate_json(js))
        return out

    def get(self, doc_id: str) -> Document | None:
        docs = self.documents([doc_id], status=None)
        return docs[0] if docs else None

    def info(self, doc_id: str) -> DocumentInfo | None:
        rows = self.list_documents([doc_id])
        return rows[0] if rows else None
