"""SQLite corpus store: the persistent library behind the app's "upload papers" feature.

Stores parsed `Document` objects (as JSON) with queryable metadata. Chunks are NOT stored: they are
derived deterministically by `chunk_document`, so changing chunking never needs a migration.

  * add_pdf / add_pdfs   ingest with PyMuPDF; duplicates (same doc_id) are skipped, failures are reported
  * add_document         store an already parsed Document (QASPER loader, fixtures)
  * documents(...)       load documents, optionally filtered by metadata (ids, year range, title text)
  * delete / list_documents / count
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from researchlens.contracts import SCHEMA_VERSION, Document

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  year INTEGER,
  source TEXT NOT NULL,
  source_path TEXT,
  n_sections INTEGER NOT NULL,
  n_references INTEGER NOT NULL,
  schema_version TEXT NOT NULL,
  added_at REAL NOT NULL,
  json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year);
"""


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)                # doc_ids
    duplicates: list[str] = field(default_factory=list)           # paths whose doc_id already existed
    failed: list[tuple[str, str]] = field(default_factory=list)   # (path, error message)


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


class CorpusStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "CorpusStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ write
    def add_document(self, doc: Document) -> bool:
        """Insert a document; returns False (and changes nothing) if its doc_id is already stored."""
        cur = self._db.execute(
            "INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)",
            (doc.doc_id, doc.title, doc.year, doc.source, doc.source_path, len(doc.sections),
             len(doc.references), doc.schema_version, time.time(), doc.model_dump_json()))
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
        cur = self._db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._db.commit()
        return cur.rowcount == 1

    # ------------------------------------------------------------------ read
    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def _where(self, doc_ids, year_min, year_max, title_contains):
        clauses, args = [], []
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
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", args

    def list_documents(self, doc_ids: Sequence[str] | None = None, year_min: int | None = None,
                       year_max: int | None = None, title_contains: str | None = None) -> list[DocumentInfo]:
        where, args = self._where(doc_ids, year_min, year_max, title_contains)
        rows = self._db.execute("SELECT doc_id,title,year,source,source_path,n_sections,n_references,added_at "
                                f"FROM documents{where} ORDER BY added_at, doc_id", args).fetchall()
        return [DocumentInfo(*r) for r in rows]

    def documents(self, doc_ids: Sequence[str] | None = None, year_min: int | None = None,
                  year_max: int | None = None, title_contains: str | None = None) -> list[Document]:
        where, args = self._where(doc_ids, year_min, year_max, title_contains)
        rows = self._db.execute(f"SELECT json, schema_version FROM documents{where} ORDER BY added_at, doc_id", args).fetchall()
        out = []
        for js, ver in rows:
            if ver.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
                raise ValueError(f"stored document has incompatible schema {ver} (current {SCHEMA_VERSION})")
            out.append(Document.model_validate_json(js))
        return out

    def get(self, doc_id: str) -> Document | None:
        docs = self.documents([doc_id])
        return docs[0] if docs else None
