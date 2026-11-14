"""Ordered SQLite migrations driven by `PRAGMA user_version` (no migration framework).

  v1  documents table of the original scaffold (kept so existing library files open)
  v2  documents scoped by workspace (PRIMARY KEY (workspace, doc_id)), status/error and file columns
  v3  jobs table (ingestion jobs: state machine, progress, error, idempotency key)

A database that predates versioning (user_version 0 but a `documents` table present) is treated as v1 and upgraded in place;
every row is preserved. Each migration runs inside one transaction: a failure leaves the previous version untouched.
Add new migrations to the end of MIGRATIONS only; never edit an applied one.
"""
from __future__ import annotations

import sqlite3

V1 = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, year INTEGER, source TEXT NOT NULL, source_path TEXT,
  n_sections INTEGER NOT NULL, n_references INTEGER NOT NULL, schema_version TEXT NOT NULL,
  added_at REAL NOT NULL, json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year);
PRAGMA user_version = 1;
"""

V2 = """
CREATE TABLE documents_v2 (
  workspace TEXT NOT NULL DEFAULT 'local',
  doc_id TEXT NOT NULL,
  title TEXT NOT NULL, year INTEGER, source TEXT NOT NULL, source_path TEXT,
  n_sections INTEGER NOT NULL, n_references INTEGER NOT NULL, schema_version TEXT NOT NULL,
  added_at REAL NOT NULL, json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'ready',
  error TEXT,
  sha256 TEXT, pdf_path TEXT, bytes INTEGER, pages INTEGER,
  PRIMARY KEY (workspace, doc_id));
INSERT INTO documents_v2 (workspace, doc_id, title, year, source, source_path, n_sections, n_references, schema_version, added_at, json)
  SELECT 'local', doc_id, title, year, source, source_path, n_sections, n_references, schema_version, added_at, json FROM documents;
DROP TABLE documents;
ALTER TABLE documents_v2 RENAME TO documents;
CREATE INDEX idx_documents_year ON documents(workspace, year);
CREATE INDEX idx_documents_status ON documents(workspace, status);
PRAGMA user_version = 2;
"""

V3 = """
CREATE TABLE jobs (
  id TEXT PRIMARY KEY, workspace TEXT NOT NULL, kind TEXT NOT NULL,
  state TEXT NOT NULL, stage TEXT, progress REAL NOT NULL DEFAULT 0,
  filename TEXT, pending_key TEXT, doc_id TEXT, outcome TEXT, error TEXT,
  attempts INTEGER NOT NULL DEFAULT 0, idempotency_key TEXT,
  created_at REAL NOT NULL, started_at REAL, finished_at REAL);
CREATE INDEX idx_jobs_ws_state ON jobs(workspace, state, created_at);
CREATE UNIQUE INDEX idx_jobs_idem ON jobs(workspace, idempotency_key) WHERE idempotency_key IS NOT NULL;
PRAGMA user_version = 3;
"""

MIGRATIONS: list[tuple[int, str]] = [(1, V1), (2, V2), (3, V3)]
LATEST = MIGRATIONS[-1][0]


def user_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def migrate(conn: sqlite3.Connection) -> int:
    """Bring the database to the latest version and return it. Safe to call on every start."""
    v = user_version(conn)
    if v == 0 and conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'").fetchone():
        conn.execute("PRAGMA user_version = 1")        # legacy file created before versioning
        v = 1
    for version, script in MIGRATIONS:
        if version > v:
            try:
                conn.executescript("BEGIN;\n" + script + "\nCOMMIT;")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            v = version
    return v
