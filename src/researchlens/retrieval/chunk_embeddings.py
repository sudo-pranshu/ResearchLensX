"""Per-text embedding store: adding one paper embeds only that paper's new chunks.

`EmbeddingCache` (embedding_cache.py) keys on the digest of the WHOLE text list, so any change to the corpus re-embeds
everything; it stays untouched because committed experiment provenance depends on it. This store is content-addressed per text:

    embeddings(space, text_sha256) -> float32 vector        space = "<encoder cache key>|doc" or "...|query"

It exposes the same `get_or_compute(encoder_key, side, texts, fn)` call, so `DenseRetriever` uses it unchanged. Rows are shared
by any document containing the identical text, so deleting a document never deletes vectors; `gc` sweeps what no live text needs.
One SQLite file (WAL), not thousands of small files.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
  space TEXT NOT NULL, text_sha TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL, created_at REAL NOT NULL,
  PRIMARY KEY (space, text_sha));
CREATE INDEX IF NOT EXISTS idx_embeddings_created ON embeddings(created_at);
"""


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ChunkEmbeddingCache:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._lock = threading.RLock()
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.executescript(_SCHEMA)
        self.hits = self.misses = 0
        self.compute_calls: list[int] = []          # number of texts handed to the encoder per call (tests, diagnostics)

    # ------------------------------------------------------------------ main call
    def get_or_compute(self, encoder_key: str, side: str, texts: Sequence[str],
                       fn: Callable[[Sequence[str]], np.ndarray]) -> np.ndarray:
        space = f"{encoder_key}|{side}"
        shas = [text_sha(t) for t in texts]
        found = self._fetch(space, shas)
        missing = [i for i, s in enumerate(shas) if s not in found]
        self.hits += len(shas) - len(missing)
        self.misses += len(missing)
        if missing:
            uniq: dict[str, int] = {}
            for i in missing:                          # identical texts are embedded once
                uniq.setdefault(shas[i], i)
            idx = list(uniq.values())
            emb = np.asarray(fn([texts[i] for i in idx]), dtype=np.float32)
            if emb.shape[0] != len(idx):
                raise ValueError(f"encoder returned {emb.shape[0]} rows for {len(idx)} texts")
            self.compute_calls.append(len(idx))
            self._store(space, [shas[i] for i in idx], emb)
            for s, row in zip(uniq.keys(), emb):
                found[s] = row
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([found[s] for s in shas]).astype(np.float32, copy=False)

    # ------------------------------------------------------------------ storage
    def _fetch(self, space: str, shas: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(shas), 500):                    # SQLite's variable limit
                chunk = list(dict.fromkeys(shas[i:i + 500]))
                q = f"SELECT text_sha, dim, vec FROM embeddings WHERE space = ? AND text_sha IN ({','.join('?' * len(chunk))})"
                for sha, dim, blob in self._db.execute(q, [space, *chunk]):
                    out[sha] = np.frombuffer(blob, dtype=np.float32, count=dim).copy()
        return out

    def _store(self, space: str, shas: Sequence[str], emb: np.ndarray) -> None:
        now = time.time()
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?)",
                                 [(space, s, int(emb.shape[1]), row.tobytes(), now) for s, row in zip(shas, emb)])
            self._db.commit()

    # ------------------------------------------------------------------ maintenance
    def count(self, space: str | None = None) -> int:
        with self._lock:
            if space is None:
                return self._db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            return self._db.execute("SELECT COUNT(*) FROM embeddings WHERE space = ?", (space,)).fetchone()[0]

    def encoder_keys(self) -> list[str]:
        with self._lock:
            return [r[0][:-4] for r in self._db.execute("SELECT DISTINCT space FROM embeddings WHERE space LIKE '%|doc'")]

    def gc(self, live: Iterable[tuple[str, str]], max_query_rows: int = 50_000) -> int:
        """Delete document-side vectors that no live text needs (`live` = (encoder_key, text) pairs for current chunks) and
        trim query-side vectors to the newest `max_query_rows`. Returns the number of rows removed."""
        keep = {(f"{k}|doc", text_sha(t)) for k, t in live}
        removed = 0
        with self._lock:
            rows = self._db.execute("SELECT space, text_sha FROM embeddings WHERE space LIKE '%|doc'").fetchall()
            dead = [r for r in rows if r not in keep]
            self._db.executemany("DELETE FROM embeddings WHERE space = ? AND text_sha = ?", dead)
            removed += len(dead)
            n = self._db.execute("SELECT COUNT(*) FROM embeddings WHERE space LIKE '%|query'").fetchone()[0]
            if n > max_query_rows:
                cur = self._db.execute("DELETE FROM embeddings WHERE rowid IN (SELECT rowid FROM embeddings WHERE space LIKE '%|query' "
                                       "ORDER BY created_at LIMIT ?)", (n - max_query_rows,))
                removed += cur.rowcount
            self._db.commit()
        return removed

    def close(self) -> None:
        with self._lock:
            self._db.close()
