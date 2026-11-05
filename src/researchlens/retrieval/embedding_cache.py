"""Persistent embedding cache: one .npy per (encoder key, side, exact text list).

Key = sha1(encoder.cache_key + side + sha1 of the texts). Any change to the model id,
revision, prefixes, pooling or the corpus text gives a new key, so stale embeddings are
never reused. Default location `artifacts/embeddings/` (gitignored; share via Drive).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "embeddings"


def texts_digest(texts: Sequence[str]) -> str:
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class EmbeddingCache:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else DEFAULT_DIR
        self.hits = self.misses = 0

    def key(self, encoder_key: str, side: str, texts: Sequence[str]) -> str:
        return hashlib.sha1(f"{encoder_key}|{side}|{texts_digest(texts)}".encode()).hexdigest()[:20]

    def get_or_compute(self, encoder_key: str, side: str, texts: Sequence[str],
                       fn: Callable[[Sequence[str]], np.ndarray]) -> np.ndarray:
        k = self.key(encoder_key, side, texts)
        path = self.root / f"{k}.npy"
        if path.exists():
            self.hits += 1
            return np.load(path)
        self.misses += 1
        t = time.time()
        emb = np.asarray(fn(texts), dtype=np.float32)
        if emb.shape[0] != len(texts):
            raise ValueError(f"encoder returned {emb.shape[0]} rows for {len(texts)} texts")
        self.root.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        (self.root / f"{k}.json").write_text(json.dumps({
            "encoder": encoder_key, "side": side, "n": len(texts), "dim": int(emb.shape[1]),
            "seconds": round(time.time() - t, 2)}))
        return emb
