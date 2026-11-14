"""ModelManager: one instance per model, loaded once, bounded memory, inference on a single thread.

Why: before this, every retrieval engine (one per document scope) built its own copy of e5 and the reranker, `ask()` mutated a
shared verifier, and nothing capped memory (about 0.4 GB per e5 copy on an 8 GB machine). Now:
  * `get(key, loader, est_bytes)` returns the single instance for `key`; concurrent callers load it exactly once;
  * a byte budget (not a model count) is enforced by evicting the least recently used models;
  * `run(fn)` executes on one dedicated inference thread, so torch is never entered from several threads at once (the
    process already needs KMP_DUPLICATE_LIB_OK / OMP_NUM_THREADS=1 on macOS, see _runtime.py) and eviction never races a use;
  * a failed load is never cached, and `status()` reports what is resident for the UI and health checks.
Not a model server: a hosted inference service would implement the same calls elsewhere (docs/deployment-readiness).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")

MB = 1024 * 1024


class ModelManager:
    def __init__(self, budget_bytes: int = 2048 * MB):
        self.budget_bytes = budget_bytes
        self._items: "OrderedDict[str, tuple[Any, int]]" = OrderedDict()
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rlx-infer")
        self._thread_id: int | None = None
        self.loads: dict[str, int] = {}          # key -> number of times the loader ran (tests and diagnostics)
        self.evictions = 0

    # ------------------------------------------------------------------ inference thread
    def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run `fn` on the inference thread and return its result (exceptions propagate). Re-entrant: a call made from the
        inference thread itself runs inline instead of deadlocking."""
        if threading.get_ident() == self._thread_id:
            return fn(*args, **kwargs)

        def task() -> T:
            self._thread_id = threading.get_ident()
            return fn(*args, **kwargs)
        return self._pool.submit(task).result()

    # ------------------------------------------------------------------ registry
    def get(self, key: str, loader: Callable[[], T], est_bytes: int = 0) -> T:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key][0]
            self._evict_for(est_bytes, keep=key)
            obj = loader()                         # a failure propagates and nothing is cached
            self._items[key] = (obj, est_bytes)
            self.loads[key] = self.loads.get(key, 0) + 1
            return obj

    def register(self, key: str, obj: Any, est_bytes: int = 0) -> None:
        """Install an already-built instance (tests, or a provider that loads models elsewhere)."""
        with self._lock:
            self._items[key] = (obj, est_bytes)
            self.loads[key] = self.loads.get(key, 0) + 1

    def _evict_for(self, incoming: int, keep: str) -> None:
        while self._items and self.resident_bytes() + incoming > self.budget_bytes:
            victim = next(iter(self._items))       # least recently used first
            if victim == keep:
                break
            obj, _ = self._items.pop(victim)
            release = getattr(obj, "release", None)
            if callable(release):
                release()
            self.evictions += 1

    def resident_bytes(self) -> int:
        return sum(b for _, b in self._items.values())

    def release(self, key: str) -> bool:
        with self._lock:
            item = self._items.pop(key, None)
        if item and callable(getattr(item[0], "release", None)):
            item[0].release()
        return item is not None

    def status(self) -> dict:
        with self._lock:
            return {"resident": [{"key": k, "est_mb": b // MB} for k, (_, b) in self._items.items()],
                    "resident_mb": self.resident_bytes() // MB, "budget_mb": self.budget_bytes // MB,
                    "loads": dict(self.loads), "evictions": self.evictions}

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


# Rough resident sizes used for the budget (weights plus working memory); measured values are logged by status().
EST_MB = {"e5": 600, "bge": 600, "scibert": 600, "specter2": 700, "reranker": 200, "nli": 700}
