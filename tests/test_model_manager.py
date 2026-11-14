import threading
import time

import pytest

from researchlens.models import MB, ModelManager


def test_single_load_under_concurrency_and_same_instance():
    m = ModelManager()
    calls, out = [], []
    gate = threading.Barrier(8)

    def loader():
        calls.append(1); time.sleep(0.05); return object()

    def worker():
        gate.wait(); out.append(m.get("e5", loader, 100 * MB))
    ts = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(calls) == 1 and len({id(o) for o in out}) == 1 and m.loads == {"e5": 1}


def test_byte_budget_evicts_least_recently_used_and_releases():
    m = ModelManager(budget_bytes=250 * MB)
    freed = []

    class Model:
        def __init__(self, n): self.n = n
        def release(self): freed.append(self.n)
    a = m.get("a", lambda: Model("a"), 100 * MB)
    m.get("b", lambda: Model("b"), 100 * MB)
    m.get("a", lambda: Model("a2"), 100 * MB)                  # touch a: b is now least recently used
    m.get("c", lambda: Model("c"), 100 * MB)                   # 300 MB > 250 MB budget
    assert freed == ["b"] and [x["key"] for x in m.status()["resident"]] == ["a", "c"] and m.evictions == 1
    assert m.get("a", lambda: Model("never"), 100 * MB) is a   # still the original instance


def test_failed_load_is_not_cached_and_can_retry():
    m = ModelManager()
    with pytest.raises(OSError):
        m.get("nli", lambda: (_ for _ in ()).throw(OSError("no weights")), 10 * MB)
    assert "nli" not in [x["key"] for x in m.status()["resident"]]
    assert m.get("nli", lambda: "ok", 10 * MB) == "ok"


def test_all_inference_runs_on_one_thread_and_run_is_reentrant():
    m = ModelManager()
    ids = []

    def work():
        ids.append(threading.get_ident())
        return m.run(lambda: threading.get_ident())             # nested call must not deadlock
    results = []
    ts = [threading.Thread(target=lambda: results.append(m.run(work))) for _ in range(6)]
    [t.start() for t in ts]; [t.join(timeout=5) for t in ts]
    assert len(results) == 6 and len(set(ids)) == 1 and set(results) == set(ids)
    assert threading.get_ident() not in ids                      # never the caller's thread


def test_exceptions_propagate_from_run():
    m = ModelManager()
    with pytest.raises(ZeroDivisionError):
        m.run(lambda: 1 / 0)
