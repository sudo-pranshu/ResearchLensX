"""Job runner (Phase 4, step 7): state machine, hostile PDFs, limits, cancel/retry, idempotency, restart recovery, concurrency."""
import threading
import time

import pytest

pymupdf = pytest.importorskip("pymupdf")
from tests.test_ingestion import make_paper                                   # noqa: E402

from researchlens import parse_worker                                          # noqa: E402
from researchlens.jobs import LocalJobRunner, pending_key                      # noqa: E402
from researchlens.storage.documents import LocalDocumentStorage                # noqa: E402
from researchlens.store.corpus import CorpusStore                              # noqa: E402
from researchlens.workspace import Settings, Workspace                         # noqa: E402


@pytest.fixture()
def pdf(tmp_path):
    return make_paper(tmp_path / "p.pdf").read_bytes()


def runner(tmp_path, rebuild=lambda: None, autostart=True, **kw):
    store = CorpusStore(tmp_path / "j.sqlite")
    storage = LocalDocumentStorage(tmp_path / "files")
    return LocalJobRunner(store, storage, rebuild, autostart=autostart, **kw), store, storage


def test_lifecycle_states_and_a_document_is_ready_only_after_the_index_rebuild(tmp_path, pdf):
    gate, seen_in_rebuild = threading.Event(), {}

    def rebuild():
        seen_in_rebuild["docs"] = [(d.doc_id, d.status) for d in store.list_documents()]
        gate.wait(5)
    r, store, storage = runner(tmp_path, rebuild)
    job = r.submit_ingest("paper.pdf", pdf)
    assert job.state == "queued" and storage.exists("local", job.pending_key)         # the original is durable before processing
    for _ in range(100):                                                                # wait until the rebuild is running
        if "docs" in seen_in_rebuild:
            break
        time.sleep(0.1)
    assert [s for _, s in seen_in_rebuild["docs"]] == ["indexing"]                      # not searchable yet
    assert store.documents() == [] and r.get(job.id).state == "indexing"
    gate.set(); assert r.wait_idle(30)
    done = r.get(job.id)
    assert done.state == "ready" and done.outcome == "added" and done.progress == 1.0 and done.doc_id
    assert store.info(done.doc_id).status == "ready" and len(store.documents()) == 1
    assert not storage.exists("local", job.pending_key) and storage.exists("local", done.doc_id)     # pending original moved to its doc id
    r.stop()


@pytest.mark.parametrize("name,data,reason", [
    ("garbage.pdf", b"%PDF-1.4\nthis is not a real pdf body at all", None),                      # truncated/corrupt, parser decides
    ("note.txt", b"hello world", "not a PDF"),
])
def test_hostile_or_invalid_inputs_fail_with_a_reason_and_leave_the_library_untouched(tmp_path, name, data, reason):
    r, store, _ = runner(tmp_path)
    job = r.submit_ingest(name, data)
    r.wait_idle(60)
    j = r.get(job.id)
    assert j.state == "failed" and j.error and (reason is None or reason in j.error)
    assert store.count() == 0
    r.stop()


def test_blank_scanned_pdf_and_page_limit_fail_clearly(tmp_path, pdf):
    blank = pymupdf.open(); blank.new_page(); blank.save(tmp_path / "blank.pdf"); blank.close()
    r, store, _ = runner(tmp_path, max_pages=2)
    a = r.submit_ingest("scan.pdf", (tmp_path / "blank.pdf").read_bytes())
    b = r.submit_ingest("long.pdf", pdf)                                                    # 3 pages > limit 2
    r.wait_idle(90)
    assert "no extractable text" in r.get(a.id).error and "OCR is not supported" in r.get(a.id).error
    assert "3 pages exceeds the limit of 2" in r.get(b.id).error and "long.pdf" in r.get(b.id).error
    r.stop()


def test_time_limit_kills_the_parser_and_retry_succeeds_with_a_sane_limit(tmp_path, pdf):
    r, store, _ = runner(tmp_path, parse_timeout_s=0.01)
    job = r.submit_ingest("p.pdf", pdf)
    r.wait_idle(60)
    j = r.get(job.id)
    assert j.state == "failed" and "time limit" in j.error
    r.parse_timeout_s = 60
    assert r.retry(job.id).state == "queued"
    r.wait_idle(90)
    assert r.get(job.id).state == "ready" and r.get(job.id).attempts == 2 and store.count() == 1
    r.stop()


def test_memory_limit_kills_the_parser(tmp_path, pdf, monkeypatch):
    monkeypatch.setattr(parse_worker, "rss_mb", lambda pid: 99999.0)
    r, _, _ = runner(tmp_path, max_rss_mb=100)
    job = r.submit_ingest("p.pdf", pdf)
    r.wait_idle(60)
    assert "more than 100 MB of memory" in r.get(job.id).error
    r.stop()


def test_cancel_queued_job_and_idempotent_submission(tmp_path, pdf):
    r, store, storage = runner(tmp_path, autostart=False)                                 # worker not started: jobs stay queued
    a, b = r.submit_ingest("p.pdf", pdf), r.submit_ingest("same-bytes.pdf", pdf)
    assert a.id == b.id and len(r.list()) == 1                                             # identical bytes in flight: one job
    assert r.cancel(a.id) and r.get(a.id).state == "cancelled" and not r.cancel(a.id)
    r.start(); r.wait_idle(10)
    assert store.count() == 0
    r.stop()


def test_cancel_a_job_that_is_parsing(tmp_path, pdf, monkeypatch):
    started = threading.Event()

    def slow_parse(path, max_pages, timeout, rss, should_cancel=lambda: False, **kw):
        started.set()
        while not should_cancel():
            time.sleep(0.05)
        return parse_worker.ParseResult(False, error="cancelled")
    monkeypatch.setattr("researchlens.jobs.parse_pdf", slow_parse)
    r, store, _ = runner(tmp_path)
    job = r.submit_ingest("p.pdf", pdf)
    assert started.wait(10) and r.cancel(job.id)
    r.wait_idle(10)
    assert r.get(job.id).state == "cancelled" and store.count() == 0
    r.stop()


def test_restart_recovery_requeues_parsing_jobs_and_rebuilds_indexing_documents(tmp_path, pdf):
    r1, store, storage = runner(tmp_path, autostart=False)
    job = r1.submit_ingest("p.pdf", pdf)
    store.update_job(job.id, state="parsing", stage="parsing the PDF")                    # the process died mid-parse
    d = make_doc_for_indexing(store)                                                       # and one document was waiting for the index
    r2 = LocalJobRunner(store, storage, lambda: None)
    assert r2.recovered == 1
    r2.wait_idle(90)
    assert r2.get(job.id).state == "ready" and store.info(d).status == "ready" and store.count("ready") == 2
    r2.stop()


def make_doc_for_indexing(store):
    from tests.test_store import mk
    store.add_document(mk("left-in-indexing"), status="indexing")
    return "left-in-indexing"


def test_failed_rebuild_marks_documents_failed_and_the_worker_survives(tmp_path, pdf):
    calls = {"n": 0}

    def bad_rebuild():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("index exploded")
    r, store, _ = runner(tmp_path, bad_rebuild)
    job = r.submit_ingest("p.pdf", pdf)
    r.wait_idle(90)
    j = r.get(job.id)
    assert j.state == "failed" and "index exploded" in j.error and store.info(j.doc_id).status == "failed"
    assert r.submit_ingest("n.txt", b"x").state == "failed"                                # worker still accepts and answers
    r.stop()


def test_searches_during_an_upload_never_see_a_half_indexed_document_and_nothing_locks(tmp_path, pdf):
    from tests.test_incremental_equals_rebuild import make_doc
    ws = Workspace(tmp_path / "ws.sqlite")
    ws.add_documents([make_doc("A")])
    job = ws.submit_uploads([("p.pdf", pdf)])[0]
    errors, stop = [], threading.Event()

    def searcher():
        while not stop.is_set():
            try:
                ps, _ = ws.search("dense retrieval recall", Settings(k=5))
                ready_after = {i.doc_id for i in ws.store.list_documents(status="ready")}
                assert {p.chunk.doc_id for p in ps} <= ready_after, "a document was searchable before it was ready"
            except Exception as e:
                errors.append(repr(e)); stop.set()
    ts = [threading.Thread(target=searcher) for _ in range(4)]
    [t.start() for t in ts]
    ws.jobs.wait_idle(120)
    stop.set(); [t.join(10) for t in ts]
    assert not errors, errors
    assert ws.jobs.get(job.id).state == "ready" and ws.store.count("ready") == 2
