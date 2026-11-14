"""Original-file storage, upload limits and identity (Phase 4, step 3). Needs PyMuPDF for the generated PDFs."""
import hashlib

import pytest

pymupdf = pytest.importorskip("pymupdf")
from tests.test_ingestion import make_paper                                  # noqa: E402

from researchlens.settings import AppConfig                                  # noqa: E402
from researchlens.storage.documents import LocalDocumentStorage              # noqa: E402
from researchlens.workspace import Workspace                                  # noqa: E402


def test_storage_roundtrip_atomic_and_scoped(tmp_path):
    s = LocalDocumentStorage(tmp_path)
    f = s.put("alice", "arxiv:1810.04805", b"%PDF-1.4 data")
    assert f.sha256 == hashlib.sha256(b"%PDF-1.4 data").hexdigest() and f.bytes == 13
    assert s.exists("alice", "arxiv:1810.04805") and not s.exists("bob", "arxiv:1810.04805")
    with s.open_read("alice", "arxiv:1810.04805") as fh:
        assert fh.read() == b"%PDF-1.4 data"
    s.put("alice", "arxiv:1810.04805", b"%PDF-new")                           # overwrite is atomic and leaves no temp files
    assert [p.name for p in s.path_for("alice", "arxiv:1810.04805").parent.iterdir()] == ["original.pdf"]
    assert s.delete("alice", "arxiv:1810.04805") and not s.exists("alice", "arxiv:1810.04805") and not s.delete("alice", "arxiv:1810.04805")
    assert s.presigned_url("alice", "x") is None                              # the local adapter cannot sign URLs


@pytest.mark.parametrize("evil", ["../../etc/passwd", "/abs/path", "a/../../b", "..", "x\x00y"])
def test_storage_ids_can_never_escape_the_root(tmp_path, evil):
    s = LocalDocumentStorage(tmp_path / "root")
    s.put("ws", evil, b"x")
    for p in (tmp_path).rglob("*"):
        assert p == tmp_path / "root" or (tmp_path / "root") in p.parents or p == tmp_path / "root"
    assert not (tmp_path / "etc").exists()


def test_distinct_ids_never_collide_after_sanitising(tmp_path):
    s = LocalDocumentStorage(tmp_path)
    s.put("w", "a/b", b"1"); s.put("w", "a_b", b"2")
    with s.open_read("w", "a/b") as f1, s.open_read("w", "a_b") as f2:
        assert (f1.read(), f2.read()) == (b"1", b"2")


@pytest.fixture()
def ws(tmp_path):
    cfg = AppConfig.from_env({"RLX_MAX_UPLOAD_MB": "1", "RLX_MAX_PAGES": "2"}, root=tmp_path)
    return Workspace.from_config(cfg), cfg, tmp_path


def test_upload_stores_original_with_hash_and_pages(ws, tmp_path):
    w, cfg, _ = ws
    data = make_paper(tmp_path / "p.pdf").read_bytes()
    big = Workspace(tmp_path / "x.sqlite")                                   # default limits (50 MB, 400 pages)
    rep = big.upload([("paper.pdf", data)])
    assert len(rep.added) == 1 and not rep.failed
    info = big.store.info(rep.added[0])
    assert info.sha256 == hashlib.sha256(data).hexdigest() and info.bytes == len(data) and info.pages == 3 and info.pdf_path
    with big.storage.open_read("local", rep.added[0]) as f:
        assert f.read() == data


def test_duplicate_is_reported_and_original_is_not_rewritten_or_orphaned(tmp_path):
    w = Workspace(tmp_path / "d.sqlite")
    data = make_paper(tmp_path / "p.pdf").read_bytes()
    first = w.upload([("a.pdf", data)])
    p = w.storage.path_for("local", first.added[0]); m1 = p.stat().st_mtime_ns
    second = w.upload([("copy.pdf", data)])
    assert second.added == [] and second.duplicates == ["copy.pdf"] and p.stat().st_mtime_ns == m1
    assert sorted(x.name for x in p.parent.iterdir()) == ["original.pdf"]


def test_limits_and_validation_report_reasons(ws, tmp_path):
    w, cfg, _ = ws                                                           # 1 MB, 2 pages
    ok = make_paper(tmp_path / "p.pdf").read_bytes()                         # 3 pages -> over the page limit
    rep = w.upload([("big.pdf", b"%PDF-" + b"0" * 2_000_000), ("note.txt", b"hello"), ("three.pdf", ok)])
    reasons = dict(rep.failed)
    assert "limit is 1 MB" in reasons["big.pdf"]
    assert "not a PDF" in reasons["note.txt"]
    assert "3 pages exceeds the limit of 2" in reasons["three.pdf"]
    assert rep.added == [] and w.store.count() == 0


def test_remove_deletes_row_and_original(tmp_path):
    w = Workspace(tmp_path / "r.sqlite")
    doc_id = w.upload([("p.pdf", make_paper(tmp_path / "p.pdf").read_bytes())]).added[0]
    path = w.storage.path_for("local", doc_id)
    assert path.exists() and w.remove(doc_id)
    assert not path.exists() and not path.parent.exists() and w.store.count() == 0
