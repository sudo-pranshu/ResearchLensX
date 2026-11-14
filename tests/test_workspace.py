"""Workspace service layer (offline: BM25, extractive generator, mock/real verifier fallback, generated PDFs)."""
import pytest

pymupdf = pytest.importorskip("pymupdf")
from tests.test_ingestion import make_paper                      # noqa: E402

from researchlens.workspace import Settings, Workspace          # noqa: E402


@pytest.fixture()
def ws(tmp_path):
    w = Workspace()
    data = make_paper(tmp_path / "p.pdf").read_bytes()
    w.upload([("paper.pdf", data)])
    return w


def test_upload_reports_added_duplicate_failed(tmp_path):
    w = Workspace()
    good = make_paper(tmp_path / "a.pdf").read_bytes()
    r = w.upload([("a.pdf", good), ("notes.txt", b"not a pdf")])
    assert len(r.added) == 1 and len(r.failed) == 1 and r.failed[0][0] == "notes.txt"
    r2 = w.upload([("again.pdf", good)])
    assert r2.added == [] and r2.duplicates == ["again.pdf"]
    assert len(w.library()) == 1


def test_empty_library_error():
    with pytest.raises(ValueError, match="library is empty"):
        Workspace().search("x", Settings())


def test_search_excludes_references_and_returns_pages(ws):
    passages, info = ws.search("reciprocal rank fusion Robertson", Settings(k=5))
    assert passages and all(p.chunk.section.lower() != "references" for p in passages)
    assert all(p.chunk.page is not None for p in passages) and info["latency_ms"] >= 0


def test_ask_returns_traceable_references_and_status_unchecked(ws):
    res, _ = ws.ask("how are ranked lists fused", Settings(verify=False))
    assert not res.abstained and res.references and all(s.status == "unchecked" for s in res.sentences)


def test_verifier_falls_back_to_visible_mock_when_model_unavailable(ws, monkeypatch):
    import researchlens.verification.nli as nli
    monkeypatch.setattr(nli, "HFNLIScorer", lambda *a, **k: (_ for _ in ()).throw(OSError("no model")))
    res, _ = ws.ask("how are ranked lists fused", Settings(verify=True))
    assert res.is_mock is True
    st = {c.name: c for c in ws.status(Settings(verify=True))}
    assert st["Verifier"].state == "mock" and "MOCK" in st["Verifier"].detail


def test_rerank_failure_falls_back_to_bm25(ws, monkeypatch):
    import researchlens.retrieval.rerank as rr
    monkeypatch.setattr(rr.CrossEncoderScorer, "scores", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    passages, info = ws.search("fusion", Settings(rerank=True))
    assert passages and "fell back" in info["note"]


def test_unavailable_dense_retriever_falls_back_visibly(ws, monkeypatch):
    import researchlens.retrieval.registry as reg
    monkeypatch.setattr(reg, "build_systems", lambda *a, **k: (_ for _ in ()).throw(OSError("no model")))
    passages, info = ws.search("fusion", Settings(retriever="e5"))
    assert passages and "fell back to BM25" in info["note"]
    st = {c.name: c for c in ws.status(Settings(retriever="e5"))}
    assert st["Retrieval"].state == "fallback"


def test_failed_rerank_is_reported_as_fallback_in_status(ws, monkeypatch):
    import researchlens.retrieval.rerank as rr
    monkeypatch.setattr(rr.CrossEncoderScorer, "scores", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    ws.search("fusion", Settings(rerank=True))
    assert {c.name: c for c in ws.status(Settings(rerank=True))}["Reranker"].state == "fallback"


def test_default_verifier_is_the_validated_passage_level_configuration(monkeypatch):
    import researchlens.verification.nli as nli

    class Fake:
        name = "fake"
        def __init__(self, *a, **k): pass
        def __call__(self, p, h):
            import numpy as np
            return np.array([[0.9, 0.05, 0.05]] * len(p))
    monkeypatch.setattr(nli, "HFNLIScorer", Fake)
    w = Workspace()
    v = w.verifier()
    assert v.granularity == "passage" and v.temperature == 2.24 and not v.combine
    c = w.verifier(combine=True)
    assert c.granularity == "sentence" and c.combine and c.temperature == 1.0       # experimental mode is uncalibrated


def test_remove_document(ws):
    doc_id = ws.library()[0].doc_id
    assert ws.remove(doc_id) and ws.library() == []
