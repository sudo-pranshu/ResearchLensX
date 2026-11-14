"""Corpus store tests (SQLite in memory / tmp; PDFs generated in-test)."""
import pytest

from researchlens.contracts import Document, Section
from researchlens.store.corpus import CorpusStore


def mk(doc_id, title="T", year=2020):
    return Document(doc_id=doc_id, title=title, year=year, abstract="a b c",
                    sections=[Section(title="Intro", paragraphs=["one two three four"], pages=[1])], source="fixture")


def test_add_dedupes_and_counts():
    s = CorpusStore()
    assert s.add_document(mk("a")) is True
    assert s.add_document(mk("a", title="changed")) is False      # same id: first write wins, nothing changes
    assert s.count() == 1 and s.get("a").title == "T"


def test_roundtrip_preserves_pages_and_fields():
    s = CorpusStore(); s.add_document(mk("a"))
    d = s.get("a")
    assert d.sections[0].pages == [1] and d.abstract == "a b c" and d.schema_version


def test_metadata_filters():
    s = CorpusStore()
    for d in (mk("a", "Dense Retrieval", 2020), mk("b", "Claim Verification", 2021), mk("c", "No Year", None)):
        s.add_document(d)
    assert {d.doc_id for d in s.documents(year_min=2021)} == {"b"}
    assert {d.doc_id for d in s.documents(year_max=2020)} == {"a"}          # unknown year never matches a range
    assert {d.doc_id for d in s.documents(title_contains="retrieval")} == {"a"}
    assert {d.doc_id for d in s.documents(doc_ids=["a", "c"])} == {"a", "c"}
    assert s.documents(doc_ids=[]) == []
    assert [i.doc_id for i in s.list_documents(title_contains="%")] == []   # LIKE wildcards are escaped


def test_delete():
    s = CorpusStore(); s.add_document(mk("a"))
    assert s.delete("a") is True and s.delete("a") is False and s.count() == 0


def test_persistence_across_connections(tmp_path):
    db = tmp_path / "sub" / "lib.sqlite"
    with CorpusStore(db) as s:
        s.add_document(mk("a"))
    with CorpusStore(db) as s:
        assert s.count() == 1


def test_incompatible_schema_is_rejected():
    s = CorpusStore(); s.add_document(mk("a"))
    s._db.execute("UPDATE documents SET schema_version='0.9'"); s._db.commit()
    with pytest.raises(ValueError, match="incompatible schema"):
        s.documents()


def test_add_pdfs_reports_added_duplicates_and_failures(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    from tests.test_ingestion import make_paper
    good = make_paper(tmp_path / "good.pdf")
    blank = tmp_path / "blank.pdf"; d = pymupdf.open(); d.new_page(); d.save(blank); d.close()
    s = CorpusStore()
    r1 = s.add_pdfs([good, blank, tmp_path / "missing.pdf"])
    assert len(r1.added) == 1 and len(r1.failed) == 2 and not r1.duplicates
    assert any("no extractable text" in e for _, e in r1.failed)
    r2 = s.add_pdfs([good])
    assert r2.added == [] and r2.duplicates == [str(good)]
    assert s.count() == 1


def test_usable_from_other_threads(tmp_path):
    """Streamlit reruns on different threads; the store must not raise sqlite3.ProgrammingError."""
    import threading
    s = CorpusStore(tmp_path / "t.sqlite"); s.add_document(mk("a"))
    out = {}
    def work():
        try:
            s.add_document(mk("b")); out["n"] = s.count(); out["ids"] = [i.doc_id for i in s.list_documents()]
        except Exception as e:                      # pragma: no cover - would be the regression
            out["err"] = repr(e)
    th = threading.Thread(target=work); th.start(); th.join()
    assert out == {"n": 2, "ids": ["a", "b"]}
