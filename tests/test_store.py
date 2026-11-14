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


# ---- versioned schema, workspaces, status (Phase 4, step 2) -----------------------------------------------------
def test_legacy_unversioned_database_is_upgraded_in_place(tmp_path):
    import sqlite3
    db = tmp_path / "legacy.sqlite"
    con = sqlite3.connect(db)
    con.executescript("""CREATE TABLE documents (doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, year INTEGER, source TEXT NOT NULL,
        source_path TEXT, n_sections INTEGER NOT NULL, n_references INTEGER NOT NULL, schema_version TEXT NOT NULL,
        added_at REAL NOT NULL, json TEXT NOT NULL);""")
    d = mk("old", "Legacy Paper", 2019)
    con.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)",
                (d.doc_id, d.title, d.year, d.source, None, 1, 0, "1.0", 1.0, d.model_dump_json()))
    con.commit(); con.close()
    with CorpusStore(db) as s:
        from researchlens.store.migrations import LATEST
        assert s.schema_version() == LATEST and s.count() == 1
        assert s.get("old").title == "Legacy Paper"
        info = s.info("old")
        assert info.status == "ready" and info.error is None and info.sha256 is None      # defaults for migrated rows
        assert s.integrity_check() == "ok"


def test_failed_migration_leaves_previous_version_intact(tmp_path, monkeypatch):
    import sqlite3
    from researchlens.store import migrations
    db = tmp_path / "x.sqlite"
    CorpusStore(db).close()
    con = sqlite3.connect(db)
    monkeypatch.setattr(migrations, "MIGRATIONS", migrations.MIGRATIONS + [(99, "CREATE TABLE t(a); THIS IS NOT SQL;")])
    with pytest.raises(sqlite3.Error):
        migrations.migrate(con)
    assert migrations.user_version(con) == migrations.LATEST and con.execute("SELECT name FROM sqlite_master WHERE name='t'").fetchone() is None


def test_workspaces_are_isolated_and_may_reuse_doc_ids(tmp_path):
    db = tmp_path / "ws.sqlite"
    a, b = CorpusStore(db, workspace="alice"), CorpusStore(db, workspace="bob")
    assert a.add_document(mk("same", "Alice's paper")) and b.add_document(mk("same", "Bob's paper"))
    assert a.get("same").title == "Alice's paper" and b.get("same").title == "Bob's paper"
    a.add_document(mk("only-a"))
    assert b.get("only-a") is None and b.count() == 1 and a.count() == 2
    assert a.delete("same") and b.get("same") is not None            # deleting in one workspace never touches the other


def test_status_lifecycle_and_search_set_only_contains_ready(tmp_path):
    s = CorpusStore(tmp_path / "st.sqlite")
    s.add_document(mk("p"), status="queued")
    assert s.documents() == [] and s.info("p").status == "queued"      # not searchable until ready
    s.set_status("p", "failed", "scanned PDF: no text layer")
    assert s.info("p").error == "scanned PDF: no text layer" and s.count("failed") == 1
    s.set_status("p", "ready")
    assert [d.doc_id for d in s.documents()] == ["p"] and s.info("p").error is None
    with pytest.raises(ValueError):
        s.set_status("p", "bogus")


def test_wal_and_busy_timeout_enabled(tmp_path):
    s = CorpusStore(tmp_path / "w.sqlite")
    assert s._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert s._db.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
