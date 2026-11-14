"""Streamlit app smoke tests via streamlit.testing (no browser). Uses a temp library and no model downloads."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from pathlib import Path

from researchlens.mocks import load_sample_documents  # noqa: E402
from researchlens.store.corpus import CorpusStore      # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture()
def app_with_library(tmp_path, monkeypatch):
    db = tmp_path / "lib.sqlite"
    with CorpusStore(db) as s:
        for d in load_sample_documents():
            s.add_document(d)
    monkeypatch.setenv("RLX_DB", str(db))
    return lambda: AppTest.from_file(APP, default_timeout=90).run()


def go(at, page):
    at.sidebar.radio[0].set_value(page).run()
    return at


def test_empty_library_shows_guidance(tmp_path, monkeypatch):
    monkeypatch.setenv("RLX_DB", str(tmp_path / "empty.sqlite"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and at.header[0].value == "Library"
    assert any("No papers yet" in i.value for i in at.info)
    go(at, "Ask")
    assert any("library is empty" in i.value for i in at.info)


def test_every_page_renders_with_library(app_with_library):
    at = app_with_library()
    assert not at.exception
    for page in ("Library", "Ask", "Search", "Verify a claim", "Compare papers", "Knowledge graph", "Evaluation"):
        go(at, page)
        assert not at.exception, (page, [e.value for e in at.exception])


def test_ask_shows_answer_references_and_no_mock_without_verification(app_with_library):
    at = go(app_with_library(), "Ask")
    at.text_input[0].set_value("How does hybrid retrieval work?").run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "rl-card" in html and "Not verified" in html and "MOCK" not in html


def test_ask_abstains_visibly_on_off_topic_question(app_with_library):
    at = go(app_with_library(), "Ask")
    at.text_input[0].set_value("What is the capital of France?").run()
    assert not at.exception and any("do not contain enough evidence" in w.value for w in at.warning)


def test_search_compare_mode(app_with_library):
    at = go(app_with_library(), "Search")
    at.text_input[0].set_value("retrieval").run()
    assert not at.exception and any("rl-card" in m.value for m in at.markdown)


def test_evaluation_page_lists_committed_experiments(app_with_library):
    at = go(app_with_library(), "Evaluation")
    assert not at.exception
    assert any("EXP-H1-006" in o for o in at.selectbox[0].options)
