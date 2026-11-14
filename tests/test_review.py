import re

import pytest

from researchlens.analysis.gaps import find_gaps
from researchlens.analysis.review import CAVEAT, build_review, review_html, review_markdown, review_pdf
from researchlens.knowledge_graph.builder import GraphBuilder
from tests.test_gaps import P1, P2
from tests.test_knowledge_graph import A, B

DOCS = [A, B, P1, P2]


def make():
    kg = GraphBuilder().build(DOCS)
    return build_review(DOCS, kg, find_gaps(DOCS, kg), title="Test review")


def test_all_sections_present_in_order():
    r = make()
    assert [h for h, _ in r.sections] == ["Research problem", "Background", "Existing methodologies", "Paper-by-paper comparison",
                                          "Experimental findings", "Contradictory evidence", "Research limitations",
                                          "Identified research gaps", "Future research directions", "References"]


def test_every_citation_resolves_and_sources_have_quotes():
    r = make()
    md = review_markdown(r)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", md.split("**Cited passages**")[0])}
    assert cited and cited <= {s.n for s in r.sources}
    assert all(s.quote and s.title for s in r.sources)


def test_extracted_statements_carry_citations():
    md = review_markdown(make())
    for section in ("Experimental findings", "Research limitations"):
        body = md.split(f". {section}")[1].split("\n## ")[0]
        for line in [l for l in body.split("\n") if l.startswith("- ")]:
            assert re.search(r"\[\d+\]", line), line


def test_no_invented_text_quoted_sentences_exist_in_sources():
    r = make()
    md = review_markdown(r)
    body_docs = " ".join(" ".join(s.paragraphs) for d in DOCS for s in d.sections) + " ".join(d.abstract for d in DOCS)
    for q in re.findall(r"“([^”]{40,})”", md.split("**Cited passages**")[0]):
        frag = q.rstrip("…")[:60]
        assert frag in body_docs or frag in " ".join(x.quote for x in r.sources) or "TF-IDF" in q, frag


def test_caveat_and_gap_fields_in_exports():
    r = make()
    md, html = review_markdown(r), review_html(r)
    assert CAVEAT in md and "Proposed research question" in md and "Scope:" in md
    assert html.startswith("<!doctype html>") and "<script" not in html.lower()


def test_empty_sections_say_so_instead_of_filling():
    kg = GraphBuilder().build([P1])
    md = review_markdown(build_review([P1], kg, []))
    assert "No gaps were identified" in md and "No cross-paper contradictions" in md


def test_html_escapes_untrusted_pdf_text():
    from researchlens.contracts import Document, Section
    evil = Document(doc_id="E", title="<script>alert(1)</script> Paper", abstract="<img src=x onerror=alert(1)> We study retrieval of papers.",
                    source="fixture", sections=[Section(title="Intro", paragraphs=["Plain text paragraph about retrieval systems."])])
    html = review_html(build_review([evil], GraphBuilder().build([evil]), []))
    assert "<script>" not in html and "<img" not in html and "&lt;script&gt;" in html


def test_pdf_export_roundtrip(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    out = review_pdf(make(), str(tmp_path / "r.pdf"))
    doc = pymupdf.open(out)
    from researchlens.ingestion.pymupdf_ingestor import _clean
    text = _clean(" ".join(p.get_text() for p in doc))      # PDF text extraction yields fi/fl ligatures
    assert doc.page_count >= 1 and "Identified research gaps" in text and "automatically from sentences" in text
