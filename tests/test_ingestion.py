"""PDF ingestion tests. PDFs are generated in-test with PyMuPDF (no external files, offline, CPU)."""
import pytest

pymupdf = pytest.importorskip("pymupdf")

from researchlens.contracts import SCHEMA_VERSION
from researchlens.ingestion.pymupdf_ingestor import PyMuPDFIngestor
from researchlens.processing.chunking import chunk_document

BODY = ("We study retrieval of scientific evidence. Our approach combines lexical and dense signals "
        "and is evaluated on a public benchmark with careful statistical testing. ")
W, H = 595, 842


def _page(doc, header=None, footer=None):
    page = doc.new_page(width=W, height=H)
    if header:
        page.insert_text((72, 30), header, fontsize=8)
    if footer:
        page.insert_text((290, 820), footer, fontsize=9)
    return page


def make_paper(path, with_arxiv=False, hyphen=True):
    doc = pymupdf.open()
    p1 = _page(doc, "Journal of Tests 2026", "1")
    p1.insert_text((72, 100), "Hybrid Evidence Retrieval for Papers", fontsize=20, fontname="hebo")
    p1.insert_text((72, 130), "A. Author and B. Writer", fontsize=11)
    if with_arxiv:
        p1.insert_text((20, 400), "arXiv:2105.12345v2 [cs.CL] 25 May 2021", fontsize=8)
    p1.insert_text((72, 170), "Abstract", fontsize=12, fontname="hebo")
    p1.insert_textbox(pymupdf.Rect(72, 180, 520, 260), "We present a method for finding evidence in papers. "
                      "It works well on benchmark data and is simple to reproduce.", fontsize=10)
    p1.insert_text((72, 290), "1 Introduction", fontsize=12, fontname="hebo")
    intro = BODY + ("The inter-\nnational community needs reliable tools. " if hyphen else "") + BODY
    p1.insert_textbox(pymupdf.Rect(72, 300, 520, 420), intro, fontsize=10)
    # paragraph that continues onto page 2
    p1.insert_textbox(pymupdf.Rect(72, 700, 520, 790), "A second paragraph begins at the bottom of this page and "
                      "carries on over the page boundary with additional supporting detail about", fontsize=10)
    p2 = _page(doc, "Journal of Tests 2026", "2")
    p2.insert_textbox(pymupdf.Rect(72, 60, 520, 140), "the experiments that were run in the study. " + BODY, fontsize=10)
    p2.insert_text((72, 170), "2 Method", fontsize=12, fontname="hebo")
    p2.insert_textbox(pymupdf.Rect(72, 180, 520, 300), BODY * 2, fontsize=10)
    p2.insert_text((72, 330), "2.1 Fusion", fontsize=11, fontname="hebo")
    p2.insert_textbox(pymupdf.Rect(72, 340, 520, 440), "We fuse ranked lists with reciprocal rank fusion. " + BODY, fontsize=10)
    p3 = _page(doc, "Journal of Tests 2026", "3")
    p3.insert_text((72, 100), "3 Conclusion", fontsize=12, fontname="hebo")
    p3.insert_textbox(pymupdf.Rect(72, 110, 520, 220), "Hybrid retrieval is not always better. " + BODY, fontsize=10)
    p3.insert_text((72, 260), "References", fontsize=12, fontname="hebo")
    p3.insert_textbox(pymupdf.Rect(72, 270, 520, 360), "[1] Robertson, S. The probabilistic relevance framework. 2009.\n"
                      "[2] Cormack, G. Reciprocal rank fusion outperforms Condorcet. 2009.", fontsize=9)
    doc.set_metadata({"title": "", "author": "A. Author, B. Writer"})
    doc.save(path)
    doc.close()
    return path


@pytest.fixture(scope="module")
def parsed(tmp_path_factory):
    p = make_paper(tmp_path_factory.mktemp("pdf") / "paper.pdf")
    return PyMuPDFIngestor().ingest(str(p))


def test_title_abstract_and_source(parsed):
    assert parsed.title == "Hybrid Evidence Retrieval for Papers"
    assert parsed.source == "pymupdf" and parsed.schema_version == SCHEMA_VERSION
    assert parsed.abstract.startswith("We present a method")
    assert len(parsed.doc_id) == 12


def test_sections_levels_and_order(parsed):
    titles = [(s.title, s.level) for s in parsed.sections]
    assert titles == [("Introduction", 1), ("Method", 1), ("Fusion", 2), ("Conclusion", 1)]


def test_headers_footers_and_page_numbers_removed(parsed):
    text = " ".join(p for s in parsed.sections for p in s.paragraphs) + parsed.abstract
    assert "Journal of Tests" not in text
    assert not any(p.strip() in {"1", "2", "3"} for s in parsed.sections for p in s.paragraphs)


def test_dehyphenation(parsed):
    intro = " ".join(parsed.sections[0].paragraphs)
    assert "international" in intro and "inter-" not in intro


def test_cross_page_paragraph_is_merged_and_keeps_start_page(parsed):
    intro = parsed.sections[0]
    merged = [(p, pg) for p, pg in zip(intro.paragraphs, intro.pages) if "second paragraph" in p]
    assert len(merged) == 1
    text, page = merged[0]
    assert page == 1 and "the experiments that were run" in text


def test_pages_align_with_paragraphs(parsed):
    for s in parsed.sections:
        assert len(s.pages) == len(s.paragraphs)
    assert parsed.sections[-1].pages[0] == 3        # Conclusion is on page 3


def test_references_split(parsed):
    assert len(parsed.references) == 2
    assert parsed.references[0].raw.startswith("[1] Robertson") and parsed.references[0].year == 2009


def test_authors_only_from_metadata(parsed):
    assert parsed.authors == ["A. Author", "B. Writer"]


def test_chunks_carry_page_numbers(parsed):
    chunks = chunk_document(parsed)
    assert all(c.page is not None for c in chunks if c.section != "Abstract")
    assert {c.page for c in chunks if c.section == "Conclusion"} == {3}
    assert next(c for c in chunks if c.section == "Abstract").page is None   # abstract page not asserted


def test_arxiv_id_and_year(tmp_path):
    p = make_paper(tmp_path / "ax.pdf", with_arxiv=True)
    d = PyMuPDFIngestor().ingest(str(p))
    assert d.doc_id == "arxiv:2105.12345" and d.year == 2021


def test_doc_id_is_stable_per_file_and_distinct_across_files(tmp_path):
    a = make_paper(tmp_path / "a.pdf"); b = make_paper(tmp_path / "b.pdf", hyphen=False)
    ing = PyMuPDFIngestor()
    assert ing.ingest(str(a)).doc_id == ing.ingest(str(a)).doc_id
    assert ing.ingest(str(a)).doc_id != ing.ingest(str(b)).doc_id


def test_image_only_pdf_raises(tmp_path):
    doc = pymupdf.open(); doc.new_page(); p = tmp_path / "blank.pdf"; doc.save(p); doc.close()
    with pytest.raises(ValueError, match="no extractable text"):
        PyMuPDFIngestor().ingest(str(p))
