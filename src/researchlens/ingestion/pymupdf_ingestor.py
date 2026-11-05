"""PyMuPDF ingestor. Owner: Gaurav. STATUS: not implemented (Day 1-2 task, issue G1).

Spec (see docs/development/GAURAV_WORKFLOW.md §G1):
  ingest(pdf_path) -> contracts.Document with source="pymupdf"
  * doc_id = sha1(file bytes)[:12] unless an arXiv id is found in the first page text
  * title = largest-font text span on page 1 (fallback: PDF metadata title, then filename)
  * sections: detect headings by font size/bold + numbering regex r"^(\\d+(\\.\\d+)*)\\.?\\s+[A-Z]"
  * paragraphs: merge text blocks within a section; de-hyphenate line breaks; drop headers/footers
    that repeat on >50% of pages and lone page numbers
  * references: everything after a heading matching /references|bibliography/i, one Reference(raw=...) per entry
"""
from researchlens.contracts import Document


class PyMuPDFIngestor:
    name = "pymupdf"

    def ingest(self, pdf_path: str) -> Document:
        raise NotImplementedError("G1: implement per docstring spec")
