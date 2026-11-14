"""PyMuPDF ingestor: text-based PDF -> contracts.Document (source="pymupdf").

Heuristics (font statistics, numbering regex, repeated-margin detection); no OCR, no ML.

  * doc_id    arXiv id if one is printed on page 1, else sha1(file bytes)[:12]
  * title     largest-font text on page 1; fallback PDF metadata title, then file stem
  * authors   PDF metadata only. Page-1 author lines are not guessed; empty list otherwise
  * year      derived from the arXiv id (YYMM) only; None otherwise (creation dates are not publication years)
  * sections  headings = short blocks that are numbered ("2.1 Method"), a well-known heading word,
              or noticeably larger than body text; level from numbering depth
  * paragraphs  blocks merged inside a section; de-hyphenated; margin headers/footers that repeat on
              >50% of pages (>=3 pages) and lone page numbers dropped; a paragraph keeps the page
              on which it starts (Section.pages)
  * references  everything after a "References"/"Bibliography" heading, split on [n] / "n." markers

Known limitations: two-column PDFs rely on PyMuPDF block order (usually correct for LaTeX output);
tables and equations become noisy text; scanned PDFs raise ValueError (no OCR).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from researchlens.contracts import Document, Reference, Section

NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+[A-Z]")
LETTERED = re.compile(r"^[A-Z](?:\.\d+)*\.?\s+[A-Z]")      # appendix headings: "A Details", "B.1 Setup"
KNOWN_HEADINGS = {
    "abstract", "introduction", "background", "related work", "preliminaries", "methods", "method",
    "methodology", "approach", "model", "experiments", "experimental setup", "results", "discussion",
    "analysis", "conclusion", "conclusions", "limitations", "future work", "acknowledgments",
    "acknowledgements", "references", "bibliography", "appendix",
}
REF_HEADING = re.compile(r"^(?:\d+\.?\s+)?(references|bibliography)$", re.I)
ARXIV = re.compile(r"arXiv:(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
PAGE_NO = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?$", re.I)
SENT_END = re.compile(r"[.!?:;)\]\"”]$")
MARGIN = 0.08


@dataclass
class _Block:
    text: str
    size: float
    bold: bool
    page: int           # 1-based
    n_lines: int
    y0: float
    y1: float
    height: float       # page height
    line_x0: tuple = ()  # left edge of each line (hanging-indent detection in references)


_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})


def _clean(text: str) -> str:
    text = text.translate(_LIGATURES)
    text = re.sub(r"(\w)-\s*\n\s*([a-z])", r"\1\2", text)     # de-hyphenate line breaks
    return re.sub(r"\s+", " ", text).strip()


def _extract_blocks(pdf) -> tuple[list[_Block], str]:
    """Horizontal text blocks, plus the text of rotated lines (e.g. the arXiv sidebar stamp)."""
    blocks: list[_Block] = []
    rotated: list[str] = []
    for pno, page in enumerate(pdf, start=1):
        h = float(page.rect.height)
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            lines, x0s, sizes, bold_chars, chars = [], [], Counter(), 0, 0
            for ln in b["lines"]:
                txt = "".join(s["text"] for s in ln["spans"])
                if not txt.strip():
                    continue
                if abs(ln["dir"][0]) < 0.99:           # rotated (vertical) text is never body text
                    rotated.append(txt)
                    continue
                lines.append(txt)
                x0s.append(float(ln["bbox"][0]))
                for s in ln["spans"]:
                    n = len(s["text"].strip())
                    if n:
                        sizes[round(s["size"], 1)] += n
                        chars += n
                        if (s["flags"] & 16) or "bold" in s["font"].lower():
                            bold_chars += n
            if not lines:
                continue
            y0, _, _, y1 = b["bbox"]
            blocks.append(_Block("\n".join(lines), sizes.most_common(1)[0][0], bold_chars > 0.6 * chars,
                                 pno, len(lines), y0, y1, h, tuple(x0s)))
    return blocks, " ".join(rotated)


def _drop_margins(blocks: list[_Block], n_pages: int) -> list[_Block]:
    def in_margin(b: _Block) -> bool:
        return b.y1 < MARGIN * b.height or b.y0 > (1 - MARGIN) * b.height

    def key(b: _Block) -> str:
        return re.sub(r"\d+", "#", _clean(b.text).lower())

    pages_with: dict[str, set[int]] = defaultdict(set)
    for b in blocks:
        if in_margin(b):
            pages_with[key(b)].add(b.page)
    kept = []
    for b in blocks:
        if in_margin(b):
            if PAGE_NO.match(_clean(b.text)):
                continue
            if n_pages >= 3 and len(pages_with[key(b)]) / n_pages > 0.5:
                continue
        kept.append(b)
    return kept


def _is_heading(b: _Block, body: float) -> bool:
    t = _clean(b.text)
    if not t or len(t) > 100 or b.n_lines > 2:
        return False
    plain = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", t).strip().rstrip(".:").lower()
    if plain in KNOWN_HEADINGS and (b.bold or b.size > body * 1.02 or NUMBERED.match(t) or len(t) < 25):
        return True
    if (NUMBERED.match(t) or LETTERED.match(t)) and (b.bold or b.size > body * 1.05) and not t.endswith("."):
        return True
    return b.size > body * 1.15 and not t.endswith(".") and len(t.split()) <= 12


def _level(t: str) -> int:
    m = NUMBERED.match(t)
    if m:
        return m.group(1).count(".") + 1
    return t.split()[0].count(".") + 1 if LETTERED.match(t) and "." in t.split()[0].rstrip(".") else 1


def _split_references(blocks: list[_Block]) -> list[Reference]:
    marker = re.compile(r"^\s*(?:\[\d+\]|\d{1,3}\.)\s+")
    lines = [ln for b in blocks for ln in b.text.split("\n") if ln.strip()]
    numbered = any(marker.match(ln) for ln in lines)
    entries: list[str] = []
    if numbered:
        for ln in lines:
            if marker.match(ln) or not entries:
                entries.append(_clean(ln))
            else:
                entries[-1] = _clean(entries[-1] + "\n" + ln)
    else:
        # Unnumbered styles use a hanging indent: an entry's first line sits at the column's left edge,
        # continuation lines are indented. PyMuPDF splits entries into several blocks, so scan lines in order.
        flat = [(x0, ln) for b in blocks for ln, x0 in zip([l for l in b.text.split("\n") if l.strip()], b.line_x0)]
        prev, left = None, 0.0
        for x0, ln in flat:
            if prev is None or abs(x0 - prev) > 100:        # first line, or column change
                left, new = x0, True
            else:
                new = x0 <= left + 2
                left = min(left, x0)
            prev = x0
            if new or not entries:
                entries.append(_clean(ln))
            else:
                entries[-1] = _clean(entries[-1] + "\n" + ln)
    refs = []
    for e in entries:
        m = re.search(r"\b(19|20)\d{2}[a-z]?\b", e)
        refs.append(Reference(raw=e, year=int(m.group(0)[:4]) if m else None))
    return refs


class PyMuPDFIngestor:
    name = "pymupdf"

    def ingest(self, pdf_path: str, max_pages: int | None = None) -> Document:
        import pymupdf

        path = Path(pdf_path)
        raw = path.read_bytes()
        pdf = pymupdf.open(stream=raw, filetype="pdf")
        try:
            n_pages = pdf.page_count
            if max_pages is not None and n_pages > max_pages:      # checked before any text extraction
                raise ValueError(f"{path.name}: {n_pages} pages exceeds the limit of {max_pages}")
            blocks, rotated_text = _extract_blocks(pdf)
            meta = pdf.metadata or {}
        finally:
            pdf.close()
        if not blocks or sum(len(b.text) for b in blocks) < 200:
            raise ValueError(f"{path.name}: no extractable text (scanned PDF? OCR is not supported)")

        first_page_text = " ".join(b.text for b in blocks if b.page == 1) + " " + rotated_text
        m = ARXIV.search(first_page_text)
        doc_id = f"arxiv:{m.group(1)}" if m else hashlib.sha1(raw).hexdigest()[:12]
        year = 2000 + int(m.group(1)[:2]) if m and 7 <= int(m.group(1)[:2]) <= 99 else None

        blocks = _drop_margins(blocks, n_pages)
        sizes = Counter()
        for b in blocks:
            sizes[b.size] += len(b.text)
        body = sizes.most_common(1)[0][0]

        title = self._title(blocks, meta, path)
        authors = [a.strip() for a in re.split(r";|,| and ", meta.get("author") or "") if a.strip()]

        sections: list[tuple[str, int, list[str], list[int | None]]] = []   # title, level, paras, pages
        ref_blocks: list[_Block] = []
        in_refs = False
        cur: tuple[str, int, list[str], list[int | None]] | None = None
        buf: list[str] | None = None        # [text, page, last_block_text]
        title_done = False

        def flush():
            nonlocal buf
            if buf is not None and cur is not None and len(buf[0].split()) >= 3:
                cur[2].append(buf[0]); cur[3].append(buf[1])
            buf = None

        for b in blocks:
            t = _clean(b.text)
            if not t:
                continue
            if not title_done and b.page == 1 and t in _clean(title):
                continue                           # the title block itself
            if _is_heading(b, body):
                flush()
                title_done = True
                if REF_HEADING.match(t):
                    in_refs = True
                    continue
                in_refs = False
                cur = (re.sub(r"^(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.?\s+", "", t).rstrip(":"), _level(t), [], [])
                sections.append(cur)
                continue
            if in_refs:
                ref_blocks.append(b)
                continue
            if cur is None:                        # front matter (authors, affiliations): not evidence
                continue
            if len(t.split()) < 3:
                continue
            if buf is not None and t[0].islower() and not SENT_END.search(buf[0]):
                buf[0] += " " + t                  # continuation across column/page break
            else:
                flush()
                buf = [t, b.page]
        flush()

        abstract, abstract_page = "", None
        body_sections = []
        for title_, level, paras, pages in sections:
            if title_.lower() == "abstract" and not abstract:
                abstract = " ".join(paras)
                abstract_page = pages[0] if pages else None
                continue
            if paras:
                body_sections.append(Section(title=title_, paragraphs=paras, level=level, pages=pages))
        if not body_sections and not abstract:
            raise ValueError(f"{path.name}: no sections detected")

        return Document(doc_id=doc_id, title=title, authors=authors, year=year, abstract=abstract, abstract_page=abstract_page, n_pages=n_pages,
                        sections=body_sections, references=_split_references(ref_blocks),
                        source="pymupdf", source_path=str(path))

    @staticmethod
    def _title(blocks: list[_Block], meta: dict, path: Path) -> str:
        p1 = [b for b in blocks if b.page == 1 and b.y0 < 0.45 * b.height]
        if p1:
            top = max(b.size for b in p1)
            cand = [b for b in p1 if b.size >= top - 0.1]
            if top > 0 and cand:
                t = _clean(" ".join(b.text for b in cand))
                if 3 <= len(t) <= 300:
                    return t
        mt = (meta.get("title") or "").strip()
        return mt or path.stem
