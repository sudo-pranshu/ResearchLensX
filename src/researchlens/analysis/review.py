"""Evidence-linked literature review: Markdown / HTML / PDF export.

No generative model writes any sentence here. Every factual statement is a verbatim extracted sentence, an extracted
name, or a measured value, followed by a numbered citation [n] that resolves to (paper, section, page, quote) in
the Sources list. Section headings and connective phrases are fixed templates. If nothing was extracted for a
section it says so instead of filling the space.
"""
from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass, field
from typing import Sequence

from researchlens.analysis.compare import COLUMNS, comparison_table
from researchlens.analysis.gaps import Gap
from researchlens.contracts import Document, KnowledgeGraph
from researchlens.generation.extractive import split_sentences

CAVEAT = ("This review was assembled automatically from sentences and names extracted verbatim from the uploaded PDFs. "
          "Extraction is rule-based and imperfect (limited recall, occasional false positives). A statement's citation shows where it "
          "came from, not that it is true. Absence of an item means it was not detected, not that the paper lacks it. "
          "Gaps and research questions are suggestions for a human to evaluate, limited to the uploaded collection.")


@dataclass
class Source:
    n: int
    doc_id: str
    title: str
    section: str
    page: int | None
    quote: str


class Citations:
    def __init__(self):
        self.sources: list[Source] = []
        self._idx: dict[tuple, int] = {}

    def cite(self, doc_id: str, title: str, section: str, page: int | None, quote: str) -> str:
        key = (doc_id, page, quote[:120])
        if key not in self._idx:
            self.sources.append(Source(len(self.sources) + 1, doc_id, title, section, page, quote))
            self._idx[key] = len(self.sources)
        return f"[{self._idx[key]}]"


@dataclass
class Review:
    title: str
    sections: list[tuple[str, str]]          # (heading, markdown)
    sources: list[Source]
    papers: list[tuple[str, str, int | None]] = field(default_factory=list)   # (doc_id, title, year)


def _q(s: str, n: int = 260) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def build_review(documents: Sequence[Document], kg: KnowledgeGraph, gaps: Sequence[Gap], conflicts: Sequence[dict] = (),
                 title: str = "Literature review") -> Review:
    cites = Citations()
    table = comparison_table(kg, documents, max_items=8)
    by_id = {d.doc_id: d for d in documents}
    sec: list[tuple[str, str]] = []

    # 1 research problem: first sentence of each abstract
    lines = []
    for d in documents:
        first = (split_sentences(d.abstract) or [""])[0]
        if first:
            c = cites.cite(d.doc_id, d.title, "Abstract", d.abstract_page, first)
            lines.append(f"- **{d.title}** ({d.year or 'year unknown'}): “{_q(first)}” {c}")
        else:
            lines.append(f"- **{d.title}** ({d.year or 'year unknown'}): no abstract was detected.")
    sec.append(("Research problem (first sentence of each abstract)", "\n".join(lines)))

    # 2 background: measured relations between the papers
    by_n = {n.id: n for n in kg.nodes}
    rel = []
    for e in kg.edges:
        a, b = by_n[e.source], by_n.get(e.target)
        if e.type == "cites":
            ev = e.attrs["evidence"][0]
            rel.append(f"- {a.label} cites {b.label}: reference entry “{_q(ev['quote'], 160)}” {cites.cite(ev['doc_id'], a.label, 'References', None, ev['quote'])}")
        elif e.type == "similar_problem":
            rel.append(f"- {a.label} and {b.label} have lexically similar titles/abstracts (TF-IDF cosine {e.attrs['cosine']}); this does not establish they study the same problem.")
        elif e.type == "shares_entity":
            rel.append(f"- {a.label} and {b.label} both involve: {', '.join(e.attrs['shared'])}.")
    sec.append(("Background", "\n".join(rel) or "No citations or shared entities were detected between these papers."))

    # 3 / 4 methodology and comparison
    def cell(items, long=False):
        if not items:
            return "not detected"
        out = []
        for i in items:
            row_ref = cites.cite(*items_meta[id(i)], i.quote) if i.quote else ""
            out.append(f"{_q(i.text, 180 if long else 60)} {row_ref}".strip())
        return "; ".join(out)
    items_meta: dict[int, tuple] = {}
    for r in table:
        for its in r.cells.values():
            for i in its:
                items_meta[id(i)] = (r.doc_id, r.title, i.section, i.page)
    meth = ["| Paper | Methods / models | Datasets | Metrics |", "|---|---|---|---|"]
    for r in table:
        meth.append(f"| {r.title[:60]} | {cell(r.cells['method'][:5])} | {cell(r.cells['dataset'][:5])} | {cell(r.cells['metric'][:5])} |")
    sec.append(("Existing methodologies", "\n".join(meth)))
    sec.append(("Paper-by-paper comparison", "\n\n".join(
        f"**{r.title}**\n" + "\n".join(f"- {label}: {cell(r.cells[key][:4], long=key in ('finding', 'limitation'))}" for key, label, _ in COLUMNS)
        for r in table)))

    # 5 findings
    f_lines = []
    for r in table:
        for i in r.cells["finding"][:4]:
            f_lines.append(f"- {r.title[:50]}: “{_q(i.text)}” {cites.cite(r.doc_id, r.title, i.section, i.page, i.quote)}")
    sec.append(("Experimental findings", "\n".join(f_lines) or "No finding sentences were detected."))

    # 6 contradictory evidence
    if conflicts:
        c_lines = []
        for c in conflicts:
            ev = cites.cite(c["evidence_doc"], by_id[c["evidence_doc"]].title if c["evidence_doc"] in by_id else "", "", c["evidence_page"], c["evidence_quote"])
            c_lines.append(f"- {c['relation'].upper()} (NLI-flagged candidate, unreviewed, confidence {c['confidence']:.2f}): finding “{_q(c['finding'], 160)}” "
                           f"vs “{_q(c['evidence_quote'], 160)}” {ev}")
        sec.append(("Contradictory evidence (NLI-flagged candidates, unreviewed)", "\n".join(c_lines)))
    else:
        sec.append(("Contradictory evidence (NLI-flagged candidates, unreviewed)", "No cross-paper contradictions or confirmations were flagged. This is not a finding that none exist: "
                                              "cross-paper NLI is optional and gated by relevance checks."))

    # 7 limitations
    l_lines = []
    for r in table:
        for i in r.cells["limitation"][:3]:
            l_lines.append(f"- {r.title[:50]}: “{_q(i.text)}” {cites.cite(r.doc_id, r.title, i.section, i.page, i.quote)}")
    sec.append(("Research limitations", "\n".join(l_lines) or "No limitation sentences were detected."))

    # 8 gaps, 9 future directions
    g_lines, f_dirs = [], []
    for g in gaps:
        ev = " ".join(cites.cite(e.doc_id, e.title, e.section, e.page, e.quote) for e in g.evidence[:3])
        quotes = "; ".join(f"“{_q(e.quote, 200)}”" for e in g.evidence[:2])
        block = (f"- **{g.id}** ({g.kind.replace('_', ' ')}; basis: {g.basis}; confidence: {g.confidence}). {g.statement} Evidence: {quotes} {ev}\n"
                 f"  - Status in the collection (not checked against the other papers): {g.why_unaddressed}\n  - Proposed research question: {g.research_question}\n"
                 f"  - Potential methodology: {g.methodology}\n  - Limitations: {g.limitations}\n  - {g.scope}")
        g_lines.append(block)
        if g.kind in ("open_question", "missing_comparison"):
            f_dirs.append(f"- {g.research_question} (from {g.id}) {ev}")
    sec.append(("Candidate research gaps", "\n".join(g_lines) or "No gaps were identified from the extracted evidence."))
    sec.append(("Future research directions", "\n".join(f_dirs) or "No open-question or missing-comparison gaps were produced."))

    refs = [f"{k}. **{t}** ({y or 'year unknown'}), `{i}`" for k, (i, t, y) in enumerate(((d.doc_id, d.title, d.year) for d in documents), 1)]
    sources = ["\n".join(f"[{s.n}] {s.title[:70]}, §{s.section or 'n/a'}, {('p. ' + str(s.page)) if s.page is not None else 'page unknown'}: “{_q(s.quote, 200)}”"
                         for s in cites.sources)]
    sec.append(("References", "**Papers analysed**\n\n" + "\n".join(refs) + "\n\n**Cited passages**\n\n" + "\n\n".join(sources)))
    return Review(title, sec, cites.sources, [(d.doc_id, d.title, d.year) for d in documents])


def review_markdown(r: Review) -> str:
    return f"# {r.title}\n\n> {CAVEAT}\n\n" + "\n\n".join(f"## {i}. {h}\n\n{body}" for i, (h, body) in enumerate(r.sections, 1))


def _md_to_html(md: str) -> str:
    """Tiny Markdown subset (headings, bold, italics, code, lists, pipe tables, blockquote) -> HTML, escaping first."""
    out, in_list, in_table = [], False, False
    for raw in md.split("\n"):
        line = _html.escape(raw)
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"(?<![\w*])\*(?!\s)(.+?)\*(?![\w*])", r"<i>\1</i>", line)
        line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)
        if raw.startswith("|"):
            if not in_table:
                out.append("<table>"); in_table = True
            if re.fullmatch(r"\|[-| ]+\|", raw):
                continue
            tag = "th" if out[-1] == "<table>" else "td"
            out.append("<tr>" + "".join(f"<{tag}>{c.strip()}</{tag}>" for c in line.strip("|").split("|")) + "</tr>")
            continue
        if in_table:
            out.append("</table>"); in_table = False
        m = re.match(r"^(\s*)- (.*)", line)
        if m:
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{m.group(2)}</li>")
            continue
        if in_list:
            out.append("</ul>"); in_list = False
        if raw.startswith("# "):
            out.append(f"<h1>{line[2:]}</h1>")
        elif raw.startswith("## "):
            out.append(f"<h2>{line[3:]}</h2>")
        elif raw.startswith("&gt; ") or raw.startswith("> "):
            out.append(f"<blockquote>{line.split(' ', 1)[1]}</blockquote>")
        elif raw.strip():
            out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    if in_table:
        out.append("</table>")
    return "\n".join(out)


def review_html(r: Review) -> str:
    css = ("body{font-family:Georgia,serif;max-width:860px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#222}"
           "h1{font-size:1.7rem}h2{font-size:1.2rem;margin-top:1.6rem;border-bottom:1px solid #ddd}blockquote{background:#f4f6fa;"
           "margin:1rem 0;padding:.6rem 1rem;border-left:4px solid #2F6FED;font-size:.92rem}table{border-collapse:collapse;width:100%;"
           "font-size:.85rem}td,th{border:1px solid #ccc;padding:4px 6px;vertical-align:top}ul{padding-left:1.2rem}code{font-size:.85em}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{_html.escape(r.title)}</title><style>{css}</style></head>"
            f"<body>{_md_to_html(review_markdown(r))}</body></html>")


def review_pdf(r: Review, path: str) -> str:
    """Render the HTML to a paginated PDF with PyMuPDF's Story (no extra dependency)."""
    import pymupdf
    html = review_html(r)
    story = pymupdf.Story(html=html)
    writer = pymupdf.DocumentWriter(path)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (48, 48, -48, -48)
    more = 1
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    return path
