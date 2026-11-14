"""Cross-paper comparison built from the knowledge graph (so every cell traces to a quote and page).

  table = comparison_table(kg, documents)        # one row per paper, columns: methods, datasets, metrics, findings, limitations
  shared = entity_overlap(kg)                    # entities used by several papers vs unique to one
  md = table_markdown(table)                     # export

Cells are lists of `Item(text, page, quote)`. Nothing is summarised or paraphrased: items are the extracted names or
verbatim sentences. Papers for which nothing was extracted get an empty cell, shown as "not detected" (absence of
extraction is not evidence of absence in the paper).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from researchlens.contracts import Document, KnowledgeGraph

COLUMNS = [("method", "Methods / models", "uses_method"), ("dataset", "Datasets", "evaluates_on"),
           ("metric", "Metrics", "uses_metric"), ("finding", "Reported findings", "reports_finding"),
           ("limitation", "Limitations", "has_limitation")]


@dataclass
class Item:
    text: str
    page: int | None
    quote: str
    mentions: int = 1


@dataclass
class Row:
    doc_id: str
    title: str
    year: int | None
    cells: dict[str, list[Item]] = field(default_factory=dict)


def comparison_table(kg: KnowledgeGraph, documents: Sequence[Document], max_items: int = 6) -> list[Row]:
    by_id = {n.id: n for n in kg.nodes}
    out = []
    for d in documents:
        pid = f"paper:{d.doc_id}"
        row = Row(d.doc_id, d.title, d.year, {key: [] for key, _, _ in COLUMNS})
        for e in kg.edges:
            if e.source != pid:
                continue
            for key, _, et in COLUMNS:
                if e.type == et:
                    ev = (e.attrs.get("evidence") or [{}])[0]
                    tgt = by_id[e.target]
                    text = tgt.attrs.get("full_text", tgt.label) if key in ("finding", "limitation") else tgt.label
                    row.cells[key].append(Item(text, ev.get("page"), ev.get("quote", ""), int(e.attrs.get("mentions", 1))))
        for key in row.cells:
            row.cells[key].sort(key=lambda i: -i.mentions)
            row.cells[key] = row.cells[key][:max_items]
        out.append(row)
    return out


def entity_overlap(kg: KnowledgeGraph) -> dict[str, dict[str, list[str]]]:
    """{'method': {'shared': [...], 'unique': [...]}, ...} over the entity nodes present in the graph."""
    res: dict[str, dict[str, list[str]]] = {}
    for kind in ("method", "dataset", "metric"):
        shared, unique = [], []
        for n in kg.nodes:
            if n.type == "entity" and n.attrs.get("entity_type") == kind:
                (shared if len(n.attrs.get("papers", [])) > 1 else unique).append(n.label)
        res[kind] = {"shared": sorted(shared), "unique": sorted(unique)}
    return res


def _cell_text(items: list[Item], long: bool) -> str:
    if not items:
        return "not detected"
    parts = []
    for i in items:
        t = i.text if long else i.text
        if long and len(t) > 220:
            t = t[:217] + "…"
        parts.append(f"{t} (p. {i.page})" if i.page is not None else f"{t} (page unknown)")
    return "; ".join(parts) if not long else "\n".join(f"- {p}" for p in parts)


def table_rows_flat(table: list[Row]) -> list[dict]:
    """For st.dataframe: plain strings."""
    return [{"Paper": f"{r.title[:70]} ({r.year or 'n.d.'})",
             **{label: _cell_text(r.cells[key], long=False) for key, label, _ in COLUMNS}} for r in table]


def table_markdown(table: list[Row]) -> str:
    lines = []
    for r in table:
        lines.append(f"### {r.title} ({r.year or 'year unknown'})")
        for key, label, _ in COLUMNS:
            lines.append(f"**{label}**\n{_cell_text(r.cells[key], long=True)}\n")
    return "\n".join(lines)


def conflicting_pairs(kg: KnowledgeGraph) -> list[dict]:
    """Cross-paper supports/contradicts edges (NLI candidates) flattened for display."""
    by_id = {n.id: n for n in kg.nodes}
    out = []
    for e in kg.edges:
        if e.type in ("supports", "contradicts"):
            ev = (e.attrs.get("evidence") or [{}])[0]
            out.append({"relation": e.type, "confidence": e.weight, "finding": by_id[e.source].attrs.get("full_text", ""),
                        "finding_doc": by_id[e.source].attrs.get("doc_id"), "evidence_doc": ev.get("doc_id"),
                        "evidence_page": ev.get("page"), "evidence_quote": ev.get("quote", ""), "status": e.attrs.get("status", "")})
    return out
