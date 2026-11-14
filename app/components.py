"""Presentation helpers for the Streamlit app. Colours use CSS variables so they work in light and dark themes."""
from __future__ import annotations

import html
import re

import streamlit as st

STATUS_STYLE = {   # status -> (label, colour)
    "supported": ("Directly supported", "#1a9850"), "inferred": ("Inferred from several passages", "#2F6FED"),
    "conflicting": ("Conflicting evidence", "#e08a00"), "contradicted": ("Contradicted", "#d73027"),
    "unsupported": ("No supporting evidence", "#7a7f87"), "unchecked": ("Not verified", "#7a7f87"),
    "abstained": ("Insufficient evidence", "#e08a00"),
    "SUPPORTS": ("Supports", "#1a9850"), "CONTRADICTS": ("Contradicts", "#d73027"), "INSUFFICIENT": ("Not enough evidence", "#7a7f87"),
    "real": ("real", "#1a9850"), "fallback": ("fallback", "#e08a00"), "mock": ("MOCK", "#d73027"), "off": ("off", "#7a7f87"),
}

CSS = """
<style>
.rl-pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.78rem;font-weight:600;color:#fff;margin-right:6px;white-space:nowrap}
.rl-card{border:1px solid rgba(128,128,128,.28);border-radius:10px;padding:12px 14px;margin:8px 0;background:var(--secondary-background-color)}
.rl-meta{opacity:.72;font-size:.82rem;margin-bottom:6px}
.rl-quote{font-size:.93rem;line-height:1.5}
.rl-quote mark{background:rgba(47,111,237,.28);color:inherit;border-radius:3px;padding:0 2px}
.rl-ref{display:inline-block;min-width:1.5em;text-align:center;border:1px solid rgba(128,128,128,.5);border-radius:6px;font-size:.75rem;padding:0 4px;margin-left:3px;opacity:.85}
.rl-h{font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;opacity:.6;margin:14px 0 4px}
.rl-mock{border:2px solid #d73027;color:#d73027;border-radius:8px;padding:6px 10px;font-weight:700;margin:6px 0}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def pill(key: str, text: str | None = None) -> str:
    label, colour = STATUS_STYLE.get(key, (key, "#7a7f87"))
    return f'<span class="rl-pill" style="background:{colour}">{html.escape(text or label)}</span>'


def mock_banner(is_mock: bool) -> None:
    if is_mock:
        st.markdown('<div class="rl-mock">MOCK: a placeholder component produced part of this output (not a real model). '
                    'Do not treat it as evidence.</div>', unsafe_allow_html=True)


def highlight(text: str, spans: list[str]) -> str:
    out = html.escape(text)
    for s in spans:
        es = html.escape(s.strip())
        if es and es in out:
            out = out.replace(es, f"<mark>{es}</mark>", 1)
    return out


def where(ref) -> str:
    page = f"p. {ref.page}" if ref.page is not None else "page unknown"
    return f"{html.escape(ref.title)} · §{html.escape(ref.section)} · {page}"


def evidence_card(ref, cited_sentences: list[str] | None = None, extra: str = "") -> None:
    st.markdown(
        f'<div class="rl-card"><div class="rl-meta"><span class="rl-ref">{ref.ref_id}</span> {where(ref)} · '
        f'{html.escape(ref.retriever)} · score {ref.score:.3f} {extra}</div>'
        f'<div class="rl-quote">{highlight(ref.text, cited_sentences or [])}</div></div>', unsafe_allow_html=True)


def cited_answer(res) -> None:
    """Render answer sentences with status pills and [n] markers linking to the evidence list below."""
    for s in res.sentences:
        refs = "".join(f'<span class="rl-ref">{i}</span>' for i in s.ref_ids)
        conf = f" ({s.confidence:.2f})" if s.confidence is not None else ""
        st.markdown(f'<div class="rl-card"><div class="rl-quote">{html.escape(s.text)} {refs}</div>'
                    f'<div style="margin-top:6px">{pill(s.status)}<span class="rl-meta">{conf}</span></div></div>',
                    unsafe_allow_html=True)


def status_panel(components) -> None:
    for c in components:
        st.markdown(f'{pill(c.state)} **{html.escape(c.name)}** <span class="rl-meta">{html.escape(c.detail)}</span>',
                    unsafe_allow_html=True)


def section_head(text: str) -> None:
    st.markdown(f'<div class="rl-h">{html.escape(text)}</div>', unsafe_allow_html=True)


def sentence_spans(passage_text: str, rationale_idx: list[int]) -> list[str]:
    from researchlens.generation.extractive import split_sentences
    sents = split_sentences(passage_text) or [passage_text]
    return [sents[i] for i in rationale_idx if 0 <= i < len(sents)]


def clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()


NODE_STYLE = {   # type -> (colour, shape)
    "paper": ("#2F6FED", "dot"), "author": ("#8e6bd1", "triangle"), "finding": ("#1a9850", "square"),
    "limitation": ("#d73027", "diamond"), "claim": ("#e08a00", "star"), "section": ("#7a7f87", "dot"),
    "entity:method": ("#00a3a3", "dot"), "entity:dataset": ("#e08a00", "database"), "entity:metric": ("#a0a4ab", "dot"),
}


def node_key(n) -> str:
    return f"entity:{n.attrs.get('entity_type', 'method')}" if n.type == "entity" else n.type


def graph_html(kg, node_types: set[str], edge_types: set[str], height: int = 620) -> str:
    """pyvis network (all JS inlined: works offline). Tooltips carry the source quote and page of each element."""
    from pyvis.network import Network
    net = Network(height=f"{height}px", width="100%", directed=True, cdn_resources="in_line", bgcolor="rgba(0,0,0,0)")
    net.barnes_hut(gravity=-6000, central_gravity=0.25, spring_length=140)
    keep = set()
    for n in kg.nodes:
        k = node_key(n)
        if k not in node_types:
            continue
        colour, shape = NODE_STYLE.get(k, ("#7a7f87", "dot"))
        ev = (n.attrs.get("evidence") or [{}])[0]
        tip = html.escape(n.label) + (f"<br><i>{html.escape(str(ev.get('quote', ''))[:260])}</i><br>p. {ev.get('page')}" if ev.get("quote") else "")
        size = 26 if n.type == "paper" else 14 + min(len(n.attrs.get("papers", [])), 3) * 3
        net.add_node(n.id, label=n.label[:34], title=tip, color=colour, shape=shape, size=size)
        keep.add(n.id)
    for e in kg.edges:
        if e.type in edge_types and e.source in keep and e.target in keep:
            ev = (e.attrs.get("evidence") or [{}])[0]
            tip = e.type + (f": {html.escape(str(ev.get('quote', ''))[:260])} (p. {ev.get('page')})" if ev.get("quote") else "")
            colour = {"contradicts": "#d73027", "supports": "#1a9850"}.get(e.type, "#9aa0a6")
            net.add_edge(e.source, e.target, title=tip, label=e.type.replace("_", " "), color=colour,
                         width=1 + min(e.weight, 5) * 0.4 if e.type in ("similar_problem", "shares_entity", "supports", "contradicts") else 1,
                         font={"size": 9, "color": "#888"}, arrows="to")
    return net.generate_html()
