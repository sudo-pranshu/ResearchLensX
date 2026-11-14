"""ResearchLens X: Streamlit front end. Talks only to researchlens.workspace.Workspace.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import researchlens._runtime  # noqa: E402,F401

import streamlit as st  # noqa: E402

import components as C  # noqa: E402
from researchlens.qa.engine import Reference  # noqa: E402
from researchlens.workspace import Settings, Workspace  # noqa: E402

st.set_page_config(page_title="ResearchLens X", page_icon="🔬", layout="wide")
C.inject_css()
DB = os.environ.get("RLX_DB", str(ROOT / "data" / "processed" / "library.sqlite"))


@st.cache_resource
def get_workspace(db_path: str) -> Workspace:
    return Workspace(db_path)          # cached per database path


ws = get_workspace(DB)
PAGES = ["Library", "Ask", "Search", "Verify a claim", "Knowledge graph", "Evaluation"]

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🔬 ResearchLens X")
    st.caption("Evidence-aware scientific literature analysis")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    infos = ws.library()
    labels = {i.doc_id: f"{C.clean_title(i.title)[:48]}" for i in infos}
    scope = st.multiselect("Papers in scope", list(labels), format_func=labels.get, placeholder="All papers",
                           help="Restrict search, QA and verification to these papers.")
    with st.expander("Retrieval settings", expanded=False):
        retr = st.selectbox("First-stage retriever", ["bm25", "tfidf"], help="Lexical retrievers, implemented from scratch.")
        rerank = st.toggle("Cross-encoder reranking", value=False,
                           help="Reranks the top candidates with a small cross-encoder (downloads ~90 MB the first time). "
                                "On QASPER dev it raised Recall@5 from .50 to .58 (EXP-H1-007).")
        dedup = st.slider("Redundancy suppression", 0.5, 1.0, 0.85, 0.05, help="Drop passages that repeat a better one.")
        k = st.slider("Passages (k)", 2, 15, 6)
        per_doc = st.slider("Max per paper", 1, 6, 3)
        excl_refs = st.toggle("Ignore reference lists", value=True)
    verify = st.toggle("Verify answer sentences with NLI", value=False,
                       help="Loads a DeBERTa NLI model (~700 MB). If it cannot load, a clearly labelled MOCK is used.")
    settings = Settings(retriever=retr, rerank=rerank, dedup=dedup if dedup < 1.0 else None, k=k, max_per_doc=per_doc,
                        verify=verify, exclude_sections=("references",) if excl_refs else ())
    st.divider()
    C.section_head("Components")
    C.status_panel(ws.status(settings))
doc_ids = scope or None


def need_library() -> bool:
    if not ws.library():
        st.info("The library is empty. Upload PDFs on the **Library** page to begin.")
        return True
    return False


# ------------------------------------------------------------------ Library
if page == "Library":
    st.header("Library")
    st.write("Upload text-based research PDFs. Each paper is parsed into sections and pages so every answer can point back to "
             "the exact page. Scanned PDFs are rejected (no OCR).")
    files = st.file_uploader("Upload papers", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Ingest uploaded PDFs", type="primary"):
        with st.spinner(f"Parsing {len(files)} PDF(s)…"):
            rep = ws.upload([(f.name, f.getvalue()) for f in files])
        if rep.added:
            st.success(f"Added {len(rep.added)} paper(s).")
        if rep.duplicates:
            st.info("Already in the library: " + ", ".join(rep.duplicates))
        for name, err in rep.failed:
            st.error(f"{name}: {err}")
    with st.expander("Or load synthetic demo fixtures (not real papers)"):
        st.caption("Tiny hand-written fixtures from `data/sample/` for trying the interface without PDFs.")
        if st.button("Load fixtures"):
            from researchlens.mocks import load_sample_documents
            st.success(f"Added {ws.add_documents(load_sample_documents())} fixture document(s).")
    infos = ws.library()
    if not infos:
        st.info("No papers yet.")
    else:
        st.subheader(f"{len(infos)} paper(s)")
        for i in infos:
            c1, c2 = st.columns([8, 1])
            c1.markdown(f"**{C.clean_title(i.title)}**  \n<span class='rl-meta'>{i.doc_id} · {i.year or 'year unknown'} · "
                        f"{i.n_sections} sections · {i.n_references} references · source: {i.source}</span>", unsafe_allow_html=True)
            if c2.button("Remove", key=f"rm{i.doc_id}"):
                ws.remove(i.doc_id); st.rerun()

# ------------------------------------------------------------------ Ask
elif page == "Ask":
    st.header("Ask across your papers")
    if not need_library():
        q = st.text_input("Question", placeholder="Which datasets do these papers use for evaluation?")
        if q:
            try:
                with st.spinner("Retrieving evidence" + (" and verifying claims…" if verify else "…")):
                    res, note = ws.ask(q, settings, doc_ids)
            except Exception as e:
                st.error(f"Could not answer: {type(e).__name__}: {e}")
                st.stop()
            if note:
                st.warning(note)
            C.mock_banner(res.is_mock)
            if res.abstained:
                st.warning(f"**{res.sentences[0].text}**  \n{res.abstain_reason}")
            else:
                C.section_head("Answer")
                C.cited_answer(res)
                st.caption("The answer is extractive: each sentence is copied from the cited passage. Statuses come from an NLI model "
                           "and indicate textual evidence, not scientific truth.")
            C.section_head(f"Evidence ({len(res.references)} passages from {len({r.doc_id for r in res.references})} paper(s))")
            cited = {r: [s.text for s in res.sentences if r in s.ref_ids] for r in range(1, len(res.references) + 1)}
            for ref in res.references:
                C.evidence_card(ref, cited[ref.ref_id])
            with st.expander("Timings and diagnostics"):
                st.json({"timings_ms": res.timings_ms, "evidence_coverage": round(res.evidence_coverage, 3),
                         "docs_used": res.docs_used, "generator": res.answer.generator})

# ------------------------------------------------------------------ Search
elif page == "Search":
    st.header("Search")
    if not need_library():
        q = st.text_input("Query", placeholder="contrastive pre-training for retrieval")
        compare = st.toggle("Compare retrievers side by side", value=False)
        if q:
            try:
                variants = ([("BM25", Settings(**{**settings.__dict__, "retriever": "bm25", "rerank": False})),
                             ("TF-IDF", Settings(**{**settings.__dict__, "retriever": "tfidf", "rerank": False})),
                             ("BM25 + rerank", Settings(**{**settings.__dict__, "retriever": "bm25", "rerank": True}))]
                            if compare else [("Selected", settings)])
                cols = st.columns(len(variants))
                ranked: dict[str, list[str]] = {}
                for col, (name, s) in zip(cols, variants):
                    with col:
                        with st.spinner(f"{name}…"):
                            passages, info = ws.search(q, s, doc_ids)
                        st.markdown(f"**{name}** · {info['latency_ms']:.0f} ms")
                        if info["note"]:
                            st.warning(info["note"])
                        ranked[name] = [p.chunk.chunk_id for p in passages]
                        if not passages:
                            st.info("No passages matched.")
                        titles = {i.doc_id: i.title for i in ws.library()}
                        for p in passages:
                            ref = Reference(ref_id=p.rank, doc_id=p.chunk.doc_id, title=C.clean_title(titles.get(p.chunk.doc_id, p.chunk.doc_id)),
                                            section=p.chunk.section, page=p.chunk.page, chunk_id=p.chunk.chunk_id,
                                            text=p.chunk.text, retriever=p.retriever, score=p.score)
                            C.evidence_card(ref)
                if compare and len(ranked) > 1:
                    names = list(ranked)
                    base = set(ranked[names[0]])
                    st.caption("Overlap with BM25 top-k: " + ", ".join(
                        f"{n}: {len(base & set(ranked[n]))}/{len(ranked[n])}" for n in names[1:]))
            except Exception as e:
                st.error(f"Search failed: {type(e).__name__}: {e}")

# ------------------------------------------------------------------ Verify a claim
elif page == "Verify a claim":
    st.header("Verify a claim against your papers")
    st.write("Enter a scientific claim. The system retrieves passages and an NLI model judges whether they entail, contradict, "
             "or say nothing about it. *Not enough evidence* means the retrieved text is silent, which is not the same as the claim being false.")
    if not need_library():
        claim = st.text_area("Claim", placeholder="Dense retrieval outperforms BM25 on scientific question answering.", height=80)
        if claim.strip() and st.button("Verify", type="primary"):
            try:
                with st.spinner("Retrieving and running NLI (first run loads the model)…"):
                    v, passages = ws.verify_claim(claim.strip(), Settings(**{**settings.__dict__, "verify": True}), doc_ids)
                st.session_state["verify_out"] = (claim.strip(), v, passages)      # survive reruns (e.g. toggling a setting)
            except Exception as e:
                st.error(f"Verification failed: {type(e).__name__}: {e}")
        out = st.session_state.get("verify_out")
        if out and out[0] == claim.strip():
            _, v, passages = out
            C.mock_banner(v.is_mock)
            st.markdown(f"{C.pill(v.label)} confidence {v.confidence:.2f}"
                        + (f" · {C.pill('inferred', 'combined evidence')}" if v.support == "combined" else ""), unsafe_allow_html=True)
            st.progress(min(max(v.confidence, 0.0), 1.0))
            st.write(v.explanation or "")
            titles = {i.doc_id: i.title for i in ws.library()}
            C.section_head("Evidence judged")
            for n, (p, e) in enumerate(zip(passages, v.evidence), start=1):
                ref = Reference(ref_id=n, doc_id=p.chunk.doc_id, title=C.clean_title(titles.get(p.chunk.doc_id, p.chunk.doc_id)),
                                section=p.chunk.section, page=p.chunk.page, chunk_id=p.chunk.chunk_id, text=p.chunk.text,
                                retriever=p.retriever, score=p.score)
                probs = " · ".join(f"{k[:3]} {x:.2f}" for k, x in e.probs.items())
                C.evidence_card(ref, C.sentence_spans(p.chunk.text, e.rationale_sentences),
                                extra=f"· {C.pill(e.label)} {probs}")

# ------------------------------------------------------------------ Knowledge graph
elif page == "Knowledge graph":
    st.header("Knowledge graph")
    st.write("Papers, authors, methods, datasets, metrics, findings and limitations extracted from the PDFs. Every node and edge "
             "keeps the sentence and page it came from (hover, or inspect below). Relations are extracted or measured, never invented; "
             "extraction is rule-based and imperfect.")
    if not need_library():
        link = st.toggle("Verify cross-paper relations with NLI (slow; adds supports / contradicts edges)", value=False,
                         help="Checks each paper's findings against the other papers' passages with the real NLI model. A mock model never adds edges.")
        try:
            with st.spinner("Building graph…" + (" running NLI between papers…" if link else "")):
                kg = ws.graph(settings, doc_ids, link_with_nli=link)
        except Exception as e:
            st.error(f"Could not build the graph: {type(e).__name__}: {e}")
            st.stop()
        if link and ws.status(settings)[-1].state == "mock":
            st.warning("The NLI model could not be loaded, so no supports/contradicts edges were created (a MOCK never adds relations).")
        all_nodes = list(C.NODE_STYLE)
        default_nodes = ["paper", "entity:method", "entity:dataset", "entity:metric"]
        nt = set(st.multiselect("Node types", all_nodes, default=default_nodes))
        et_all = sorted({e.type for e in kg.edges})
        et = set(st.multiselect("Relations", et_all, default=[e for e in et_all if e not in ("reports_finding", "has_limitation", "authored_by")]))
        from collections import Counter as _C
        st.caption(" · ".join(f"{k}: {v}" for k, v in sorted(_C(C.node_key(n) for n in kg.nodes).items())) + "  |  "
                   + " · ".join(f"{k}: {v}" for k, v in sorted(_C(e.type for e in kg.edges).items())))
        import streamlit.components.v1 as components
        components.html(C.graph_html(kg, nt, et), height=640, scrolling=False)
        C.section_head("Inspect a node")
        by_id = {n.id: n for n in kg.nodes}
        pick = st.selectbox("Node", list(by_id), format_func=lambda i: f"[{C.node_key(by_id[i])}] {by_id[i].label[:90]}")
        n = by_id[pick]
        st.markdown(f"**{n.label}**")
        for ev in n.attrs.get("evidence", []):
            st.markdown(f"> {ev.get('quote') or ev.get('source')}  \n<span class='rl-meta'>{ev.get('doc_id', '')} · {ev.get('section', '')} · "
                        f"p. {ev.get('page') if ev.get('page') is not None else 'unknown'}</span>", unsafe_allow_html=True)
        rel = [e for e in kg.edges if pick in (e.source, e.target)]
        if rel:
            st.dataframe([{"relation": e.type, "from": by_id[e.source].label[:50], "to": by_id[e.target].label[:50], "weight": round(e.weight, 2),
                           "evidence": (e.attrs.get("evidence") or [{}])[0].get("quote", "")[:160]} for e in rel], hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ Evaluation
else:
    st.header("Evaluation results")
    st.caption("Read from the committed `results/` folder; nothing here is computed in the app. Dev results guided tuning; "
               "the QASPER test split was run once (EXP-H1-006).")
    runs = sorted(p for p in (ROOT / "results").glob("EXP-*") if (p / "summary.json").exists())
    if not runs:
        st.info("No result folders found.")
    else:
        name = st.selectbox("Experiment", [r.name for r in runs], index=len(runs) - 1)
        summary = json.loads((ROOT / "results" / name / "summary.json").read_text())
        cfg_p = ROOT / "results" / name / "config.json"
        cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
        if cfg:
            st.markdown(f"**Split:** {cfg.get('split_name') or cfg.get('split_reported', '?')} · **git:** `{cfg.get('git', '?')}` · "
                        f"**questions/claims:** {cfg.get('n_questions') or cfg.get('n_pairs', '?')}")
        import plotly.graph_objects as go
        first = next(iter(summary.values())) if summary else None
        if isinstance(first, dict) and "recall@5" in first:
            metric = st.selectbox("Metric", ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"], index=1)
            systems = list(summary)
            fig = go.Figure(go.Bar(x=[summary[s][metric] for s in systems], y=systems, orientation="h",
                                   text=[f"{summary[s][metric]:.3f}" for s in systems], textposition="outside"))
            fig.update_layout(height=60 + 38 * len(systems), margin=dict(l=0, r=30, t=10, b=0), xaxis_title=metric,
                              yaxis=dict(autorange="reversed"), template=None)
            st.plotly_chart(fig, use_container_width=True)
            cp = ROOT / "results" / name / "comparisons.json"
            if cp.exists():
                comps = json.loads(cp.read_text())
                if comps:
                    C.section_head("Significance (paper-clustered bootstrap, Holm-adjusted)")
                    st.dataframe([{"baseline": c["system_a"], "system": c["system_b"], "Δ": round(c["delta"], 4),
                                   "95% CI": f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]", "p (Holm)": round(c["p_holm"], 4)}
                                  for c in comps], hide_index=True, use_container_width=True)
        elif isinstance(first, dict) and "macro_f1" in first:
            systems = list(summary)
            fig = go.Figure(go.Bar(x=[summary[s]["macro_f1"] for s in systems], y=systems, orientation="h",
                                   error_x=dict(type="data", symmetric=False,
                                                array=[summary[s]["macro_f1_ci95"][1] - summary[s]["macro_f1"] for s in systems],
                                                arrayminus=[summary[s]["macro_f1"] - summary[s]["macro_f1_ci95"][0] for s in systems])))
            fig.update_layout(height=60 + 42 * len(systems), margin=dict(l=0, r=30, t=10, b=0), xaxis_title="macro-F1 (95% CI)",
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.json(summary)
