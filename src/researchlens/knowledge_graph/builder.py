"""Knowledge graph of the uploaded collection.

Every edge is either (a) extracted with provenance (quote, chunk, page) from a paper's text/metadata,
(b) computed from a stated, auditable measure (reference matching, TF-IDF cosine of abstracts, shared entity
nodes), or (c) produced by NLI verification and tagged with the model and the evidence. A mock verifier is
never allowed to create supports/contradicts edges. There are no "extends" edges: that relation cannot be
established reliably by rules, so it is not claimed.

Node ids: paper:<doc_id> · author:<name> · entity:<type>:<key> · finding:<doc>:<n> · limitation:<doc>:<n>
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Sequence

from researchlens.contracts import Claim, Document, KGEdge, KGNode, KnowledgeGraph, VerificationResult
from researchlens.knowledge_graph.extract import extract_from_chunk, title_method
from researchlens.processing.chunking import chunk_document

ENTITY_EDGE = {"method": "uses_method", "dataset": "evaluates_on", "metric": "uses_metric"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class GraphBuilder:
    name = "rule-based-graph-v1"
    is_mock = False

    def __init__(self, min_mentions: int = 2, max_findings: int = 6, max_limitations: int = 4,
                 similarity_threshold: float = 0.30, evidence_per_edge: int = 3):
        self.min_mentions, self.max_findings, self.max_limitations = min_mentions, max_findings, max_limitations
        self.sim_threshold, self.k_ev = similarity_threshold, evidence_per_edge

    # ------------------------------------------------------------------ build
    def build(self, documents: Sequence[Document], verifications: list[VerificationResult] | None = None,
              verifier=None, index=None, link_threshold: float = 0.8) -> KnowledgeGraph:
        nodes: dict[str, KGNode] = {}
        edges: list[KGEdge] = []

        def node(id_, type_, label, **attrs):
            if id_ not in nodes:
                nodes[id_] = KGNode(id=id_, type=type_, label=label, attrs=attrs)
            return nodes[id_]

        per_doc_entities: dict[str, dict[tuple[str, str], list]] = {}
        for d in documents:
            pid = f"paper:{d.doc_id}"
            node(pid, "paper", d.title, doc_id=d.doc_id, year=d.year, evidence=[{"source": f"{d.source} ingestion", "doc_id": d.doc_id}])
            for a in d.authors:
                aid = f"author:{_norm(a)}"
                node(aid, "author", a, evidence=[{"source": "PDF metadata", "doc_id": d.doc_id}])
                edges.append(KGEdge(source=pid, target=aid, type="authored_by", attrs={"evidence": [{"source": "PDF metadata"}]}))
            mentions = [m for c in chunk_document(d) for m in extract_from_chunk(c)]
            ents: dict[tuple[str, str], list] = defaultdict(list)
            for m in mentions:
                if m.kind in ENTITY_EDGE:
                    ents[(m.kind, _norm(m.text))].append(m)
            tm = title_method(d.title)
            if tm:
                ents[("method", _norm(tm))].append(type("M", (), dict(kind="method", text=tm, evidence=lambda self=None, d=d: {
                    "doc_id": d.doc_id, "chunk_id": None, "section": "Title", "page": 1, "quote": d.title}))())
            for kind_key in [k for k in ents if k[0] == "method" and ("dataset", k[1]) in ents]:
                # the same name used as dataset and method in one paper: keep the more frequent reading
                if len(ents[("dataset", kind_key[1])]) >= len(ents[kind_key]):
                    del ents[kind_key]
            per_doc_entities[d.doc_id] = ents
            for (kind, key), ms in ents.items():
                if len(ms) < self.min_mentions and not (kind == "metric" or (kind == "method" and tm and key == _norm(tm))):
                    continue
                surface = Counter(m.text for m in ms).most_common(1)[0][0]
                eid = f"entity:{kind}:{key}"
                node(eid, "entity", surface, entity_type=kind)
                nodes[eid].attrs.setdefault("papers", [])
                if d.doc_id not in nodes[eid].attrs["papers"]:
                    nodes[eid].attrs["papers"].append(d.doc_id)
                edges.append(KGEdge(source=pid, target=eid, type=ENTITY_EDGE[kind], weight=float(len(ms)),
                                    attrs={"evidence": [m.evidence() for m in ms[: self.k_ev]], "mentions": len(ms)}))
            n_f = n_l = 0
            seen: set[str] = set()
            for m in mentions:
                if m.text in seen:
                    continue
                if m.kind == "finding" and n_f < self.max_findings:
                    seen.add(m.text); n_f += 1
                    fid = f"finding:{d.doc_id}:{n_f}"
                    node(fid, "finding", m.text[:140] + ("…" if len(m.text) > 140 else ""), full_text=m.text, doc_id=d.doc_id, evidence=[m.evidence()])
                    edges.append(KGEdge(source=pid, target=fid, type="reports_finding", attrs={"evidence": [m.evidence()]}))
                elif m.kind == "limitation" and n_l < self.max_limitations:
                    seen.add(m.text); n_l += 1
                    lid = f"limitation:{d.doc_id}:{n_l}"
                    node(lid, "limitation", m.text[:140] + ("…" if len(m.text) > 140 else ""), full_text=m.text, doc_id=d.doc_id, evidence=[m.evidence()])
                    edges.append(KGEdge(source=pid, target=lid, type="has_limitation", attrs={"evidence": [m.evidence()]}))

        edges += self._citations(documents)
        edges += self._similar(documents)
        edges += self._shared(nodes)
        if verifier is not None and index is not None:
            edges += self._link_findings(nodes, documents, verifier, index, link_threshold)
        return KnowledgeGraph(nodes=list(nodes.values()), edges=edges)

    # ------------------------------------------------------------------ cross-paper relations
    def _citations(self, documents) -> list[KGEdge]:
        out = []
        for a in documents:
            for b in documents:
                if a.doc_id == b.doc_id:
                    continue
                tb = {w for w in _norm(b.title).split() if len(w) > 2}
                aid = b.doc_id.split(":")[-1]
                for r in a.references:
                    rn = _norm(r.raw)
                    hit = (len(tb) >= 3 and sum(w in rn.split() for w in tb) / len(tb) >= 0.85) or (
                        re.fullmatch(r"\d{4}\.\d{4,5}", aid) is not None and aid in r.raw)
                    if hit:
                        out.append(KGEdge(source=f"paper:{a.doc_id}", target=f"paper:{b.doc_id}", type="cites",
                                          attrs={"evidence": [{"doc_id": a.doc_id, "section": "References", "quote": r.raw[:300]}],
                                                 "method": "reference string contains the cited paper's title/arXiv id"}))
                        break
        return out

    def _similar(self, documents) -> list[KGEdge]:
        if len(documents) < 2:
            return []
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [f"{d.title}. {d.abstract}" for d in documents]
        X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
        S = (X @ X.T).toarray()
        out = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                if S[i, j] >= self.sim_threshold:
                    out.append(KGEdge(source=f"paper:{documents[i].doc_id}", target=f"paper:{documents[j].doc_id}",
                                      type="similar_problem", weight=float(S[i, j]),
                                      attrs={"method": "TF-IDF cosine of title+abstract", "cosine": round(float(S[i, j]), 3),
                                             "evidence": [{"source": "title+abstract similarity"}]}))
        return out

    def _shared(self, nodes) -> list[KGEdge]:
        pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
        for n in nodes.values():
            if n.type == "entity" and n.attrs.get("entity_type") in ("method", "dataset"):
                ps = sorted(n.attrs.get("papers", []))
                for i in range(len(ps)):
                    for j in range(i + 1, len(ps)):
                        pairs[(ps[i], ps[j])].append(n.label)
        return [KGEdge(source=f"paper:{a}", target=f"paper:{b}", type="shares_entity", weight=float(len(l)),
                       attrs={"shared": sorted(l), "evidence": [{"source": "shared extracted entities"}]})
                for (a, b), l in pairs.items()]

    def _link_findings(self, nodes, documents, verifier, index, threshold, min_coverage: float = 0.4) -> list[KGEdge]:
        """Check each finding against OTHER papers' passages with NLI.

        An NLI model asked about an unrelated passage will often answer "contradiction" with high confidence
        (observed on real papers: 9/9 raw edges were off-topic). So an edge needs ALL of: topical relevance
        (>= `min_coverage` of the finding's content words occur in the passage), at least one shared named term
        (capitalised token, e.g. a method/dataset name), and a confident SUPPORTS/CONTRADICTS verdict.
        Surviving edges are still only candidates for human review (attrs["status"])."""
        if getattr(verifier, "is_mock", False):
            return []                                           # a mock must never create scientific relations
        from researchlens.retrieval.lexical import tokenize
        out = []
        for n in nodes.values():
            if n.type != "finding":
                continue
            src, text = n.attrs["doc_id"], n.attrs["full_text"]
            f_tok = set(tokenize(text))
            f_names = {w.strip(".,;:()") for w in text.split() if w[:1].isupper() and len(w) > 2 and not text.startswith(w)}
            for other in (d.doc_id for d in documents if d.doc_id != src):
                ps = [p for p in index.search(text, k=3, doc_ids=[other])
                      if f_tok and len(f_tok & set(tokenize(p.chunk.text))) / len(f_tok) >= min_coverage
                      and any(nm in p.chunk.text for nm in f_names)]
                if not ps:
                    continue
                v = verifier.verify(Claim(claim_id=n.id, text=text, source="document", doc_id=src), ps)
                if v.label in ("SUPPORTS", "CONTRADICTS") and v.confidence >= threshold:
                    best = max(v.evidence, key=lambda e: e.probs[v.label])
                    chunk = next(p.chunk for p in ps if p.chunk.chunk_id == best.chunk_id)
                    out.append(KGEdge(source=n.id, target=f"paper:{other}", type="supports" if v.label == "SUPPORTS" else "contradicts",
                                      weight=v.confidence,
                                      attrs={"verified_by": v.model, "confidence": round(v.confidence, 3), "support": v.support,
                                             "status": "candidate: NLI-flagged, needs human review",
                                             "evidence": [{"doc_id": other, "chunk_id": chunk.chunk_id, "section": chunk.section,
                                                           "page": chunk.page, "quote": chunk.text[:400]}]}))
        return out


def to_networkx(kg: KnowledgeGraph):
    import networkx as nx
    g = nx.MultiDiGraph()
    for n in kg.nodes:
        g.add_node(n.id, type=n.type, label=n.label, **{k: v for k, v in n.attrs.items() if k != "evidence"})
    for e in kg.edges:
        g.add_edge(e.source, e.target, key=f"{e.type}:{len(g.edges)}", type=e.type, weight=e.weight)
    return g
