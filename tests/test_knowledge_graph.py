"""Knowledge graph: extraction with provenance, cross-paper relations, mock-verifier guard (offline)."""
import numpy as np

from researchlens.contracts import Document, Reference, Section
from researchlens.knowledge_graph.builder import GraphBuilder, to_networkx
from researchlens.knowledge_graph.extract import extract_from_chunk, title_method
from researchlens.contracts import Chunk
from researchlens.mocks import MockVerifier
from researchlens.pipeline import build_default_pipeline
from researchlens.verification.nli import NLIVerifier


def doc(i, title, abstract, paras_by_section, authors=(), refs=()):
    secs = [Section(title=s, paragraphs=ps, pages=list(range(1, len(ps) + 1))) for s, ps in paras_by_section.items()]
    return Document(doc_id=i, title=title, abstract=abstract, authors=list(authors), sections=secs, source="fixture",
                    references=[Reference(raw=r) for r in refs])


A = doc("A", "ColBERTish: Late Interaction Retrieval", "We study late interaction retrieval for passage ranking.",
        {"Experiments": ["We evaluate on the MS MARCO dataset using MRR and Recall@k. The MS MARCO dataset is large. "
                         "Our ColBERTish model outperforms BM25 by 4.1 points on MRR.",
                         "The ColBERTish model uses late interaction over token embeddings."],
         "Limitations": ["However, our approach fails to generalize to out-of-domain corpora and we leave this for future work."]},
        authors=["A. Writer"])
B = doc("B", "A Benchmark of Retrieval Models", "We study retrieval benchmarks for passage ranking and report results.",
        {"Results": ["Results on the MS MARCO dataset show that BM25 remains strong. The MS MARCO dataset is widely used.",
                     "In our replication the ColBERTish model does not outperform BM25 by 4.1 points on MRR."]},
        refs=["Smith. 2020. ColBERTish: Late Interaction Retrieval. In Proceedings."])


def test_extraction_has_provenance_and_is_verbatim():
    ch = Chunk(chunk_id="A::s0::p0::w0", doc_id="A", section="Experiments", order=0, page=3, text=A.sections[0].paragraphs[0])
    ms = extract_from_chunk(ch)
    kinds = {(m.kind, m.text) for m in ms}
    assert ("dataset", "MS MARCO") in kinds and ("metric", "MRR") in kinds and ("metric", "Recall@k") in kinds
    assert ("method", "ColBERTish") in kinds
    f = [m for m in ms if m.kind == "finding"]
    assert f and all(m.quote in ch.text and m.page == 3 and m.chunk_id == ch.chunk_id for m in ms)


def test_title_method():
    assert title_method("BERT: Pre-training of Deep Bidirectional Transformers") == "BERT"
    assert title_method("A study of many things with a long title and no colon") is None


def test_graph_nodes_edges_and_evidence_everywhere():
    kg = GraphBuilder().build([A, B])
    types = {n.type for n in kg.nodes}
    assert {"paper", "author", "entity", "finding", "limitation"} <= types
    ids = {n.id for n in kg.nodes}
    assert all(e.source in ids and e.target in ids for e in kg.edges)
    assert all(e.attrs.get("evidence") for e in kg.edges)                   # no edge without provenance
    et = {(e.type) for e in kg.edges}
    assert {"authored_by", "evaluates_on", "uses_metric", "uses_method", "reports_finding", "has_limitation", "cites",
            "shares_entity", "similar_problem"} <= et
    assert "extends" not in et
    cite = next(e for e in kg.edges if e.type == "cites")
    assert (cite.source, cite.target) == ("paper:B", "paper:A")
    shared = next(e for e in kg.edges if e.type == "shares_entity")
    assert "MS MARCO" in shared.attrs["shared"]


def test_min_mentions_filters_one_off_entities():
    one = doc("C", "Plain Title", "abstract text here", {"Methods": ["We evaluate on the Zork dataset once. Nothing else matters here at all."]})
    kg = GraphBuilder(min_mentions=2).build([one])
    assert not any(n.type == "entity" and n.label == "Zork" for n in kg.nodes)
    kg1 = GraphBuilder(min_mentions=1).build([one])
    assert any(n.label == "Zork" for n in kg1.nodes)


def scorer(prem, hyp):
    # contradicts when the passage is the replication sentence; (wrongly) also "contradicts" everything else,
    # imitating NLI's behaviour on off-topic text, which the relevance gates must filter out
    return np.array([[0.02, 0.95, 0.03] for p in prem])


def test_nli_links_only_with_real_verifier_and_carry_model_and_quote():
    pipe = build_default_pipeline([A, B])
    kg_mock = GraphBuilder().build([A, B], verifier=MockVerifier(), index=pipe.retriever)
    assert not any(e.type in ("supports", "contradicts") for e in kg_mock.edges)      # mock may not create relations
    real = NLIVerifier(scorer, granularity="sentence")
    kg = GraphBuilder().build([A, B], verifier=real, index=pipe.retriever, link_threshold=0.8)
    links = [e for e in kg.edges if e.type in ("supports", "contradicts")]
    assert links and all(e.attrs["verified_by"] and e.attrs["evidence"][0]["quote"] for e in links)
    from_a = [e for e in links if e.source.startswith("finding:A:")]
    assert from_a and all(e.target == "paper:B" and e.type == "contradicts" for e in from_a)   # A's finding checked against B
    assert all("replication" in e.attrs["evidence"][0]["quote"] for e in from_a)       # off-topic passages never produce edges
    assert all("needs human review" in e.attrs["status"] for e in links)


def test_networkx_export():
    g = to_networkx(GraphBuilder().build([A, B]))
    assert g.number_of_nodes() > 5 and "paper:A" in g
