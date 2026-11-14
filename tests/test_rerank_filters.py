"""Reranker, redundancy suppression and metadata filters. Offline: the cross-encoder is replaced by a fake scorer."""
import numpy as np

from researchlens.contracts import Chunk, RetrievedPassage
from researchlens.retrieval.base import Retriever
from researchlens.retrieval.index import ChunkIndex
from researchlens.retrieval.lexical import BM25Retriever
from researchlens.retrieval.postprocess import section_filter, suppress_redundant
from researchlens.retrieval.registry import components
from researchlens.retrieval.rerank import RerankRetriever


class FakeScorer:
    """Scores a passage by how many times 'gold' appears: a stand-in for a cross-encoder."""
    def __init__(self): self.calls = []
    def scores(self, query, texts):
        self.calls.append(len(texts))
        return np.array([t.count("gold") for t in texts], dtype=float)


DOCS = ["alpha beta retrieval baseline text", "retrieval retrieval retrieval systems gold gold",
        "retrieval of scientific papers gold", "unrelated cooking recipe pasta", "retrieval evaluation metrics"]


def test_rerank_reorders_first_stage_and_is_deterministic():
    first = BM25Retriever().fit(DOCS)
    rr = RerankRetriever(first, FakeScorer(), depth=4, fit_first_stage=False).fit(DOCS)
    hits = rr.search("retrieval", k=3)
    ids = [i for i, _ in hits]
    assert ids[0] == 1 and ids[1] == 2                      # most 'gold' first
    assert hits == rr.search("retrieval", k=3)


def test_rerank_depth_limits_scored_pairs_and_respects_candidates():
    sc = FakeScorer()
    rr = RerankRetriever(BM25Retriever(), sc, depth=2).fit(DOCS)
    rr.search("retrieval", k=1)
    assert sc.calls == [2]
    hits = rr.search("retrieval", k=5, candidates=np.array([0, 3, 4]))
    assert {i for i, _ in hits} <= {0, 3, 4}


def test_rerank_empty_candidates():
    rr = RerankRetriever(BM25Retriever(), FakeScorer()).fit(DOCS)
    assert rr.search("retrieval", k=3, candidates=np.array([], dtype=int)) == []


def mk(i, text, section="Intro", doc="d"):
    return RetrievedPassage(chunk=Chunk(chunk_id=f"{doc}::{i}", doc_id=doc, section=section, text=text, order=i),
                            score=1.0 / (i + 1), rank=i + 1, retriever="t")


def test_suppress_redundant_exact_contained_and_overlap():
    long_ = "we evaluate dense retrieval on scientific questions using paragraph level evidence and report recall"
    ps = [mk(0, long_), mk(1, long_), mk(2, "evaluate dense retrieval scientific questions paragraph evidence"),
          mk(3, "completely different content about knowledge graphs and entity extraction methods"),
          mk(4, "short")]
    out = suppress_redundant(ps, threshold=0.85)
    assert [p.chunk.chunk_id for p in out] == ["d::0", "d::3", "d::4"]       # exact dup and contained passage dropped
    assert [p.rank for p in out] == [1, 2, 3]                                 # re-ranked; short passage kept
    assert len(suppress_redundant(ps, k=1)) == 1


def test_suppress_keeps_windows_with_small_overlap():
    a = " ".join(f"w{i}" for i in range(100)); b = " ".join(f"w{i}" for i in range(90, 190))
    assert len(suppress_redundant([mk(0, a), mk(1, b)], 0.85)) == 2


def chunks():
    rows = [("A", "Introduction", "retrieval methods for papers"), ("A", "Related Work", "retrieval prior art survey"),
            ("B", "Method", "retrieval pipeline details"), ("B", "References", "retrieval citations list")]
    return [Chunk(chunk_id=f"{d}::{i}", doc_id=d, section=s, text=t, order=i) for i, (d, s, t) in enumerate(rows)]


def test_chunk_index_where_applies_before_ranking():
    idx = ChunkIndex(BM25Retriever(), chunks())
    flt = section_filter(exclude=["related work", "references"])
    res = idx.search("retrieval", k=4, where=flt)
    assert {p.chunk.section for p in res} == {"Introduction", "Method"} and len(res) == 2
    assert [p.chunk.section for p in idx.search("retrieval", k=4, where=section_filter(include=["method"]))] == ["Method"]
    assert idx.search("retrieval", k=4, doc_ids=["A"], where=section_filter(include=["method"])) == []


def test_chunk_index_dedup_option():
    long_ = "retrieval methods for scientific papers using lexical dense signals evaluated carefully"
    cs = chunks() + [Chunk(chunk_id="C::8", doc_id="C", section="Intro", text=long_, order=8),
                     Chunk(chunk_id="C::9", doc_id="D", section="Intro", text=long_, order=9)]
    plain = ChunkIndex(BM25Retriever(), cs).search("retrieval methods papers", k=6)
    dd = ChunkIndex(BM25Retriever(), cs, dedup_threshold=0.85).search("retrieval methods papers", k=6)
    assert len(dd) == len(plain) - 1


def test_registry_rerank_spec_components():
    assert components("rerank:rrf:bm25+e5") == ["bm25", "e5"] and components("rerank:bm25") == ["bm25"]
