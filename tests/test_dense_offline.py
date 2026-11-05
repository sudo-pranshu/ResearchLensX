"""Dense-retrieval machinery tested offline with a deterministic fake encoder (no downloads)."""
import numpy as np
import pytest

from researchlens.evaluation.reporting import summarise, write_run
from researchlens.retrieval.dense import DenseRetriever, l2norm
from researchlens.retrieval.embedding_cache import EmbeddingCache
from researchlens.retrieval.registry import RetrievalConfig, build_systems, components

DOCS = ["alpha beta", "gamma delta", "alpha gamma", "epsilon"]


class FakeEncoder:
    """Bag-of-letters embedding; counts calls so caching can be asserted."""
    cache_key = "fake|v1"
    meta = {"model": "fake"}

    def __init__(self):
        self.calls = 0

    def _e(self, texts):
        self.calls += 1
        return np.array([[t.count(c) for c in "abcdeglmpst"] for t in texts], dtype=float) + 1e-3

    encode_docs = _e
    encode_queries = _e


def test_cache_reuses_and_keys_on_text(tmp_path):
    cache = EmbeddingCache(tmp_path)
    enc = FakeEncoder()
    r1 = DenseRetriever(enc, cache=cache, use_faiss=False).fit(DOCS)
    r2 = DenseRetriever(enc, cache=cache, use_faiss=False).fit(DOCS)
    assert enc.calls == 1 and cache.hits == 1 and np.allclose(r1.emb, r2.emb)
    DenseRetriever(enc, cache=cache, use_faiss=False).fit(DOCS + ["new"])
    assert enc.calls == 2                                   # different corpus -> new key


def test_embeddings_normalised_and_query_memo():
    enc = FakeEncoder()
    r = DenseRetriever(enc, use_faiss=False).fit(DOCS)
    assert np.allclose(np.linalg.norm(r.emb, axis=1), 1.0)
    r.prime_queries(["alpha", "alpha", "gamma"])
    calls = enc.calls
    r.search("alpha", 2); r.search("gamma", 2)
    assert enc.calls == calls                               # memoised, no re-encoding


def test_faiss_matches_numpy():
    pytest.importorskip("faiss")
    a = DenseRetriever(FakeEncoder(), use_faiss=True).fit(DOCS)
    b = DenseRetriever(FakeEncoder(), use_faiss=False).fit(DOCS)
    fa, fb = a.search("alpha gamma", 3), b.search("alpha gamma", 3)
    assert [d for d, _ in fa] == [d for d, _ in fb]
    assert np.allclose([s for _, s in fa], [s for _, s in fb], atol=1e-5)
    # candidate-restricted search bypasses FAISS and respects the filter
    assert {d for d, _ in a.search("alpha", 4, candidates=np.array([1, 3]))} <= {1, 3}


def test_registry_builds_hybrids_sharing_bases():
    systems, base, meta = build_systems(["bm25", "lsa", "rrf:bm25+lsa", "linear:bm25+lsa"], DOCS * 3,
                                        RetrievalConfig(lsa_dim=3), log=lambda m: None)
    assert set(base) == {"bm25", "lsa"} and len(systems) == 4
    assert components("rrf:bm25+specter2") == ["bm25", "specter2"]
    assert systems["rrf:bm25+lsa"].retrievers[0] is base["bm25"]


def test_write_run_outputs(tmp_path):
    pq = {"a": [{"qid": "1", "recall@5": 1.0, "top5": [1]}, {"qid": "2", "recall@5": 0.0, "top5": [2]}]}
    assert summarise(pq) == {"a": {"recall@5": 0.5}}
    out = write_run(tmp_path / "run", {"x": 1}, pq, [{"system_a": "a", "delta": 0.1}])
    for f in ["config.json", "summary.json", "summary.csv", "comparisons.json", "comparisons.csv", "per_query.jsonl"]:
        assert (out / f).exists()


def test_release_after_priming():
    r = DenseRetriever(FakeEncoder(), use_faiss=False).fit(DOCS)
    r.prime_queries(["alpha"])
    r.release()
    assert r.search("alpha", 1)
    with pytest.raises(KeyError):
        r.search("never primed", 1)
