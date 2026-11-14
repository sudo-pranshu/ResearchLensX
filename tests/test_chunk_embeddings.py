import numpy as np

from researchlens.retrieval.chunk_embeddings import ChunkEmbeddingCache
from researchlens.retrieval.dense import DenseRetriever


class CountingEncoder:
    """Deterministic encoder: a text's vector depends only on the text, so cached and fresh vectors are comparable."""
    cache_key = "fake|v1"
    meta = {"model": "fake"}

    def __init__(self):
        self.seen: list[str] = []

    def _vec(self, t):
        rng = np.random.default_rng(abs(hash(t)) % (2 ** 32))
        return rng.normal(size=8).astype(np.float32)

    def encode_docs(self, texts):
        self.seen += list(texts); return np.stack([self._vec(t) for t in texts])

    def encode_queries(self, texts):
        return np.stack([self._vec("q:" + t) for t in texts])


def test_only_new_texts_are_embedded_and_order_is_preserved():
    c, enc = ChunkEmbeddingCache(), CountingEncoder()
    a = c.get_or_compute("k", "doc", ["alpha", "beta", "gamma"], enc.encode_docs)
    assert enc.seen == ["alpha", "beta", "gamma"]
    b = c.get_or_compute("k", "doc", ["beta", "delta", "alpha"], enc.encode_docs)         # one new text, reordered
    assert enc.seen[3:] == ["delta"] and c.compute_calls == [3, 1]
    assert np.allclose(b[0], a[1]) and np.allclose(b[2], a[0])
    assert (c.hits, c.misses) == (2, 4)


def test_duplicate_texts_in_one_call_are_embedded_once():
    c, enc = ChunkEmbeddingCache(), CountingEncoder()
    out = c.get_or_compute("k", "doc", ["same", "same", "other"], enc.encode_docs)
    assert enc.seen == ["same", "other"] and np.allclose(out[0], out[1])


def test_spaces_do_not_mix_encoders_or_sides():
    c, enc = ChunkEmbeddingCache(), CountingEncoder()
    c.get_or_compute("k1", "doc", ["t"], enc.encode_docs)
    c.get_or_compute("k2", "doc", ["t"], enc.encode_docs)
    c.get_or_compute("k1", "query", ["t"], enc.encode_queries)
    assert c.count() == 3 and enc.seen == ["t", "t"]


def test_gc_removes_unneeded_vectors_but_keeps_shared_ones_and_trims_queries():
    c, enc = ChunkEmbeddingCache(), CountingEncoder()
    c.get_or_compute("k", "doc", ["shared", "only-b", "only-c"], enc.encode_docs)
    removed = c.gc(live=[("k", "shared"), ("k", "only-c")])                    # document B deleted
    assert removed == 1 and c.count("k|doc") == 2
    again = c.get_or_compute("k", "doc", ["shared"], enc.encode_docs)
    assert enc.seen == ["shared", "only-b", "only-c"] and again.shape == (1, 8)    # shared vector survived, nothing recomputed
    c.get_or_compute("k", "query", [f"q{i}" for i in range(10)], enc.encode_queries)
    assert c.gc(live=[("k", "shared"), ("k", "only-c")], max_query_rows=4) == 6 and c.count("k|query") == 4


def test_persists_across_connections(tmp_path):
    p = tmp_path / "e.sqlite"
    enc = CountingEncoder()
    ChunkEmbeddingCache(p).get_or_compute("k", "doc", ["x", "y"], enc.encode_docs)
    c2 = ChunkEmbeddingCache(p)
    c2.get_or_compute("k", "doc", ["x", "y"], enc.encode_docs)
    assert enc.seen == ["x", "y"] and c2.hits == 2


def test_dense_retriever_adding_a_paper_embeds_only_its_chunks():
    cache, enc = ChunkEmbeddingCache(), CountingEncoder()
    old = [f"paper A chunk {i}" for i in range(5)] + [f"paper B chunk {i}" for i in range(5)]
    DenseRetriever(enc, name="d", cache=cache, use_faiss=False).fit(old)
    assert len(enc.seen) == 10
    new = old + [f"paper C chunk {i}" for i in range(3)]
    r = DenseRetriever(enc, name="d", cache=cache, use_faiss=False).fit(new)
    assert len(enc.seen) == 13 and cache.compute_calls == [10, 3]                 # only paper C's 3 chunks were embedded
    fresh = DenseRetriever(CountingEncoder(), name="d", cache=ChunkEmbeddingCache(), use_faiss=False).fit(new)
    assert np.allclose(r.emb, fresh.emb)                                          # identical to embedding everything from scratch
