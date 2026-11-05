"""Build retrievers from short spec strings, shared by experiments and the pipeline builder.

Specs: bm25 | tfidf | lsa | e5 | bge | scibert | specter2 | rrf:a+b[+c] | linear:lex+dense
Base systems are fitted once and shared by every hybrid that references them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .embedding_cache import EmbeddingCache
from .fusion import LinearFusionRetriever, RRFRetriever
from .lexical import BM25Retriever, TfidfRetriever

DENSE_NAMES = {"lsa", "e5", "bge", "scibert", "specter2"}
NEURAL_NAMES = {"e5", "bge", "scibert", "specter2"}


@dataclass
class RetrievalConfig:
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    lsa_dim: int = 256
    rrf_k: int = 60
    depth: int = 100
    alpha: float = 0.5
    batch_size: int = 32
    cache_dir: str | None = None
    extra: dict = field(default_factory=dict)


def make_base(name: str, cfg: RetrievalConfig, cache: EmbeddingCache | None = None):
    from . import dense as D
    if name == "bm25":
        return BM25Retriever(k1=cfg.bm25_k1, b=cfg.bm25_b)
    if name == "tfidf":
        return TfidfRetriever()
    if name == "lsa":
        return D.DenseRetriever(D.LSAEncoder(n_components=cfg.lsa_dim), name="lsa")
    if name == "e5":
        enc = D.SentenceTransformerEncoder("intfloat/e5-base-v2", "query: ", "passage: ", batch_size=cfg.batch_size)
    elif name == "bge":
        enc = D.SentenceTransformerEncoder("BAAI/bge-base-en-v1.5",
                                           "Represent this sentence for searching relevant passages: ",
                                           batch_size=cfg.batch_size)
    elif name == "scibert":
        enc = D.MeanPoolEncoder(batch_size=cfg.batch_size)
    elif name == "specter2":
        enc = D.Specter2Encoder(batch_size=cfg.batch_size)
    else:
        raise ValueError(f"unknown retriever spec {name!r}")
    return D.DenseRetriever(enc, name=name, cache=cache)


def components(spec: str) -> list[str]:
    if spec.startswith("rrf:"):
        return spec[4:].split("+")
    if spec.startswith("linear:"):
        return spec[7:].split("+")
    return [spec]


def build_systems(specs, docs, cfg: RetrievalConfig, log=print):
    cache = EmbeddingCache(cfg.cache_dir)
    base, timings, meta = {}, {}, {}

    def get(n):
        if n not in base:
            t = time.time()
            base[n] = make_base(n, cfg, cache).fit(docs)
            timings[n] = round(time.time() - t, 2)
            enc = getattr(base[n], "encoder", None)
            meta[n] = getattr(enc, "meta", {"model": n})
            log(f"  fitted {n} in {timings[n]:.1f}s")
        return base[n]

    systems = {}
    for spec in specs:
        if spec.startswith("rrf:"):
            systems[spec] = RRFRetriever([get(p) for p in components(spec)], k=cfg.rrf_k, depth=cfg.depth,
                                         name=spec, fit_components=False).fit(docs)
        elif spec.startswith("linear:"):
            lex, den = components(spec)
            r = LinearFusionRetriever(get(lex), get(den), alpha=cfg.alpha, fit_components=False).fit(docs)
            r.name = spec
            systems[spec] = r
        else:
            systems[spec] = get(spec)
    return systems, base, {"fit_seconds": timings, "models": meta,
                           "embedding_cache": {"hits": cache.hits, "misses": cache.misses}}


def prime_queries(base: dict, queries) -> None:
    """Batch-encode all queries once for every dense base system."""
    for r in base.values():
        if hasattr(r, "prime_queries"):
            r.prime_queries(list(queries))
