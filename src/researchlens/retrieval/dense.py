"""Dense retrieval with pluggable encoders, persistent embedding cache and FAISS.

Encoders (all expose encode_docs / encode_queries / cache_key / meta):
  * LSAEncoder                 - TF-IDF -> truncated SVD (classical; no downloads)
  * SentenceTransformerEncoder - e5 / bge (general-purpose dense baseline)
  * MeanPoolEncoder            - mean-pooled HF encoder, e.g. SciBERT (diagnostic control)
  * Specter2Encoder            - SPECTER2 base + proximity (docs) / adhoc_query (queries) adapters

Transformer libraries are imported lazily so the core package and CI stay CPU/offline.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .base import Retriever
from .embedding_cache import EmbeddingCache


class Encoder(Protocol):
    cache_key: str
    meta: dict
    def encode_docs(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


class DenseRetriever(Retriever):
    """Cosine similarity over L2-normalised embeddings.

    Full-corpus search uses FAISS IndexFlatIP (exact) when faiss is installed, else NumPy.
    Candidate-restricted search (QASPER within-paper) uses NumPy on the candidate rows.
    Query embeddings are memoised; call `prime_queries` to batch-encode them up front.
    """

    def __init__(self, encoder, name: str | None = None, cache: EmbeddingCache | None = None,
                 use_faiss: bool | None = None):
        self.encoder = encoder
        self.name = name or type(encoder).__name__
        self.cache = cache
        self.use_faiss = _faiss_available() if use_faiss is None else use_faiss
        self._q: dict[str, np.ndarray] = {}

    def _encode(self, side: str, texts: Sequence[str]) -> np.ndarray:
        fn = self.encoder.encode_docs if side == "doc" else self.encoder.encode_queries
        if self.cache is None or not getattr(self.encoder, "cache_key", None):
            return np.asarray(fn(texts), dtype=np.float32)
        return self.cache.get_or_compute(self.encoder.cache_key, side, list(texts), fn)

    def fit(self, docs: Sequence[str]) -> "DenseRetriever":
        if hasattr(self.encoder, "fit"):
            self.encoder.fit(docs)
        self.emb = l2norm(self._encode("doc", docs))
        self.n_docs = len(docs)
        self._index = None
        if self.use_faiss:
            import faiss
            self._index = faiss.IndexFlatIP(self.emb.shape[1])
            self._index.add(self.emb)
        return self

    def prime_queries(self, queries: Sequence[str]) -> None:
        todo = list(dict.fromkeys(q for q in queries if q not in self._q))
        if todo:
            for q, e in zip(todo, l2norm(self._encode("query", todo))):
                self._q[q] = e

    def _qvec(self, query: str) -> np.ndarray:
        if query not in self._q:
            self.prime_queries([query])
        return self._q[query]

    def score(self, query: str) -> np.ndarray:
        return self.emb @ self._qvec(query)

    def search(self, query, k=10, candidates=None):
        if self._index is not None and candidates is None:
            D, I = self._index.search(self._qvec(query)[None, :], min(k, self.n_docs))
            return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]
        return super().search(query, k, candidates)


# ---------------------------------------------------------------- encoders
class LSAEncoder:
    def __init__(self, n_components: int = 256, seed: int = 0):
        self.n_components, self.seed = n_components, seed
        self.cache_key = ""          # corpus-fitted: never cached
        self.meta = {"model": f"lsa-{n_components}"}

    def fit(self, docs):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from .lexical import tokenize
        self._tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        X = self._tfidf.fit_transform(docs)
        n = max(2, min(self.n_components, X.shape[1] - 1, X.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=n, random_state=self.seed).fit(X)
        return self

    def encode_docs(self, texts):
        return self._svd.transform(self._tfidf.transform(texts))

    encode_queries = encode_docs


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _revision(config) -> str:
    return getattr(config, "_commit_hash", None) or "unknown"


def _hf_batches(texts, tok, model, device, batch_size, pool, max_length=512):
    """Length-sorted batching (less padding), restored to input order."""
    import torch
    order = np.argsort([-len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for s in range(0, len(texts), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        with torch.inference_mode():
            h = model(**enc).last_hidden_state
        if pool == "cls":
            v = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out[idx] = v.float().cpu().numpy()
    return out


class SentenceTransformerEncoder:
    """'intfloat/e5-base-v2' needs query_prefix='query: ', doc_prefix='passage: '."""

    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = "", batch_size: int = 32,
                 max_seq_length: int = 512, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.device = device or pick_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length
        self.qp, self.dp, self.bs = query_prefix, doc_prefix, batch_size
        rev = _revision(self.model[0].auto_model.config)
        self.meta = {"model": model_name, "revision": rev, "query_prefix": query_prefix,
                     "doc_prefix": doc_prefix, "max_seq_length": max_seq_length, "pooling": "model-default"}
        self.cache_key = f"st|{model_name}|{rev}|{query_prefix}|{doc_prefix}|{max_seq_length}"

    def encode_docs(self, texts):
        return self.model.encode([self.dp + t for t in texts], batch_size=self.bs,
                                 normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)

    def encode_queries(self, texts):
        return self.model.encode([self.qp + t for t in texts], batch_size=self.bs,
                                 normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)


class MeanPoolEncoder:
    """Mean-pooled HF encoder. With SciBERT: domain-pretrained but NOT retrieval-trained (control)."""

    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased", batch_size: int = 32, max_length: int = 512):
        from transformers import AutoModel, AutoTokenizer
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.bs, self.max_length = batch_size, max_length
        rev = _revision(self.model.config)
        self.meta = {"model": model_name, "revision": rev, "pooling": "mean", "max_length": max_length}
        self.cache_key = f"mean|{model_name}|{rev}|{max_length}"

    def encode_docs(self, texts):
        return _hf_batches(list(texts), self.tok, self.model, self.device, self.bs, "mean", self.max_length)

    encode_queries = encode_docs


class Specter2Encoder:
    """SPECTER2 base + adapters (`pip install adapters`). Documents: 'allenai/specter2' (proximity);
    queries: 'allenai/specter2_adhoc_query', as recommended on the SPECTER2 model card for search. [CLS] pooling."""

    def __init__(self, batch_size: int = 32, max_length: int = 512):
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
        self.model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query")
        self.model.to(self.device).eval()
        self.bs, self.max_length = batch_size, max_length
        rev = _revision(self.model.config)
        self.meta = {"model": "allenai/specter2_base", "revision": rev,
                     "adapters": {"doc": "allenai/specter2", "query": "allenai/specter2_adhoc_query"},
                     "pooling": "cls", "max_length": max_length}
        self.cache_key = f"specter2|{rev}|{max_length}"

    def _enc(self, texts, adapter):
        self.model.set_active_adapters(adapter)
        return _hf_batches(list(texts), self.tok, self.model, self.device, self.bs, "cls", self.max_length)

    def encode_docs(self, texts):
        return self._enc(texts, "proximity")

    def encode_queries(self, texts):
        return self._enc(texts, "adhoc_query")
