"""Dense retrieval with pluggable encoders.

Encoders:
  * LSAEncoder            - TF-IDF -> truncated SVD (classical, no downloads; B3 baseline)
  * SentenceTransformerEncoder - e5 / bge / any sentence-transformers model (D1)
  * MeanPoolEncoder       - mean-pooled HF encoder, e.g. SciBERT (D2, negative control)
  * Specter2Encoder       - SPECTER2 base + proximity / adhoc_query adapters (D3)

The transformer encoders import their libraries lazily and have NOT been run in
this scaffold yet (they need model downloads). Verify on a few examples first.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .base import Retriever


class Encoder(Protocol):
    def encode_docs(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class DenseRetriever(Retriever):
    def __init__(self, encoder, name: str | None = None, use_faiss: bool = False):
        self.encoder = encoder
        self.name = name or type(encoder).__name__
        self.use_faiss = use_faiss

    def fit(self, docs: Sequence[str]) -> "DenseRetriever":
        if hasattr(self.encoder, "fit"):
            self.encoder.fit(docs)
        self.emb = _l2norm(np.asarray(self.encoder.encode_docs(docs), dtype=np.float32))
        self.n_docs = len(docs)
        self._index = None
        if self.use_faiss:
            import faiss  # optional dependency
            self._index = faiss.IndexFlatIP(self.emb.shape[1])
            self._index.add(self.emb)
        return self

    def score(self, query: str) -> np.ndarray:
        q = _l2norm(np.asarray(self.encoder.encode_queries([query]), dtype=np.float32))[0]
        return self.emb @ q

    def search(self, query, k=10, candidates=None):
        if self._index is not None and candidates is None:
            q = _l2norm(np.asarray(self.encoder.encode_queries([query]), dtype=np.float32))
            D, I = self._index.search(q, k)
            return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]
        return super().search(query, k, candidates)


class LSAEncoder:
    def __init__(self, n_components: int = 256, seed: int = 0):
        self.n_components, self.seed = n_components, seed

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


class SentenceTransformerEncoder:
    """e.g. 'intfloat/e5-base-v2' (query_prefix='query: ', doc_prefix='passage: ')
    or 'BAAI/bge-base-en-v1.5' (query_prefix='Represent this sentence for searching relevant passages: ')."""

    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = "",
                 batch_size: int = 64, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.qp, self.dp, self.bs = query_prefix, doc_prefix, batch_size

    def encode_docs(self, texts):
        return self.model.encode([self.dp + t for t in texts], batch_size=self.bs,
                                 normalize_embeddings=True, show_progress_bar=True)

    def encode_queries(self, texts):
        return self.model.encode([self.qp + t for t in texts], batch_size=self.bs, normalize_embeddings=True)


def _hf_batches(texts, tok, model, device, batch_size, pool, max_length=512):
    import torch
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(list(texts[i:i + batch_size]), padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        if pool == "cls":
            v = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(v.cpu().numpy())
    return np.vstack(out)


def _pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MeanPoolEncoder:
    """Mean-pooled HF encoder. With 'allenai/scibert_scivocab_uncased' this is the
    D2 control: domain-pretrained but not retrieval-trained."""

    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased", batch_size: int = 32):
        from transformers import AutoModel, AutoTokenizer
        self.device = _pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.bs = batch_size

    def encode_docs(self, texts):
        return _hf_batches(texts, self.tok, self.model, self.device, self.bs, "mean")

    encode_queries = encode_docs


class Specter2Encoder:
    """SPECTER2: base model + task adapters (requires `pip install adapters`).

    Documents use the 'proximity' adapter; queries use 'adhoc_query', as recommended
    on the allenai/specter2 model card for ad-hoc search. Pooling is [CLS].
    """

    def __init__(self, batch_size: int = 32):
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer
        self.device = _pick_device()
        self.tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter("allenai/specter2", source="hf", load_as="proximity")
        self.model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query")
        self.model.to(self.device).eval()
        self.bs = batch_size

    def _enc(self, texts, adapter):
        self.model.set_active_adapters(adapter)
        return _hf_batches(texts, self.tok, self.model, self.device, self.bs, "cls")

    def encode_docs(self, texts):
        return self._enc(texts, "proximity")

    def encode_queries(self, texts):
        return self._enc(texts, "adhoc_query")
