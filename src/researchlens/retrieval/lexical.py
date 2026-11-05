"""Lexical retrievers: BM25 (implemented from scratch) and TF-IDF cosine."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

from .base import Retriever

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, keep alphanumerics and hyphenated terms (e.g. 'bert-base', 'f1')."""
    toks = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in ENGLISH_STOP_WORDS]
    return toks


class BM25Retriever(Retriever):
    """Okapi BM25 with Lucene-style non-negative IDF.

    score(q, d) = sum_{t in q} idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))

    The per-(doc, term) weight is precomputed into a sparse matrix, so scoring a
    query is one sparse column-slice + row-sum.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75, remove_stopwords: bool = True):
        self.k1, self.b, self.remove_stopwords = k1, b, remove_stopwords

    def fit(self, docs: Sequence[str]) -> "BM25Retriever":
        self._vec = CountVectorizer(
            tokenizer=lambda s: tokenize(s, self.remove_stopwords),
            lowercase=False, token_pattern=None,
        )
        tf = self._vec.fit_transform(docs).tocsr().astype(np.float64)  # (N, V)
        n_docs = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n_docs else 0.0
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))

        # Row-wise length normalisation applied to the non-zeros only.
        norm = self.k1 * (1.0 - self.b + self.b * dl / max(avgdl, 1e-9))
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        data = tf.data * (self.k1 + 1.0) / (tf.data + norm[rows])
        data *= self.idf[tf.indices]
        # Built as CSR (same sparsity as tf), stored as CSC for fast column slicing by query term.
        self._W = sparse.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape).tocsc()
        self.n_docs = n_docs
        return self

    def score(self, query: str) -> np.ndarray:
        vocab = self._vec.vocabulary_
        cols = [vocab[t] for t in tokenize(query, self.remove_stopwords) if t in vocab]
        if not cols:
            return np.zeros(self.n_docs)
        # Repeated query terms count once per occurrence (standard BM25 with qtf=raw count).
        return np.asarray(self._W[:, cols].sum(axis=1)).ravel()


class TfidfRetriever(Retriever):
    """TF-IDF (sublinear tf, L2-normalised) with cosine similarity."""

    name = "tfidf"

    def __init__(self, remove_stopwords: bool = True, ngram_range: tuple[int, int] = (1, 1)):
        self.remove_stopwords, self.ngram_range = remove_stopwords, ngram_range

    def fit(self, docs: Sequence[str]) -> "TfidfRetriever":
        self._vec = TfidfVectorizer(
            tokenizer=lambda s: tokenize(s, self.remove_stopwords),
            lowercase=False, token_pattern=None, sublinear_tf=True,
            ngram_range=self.ngram_range,
        )
        self._X = self._vec.fit_transform(docs).tocsr()
        self.n_docs = self._X.shape[0]
        return self

    def score(self, query: str) -> np.ndarray:
        q = self._vec.transform([query])
        return np.asarray((self._X @ q.T).todense()).ravel()
