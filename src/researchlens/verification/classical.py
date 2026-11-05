"""Classical verifier baseline: hand-built lexical features + Logistic Regression (3-way).
Owner: Pawan (initial version: Pranshu, sprint). A supporting syllabus component, not a contribution.

Features for (claim, premise sentences):
  tf-idf cosine to the whole premise and to the best sentence, token Jaccard with the best
  sentence, number overlap, negation cues and increase/decrease polarity cues in claim and
  best sentence, and a polarity-mismatch indicator.
"""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from researchlens.contracts import VERDICTS
from researchlens.retrieval.lexical import tokenize

NEG = {"not", "no", "never", "without", "neither", "nor", "none", "cannot", "lack", "lacks", "absence", "fail", "fails"}
UP = {"increase", "increases", "increased", "higher", "enhance", "enhances", "promote", "promotes", "improve",
      "improves", "elevated", "greater", "more", "upregulates", "activates", "induces"}
DOWN = {"decrease", "decreases", "decreased", "lower", "reduce", "reduces", "reduced", "inhibit", "inhibits",
        "suppress", "suppresses", "less", "fewer", "downregulates", "impairs", "prevents"}
_NUM = re.compile(r"\d+(?:\.\d+)?")


def _raw_tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class LRVerifierBaseline:
    name = "tfidf-features+logreg"

    def __init__(self, C: float = 1.0, seed: int = 0):
        self.C, self.seed = C, seed

    def fit_vocab(self, corpus_texts: Sequence[str]) -> "LRVerifierBaseline":
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        self.tfidf.fit(corpus_texts)
        return self

    def features(self, claim: str, sentences: Sequence[str]) -> np.ndarray:
        sentences = list(sentences) or [""]
        V = self.tfidf.transform([claim, " ".join(sentences)] + sentences)
        c, whole, sents = V[0], V[1], V[2:]
        sims = (sents @ c.T).toarray().ravel()
        b = int(np.argmax(sims))
        best = sentences[b]
        ct, bt = set(_raw_tokens(claim)), set(_raw_tokens(best))
        cn, bn = set(_NUM.findall(claim)), set(_NUM.findall(best))
        c_up, c_dn = len(ct & UP), len(ct & DOWN)
        b_up, b_dn = len(bt & UP), len(bt & DOWN)
        mismatch = float((c_up > 0 and b_dn > 0 and b_up == 0) or (c_dn > 0 and b_up > 0 and b_dn == 0))
        return np.array([
            float((whole @ c.T).toarray()[0, 0]), float(sims[b]), float(np.sort(sims)[-2] if len(sims) > 1 else 0),
            len(ct & bt) / max(len(ct | bt), 1), len(cn & bn) / max(len(cn), 1) if cn else 0.0, float(bool(cn)),
            len(ct & NEG), len(bt & NEG), float((len(ct & NEG) > 0) != (len(bt & NEG) > 0)),
            c_up, c_dn, b_up, b_dn, mismatch,
        ], dtype=float)

    def fit(self, X: np.ndarray, y: Sequence[str]) -> "LRVerifierBaseline":
        self.clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000, random_state=self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.clf.fit((X - self.mu) / self.sd, list(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        P = self.clf.predict_proba((X - self.mu) / self.sd)
        cols = [list(self.clf.classes_).index(v) for v in VERDICTS]
        return P[:, cols]

    def predict(self, X: np.ndarray) -> list[str]:
        return [VERDICTS[i] for i in self.predict_proba(X).argmax(1)]
