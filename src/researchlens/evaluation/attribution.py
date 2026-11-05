"""Answer-level evaluation for H2. Owner: Pawan (initial: Pranshu, sprint).

Citation recall / precision follow ALCE (Gao et al., 2023), with an NLI *judge* that must
differ from the pipeline's verifier:
  recall(sentence)    = 1 if judge(concat(cited passages) entails sentence) else 0 (0 if uncited)
  precision(citation) = 1 if the citation alone entails the sentence, or the full citation set
                        entails it and removing this citation breaks entailment; else 0
Abstention sentences are excluded. Answer-F1 is SQuAD-style token F1, max over annotators.
"""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

from researchlens.contracts import Answer
from researchlens.generation.llm_rag import is_abstention


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    common = Counter(p) & Counter(g)
    tp = sum(common.values())
    if tp == 0:
        return 0.0
    pr, rc = tp / len(p), tp / len(g)
    return 2 * pr * rc / (pr + rc)


def answer_f1(answer_text: str, golds: list[str]) -> float:
    return max((token_f1(answer_text, g) for g in golds), default=0.0)


class CitationJudge:
    def __init__(self, scorer, threshold: float = 0.5):
        self.scorer, self.threshold = scorer, threshold
        self._memo: dict[tuple[str, str], bool] = {}

    def entails_batch(self, pairs: list[tuple[str, str]]) -> list[bool]:
        todo = [p for p in dict.fromkeys(pairs) if p not in self._memo]
        if todo:
            P = self.scorer([a for a, _ in todo], [b for _, b in todo])
            for pair, row in zip(todo, P):
                self._memo[pair] = bool(row[0] >= self.threshold and row[0] >= row[1] and row[0] >= row[2])
        return [self._memo[p] for p in pairs]

    def evaluate(self, answer: Answer) -> dict:
        texts = [p.chunk.text for p in answer.passages]
        sents = [s for s in answer.sentences if not is_abstention(s.text)]
        n_abst = len(answer.sentences) - len(sents)
        if not sents:
            return {"n_sentences": 0, "n_abstain_sentences": n_abst, "abstained": True,
                    "citation_recall": float("nan"), "citation_precision": float("nan"), "n_citations": 0}
        join = lambda ks: "\n".join(texts[k] for k in ks)
        # all entailment queries for this answer in one batch
        q = []
        for s in sents:
            if s.citations:
                q.append((join(s.citations), s.text))
                for c in s.citations:
                    q.append((texts[c], s.text))
                    if len(s.citations) > 1:
                        q.append((join([k for k in s.citations if k != c]), s.text))
        res = dict(zip(q, self.entails_batch(q))) if q else {}
        rec, prec = [], []
        for s in sents:
            if not s.citations:
                rec.append(0.0)
                continue
            full = res[(join(s.citations), s.text)]
            rec.append(float(full))
            for c in s.citations:
                alone = res[(texts[c], s.text)]
                rest = res[(join([k for k in s.citations if k != c]), s.text)] if len(s.citations) > 1 else False
                prec.append(float(alone or (full and not rest)))
        return {"n_sentences": len(sents), "n_abstain_sentences": n_abst, "abstained": False,
                "citation_recall": float(np.mean(rec)), "citation_precision": float(np.mean(prec)) if prec else 0.0,
                "n_citations": len(prec)}
