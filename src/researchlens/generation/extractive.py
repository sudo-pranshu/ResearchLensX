"""ExtractiveGenerator: CPU-only, deterministic, no LLM. Owner: Pawan (initial version by Pranshu).

For each of the top-n passages it selects the sentence with the highest lexical overlap
with the question and cites that passage. It is a genuine (weak) baseline, not a mock:
it never invents text, every sentence is copied from a cited passage.
"""
from __future__ import annotations

import re
import time

from researchlens.contracts import Answer, AnswerSentence, RetrievedPassage
from researchlens.retrieval.lexical import tokenize

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


class ExtractiveGenerator:
    name = "extractive-overlap-v0"
    is_mock = False

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, question: str, passages: list[RetrievedPassage]) -> Answer:
        t = time.perf_counter()
        q = set(tokenize(question))
        sentences = []
        for i, p in enumerate(passages[: self.max_sentences]):
            sents = split_sentences(p.chunk.text) or [p.chunk.text]
            best = max(sents, key=lambda s: (len(q & set(tokenize(s))), -sents.index(s)))
            sentences.append(AnswerSentence(text=best, citations=[i]))
        return Answer(question=question, sentences=sentences, passages=passages, mode="extractive",
                      generator=self.name, timings_ms={"generate": (time.perf_counter() - t) * 1e3})
