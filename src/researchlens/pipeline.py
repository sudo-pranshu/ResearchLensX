"""Integration facade used by the app, the API and experiments. Owner: Pranshu.

The app (Gaurav) talks ONLY to Pipeline; it never imports retrieval/verification internals.
"""
from __future__ import annotations

import time
from typing import Sequence

from researchlens.contracts import Answer, Claim, Document, Generator, PassageRetriever, Verifier


class Pipeline:
    def __init__(self, retriever: PassageRetriever, generator: Generator, verifier: Verifier | None = None):
        self.retriever, self.generator, self.verifier = retriever, generator, verifier

    def ask(self, question: str, k: int = 5, doc_ids: Sequence[str] | None = None, verify: bool = True) -> Answer:
        t0 = time.perf_counter()
        passages = self.retriever.search(question, k=k, doc_ids=doc_ids)
        t1 = time.perf_counter()
        answer = self.generator.generate(question, passages)
        t2 = time.perf_counter()
        if verify and self.verifier is not None:
            for i, s in enumerate(answer.sentences):
                cited = [answer.passages[c] for c in s.citations] or answer.passages
                s.verification = self.verifier.verify(Claim(claim_id=f"a{i}", text=s.text, source="answer"), cited)
            answer.mode = "verified_rag" if answer.mode == "rag" else answer.mode
        answer.is_mock = bool(answer.is_mock or getattr(self.generator, "is_mock", False)
                              or (verify and self.verifier is not None and self.verifier.is_mock))
        answer.timings_ms.update({"retrieve": (t1 - t0) * 1e3, "verify": (time.perf_counter() - t2) * 1e3})
        return answer

    def compare(self, question: str, doc_ids: Sequence[str], k: int = 3, verify: bool = True) -> dict[str, Answer]:
        """Basic multi-paper comparison: the same question answered within each paper separately."""
        return {d: self.ask(question, k=k, doc_ids=[d], verify=verify) for d in doc_ids}


def build_default_pipeline(documents: list[Document], retriever: str = "bm25", verifier=None) -> Pipeline:
    """CPU-only default: BM25 + extractive generator (+ optional verifier). Always works offline."""
    from researchlens.generation.extractive import ExtractiveGenerator
    from researchlens.processing.chunking import chunk_document
    from researchlens.retrieval.index import ChunkIndex
    from researchlens.retrieval.lexical import BM25Retriever, TfidfRetriever

    base = {"bm25": BM25Retriever, "tfidf": TfidfRetriever}[retriever]()
    chunks = [c for d in documents for c in chunk_document(d)]
    index = ChunkIndex(base, chunks, titles={d.doc_id: d.title for d in documents})
    return Pipeline(index, ExtractiveGenerator(), verifier)
