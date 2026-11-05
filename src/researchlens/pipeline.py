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


# ---------------------------------------------------------------- config-driven builder (issue P6)
class RevisingPipeline(Pipeline):
    """Pipeline whose verification step is the full verified-RAG reviser (prune/reassign/drop)."""

    def __init__(self, retriever, generator, reviser):
        super().__init__(retriever, generator, reviser.verifier)
        self.reviser = reviser

    def ask(self, question, k=5, doc_ids=None, verify=True):
        if not verify:
            return super().ask(question, k, doc_ids, verify=False)
        t0 = time.perf_counter()
        passages = self.retriever.search(question, k=k, doc_ids=doc_ids)
        t1 = time.perf_counter()
        answer = self.generator.generate(question, passages)
        t2 = time.perf_counter()
        answer, log = self.reviser.revise(answer)
        answer.timings_ms.update({"retrieve": (t1 - t0) * 1e3, "verify": (time.perf_counter() - t2) * 1e3})
        answer.is_mock = bool(getattr(self.generator, "is_mock", False) or self.verifier.is_mock)
        return answer


def build_pipeline(documents: list[Document], retriever: str = "bm25", generator: str = "extractive",
                   verifier: str | None = None, revise: bool = True, support_threshold: float = 0.5,
                   contradict_threshold: float = 0.5, cache_dir: str | None = None):
    """Build the best pipeline the machine supports; never fails on missing optional components.

    retriever : any registry spec, e.g. "bm25", "rrf:bm25+specter2"
    generator : "extractive" | "hf:<model id>"  (e.g. "hf:Qwen/Qwen2.5-1.5B-Instruct")
    verifier  : None | "nli:<model id>"         (e.g. "nli:MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    Returns (pipeline, status) where status lists what is actually in use and why anything fell back.
    """
    from researchlens.generation.extractive import ExtractiveGenerator
    from researchlens.processing.chunking import chunk_document
    from researchlens.retrieval.index import ChunkIndex
    from researchlens.retrieval.registry import RetrievalConfig, build_systems

    status = {"requested": {"retriever": retriever, "generator": generator, "verifier": verifier}, "fallbacks": []}
    chunks = [c for d in documents for c in chunk_document(d)]
    titles = {d.doc_id: d.title for d in documents}
    texts = [c.index_text(titles.get(c.doc_id, "")) for c in chunks]
    try:
        systems, _, _ = build_systems([retriever], texts, RetrievalConfig(cache_dir=cache_dir), log=lambda m: None)
        index = ChunkIndex(systems[retriever], chunks, titles, name=retriever, prefit=True)
    except Exception as e:                                         # missing torch / model / offline
        status["fallbacks"].append(f"retriever {retriever!r} unavailable ({type(e).__name__}: {e}); using bm25")
        systems, _, _ = build_systems(["bm25"], texts, RetrievalConfig(), log=lambda m: None)
        index = ChunkIndex(systems["bm25"], chunks, titles, name="bm25", prefit=True)

    gen = ExtractiveGenerator()
    if generator.startswith("hf:"):
        try:
            from researchlens.generation.llm_rag import HFChatLLM, LLMRAGGenerator
            gen = LLMRAGGenerator(HFChatLLM(generator[3:]))
        except Exception as e:
            status["fallbacks"].append(f"generator {generator!r} unavailable ({type(e).__name__}); using extractive")

    ver = None
    if verifier and verifier.startswith("nli:"):
        try:
            from researchlens.verification.nli import HFNLIScorer, NLIVerifier
            ver = NLIVerifier(HFNLIScorer(verifier[4:]), support_threshold=support_threshold,
                              contradict_threshold=contradict_threshold)
        except Exception as e:
            status["fallbacks"].append(f"verifier {verifier!r} unavailable ({type(e).__name__}); answers NOT verified")

    if ver is not None and revise:
        from researchlens.generation.verified import VerifiedReviser
        pipe = RevisingPipeline(index, gen, VerifiedReviser(ver))
    else:
        pipe = Pipeline(index, gen, ver)
    status["in_use"] = {"retriever": index.name, "generator": gen.name,
                        "verifier": getattr(getattr(ver, "scorer", None), "name", None), "revise": bool(ver and revise)}
    return pipe, status
