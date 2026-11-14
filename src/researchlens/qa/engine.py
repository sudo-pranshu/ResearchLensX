"""Evidence-grounded multi-document QA.

  qa = MultiDocQA(index, generator, documents, verifier=None)
  res = qa.answer("Which datasets do these papers evaluate on?", k=6, max_per_doc=3)

What it adds on top of Pipeline.ask:
  * multi-document aggregation: passages are balanced across papers (round-robin by each paper's best
    hit, capped at `max_per_doc`) so one long paper cannot crowd out the others
  * source traceability: every cited passage becomes a Reference (paper title, section, page, chunk id,
    the exact passage text). A page is reported only if the PDF ingestor recorded one; otherwise None
  * explicit insufficient-evidence handling: if no retrieved passage covers `min_coverage` of the question's
    content words, the engine abstains instead of generating. This is a WEAK, off-topic guard: on QASPER dev
    the signal has AUROC 0.64 for unanswerability (EXP-QA-001), so the default is deliberately low (0.2).
    The real insufficient-evidence check is claim verification: sentences the verifier cannot support are
    labelled "unsupported".
  * claim status per sentence (when a verifier is supplied): supported (one passage entails it), inferred
    (only several passages jointly entail it; needs NLIVerifier(combine=True)), conflicting (passages disagree),
    contradicted, unsupported. Statuses come from one NLI model and are evidence, not scientific truth.

Nothing is invented: sentences come from the injected generator and cite only passages it was given.
"""
from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field

from researchlens.contracts import Answer, AnswerSentence, Claim, Document, RetrievedPassage
from researchlens.retrieval.lexical import tokenize

ABSTAIN_TEXT = "The uploaded papers do not contain enough evidence to answer this question."
Status = str   # "supported" | "inferred" | "conflicting" | "contradicted" | "unsupported" | "unchecked" | "abstained"


class Reference(BaseModel):
    ref_id: int                          # 1-based, shown to the reader as [n]
    doc_id: str
    title: str
    section: str
    page: int | None                     # 1-based PDF page, None if the source has no page information
    chunk_id: str
    text: str                            # the exact passage the claim is attributed to
    retriever: str
    score: float


class SentenceStatus(BaseModel):
    text: str
    ref_ids: list[int]
    status: Status
    confidence: float | None = None


class QAResult(BaseModel):
    question: str
    answer: Answer
    references: list[Reference]
    sentences: list[SentenceStatus]
    abstained: bool
    abstain_reason: str | None = None
    evidence_coverage: float             # best coverage of the question's content words by one passage, 0..1
    docs_used: list[str]
    is_mock: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)


def evidence_coverage(question: str, passages: Sequence[RetrievedPassage]) -> float:
    """Max over passages of |content words of question found in passage| / |content words of question|.
    A cheap, deterministic proxy for "does any retrieved passage even talk about what was asked"."""
    q = set(tokenize(question))
    if not q or not passages:
        return 0.0
    return max(len(q & set(tokenize(p.chunk.text))) / len(q) for p in passages)


def balance_across_documents(passages: Sequence[RetrievedPassage], k: int, max_per_doc: int) -> list[RetrievedPassage]:
    """Round-robin over documents (ordered by their best passage), at most `max_per_doc` each, up to k.
    The result keeps the original relative order and is re-ranked 1..n."""
    by_doc: dict[str, list[RetrievedPassage]] = {}
    for p in passages:                                   # input is rank-ordered, so first-seen order = doc order
        by_doc.setdefault(p.chunk.doc_id, []).append(p)
    chosen: list[RetrievedPassage] = []
    depth = 0
    while len(chosen) < k and depth < max_per_doc:
        added = False
        for plist in by_doc.values():
            if depth < len(plist) and len(chosen) < k:
                chosen.append(plist[depth]); added = True
        if not added:
            break
        depth += 1
    chosen.sort(key=lambda p: p.rank)
    return [p.model_copy(update={"rank": r}) for r, p in enumerate(chosen, start=1)]


class MultiDocQA:
    def __init__(self, index, generator, documents: Sequence[Document], verifier=None,
                 min_coverage: float = 0.2):
        self.index, self.generator, self.verifier = index, generator, verifier
        self.titles = {d.doc_id: d.title for d in documents}
        self.min_coverage = min_coverage

    def answer(self, question: str, k: int = 6, max_per_doc: int = 3, doc_ids: Sequence[str] | None = None,
               where=None, verify: bool = True, verifier=None) -> QAResult:
        """`verifier` (optional) is used for this call only; the engine's own verifier is never mutated, so concurrent calls with
        and without verification cannot interfere."""
        verifier = verifier if verifier is not None else self.verifier
        import time
        t0 = time.perf_counter()
        kw = {} if where is None else {"where": where}
        pool = self.index.search(question, k=max(4 * k, 20), doc_ids=doc_ids, **kw)
        passages = balance_across_documents(pool, k, max_per_doc)
        cov = evidence_coverage(question, passages)
        t1 = time.perf_counter()
        if not passages or cov < self.min_coverage:
            reason = "no passages retrieved" if not passages else (
                f"best passage covers only {cov:.0%} of the question's content words (threshold {self.min_coverage:.0%})")
            ans = Answer(question=question, sentences=[AnswerSentence(text=ABSTAIN_TEXT)], passages=passages,
                         mode="extractive", generator="abstain", is_mock=False)
            return QAResult(question=question, answer=ans, references=self._refs(passages), abstained=True,
                            abstain_reason=reason, evidence_coverage=cov, docs_used=[],
                            sentences=[SentenceStatus(text=ABSTAIN_TEXT, ref_ids=[], status="abstained")],
                            timings_ms={"retrieve": (t1 - t0) * 1e3})
        ans = self.generator.generate(question, passages)
        statuses = []
        do_verify = verify and verifier is not None
        for i, s in enumerate(ans.sentences):
            ids = [c + 1 for c in s.citations if 0 <= c < len(ans.passages)]
            if do_verify:
                cited = [ans.passages[c - 1] for c in ids] or ans.passages
                v = verifier.verify(Claim(claim_id=f"a{i}", text=s.text, source="answer"), cited)
                s.verification = v
                labels = {e.label for e in v.evidence}
                if {"SUPPORTS", "CONTRADICTS"} <= labels:
                    st = "conflicting"                       # different passages disagree about this sentence
                elif v.label == "SUPPORTS":
                    st = "inferred" if v.support == "combined" else "supported"
                else:
                    st = {"CONTRADICTS": "contradicted"}.get(v.label, "unsupported")
                statuses.append(SentenceStatus(text=s.text, ref_ids=ids, status=st, confidence=v.confidence))
            else:
                statuses.append(SentenceStatus(text=s.text, ref_ids=ids, status="unchecked"))
        is_mock = bool(ans.is_mock or getattr(self.generator, "is_mock", False)
                       or (do_verify and getattr(verifier, "is_mock", False)))
        ans.is_mock = is_mock
        used = list(dict.fromkeys(ans.passages[c - 1].chunk.doc_id for s in statuses for c in s.ref_ids))
        return QAResult(question=question, answer=ans, references=self._refs(ans.passages), sentences=statuses,
                        abstained=False, evidence_coverage=cov, docs_used=used, is_mock=is_mock,
                        timings_ms={"retrieve": (t1 - t0) * 1e3, "generate_verify": (time.perf_counter() - t1) * 1e3})

    def _refs(self, passages: Sequence[RetrievedPassage]) -> list[Reference]:
        return [Reference(ref_id=i, doc_id=p.chunk.doc_id, title=self.titles.get(p.chunk.doc_id, p.chunk.doc_id),
                          section=p.chunk.section, page=p.chunk.page, chunk_id=p.chunk.chunk_id, text=p.chunk.text,
                          retriever=p.retriever, score=p.score) for i, p in enumerate(passages, start=1)]


def format_reference(r: Reference) -> str:
    page = f"p. {r.page}" if r.page is not None else "page unknown"
    return f"[{r.ref_id}] {r.title}, §{r.section}, {page}"
