"""Shared data contracts and module interfaces for ResearchLens X.

OWNER: Pranshu (architecture). Changing anything in this file requires a PR labelled
`interface-change`, approval from Pranshu, and a CHANGELOG entry. Additive optional
fields are minor changes; renaming/removing fields or changing types is a breaking
change and bumps SCHEMA_VERSION.

Data flow:
  Ingestor (Gaurav) -> Document -> chunk_document (Gaurav) -> list[Chunk]
  -> PassageRetriever (Pranshu) -> list[RetrievedPassage]
  -> Generator (Pawan) -> Answer -> Verifier (Pawan) -> VerificationResult per sentence
  -> GraphBuilder (Gaurav) -> KnowledgeGraph -> app (Gaurav)
"""
from __future__ import annotations

from typing import Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

# Verification label set. Mapping from SciFact: SUPPORT->SUPPORTS, CONTRADICT->CONTRADICTS, NEI->INSUFFICIENT.
Verdict = Literal["SUPPORTS", "CONTRADICTS", "INSUFFICIENT"]
VERDICTS: tuple[str, ...] = ("SUPPORTS", "CONTRADICTS", "INSUFFICIENT")


# ---------------------------------------------------------------- documents (Gaurav)
class Reference(BaseModel):
    raw: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None


class Section(BaseModel):
    title: str
    paragraphs: list[str]
    level: int = 1


class Document(BaseModel):
    doc_id: str                                   # stable id: arXiv id, QASPER id, or sha1 of PDF bytes[:12]
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    sections: list[Section]
    references: list[Reference] = Field(default_factory=list)
    source: Literal["pymupdf", "grobid", "qasper", "fixture"]
    source_path: str | None = None
    schema_version: str = SCHEMA_VERSION


class Chunk(BaseModel):
    chunk_id: str                                 # f"{doc_id}::s{section_idx}::p{para_idx}::w{window}"
    doc_id: str
    section: str
    text: str
    order: int                                    # position of the chunk within its document

    def index_text(self, title: str = "") -> str:
        """Text handed to retrievers (section-prefixed, see EXPERIMENT_PROTOCOL H1 ablations)."""
        head = " | ".join(x for x in (title, self.section) if x)
        return f"{head}\n{self.text}" if head else self.text


# ---------------------------------------------------------------- retrieval (Pranshu)
class RetrievedPassage(BaseModel):
    chunk: Chunk
    score: float
    rank: int                                     # 1-based
    retriever: str                                # e.g. "bm25", "rrf(bm25+specter2)"


# ---------------------------------------------------------------- verification (Pawan)
class Claim(BaseModel):
    claim_id: str
    text: str
    source: Literal["answer", "document", "user"]
    doc_id: str | None = None


class EvidenceJudgement(BaseModel):
    chunk_id: str
    label: Verdict
    probs: dict[str, float]                       # keys = VERDICTS, sums to ~1
    rationale_sentences: list[int] = Field(default_factory=list)


class VerificationResult(BaseModel):
    claim: Claim
    label: Verdict                                # aggregated over evidence
    confidence: float                             # calibrated where possible; document how in model card
    evidence: list[EvidenceJudgement]
    model: str                                    # HF model id + revision, or "mock-*"
    is_mock: bool = False


# ---------------------------------------------------------------- generation (Pawan)
class AnswerSentence(BaseModel):
    text: str
    citations: list[int] = Field(default_factory=list)   # indices into Answer.passages (0-based)
    verification: VerificationResult | None = None


class Answer(BaseModel):
    question: str
    sentences: list[AnswerSentence]
    passages: list[RetrievedPassage]
    mode: Literal["extractive", "rag", "verified_rag"]
    generator: str
    is_mock: bool = False                         # True if ANY component that produced this was a mock
    timings_ms: dict[str, float] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------- knowledge graph (Gaurav)
NodeType = Literal["paper", "section", "entity", "claim"]
EdgeType = Literal["has_section", "mentions", "cites", "supports", "contradicts", "shares_entity"]


class KGNode(BaseModel):
    id: str
    type: NodeType
    label: str
    attrs: dict = Field(default_factory=dict)     # entity nodes: {"entity_type": "dataset"|"method"|"metric"|"task"}


class KGEdge(BaseModel):
    source: str
    target: str
    type: EdgeType
    weight: float = 1.0
    attrs: dict = Field(default_factory=dict)


class KnowledgeGraph(BaseModel):
    nodes: list[KGNode]
    edges: list[KGEdge]
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------- module interfaces
@runtime_checkable
class Ingestor(Protocol):
    name: str
    def ingest(self, pdf_path: str) -> Document: ...


@runtime_checkable
class PassageRetriever(Protocol):
    name: str
    def search(self, query: str, k: int = 10, doc_ids: Sequence[str] | None = None) -> list[RetrievedPassage]: ...


@runtime_checkable
class Generator(Protocol):
    name: str
    is_mock: bool
    def generate(self, question: str, passages: list[RetrievedPassage]) -> Answer: ...


@runtime_checkable
class Verifier(Protocol):
    name: str
    is_mock: bool
    def verify(self, claim: Claim, passages: list[RetrievedPassage]) -> VerificationResult: ...


@runtime_checkable
class GraphBuilder(Protocol):
    def build(self, documents: list[Document], verifications: list[VerificationResult] | None = None) -> KnowledgeGraph: ...
