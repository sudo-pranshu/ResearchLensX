"""Mock components so every member can develop against the contracts before the real
modules exist. Every mock sets is_mock=True; the app MUST show a visible "MOCK" badge for
any output where Answer.is_mock or VerificationResult.is_mock is True (see ARCHITECTURE.md).
"""
from __future__ import annotations

import json
from pathlib import Path

from researchlens.contracts import Claim, Document, EvidenceJudgement, RetrievedPassage, VerificationResult
from researchlens.retrieval.lexical import tokenize

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"


class MockVerifier:
    """Word-overlap heuristic. NOT an NLI model; never predicts CONTRADICTS."""
    name = "mock-overlap"
    is_mock = True

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def verify(self, claim: Claim, passages: list[RetrievedPassage]) -> VerificationResult:
        c = set(tokenize(claim.text))
        judgements = []
        for p in passages:
            overlap = len(c & set(tokenize(p.chunk.text))) / max(len(c), 1)
            label = "SUPPORTS" if overlap >= self.threshold else "INSUFFICIENT"
            probs = {"SUPPORTS": overlap, "CONTRADICTS": 0.0, "INSUFFICIENT": 1 - overlap}
            judgements.append(EvidenceJudgement(chunk_id=p.chunk.chunk_id, label=label, probs=probs))
        best = max(judgements, key=lambda j: j.probs["SUPPORTS"], default=None)
        label = best.label if best else "INSUFFICIENT"
        conf = best.probs[label] if best else 1.0
        return VerificationResult(claim=claim, label=label, confidence=conf, evidence=judgements,
                                  model=self.name, is_mock=True)


def load_sample_documents() -> list[Document]:
    data = json.loads((SAMPLE_DIR / "documents.json").read_text())
    return [Document.model_validate(d) for d in data]
