"""Claim extraction from generated answers. Owner: Pawan (initial: Pranshu, sprint).
One claim per answer sentence (citation markers already stripped by the parser); abstention
sentences ("the passages do not contain ...") and wordless fragments are skipped; every other
sentence, however short, is a claim that must be verified."""
from __future__ import annotations

from researchlens.contracts import Answer, Claim
from researchlens.generation.llm_rag import is_abstention


def extract_claims(answer: Answer) -> list[tuple[int, Claim]]:
    """-> [(sentence index, Claim)] for verifiable sentences."""
    out = []
    for i, s in enumerate(answer.sentences):
        # only abstentions and fragments without any word are exempt; short claims MUST be verified
        if is_abstention(s.text) or not any(ch.isalnum() for ch in s.text):
            continue
        out.append((i, Claim(claim_id=f"s{i}", text=s.text, source="answer")))
    return out
