"""SciFact loader (Wadden et al., 2020; CC BY-NC 2.0). Owner: Pawan (initial version: Pranshu, sprint).

Files (data/raw/scifact/data/): corpus.jsonl, claims_{train,dev,test}.jsonl.
  corpus: {"doc_id": int, "title": str, "abstract": [sentence, ...], "structured": bool}
  claims: {"id": int, "claim": str, "cited_doc_ids": [int],
           "evidence": {"<doc_id>": [{"sentences": [int], "label": "SUPPORT"|"CONTRADICT"}, ...]}}
Test claims have no evidence/labels (leaderboard). Labels map to contracts.Verdict:
SUPPORT -> SUPPORTS, CONTRADICT -> CONTRADICTS, cited doc without evidence -> INSUFFICIENT (NEI).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

LABEL_MAP = {"SUPPORT": "SUPPORTS", "CONTRADICT": "CONTRADICTS"}
DEFAULT_DIR = Path("data/raw/scifact/data")


@dataclass
class SciFactDoc:
    doc_id: str
    title: str
    sentences: list[str]
    structured: bool = False

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


@dataclass
class SciFactClaim:
    claim_id: int
    claim: str
    cited_doc_ids: list[str]
    evidence: dict[str, dict] = field(default_factory=dict)   # doc_id -> {"label": Verdict, "rationales": [[int]]}

    @property
    def evidence_doc_ids(self) -> set[str]:
        return set(self.evidence)

    def label_for(self, doc_id: str) -> str:
        """Abstract-level gold label for a (claim, cited doc) pair."""
        return self.evidence[doc_id]["label"] if doc_id in self.evidence else "INSUFFICIENT"

    def rationale_sentences(self, doc_id: str) -> list[int]:
        return sorted({i for r in self.evidence.get(doc_id, {}).get("rationales", []) for i in r})


def _read_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(root: str | Path = DEFAULT_DIR) -> dict[str, SciFactDoc]:
    return {str(d["doc_id"]): SciFactDoc(str(d["doc_id"]), d["title"], [s.strip() for s in d["abstract"]],
                                         bool(d.get("structured", False)))
            for d in _read_jsonl(Path(root) / "corpus.jsonl")}


def load_claims(split: str, root: str | Path = DEFAULT_DIR) -> list[SciFactClaim]:
    claims = []
    for c in _read_jsonl(Path(root) / f"claims_{split}.jsonl"):
        ev = {}
        for doc_id, sets in (c.get("evidence") or {}).items():
            labels = {s["label"] for s in sets}
            if len(labels) != 1:                       # never happens in the release; guard anyway
                raise ValueError(f"claim {c['id']} doc {doc_id}: mixed labels {labels}")
            ev[str(doc_id)] = {"label": LABEL_MAP[labels.pop()], "rationales": [s["sentences"] for s in sets]}
        claims.append(SciFactClaim(int(c["id"]), c["claim"], [str(d) for d in c.get("cited_doc_ids", [])], ev))
    return claims


def audit(claims: list[SciFactClaim], corpus: dict[str, SciFactDoc]) -> dict:
    from collections import Counter
    pairs = Counter(c.label_for(d) for c in claims for d in c.cited_doc_ids)
    return {
        "claims": len(claims),
        "claims_with_evidence": sum(bool(c.evidence) for c in claims),
        "evidence_docs_total": sum(len(c.evidence) for c in claims),
        "claims_multi_evidence_docs": sum(len(c.evidence) > 1 for c in claims),
        "cited_pairs_by_label": dict(pairs),
        "evidence_docs_missing_from_corpus": sum(d not in corpus for c in claims for d in c.evidence),
        "corpus_docs": len(corpus),
    }
