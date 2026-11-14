from __future__ import annotations

from dataclasses import dataclass

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_TEMPERATURE = 2.24      # fitted on SciFact dev PASSAGE-level probabilities (EXP-H2-003); valid for granularity="passage" only
RETRIEVERS = {"bm25": "BM25 (lexical)", "tfidf": "TF-IDF (lexical)", "e5": "e5-base-v2 (dense)", "rrf:bm25+e5": "Hybrid: RRF(BM25 + e5)"}


@dataclass
class ComponentStatus:
    name: str
    state: str                  # "real" | "fallback" | "mock" | "off"
    detail: str = ""


@dataclass
class Settings:
    """Per-request options. (Process-level configuration lives in researchlens.settings.AppConfig.)"""
    retriever: str = "bm25"            # bm25 | tfidf | e5 | rrf:bm25+e5 (dense/hybrid need the dense extra; fall back to BM25 with a note)
    rerank: bool = False
    dedup: float | None = 0.85
    k: int = 6
    max_per_doc: int = 3
    verify: bool = False
    combine_evidence: bool = False      # experimental: multi-passage support; sentence-level, uncalibrated, never evaluated on real data
    exclude_sections: tuple[str, ...] = ("references",)
