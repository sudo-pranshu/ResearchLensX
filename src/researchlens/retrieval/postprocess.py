"""Passage-level post-processing: redundancy suppression and metadata (section) filters."""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

from researchlens.contracts import Chunk, RetrievedPassage

from .lexical import tokenize


def _tokens(text: str) -> frozenset[str]:
    return frozenset(tokenize(text))


def suppress_redundant(passages: Sequence[RetrievedPassage], threshold: float = 0.85, min_tokens: int = 5,
                       k: int | None = None) -> list[RetrievedPassage]:
    """Drop a passage if a better-ranked kept passage already contains >= `threshold` of its content words
    (token-set containment |A∩B| / min(|A|,|B|)). Catches exact duplicates and a short passage repeated
    inside a longer one; overlapping sliding windows (small overlap) are not duplicates. Kept passages
    are re-ranked 1..n; order is otherwise unchanged. Very short passages are never suppressed."""
    kept: list[RetrievedPassage] = []
    kept_tok: list[frozenset[str]] = []
    for p in passages:
        t = _tokens(p.chunk.text)
        if len(t) >= min_tokens and any(
                len(t & o) / min(len(t), len(o)) >= threshold for o in kept_tok if len(o) >= min_tokens):
            continue
        kept.append(p)
        kept_tok.append(t)
        if k is not None and len(kept) >= k:
            break
    return [p.model_copy(update={"rank": r}) for r, p in enumerate(kept, start=1)]


def section_filter(include: Iterable[str] | None = None, exclude: Iterable[str] | None = None) -> Callable[[Chunk], bool]:
    """Chunk predicate on the section title (case-insensitive substring match).
    e.g. section_filter(exclude=["related work"]) or section_filter(include=["method", "experiment"])."""
    inc = [s.lower() for s in include] if include else None
    exc = [s.lower() for s in exclude] if exclude else []

    def keep(c: Chunk) -> bool:
        s = c.section.lower()
        if any(x in s for x in exc):
            return False
        return True if inc is None else any(x in s for x in inc)
    return keep
