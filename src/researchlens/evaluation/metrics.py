"""Per-query ranking metrics (binary relevance). All return floats in [0, 1]."""
from __future__ import annotations

import math
from typing import Collection, Sequence


def recall_at_k(ranked: Sequence[int], gold: Collection[int], k: int) -> float:
    if not gold:
        return float("nan")
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def hit_at_k(ranked: Sequence[int], gold: Collection[int], k: int) -> float:
    return float(any(d in gold for d in ranked[:k]))


def precision_at_k(ranked: Sequence[int], gold: Collection[int], k: int) -> float:
    return sum(d in gold for d in ranked[:k]) / k


def mrr_at_k(ranked: Sequence[int], gold: Collection[int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[int], gold: Collection[int], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else float("nan")


def set_f1(pred: Collection, gold: Collection) -> float:
    """Token/evidence set F1, as used for QASPER Evidence-F1 (per annotator; take max)."""
    pred, gold = set(pred), set(gold)
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def all_metrics(ranked: Sequence[int], gold: Collection[int], ks=(1, 3, 5, 10, 20)) -> dict[str, float]:
    gold = set(gold)
    out = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in ks}
    out.update({f"hit@{k}": hit_at_k(ranked, gold, k) for k in ks})
    out["mrr@10"] = mrr_at_k(ranked, gold, 10)
    out["ndcg@10"] = ndcg_at_k(ranked, gold, 10)
    return out
