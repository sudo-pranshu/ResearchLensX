"""Classification metrics for verification: per-class P/R/F1, macro-F1, confusion matrix,
cluster bootstrap CI for macro-F1 (clusters = claims), and McNemar between two systems."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from researchlens.contracts import VERDICTS


def confusion(gold: Sequence[str], pred: Sequence[str], labels=VERDICTS) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for g, p in zip(gold, pred):
        M[idx[g], idx[p]] += 1
    return M                                           # rows = gold, cols = predicted


def prf_from_confusion(M: np.ndarray, labels=VERDICTS) -> dict:
    out = {}
    for i, l in enumerate(labels):
        tp, fp, fn = M[i, i], M[:, i].sum() - M[i, i], M[i, :].sum() - M[i, i]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[l] = {"precision": float(p), "recall": float(r), "f1": float(2 * p * r / (p + r) if p + r else 0.0),
                  "support": int(M[i, :].sum())}
    out["macro_f1"] = float(np.mean([out[l]["f1"] for l in labels]))
    out["accuracy"] = float(np.trace(M) / M.sum()) if M.sum() else 0.0
    return out


def classification_report(gold, pred, groups=None, n_resamples: int = 5000, seed: int = 0) -> dict:
    M = confusion(gold, pred)
    rep = prf_from_confusion(M)
    rep["confusion"] = {"labels_rows_gold_cols_pred": list(VERDICTS), "matrix": M.tolist()}
    g = np.asarray(groups if groups is not None else np.arange(len(gold)))
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    gold, pred = np.asarray(gold), np.asarray(pred)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_resamples):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        boots.append(prf_from_confusion(confusion(gold[idx], pred[idx]))["macro_f1"])
    rep["macro_f1_ci95"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    rep["n"] = int(len(gold))
    return rep
