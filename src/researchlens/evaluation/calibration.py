"""Confidence calibration for NLI probabilities: temperature scaling, NLL, ECE.

Temperature scaling on probabilities: softmax(log p / T) == softmax(logits / T), so no logits are needed.
Calibration makes confidences better match accuracy; it does NOT make them evidence of scientific truth.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(np.asarray(probs, dtype=float), EPS, 1.0)) / T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(probs)[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.log(p).mean())


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error on the top-label confidence, equal-width bins."""
    probs = np.asarray(probs)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(probs: np.ndarray, y: np.ndarray, lo: float = 0.25, hi: float = 8.0) -> float:
    """T minimising NLL (golden-section search on log T; NLL is unimodal in T for this family)."""
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    f = lambda lt: nll(apply_temperature(probs, float(np.exp(lt))), y)
    for _ in range(60):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return float(np.exp((a + b) / 2))
