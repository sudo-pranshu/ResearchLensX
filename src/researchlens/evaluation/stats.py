"""Paired significance testing for per-query system comparisons.

Use these on per-query metric arrays (same queries, same order) for two systems.
  * paired_bootstrap      - 95% CI on the mean difference (Koehn 2004; Efron & Tibshirani)
  * paired_randomization  - sign-flip permutation test p-value (two-sided)
  Both accept `groups` (e.g. QASPER paper ids): questions from the same paper are not
  independent, so resampling / sign-flipping is then done at the cluster level.
  * holm                  - Holm-Bonferroni step-down adjustment across comparisons
  * mcnemar               - for paired binary outcomes (e.g. claim supported y/n)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy import stats as sps


@dataclass
class Comparison:
    system_a: str
    system_b: str
    metric: str
    n: int
    mean_a: float
    mean_b: float
    delta: float          # mean_b - mean_a
    ci_low: float
    ci_high: float
    p_randomization: float
    p_wilcoxon: float        # question-level; ignores clustering, reported for reference only
    ci_method: str = "bootstrap"
    p_holm: float | None = None

    def to_dict(self):
        return asdict(self)


def _clean(a, b, groups=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    mask = ~(np.isnan(a) | np.isnan(b))
    if groups is None:
        return a[mask], b[mask], None
    g = np.asarray(groups)
    if g.shape != a.shape:
        raise ValueError("groups must align with the score arrays")
    return a[mask], b[mask], g[mask]


def _cluster_sums(d, groups):
    _, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    counts = np.bincount(inv).astype(float)
    return sums, counts


def paired_bootstrap(a, b, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0, groups=None):
    a, b, g = _clean(a, b, groups)
    d = b - a
    rng = np.random.default_rng(seed)
    if g is None:
        idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
        boots = d[idx].mean(axis=1)
    else:
        sums, counts = _cluster_sums(d, g)
        idx = rng.integers(0, len(sums), size=(n_resamples, len(sums)))
        boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def paired_randomization(a, b, n_resamples: int = 10_000, seed: int = 0, groups=None) -> float:
    a, b, g = _clean(a, b, groups)
    d = b - a
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    if g is None:
        signs = rng.choice([-1.0, 1.0], size=(n_resamples, len(d)))
        null = np.abs((signs * d).mean(axis=1))
    else:
        sums, counts = _cluster_sums(d, g)
        signs = rng.choice([-1.0, 1.0], size=(n_resamples, len(sums)))
        null = np.abs((signs * sums).sum(axis=1) / counts.sum())
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_resamples + 1))


def wilcoxon_p(a, b) -> float:
    a, b, _ = _clean(a, b)
    if np.allclose(a, b):
        return 1.0
    return float(sps.wilcoxon(a, b, zero_method="wilcox").pvalue)


def holm(pvals: Sequence[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def mcnemar(a_correct, b_correct) -> float:
    """Exact McNemar test (binomial on discordant pairs)."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    if n01 + n10 == 0:
        return 1.0
    return float(sps.binomtest(n01, n01 + n10, 0.5).pvalue)


def compare(name_a, a, name_b, b, metric, n_resamples=10_000, seed=0, groups=None) -> Comparison:
    ac, bc, g = _clean(a, b, groups)
    delta, lo, hi = paired_bootstrap(ac, bc, n_resamples, seed=seed, groups=g)
    return Comparison(name_a, name_b, metric, len(ac), float(ac.mean()), float(bc.mean()), delta, lo, hi,
                      paired_randomization(ac, bc, n_resamples, seed, groups=g), wilcoxon_p(ac, bc),
                      ci_method="cluster_bootstrap" if g is not None else "bootstrap")


def compare_many(baselines: dict[str, Sequence[float]], treatment_name: str, treatment, metric: str, **kw):
    """Treatment vs each baseline, Holm-adjusted across the family."""
    comps = [compare(n, s, treatment_name, treatment, metric, **kw) for n, s in baselines.items()]
    for c, p in zip(comps, holm([c.p_randomization for c in comps])):
        c.p_holm = p
    return comps
