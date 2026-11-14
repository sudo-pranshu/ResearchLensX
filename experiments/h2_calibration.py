"""EXP-H2-003: temperature scaling of the NLI verifier on SciFact dev (passage-level probabilities).

Reuses the dev probabilities stored by EXP-H2-001 (no model re-run). Because T is fitted on dev, calibration quality
is estimated by claim-clustered K-fold CROSS-FITTING: T is fitted on K-1 folds and ECE/NLL measured on the held-out
fold, so reported numbers are out-of-fold. The T applied downstream is the one fitted on all of dev. Test untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchlens.evaluation.calibration import apply_temperature, ece, fit_temperature, nll  # noqa: E402
from researchlens.evaluation.reporting import environment, git_rev  # noqa: E402

SRC = Path("results/EXP-H2-001_scifact_verifier_dev/per_query.jsonl")
OUT = Path("results/EXP-H2-003_calibration_dev")
LAB = {"SUPPORTS": 0, "CONTRADICTS": 1, "INSUFFICIENT": 2}


def main(k_folds: int = 5, seed: int = 0, n_boot: int = 2000, out_dir: Path = OUT):
    rows = [json.loads(l) for l in SRC.read_text().splitlines()]
    P = np.array([r["V_passage_probs"] for r in rows]); P = P / P.sum(1, keepdims=True)
    y = np.array([LAB[r["gold"]] for r in rows]); claims = np.array([r["claim_id"] for r in rows])
    uniq = np.unique(claims); rng = np.random.default_rng(seed); rng.shuffle(uniq)
    fold_of = {c: i % k_folds for i, c in enumerate(uniq)}
    folds = np.array([fold_of[c] for c in claims])
    oof = np.zeros_like(P); Ts = []
    for f in range(k_folds):
        T = fit_temperature(P[folds != f], y[folds != f]); Ts.append(T)
        oof[folds == f] = apply_temperature(P[folds == f], T)
    T_all = fit_temperature(P, y)
    def metrics(Q):
        return {"ece": ece(Q, y), "nll": nll(Q, y), "accuracy": float((Q.argmax(1) == y).mean())}
    before, after = metrics(P), metrics(oof)
    # claim-clustered bootstrap CI for the ECE change (out-of-fold calibrated vs raw)
    by = {c: np.where(claims == c)[0] for c in uniq}; deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([by[c] for c in rng.choice(uniq, len(uniq))])
        deltas.append(ece(oof[idx], y[idx]) - ece(P[idx], y[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    out = {"n_pairs": len(y), "n_claims": len(uniq), "temperature_all_dev": round(T_all, 3),
           "temperature_per_fold": [round(t, 3) for t in Ts],
           "raw": {k: round(v, 4) for k, v in before.items()},
           "calibrated_out_of_fold": {k: round(v, 4) for k, v in after.items()},
           "ece_change_ci95": [round(float(lo), 4), round(float(hi), 4)],
           "note": "passage-level V probabilities from EXP-H2-001; probabilities rounded to 4 decimals in that artefact"}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    (out_dir / "config.json").write_text(json.dumps({"experiment": "EXP-H2-003", "source": str(SRC), "k_folds": k_folds, "seed": seed,
        "n_boot": n_boot, "git": git_rev(), "environment": environment(), "dataset": "SciFact 2020 release (dev, via EXP-H2-001 probabilities)"}, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    main(out_dir=Path(ap.parse_args().out))
