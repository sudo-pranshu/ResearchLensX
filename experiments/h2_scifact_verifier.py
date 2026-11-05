"""H2 component evaluation: claim verification on SciFact (abstract-level, oracle cited abstracts).

  python experiments/h2_scifact_verifier.py --config configs/h2_scifact_verifier.yaml

Task: for every (claim, cited abstract) pair, predict SUPPORTS / CONTRADICTS / INSUFFICIENT (NEI).
Model selection (LR C, NLI thresholds) uses SciFact TRAIN only; DEV is the reported split.
Systems:
  lr                 TF-IDF features + Logistic Regression (classical baseline)
  V:sentence         verifier, sentence-level premises, default thresholds 0.5
  V:sentence:tuned   same, thresholds tuned on train (macro-F1 grid)
  V:passage          verifier, whole abstract as premise (granularity ablation)
  J:sentence         the H2 *judge* model on the same task (judge-validity check; must differ from V)
  V:oracle_rationale diagnostic: gold rationale sentences as premise (SUPPORTS/CONTRADICTS pairs only)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from researchlens.contracts import VERDICTS                                   # noqa: E402
from researchlens.data.scifact import load_claims, load_corpus                # noqa: E402
from researchlens.evaluation.classification import classification_report      # noqa: E402
from researchlens.evaluation.reporting import environment, git_rev            # noqa: E402
from researchlens.evaluation.stats import mcnemar                             # noqa: E402
from researchlens.verification.classical import LRVerifierBaseline            # noqa: E402
from researchlens.verification.nli import decide                              # noqa: E402


def pairs(claims, corpus):
    return [(c, d, c.label_for(d)) for c in claims for d in c.cited_doc_ids if d in corpus]


def score_units(scorer, items, corpus):
    """Sentence-level and passage-level probabilities for every (claim, doc) pair."""
    units, owner = [], []
    for k, (c, d, _) in enumerate(items):
        for si, s in enumerate(corpus[d].sentences):
            units.append((s, c.claim)); owner.append((k, si))
    t = time.time()
    P = scorer([u[0] for u in units], [u[1] for u in units])
    Pp = scorer([corpus[d].text for _, d, _ in items], [c.claim for c, _, _ in items])
    sent = [[] for _ in items]
    for (k, si), p in zip(owner, P):
        sent[k].append(p)
    return [np.array(s) for s in sent], Pp, time.time() - t


def sentence_decision(S: np.ndarray, ts: float, tc: float):
    best = int(np.argmax(np.maximum(S[:, 0], S[:, 1])))
    label, conf = decide(S[best], ts, tc)
    return label, conf, best


def tune_thresholds(sent_probs, gold):
    from researchlens.evaluation.classification import confusion, prf_from_confusion
    grid = np.round(np.arange(0.3, 0.96, 0.05), 2)
    best = (-1, 0.5, 0.5)
    for ts in grid:
        for tc in grid:
            pred = [sentence_decision(S, ts, tc)[0] for S in sent_probs]
            f = prf_from_confusion(confusion(gold, pred))["macro_f1"]
            if f > best[0] + 1e-12:
                best = (f, float(ts), float(tc))
    return {"train_macro_f1": best[0], "support_threshold": best[1], "contradict_threshold": best[2]}


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--root", default="data/raw/scifact/data")
    ap.add_argument("--verifier", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("--judge", default="roberta-large-mnli")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N claims per split")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--run-name", default=None)
    if known.config:
        import yaml
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in (yaml.safe_load(Path(known.config).read_text()) or {}).items()})
    args = ap.parse_args(argv)
    if args.verifier == args.judge:
        ap.error("verifier and judge must be different models")

    from researchlens.verification.nli import HFNLIScorer
    corpus = load_corpus(args.root)
    train = pairs(load_claims("train", args.root)[: args.limit], corpus)
    dev = pairs(load_claims("dev", args.root)[: args.limit], corpus)
    g_train, g_dev = [y for *_, y in train], [y for *_, y in dev]
    groups = [c.claim_id for c, _, _ in dev]
    print(f"pairs: train {len(train)}  dev {len(dev)}", flush=True)
    preds, extra, timings, models = {}, {}, {}, {}

    # ---- classical LR baseline: C chosen on an 80/20 claim-level split of train -----------
    t = time.time()
    lr = LRVerifierBaseline().fit_vocab([d.text for d in corpus.values()])
    feats = lambda items: np.array([lr.features(c.claim, corpus[d].sentences) for c, d, _ in items])
    Xtr, Xdv = feats(train), feats(dev)
    rng = np.random.default_rng(0)
    claim_ids = sorted({c.claim_id for c, _, _ in train})
    held = set(rng.choice(claim_ids, size=len(claim_ids) // 5, replace=False).tolist())
    m_fit = np.array([c.claim_id not in held for c, _, _ in train])
    from researchlens.evaluation.classification import confusion, prf_from_confusion
    best_C, best_f = 1.0, -1
    for C in [0.01, 0.1, 1.0, 10.0]:
        m = LRVerifierBaseline(C=C); m.tfidf = lr.tfidf
        m.fit(Xtr[m_fit], np.array(g_train)[m_fit])
        f = prf_from_confusion(confusion(np.array(g_train)[~m_fit], m.predict(Xtr[~m_fit])))["macro_f1"]
        if f > best_f:
            best_C, best_f = C, f
    final = LRVerifierBaseline(C=best_C); final.tfidf = lr.tfidf
    final.fit(Xtr, g_train)
    preds["lr"] = final.predict(Xdv)
    extra["lr"] = {"C": best_C, "heldout_train_macro_f1": best_f}
    timings["lr"] = round(time.time() - t, 1)
    print(f"  lr done (C={best_C})", flush=True)

    # ---- NLI verifier V ----------------------------------------------------------------------
    sV = HFNLIScorer(args.verifier, batch_size=args.batch_size)
    models["V"] = sV.meta
    Str, _, ttr = score_units(sV, train, corpus)
    Sdv, Pdv, tdv = score_units(sV, dev, corpus)
    timings["V_train_scoring"], timings["V_dev_scoring"] = round(ttr, 1), round(tdv, 1)
    tuned = tune_thresholds(Str, g_train)
    extra["V:sentence:tuned"] = tuned
    preds["V:sentence"] = [sentence_decision(S, 0.5, 0.5)[0] for S in Sdv]
    preds["V:sentence:tuned"] = [sentence_decision(S, tuned["support_threshold"], tuned["contradict_threshold"])[0] for S in Sdv]
    preds["V:passage"] = [decide(p, 0.5, 0.5)[0] for p in Pdv]
    # oracle rationale diagnostic (S/C pairs only)
    sc = [k for k, (_, _, y) in enumerate(dev) if y != "INSUFFICIENT"]
    prem = [" ".join(corpus[dev[k][1]].sentences[i] for i in dev[k][0].rationale_sentences(dev[k][1])) for k in sc]
    Por = sV(prem, [dev[k][0].claim for k in sc])
    oracle_pred = [decide(p, 0.5, 0.5)[0] for p in Por]
    print(f"  V done (tuned ts={tuned['support_threshold']} tc={tuned['contradict_threshold']})", flush=True)
    del sV; gc.collect()

    # ---- judge J on the same task -----------------------------------------------------------
    Sj = None
    if not args.skip_judge:
        sJ = HFNLIScorer(args.judge, batch_size=args.batch_size)
        models["J"] = sJ.meta
        Sj, _, tj = score_units(sJ, dev, corpus)
        timings["J_dev_scoring"] = round(tj, 1)
        preds["J:sentence"] = [sentence_decision(S, 0.5, 0.5)[0] for S in Sj]
        del sJ; gc.collect()
        print("  J done", flush=True)

    # ---- reports -----------------------------------------------------------------------------
    reports = {n: classification_report(g_dev, p, groups=groups) for n, p in preds.items()}
    reports["V:oracle_rationale(S/C only)"] = classification_report([dev[k][2] for k in sc], oracle_pred,
                                                                    groups=[groups[k] for k in sc])
    correct = {n: np.array(p) == np.array(g_dev) for n, p in preds.items()}
    tests = []
    for a, b in [("lr", "V:sentence"), ("lr", "V:sentence:tuned"), ("V:passage", "V:sentence"),
                 ("V:sentence", "J:sentence")]:
        if a in correct and b in correct:
            tests.append({"a": a, "b": b, "acc_a": float(correct[a].mean()), "acc_b": float(correct[b].mean()),
                          "mcnemar_p": mcnemar(correct[a], correct[b])})

    run = Path(args.out) / (args.run_name or f"h2_scifact_verifier_{int(time.time())}")
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({**vars(args), "git": git_rev(), "environment": environment(),
        "dataset": "SciFact 2020 release", "split_reported": "dev", "selection_split": "train",
        "n_pairs": {"train": len(train), "dev": len(dev)}, "models": models, "timings_s": timings,
        "selection": extra}, indent=2))
    (run / "summary.json").write_text(json.dumps(reports, indent=2))
    (run / "comparisons.json").write_text(json.dumps(tests, indent=2))
    import csv
    with open(run / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "macro_f1", "ci_low", "ci_high", "accuracy"] + [f"{l}_{m}" for l in VERDICTS for m in ("p", "r", "f1")])
        for n, r in reports.items():
            w.writerow([n, round(r["macro_f1"], 4), round(r["macro_f1_ci95"][0], 4), round(r["macro_f1_ci95"][1], 4),
                        round(r["accuracy"], 4)] + [round(r[l][m], 4) for l in VERDICTS for m in ("precision", "recall", "f1")])
    with open(run / "confusion_matrices.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "gold\\pred"] + list(VERDICTS))
        for n, r in reports.items():
            for lab, row in zip(VERDICTS, r["confusion"]["matrix"]):
                w.writerow([n, lab] + row)
    with open(run / "per_query.jsonl", "w") as f:
        for k, (c, d, y) in enumerate(dev):
            rec = {"claim_id": c.claim_id, "doc_id": d, "gold": y, **{f"pred:{n}": p[k] for n, p in preds.items()},
                   "V_sentence_probs_best": Sdv[k][int(np.argmax(np.maximum(Sdv[k][:, 0], Sdv[k][:, 1])))].round(4).tolist(),
                   "V_passage_probs": Pdv[k].round(4).tolist()}
            f.write(json.dumps(rec) + "\n")

    print(f"\n{'system':<30}{'macroF1':>9}{'95% CI':>18}{'acc':>7}   F1 S/C/N")
    for n, r in reports.items():
        print(f"{n:<30}{r['macro_f1']:>9.3f}   [{r['macro_f1_ci95'][0]:.3f}, {r['macro_f1_ci95'][1]:.3f}]{r['accuracy']:>7.3f}   "
              + "/".join(f"{r[l]['f1']:.2f}" for l in VERDICTS))
    for t in tests:
        print(f"  McNemar {t['a']} ({t['acc_a']:.3f}) vs {t['b']} ({t['acc_b']:.3f}): p={t['mcnemar_p']:.4f}")
    print("wrote", run)


if __name__ == "__main__":
    main()
