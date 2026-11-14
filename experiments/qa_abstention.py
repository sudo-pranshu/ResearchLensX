"""EXP-QA-001: can lexical evidence coverage detect unanswerable QASPER questions?

For every dev question (answerable AND unanswerable) retrieve the top-5 paragraphs of the question's own paper
with BM25 and compute `evidence_coverage` (the signal MultiDocQA uses to abstain). Label = the question's
majority annotator answer type is "none" (the loader maps unanswerable and empty answers to "none"). Reports AUROC (paper-clustered bootstrap CI), and
abstention precision/recall at several thresholds. Tune on dev; do not run on test without registering it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import researchlens._runtime  # noqa: E402,F401

from researchlens.data.qasper import load_qasper                      # noqa: E402
from researchlens.evaluation.reporting import environment, git_rev      # noqa: E402
from researchlens.qa.engine import evidence_coverage                    # noqa: E402
from researchlens.contracts import Chunk, RetrievedPassage              # noqa: E402
from researchlens.retrieval.lexical import BM25Retriever                # noqa: E402


def auroc(pos_scores, neg_scores) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie), computed by ranks (Mann-Whitney)."""
    s = np.concatenate([pos_scores, neg_scores]); ranks = s.argsort().argsort().astype(float) + 1
    for v in np.unique(s):                                   # average ranks over ties
        m = s == v
        ranks[m] = ranks[m].mean()
    n1, n0 = len(pos_scores), len(neg_scores)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasper", default="data/raw/qasper/qasper-dev-v0.3.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/EXP-QA-001_abstention_dev")
    args = ap.parse_args(argv)

    split = load_qasper(args.qasper)
    texts = [p.indexed_text(split.titles[p.paper_id]) for p in split.paragraphs]
    bm = BM25Retriever().fit(texts)
    rows = []
    for q in split.questions:
        hits = bm.search(q.question, args.k, np.array(split.paper_pids[q.paper_id]))
        ps = [RetrievedPassage(chunk=Chunk(chunk_id=str(i), doc_id=q.paper_id, section=split.paragraphs[i].section,
                                           text=split.paragraphs[i].text, order=i), score=s, rank=r, retriever="bm25")
              for r, (i, s) in enumerate(hits, 1)]
        rows.append({"qid": q.qid, "paper": q.paper_id, "unanswerable": q.majority_type == "none",
                     "coverage": evidence_coverage(q.question, ps)})
    cov = np.array([r["coverage"] for r in rows]); un = np.array([r["unanswerable"] for r in rows])
    # lower coverage => more likely unanswerable, so score for "unanswerable" is -coverage
    a = auroc(-cov[un], -cov[~un])
    by_paper = defaultdict(list)
    for i, r in enumerate(rows):
        by_paper[r["paper"]].append(i)
    papers = list(by_paper); rng = np.random.default_rng(0); boots = []
    for _ in range(args.n_boot):
        idx = np.concatenate([by_paper[papers[j]] for j in rng.integers(0, len(papers), len(papers))])
        if un[idx].any() and (~un[idx]).any():
            boots.append(auroc(-cov[idx][un[idx]], -cov[idx][~un[idx]]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    table = []
    for tau in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        abst = cov < tau
        tp = int((abst & un).sum())
        table.append({"threshold": tau, "abstain_rate": round(float(abst.mean()), 3),
                      "precision": round(tp / max(int(abst.sum()), 1), 3), "recall": round(tp / max(int(un.sum()), 1), 3),
                      "answerable_wrongly_abstained": round(float((abst & ~un).sum() / max(int((~un).sum()), 1)), 3)})
    out = {"n_questions": len(rows), "n_unanswerable": int(un.sum()), "unanswerable_rate": round(float(un.mean()), 3),
           "auroc": round(a, 3), "auroc_ci95_paper_bootstrap": [round(float(lo), 3), round(float(hi), 3)],
           "thresholds": table, "dataset": "QASPER v0.3 dev", "retriever": "bm25 within-paper top-%d" % args.k}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(out, indent=2))
    (Path(args.out) / "config.json").write_text(json.dumps({"experiment": "EXP-QA-001", **vars(args), "git": git_rev(),
        "environment": environment(), "dataset": "QASPER v0.3 dev"}, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
