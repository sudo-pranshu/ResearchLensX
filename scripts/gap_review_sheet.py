"""Write a CSV for MANUAL review of automatically proposed gaps, then score the human labels.

  python scripts/gap_review_sheet.py make  --db data/processed/library.sqlite --out docs/review/gap_review.csv
  python scripts/gap_review_sheet.py score --csv docs/review/gap_review.csv

A reviewer fills the `label` column with: valid | borderline | invalid, and `evidence_ok` with y/n (does the quote really say it?).
`score` reports counts and precision (valid / all, and valid+borderline / all) per gap kind. Nothing is scored automatically.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make(db, out):
    from researchlens.workspace import Settings, Workspace
    gaps, _ = Workspace(db).gaps(Settings())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "basis", "paper(s)", "statement", "evidence_quote", "page", "label", "evidence_ok", "notes"])
        for g in gaps:
            e = g.evidence[0]
            w.writerow([g.id, g.kind, g.basis, " | ".join(g.papers)[:120], g.statement, e.quote, e.page, "", "", ""])
    print(f"wrote {len(gaps)} gaps to {out}; fill `label` (valid/borderline/invalid) and `evidence_ok` (y/n)")


def score(path):
    rows = [r for r in csv.DictReader(open(path)) if r["label"].strip()]
    if not rows:
        sys.exit("no labelled rows")
    by = defaultdict(Counter)
    for r in rows:
        by[r["kind"]][r["label"].strip().lower()] += 1
        by["ALL"][r["label"].strip().lower()] += 1
    for kind, c in by.items():
        n = sum(c.values())
        print(f"{kind:<22} n={n:<3} valid={c['valid']:<3} borderline={c['borderline']:<3} invalid={c['invalid']:<3} "
              f"precision(valid)={c['valid']/n:.2f}  precision(valid+borderline)={(c['valid']+c['borderline'])/n:.2f}")
    ok = Counter(r["evidence_ok"].strip().lower() for r in rows)
    print("evidence quote supports the statement:", dict(ok))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make"); m.add_argument("--db", default="data/processed/library.sqlite"); m.add_argument("--out", default="docs/review/gap_review.csv")
    s = sp.add_parser("score"); s.add_argument("--csv", default="docs/review/gap_review.csv")
    a = ap.parse_args()
    make(a.db, a.out) if a.cmd == "make" else score(a.csv)
