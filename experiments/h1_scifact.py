"""H1 open-corpus test: SciFact claim -> abstract retrieval (5,183 abstracts).

  python experiments/h1_scifact.py --config configs/h1_scifact_dev.yaml

Gold = abstracts with official evidence annotations for the claim (claims without evidence
are excluded and counted). Limitations (also in EXPERIMENT_PROTOCOL):
  * abstract-level only: retrieving the right abstract does not mean the rationale sentences are found;
  * incomplete judgements: only cited abstracts were annotated, so an unannotated but relevant
    abstract counts as a miss (scores are lower bounds, comparisons stay paired/fair);
  * ~1.1 relevant abstracts per claim, so Recall@k ~ Hit@k.
Never aggregate these numbers with QASPER paragraph-level metrics.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import researchlens._runtime  # noqa: E402,F401  (must precede numpy/torch/faiss use)

from researchlens.data.scifact import audit, load_claims, load_corpus           # noqa: E402
from researchlens.evaluation.metrics import all_metrics                         # noqa: E402
from researchlens.evaluation.reporting import write_run                         # noqa: E402
from researchlens.evaluation.stats import compare_many                          # noqa: E402
from researchlens.retrieval.registry import RetrievalConfig, build_systems, prime_queries  # noqa: E402


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--root", default="data/raw/scifact/data")
    ap.add_argument("--split", default="dev", choices=["train", "dev"])
    ap.add_argument("--systems", nargs="+", default=["bm25", "tfidf", "rrf:bm25+tfidf"])
    ap.add_argument("--primary", default="ndcg@10")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lsa-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--run-name", default=None)
    if known.config:
        import yaml
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in (yaml.safe_load(Path(known.config).read_text()) or {}).items()})
    args = ap.parse_args(argv)

    corpus = load_corpus(args.root)
    claims = load_claims(args.split, args.root)
    stats = audit(claims, corpus)
    doc_ids = list(corpus)
    pos = {d: i for i, d in enumerate(doc_ids)}
    docs = [f"{corpus[d].title}. {corpus[d].text}" for d in doc_ids]
    queries = [c for c in claims if c.evidence]
    print(f"SciFact {args.split}: {len(queries)} claims with evidence / {len(claims)}; corpus {len(docs)}", flush=True)

    rcfg = RetrievalConfig(lsa_dim=args.lsa_dim, rrf_k=args.rrf_k, depth=args.depth, alpha=args.alpha,
                           batch_size=args.batch_size, cache_dir=args.cache_dir)
    systems, base, meta = build_systems(args.systems, docs, rcfg, log=lambda m: print(m, flush=True),
                                        queries=[c.claim for c in queries])
    prime_queries(base, [c.claim for c in queries])

    per_query, search_seconds = {}, {}
    for name, r in systems.items():
        t = time.time()
        rows = []
        for c in queries:
            ranked = [d for d, _ in r.search(c.claim, k=20)]
            gold = {pos[d] for d in c.evidence_doc_ids}
            rows.append({"qid": c.claim_id, "n_gold": len(gold), "top5": [doc_ids[i] for i in ranked[:5]],
                         **all_metrics(ranked, gold)})
        per_query[name] = rows
        search_seconds[name] = round(time.time() - t, 2)
        print(f"  searched {name} in {search_seconds[name]:.1f}s", flush=True)

    comparisons = []
    singles = {n: [r[args.primary] for r in rows] for n, rows in per_query.items() if ":" not in n}
    for n, rows in per_query.items():
        if ":" in n:
            comparisons += [c.to_dict() for c in compare_many(singles, n, [r[args.primary] for r in rows], args.primary)]

    out = write_run(Path(args.out) / (args.run_name or f"h1_scifact_{args.split}_{int(time.time())}"),
                    {**vars(args), "dataset": "SciFact 2020 release", "audit": stats, "n_queries": len(queries),
                     "stats": "claim-level bootstrap (10k) + sign-flip randomisation, Holm per hybrid"},
                    per_query, comparisons, {**meta, "search_seconds": search_seconds})
    import json
    summary = json.loads((out / "summary.json").read_text())
    cols = ["recall@1", "recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10"]
    print(f"\n{'system':<24}" + "".join(f"{c:>11}" for c in cols))
    for n, s in summary.items():
        print(f"{n:<24}" + "".join(f"{s[c]:>11.4f}" for c in cols))
    for c in comparisons:
        print(f"  {c['system_b']} vs {c['system_a']}: Δ={c['delta']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] p_holm={c['p_holm']:.4f}")
    print("wrote", out)
    return summary, comparisons


if __name__ == "__main__":
    main()
