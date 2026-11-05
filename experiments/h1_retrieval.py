"""H1: hybrid lexical-semantic retrieval vs lexical-only and dense-only, on QASPER.

Examples
  python experiments/h1_retrieval.py --qasper data/raw/qasper/qasper-dev-v0.3.json \
      --systems bm25 tfidf lsa rrf:bm25+lsa --setting within_paper

  # with neural encoders (needs `pip install -e .[dense]`)
  python experiments/h1_retrieval.py --qasper ... --systems bm25 e5 specter2 rrf:bm25+specter2 rrf:bm25+e5

Tune on dev. Run the test split once per final configuration and keep the log.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import researchlens._runtime  # noqa: E402,F401  (must precede numpy/torch/faiss use)

from researchlens.data.qasper import load_qasper                      # noqa: E402
from researchlens.evaluation.metrics import all_metrics                      # noqa: E402
from researchlens.evaluation.stats import compare_many                       # noqa: E402
from researchlens.evaluation.reporting import write_run                     # noqa: E402
from researchlens.retrieval.registry import RetrievalConfig, build_systems, prime_queries  # noqa: E402


def build_windows(split, w: int):
    """Section-agnostic units: each paper's paragraphs concatenated in order (captions excluded),
    cut into windows of w words with stride w/2. Returns (texts, unit->paragraph ids, paper->unit ids)."""
    texts, owners, paper_units = [], [], {}
    for paper_id, pids in split.paper_pids.items():
        words, wp = [], []
        for pid in pids:
            para = split.paragraphs[pid]
            if para.is_float:
                continue
            toks = para.text.split()
            words += toks
            wp += [pid] * len(toks)
        paper_units[paper_id] = []
        for start in range(0, max(len(words), 1), max(w // 2, 1)):
            chunk = words[start:start + w]
            if not chunk:
                break
            paper_units[paper_id].append(len(texts))
            texts.append(f"{split.titles[paper_id]}\n" + " ".join(chunk))
            owners.append(list(dict.fromkeys(wp[start:start + w])))
            if start + w >= len(words):
                break
    return texts, owners, paper_units


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="YAML file whose keys set argument defaults (CLI wins)")
    known, _ = pre.parse_known_args(argv)
    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--qasper", required=False)
    ap.add_argument("--systems", nargs="+", default=["bm25", "tfidf", "lsa", "rrf:bm25+lsa"])
    ap.add_argument("--setting", choices=["within_paper", "open"], default="within_paper")
    ap.add_argument("--no-section-prefix", action="store_true")
    ap.add_argument("--no-floats", action="store_true")
    ap.add_argument("--primary", default=None, help="metric for significance (default by setting)")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--bm25-k1", type=float, default=1.5)
    ap.add_argument("--bm25-b", type=float, default=0.75)
    ap.add_argument("--lsa-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cache-dir", default=None, help="embedding cache (default artifacts/embeddings)")
    ap.add_argument("--split-name", default=None, help="label recorded in config, e.g. dev/test")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N questions")
    ap.add_argument("--out", default="results")
    ap.add_argument("--title-query", action="store_true", help="prepend paper title to the question")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--unit", choices=["paragraph", "window"], default="paragraph",
                    help="paragraph = section-aware units (default); window = section-agnostic fixed windows "
                         "(ablation), mapped back to paragraphs by max window score")
    ap.add_argument("--window-words", type=int, default=128)
    if known.config:
        import yaml
        cfg = yaml.safe_load(Path(known.config).read_text()) or {}
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    args = ap.parse_args(argv)
    if not args.qasper:
        ap.error("--qasper is required (CLI or config)")
    primary = args.primary or ("recall@5" if args.setting == "within_paper" else "ndcg@10")

    split = load_qasper(args.qasper, include_floats=not args.no_floats)
    print("data:", json.dumps(split.stats))
    if args.unit == "paragraph":
        docs = [p.indexed_text(split.titles[p.paper_id], with_section=not args.no_section_prefix) for p in split.paragraphs]
        unit_pids = [[p.pid] for p in split.paragraphs]
        paper_units = split.paper_pids
    else:
        docs, unit_pids, paper_units = build_windows(split, args.window_words)
        print(f"section-agnostic windows: {len(docs)} windows of {args.window_words} words", flush=True)
    questions = [q for q in split.questions if q.gold_union][: args.limit]
    print(f"evaluating {len(questions)} questions with evidence ({args.setting})")

    rcfg = RetrievalConfig(bm25_k1=args.bm25_k1, bm25_b=args.bm25_b, lsa_dim=args.lsa_dim, rrf_k=args.rrf_k,
                           depth=args.depth, alpha=args.alpha, batch_size=args.batch_size, cache_dir=args.cache_dir)
    queries = [f"{split.titles[q.paper_id]}. {q.question}" if args.title_query else q.question for q in questions]
    systems, base, sys_meta = build_systems(args.systems, docs, rcfg, log=lambda m: print(m, flush=True),
                                            queries=queries)
    t = time.time()
    prime_queries(base, queries)
    sys_meta["query_encode_seconds"] = round(time.time() - t, 2)
    per_query: dict[str, list[dict]] = {name: [] for name in systems}
    search_seconds = {}
    for name, sys_ in systems.items():
        t = time.time()
        for q in questions:
            cands = np.array(paper_units[q.paper_id]) if args.setting == "within_paper" else None
            query = f"{split.titles[q.paper_id]}. {q.question}" if args.title_query else q.question
            depth = 20 if args.unit == "paragraph" else 200
            ranked = []
            for u, _ in sys_.search(query, k=depth, candidates=cands):   # units -> paragraphs, first hit wins
                for pid in unit_pids[u]:
                    if pid not in ranked:
                        ranked.append(pid)
            ranked = ranked[:20]
            per_query[name].append({"qid": q.qid, "paper_id": q.paper_id, "n_gold": len(q.gold_union),
                                    "top5": ranked[:5], **all_metrics(ranked, q.gold_union)})
        search_seconds[name] = round(time.time() - t, 2)
        print(f"  searched {name} in {search_seconds[name]:.1f}s", flush=True)

    # Significance: each hybrid vs every non-hybrid system, Holm-adjusted per hybrid.
    comparisons = []
    # Questions are clustered by paper -> paper-level cluster bootstrap / sign-flip.
    groups = [q.paper_id for q in questions]
    singles = {n: [r[primary] for r in rows] for n, rows in per_query.items() if ":" not in n}
    for n, rows in per_query.items():
        if ":" in n and singles:
            comparisons += [c.to_dict() for c in compare_many(singles, n, [r[primary] for r in rows], primary,
                                                               groups=groups)]

    run_id = args.run_name or f"h1_{Path(args.qasper).stem}_{args.setting}_{int(time.time())}"
    out = write_run(Path(args.out) / run_id,
                    {**vars(args), "dataset": "QASPER v0.3", "data_stats": split.stats,
                     "n_questions": len(questions), "n_papers": len(set(groups)), "primary": primary,
                     "stats": "paper-level cluster bootstrap (10k) + cluster sign-flip randomisation, Holm per hybrid"},
                    per_query, comparisons, {**sys_meta, "search_seconds": search_seconds})
    summary = json.loads((out / "summary.json").read_text())

    cols = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    print(f"\n{'system':<24}" + "".join(f"{c:>11}" for c in cols))
    for n, s in summary.items():
        print(f"{n:<24}" + "".join(f"{s[c]:>11.4f}" for c in cols))
    if comparisons:
        print(f"\nsignificance on {primary} (delta = hybrid - baseline):")
        for c in comparisons:
            print(f"  {c['system_b']} vs {c['system_a']}: delta={c['delta']:+.4f} "
                  f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  p_rand={c['p_randomization']:.4f}  p_holm={c['p_holm']:.4f}")
    print(f"\nwrote {out}")
    return summary, comparisons


if __name__ == "__main__":
    main()
