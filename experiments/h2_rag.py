"""H2: conventional RAG vs claim-verified RAG on QASPER (same generator output in every condition).

  python experiments/h2_rag.py --config configs/h2_rag_dev_pilot.yaml

Stages run sequentially (one model in memory at a time; 8 GB laptops are fine):
  1. retrieve  : hybrid within-paper top-k (cached embeddings)
  2. generate  : LLM, greedy; cached to <run>/generations.jsonl and reused on re-runs
  3. verify    : verifier V (NLI) -> conditions: rag | verify_only | verified | ablations
  4. judge     : judge J (different NLI model) -> citation recall/precision; Answer-F1 vs gold
Question set: questions with gold evidence whose majority answer type is extractive/abstractive
(review R7), sampled with a fixed seed.
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

from researchlens.contracts import Answer                                          # noqa: E402
from researchlens.data.qasper import load_qasper, load_qasper_documents            # noqa: E402
from researchlens.evaluation.attribution import CitationJudge, answer_f1           # noqa: E402
from researchlens.evaluation.reporting import write_run                            # noqa: E402
from researchlens.evaluation.stats import compare_many                             # noqa: E402
from researchlens.generation.llm_rag import LLMRAGGenerator, build_messages, prompt_hash  # noqa: E402
from researchlens.processing.chunking import chunk_document                        # noqa: E402
from researchlens.retrieval.index import ChunkIndex                                # noqa: E402
from researchlens.retrieval.registry import RetrievalConfig, build_systems, prime_queries  # noqa: E402

CONDITIONS = {  # name -> (mode, reassign, drop_unsupported)
    "rag": None,
    "verify_only": ("verify_only", True, True),
    "verified": ("revise", True, True),
    "ablation:no_reassign": ("revise", False, True),
    "ablation:no_drop": ("revise", True, False),
}


def free():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--qasper", default="data/raw/qasper/qasper-dev-v0.3.json")
    ap.add_argument("--split-name", default="dev")
    ap.add_argument("--n-questions", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retriever", default="rrf:bm25+specter2")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--generator", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--verifier", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("--verifier-granularity", default="sentence")
    ap.add_argument("--support-threshold", type=float, default=0.5)
    ap.add_argument("--contradict-threshold", type=float, default=0.5)
    ap.add_argument("--judge", default="roberta-large-mnli")
    ap.add_argument("--judge-threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results")
    ap.add_argument("--run-name", default=None)
    if known.config:
        import yaml
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in (yaml.safe_load(Path(known.config).read_text()) or {}).items()})
    args = ap.parse_args(argv)
    if args.verifier == args.judge:
        ap.error("verifier and judge must be different models (circular evaluation)")
    run = Path(args.out) / (args.run_name or f"h2_rag_{args.split_name}_{int(time.time())}")
    run.mkdir(parents=True, exist_ok=True)
    timings, models = {}, {}

    # ---- question set ---------------------------------------------------------------------
    split = load_qasper(args.qasper)
    pool = [q for q in split.questions if q.gold_union and q.majority_type in ("extractive", "abstractive")]
    rng = np.random.default_rng(args.seed)
    qs = [pool[i] for i in sorted(rng.choice(len(pool), size=min(args.n_questions, len(pool)), replace=False))]
    print(f"H2 question pool {len(pool)} -> sampled {len(qs)} ({args.split_name})", flush=True)

    # ---- 1. retrieval -----------------------------------------------------------------------
    t = time.time()
    docs = load_qasper_documents(args.qasper, paper_ids=sorted({q.paper_id for q in qs}))
    chunks = [c for d in docs for c in chunk_document(d)]
    titles = {d.doc_id: d.title for d in docs}
    texts = [c.index_text(titles[c.doc_id]) for c in chunks]
    systems, base, rmeta = build_systems([args.retriever], texts, RetrievalConfig(), log=lambda m: print(m, flush=True))
    prime_queries(base, [q.question for q in qs])
    index = ChunkIndex(systems[args.retriever], chunks, titles, name=args.retriever, prefit=True)
    retrieved = {q.qid: index.search(q.question, k=args.k, doc_ids=[q.paper_id]) for q in qs}
    timings["retrieval_s"] = round(time.time() - t, 1)
    models["retriever"] = rmeta.get("models", {})
    del systems, base
    free()

    # ---- 2. generation (cached) -------------------------------------------------------------
    gen_path = run / "generations.jsonl"
    cache = {}
    if gen_path.exists():
        for line in gen_path.read_text().splitlines():
            r = json.loads(line)
            cache[(r["qid"], r["prompt_hash"])] = r["raw"]
    need = [q for q in qs if (q.qid, prompt_hash(build_messages(q.question, retrieved[q.qid]))) not in cache]
    t = time.time()
    if need:
        from researchlens.generation.llm_rag import HFChatLLM
        llm = HFChatLLM(args.generator, max_new_tokens=args.max_new_tokens)
        models["generator"] = llm.meta
        with open(gen_path, "a") as f:
            for i, q in enumerate(need):
                msgs = build_messages(q.question, retrieved[q.qid])
                t0 = time.time()
                raw = llm(msgs)
                rec = {"qid": q.qid, "paper_id": q.paper_id, "prompt_hash": prompt_hash(msgs), "raw": raw,
                       "seconds": round(time.time() - t0, 2), "model": llm.meta}
                cache[(q.qid, rec["prompt_hash"])] = raw
                f.write(json.dumps(rec) + "\n"); f.flush()
                if i % 10 == 0:
                    print(f"  generated {i + 1}/{len(need)} ({rec['seconds']}s)", flush=True)
        del llm
        free()
    else:
        models["generator"] = {"model": args.generator, "note": "replayed from cached generations.jsonl"}
    timings["generation_s"] = round(time.time() - t, 1)
    gen = LLMRAGGenerator(llm=None, name=f"rag:{args.generator}")
    answers = {"rag": {q.qid: gen.generate(q.question, retrieved[q.qid],
                                           raw=cache[(q.qid, prompt_hash(build_messages(q.question, retrieved[q.qid])))])
                       for q in qs}}

    # ---- 3. verification conditions ----------------------------------------------------------
    t = time.time()
    from researchlens.generation.verified import VerifiedReviser
    from researchlens.verification.nli import HFNLIScorer, NLIVerifier
    vs = HFNLIScorer(args.verifier)
    models["verifier"] = vs.meta
    verifier = NLIVerifier(vs, granularity=args.verifier_granularity, support_threshold=args.support_threshold,
                           contradict_threshold=args.contradict_threshold)
    actions = {}
    for cond, spec in CONDITIONS.items():
        if spec is None:
            continue
        mode, reassign, drop = spec
        rev = VerifiedReviser(verifier, mode=mode, reassign=reassign, drop_unsupported=drop)
        answers[cond], actions[cond] = {}, {}
        for q in qs:
            answers[cond][q.qid], actions[cond][q.qid] = rev.revise(answers["rag"][q.qid])
    timings["verification_s"] = round(time.time() - t, 1)
    del vs, verifier
    free()

    # ---- 4. judge + answer F1 ----------------------------------------------------------------
    t = time.time()
    js = HFNLIScorer(args.judge)
    models["judge"] = js.meta
    judge = CitationJudge(js, threshold=args.judge_threshold)
    per_query = {}
    for cond, by_q in answers.items():
        rows = []
        for q in qs:
            a: Answer = by_q[q.qid]
            m = judge.evaluate(a)
            text = " ".join(s.text for s in a.sentences)
            acts = [x["action"] for x in actions.get(cond, {}).get(q.qid, [])]
            rows.append({"qid": q.qid, "paper_id": q.paper_id, **m, "answer_f1": answer_f1(text, q.answers),
                         "n_kept": acts.count("kept"), "n_pruned": acts.count("pruned"),
                         "n_reassigned": acts.count("reassigned"),
                         "n_dropped": sum(a_.startswith("dropped") for a_ in acts),
                         "answer": text, "citations": [s.citations for s in a.sentences]})
        per_query[cond] = rows
    timings["judge_s"] = round(time.time() - t, 1)
    del js
    free()

    # ---- statistics: paired, clustered by paper ---------------------------------------------
    groups = [q.paper_id for q in qs]
    comparisons = []
    for metric in ["citation_precision", "citation_recall", "answer_f1"]:
        base_scores = {"rag": [r[metric] for r in per_query["rag"]]}
        for cond in ["verified", "ablation:no_reassign", "ablation:no_drop"]:
            for c in compare_many(base_scores, cond, [r[metric] for r in per_query[cond]], metric, groups=groups):
                comparisons.append(c.to_dict())
    extra = {"models": models, "timings_s": timings, "n_pool": len(pool),
             "abstained": {c: int(sum(r["abstained"] for r in rows)) for c, rows in per_query.items()}}
    write_run(run, {**vars(args), "dataset": "QASPER v0.3", "n_questions": len(qs),
                    "question_ids": [q.qid for q in qs],
                    "stats": "paper-clustered bootstrap (10k) + sign-flip; NaN (fully abstained) rows dropped pairwise"},
              per_query, comparisons, extra)
    summary = json.loads((run / "summary.json").read_text())
    print(f"\n{'condition':<24}{'cit_P':>8}{'cit_R':>8}{'ansF1':>8}{'abst':>6}{'dropped':>9}{'reassign':>9}")
    for c, s in summary.items():
        print(f"{c:<24}{s['citation_precision']:>8.3f}{s['citation_recall']:>8.3f}{s['answer_f1']:>8.3f}"
              f"{extra['abstained'][c]:>6}{s['n_dropped']:>9.2f}{s['n_reassigned']:>9.2f}")
    for c in comparisons:
        print(f"  {c['metric']:<20} {c['system_b']:<22} Δ={c['delta']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] "
              f"p={c['p_randomization']:.4f} n={c['n']}")
    print("timings", timings, "\nwrote", run)


if __name__ == "__main__":
    main()
