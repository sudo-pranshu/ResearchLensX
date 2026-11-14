# ResearchLens X — Repository Audit (2026-11-14)

Scope: verified against the working tree on `feature/pranshu-h1-final-test` (HEAD `19a34de`, 2 commits unpushed).
Tests: `pytest -q` → **40 passed** (4.2 s, CPU, offline). Nothing below is claimed from memory; "Verified" means I read the code or ran it.

## 1. Implementation matrix (vs. the six target subsystems)

| Subsystem / capability | Status | Evidence |
|---|---|---|
| BM25 (from scratch), TF-IDF, LSA | **Done, tested, benchmarked** | `retrieval/lexical.py`, EXP-H1-001/002/006 |
| Dense (e5-base-v2, SPECTER2, SciBERT), FAISS, embedding cache | **Done, tested, benchmarked** | `retrieval/dense.py`, `embedding_cache.py`, EXP-H1-003/004 |
| RRF + linear fusion | **Done, benchmarked** | `retrieval/fusion.py`; hybrid does not beat e5 (H1 not supported, EXP-H1-006) |
| Cross-encoder reranking | **Implemented 2026-11-14, unevaluated** | `retrieval/rerank.py` |
| Metadata-aware filtering | **Partial**: document-level filters (ids, year, title) in `CorpusStore`; not yet exposed in retrieval/UI | `store/corpus.py` |
| Duplicate/redundant passage suppression | **Implemented** (lexical containment) | `retrieval/postprocess.py` |
| Document-level vs chunk-level retrieval | Partial | chunk-level only; SciFact is abstract-level in a separate script |
| Retrieval metrics: Recall@K, nDCG, MRR, paired stats, latency | Mostly done | `evaluation/metrics.py`, `stats.py`; **Precision@K and per-query-category breakdown not verified/absent**; latency is logged per run only |
| QA over multiple docs with citations | **Implemented 2026-11-14 (`qa/engine.py`); extractive generator only until a stronger LLM is used** | `generation/extractive.py`, `llm_rag.py` (Qwen2.5-1.5B), `Pipeline.ask`; no page numbers (Document model has no page info), no uncertainty/"insufficient evidence" abstention in generator |
| Claim verification (NLI) | **Done on benchmark** | `verification/nli.py` (DeBERTa-v3 + roberta-large-mnli judge), EXP-H2-001 SciFact macro-F1 .596–.681; no calibration, no multi-evidence aggregation beyond max-pool; no natural-language explanations |
| Verified-RAG reviser | **Done, result inconclusive** | `generation/verified.py`, EXP-H2-002 (generator emits 0.15 citations/answer → confounded) |
| PDF ingestion | **Implemented 2026-11-14, tested on synthetic PDFs and 5 real arXiv PDFs; reference splitting approximate** | `ingestion/pymupdf_ingestor.py`, `tests/test_ingestion.py` |
| Knowledge graph | **Implemented 2026-11-14** (rule-based extraction, provenance, interactive view; NLI links are gated candidates) | `knowledge_graph/`, app page |
| Research-gap engine | **Missing** (no module, no design doc) | — |
| Literature-review / report generator + export | **Missing** | — |
| Frontend (Streamlit) / FastAPI | **Missing** (`app/` has only a README); `streamlit`, `fitz` import-fail | — |
| Evaluation reproducibility | **Strong** | pinned configs, git hash + dataset version per run, `results/` committed, tests for stats |
| Mock verifier | Exists, flagged `is_mock`, used by the demo/app path only (not in evaluation) | `mocks.py` |

Honest headline: the **research core (H1/H2 retrieval + verification + evaluation) is real; the entire product layer (ingest → app → graph → gaps → report) does not exist yet.** Definition-of-Done items 1, 2, 7, 8, 9 cannot be demonstrated today; 3–6 and 10 work only on the benchmark corpora, not on uploaded PDFs.

## 2. Scientific status (real results, nothing new run today)

- H1 hybrid vs single retriever: **NOT supported** on QASPER test (RRF .486 vs e5 .501 R@5, p_holm .10) and SciFact dev. Hybrid does beat all lexical baselines.
- H2 verifier: SciFact dev macro-F1 .518 (LR) → .681 (DeBERTa, passage-level). Strong.
- H2 verified RAG: **inconclusive** (confounded by a generator that rarely cites; answer-F1 fell 3.7 pts).
- Note the directive's "dense beats BM25 / hybrid beats individuals / reranking vs none" experiment list: items 1, 2 are done; **3 (reranking) is not runnable until a reranker exists**.

## 3. Technical-debt report

1. **Team-era artefacts** (ownership matrix, per-member workflow docs, owner names in docstrings, README team list): **removed on 2026-11-14**; the project is solo.
2. **`Document` contract lacks page provenance** — page-level citation is a stated requirement; adding `page` to `Paragraph`/`Chunk` is a contract change (schema 1.0 → 1.1). Must be done before ingestion.
3. **Two experiment pipelines** (QASPER paragraph-level, SciFact abstract-level) duplicate loader/eval glue (`h1_retrieval.py`/`h1_scifact.py`). Acceptable (metrics must not be combined) but share a runner later.
4. **Naming mismatch with the directive's doc set**: required `EXPERIMENT_LOG.md`, `IMPLEMENTATION_ROADMAP.md`, `MODEL_USAGE_LEDGER.md`, `TECHNICAL_DECISIONS.md`, `KNOWN_LIMITATIONS.md`. Repo has `PROJECT_BIBLE.md` (register + ADRs), `PROJECT_STATE.md`, `EXPERIMENT_PROTOCOL.md`. Plan: add the missing ones as thin documents that point into the Bible instead of duplicating (no second source of truth).
5. `docs/adr/` is empty; ADRs live inside the Bible. Fine, but confirm one location.
6. Weak spots in generator: 1.5B Qwen cannot follow citation format → the app's QA quality is bounded until a bigger local model or Colab generation is used. This is the largest functional risk.
7. 8 GB RAM constraint: only one transformer resident at a time; cross-encoder + NLI + e5 in an interactive app need lazy loading/unloading and a CPU latency budget.
8. Unpushed work: 2 local commits. Remote `main`/`develop` clean of Claude attribution (verified earlier).
9. No CI check for the app layer yet; no lint config verified today.
10. Dependencies: `pymupdf`, `streamlit`, `pyvis` not installed (install was interrupted earlier).

## 4. Schedule conflict you need to decide on

The directive asks for a 14-day plan; PROJECT_STATE records **submission freeze 2026-11-18 (4 days from today)**. These cannot both hold. Realistic plan below assumes either (a) the freeze moves, or (b) we ship a **4-day MVP** (items 1–6 of Definition of Done on real PDFs, a Streamlit UI, and a report generator) and explicitly list the rest as future work.

## 5. Prioritised roadmap

**Tier A — end-to-end MVP (Sonnet, ~4 days if freeze holds):**
A1. Install `pymupdf streamlit pyvis`; implement `PyMuPDFIngestor` with page numbers (contract bump 1.1 + tests using self-generated PDFs). 
A2. Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, small) + dedup + metadata filter in retrieval; one small evaluation run (EXP-H1-007 rerank vs none, dev only).
A3. Streamlit app on `Pipeline` facade: upload, search, cited QA, claim check, visible MOCK/fallback badges.
A4. Rule-based entity/relation extraction → NetworkX graph + pyvis view; only extracted edges.
A5. Gap & review report generator (extractive, evidence-linked, "within-collection" wording enforced) with Markdown/HTML export.

**Tier B — science (if time):** H2 rerun with a few-shot citation prompt / Colab 7–8B generator (fixes the confound), H1 ablations (`run_h1_ablations.sh`), calibration of the verifier (temperature scaling on dev), per-query-type retrieval breakdown, Precision@K.

**Tier C — polish:** dark/light theme, FastAPI wrapper, Colab notebooks, technical report, solo-doc cleanup.

## 6. Files to create / modify

Create: `src/researchlens/retrieval/rerank.py`, `retrieval/dedup.py`, `ingestion/` (real implementation), `knowledge_graph/extract.py`, `analysis/gaps.py`, `analysis/review.py`, `app/streamlit_app.py`, `app/components/*`, tests for each, `IMPLEMENTATION_ROADMAP.md`, `MODEL_USAGE_LEDGER.md`, `TECHNICAL_DECISIONS.md`, `KNOWN_LIMITATIONS.md`, `EXPERIMENT_LOG.md`, `configs/h1_rerank_dev.yaml`.
Modify: `contracts.py` (page provenance), `processing/chunking.py`, `pipeline.py` (reranker, filters), `retrieval/registry.py` (rerank spec), `requirements*.txt`, `PROJECT_STATE.md`, `CLAUDE.md`, `README.md`.
Do **not** touch: retrieval/evaluation code that produced committed results (frozen; only additive changes).

## 7. Model allocation (see `MODEL_USAGE_LEDGER.md`)

Sonnet: everything in Tier A. Opus: one review of the gap-discovery methodology and one of the NLI calibration/explanation design. Fable (5 h cap, 0 h used): reserve for (1) adversarial review of the verified-RAG + gap-discovery scientific claims before the report, (2) final QA. Nothing in Tier A needs it.
