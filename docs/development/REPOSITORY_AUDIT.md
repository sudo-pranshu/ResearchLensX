# Repository audit

Audit date 2026-11-14 · branch `chore/open-source-release` (from `develop` at `a731759`, PRs #23 and #24 already merged by the maintainer) · `/Users/pranshu/entre/researchlens-x`.
Everything here was read or measured in this repository unless marked **unverified**. Hardware for all timings: Apple M2, 8 GB RAM, 29 GB free disk, Python 3.12.11, Node v25.9.0 / npm 11.12.1 (pnpm not installed).

## 1. Baseline
| Check | Result |
|---|---|
| Tests, `.venv` | **133 passed** in 7.0 s (re-established today, also after the file moves) |
| Tests, base Python | 101 passed, 5 skipped (modules needing PyMuPDF/Streamlit skip) |
| Code | `src/researchlens` 4,176 lines (26 modules reachable from the service layer, about 900 lines used only by experiments: dataset loaders, evaluation statistics, the classical baseline) · Streamlit app 530 lines · tests 1,563 lines (16 files) · experiments 848 lines |
| Git | `main` = original scaffold (42 commits behind), `develop` = default branch and holds all implementation (PR #24 merged 2026-11-14 07:53 UTC), `feature/*` branches merged. Open work: branch `chore/open-source-release` (uncommitted restructure, governance, launcher) |
| Startup today | `./run.sh` (new, tested) starts one Streamlit process that contains UI and engine |

## 2. Functionality inventory
| Class | Items | Evidence |
|---|---|---|
| **Fully working** (tested, exercised on 5 real arXiv PDFs) | PDF ingestion with page provenance; SQLite library (duplicates, failures, filters, thread-safe); BM25, TF-IDF, e5, RRF(BM25+e5) in the app; redundancy suppression; section filters; extractive multi-document QA with references and abstention; passage-level NLI verification with explanations and calibrated confidence; knowledge graph with provenance; comparison table; candidate gaps; cited literature review with Markdown/HTML/PDF export; evaluation framework and committed results | 133 tests; browser checks; `results/` |
| **Partially implemented** | Cross-encoder rerank (works, measured only on dev); research-gap detection (rules tuned on 5 papers, no independent precision); reference splitting (counts approximate); two-column/table text (noisy); the app has no job/progress model, no stored originals, no incremental indexing | measured, see section 4 |
| **Experimental** | Combined multi-passage evidence (sentence-level, uncalibrated, never evaluated); NLI-linked cross-paper relations (relevance-gated candidates; 0 edges on the 5 unrelated papers); LLM generators (`generation/llm_rag.py`, `verified.py`) used only by H2 experiments | EXP-H2-004, EXP-KG-001 |
| **Mock** | `MockVerifier` (word overlap) is a labelled fallback shown with a red MOCK banner; `scripts/demo.py` uses it; synthetic fixtures behind the Library's "load fixtures" expander | `mocks.py`, `Workspace.verifier` |
| **Broken** | Nothing known broken. Not run: `scripts/run_h1_ablations.sh`, both Colab notebooks | KNOWN_LIMITATIONS |
| **Missing for the target product** | HTTP API; modern frontend; job/progress model; stored original PDFs and page preview; incremental indexing; settings screen; frontend, API and browser tests | none exist |

## 3. Measured performance (M2, 8 GB; 5 papers, 0.5-4 MB each)
| Operation | Measured | Note |
|---|---|---|
| Ingest one PDF | 0.19-0.96 s | 13-30 sections each; fast enough to run inline, but a job model is still needed for large batches |
| BM25 first query | 1.2 s | index build for the whole library; warm 2 ms |
| e5 first query | 10.5 s | model load + FAISS build from cached embeddings; warm ≈0.53 s per new query (query encoding) |
| BM25 + cross-encoder first query | 7.7 s | reranker load; warm ≈0.23 s |
| Knowledge graph, gaps, review | 0.05-0.10 s each | rule-based; cheap on 5 papers (scaling not measured) |
| Peak memory, one process with e5 + reranker (+ NLI in an earlier run) | ≈0.5-0.7 GB RSS | measured twice, not on a large corpus |
| First embedding of a **new** corpus | not re-measured today | historical: dev-set embedding with e5 took ≈530 s before batching changes |
Never invented: no speed-up is claimed anywhere; section 6 of LOCAL_ARCHITECTURE lists what will be measured before and after each optimisation.

## 4. Architectural weaknesses (verified in code)
| # | Weakness | Evidence | Consequence |
|---|---|---|---|
| A1 | No HTTP API; the UI imports the engine in-process | `apps/streamlit` calls `Workspace` directly | cannot host a separate frontend; no clean boundary for tests |
| A2 | **Embedding cache is keyed on the entire text list** | `retrieval/embedding_cache.py` (`texts_digest` of all texts) | adding one paper to the library re-embeds everything: cost O(library) on every upload |
| A3 | **Indexes are rebuilt, not persisted** | `Workspace._engines.clear()` on upload/remove; BM25 and TF-IDF fit per engine; FAISS built from cache | first query after any change pays the full rebuild |
| A4 | **Original PDFs are discarded** after parsing | `Workspace.upload` writes to a temp dir | no page preview, no re-processing after parser improvements, no export of originals |
| A5 | Uploads parse synchronously inside the call; no progress or per-document status | `Workspace.upload` | UI cannot show processing progress or retry |
| A6 | Models are created inside engine modules | `HFNLIScorer`, `SentenceTransformerEncoder`, `CrossEncoderScorer` | no single place to cap memory, lazy-load or later swap for hosted inference |
| A7 | Paths and settings partly hard-coded | `RLX_DB` is the only env var; embedding cache default `artifacts/embeddings` is relative to the package | needs central configuration |
| A8 | Local assumptions are scattered (single user, one global library) | `Workspace` owns one store | later multi-user needs a workspace id parameter now |
| A9 | `Workspace` mixes service orchestration, caching and verifier policy | 205 lines | split into services behind the API |

## 5. Redundant or stale code
- `scripts/demo.py` duplicates what the app does and uses the mock; replace with a smoke script or remove after the API exists.
- `app` (Streamlit) will be redundant once the new UI reaches parity; keep until the end-to-end suite passes on the new UI, then archive.
- `build_pipeline`/`RevisingPipeline` in `pipeline.py` serve experiments only; the app uses `build_default_pipeline` and `Workspace.engine` (two construction paths for one retriever stack).
- `verification/classical.py`, `generation/llm_rag.py`, `generation/verified.py`, `data/`, `evaluation/` are experiment-only (about 900 lines): keep, but move behind an `experiments` extra when packaging.

## 6. Documentation inconsistencies found
| File | Problem | Fix (planned, phase 11) |
|---|---|---|
| `docs/research/PROJECT_BIBLE.md` abstract and section C | says the hybrid is BM25 + **SPECTER2**; the selected dense model is **e5** (ADR-009) and SPECTER2 is not used by the app | correct to e5, keep SPECTER2 as an evaluated alternative |
| `docs/architecture/ARCHITECTURE.md` and ADR-003 | describe GROBID as an optional primary parser and a fallback matrix row; **GROBID is not implemented** | mark as not implemented / future |
| `docs/research/RESEARCH_DESIGN.md` | the original v0.1 design with a day-by-day plan and thesis wording | keep as a historical record with a banner pointing to the Bible |
| `docs/research/EXPERIMENT_PROTOCOL.md`, `DATASET_REGISTER.md` | "Day 1", "once, Day 7" scheduling language | neutral wording |
| `README.md` (root) | pre-restructure; still contains course-project wording | rewritten in phase 11 with the new hierarchy |
| `CLAUDE.md` | assistant instruction file; keep at root, paths updated | maintained |
Link and wording gate: `scripts/check_docs.py` (broken links and stale framing), run in CI.

## 7. Compatibility constraints for the migration
- `researchlens` imports, `configs/*.yaml` paths and `results/EXP-*` folders are referenced by committed provenance (`config.json` records git hashes and paths) and by the experiment register. **They must not move or be renamed**; the target structure is applied around them (see LOCAL_ARCHITECTURE §8).
- `tests/` currently imports `tests.test_ingestion` helpers; the unit/integration/e2e split must keep those imports working.
