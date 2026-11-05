# ResearchLens X: Project Bible

**Evidence-Aware Scientific Literature Understanding and Cross-Paper Reasoning Using Hybrid NLP**

B.Tech E&TC final-year project · Team: Pranshu Kumar (lead), Pawan, Gaurav · Version 1.0 · 2026-11-05 (Day 1 of 14) · Submission freeze: **2026-11-18 (Day 14)**

This document is the single source of truth. The other documents expand on one section each:

| Section | Detailed in |
|---|---|
| C Architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| D Team | [TEAM_RESPONSIBILITIES.md](TEAM_RESPONSIBILITIES.md), [docs/development/](docs/development/) |
| B/G Experiments | [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) |
| F Datasets | [DATASET_REGISTER.md](DATASET_REGISTER.md) |
| H Conventions | [CONTRIBUTING.md](CONTRIBUTING.md) |
| J Status | [PROJECT_STATE.md](PROJECT_STATE.md) |

If a detailed document disagrees with this one, **this one wins** and the other must be fixed in the same PR that notices the conflict.

---

## A. Project overview

**Abstract.** ResearchLens X ingests scientific papers, retrieves evidence with a hybrid of lexical (BM25/TF-IDF) and scientific dense (SPECTER2) retrieval fused by Reciprocal Rank Fusion, and answers questions with passage-level citations. Every generated claim is checked against its cited evidence by an NLI verifier, which labels it SUPPORTS / CONTRADICTS / INSUFFICIENT. A knowledge graph and per-paper comparison view expose relationships across papers. Two hypotheses are evaluated against conventional baselines on QASPER and SciFact with pre-registered metrics and cluster-aware significance tests: hybrid retrieval (H1) and claim-level verification (H2). Cross-paper reasoning (H3) is explored qualitatively.

**Problem.** LLM-based literature assistants produce fluent answers whose citations often do not support the claims attached to them. Single-method retrievers also miss evidence: lexical retrieval fails on paraphrase, dense retrieval on exact technical terms.

**Motivation.** Researchers need answers they can *audit*: each claim traceable to a passage, with an explicit signal when evidence is missing or contradictory.

**Research question.** Can a hybrid architecture combining scientific-domain retrieval, claim-level verification and evidence attribution improve the factual reliability and evidence coverage of scientific QA compared with conventional RAG baselines?

**Objectives (14-day scope).**
1. Working end-to-end system: PDF → structured document → hybrid retrieval → cited answer → claim verification → KG + comparison UI.
2. Rigorous H1 evaluation (QASPER within-paper; SciFact open-corpus).
3. Rigorous H2 evaluation (QASPER test; SciFact verifier evaluation; human evaluation on 50 questions).
4. Exploratory H3 case studies.
5. Reproducible artefacts: configs, commits, cached generations, per-query results.

**In scope:** items 1–10 of the 14-day brief. **Out of scope:** training large models, cross-encoder reranking (stretch only), layout models, multi-domain benchmarks, a quantitative H3 claim.

**Expected contributions** (stated modestly, see review R14):
- C1. A controlled comparison of lexical, dense and hybrid retrieval for scientific evidence retrieval, using cluster-aware statistics.
- C2. A controlled comparison of conventional RAG vs claim-verified RAG on citation precision/recall, with judge validation and human evaluation.
- C3. An integrated, reproducible, CPU-runnable application.

The components themselves (BM25, SPECTER2, RRF, NLI verification) are established techniques. We do **not** claim new algorithms.

## B. Research specification

Full procedures: [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). Changes from v0.1, with reasons: [docs/RESEARCH_DESIGN_REVIEW.md](docs/RESEARCH_DESIGN_REVIEW.md).

**Decision rule (all confirmatory tests):** a hypothesis is *supported* if the primary metric improves on **every** named baseline with Holm-adjusted p < 0.05 **and** the 95% CI of the improvement excludes 0. Otherwise it is *not supported*, and we report that.

### H1 (primary): Hybrid lexical–semantic retrieval beats lexical-only and dense-only
| | |
|---|---|
| Treatment (pre-registered) | RRF(k=60, depth=100) of BM25 + best dense retriever D*. D* is chosen on QASPER **dev** by R@5 among {SPECTER2, e5-base-v2, SciBERT-mean} |
| Baselines | BM25, TF-IDF, LSA, D* alone |
| Test 1 | QASPER **test**, within-paper evidence-paragraph retrieval, gold = union of annotators' evidence (headings excluded). Primary metric **Recall@5** |
| Test 2 | SciFact abstract retrieval (claims → 5,183 abstracts), dev set (test labels hidden). Primary metric **nDCG@10** |
| Statistics | QASPER: paper-level cluster bootstrap CI (10k) + cluster sign-flip randomisation, Holm over baselines. SciFact: claim-level bootstrap + randomisation |
| Secondary (no confirmatory claim) | Linear fusion, RRF k ∈ {10,30,60,100}, depth {20,50,100}, section prefix on/off, title-conditioned open-corpus QASPER |
| Stretch | Cross-encoder reranking on top of the hybrid, a separate factor that is never mixed into the H1 claim |

### H2 (primary): Claim-level verification improves citation correctness
| | |
|---|---|
| Baseline | Conventional RAG: hybrid top-5 → open-weight LLM → answer with `[n]` citations. Fixed prompt, greedy decoding |
| Treatment | The same generation → sentence-level claims → NLI verifier on (claim, cited passage) and (claim, every retrieved passage) → re-attribute to the best supporting passage; claims that are unsupported are dropped or marked "insufficient evidence" |
| Test set | QASPER test, answerable questions whose majority answer type is extractive/abstractive. Dev subset (200 Q) is used for prompt/threshold tuning |
| Primary metrics | **Citation precision** and **citation recall** (ALCE-style), judged by an NLI model **different from** the verifier |
| Guard metric | QASPER Answer-F1 (verification must not reduce it by >2 pts; reported either way) |
| Component evaluation | Verifier on SciFact dev (label macro-F1 given gold abstracts); classical baseline = TF-IDF + Logistic Regression |
| Human eval | 50 Q × 2 systems, blinded; 15-Q overlap annotated by all 3 (Fleiss' κ); judge–human agreement reported |
| Statistics | Paired bootstrap + randomisation on per-question citation P/R (cluster by paper); McNemar on per-claim supported/unsupported in human eval |
| Ablations | Verify-only (flag, no revision) vs verify+revise; verifier model; threshold |

### H3 (exploratory): Cross-paper reasoning
About 10 documented case studies using `Pipeline.compare` + KG (shared datasets/methods, support/contradiction edges). Reported qualitatively, with **no significance claims**.

### Validity threats and limitations
SPECTER2 document exposure to S2 papers (R13); LLM memorisation of public datasets (mitigated by the paired design, and absolute F1 is not claimed); annotators are system builders (blinding only); single dataset family (NLP papers) limits generality; small human-eval sample; judge-model errors (validated against humans); heading-evidence exclusion (R6).

## C. Architecture (summary)
```
PDF ─► Ingestor ─► Document ─► chunk_document ─► Chunk[] ─► ChunkIndex(BM25 | TF-IDF | SPECTER2+FAISS | RRF)
                                                               │ RetrievedPassage[]
                                                               ▼
                                Generator (extractive | LLM-RAG) ─► Answer ─► Verifier (NLI) ─► VerificationResult/sentence
                                                               │
                         GraphBuilder(Document[], VerificationResult[]) ─► KnowledgeGraph ─► Streamlit app
```
Contracts: `src/researchlens/contracts.py` (pydantic). Integration facade: `src/researchlens/pipeline.py`. The app talks only to `Pipeline`. **Degradation rule:** if GROBID, the GPU, an LLM or SPECTER2 is unavailable, the system falls back to PyMuPDF / CPU / extractive / BM25 and the UI says so. Mock components carry `is_mock=True`, and the UI must show a MOCK badge for them.

## D. Team (summary)
| Member | Role | Owns (code) | Hypothesis | Compute |
|---|---|---|---|---|
| **Pranshu Kumar** | Lead, research architect, retrieval | `contracts.py`, `pipeline.py`, `mocks.py`, `retrieval/`, `evaluation/{metrics,stats}.py`, `data/qasper.py`, `experiments/h1_*`, Colab notebooks | H1 | Colab Pro, Claude Pro |
| **Pawan** | NLP modelling & verification | `verification/`, `generation/`, `data/scifact.py`, `evaluation/attribution.py`, `experiments/h2_*` | H2 | Claude Pro, CPU (GPU jobs through Pranshu) |
| **Gaurav** | Document intelligence & application | `ingestion/`, `processing/`, `knowledge_graph/`, `app/` | H3 case studies (with Pranshu) | CPU only, open-source only |

Pranshu has final architectural approval. Disagreements are recorded in §I, never silently resolved.

## E. Technology stack
Python 3.11 (3.10+ supported) · NumPy/SciPy/scikit-learn · pydantic v2 · PyTorch + Hugging Face Transformers + sentence-transformers + `adapters` (SPECTER2) · FAISS (cpu) · PyMuPDF (primary PDF parser) · GROBID (optional, Docker) · NetworkX + Plotly/pyvis (KG) · **Streamlit** (frontend) · FastAPI (thin optional API over `Pipeline`) · pytest + GitHub Actions. Models are proposals until pinned by HF revision in a model card: `allenai/specter2_base`+adapters, `intfloat/e5-base-v2`, `allenai/scibert_scivocab_uncased`, NLI verifier/judge from DeBERTa-v3 MNLI-family checkpoints, and a 7–8B open-weight instruct LLM.

## F. Dataset register (summary)
| ID | Dataset | Version | Licence | Location | Use | Status |
|---|---|---|---|---|---|---|
| DS-01 | QASPER | v0.3 | CC BY 4.0 | `data/raw/qasper/` (not committed) | H1, H2, demo corpus | downloaded + audited Day 1 |
| DS-02 | SciFact | 2020 release | CC BY-NC 2.0 | `data/raw/scifact/` | H1 test 2, H2 verifier | to download (Pawan, Day 1) |

Full detail: [DATASET_REGISTER.md](DATASET_REGISTER.md).

## G. Experiment register
Each row needs a config file, a git commit and saved results. Rows are added by PR.

| ID | Hyp. | Config | Commit | Models | Data | Primary result | Date | Status |
|---|---|---|---|---|---|---|---|---|
| EXP-H1-001 | H1 (dev, classical) | `configs/h1_classical_dev.yaml` | `02120d9` | BM25, TF-IDF, LSA-256 | QASPER v0.3 dev, within-paper, 923 Q / 281 papers | R@5: BM25 .435 · TF-IDF .391 · LSA .427 · RRF(BM25+LSA) .450 (Δ vs BM25 +.015 [−.003, +.034], p_holm .11, **n.s.**) · linear(BM25+LSA) .461 (Δ +.026 [+.010, +.043], p_holm .002) | 2026-11-05 | ✅ done (dev; classical only, so not a test of H1) |
| EXP-H1-002 | H1 diagnostic | same, `--setting open` | `02120d9` | same | QASPER dev, open corpus (15,122 paragraphs) | nDCG@10: BM25 .142 · TF-IDF .104 · LSA .071 · RRF(BM25+LSA) .102 · linear .126. All hybrids are **worse** than BM25 | 2026-11-05 | ✅ done |
| EXP-H1-002b | H1 diagnostic | same, `--setting open --title-query` | `02120d9` | BM25, LSA | QASPER dev, open corpus, title + question | nDCG@10: BM25 .183 · LSA .135 · RRF .154. Title conditioning → 99% of the top-10 is from the right paper (review R2) | 2026-11-05 | ✅ done |
| EXP-H1-003 | H1 (dev, dense; selects D*) | `configs/h1_dense_dev.yaml` | `3531f78` | e5-base-v2 `f52bf8e`, SciBERT `24f92d3`, SPECTER2 base `3447645` + proximity/adhoc_query adapters | QASPER dev within-paper, 923 Q / 281 papers | R@5: e5 **.500** · BM25 .435 · SPECTER2 .288 · SciBERT .181 · **RRF(BM25+e5) .502**: Δ vs e5 +.002 [−.021, +.027] **n.s.**, vs BM25 +.067 [+.046, +.090] p_holm<.001 · linear(BM25+e5) .514 (vs e5 +.014, n.s.) | 2026-11-05 | ✅ done. D* = e5 (ADR-009). On dev the H1 decision rule is **not met** (hybrid ≯ D*) |
| EXP-H1-004 | H1 test 2 (open corpus) | `configs/h1_scifact_dev.yaml` | `d161678`+ | same models | SciFact dev, 188 claims with evidence, 5,183 abstracts | nDCG@10: e5 **.870** · RRF(BM25+e5) .855 (Δ −.015 [−.047, +.015], n.s.) · SPECTER2 .804 · BM25 .801 · RRF(BM25+SPECTER2) .841 · SciBERT .121 | 2026-11-05 | ✅ done. Replicates QASPER: RRF helps weaker components, not the strongest dense model |
| EXP-H2-001 | H2 component | `configs/h2_scifact_verifier.yaml` | `c3bf204`+ | V = MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli `6f5cf0a`; J = roberta-large-mnli `2a8f12d` | SciFact dev, 340 (claim, cited abstract) pairs; selection on train (919 pairs) | Macro-F1 [95% CI]: LR .518 [.46,.57] · V sentence .596 · V sentence+tuned .633 [.58,.69] (vs LR McNemar p=.026) · **V passage .681 [.62,.74]** (vs sentence p<.001) · J .565 | 2026-11-05 | ✅ done. The judge is weaker than the verifier, so judge-based H2 metrics need human validation |
| EXP-H1-005 | H1 secondary | `scripts/run_h1_ablations.sh` | — | — | QASPER dev | — | — | ⏳ not run (section prefix / window units / RRF k / open-corpus diagnostics) |
| EXP-H2-002 | H2 (dev **pilot**) | `configs/h2_rag_dev_pilot.yaml` | see run config | Qwen2.5-1.5B-Instruct (local), V, J | QASPER dev, 100 Q (extractive/abstractive, seed 0) | see PR | 2026-11-05 | ⏳ running |
| EXP-H1-006 | H1 **final test** | to write | — | — | QASPER test | — | — | ⏳ not run (issue #18) |

## H. Engineering conventions (summary)
Branches `main` (stable) ← `develop` (integration) ← `feature/<member>-<topic>`. PRs only; CI green; one review; owner of touched module approves. Conventional commits (`feat(retrieval): …`). No data, checkpoints, secrets or PDFs in git. Full rules: [CONTRIBUTING.md](CONTRIBUTING.md).

## I. Decisions log (ADRs)
| ID | Date | Decision | Alternatives | Reason | Owner |
|---|---|---|---|---|---|
| ADR-001 | 11-05 | Package layout `src/researchlens/<module>/` instead of top-level `src/ingestion` etc. | Top-level packages | Generic top-level names (`evaluation`, `generation`) collide with other installed packages; a single namespace is installable | Pranshu |
| ADR-002 | 11-05 | pydantic v2 contracts in one file | dataclasses; JSON Schema | Runtime validation, JSON round-trip, FastAPI-native | Pranshu |
| ADR-003 | 11-05 | PyMuPDF primary, GROBID optional | GROBID primary | GROBID needs Docker/Java; the app must run on CPU and degrade gracefully | Pranshu/Gaurav |
| ADR-004 | 11-05 | Streamlit frontend calling `Pipeline` in-process; FastAPI optional | React + FastAPI | 14 days, a Python-only team member, fewer moving parts | Pranshu/Gaurav |
| ADR-005 | 11-05 | QASPER open-corpus dropped as H1 primary; SciFact used as open-corpus test | Keep both | Review R2 (title conditioning: 25% → 99% right-paper hits) | Pranshu |
| ADR-006 | 11-05 | Cluster bootstrap by paper for QASPER | IID bootstrap | Review R5 | Pranshu |
| ADR-007 | 11-05 | All LLM generations cached and committed as JSONL | Regenerate on demand | Only Pranshu has a GPU; keeps H2 re-runnable on CPU | Pranshu/Pawan |
| ADR-008 | 11-05 | No paid API in the evaluated pipeline | Hosted LLM | Reproducibility and resource rules | Pranshu |
| ADR-009 | 11-05 | D* = e5-base-v2; the pre-registered H1 hybrid is RRF(BM25 + e5), k=60, depth=100 | SPECTER2 (expected), SciBERT | Pre-registered rule: best dense R@5 on QASPER dev (e5 .500 vs SPECTER2 .288 vs SciBERT .181, EXP-H1-003) | Pranshu |
| ADR-010 | 11-05 | H2 verifier uses passage-level premises, thresholds 0.5 | sentence-level (+tuned) | Best on SciFact dev (EXP-H2-001, p<.001). H2 itself is measured on QASPER, so no selection leakage | Pranshu/Pawan |
| ADR-011 | 11-05 | Experiments run one transformer at a time and set KMP_DUPLICATE_LIB_OK / OMP_NUM_THREADS=1 on macOS | — | 8 GB laptops thrash; torch+faiss libomp clash caused aborts/segfaults | Pranshu |

## J. Status
See [PROJECT_STATE.md](PROJECT_STATE.md). 14-day plan:

| Days | Dates | Goal | Exit criterion |
|---|---|---|---|
| 1–2 | 11-05→11-06 | Setup, contracts, datasets, baselines | Repo + CI green; QASPER & SciFact audited; classical H1 dev run committed; each member's first module skeleton passing tests |
| 3–4 | 11-07→11-08 | Independent modules | SPECTER2/e5 dev results; zero-shot NLI on SciFact dev; PyMuPDF ingestor on 10 PDFs |
| 5–6 | 11-09→11-10 | Integration | `Pipeline` with hybrid retriever + real verifier; Streamlit app shows real outputs |
| 7–8 | 11-11→11-12 | Evaluation | H1 test runs (once); H2 generations cached; H2 metrics; human eval done |
| 9–10 | 11-13→11-14 | Advanced features & UI | KG view, comparison view, H3 case studies |
| 11–12 | 11-15→11-16 | Testing & docs | Test coverage of every module; docs consistent; demo script |
| 13–14 | 11-17→11-18 | Report, slides, freeze | Tag `v1.0`; no code changes after the freeze except blocker fixes |
