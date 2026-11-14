# ResearchLens X

**Evidence-aware scientific literature understanding and cross-paper reasoning with hybrid NLP**
B.Tech E&TC final-year project by Pranshu Kumar.

Upload research PDFs, search them, ask questions across them with **page-level citations**, check claims with an **NLI model**, compare papers, explore a **provenance-grounded knowledge graph**, see **candidate research gaps** and export a **cited literature review**. It runs locally, needs no API key, and never writes text with a generative model: answers are sentences copied from the papers, and every claim points to a paper, section, page and quote.

It is also a research framework. Two hypotheses were pre-registered: **H1 (hybrid retrieval beats every single retriever) was tested once on the QASPER test split and not supported**; H2 (claim-level verification improves citation correctness) was piloted on dev only and is partially supported, not confirmed. Negative and inconclusive results are reported as such.

## What works (all tested)
| Capability | Where |
|---|---|
| PDF ingestion with sections, per-paragraph pages, references | `ingestion/pymupdf_ingestor.py` |
| Persistent multi-paper library (duplicates, failures reported) | `store/corpus.py` |
| Retrieval: BM25, TF-IDF, LSA, e5, SPECTER2, SciBERT, RRF, linear fusion (experiments); **in the app: BM25, TF-IDF, e5 and RRF(BM25+e5)**, optional cross-encoder rerank, redundancy suppression, section filters | `retrieval/`, `workspace.py` |
| Multi-document QA with references and visible abstention | `qa/engine.py` |
| NLI claim verification: status, explanation, calibrated confidence (app default = the validated passage-level configuration); combined multi-passage evidence is experimental and uncalibrated | `verification/`, `evaluation/calibration.py` |
| Knowledge graph with evidence on every edge (mostly an entity graph: few paper-to-paper edges, cross-paper NLI links are gated candidates) | `knowledge_graph/` |
| Comparison table, candidate gaps, cited review (Markdown/HTML/PDF) | `analysis/` |
| Streamlit app with MOCK/fallback badges | `app/`, `workspace.py` |
| Evaluation framework: paper-clustered bootstrap, randomisation, Holm, McNemar, ECE | `evaluation/`, `experiments/` |

## Headline results (real runs; see `results/` and `PROJECT_BIBLE.md` §G)
| Experiment | Result |
|---|---|
| **H1 final test**, QASPER test, R@5 | e5 **.501** · RRF(BM25+e5) .486 · BM25 .415. RRF vs e5 Δ −.015 [−.032, +.002], p_holm .10: **not supported** |
| Cross-encoder rerank, QASPER **dev** (within-paper, ~45 candidate paragraphs) | R@5 .502 → .580 on the hybrid (+.078, p_holm .0003); dev only |
| NLI verifier, SciFact dev macro-F1 | LR .518 → DeBERTa passage-level .681 |
| Confidence calibration (dev, out-of-fold) | ECE .183 → .067 |
| H2 verified RAG, QASPER **dev** pilot (100 Q, 1.5B generator) | citation precision .420 → .785 (77 paired questions), but Answer-F1 −.060 fails the 2-pt guard; **partially supported, not confirmed**; test split not run |

Limitations are in [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md). Gaps are collection-scoped and never claimed novel; extraction is rule-based; Colab notebooks were not executed here.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dense.txt -r requirements-app.txt
pip install -e .
python -m pytest -q                          # 129 tests with all three requirements files (100 + skips with requirements.txt only), CPU, offline
streamlit run app/streamlit_app.py           # the app (first NLI/rerank use downloads models)
./scripts/run_eval.sh                        # tests + CPU experiments; add --full for the model-based dev runs
```
Walk-through: [docs/DEMO_WORKFLOW.md](docs/DEMO_WORKFLOW.md). Data: `./scripts/download_qasper.sh`, `./scripts/download_scifact.sh` (QASPER CC BY 4.0, SciFact CC BY-NC 2.0; raw data and PDFs are not redistributed).

## Documents
| Doc | Content |
|---|---|
| [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) | the report |
| [ARCHITECTURE.md](ARCHITECTURE.md), [TECHNICAL_DECISIONS.md](TECHNICAL_DECISIONS.md) | design and decisions |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), [DATASET_REGISTER.md](DATASET_REGISTER.md) | experiments and data |
| [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) | honest limits |
| [PROJECT_BIBLE.md](PROJECT_BIBLE.md), [PROJECT_STATE.md](PROJECT_STATE.md), [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) | plan, state, roadmap |
| [docs/REPOSITORY_AUDIT.md](docs/REPOSITORY_AUDIT.md), [MODEL_USAGE_LEDGER.md](MODEL_USAGE_LEDGER.md) | audit and model-usage ledger |
