# ResearchLens X

**Evidence-Aware Scientific Literature Understanding and Cross-Paper Reasoning Using Hybrid NLP**
B.Tech E&TC final-year project: Pranshu Kumar (lead), Pawan, Gaurav.

ResearchLens X answers questions over scientific papers with **hybrid retrieval** (BM25 + SPECTER2, fused by Reciprocal Rank Fusion) and **passage-level citations**. Each claim is **verified with NLI** as SUPPORTS / CONTRADICTS / INSUFFICIENT. A **knowledge graph** and **multi-paper comparison** view sit on top. The research contribution is a controlled evaluation of hybrid retrieval (H1) and claim-level verification (H2) on QASPER and SciFact.

📘 Start with **[PROJECT_BIBLE.md](PROJECT_BIBLE.md)**, the single source of truth.

| Doc | What it holds |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | components, contracts, fallbacks |
| [TEAM_RESPONSIBILITIES.md](TEAM_RESPONSIBILITIES.md) | ownership matrix, acceptance criteria |
| [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) | how runs are executed and reported |
| [DATASET_REGISTER.md](DATASET_REGISTER.md) | datasets, licences, audits |
| [PROJECT_STATE.md](PROJECT_STATE.md) | current status |
| [CONTRIBUTING.md](CONTRIBUTING.md) | setup, branches, PRs |
| [docs/development/](docs/development/) | per-member workflows |
| [docs/RESEARCH_DESIGN_REVIEW.md](docs/RESEARCH_DESIGN_REVIEW.md) | v0.1 → v1.0 review |

## Quick start (CPU, no GPU or API keys)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q
python scripts/demo.py "Which datasets are used for evaluation?"
./scripts/download_qasper.sh
python scripts/demo.py --qasper data/raw/qasper/qasper-dev-v0.3.json --papers 30 "What baselines do they compare to?"
python experiments/h1_retrieval.py --config configs/h1_classical_dev.yaml
```

## Status (Day 1)
| Component | State |
|---|---|
| Contracts, Pipeline facade, mocks, fixtures | ✅ tested |
| BM25 / TF-IDF / LSA / RRF / linear fusion | ✅ tested; H1 classical dev runs done |
| Clustered bootstrap / randomisation / Holm | ✅ tested |
| QASPER loader + audit | ✅ |
| Extractive CPU generator | ✅ tested (weak baseline) |
| SPECTER2 / e5 / SciBERT encoders | ⚠️ written, not yet executed |
| NLI verifier, LLM RAG, SciFact | ⏳ Pawan, Days 1–4 |
| PDF ingestion, KG, Streamlit app | ⏳ Gaurav, Days 1–10 |

The verifier currently in the demo is a **mock** (word overlap) and is labelled as such in all outputs.

## Data licences
QASPER: CC BY 4.0. SciFact: CC BY-NC 2.0 (non-commercial). Raw data is not redistributed in this repository.
