# Implementation Roadmap: 14-day solo plan

Developer: Pranshu Kumar (sole). Window: **2026-11-14 → 2026-11-27** (freeze moved from 11-18).
Baseline: see [docs/REPOSITORY_AUDIT.md](docs/REPOSITORY_AUDIT.md). Rule: working end-to-end pipeline before optional features. Each day ends with `pytest -q` green and PROJECT_STATE.md updated. Estimates are hours of focused work.

| Day | Date | Objectives and tasks | Files | Tests / deliverable | Est. h |
|---|---|---|---|---|---|
| 1 | 11-14 | Solo cleanup (done). Isolated venv with app deps. Contract 1.1: page provenance. `PyMuPDFIngestor` | `contracts.py`, `ingestion/pymupdf_ingestor.py`, `processing/chunking.py` | Tests generate their own PDFs; page numbers survive to chunks | 6 |
| 2 | 11-15 | Finish ingestion (headings, header/footer removal, references); multi-PDF corpus store (SQLite) | `ingestion/`, `store/corpus.py` | Round-trip tests; ingest 5 real open-access PDFs | 6 |
| 3 | 11-16 | Retrieval additions: cross-encoder reranker, metadata filter, dedup (MMR/near-dup) | `retrieval/rerank.py`, `dedup.py`, `registry.py` | Unit tests offline; spec strings `rerank:...` | 6 |
| 4 | 11-17 | EXP-H1-007 rerank vs none (dev only); Precision@K, MRR, latency, per-query-type breakdown | `experiments/h1_retrieval.py` (additive), `configs/h1_rerank_dev.yaml` | Committed results + register row | 6 |
| 5 | 11-18 | Multi-document QA: evidence aggregation across papers, abstention on insufficient evidence, page-level citations | `generation/`, `pipeline.py` | Tests: no fabricated citation; abstention case | 6 |
| 6 | 11-19 | Verification upgrades: multi-evidence aggregation, contradiction surfacing, templated explanations, temperature scaling on dev | `verification/nli.py`, `calibration.py` | ECE on SciFact dev reported; tests | 6 |
| 7 | 11-20 | Streamlit app v1: upload, library, search, cited QA with evidence panel, MOCK/fallback badges | `app/` | Smoke test via `streamlit.testing`; manual demo | 7 |
| 8 | 11-21 | Knowledge graph: rule-based entity/relation extraction; edges only from extraction or verified claims; pyvis view | `knowledge_graph/` | Edge-provenance tests | 6 |
| 9 | 11-22 | Cross-paper comparison tables and claim check UI; retrieval comparison view | `analysis/compare.py`, `app/` | Tests; screenshot walk-through | 6 |
| 10 | 11-23 | Research-gap engine (within-collection wording enforced) and literature-review report with export (Markdown/HTML) | `analysis/gaps.py`, `analysis/review.py` | Every sentence traceable; hand-reviewed examples | 7 |
| 11 | 11-24 | H2 rerun on dev with fixed citation prompt (Colab generator if needed); H1 ablations | `experiments/h2_rag.py`, notebooks | Registered results, positive or negative | 6 |
| 12 | 11-25 | H2 final test (once) if dev criteria met; evaluation suite one-command runner | `scripts/run_eval.sh` | Reproducible from clean clone | 6 |
| 13 | 11-26 | Theming/polish, empty/error states, Colab notebooks, sample data, KNOWN_LIMITATIONS, TECHNICAL_DECISIONS | `app/`, docs | Full regression | 6 |
| 14 | 11-27 | Technical report, README, demo workflow, tag `v1.0`, freeze | docs | Fresh-clone install + demo run | 6 |

## Gates
- No test-split run unless the dev criterion in EXPERIMENT_PROTOCOL.md is met.
- Any day that slips pushes optional items (FastAPI, GROBID, ablations) out, never the end-to-end path.
- Fable (5 h cap) is not scheduled in Days 1–10; first candidate use is Day 10 (gap methodology) or Day 13 (final QA). Log in MODEL_USAGE_LEDGER.md first.
