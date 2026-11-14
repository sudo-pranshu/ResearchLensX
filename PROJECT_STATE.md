# Project State

Updated: 2026-11-14 by Pranshu. Update this file at every milestone (PROJECT_BIBLE §J).

## Completed
- Research design v0.1 → critical review → v1.0 ([review](docs/RESEARCH_DESIGN_REVIEW.md)).
- Contracts (`contracts.py`), `Pipeline` facade, mocks, synthetic fixtures.
- Retrieval: BM25 (from scratch), TF-IDF, LSA, RRF, linear fusion, `ChunkIndex` adapter. Dense encoders (SPECTER2, e5, SciBERT) with embedding cache + FAISS, executed on QASPER and SciFact.
- Evaluation: ranking metrics; bootstrap / randomisation with paper-level clustering; Holm; McNemar.
- QASPER v0.3 downloaded and audited; loader + `Document` adapter.
- Section-aware chunker.
- CPU end-to-end demo: `scripts/demo.py` (BM25 → extractive answer → mock verification) runs on real QASPER papers.
- H1 classical dev baselines (EXP-H1-001/002).
- H1 dense dev (EXP-H1-003): D* = e5; RRF(BM25+e5) is not better than e5 alone. SciFact retrieval (EXP-H1-004) shows the same pattern.
- Real NLI verifier on SciFact dev (EXP-H2-001): passage-level macro-F1 .681 vs LR .518. The judge model is weaker (.565).
- **H1 final test (EXP-H1-006): not supported.** RRF(BM25+e5) R@5 .486 vs e5 .501 (Δ −.015, n.s.); it beats BM25/TF-IDF/LSA. Reported as a negative result.
- H2 dev pilot (EXP-H2-002): inconclusive, because the small generator rarely cites; Answer-F1 guard failed.

## Active
Day 1 done (11-14): schema 1.1 page provenance; `PyMuPDFIngestor` + tests (52 pass in `.venv`; ingestion tests skip without PyMuPDF). Validated on 5 real arXiv PDFs (BERT, RAG, BEIR, SciFact, QASPER papers; kept in gitignored `data/raw/pdfs/`): titles, abstracts, sections and page-cited BM25 retrieval work. Known issues: reference counts approximate (BERT over-counts appendix text), appendix heading "A X" strips the letter, authors empty when PDF metadata lacks them, two-column and table text is noisy.

See [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) (14-day solo plan, 2026-11-14 to 2026-11-27) and [docs/REPOSITORY_AUDIT.md](docs/REPOSITORY_AUDIT.md).

## Blocked
- Branch protection on a private repo needs GitHub Pro/Team; the no-direct-commits-to-main rule is by convention.
- PDF ingestion, graph, app, reranker, gap and report modules are not implemented yet.

## Freeze
2026-11-27 (moved from 2026-11-18).