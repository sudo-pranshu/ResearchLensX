# Project State

Updated: 2026-11-13 by Pranshu. Update this file at every milestone (PROJECT_BIBLE §J).

## Completed
- Research design v0.1 → critical review → v1.0 ([review](docs/RESEARCH_DESIGN_REVIEW.md)).
- Contracts (`contracts.py`), `Pipeline` facade, mocks, synthetic fixtures.
- Retrieval: BM25 (from scratch), TF-IDF, LSA, RRF, linear fusion, `ChunkIndex` adapter. Dense encoders (SPECTER2, e5, SciBERT) with embedding cache + FAISS, executed on QASPER and SciFact.
- Evaluation: ranking metrics; bootstrap / randomisation with paper-level clustering; Holm; McNemar.
- QASPER v0.3 downloaded and audited; loader + `Document` adapter.
- Section-aware chunker (ownership → Gaurav).
- CPU end-to-end demo: `scripts/demo.py` (BM25 → extractive answer → mock verification) runs on real QASPER papers.
- H1 classical dev baselines (EXP-H1-001/002).
- H1 dense dev (EXP-H1-003): D* = e5; RRF(BM25+e5) is not better than e5 alone. SciFact retrieval (EXP-H1-004) shows the same pattern.
- Real NLI verifier on SciFact dev (EXP-H2-001): passage-level macro-F1 .681 vs LR .518. The judge model is weaker (.565).
- **H1 final test (EXP-H1-006): not supported.** RRF(BM25+e5) R@5 .486 vs e5 .501 (Δ −.015, n.s.); it beats BM25/TF-IDF/LSA. Reported as a negative result.
- H2 dev pilot (EXP-H2-002): inconclusive, because the small generator rarely cites; Answer-F1 guard failed.

## Active (Days 1–2): see GitHub issues, milestone "Day 2: setup & baselines"
| Owner | Task |
|---|---|
| Pranshu | Dense encoders on Colab (QASPER dev); SciFact retrieval eval; select D* |
| Pawan | SciFact loader + audit; LR baseline; zero-shot NLI verifier on SciFact dev |
| Gaurav | Environment setup; PyMuPDF ingestor on 10 PDFs; Streamlit skeleton on `Pipeline` + fixtures |

## Blocked
- GitHub usernames for Pawan and Gaurav are needed to add them as collaborators and assign their issues.
- Branch protection on a *private* repo requires GitHub Pro/Team. Until then the "no direct commits to main" rule is enforced by convention.

## Next milestones
Day 2 (11-06) setup & baselines · Day 4 (11-08) modules · Day 6 (11-10) integration · Day 8 (11-12) evaluation · Day 10 (11-14) features · Day 12 (11-16) testing & docs · Day 14 (11-18) freeze.
