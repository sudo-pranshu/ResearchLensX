# Technical decisions

Early decisions (ADR-001 to ADR-011) are in [PROJECT_BIBLE.md §I](PROJECT_BIBLE.md). Decisions made while building the product layer:

| ID | Decision | Why | Alternative rejected |
|---|---|---|---|
| TD-01 | Schema versions are additive (1.1 page provenance, 1.2 verifier support/explanation, 1.3 graph types); old fields never change | Keeps committed results and fixtures valid | Breaking rename |
| TD-02 | `CorpusStore` = SQLite storing parsed `Document` JSON; chunks are derived, not stored | Chunking can change without a migration; trivial to back up | Store chunks; a vector DB |
| TD-03 | Reranker is a `Retriever` wrapper (`rerank:<spec>`), shared scorer cache | Works with within-paper candidate subsets and every first stage; same experiment runner | Separate rerank stage in the app only |
| TD-04 | Answers are extractive and cited; the NLI verifier labels each sentence | No hallucinated citations or page numbers possible; fully offline | LLM generation as the default (kept for experiments only) |
| TD-05 | Abstention threshold 0.2 on lexical coverage | EXP-QA-001: the signal is weak; higher thresholds wrongly abstain on answerable questions | 0.4 (19% abstention at precision .135) |
| TD-06 | Temperature scaling changes reported confidence only, labels use raw probabilities | Verifier thresholds were tuned on raw probabilities | Calibrate then re-tune thresholds |
| TD-07 | Graph edges require provenance; mock verifiers cannot create supports/contradicts; NLI edges need relevance gates and are labelled candidates | Raw NLI edges were 0/9 valid | Trust NLI output |
| TD-08 | Gap confidence capped at "medium"; no novelty claims; templates labelled | No expert review and no external literature search | Score "high" confidence from evidence count |
| TD-09 | Streamlit app talks only to `Workspace` (service layer, no UI imports) | Testable offline; visible fallbacks (MOCK badge) in one place | Logic inside page code |
| TD-10 | Review export: Markdown + HTML + PDF (PyMuPDF Story) | No extra dependency; PyMuPDF is already required | Pandoc / LaTeX |
| TD-11 | Separate `.venv` (system-site-packages) for PyMuPDF/Streamlit/pyvis | Leaves the base Python untouched | Install into base |
