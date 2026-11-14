# Project state

Updated 2026-11-14. Maintainer: Pranshu Kumar. The experiment register is in [PROJECT_BIBLE.md](../research/PROJECT_BIBLE.md) (section G); open limitations are in [KNOWN_LIMITATIONS.md](../user-guide/KNOWN_LIMITATIONS.md).

## Implemented (tested: 189 tests with the app extras; 177 + 1 skipped without PyTorch)
- **Ingestion:** PyMuPDF text PDFs with sections, per-paragraph pages, references, arXiv id; scanned PDFs rejected (no OCR).
- **Library:** SQLite store, duplicate and failure reporting, metadata filters, thread-safe.
- **Retrieval:** BM25, TF-IDF, LSA, e5, SPECTER2, SciBERT, RRF, linear fusion, cross-encoder rerank, redundancy suppression, section filters. The app offers BM25, TF-IDF, e5 and RRF(BM25+e5) with a visible fallback to BM25.
- **Question answering:** extractive, multi-document, page-cited references, visible abstention.
- **Verification:** DeBERTa-v3 NLI with explanations; the app default is the validated passage-level configuration with calibrated confidence. Combined multi-passage evidence is an experimental, uncalibrated option.
- **Analysis:** provenance-grounded knowledge graph, comparison table, candidate research gaps, cited literature review (Markdown, HTML, PDF).
- **Application:** Streamlit app with a real/fallback/MOCK status panel (to be replaced by the new frontend once it reaches parity).
- **Launcher:** `./run.sh`, `scripts/setup.sh`, `scripts/healthcheck.sh`, `scripts/stop.sh`, `Makefile`.

## Experimentally evaluated (details in the register)
- H1 (hybrid beats every single retriever): **not supported** on the QASPER test split (EXP-H1-006).
- Cross-encoder reranking: improves dev Recall@5 (.502 to .580, EXP-H1-007); no test-split run.
- NLI verifier on SciFact dev: macro-F1 .518 (logistic regression) to .681 (DeBERTa passage-level); calibration ECE .183 to .067 out-of-fold.
- H2 (verified RAG): partially supported on a 100-question dev pilot (citation precision .420 to .785), but the Answer-F1 guard fails (EXP-H2-004). Not confirmed.

## Experimental or unevaluated
Combined-evidence verification; cross-paper NLI relations (candidates only); research-gap precision (rules tuned on 5 papers, no independent labels yet).

## Direction (2026-11-14)
Build the complete product **locally first**: Next.js frontend, local FastAPI API over the existing engine, local SQLite and filesystem storage, one-command startup. Online deployment is on hold and nothing cloud-related is built; boundaries are kept deployment-ready. Plan: [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) · architecture: [LOCAL_ARCHITECTURE.md](../architecture/LOCAL_ARCHITECTURE.md) · UX: [UI_UX_SPECIFICATION.md](../architecture/UI_UX_SPECIFICATION.md) · audit: [REPOSITORY_AUDIT.md](REPOSITORY_AUDIT.md) · later deployment: [FUTURE_DEPLOYMENT_READINESS.md](../deployment-readiness/FUTURE_DEPLOYMENT_READINESS.md).

## Phase 4 (engine hardening) complete
See [PHASE4_RESULTS.md](PHASE4_RESULTS.md): settings, versioned store, original-file storage, model manager, per-text embeddings, index generations, job runner, services split. 189 tests. Next: Phase 5 (local API with browser-attack protection).

## Earlier in-progress work
Branch `chore/open-source-release` (not pushed): repository restructure, governance files, licence notes, `./run.sh` one-command launcher for the current Streamlit app (tested: first start, duplicate prevention, occupied port, stop, restart, Ctrl+C). Awaiting approval of the plan before the API and frontend migration.

## Not run
H2 with a 7-8B generator and on the test split; reranking on the test split; H1 ablations; independent gap labelling.
