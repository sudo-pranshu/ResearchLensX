# Team Responsibilities

Summary: [PROJECT_BIBLE.md §D](PROJECT_BIBLE.md#d-team-summary). Day-by-day instructions: [docs/development/](docs/development/).

## Ownership matrix (files)

Only the owner edits a path. Anyone else proposes a change through a PR that the owner approves.

| Path | Owner | Notes |
|---|---|---|
| `src/researchlens/contracts.py`, `pipeline.py`, `mocks.py` | Pranshu | `interface-change` label required |
| `src/researchlens/retrieval/**`, `evaluation/metrics.py`, `evaluation/stats.py`, `data/qasper.py` | Pranshu | |
| `experiments/h1_*`, `configs/h1_*`, `notebooks/*` | Pranshu | |
| `src/researchlens/verification/**`, `generation/**`, `data/scifact.py`, `evaluation/attribution.py` | Pawan | |
| `experiments/h2_*`, `configs/h2_*`, `results/h2/**` | Pawan | |
| `src/researchlens/ingestion/**`, `processing/**`, `knowledge_graph/**` | Gaurav | `processing/chunking.py` was written by Pranshu on Day 1 and transferred to Gaurav |
| `app/**`, `tests/test_app*`, `tests/test_ingestion*`, `tests/test_kg*` | Gaurav | |
| `PROJECT_BIBLE.md`, `PROJECT_STATE.md`, `CHANGELOG.md`, CI, `pyproject.toml`, `requirements*.txt` | Pranshu | others append via PR |
| `tests/test_<module>*.py` | the module's owner | |

## Pranshu Kumar: lead, research architect, retrieval (H1)

**Deliverables:** classical + dense + hybrid `PassageRetriever`s; FAISS indexes; H1 dev/test experiments with cluster stats; the `Pipeline` integration; running GPU jobs for the team; final integration and the experiment register.

- **Inputs:** QASPER/SciFact raw data; `Chunk[]` from Gaurav (or `load_qasper_documents`).
- **Outputs:** `list[RetrievedPassage]`, `results/h1_*`, `artifacts/indexes/`.
- **Acceptance:**
  - Every retriever passes `isinstance(x, PassageRetriever)`.
  - Unit tests for BM25 math, RRF math and the candidate filter.
  - H1 dev and test runs committed with their configs.
  - Test run exactly once per final system.
- **Depends on:** Gaurav for `chunk_document` (already available); Pawan for nothing.

## Pawan: NLP modelling & claim verification (H2)

**Deliverables:** SciFact loader; claim splitter; TF-IDF + LR baseline verifier; NLI `Verifier` (zero-shot, optionally fine-tuned); `LLMRAGGenerator` and verified-RAG revision; citation P/R evaluation (`evaluation/attribution.py`); H2 experiments and ablations; confidence calibration (temperature scaling on SciFact held-out train, ECE reported).

- **Inputs:** `RetrievedPassage[]` (use `mocks.py` / `build_default_pipeline` until hybrid is ready); SciFact.
- **Outputs:** `VerificationResult`, `Answer`, `results/h2/*`.
- **Acceptance:**
  - `Verifier` and `Generator` protocol checks pass.
  - Verifier macro-F1 on SciFact dev reported, alongside the LR baseline.
  - The generation script runs on CPU with `--limit 3` and on GPU unchanged.
  - Generations are cached as JSONL.
  - Citation P/R code has unit tests with hand-labelled toy cases.
- **Depends on:** Pranshu for GPU runs and the hybrid retriever (by Day 5); nothing from Gaurav.

## Gaurav: document intelligence & application

**Deliverables:** `PyMuPDFIngestor` (+ optional GROBID client); metadata (title, authors, year, references); chunker ownership; rule/gazetteer entity extraction (dataset, method, metric, task); `GraphBuilder`; Streamlit app (upload, ask, citations, verification badges, compare view, KG view); optional FastAPI; app tests; demo workflow doc.

- **Inputs:** PDFs; `Answer`/`VerificationResult` from `Pipeline` (mock until Day 5).
- **Outputs:** `Document`, `Chunk[]`, `KnowledgeGraph`, the app.
- **Acceptance:**
  - Ingestor recovers the title and ≥80% of top-level section headings on 10 test PDFs (hand-checked table committed).
  - Builds a graph from `data/sample/documents.json`.
  - The app runs with `streamlit run app/streamlit_app.py` on a laptop without a GPU or internet once models are cached.
  - MOCK badge shown whenever `is_mock`.
- **Depends on:** nothing paid. All interfaces and fixtures are in the repo, and no Claude or GPU is needed.

## Integration responsibilities
- Pranshu merges `develop` → `main` at the end of Days 2, 6, 10, 12, 14 after CI passes and a smoke demo.
- Interface disagreements → open an issue labelled `decision`, record the outcome in PROJECT_BIBLE §I.

## Weekly/daily progress format (post in the team chat and append to PROJECT_STATE.md via PR at each milestone)
```
[Name] Day N — YYYY-MM-DD
Done:      <issue #s + one line each, link PRs>
Results:   <metric = value, run id> (or "none")
Next:      <issue #s>
Blocked:   <what, waiting on whom> (or "none")
Interface: <any contract change requested> (or "none")
```
