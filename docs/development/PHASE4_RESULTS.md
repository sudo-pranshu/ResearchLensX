# Phase 4 results: engine hardening

Branch `feature/phase4-engine-hardening`, 2026-11-14. Goal: make the research engine safe and fast enough to sit behind an API, without changing any scientific behaviour. Baseline 133 tests; now **189 passing** (`.venv`) and 177 passing + 1 skipped without PyTorch (the CI configuration). Design source: `docs/architecture/LOCAL_ARCHITECTURE.md` (with the architecture-review amendments, §12).

## What was built
| Step | Result | Main files |
|---|---|---|
| 1 Central settings | `AppConfig` from environment (`RLX_*`), validated; wildcard bind addresses refused (no authentication yet) | `settings.py` |
| 2 Versioned store | SQLite `user_version` migrations v1-v3, WAL, busy timeout; documents scoped by workspace (`PRIMARY KEY (workspace, doc_id)`), status and file columns, `jobs` table; legacy library files upgrade in place (verified on a copy of the real library) | `store/migrations.py`, `store/corpus.py` |
| 3 Original files | `DocumentStorage` protocol with an atomic local adapter; originals stored by document id with SHA-256; upload size, type and page limits with explained failures; `Document.n_pages` (schema 1.4) | `storage/documents.py`, `workspace.py`, `ingestion/pymupdf_ingestor.py` |
| 4 Model manager | one instance per model, loaded once under a lock, byte budget with LRU eviction, all inference on one thread; per-call verifier (the shared engine is never mutated); bounded caches; fallbacks cleared when the requested configuration works again | `models.py`, `qa/engine.py`, `retrieval/registry.py` |
| 5 Per-text embeddings | content-addressed `embeddings` table; adding a paper embeds only its new chunks; garbage collection instead of delete-on-removal; the old corpus-keyed cache is untouched (experiment provenance) | `retrieval/chunk_embeddings.py` |
| 6 Index generations | one library-wide index per library snapshot; a scope is a candidate filter, so scoped and unscoped results come from the same generation | `services/search.py` |
| 7 Jobs | state machine `queued → parsing → indexing → ready / failed / cancelled`; original stored before enqueue; PDFs parsed in a spawned subprocess with time and memory limits; one rebuild per drained queue; cancel, retry, idempotent submission, restart recovery | `jobs.py`, `parse_worker.py` |
| 8 Services | `Workspace` is now a facade over `LibraryService`, `SearchService`, `AnalysisService` | `services/`, `workspace.py` |

## Acceptance tests added (from the architecture review)
1. **Incremental equals rebuild** (`tests/test_incremental_equals_rebuild.py`): for BM25, TF-IDF, LSA, e5 (offline hashing encoder), RRF(BM25+e5) and RRF(BM25+LSA), add A, B, C, delete B, add D: ranked ids and scores (6 d.p.) for 20 queries, unscoped and scoped, equal a fresh build; only the new paper's chunks are embedded; vectors shared with remaining papers survive deletion.
2. **Concurrency** (`tests/test_workspace.py`, `tests/test_jobs.py`): 16 concurrent asks with and without verification never interfere and the NLI model is built once; every model-touching call runs on the single inference thread; searches during an upload never see a half-indexed document and nothing raises `SQLITE_BUSY`; restart re-queues interrupted jobs.
3. **Browser-attack rejection** belongs to the API layer (Phase 5) and is **not done yet**.
Also covered: hostile and corrupt PDFs, blank (scanned) PDFs, page limit, parser time limit and memory limit (parser killed), cancel while parsing, retry, duplicate re-upload without re-parsing, failed rebuild marks documents failed while the worker survives, migration failure leaves the previous version intact, workspace isolation in the store.

## Measurements (Apple M2, 8 GB; the 5 real arXiv PDFs; real models; `scripts/bench_phase4.py`, three runs)
| Measure | Before | After | Notes |
|---|---|---|---|
| Add the last paper to a library, e5 index, model already loaded | full re-embed of the whole library: **10.9-11.0 s** (911 vectors) | **1.5 s** (133 new vectors) | same code path; "before" is what the old corpus-keyed cache cost for any change; time scales with chunks, so the gap grows with library size |
| Upload 5 PDFs end to end | inline parse 0.19-0.96 s per file (≈2.2 s for 5, measured earlier) | **4.3-4.8 s for 5 (≈0.9 s per file)** through the job queue | slower by about 2 s for 5 files: each parse now starts an isolated process (safety, limits, crash isolation); not an improvement |
| Peak resident memory, e5 + reranker + NLI used across 4 scopes in one process | one e5 and reranker copy **per scope** (unbounded) | **1.17 / 1.48 / 1.62 GB** over three runs; each model loaded exactly once | run-to-run variance is large (torch/MPS buffers); the byte budget (2 GB estimate) is enforced on model weights only |
No other improvement is claimed. BM25 and TF-IDF still rebuild from texts on each library change (milliseconds at this size); first-query latency after a restart was not changed by this phase.

## Not done in Phase 4 (carried forward)
- `InferenceProvider` as an awaitable, batch-shaped interface (the local `ModelManager` is the only implementation today) and `presigned_url` is a stub that returns `None`.
- API-level protections (host/origin/token checks, streaming size limit, nosniff/CSP) and the browser-attack acceptance tests: Phase 5.
- Hosted-scale concerns (distributed jobs, quotas): deployment readiness only.
- The Streamlit app still calls `Workspace.upload` (synchronous); progress UI comes with the new frontend.
