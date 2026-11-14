# Future deployment readiness

Status: **planning document. Online deployment is on hold; no cloud service, provider account, database, storage bucket, authentication service or public URL is configured or implemented.** Purpose: record exactly what will have to change when ResearchLens X is eventually deployed online, and which local-first design choices keep that change small. Companion notes with provider research (verified 2026-11-14, will go stale) are in `DEPLOYMENT_ARCHITECTURE.md`; the control list for a multi-user service is in `SECURITY_REVIEW.md`.

## 1. Rules that keep the local product deployable later
1. The research engine (`src/researchlens`) imports no web framework, no filesystem layout and no provider SDK; it reaches the outside world only through ports.
2. Every path, host, port, backend choice and limit comes from `researchlens.settings` (environment based). No literal path or provider name elsewhere.
3. The frontend talks only to the HTTP API through a generated client; it never reads files, the database or models.
4. All API operations are scoped by a **workspace id** (`local` today). Storage keys, database rows, caches and job records carry it from day one.
5. Anything slow is a job (queued → processing → ready / failed) with persisted state, so a distributed queue can replace the local runner.
6. Uploads and PDF parsing are treated as untrusted-input handling today (limits, subprocess, timeouts), not only after deployment.
7. The API never assumes it is the only client on the machine: bind to loopback by default, validate `Host` and `Origin`, require a per-launch token for state-changing calls (local-API hardening; see the architecture review in the ledger).
8. The static frontend contains no server-side rendering dependency, so it can be hosted on any static host.

## 2. Interfaces and what replaces the local adapter
| Port | Local adapter (now) | Plausible hosted adapter (later, not implemented) | What the engine must already guarantee |
|---|---|---|---|
| `DocumentStorage` (put / open / delete / exists / list by workspace) | filesystem under `RLX_DATA_DIR` | S3-compatible object storage with signed URLs | keys include workspace id; atomic writes; no direct path use outside the adapter |
| `MetadataStore` (documents, jobs, settings, index state) | SQLite (WAL) | hosted PostgreSQL | SQL kept portable (no SQLite-only features in business code); migrations by version; transactions per operation |
| `InferenceProvider` (embed, rerank, NLI) | local models via `ModelManager` (CPU / MPS) | hosted inference endpoint or GPU worker | batch interface, explicit timeouts, typed `ModelUnavailable`; results identical in shape; calibration constants tied to a model id and revision |
| `JobRunner` (submit / status / cancel / retry) | thread and subprocess workers | distributed task queue with workers | idempotent job functions; job payloads are plain data; progress stored in the metadata store |
| `Settings` | `.env` and defaults | environment/secret manager | no secrets in code; validation at startup |
| Auth | none (local single user) | token verification middleware plus per-user workspace id | endpoints already resolve a workspace id from a single dependency |

## 3. What will have to change (and what will not)
| Area | Change at deployment time | Effort driver |
|---|---|---|
| Authentication and authorisation | add identity (an identity service), derive workspace id from the token, enforce it on every query | the single "resolve workspace" dependency in the API; isolation tests already specified |
| Multi-user isolation | per-user storage prefixes and database scoping (already present as the workspace id); caches per workspace | verify with the cross-user test suite |
| Storage | implement the object-storage adapter; presigned upload/download; lifecycle and backups | adapter only |
| Database | switch URL to a hosted relational database; run migrations | adapter and migration check |
| Models | choose where inference runs (same host CPU, GPU worker or service); warm-up and health checks; batching under concurrency | inference adapter plus capacity measurements |
| Jobs | replace the local runner with a queue and workers; retries and dead-letter handling | runner adapter |
| API hardening | rate limits, quotas, request-size limits at the edge, CORS allow-list, structured logs without content, TLS | configuration plus gateway |
| Frontend hosting | serve the static export from a CDN/static host; set the API base URL | configuration |
| Observability | metrics, tracing, alerts | new, small |
| Privacy and compliance | privacy policy, data deletion and retention, backups lifecycle, terms | documents plus processes |
| Cost and licence checks | dataset terms (SciFact is non-commercial), PyMuPDF AGPL obligations for a network service, free-tier limits re-verified | legal and financial review before launch |
What does **not** change: the research engine, evaluation code, the typed API contract, the frontend, and the scientific-honesty fields.

## 4. Blockers recorded for a multi-user service (from the earlier readiness audit)
B1 no API layer (planned Phase 5) · B2 no authentication or isolation (workspace id now, auth later) · B3 synchronous uploads (job model in Phase 4) · B4 untrusted PDFs parsed in-process (subprocess with limits in Phase 4/5) · B5 whole-engine cache invalidation (per-workspace incremental indexes) · B6 lazy model loading and warm-up (`ModelManager`) · B7 shared embedding cache (per-chunk hash, per workspace) · B8 no rate limits or quotas (later) · B9 evaluation page reads local files (served by `/results`) · B10 synthetic fixtures in the UI (demo data clearly labelled) · B11 no container, TLS, backups (later, after local acceptance) · B12 single-user Streamlit UI (replaced) · B13 CPU latency for first embedding (jobs and progress) · B14 Arm/ platform wheels (**unverified**, test on the target host).

## 5. Pre-deployment checklist (not started)
Decide the hosting strategy from current provider terms (re-read the official documents; the 2026-11-14 notes show a free hosted Python backend with persistent storage was **not** available at zero cost among the providers checked) · implement the adapters in §2 behind existing ports · add authentication and the cross-user isolation test gate · load-test CPU latency and memory · write and review the privacy policy · confirm licences (AGPL, dataset and model terms) · set up backups and a restore drill · security review of upload, API and jobs · staged private beta.

## 6. Explicitly out of scope now
Vercel, Hugging Face Spaces, Supabase, any cloud database, object store, authentication service, cloud GPU, public URL, paid API or infrastructure. No code or configuration for them is added to the repository.
