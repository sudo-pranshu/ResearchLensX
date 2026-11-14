# Implementation roadmap (local-first)

Status: **proposal for approval.** No phase beyond the audit has started. Principle: preserve the scientific engine (133 tests stay green), build around it, ship in small milestones, measure before and after, never deploy.
Estimates are focused-work days for one developer; the scope is large, so expect **roughly 8-10 weeks** of calendar time. Re-plan after Phase 3.

| Phase | Goal | Main work | Exit criteria | Tests added | Model use | Days |
|---|---|---|---|---|---|---|
| 1 Audit and baseline | known facts | repository audit, baseline measurements, plan (this set of documents) | audit approved by the maintainer | — | Sonnet | done |
| 2 Local architecture | settle boundaries | review `LOCAL_ARCHITECTURE.md` (ports, API, storage, jobs, config) with an independent architect review; decide static-export serving; define OpenAPI contract draft | architecture approved; decisions recorded in `docs/architecture/TECHNICAL_DECISIONS.md` | — | **Opus** (architecture and API review) | 2 |
| 3 Design system | premium UX | design review A, design system B, key screens C (see `UI_UX_SPECIFICATION.md` §9); finalise tokens and component list | approved design system and page designs, contrast report | — | **Fable** (A-C, ≈ 3.0 h ceiling), Sonnet integrates | 4 |
| 4 Engine hardening for an API (**done 2026-11-14**, see PHASE4_RESULTS.md) | ports, storage, index generations | in this order, each step independently testable with the 133 tests green: (1) `researchlens.settings` (`RLX_DB` alias); (2) SQLite `user_version` migration with WAL, busy timeout, foreign keys, new `documents` columns; (3) original PDFs in `DocumentStorage` (atomic write, `doc_id` identity); (4) `ModelManager` (one instance per model, single inference thread, bounded caches, per-call verifier); (5) content-addressed `embeddings` table + per-document term counts; (6) index generations (one library index, scope as filter, atomic swap); (7) jobs table, state machine, spawn-based parse worker, restart recovery; (8) split `Workspace` into services | all 133 tests green; **acceptance tests** below pass; measured: time to add one paper to a 20-paper library before vs after | see acceptance tests | Sonnet | 10 |
| 5 Local API | HTTP contract | `apps/api` FastAPI with the **local-API security middleware from the first commit** (host allow-list, Origin / Sec-Fetch-Site / token on non-GET, streaming size limit, nosniff + CSP), routers per §4 of the architecture, error mapping, upload to jobs, exports, results; OpenAPI → generated TypeScript client | API test suite green (temp data dir; hostile PDFs; MOCK/fallback states); contract frozen | API tests | Sonnet; Opus reviews security and contract | 7 |
| 6 Frontend foundation | app shell | `apps/web` Next.js + TypeScript + Tailwind; tokens, themes, component library, layout, routing, command palette, status pill, API client, error boundary | shell renders in both themes, keyboard navigable, component tests pass | Vitest + axe | Sonnet implements Fable's system | 6 |
| 7 Library and upload | first real workflow | Library page, dropzone, job progress, document detail, deletion, filters, failure explanations | upload real PDFs end to end in the UI; persistence across restart | component + Playwright | Sonnet | 6 |
| 8 Research modules | all capabilities in the UI | Search, Ask, Verify, Compare, Graph (Cytoscape), Gaps, Review (+ exports), Evidence results, Settings | every module works through the UI against the real API; no inert controls | Playwright per module | Sonnet; Fable guidance on key screens | 14 |
| 9 Visual refinement | polish | Fable Task D on built screenshots; spacing, states, responsive and mobile layouts, motion, accessibility fixes | Lighthouse a11y ≥ 95, zero serious axe violations, no console errors, both themes, mobile | visual and a11y checks | **Fable** (D) + Sonnet | 5 |
| 10 One-command startup | `./run.sh` for the whole stack | extend the existing launcher: Node check, `npm ci`, frontend build, API + static serving, health checks, port handling; keep idempotency and Ctrl+C shutdown | clean-clone start, stop, restart works twice; documented Windows/WSL limits | startup tests (scripted) | Sonnet | 3 |
| 11 Performance | measured improvements | profile; lazy loading, device choice, batching, caching; frontend bundle and rendering | before/after table on the same machine and corpus, none invented | benchmark scripts (committed) | Sonnet; Opus for hard problems | 5 |
| 12 Repository and docs | open-source quality | finish restructure (`tests/{unit,integration,e2e}`, `infrastructure/local`), README with the agreed hierarchy and real screenshots, Mermaid diagrams, module docs, fix documentation inconsistencies (audit §6), governance and CI (tests, docs gate, frontend, e2e) | README verified command by command; docs gate green; CI green | docs gate | Sonnet | 6 |
| 13 Local E2E QA and deployment readiness | final acceptance | full real-user workflow (15 steps in the brief) from a clean clone, desktop and mobile, console clean; write `FUTURE_DEPLOYMENT_READINESS.md` against the final code; final Fable visual review within the reserve; Opus review of hardening | acceptance checklist signed off; ledger complete | Playwright full workflow | Sonnet, Opus, Fable (reserve) | 5 |
Total ≈ 72 focused days.

## Acceptance tests that gate Phase 4 and 5 (from the architecture review)
1. **Incremental equals rebuild:** for BM25, TF-IDF, LSA-dense and `rrf:bm25+lsa`, add A, B, C, delete B, add D; top-10 ids and scores for 20 fixed queries, with and without scope, equal a fresh full build over A, C, D (scores to 1e-6); embeddings are computed only for D's chunks; shared cache entries survive deletion.
2. **Concurrency:** 50 mixed concurrent search, ask(verify on/off) and scoped requests during an upload job: each response's verify state matches its request, the encoder loads once, every search sees exactly one index generation, no `SQLITE_BUSY`, and a restart re-queues a `processing` job.
3. **Browser-attack rejection:** multipart POST with a foreign `Origin`, a foreign `Host`, a missing token or `Sec-Fetch-Site`, an oversized stream, or a PDF over the page limit is rejected with 400/403 and nothing is written; the same-origin path succeeds.

## Decisions needed before Phase 2
1. Serve the static frontend from the API (recommended) or run `next start` separately.
2. Component stack: Radix + Tailwind (shadcn-style), Cytoscape.js, Recharts, TanStack Query (§8 of the UI specification).
3. Keep `configs/` and `results/` at the repository root (recommended) or migrate with a path-rewriting tool.
4. Retire the Streamlit app after Phase 8 parity, or keep it as an engineering console.
5. Upload limits and quota defaults (50 MB per file, 400 pages are proposals).

## Quality gates for every phase
`pytest` green (133+), new tests for the phase, docs updated (`PROJECT_STATE.md`, `CHANGELOG.md`, `KNOWN_LIMITATIONS.md`), scientific-honesty checks (MOCK, experimental, calibrated, candidate, collection-scoped) verified in the UI, measured numbers only, no cloud service referenced or configured, commits and PRs authored by Pranshu Kumar only.

## Model and ledger plan
Sonnet builds. Opus: architecture and API review (Phase 2), security review of upload/jobs/API (Phase 5), hard performance or integration failures. Fable: A, B, C (Phase 3, ceiling 3.0 h cumulative with B and C) and D (Phase 9, 0.5 h); ≥ 1.0 h reserve kept for integration problems and the final review. Used so far ≈ 0.05 h; every Fable task is logged before and after in `MODEL_USAGE_LEDGER.md`.

## Risks
Scope (mitigated by phase gates and re-planning); Node/toolchain differences on user machines (mitigated by prerequisite checks and clear messages); CPU latency of NLI and embeddings on 8 GB laptops (mitigated by lazy loading, jobs and progress); loss of scientific honesty in a polished UI (mitigated by typed honesty fields and tests); design quality dependent on the Fable allowance (mitigated by a precise ledger and a reserve).
