# Model Usage Ledger

Fable 5.2 hard cap: **5.0 h total**. Update this file before and after every Fable task.
Time is wall-clock as reported by the app; if unknown, record an estimate and mark it "est.".

## Fable 5.2 budget

| # | Date | Objective | Context supplied | Expected output | Acceptance criterion | Est. h | Actual h | Cumulative |
|---|------|-----------|------------------|-----------------|----------------------|--------|----------|-----------|
| 1 | 2026-11-14 | Final QA against the Definition of Done and overclaim audit of docs vs code vs results | repo path; DoD list; instruction to run tests and inspect, read-only | pass/fail per DoD item with reproduction steps + list of overclaimed statements with file/line | every DoD item has a verdict with evidence; every number in README/report checked against results/ | 1.0 | ≈0.05 (155 s wall-clock, ~115k tokens, est.) | ≈0.05 / 5.0 |

Planned for the local-first product (see `docs/architecture/UI_UX_SPECIFICATION.md` §9 and the roadmap). **Ceiling for planned design tasks A-D is 3.5 h; ≥ 1.0 h stays in reserve; remaining allowance on 2026-11-14 ≈ 4.95 h.** Original plan items:
1. Adversarial review of the verified-RAG and gap-discovery claims before the technical report (~1.0 h). Accept: written list of invalid/overclaimed statements, each mapped to a doc line.
2. Final system QA against Definition of Done (~1.5 h). Accept: pass/fail per DoD item with reproduction steps.
3. Reserve (~2.5 h) for an integration failure Sonnet/Opus could not resolve.

## Opus 5.5 (selective)

| Date | Task | Output |
|------|------|--------|
| 2026-11-14 | Methodology review of gap discovery, graph NLI linking and review wording (read-only, ~1.5 min, ~89k tokens) | 5 validity threats + evaluation protocol; all five code fixes applied (wording, real section labels, related-work/citation exclusion, complete-linkage clusters, mention-level missing-comparison, entity-gated NLI edges). Held-out evaluation protocol not yet run. |

| 2026-11-14 | Architecture review of `LOCAL_ARCHITECTURE.md` (read-only): ports, caching/index consistency, jobs, local-API security, memory policy, static export, deployment readiness | ≈ 2 min, ~111k tokens; 6 findings, all accepted and folded into `LOCAL_ARCHITECTURE.md` §5, §6, §11, §12 and the roadmap |

Planned: security review of upload/jobs/API (Phase 5); hard performance or integration failures as needed.

## Sonnet 5.5

Primary implementation. 2026-11-14: repository audit (`docs/REPOSITORY_AUDIT.md`).
