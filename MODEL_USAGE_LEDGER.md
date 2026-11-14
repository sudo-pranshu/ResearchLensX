# Model Usage Ledger

Fable 5.2 hard cap: **5.0 h total**. Update this file before and after every Fable task.
Time is wall-clock as reported by the app; if unknown, record an estimate and mark it "est.".

## Fable 5.2 budget

| # | Date | Objective | Context supplied | Expected output | Acceptance criterion | Est. h | Actual h | Cumulative |
|---|------|-----------|------------------|-----------------|----------------------|--------|----------|-----------|
| — | — | (none used yet) | — | — | — | — | — | 0.0 / 5.0 |

Planned (not started, each needs a concrete trigger):
1. Adversarial review of the verified-RAG and gap-discovery claims before the technical report (~1.0 h). Accept: written list of invalid/overclaimed statements, each mapped to a doc line.
2. Final system QA against Definition of Done (~1.5 h). Accept: pass/fail per DoD item with reproduction steps.
3. Reserve (~2.5 h) for an integration failure Sonnet/Opus could not resolve.

## Opus 5.5 (selective)

| Date | Task | Output |
|------|------|--------|
| 2026-11-14 | Methodology review of gap discovery, graph NLI linking and review wording (read-only, ~1.5 min, ~89k tokens) | 5 validity threats + evaluation protocol; all five code fixes applied (wording, real section labels, related-work/citation exclusion, complete-linkage clusters, mention-level missing-comparison, entity-gated NLI edges). Held-out evaluation protocol not yet run. |

Planned: gap-discovery methodology review; NLI calibration/explanation design.

## Sonnet 5.5

Primary implementation. 2026-11-14: repository audit (`docs/REPOSITORY_AUDIT.md`).
