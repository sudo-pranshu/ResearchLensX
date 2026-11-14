# Branch strategy and release plan

## Current state (verified 2026-11-14)
| Branch | Tip | Role today |
|---|---|---|
| `main` | `96e0f04` | **original scaffold**, 42 commits behind `develop`; shows an outdated repository to anyone who opens it |
| `develop` | `a731759` | **GitHub default branch**; contains all implementation (PRs #23 and #24 merged by the maintainer on 2026-11-14) and a README that still needed rewriting |
| `chore/open-source-release` | local | restructure, governance, launcher; not pushed |
| `feature/*` | merged | historical |

## Strategy
- `main`: stable, tagged releases only; updated by a reviewed pull request from `develop` (never a blind merge; no force-push, no history rewrite).
- `develop`: integration; all work arrives by pull request from `feature/<topic>`, `fix/<topic>`, `docs/<topic>`, `chore/<topic>`. CI (tests, docs gate) must pass.
- Short-lived feature branches; conventional commits; every PR uses the template.
- **Default branch:** keep `develop` until the first release exists, then switch the default to `main` so the repository front page shows the released, polished state. Switching the default is a repository setting; do it deliberately after `main` is brought up to date.

## Sequence
1. Land the repository hygiene work (this branch) by pull request into `develop` after review.
2. Build the local product in phases (`IMPLEMENTATION_ROADMAP.md`), each phase one or more PRs into `develop`.
3. When the local product passes the acceptance workflow: open a release PR `develop → main`, tag **`v0.2.0`** (first release with a UI redesign and one-command launch; `0.1.0` is the original scaffold version in `pyproject.toml`), publish release notes from `CHANGELOG.md`.
4. Switch the default branch to `main`.
5. Branch protection (require PR and CI) when the plan allows it.

## Release checklist
Tests green (Python, frontend, e2e); docs gate green; `CHANGELOG.md` updated; README commands re-verified from a clean clone; screenshots current; `LICENSE`, `CITATION.cff` version and date updated; results register consistent with `results/`; no secrets or data tracked.
