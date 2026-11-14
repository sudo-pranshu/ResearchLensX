# GitHub repository audit (read-only, 2026-11-14)

Scope: presentation and state of `github.com/sudo-pranshu/ResearchLensX` compared with the local clone. No merge, rewrite or settings change was performed to produce this audit.

## Findings
| Question | Answer (verified) |
|---|---|
| Default branch | `develop` |
| Branch with the latest verified implementation | `develop` (merge commit `a731759` of PR #24); its tree equals `feature/pranshu-h1-final-test` at `6002f71` |
| PR #24 | **merged** 2026-11-14 07:53 UTC by the maintainer (not open as previously assumed) |
| PR #23 | **merged** 2026-11-14 07:53 UTC; it was redundant (its head is an ancestor of #24's) |
| `main` | original scaffold at `96e0f04`, 42 commits behind `develop`, still shows old team and scaffold documents |
| Commits missing from `develop` | none of the feature work; only the cleanup done on `chore/open-source-release` |
| Documentation fixes already on another branch | the teammate-name removal and README fixes are on `develop`; the course-project wording in the README was still present |
| Repository description | "…(B.Tech final-year project)" (outdated, a repository setting) |
| Licence | none declared on GitHub (`licenseInfo: null`) |
| Topics | none |
| CI | `.github/workflows/ci.yml` existed (tests only) |
| Issue templates | none (a PR template existed) |

## Corrective actions (this branch)
Wording scrub, restructure, governance files (`LICENSE`: AGPL-3.0-or-later after the licence review in `docs/user-guide/LICENSES.md`, `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `.env.example`, issue templates), CI extended with a docs gate, new README (Phase 12). **Not done without your approval:** change the repository description and topics, set the default branch, publish a release, bring `main` up to date.
Historical wording remains only in this file by design; `scripts/check_docs.py` allow-lists it.
