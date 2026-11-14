# Security review (design-stage)

Scope: the planned public, multi-user ResearchLens X (frontend + FastAPI + Supabase Auth + VM storage). Status: **design review, no code changed**. To be reviewed again by an independent model/human after the API exists. Nothing here is a claim that the system is secure; it is the control list the build must meet and test.

## 1. Assets and threats
| Asset | Threat |
|---|---|
| Users' uploaded PDFs and derived data (possibly unpublished research) | Cross-user access (IDOR), leakage through caches/logs/backups, accidental public exposure |
| Accounts | Credential stuffing, weak sessions, token theft |
| Server resources (CPU, RAM, disk) | Abuse (huge/zip-bomb PDFs, many uploads, expensive queries), resource exhaustion on a free VM |
| Server integrity | Malicious PDFs exploiting the parser; dependency vulnerabilities; SSRF; injection |
| Scientific integrity | Users or attackers making outputs look validated (spoofed badges), mock output shown as real |

## 2. Authentication and authorisation
- **Identity:** Supabase Auth (email magic link and/or OAuth). The API never stores passwords. Verify the Supabase JWT (signature, `exp`, `aud`, `iss`) on every request; user id = `sub`. Unverifiable or expired token ⇒ 401.
- **Authorisation:** every data operation takes the user id from the verified token, never from the request body or URL. Storage paths are derived: `/data/users/<sha256(sub)[:32]>/…`; a document id is meaningful only inside its user's own SQLite database, so even a guessed id cannot cross users.
- **Isolation tests (mandatory):** two users A and B; B requests A's document ids on every endpoint (list, get, search, ask, verify, graph, gaps, review, delete, export, job status); all must return 404 (not 403, to avoid existence oracles). Direct file path traversal attempts blocked. These tests gate every release.
- **Sessions:** short-lived access tokens, refresh handled by Supabase client; logout revokes locally; account deletion deletes the user directory and the Supabase user.
- **Row-level security** on any Supabase table we use (profiles, quotas): `auth.uid() = user_id`.

## 3. File upload and PDF handling (highest risk area)
| Control | Detail |
|---|---|
| Validation | Extension and magic bytes (`%PDF-`), max size (e.g. 25 MB), max pages (e.g. 150), max files per request, per-user document and storage quotas |
| Sandboxed parsing | Run ingestion in a **subprocess** per document with CPU-time, wall-clock (e.g. 60 s) and memory limits (`resource.setrlimit`), no network, working dir inside the user's temp area; kill on timeout; report "failed: reason" |
| Encrypted/scanned/corrupt PDFs | Rejected with an explicit reason; no OCR is claimed |
| Storage | Store the original only if needed for re-processing; random server-side file names; never serve uploaded files back inline from a path derived from user input; downloads via authenticated endpoint with `Content-Disposition: attachment` |
| Deletion | "Delete document" removes the PDF, parsed JSON, chunks, embeddings and graph/review caches; "delete account" removes everything; verify by test that the directory is gone |
| PyMuPDF/dependency risk | Pin versions, run `pip-audit` in CI, keep the image minimal; parser runs unprivileged (non-root, read-only root filesystem where possible) |

## 4. API security
- HTTPS only (HSTS); CORS allow-list of the exact frontend origin (none needed in the single-origin deployment); no wildcard.
- Input models via Pydantic with length limits (question ≤ 1,000 chars, claim ≤ 1,000, k ≤ 20); reject unknown fields.
- **Rate limiting** per user and per IP (e.g. 60 requests/min general, 5 uploads/min, 10 expensive analyses/min) and a global cap; a job queue with per-user concurrency 1; request timeouts; 429 with `Retry-After`.
- No endpoint fetches URLs supplied by users (no SSRF surface). No dynamic code evaluation; no shell construction from user input.
- Output encoding: the existing HTML export escapes untrusted PDF text (tested); the frontend renders PDF-derived text as text, never as HTML; links from PDF text are not auto-followed; Markdown rendered with a sanitising renderer.
- Error responses are generic; detailed errors go to server logs without document content.
- Security headers on the frontend: CSP (no inline script where avoidable), `X-Content-Type-Options`, `Referrer-Policy`, frame-ancestors none.

## 5. Privacy and data handling
- Documents and derived data live only on the application server in the user's directory; they are **not** sent to any third-party service, and no model API is called (models run locally). Telemetry: none that includes content; aggregate counters only, if any.
- Logs: request metadata only (user hash, route, status, duration); never document text or claims.
- Retention: data kept until the user deletes it or the account; a stated inactivity policy (e.g. 180 days) if storage requires it; backups rotate and are deleted with the account on the next cycle (state this honestly in the privacy page).
- A privacy page must state exactly: what is stored, where, who can access it (the operator can technically read server disk), deletion behaviour, no sale or sharing, no tracking cookies beyond auth. No claims of encryption at rest unless enabled (the VM disk encryption state is **unverified**).

## 6. Infrastructure
- VM: non-root service user, firewall allowing only 443 (and 22 restricted to the owner's IP or key-only SSH), automatic security updates, fail2ban or equivalent, container with read-only filesystem and a writable `/data` volume, secrets in environment files outside the repo (`.env` ignored), no secrets in images or git history.
- Backups: nightly encrypted archive; restore tested once before launch.
- Monitoring: health endpoint, disk/RAM alerts, failed-login and 429 counters.
- Supply chain: lockfiles, `pip-audit` and `npm audit` in CI, Dependabot-style updates reviewed manually.

## 7. Scientific-integrity controls (product security of the claims)
- Mock components cannot be confused with real ones: the response carries `is_mock`, the frontend shows the red MOCK banner and refuses to style it as verified; test that a mock result never renders "supported".
- Badges and numbers come from the API's typed responses and committed result files; the frontend contains no hard-coded metrics.
- Calibrated-confidence display only when the response says the validated configuration was used (`calibrated=true`); experimental modes are tagged `experimental`.
- Gaps are always rendered with the collection-scope caveat; no code path labels a gap "novel".

## 8. Risk register (initial)
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Malicious/huge PDF exhausts the VM | Medium | High | Limits, subprocess, quotas, rate limiting |
| IDOR between users | Low if tests pass | Critical | Derived paths, 404 policy, isolation test gate |
| VM reclaimed or lost | Medium | High | Backups, restore drill, health alerts |
| Provider terms or limits change | Medium | Medium | Re-verify before launch, local-first fallback documented |
| Operator can read documents | Certain | Medium | State plainly in the privacy page |
| Vulnerable dependency | Medium | Medium | Pinning, audit in CI |
| Abuse by anonymous users | Medium | Medium | Auth required for everything that uses resources |

## 9. Open review items
An independent (Opus-class) review of sections 2-4 is planned once the API design is final; the findings and fixes will be recorded in `MODEL_USAGE_LEDGER.md`.
