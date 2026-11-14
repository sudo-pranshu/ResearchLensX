# Security policy

## Supported versions
Only the latest commit on `main` (once releases exist) and `develop` receive fixes. There are no tagged releases yet.

## Reporting a vulnerability
Please **do not** open a public issue. Report privately through GitHub: [Report a vulnerability](https://github.com/sudo-pranshu/ResearchLensX/security/advisories/new). Include what you found, how to reproduce it, and the impact. You can expect an acknowledgement within a few days; this is a single-maintainer project, so there is no guaranteed response time.

## Scope and current posture
ResearchLens X currently runs **locally** as a single-user application. It has no authentication, so it must not be exposed to an untrusted network as is. The planned hosted version (authentication, per-user isolation, upload sandboxing, rate limiting) is specified in [docs/production/SECURITY_REVIEW.md](docs/deployment-readiness/SECURITY_REVIEW.md); none of that is implemented yet and no hosted service exists.

Things to know when running it:
- Uploaded PDFs are parsed in-process with PyMuPDF. Treat untrusted PDFs with care.
- Documents and derived data stay on your machine; the application calls no external service except to download model weights from the Hugging Face Hub on first use.
- Never commit `.env`, API keys or user documents (`.gitignore` covers common cases; `.env.example` lists the supported variables).
- The HTML export escapes text taken from PDFs; the Streamlit app renders PDF text with HTML escaping.

## Dependencies
Dependencies are pinned loosely in `requirements*.txt`. CI does not yet run a vulnerability audit; adding `pip-audit` is on the roadmap.
