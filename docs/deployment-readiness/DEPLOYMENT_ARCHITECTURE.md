# Deployment architecture (zero additional cost)

Status: **proposal, not approved. Nothing has been deployed.** Provider facts were read from official documentation on 2026-11-14 (links in section 7). Free tiers change; each fact marked **unverified** must be re-checked before launch.

## 1. Workload that has to be hosted
- Python ML API: PyMuPDF ingestion, BM25/TF-IDF, e5-base-v2 + FAISS, MiniLM cross-encoder, DeBERTa-v3 NLI, knowledge graph, gap and review generation. Models ≈0.87 GB on disk; torch CPU ≈0.6 GB; plan 2-3 GB RAM resident; inference is CPU-bound and takes seconds to minutes (first embedding of a corpus).
- Persistent per-user storage: PDFs, parsed documents (SQLite), embedding caches.
- Authentication, multi-user isolation, background jobs, file uploads (multi-MB PDFs).
- A static/SSR frontend (Next.js + TypeScript + Tailwind) with a landing page.

## 2. Verified provider facts
| Provider / plan | What the official docs say (checked) | Consequence for us |
|---|---|---|
| **Hugging Face Spaces** | "Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts…". Static Spaces free. Free accounts may host up to 2 Gradio Spaces on ZeroGPU. CPU Basic hardware (2 vCPU, 16 GB, 50 GB **not persistent** disk) has "no hourly cost" but needs the paid plan to create; PRO is listed at $9/month. Free hardware sleeps when unused; Docker disk is lost on restart; persistence needs a paid Storage Bucket or an external store | **A free Docker/Streamlit backend is not available**, so HF fails the zero-cost rule. ZeroGPU/Gradio is a different programming model (GPU time slices, no persistence) and is not a fit for a stateful multi-user API |
| **Vercel Hobby** | Functions: max duration 300 s, 2 GB / 1 vCPU. Hobby includes 100 GB fast data transfer, 1 M function invocations, 4 CPU-hours active CPU. Exceeding a limit pauses the feature until 30 days pass. Runtime logs 1 h. 100 deployments/day. Source upload limit 100 MB via CLI | Good for a static/SSR frontend. **Not** a persistent ML server: no model residency, 1 vCPU, no persistent disk. Request-body limit for functions and the Hobby "personal / non-commercial" terms were **not** on the pages fetched (**unverified**: read Vercel's Fair Use Guidelines and terms before launch). Never route PDF uploads through Vercel functions |
| **Supabase Free** | 500 MB database, 1 GB file storage, 50,000 monthly active users, 5 GB egress. Projects with low activity over 7 days are **paused** (restorable within one year). Max upload size on Free: not stated on the pages fetched (**unverified**) | Excellent for auth (and row-level security). 1 GB storage is too small for research PDFs; use it for identity and small metadata only. Pausing means a keep-alive or tolerance of occasional resume |
| **Render Free web service** | "Do not use them for production applications." Spins down after 15 min idle (~1 min cold start), **ephemeral** filesystem, no persistent disk, 750 instance hours/month, may restart at any time. RAM/CPU not stated on the fetched page (**unverified**) | Fails reliability and persistence; almost certainly too little RAM for torch + three models |
| **Oracle Cloud Always Free** | Ampere A1 (Arm): first 1,500 OCPU-hours and 9,000 GB-hours per month = **2 OCPU, 12 GB RAM**; 2 AMD micro VMs (1 GB each); **200 GB** boot+block storage, 5 backups; **10 TB/month** outbound. Instances may be reclaimed if idle over 7 days (95th-percentile CPU, network and (A1) memory all below 20%). "Out of host capacity" errors can occur. Card-verification requirement not stated on the fetched page (**unverified**; commonly required, so confirm no charge path before signing up) | The only verified free option with enough RAM, a real disk and no sleep. Costs: you administer a Linux VM (patching, TLS, backups, monitoring), Arm builds must be tested, and idle-reclaim and capacity risks exist |
| **Google Cloud Run / others** | not fetched | Excluded from consideration: free usage there is tied to a billing account (**unverified**); the zero-cost rule makes a card-linked metered service a risk |

## 3. Candidate architectures
### A. Vercel frontend + Python API on a free VM + Supabase for identity
```
Browser ──> Vercel (Next.js, static + SSR)  ──(HTTPS, CORS allow-list, JWT)──> API VM (Oracle A1)
   │                                                                    │ FastAPI + worker
   └── Supabase Auth (email link / OAuth) <── JWT verification ─────────┤ /data/users/<uid>/ (PDFs, SQLite, caches)
```
- Frontend on Vercel Hobby: fast CDN, preview deployments, the landing page stays up when the VM restarts.
- API on the Oracle A1 VM: FastAPI + a job worker, Caddy for TLS on a free hostname (e.g. DuckDNS) or a Cloudflare Tunnel; models resident; data on the 200 GB disk.
- Supabase: sign-up/login and JWTs only; documents never leave the VM.
- Pros: best user experience and resilience of the public site; clean separation; free tiers all fit. Cons: three providers, CORS/cookies across origins, Vercel terms to confirm, one VM is still a single point of failure.

### B. Single host: FastAPI serves the built Next.js static export and the API from the same VM
```
Browser ──HTTPS──> Caddy ──> FastAPI (API + static frontend)  [Oracle A1 VM]  <── Supabase Auth JWT
```
- One origin (no CORS), one deployment, one place to monitor.
- Pros: simplest for a single developer, fewest moving parts, no Vercel terms risk. Cons: landing page shares the VM's fate and bandwidth (10 TB/month is ample); no global CDN; Next.js limited to static export (no SSR), which this app does not need.

### C. Rejected: HF Spaces, Render, Vercel-only
Excluded above on verified grounds (paid plan needed, not for production/ephemeral, not an ML server).

## 4. Comparison
| Criterion | A: Vercel + VM + Supabase | B: single VM + Supabase |
|---|---|---|
| Reliability | Landing page survives backend outage; API still single VM | Everything down if the VM is down |
| Free-tier sustainability | Three free tiers (limits verified/unverified as above) | Two |
| Simplicity (one developer) | Medium (CORS, 3 consoles) | **High** |
| Security surface | Cross-origin cookies/CORS to get right | Same-origin, smaller surface |
| Model inference | Same VM in both | Same |
| Uploads | Direct to VM API (never via Vercel) | Direct |
| Persistent storage | VM disk 200 GB (+ backups) | Same |
| Maintenance | Vercel is zero-ops | Zero extra ops beyond the VM |

## 5. Recommendation
**Start with B (single VM + Supabase Auth), keeping the frontend a pure static export so it can be moved to Vercel (A) later with a configuration change.** It is the simplest to secure and run alone. Move to A only if you want the landing page to survive VM downtime and Vercel's terms are confirmed to permit this use.
**Hard prerequisites before any deployment** (to be settled with you): (1) confirm an Oracle Always Free account can be created without a payment path that can incur charges; (2) build and run the Docker image on an Arm host and measure RAM, first-embedding time and query latency; (3) confirm Supabase Free auth limits suit expected traffic; (4) decide the public hostname (free subdomain) and TLS approach; (5) agree quotas (documents per user, MB per file, total MB per user).
**Fallback if Oracle is not acceptable:** a local-first release (the same app runs on the user's machine) plus a static public site with a recorded demonstration. A zero-cost hosted multi-user ML backend with persistent storage does not otherwise exist among the providers checked.

## 6. Capacity plan (to be validated, not promised)
- Resident: torch + e5 + MiniLM + DeBERTa ≈ 2-3 GB (A1 has 12 GB) → one model set shared by all users.
- Concurrency: CPU-bound; a job queue with 1-2 workers and per-user limits; upload returns immediately with a job id; the UI polls status.
- Storage: quota e.g. 20 documents / 25 MB each / 300 MB per user → ≈ 30 users per 10 GB before caches; caches are deletable and rebuildable.
- Cold start: models load at API start (warm-up state shown in `/health`).
- Backups: nightly SQLite + file archive to the free 5 block-volume backups.

## 7. Sources read (2026-11-14)
huggingface.co/docs/hub/spaces-overview · huggingface.co/docs/hub/spaces-sdks-docker · huggingface.co/pricing · vercel.com/docs/limits · vercel.com/docs/plans/hobby · supabase.com/docs/guides/platform/billing-on-supabase · supabase.com/docs/guides/platform/free-plan-pausing · render.com/docs/free · docs.oracle.com (Always Free resources). Two of the fetched pages contained text addressed to AI agents (feedback-report requests); it was ignored and is not part of this analysis.

## 8. Not decided here
Final choice (A or B), domain/hostname, quotas, whether the hosted product is invite-only at first. Deployment starts only after you approve this document.
