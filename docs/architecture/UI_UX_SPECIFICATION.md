# UI / UX specification

Status: **initial specification for approval, to be challenged and refined by the design review (Fable Tasks A-D, see §9).** Tokens and component choices below are starting points, not final design. Positioning: **ResearchLens X — Scientific Research Intelligence Platform**: "An evidence-aware scientific intelligence platform for understanding, verifying and connecting research across multiple publications."

## 1. Why replace the Streamlit interface
Evidence from the current app (screenshots in the audit pass, source `apps/streamlit/`): single global library, top-to-bottom reruns, no routing or deep links (a result cannot be linked or restored), one fixed sidebar for navigation and settings, results disappear on rerun except where patched with session state, no upload progress or per-document status, limited layout control (no resizable panes, no evidence drawer), custom CSS limited to injected HTML, no keyboard shortcuts, little responsive behaviour, graph embedded as an iframe. It communicates function honestly but cannot reach a premium product experience. The new frontend is a dedicated Next.js application over a local API; Streamlit stays until the new UI passes the same end-to-end checks.

## 2. Principles
1. **Evidence first.** Every statement shows where it came from; provenance is one interaction away (hover a citation → quote, page, section; click → source drawer).
2. **Honest by construction.** Mock, experimental, calibrated, candidate and collection-scoped states are first-class, visually distinct and never styled as verified.
3. **Calm density.** Scientific users read a lot: generous line length control, clear hierarchy, restrained colour; no decorative charts, no fabricated metrics.
4. **Fast and predictable.** Skeletons not spinners for known layouts, optimistic UI only where safe, no layout shift, keyboard reachable everywhere.
5. **Reversible and safe.** Destructive actions confirm and say exactly what is deleted.

## 3. Information architecture
Left rail (collapsible, icons + labels): **Workspace** (home) · **Library** · **Search** · **Ask** · **Verify** · **Compare** · **Graph** · **Gaps** · **Review** · **Evidence** (results of evaluation) · **Settings**. Top bar: collection scope selector (all papers or a chosen set, persisted in the URL), system status pill (components real / fallback / MOCK / models downloading), theme toggle, command palette (`Ctrl/Cmd+K`). Right side: contextual **Evidence drawer** (source passage, page, section, retrieval info, status). Every page is a route with deep-linkable state (`/ask?q=...&scope=...`).

## 4. Design system (starting proposal)
- **Typography:** UI in Inter (variable), reading passages in Source Serif 4, identifiers and numbers in JetBrains Mono; all **self-hosted** (local-first, works offline). Scale 12 / 14 / 16 / 18 / 20 / 24 / 30 / 38 px, line heights 1.4-1.6, passage measure 62-72 characters. Font licences (SIL OFL) to be verified when files are bundled.
- **Spacing:** 4 px base grid (4, 8, 12, 16, 24, 32, 48, 64); content max width 1200 px, reading columns 720 px.
- **Colour tokens** (CSS variables; light and dark sets derived, contrast-checked): neutral slate ramp for surfaces and text; one accent (indigo-blue, close to the current `#2F6FED`) for actions and focus; semantic set that is **never decorative**: `supports` green, `contradicts` red, `insufficient` grey, `conflicting` amber, `inferred` blue, `candidate` dashed outline, `experimental` violet tag, `mock` red outline banner. Colour is never the only signal (icon + text label always).
- **Components** (Radix UI primitives + Tailwind, shadcn-style copy-in components, MIT): button, input, textarea, select, combobox, tabs, dialog, drawer, popover/tooltip, toast, table (virtualised), badge/status chip, citation chip, evidence card, progress, skeleton, empty state, error state, file dropzone, command palette, resizable split pane, breadcrumb.
- **Iconography:** one set (Lucide, ISC licence), 16/20 px, stroke 1.5.
- **Motion:** 120-200 ms ease-out for state changes, 200-250 ms for drawers and route transitions, no looping animation, full support for `prefers-reduced-motion`.
- **Themes:** light and dark designed together; system default, user override stored locally.
- **Data visualisation:** graph with Cytoscape.js (fcose layout, canvas rendering, MIT); charts with Recharts (bar/forest-style plots for experiment results with confidence intervals); every chart has a table alternative and text summary.
- **Accessibility target:** WCAG 2.2 AA: contrast ≥ 4.5:1 text and 3:1 UI, visible focus rings, full keyboard operation including graph (node list alternative), ARIA live regions for job progress and results, semantic landmarks, labelled form controls, no information by colour alone, zoom to 200 % without loss.
- **Responsive:** breakpoints 640 / 1024 / 1440; rail becomes a bottom bar on mobile; evidence drawer becomes a full-screen sheet; tables collapse to cards; graph offers a list view on small screens.

## 5. Pages
Each page lists purpose, main elements, data source, states, and honesty elements. **Every control must be wired to a real endpoint or be removed.**
| # | Page | Purpose and main elements | API | Empty / loading / error | Honesty elements |
|---|---|---|---|---|---|
| 1 | **Workspace (home / entry)** | what the product does in 3 lines, collection summary (papers, ready, failed, sections), recent questions, quick actions (upload, ask), system readiness (models cached or not, memory) | `/documents`, `/status` | empty: guided first-upload card; skeleton for stats | no invented counts, readiness from real status |
| 2 | **Library + upload** | dropzone for many PDFs, per-file progress (queued → parsing → indexing → ready / failed with reason), table with title, year, id, sections, references, pages, status, filters, bulk select, remove with confirmation, reuse of already indexed papers | `/documents`, `/jobs` | empty dropzone; skeleton rows; failed rows with explained reason ("scanned PDF: no text layer, OCR not supported") | scanned/encrypted PDFs explained, duplicates reported |
| 3 | **Search** | query box, retriever selector (BM25 / TF-IDF / e5 / hybrid RRF) with an "evaluated on QASPER" note, rerank toggle (dev evidence), section include/exclude, scope, results as evidence cards (paper, section, page, score, retriever, latency), compare-retrievers view | `/search` | empty hint; skeleton cards; model-download notice; fallback notice | experimental configs labelled; fallbacks shown |
| 4 | **Ask** | question, answer sentences with citation chips and status chips, evidence coverage meter, abstention card with the reason, evidence list synced to the answer, copy/cite | `/ask` | empty examples drawn from the library's own sections; streaming-style skeleton; abstention is a normal result not an error | extractive notice; coverage threshold explained; MOCK banner |
| 5 | **Verify** | claim input, verdict (SUPPORTS / CONTRADICTS / INSUFFICIENT) with calibrated confidence only when the validated configuration ran, explanation, per-passage judgements with highlighted rationale sentence, experimental "combined evidence" toggle | `/verify` | empty examples; NLI model download notice with size; model-unavailable state | "not enough evidence ≠ false", calibration scope, experimental label |
| 6 | **Compare** | paper picker, comparison table (methods, datasets, metrics, findings, limitations), each cell with citation chip, shared vs unique entities, same-question-per-paper view, export Markdown | `/compare` | needs ≥ 2 papers state; "not detected" cells | extraction caveat; absent = not detected |
| 7 | **Graph** | interactive graph (papers, methods, datasets, metrics, findings, limitations, authors), filters by node and relation type, node inspector with provenance quotes and pages, **candidate edges dashed and labelled**, list view alternative | `/graph` | empty; layout skeleton; large-graph guard | candidate vs extracted vs measured distinguished; "similar" means lexical similarity |
| 8 | **Gaps** | candidate gaps list with basis chips (author-stated / collection-pattern / absence-in-collection), evidence quotes with pages, suggested research question and methodology template, confidence note | `/gaps` | empty "no gaps from extracted evidence" explanation | collection-scope banner, "not checked against other papers", no novelty wording |
| 9 | **Review** | generate cited literature review, section navigation, citation list, export Markdown / HTML / PDF | `/review` | needs papers; progress; export failures | assembled from verbatim sentences; caveat in header |
| 10 | **Evidence (results)** | committed experiment results with CIs, split labelling (dev / test), register links, negative H1 shown prominently | `/results` | none expected | numbers only from `results/`; dev vs test explicit |
| 11 | **Settings** | retrieval defaults, device, models (cached or not, sizes, download on demand), storage location and size, data export, delete all data, theme | `/config`, `/status` | | sizes and memory stated before downloads |
| 12 | **Document detail** | parsed structure (sections, references), page-by-page text with provenance, original PDF view, delete | `/documents/{id}` | | shows exactly what was extracted, including failures |
Cross-cutting: global error boundary, offline/API-unreachable banner with restart hint, toasts for background job completion, undo for removing a document from a selection (not for deletion).

## 6. Interaction details to get right
- **Citations:** `[n]` chips with hover preview and focus-visible; click opens the drawer at the passage; "page unknown" shown explicitly.
- **Upload:** drag and drop plus file picker, per-file progress from `/jobs`, concurrent files queued, cancel, retry, success toast, no layout jump.
- **Long operations:** job cards with stage names, time elapsed, and a cancel button; the UI stays usable.
- **Model downloads:** a dialog states model, size, destination and memory use before the first download; progress shown.
- **Graph:** pan/zoom, search, focus neighbourhood, node/edge filters, keyboard navigation through a list, selection syncs to the drawer.
- **Print/export:** review view has a print stylesheet; exports download with a descriptive file name.

## 7. Quality gates
Lighthouse accessibility ≥ 95 on every page; axe: zero serious violations; keyboard-only run-through of the full workflow; contrast report; no console errors in a full run; desktop and mobile screenshots for every page stored under `docs/images/`; Vitest component tests; Playwright end-to-end tests; visual regression only where stable.

## 8. Technology choices (to confirm)
Next.js (App Router, TypeScript, static export for local serving), React, Tailwind CSS, Radix primitives, Lucide, Cytoscape.js, Recharts, TanStack Query for server state, a typed client generated from the OpenAPI schema (`openapi-typescript`), Vitest + Testing Library, Playwright, axe-core. All are permissively licensed (**to be verified at install time**) and run offline.

## 9. Design review plan (Fable 5.2, strict five-hour cumulative cap)
Ledger on 2026-11-14: ≈ 0.05 h used (`docs/development/MODEL_USAGE_LEDGER.md`). Keep ≥ 1.0 h reserve ⇒ plan ceiling ≈ 3.9 h; planned allocation 3.5 h. Each task is logged before and after; no task starts without an acceptance criterion; Fable does not write routine components.
| Task | Ceiling | Inputs | Output | Acceptance |
|---|---|---|---|---|
| A. Visual and UX audit | 0.6 h | screenshots of all 9 current pages, Streamlit source, this specification, product requirements | ranked critique: navigation, hierarchy, typography, layout, colour, components, interactions, loading/error states, accessibility, responsive; proposed direction | every page has ≥ 3 concrete, actionable findings; conflicts with this spec resolved in writing |
| B. Design system | 1.2 h | Task A output, §4 | final tokens (type scale, spacing, colour light/dark), component conventions (buttons, panels, modals, citations, status chips, graph, charts), motion principles | tokens file + component sheet rendering in both themes; contrast report passes |
| C. Key screens | 1.2 h | design system, API contract | designs/implementations guidance for the 12 pages, priority on Workspace, Library + upload, Ask, Graph, Verify | each page has layout, states (empty / loading / error), and interaction notes that engineering can build without further questions |
| D. Final visual QA | 0.5 h | screenshots of the built UI (desktop, mobile, both themes), console logs | defect list: inconsistencies, spacing, awkward interactions, broken layouts, missing states | zero "blocker" findings remain or are consciously deferred in writing |
Opus reviews the API contract and the storage/job design before implementation starts; Sonnet implements and integrates all approved recommendations.
