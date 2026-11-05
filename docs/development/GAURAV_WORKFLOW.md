# Gaurav: Document Intelligence & Application

This guide is written to be followed **without any AI assistant, paid tool or GPU**. Everything you need is in the repo: contracts, fixtures, tests and the `Pipeline` facade.

**You may edit:** `src/researchlens/ingestion/**`, `processing/**`, `knowledge_graph/**`, `app/**`, `tests/test_ingestion*.py`, `tests/test_chunking*.py`, `tests/test_kg*.py`, `tests/test_app*.py`, `data/external/README.md`, `docs/demo/**`.
**Do not edit:** `contracts.py` (if a field is missing, open an issue labelled `interface-change` and tag Pranshu), `pipeline.py`, `retrieval/**`, `verification/**`, `generation/**`.
**Validate:** `python -m pytest -q` must pass before every push; CI runs it again on your PR.

## Setup
```bash
git clone https://github.com/sudo-pranshu/ResearchLensX.git && cd ResearchLensX
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-app.txt && pip install -e .
python -m pytest -q
python scripts/demo.py "Which datasets are used for evaluation?"     # pipeline works on your machine
```
Reading list (≈1 hour): `src/researchlens/contracts.py` (all data shapes), `src/researchlens/pipeline.py` (what the app calls), `data/sample/documents.json` (example Document), PyMuPDF docs on `page.get_text("dict")` (spans with font size/flags).

## G1: PyMuPDF ingestor (Day 1–3) · `src/researchlens/ingestion/pymupdf_ingestor.py`
Input: path to a PDF. Output: `contracts.Document` with `source="pymupdf"`. The step-by-step spec is in the module docstring. In short:
1. `fitz.open(path)`. For each page, `page.get_text("dict")` gives blocks → lines → spans (`text`, `size`, `flags`; bold = `flags & 16`).
2. **Body font size** = the most common span size, weighted by character count.
3. **Title** = the text of the largest spans on page 1 (join consecutive lines of that size). Fallbacks: `doc.metadata["title"]`, then the filename.
4. **Headings** = lines whose size > body + 0.5 pt **or** that are bold and match `^(\d+(\.\d+)*)\.?\s+[A-Z]` or a known name (Abstract, Introduction, Related Work, Method(s), Experiments, Results, Discussion, Conclusion(s), References).
5. **Paragraphs** = body-size blocks between headings. Join lines; fix hyphenation (`"-\n"` → `""`); drop lines repeated on >50% of pages (headers/footers) and lines that are just a page number.
6. **Abstract** = the text after an "Abstract" heading until the next heading. **References** = entries after a References/Bibliography heading, split on `^\[\d+\]` or on a new line starting with a capitalised surname followed by a comma.
7. `doc_id` = first 12 hex chars of `sha1(file bytes)`, or the arXiv id if `arXiv:\d{4}\.\d{4,5}` appears on page 1.

**Tests** (`tests/test_ingestion.py`): generate a 2-page PDF *inside the test* with PyMuPDF (`page.insert_text` with a big font for the title and headings), ingest it, and assert title/sections/paragraphs. No binary PDFs committed. Skip with `pytest.importorskip("fitz")` if PyMuPDF is missing.
**Acceptance:** on 10 open-access arXiv/ACL PDFs (ids listed in `data/external/README.md`), the title is correct on ≥9 and ≥80% of top-level headings are found. Commit the hand-check table as `docs/demo/ingestion_check.md`.

## G2: GROBID client, optional (Day 4, only if Docker is available) · `ingestion/grobid_ingestor.py`
`docker run --rm -p 8070:8070 lfoppiano/grobid:<pinned tag>` → POST the PDF to `/api/processFulltextDocument` → parse the TEI XML (`lxml`) into a `Document` with `source="grobid"`. `get_ingestor()` must return the GROBID ingestor only if `GROBID_URL` responds, otherwise PyMuPDF.

## G3: Chunking (owned; already working) · `processing/chunking.py`
`chunk_document(doc)` exists and is tested. Your job: keep it correct for PDFs (e.g. drop the References section from chunks; merge very short paragraphs < 20 words with the next one in the same section). Add tests for each rule.

## G4: Entities + knowledge graph (Day 5–9) · `knowledge_graph/`
- `entities.py`: `extract_entities(text) -> list[tuple[name, entity_type]]` for dataset/method/metric/task using (a) a gazetteer YAML you write in `knowledge_graph/gazetteer.yaml` (start with ~100 common NLP names: SQuAD, GLUE, BERT, F1, BLEU…) and (b) patterns such as `"(\w+) dataset"`, `"(\w+)-(base|large)"`. Normalise case and hyphens.
- `builder.py`: `GraphBuilder.build(documents, verifications)` → `KnowledgeGraph` following the docstring spec. Use NetworkX internally and convert to the contract at the end.
- Tests: build the graph for `data/sample/documents.json`; both fixture docs must share an entity node for "SciFact", connected by a `shares_entity` edge.

## G5: Streamlit app (Day 1–2 skeleton, Day 5–10 full) · `app/streamlit_app.py`
Pages:
1. **Library**: upload PDFs → ingest → list documents. A "Load QASPER sample (20 papers)" button uses `load_qasper_documents`.
2. **Ask**: question box, k slider, paper filter → `pipe.ask(...)`. Show each answer sentence with `[n]` citation chips, a verification badge (green SUPPORTS / red CONTRADICTS / grey INSUFFICIENT with confidence), and source passages with title + section. **If `answer.is_mock`, show a red "MOCK COMPONENT — not a model prediction" banner.**
3. **Compare**: pick 2–4 papers + a question → `pipe.compare(...)` → one column per paper.
4. **Graph**: KG rendered with pyvis (`st.components.v1.html`) or Plotly. Filter by node type.
5. **About**: component names + versions + fallback status (`answer.generator`, verifier `model`, retriever name).

Build the pipeline once with `@st.cache_resource`. Day 1–2 skeleton: pages 1 (sample load only) and 2 working on `build_default_pipeline(load_sample_documents(), verifier=MockVerifier())`.
Tests (`tests/test_app.py`): `streamlit.testing.v1.AppTest.from_file("app/streamlit_app.py").run()` without exceptions; skip if streamlit is missing.

## Weekly report
Use the format in TEAM_RESPONSIBILITIES.md. If you're blocked for more than 2 hours, post in the team chat with the error and the command you ran.
