# Demonstration workflow (about 10 minutes)

Prerequisites: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt -r requirements-dense.txt -r requirements-app.txt && pip install -e .` Two or three open-access PDFs on different but related topics (e.g. arXiv papers). Nothing needs an API key.

```bash
streamlit run app/streamlit_app.py
```

| Step | Page | What to show | What it demonstrates |
|---|---|---|---|
| 1 | Library | Upload 3 PDFs; upload one a second time; upload a non-PDF | Parsing with pages; duplicate and failure reporting |
| 2 | Search | Query a method name; turn on *Compare retrievers* | BM25 vs TF-IDF vs rerank side by side with latencies; page-cited passages |
| 3 | Ask | "Which datasets are used for evaluation?" | Cross-paper evidence balancing, cited extractive answer, page references; then an off-topic question to show visible abstention |
| 4 | Ask (toggle *Verify answer sentences with NLI*) | Re-run the question | Per-sentence status pills (supported / inferred / conflicting / unsupported), calibrated confidence |
| 5 | Verify a claim | "BEIR contains 18 zero-shot evaluation datasets." | NLI verdict with explanation and the exact evidence sentence |
| 6 | Compare papers | Select all papers; ask one question of every paper | Methodology table with page references; shared vs unique methods/datasets; Markdown export |
| 7 | Knowledge graph | Toggle node types; inspect a node | Provenance for every node and edge; (optional) NLI-linked relations marked as candidates |
| 8 | Research gaps | Expand two gaps | Basis, evidence quotes, template research question; "not checked" wording; no novelty claim |
| 9 | Literature review | Generate; download PDF | Every statement cited [n] to paper/section/page/quote |
| 10 | Evaluation | Pick EXP-H1-006 and EXP-H1-007 | The negative H1 result and the reranking gain, with CIs |

If a model cannot load (offline, no download), the sidebar shows the component as **fallback** or **MOCK**, and a red MOCK banner appears on any output it touched.
