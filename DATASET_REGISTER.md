# Dataset Register

Summary: [PROJECT_BIBLE.md §F](PROJECT_BIBLE.md#f-dataset-register-summary). Raw data is **never committed**. Each dataset has a download script, and its licence is recorded here before first use.

| ID | DS-01 QASPER | DS-02 SciFact |
|---|---|---|
| Version | v0.3 (files dated 2022-03) | 2020 release (`data.tar.gz`) |
| Source | AllenAI S3: `qasper-train-dev-v0.3.tgz`, `qasper-test-and-evaluator-v0.3.tgz` | AllenAI SciFact repo release (verify the URL in `scripts/download_scifact.sh`, Pawan) |
| Licence | CC BY 4.0 | CC BY-NC 2.0: non-commercial use only; fine for coursework |
| Local path | `data/raw/qasper/` | `data/raw/scifact/` |
| Download | `scripts/download_qasper.sh` (~14 MB compressed, ~62 MB extracted) | `scripts/download_scifact.sh` (to write, Pawan, Day 1) |
| Splits | train 888 papers / 2,593 Q · dev 281 / 1,005 · test 416 / 1,451 (**verified Day 1**) | claims train 809 · dev 300 · test 300 (test labels hidden); corpus 5,183 abstracts |
| Used for | H1 primary (test), H2 (test; dev for tuning), app demo corpus | H1 test 2 (dev), H2 verifier (train/dev) |
| Audit (Day 1) | Paragraphs: train 52,959 · dev 15,122 · test 22,588. Questions without evidence: 356 / 82 / 100. Unmatched evidence strings: 147 / 129 / 222, all classified as section headings in dev (see review R6) | pending |
| Leakage notes | SPECTER2/SciBERT pretraining corpora may include these papers (document exposure only) | Claims written from citances; LLMs may have seen them |
| Status | ✅ downloaded, audited | ⏳ Day 1–2 |

## Explicitly not used (14-day scope)
SciRepEval, SciDocs, PubMedQA, DocLayNet (≈28 GB), ScholarQABench. See review R1/R13. Adding one requires a register entry and a concrete experiment.

## User-supplied PDFs
Demo PDFs go in `data/external/pdfs/` (gitignored). Use only open-access papers (arXiv / ACL Anthology) and record their ids in `data/external/README.md`.
