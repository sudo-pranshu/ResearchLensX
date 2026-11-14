# Licences and third-party notices

Reviewed 2026-11-14 from package metadata installed in the development environment and from the Hugging Face model metadata API. Re-check before redistributing anything.

## This project
**GNU Affero General Public License v3.0 or later** (`LICENSE`, SPDX `AGPL-3.0-or-later`).
Why AGPL: the PDF ingestion engine depends on **PyMuPDF**, which is dual-licensed *GNU AGPL 3.0 or an Artifex commercial licence*. A project that imports PyMuPDF and is distributed or offered as a network service must itself comply with the AGPL, so a permissive licence on top would be misleading. Consequences: anyone running a modified version as a network service must offer its source to its users. To relicense permissively, replace PyMuPDF (for example with another PDF text extractor under a permissive licence) and re-audit; the ingestion code is isolated in `src/researchlens/ingestion/`. This is the maintainer's decision to confirm.

## Python dependencies (as installed)
| Package | Licence |
|---|---|
| PyMuPDF 2.1 | AGPL-3.0 or Artifex commercial (dual) |
| Streamlit | Apache-2.0 |
| pyvis | BSD |
| Plotly | MIT |
| NetworkX | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| SciPy, NumPy | BSD-3-Clause (NumPy bundles 0BSD/MIT/Zlib/CC0 components) |
| Pydantic | MIT |
| PyYAML | MIT |
| lxml | BSD-3-Clause |
| PyTorch | BSD-style and Apache-2.0 components |
| Transformers, Sentence-Transformers, Adapters | Apache-2.0 |
| faiss-cpu | MIT |
| SentencePiece | Apache-2.0 |
| protobuf | BSD-3-Clause |
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| pytest | MIT |

## Pretrained models (downloaded from the Hugging Face Hub at run time; not redistributed here)
| Model | Use | Licence (per Hub metadata) |
|---|---|---|
| intfloat/e5-base-v2 | dense retrieval | MIT |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | reranking | Apache-2.0 |
| MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | NLI verifier | MIT (its training data include MNLI, FEVER and ANLI under their own terms) |
| roberta-large-mnli | independent NLI judge (experiments) | MIT |
| Qwen/Qwen2.5-1.5B-Instruct | generator in H2 experiments only | Apache-2.0 |
| allenai/specter2_base | dense retrieval (experiments) | Apache-2.0 |
| allenai/specter2 and adapters, allenai/scibert_scivocab_uncased | experiments | **licence not declared in the Hub metadata read; verify before any redistribution** |

## Datasets (downloaded by scripts; not redistributed)
| Dataset | Use | Licence |
|---|---|---|
| QASPER v0.3 | retrieval and H2 evaluation | CC BY 4.0 |
| SciFact | verifier evaluation, retrieval | CC BY-NC 2.0 (**non-commercial**; relevant if the project is ever offered commercially) |

The repository's `results/` folders contain metrics, identifiers and model outputs derived from these datasets, but not their source text. `tests/fixtures/` and `data/sample/` are small synthetic fixtures written for this project.

## Papers used for demonstrations
Papers uploaded by users or used in demonstrations remain under their authors' licences. The repository does not contain research PDFs; screenshots show short excerpts from openly available arXiv papers for illustration.
