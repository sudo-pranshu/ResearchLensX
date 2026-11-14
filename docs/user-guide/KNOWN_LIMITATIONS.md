# Known limitations

Honest list, kept current. "Measured" means a committed run in `results/`; everything else is a design limit or an untested area.

## Scientific status
- **H1 (hybrid beats every single retriever): not supported** on the QASPER test split (EXP-H1-006): RRF(BM25+e5) R@5 .486 vs e5 .501 (Δ −.015, p_holm .10). The hybrid does beat BM25, TF-IDF and LSA.
- **Reranking helps on dev only** (EXP-H1-007: R@5 .502 → .580 with the hybrid first stage). It has not been run on the test split, and the reranker is a web-search model (MS MARCO), not a scientific one.
- **H2 (verified RAG improves citation correctness): partially supported on a dev pilot only.** EXP-H2-004 (100 Q, 1.5B generator, one-shot citation prompt): citation precision .420 → .785 on the 77 paired non-abstained questions (CI excludes 0; baseline .443 over all 100), but the full treatment fails the pre-set Answer-F1 guard (−.060) because dropped unsupported sentences make 23% of answers abstain. The "no-drop" variant that keeps Answer-F1 is a post-hoc ablation and must be pre-registered before a test run. The verifier V and judge J are both NLI models (different checkpoints), Answer-F1 is lexical, and no 7-8B generator or test-split run exists (needs Colab; not executed).
- **H3 (cross-paper reasoning) is exploratory only**; cross-paper NLI edges were invalid on raw output (0/9) and are now gated and labelled candidates.

## Components
- **Ingestion:** text PDFs only (no OCR). Two-column order relies on PyMuPDF block order; tables/equations become noisy text; references are split heuristically (counts approximate; BERT over-counts); authors come only from PDF metadata (often empty); year is the arXiv id's year when present, else unknown. Validated on 5 real arXiv papers plus synthetic PDFs, not on a broad sample.
- **Answers are extractive** (sentences copied from cited passages). No LLM generation in the app. This guarantees traceability and limits fluency and synthesis.
- **Abstention** by lexical coverage is a weak signal (AUROC .64, EXP-QA-001) and is set conservatively (0.2). The better signal is NLI status on each sentence.
- **NLI** is one model (DeBERTa-v3 trained on MNLI/FEVER/ANLI) used out of domain. SciFact dev macro-F1 .60-.68. Confidence is calibrated (ECE .18 → .07 out-of-fold on dev) but calibrated confidence is not truth. The app default is the validated configuration (passage-level premises, temperature fitted on those probabilities). "Combined evidence" support (several passages jointly entail a claim, status *inferred*) is an opt-in experimental mode: sentence-level, uncalibrated, unit-tested but **never evaluated** on real data.
- **Knowledge graph:** rule-based extraction; in practice an entity graph with few paper-to-paper edges (reference matches, shared entities). Manual audit on 5 papers (single annotator): methods ~76% precise, datasets 10/10 precise with low recall. No "extends" relation exists. The graph is not a citation database: `cites` edges only link papers inside the uploaded collection.
- **Research gaps:** an observed false positive: a method-description sentence ("…we cannot solve it with a single beam search") was emitted as a stated limitation; precision is unmeasured.  three bases (author-stated, collection-pattern, absence-in-collection). They concern the uploaded papers only; **novelty in the wider literature is never claimed or checked**. Confidence is capped at "medium"; proposed research questions and methodologies are templates. Rules were tightened by inspecting the same 5 papers, so apparent quality on them is optimistic. A blank manual-review sheet is in `docs/review/` for an independent label. An external methodology review set out a held-out protocol (3 collections from other sub-areas fixed in advance, labels valid/partial/invalid with attribution and fidelity scored separately, kappa on a re-labelled subset, Wilson CIs); it has **not** been run. "Missing comparison" gaps mean only that the extractor did not detect the pairing.
- **Literature review:** assembled from verbatim sentences, no paraphrase; reads as a list of cited evidence, not prose.

## Engineering
- Developed and tested on an 8 GB Apple-silicon laptop; one transformer is kept resident at a time. Colab GPU notebooks are provided but were not executed in this project.
- Tested on Python 3.12 / macOS only. The app is single-user (local SQLite, no authentication).
- Dev-split results guided design choices (rerank depth, thresholds, calibration temperature); only EXP-H1-006 is a test-split result.
- The `.venv` is needed for PyMuPDF/Streamlit/pyvis; the base environment suite skips the tests that need them.
