# ResearchLens X: technical report

**Evidence-aware scientific literature understanding and cross-paper reasoning with hybrid NLP**
Pranshu Kumar, B.Tech E&TC final-year project. Version of 2026-11-14. Source of truth for numbers: `results/` and the register in `PROJECT_BIBLE.md` §G. Limits: `KNOWN_LIMITATIONS.md`.

## 1. Summary
ResearchLens X is a local application and an evaluation framework for working with a set of uploaded research papers: parse PDFs, search them, answer questions with page-level citations, check claims against the text with an NLI model, compare papers, build a provenance-grounded knowledge graph, list candidate research gaps and export a cited literature review. Two hypotheses were pre-registered and tested; one was **not supported** (H1) and one is **not yet decided** (H2). A third (H3, cross-paper reasoning) is exploratory. Everything the application shows is traceable to a quote and a page, and no text is written by a generative model in the application.

## 2. Hypotheses and results

### H1: hybrid lexical + dense retrieval beats every single retriever
Setting: QASPER v0.3, within-paper paragraph retrieval, Recall@5, paper-clustered bootstrap (10k), Holm-adjusted. Treatment fixed before the test run: RRF(BM25 + e5-base-v2, k=60, depth 100).

| System (test split, 1,351 Q / 414 papers) | R@5 |
|---|---|
| e5-base-v2 | **.501** |
| linear(BM25+e5) | .496 |
| RRF(BM25+e5) | .486 |
| LSA | .430 |
| BM25 | .415 |
| TF-IDF | .382 |

RRF beats BM25, TF-IDF and LSA (all p_holm < .001) but not e5 (Δ −.015, 95% CI [−.032, +.002], p_holm .10). **H1 is not supported.** The same pattern held on QASPER dev and on SciFact abstract retrieval (nDCG@10: e5 .870, RRF .855, n.s.). Open-corpus retrieval on QASPER is ill-posed (questions are written about one paper) and was dropped as a primary setting (EXP-H1-002).

### Cross-encoder reranking (secondary, dev split only)
EXP-H1-007 (923 Q / 279 papers, first-stage depth 20, set before the run): reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2` raised R@5 for every first stage: BM25 .435 → .554, e5 .500 → .575, RRF .502 → **.580** (RRF vs rerank(RRF): +.078 [+.052, +.104], p_holm .0003). This is a dev result with a web-search reranker; no test-split run exists.

### H2: claim-level NLI verification improves citation correctness over conventional RAG
Verifier component (SciFact dev, 340 claim-abstract pairs, macro-F1): TF-IDF+LR .518, DeBERTa-v3 (MNLI+FEVER+ANLI) sentence .596, tuned .633, passage-level .681; independent judge (roberta-large-mnli) .565. Calibration (EXP-H2-003): ECE .183 → .067 out-of-fold with temperature 2.24; accuracy unchanged.
End-to-end pilot EXP-H2-002 (100 dev questions, Qwen2.5-1.5B) was **inconclusive**: the generator emitted 0.15 citations per answer, so the large gain in citation precision/recall (.08 → .70) came from the verifier adding citations, and answer-F1 fell 3.7 points. EXP-H2-004 repeats the pilot with a one-shot citation prompt (section 4).

### H3: cross-paper reasoning (exploratory)
Raw NLI cross-paper "contradicts" edges between findings were 0/9 valid on 5 real papers (off-topic passages are called contradictions with high confidence). Edges now require topical relevance and a shared extracted entity, are labelled candidates, and are never created by a mock. On the 5 unrelated test papers no edge survives, which is the honest outcome for that collection.

## 3. System
Pipeline: PDF → `PyMuPDFIngestor` (sections, per-paragraph pages, references) → `CorpusStore` (SQLite) → chunks → BM25/TF-IDF (+ optional cross-encoder rerank, redundancy suppression, section filters) → `MultiDocQA` (cross-paper balancing, cited extractive answers, coverage-based abstention) → `NLIVerifier` (supported / inferred / conflicting / contradicted / unsupported, deterministic explanations, calibrated confidence) → knowledge graph, comparison table, candidate gaps, cited review (Markdown/HTML/PDF). See `ARCHITECTURE.md` §7 and `TECHNICAL_DECISIONS.md`.

Guarantees tested in the suite (129 tests): every answer sentence is copied from a passage it cites; page numbers come from the PDF parse and are `None` when unknown, never invented; every graph edge carries evidence; mock components are flagged and cannot create scientific relations; every review citation resolves to a paper, section, page and quote; HTML export escapes untrusted PDF text.

Measured side results: abstention by lexical evidence coverage separates unanswerable QASPER questions weakly (AUROC .636, EXP-QA-001), so the default threshold is conservative; graph extraction audit on 5 papers (single annotator): methods about 76% precise, datasets 10/10 precise but low recall (EXP-KG-001).

## 4. EXP-H2-004 (RAG vs verified RAG, one-shot prompt)
Same 100 dev questions, retriever, verifier and thresholds as EXP-H2-002; only the prompt changed (one worked example showing the `[n]` citation format, shared by every condition). Manipulation check, set beforehand (baseline must cite at least 0.7 times per answer): **passed** (0.99 vs 0.15).

| Condition | Citation P | Citation R | Answer-F1 | Abstained answers |
|---|---|---|---|---|
| Conventional RAG | .443 (paired 77: .420) | .435 (paired: .409) | .286 | 0 |
| Verified RAG (full) | **.785** (paired 77) | **.768** (paired 77) | .225 | 23 |
| Verified, no reassignment | .442 | .442 | .225 | 23 |
| Verified, no dropping (post hoc) | .703 | .680 | .286 | 0 |

Citation scores are only defined for answers that still contain sentences, so the paired comparison uses the 77 questions where verified RAG did not abstain; the other rows are averaged over all 100.

Verified RAG raises citation precision by +.365 [+.264, +.468] and recall by +.359 (cluster bootstrap over the 77 paired questions, Holm-adjusted p = .0001), but **fails the pre-set guard** that Answer-F1 may fall by at most 2 points (Δ −.060 [−.088, −.035]) because unsupported sentences are dropped. Reassignment (re-citing the best supporting passage found by the verifier) is what produces the gain: without it the effect vanishes (+.022, n.s.). The no-dropping variant keeps Answer-F1 unchanged and still gains +.259 precision, but it was selected after seeing results, so it is exploratory and must be pre-registered before any test run. Caveats: 100 dev questions, a 1.5B generator, V and J are both NLI models, Answer-F1 is lexical overlap. **H2 is partially supported on dev; it is not confirmed.**

## 5. Validity and limitations
Read `KNOWN_LIMITATIONS.md`. In short: one hypothesis tested on its final split and rejected; the others rest on dev results; the extractive generator limits answer fluency; the gap engine is rule-based, collection-scoped and its quality is only audited informally (an independent labelled evaluation protocol is specified but not run); Colab notebooks are provided but were not executed here.

## 6. Reproducibility
`./scripts/run_eval.sh` (tests + CPU experiments) and `./scripts/run_eval.sh --full`. Every committed experiment run has `config.json` with the git hash, dataset version and environment (the two small analysis runs EXP-QA-001 and EXP-H2-003 write theirs too). QASPER v0.3 is CC BY 4.0 and SciFact CC BY-NC 2.0; raw data and PDFs are not redistributed. The application needs no API key and runs offline once the models are cached.
