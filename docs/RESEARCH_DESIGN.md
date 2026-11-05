> **Superseded** by v1.0 (PROJECT_BIBLE.md §B); see docs/RESEARCH_DESIGN_REVIEW.md for what changed and why. Kept unchanged as a record.

# ResearchLens X — Research Design (v0.1)

*Evidence-Aware Scientific Literature Understanding and Cross-Paper Reasoning Using Hybrid NLP*

Status: draft for supervisor review. Sections A–D of the architecture are worked out in detail. Verification, the knowledge graph and generation are specified only as far as the H2/H3 experiments need. Dataset licences and sizes below come from the dataset papers and repos. **Every row marked ⚠ must be re-checked at the source before you download or cite it.**

---

## 1. Research question

> Can a hybrid architecture combining scientific-domain retrieval, claim-level verification, evidence attribution and cross-document relationship reasoning improve the factual reliability and evidence coverage of scientific literature synthesis compared with conventional RAG baselines?

This breaks into three falsifiable hypotheses. Each one is tested against a named baseline, on a held-out test set, with a primary metric fixed **before** looking at test results. A hypothesis counts as *supported* only if the primary metric improves by a statistically significant margin (α = 0.05 after Holm correction) **and** the 95% CI of the improvement excludes zero. A negative result is a valid outcome and gets reported as one.

---

## 2. Hypotheses, operationalised

### H1 — Hybrid lexical + semantic retrieval beats lexical-only and dense-only

| Item | Specification |
|---|---|
| Task | Evidence-paragraph retrieval for QASPER questions |
| Settings | **(a) within-paper**: rank only that paper's paragraphs (the QASPER-native setting). **(b) open-corpus**: rank every paragraph of every paper in the split (harder; closer to the multi-paper use case) |
| Systems | B1 BM25 · B2 TF-IDF cosine · B3 LSA (TF-IDF→SVD; classical dense) · D1 general-purpose dense encoder (e.g. `e5-base-v2` or `bge-base-en-v1.5`) · D2 SciBERT mean-pooled (expected to be weak untuned: a useful negative control) · D3 SPECTER2 (proximity adapter for passages, adhoc-query adapter for questions) · **H** RRF(best lexical, best dense) · H-w weighted RRF / linear score fusion |
| Gold | Union of evidence paragraphs across annotators; questions with no evidence (unanswerable, or every annotator left evidence empty) are excluded and the count is reported |
| Primary metric | Recall@5 (setting a), nDCG@10 (setting b) |
| Secondary | MRR@10, Recall@{1,3,10,20}, Hit@k; latency per query |
| Statistics | Per-question paired comparison: paired bootstrap (10k resamples) for CIs, paired randomisation test for p-values; Holm–Bonferroni over the comparisons H vs each single system |
| Ablations | RRF k ∈ {10, 30, 60, 100}; fusion depth {20, 50, 100}; with/without section-title prefixing; paragraph vs fixed-window chunks; stop-word removal on/off for BM25 |
| Extra test | SciFact abstract retrieval (claim → abstracts, 5,183-doc corpus) as an **out-of-dataset replication** of H1 |

The SciDocs/BEIR numbers are reported for context only, because SPECTER2 was trained on related S2 data (see §4).

### H2 — Claim-level verification improves factual consistency and citation correctness

| Item | Specification |
|---|---|
| Baseline (conventional RAG) | Best H1 retriever → top-k passages → LLM generates an answer with inline `[n]` citations. Fixed prompt, temperature 0 |
| Treatment | The same generation, then: (1) split the answer into atomic claims, (2) run an NLI/claim-verification model on each (claim, cited passage) pair, (3) re-attribute each claim to its best-supporting retrieved passage, or drop/hedge it if unsupported, (4) regenerate from verified claims only |
| Test set | QASPER test questions that are answerable and have evidence. The **verifier component** is evaluated separately on SciFact dev for label prediction (SUPPORT / CONTRADICT / NEI) |
| Primary metric | Citation precision & citation recall (ALCE-style, Gao et al. 2023). Computed with an NLI judge **different from the verifier in the pipeline**, since using the same model for both would be circular |
| Secondary | QASPER Answer-F1 (to check that verification doesn't hurt answer quality); unsupported-claim rate; answer length; SciFact macro-F1 for the verifier |
| Human evaluation | 100 questions × 2 systems, blinded and randomised order. Each claim is rated supported / partially / unsupported, and each citation correct or not. A second annotator does 30% of items; report Cohen's κ |
| Statistics | Paired bootstrap on per-question citation P/R; McNemar on per-claim supported/unsupported; Wilcoxon signed-rank on human scores |
| Ablations | Verify-only (flag, no revision) vs verify + revise; NLI model choice; claim-splitting by an LLM vs rule-based sentence splitting; k retrieved passages |

### H3 — Cross-paper relationship reasoning improves multi-document QA

| Item | Specification |
|---|---|
| Baseline | Independent passage retrieval: top-k from the open corpus using the best H1 system, passages treated independently |
| Treatment | Relation-aware evidence selection over a cross-paper graph. Nodes are papers, claims and entities (task, dataset, method, metric). Edges are citations, shared entities, and claim–claim support/contradiction from the H2 verifier. Retrieval is expanded or reranked to cover the papers the graph connects to the question |
| Test sets | **Known gap:** none of the primary datasets is a ready-made cross-paper QA benchmark (QASPER is single-paper). Plan: (i) ScholarQABench multi-paper subset, if its licence allows (⚠); (ii) SciFact claims whose evidence spans ≥2 abstracts (count them first, since the subset may be small); (iii) a **constructed set of ~120 cross-paper comparison questions** over a QASPER paper subset, written under fixed guidelines, with a second annotator checking 30% |
| Primary metric | Evidence coverage: the fraction of gold-required papers that appear among the cited evidence |
| Secondary | Multi-doc Recall@k; human rubric (correctness, completeness, contradiction surfaced Y/N) |
| Statistics | Paired bootstrap / randomisation. With n≈120 the test is underpowered for small effects, so report effect sizes with CIs rather than only p-values |
| Ablations | Drop each edge type (citation / entity / claim-relation); graph hop depth 1 vs 2 |

---

## 3. Datasets

| Dataset | Size | Licence | Access | Splits used | Role |
|---|---|---|---|---|---|
| **QASPER** v0.3 | 5,049 Q over 1,585 NLP papers; train 888 papers / 2,593 Q, dev 281 / 1,005, test 416 / 1,451 | CC BY 4.0 | Raw JSON from AllenAI S3 (`qasper-train-dev-v0.3.tgz`, `qasper-test-and-evaluator-v0.3.tgz`); also `allenai/qasper` on HF | dev = development, test = final, run once | H1 primary, H2 primary, H3 paper pool |
| **SciFact** | 1,409 claims, 5,183-abstract corpus; train 809 / dev 300 / test 300 | CC BY-NC 2.0 ⚠ (non-commercial; fine for a thesis) | `allenai/scifact` GitHub / HF; also in BEIR | train = fit verifier, dev = report. Test labels are hidden (leaderboard) ⚠ | H1 replication, H2 verifier, H3 multi-evidence subset |
| **SciRepEval** | 24 tasks, several are large | Mostly ODC-BY; per-task varies ⚠ | `allenai/scirepeval` on HF | Only selected eval tasks (e.g. search, proximity) | Representation sanity checks. SPECTER2 was trained on its training tasks, so treat it as in-domain for SPECTER2 |
| PubMedQA | PQA-L 1k expert-labelled; PQA-U 61.2k; PQA-A 211.3k | MIT | GitHub `pubmedqa/pubmedqa` | PQA-L only | Optional domain-transfer check for H2 |
| SciDocs | 7 tasks; BEIR SCIDOCS: 25,657 docs, 1,000 queries | ⚠ verify (repo licence differs from BEIR redistribution) | BEIR / `allenai/scidocs` | test | Context numbers for H1 only |
| DocLayNet | 80,863 pages, ~28 GB core | CDLA-Permissive-1.0 | IBM / HF | **Do not download.** Only needed if you train a layout model, and GROBID already covers layout | Optional |
| ScholarQABench | Multi-domain literature-synthesis QA (OpenScholar, 2024) | ⚠ verify | OpenScholar release | multi-paper subset | H3 candidate |

**Download budget:** QASPER + SciFact together are well under 100 MB, which is enough for H1, H2 and most of H3. Download nothing larger until a concrete experiment needs it.

---

## 4. Leakage and validity register

| Risk | Where | Mitigation |
|---|---|---|
| Tuning on test | All | Tune every hyperparameter (RRF k, fusion depth, chunking, thresholds) on QASPER **dev** or SciFact **dev**. Run test exactly once per final system and log the commit hash |
| SPECTER2 pre-training overlap | SPECTER2 was trained on S2 citation data and SciRepEval train tasks, so QASPER papers may be in its training corpus | Report it. Training was citation-based, not on QASPER QA labels, so this is document exposure, not label leakage. The SciDocs comparison is unfair to non-SPECTER systems and is reported only as context |
| LLM memorisation | QASPER/SciFact are public; the generator LLM may have seen them | Same LLM for baseline and treatment, so the comparison is paired. The absolute answer-F1 is not claimed as the contribution |
| Circular evaluation | H2: verifier = judge | Judge NLI model ≠ verifier model; plus human evaluation |
| Annotator disagreement | QASPER has multiple annotations per question | H1 uses the union of evidence; report the max-over-annotators variant as well |
| Self-constructed H3 set | Author bias | Write the questions **before** building the H3 system; written guidelines; second annotator; release the set |
| Multiple comparisons | Many system pairs | Holm–Bonferroni within each hypothesis |

---

## 5. Architecture A–D

### A. Document processing
- **GROBID** (Docker, `lfoppiano/grobid`) is the primary PDF → TEI XML parser, providing sections, headers, references, figure/table captions and citation anchors. It is the established tool for scientific PDFs.
- **PyMuPDF** is the fallback for fast text extraction, page geometry and PDFs where GROBID fails. Also use it to compare the two parsers on 20 hand-checked papers (section-boundary accuracy), which gives a small intrinsic evaluation for the thesis.
- QASPER already ships parsed full text, so PDF parsing is **not on the critical path** for H1–H3. That makes it the demo/ingestion path, not an experimental variable.
- **Section-aware chunking:** chunks never cross section boundaries, and each is prefixed with `"{title} | {section}"`. Default unit is the paragraph, split if over 256 tokens with a 32-token overlap. Tables are serialised as caption + row-major text.

### B. Classical NLP
Choose these components for their usefulness to the system, not to fill out a syllabus:

| Component | Where it does real work | Evaluation |
|---|---|---|
| Tokenisation, TF-IDF, BM25 | Lexical retrieval (H1 B1/B2) | H1 |
| LSA (SVD) | Classical dense baseline B3 | H1 |
| Logistic Regression | Claim-sentence detector (is this sentence a finding/claim?), using SciFact rationales as positives; also a cheap first-stage NLI baseline | P/R/F1 on SciFact dev |
| EM topic model (LDA / PLSA) | Corpus browsing and topic facets in the UI; candidate H3 edge type ("same topic") | Coherence (NPMI); H3 ablation |
| HMM POS tagger | Features for entity/claim extraction (noun-phrase chunks for method/dataset names) | Tagging accuracy on the NLTK Penn Treebank sample, versus a spaCy tagger |
| CFG + Earley | Parsing structured comparison queries ("compare X and Y on Z") into a query frame for H3 | Parse coverage on the H3 question set |

The honest framing for the thesis: HMM POS tagging and Earley parsing are **supporting modules with their own intrinsic evaluations**. Neither one is expected to move H1–H3 metrics.

### C. Scientific semantic representation
- **SciBERT** (`allenai/scibert_scivocab_uncased`): not trained for retrieval, so mean-pooled it should underperform. Keep it as a control showing that domain pretraining ≠ retrieval ability. Use it for the claim classifier / NLI fine-tune.
- **SPECTER2** (`allenai/specter2_base` + adapters `allenai/specter2` for documents and `allenai/specter2_adhoc_query` for queries; needs the `adapters` package). Use the [CLS] embedding.
- **General-purpose baseline:** `intfloat/e5-base-v2` (needs `query:` / `passage:` prefixes) or `BAAI/bge-base-en-v1.5`. Pick one, state why, and keep it fixed.
- **FAISS** `IndexFlatIP` on L2-normalised vectors. The corpora are small (QASPER dev ≈ 10–15k paragraphs), so exact search is affordable and approximate indexing would only add noise. NumPy is the fallback.

Compute: all encoders above are base-size (~110M params). Encoding QASPER dev/test on an Apple-silicon Mac (MPS) or CPU takes minutes, so no GPU is required for H1.

### D. Retrieval
- Lexical: BM25 (Okapi, k1=1.5, b=0.75 tuned on dev) with Lucene-style non-negative IDF.
- Dense: cosine on normalised embeddings.
- **Reciprocal Rank Fusion**: `score(d) = Σ_r w_r / (k + rank_r(d))`, k=60 by default (Cormack et al. 2009), fusing the top-100 from each retriever.
- Also a normalised linear-score fusion as a second hybrid, so the H1 claim is about *hybridisation* rather than only RRF.

---

## 6. Evaluation protocol

1. Every run writes a config + per-query metrics file (JSONL) + aggregate JSON under `results/`, keyed by git commit.
2. Fixed seeds; dense encoders run in eval mode, so there is no stochasticity. LLM generation uses temperature 0, and the prompt text is stored with the results.
3. Significance tests are on per-query paired scores. Every headline number is reported as mean [95% CI].
4. Test splits are touched once at the end of each hypothesis, and that run is logged.

## 7. Milestones (suggested)

| Weeks | Deliverable |
|---|---|
| 1–2 | QASPER loader, BM25/TF-IDF/LSA, metrics, stats ✔ (scaffolded) |
| 3–4 | Dense encoders (e5, SciBERT, SPECTER2), RRF; H1 dev results |
| 5 | H1 SciFact replication; H1 test run; write-up of H1 |
| 6–8 | Verifier (SciFact-trained), claim splitter, RAG baseline, H2 dev |
| 9 | H2 human evaluation; H2 test |
| 10–12 | Entity extraction, knowledge graph, H3 question set (written before the system), H3 experiments |
| 13–14 | GROBID ingestion + UI demo; thesis writing |

## 8. Main risks

- **H3 test data** is the weakest part of the design (§2, H3). Start the constructed question set early.
- **Scope:** nine capabilities is a lot for an individual project. If time runs short, cut the UI and the "research gap" feature, never the evaluation.
- **LLM dependence:** the generator is a component held fixed across conditions. The contributions are the retrieval, verification and graph layers, each evaluated on its own.
