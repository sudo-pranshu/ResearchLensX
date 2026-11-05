# Research Design v0.1: Critical Review → v1.0

Date: 2026-11-05 (Day 1). Reviewer: Pranshu (with Claude). Status: **accepted**, and v1.0 is now the specification in [PROJECT_BIBLE.md §B](../PROJECT_BIBLE.md#b-research-specification).
v0.1 is kept unchanged for the record in [RESEARCH_DESIGN.md](RESEARCH_DESIGN.md).

The review is grounded in two pieces of evidence: (a) the dataset audit on Day 1, and (b) H1 classical baselines on QASPER dev, which were run before the review (uncommitted, question-level CIs; see EXP-H1-001/002 for the committed reruns).

## Findings

| # | Area | Finding | Evidence | Change |
|---|---|---|---|---|
| R1 | Scope | v0.1 assumes ~14 weeks; we have **14 days** and 3 people | Brief | H3 becomes **exploratory** (case studies, no hypothesis test). Dropped from scope: cross-encoder reranking (stretch goal), DocLayNet, SciRepEval, PubMedQA, SciDocs, ScholarQABench, HMM POS, Earley parser (all optional extras) |
| R2 | H1 setting | QASPER **open-corpus** retrieval is ill-posed. Questions such as "What dataset do they use?" were written about *one* paper; in a 15k-paragraph pool most of the difficulty is guessing the paper | Bare question: only 25% of BM25's top-10 comes from the right paper. With the title prepended: 99%, and BM25 nDCG@10 goes 0.142 → 0.183 | Open-corpus QASPER is removed as a primary setting. H1 primary = QASPER **within-paper**. The open-corpus test moves to **SciFact** (claims → 5,183 abstracts), where queries really are corpus-level |
| R3 | H1 hybrid choice | v0.1 lists several hybrids (RRF, weighted RRF, linear). Picking the best one on test = garden of forking paths | Dev: linear(BM25+LSA) +2.6 pt R@5, RRF(BM25+LSA) +1.5 pt (n.s.), RRF(BM25+TF-IDF) −1.6 pt | **Pre-register one primary hybrid:** RRF(k=60, depth=100) of BM25 + the best dense retriever *selected on dev by R@5*. Others are reported as secondary, with no confirmatory claim |
| R4 | H1 "semantic" | LSA is not the "semantic" retriever the hypothesis means; classical-only hybrids do not test H1 | Above | H1 confirmatory comparison needs a **neural** dense retriever (SPECTER2 / e5 / SciBERT). LSA stays as a classical baseline |
| R5 | Statistics | QASPER questions are clustered by paper (dev: 923 Q / 281 papers ≈ 3.3 Q/paper); the question-level bootstrap in v0.1 understates variance | Design property | **Paper-level cluster bootstrap and cluster sign-flip randomisation** implemented (`evaluation/stats.py`, tested). Wilcoxon is reported only as a question-level reference |
| R6 | Gold labels | 129/2,808 dev evidence strings are **section headings**, not paragraphs; 6 annotations lose all evidence | Loader audit (all 129 classified) | Headings are excluded from gold (mapping a heading to a whole section would inflate gold). Counts are logged in every run config |
| R7 | H2 answer types | Many QASPER answers are yes/no or one-span, which leaves claim verification very little to do | QASPER answer-type mix | H2 primary set = answerable test questions whose majority answer type is **extractive or abstractive**. Boolean answers are reported separately |
| R8 | H2 generator | v0.1 says "LLM" without saying which. A paid API would break reproducibility and the resource rules | Team constraints | **Open-weight instruct model** (7–8B class, chosen on Day 3, pinned by HF revision) run on Colab by Pranshu. *All generations cached to JSONL and committed*, so verification and evaluation re-run on CPU. No paid API anywhere in the evaluated pipeline |
| R9 | H2 triviality | Verifying **extractive** answers is trivially "supported" (sentences are copied from their citations) | Day-1 demo: the mock verifier gives every extractive sentence SUPPORTS 1.00 | H2 compares *generative* RAG vs generative verified RAG. The extractive generator is only the CPU app fallback |
| R10 | H2 judge | v0.1 requires judge ≠ verifier but never checks the judge is any good | — | The judge NLI model is validated against human labels on the human-eval sample (report judge–human agreement). If agreement is poor, human labels become the primary H2 evidence |
| R11 | Human eval | 100 Q × 2 systems + κ is too much annotation for 14 days | Time | **50 Q × 2 systems**, blinded, order randomised. A 15-question overlap is annotated by all three members (Fleiss' κ); the remaining 35 are split. Annotators are system builders, which is a validity threat mitigated only by blinding (stated in limitations) |
| R12 | SciFact | Test labels are hidden (leaderboard) | Dataset docs | Verifier model selection uses a held-out 20% of SciFact **train**; the final verifier is reported on SciFact **dev** once. Licence CC BY-NC 2.0: fine for coursework, noted in README |
| R13 | SPECTER2 exposure | SPECTER2 was trained on S2 citation data, which may include QASPER/SciFact papers (document exposure, not label leakage) | Model card | Stated as a validity threat. SciDocs/SciRepEval comparisons are dropped entirely, so no unfair benchmark is reported. e5 (general-purpose) is the control |
| R14 | Novelty | v0.1 risks implying novel methods. Hybrid retrieval with RRF, NLI claim verification and ALCE-style citation evaluation are all established | Literature | Contributions are re-stated as: (1) a **controlled, statistically sound evaluation** of these techniques for scientific QA on QASPER/SciFact, and (2) an **integrated, reproducible, CPU-runnable system**. No claim of new algorithms |
| R15 | H3 data | No cross-paper benchmark exists in the chosen data; the 120-question set from v0.1 is not feasible in 14 days | Time | H3 is exploratory: about 10 documented comparison case studies via `Pipeline.compare` + the KG, presented qualitatively, with **no significance claims** |
| R16 | Syllabus NLP | Classical components must not be passed off as contributions | — | Kept and labelled as "supporting": TF-IDF/BM25/LSA (H1 baselines), Logistic Regression claim classifier (Pawan), optional LDA topic facet (Gaurav). HMM/Earley are optional only if time allows |

## Preliminary classical results (dev, uncommitted run, question-level CIs, superseded by EXP-H1-001)

Within-paper, 923 questions: BM25 R@5 0.435; TF-IDF 0.391; LSA 0.427; RRF(BM25+LSA) 0.450; linear(BM25+LSA) 0.461.
These are development numbers and **must not appear in the report as results**. The committed reruns with cluster statistics are in the experiment register.
