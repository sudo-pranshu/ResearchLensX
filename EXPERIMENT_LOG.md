# Experiment log

The authoritative register (config, git hash, models, data, headline numbers, status) is **PROJECT_BIBLE.md §G**. This file is the narrative: what was tried, what happened, what was decided. Raw outputs are in `results/<EXP-ID>/`.

| ID | Question | Outcome (short) | Decision |
|---|---|---|---|
| EXP-H1-001/002/002b | Classical retrievers; within-paper vs open corpus | QASPER open-corpus retrieval is ill-posed (questions are written about one paper) | Evaluate within-paper; use SciFact for open corpus (ADR-005) |
| EXP-H1-003 | Dense models and hybrids (dev) | e5 best single (R@5 .500); hybrid not better than e5 | D* = e5 (ADR-009) |
| EXP-H1-004 | SciFact retrieval (dev) | e5 .870 nDCG@10 vs RRF .855 (n.s.) | Same conclusion on a second corpus |
| EXP-H2-001 | NLI verifier on SciFact (dev) | macro-F1 .518 (LR) → .596-.681 (DeBERTa); judge .565 | Passage-level premises (ADR-010) |
| EXP-H2-002 | RAG vs verified RAG pilot | Inconclusive: generator emitted 0.15 citations/answer, the gain came from the verifier adding citations | Fix prompt (EXP-H2-004) |
| EXP-H1-006 | **H1 final test (once)** | R@5 e5 .501, RRF .486: **H1 not supported**; hybrid beats lexical baselines | Report as a negative result |
| EXP-H1-007 | Cross-encoder rerank vs none (dev) | R@5 +.075 to +.120 for every first stage; best: rerank(RRF) .580 | Rerank is optional in the app; test run still pending |
| EXP-QA-001 | Can lexical coverage detect unanswerable questions? | AUROC .636; weak | Default abstention threshold 0.2; rely on NLI status |
| EXP-H2-003 | Calibrate the NLI confidence | ECE .183 → .067 out-of-fold, T = 2.24 | Calibrated confidence in the app |
| EXP-KG-001 | Graph extraction audit (qualitative) | Methods 76%, datasets 100% precise (low recall); raw cross-paper NLI edges 0/9 valid | Gate NLI edges; label as candidates |
| EXP-H2-004 | RAG vs verified RAG, one-shot citation prompt (dev pilot) | Baseline now cites (0.99/answer). Verified: citation precision .443 → .785 (CI excludes 0) but Answer-F1 −.060 (fails the 2-pt guard; 23% abstentions). Reassignment drives the gain; the no-drop variant keeps Answer-F1 and still gains +.259 | H2 partially supported on dev only. Pre-register the no-drop variant before any test run; needs a larger generator on Colab to be convincing |

## Not run
- H2 with a 7-8B generator and the QASPER test split (needs Colab).
- H1 ablations (`scripts/run_h1_ablations.sh`: section prefix, windows, RRF k).
- Rerank on the test split.
- Independent manual labelling of research gaps (`docs/review/gap_review.csv`).
- Evaluation of "combined evidence" verification on real data.
