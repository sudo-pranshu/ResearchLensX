# Changelog

Format: date · type (added / changed / interface / fixed / result) · description · owner.

## 2026-11-05 (Day 1)
- added · Initial scaffold: contracts v1.0, `Pipeline`, mocks, fixtures, retrieval (BM25/TF-IDF/LSA/RRF/linear), evaluation (metrics, clustered stats), QASPER loader, chunker, demo CLI, CI · Pranshu
- interface · `contracts.SCHEMA_VERSION = "1.0"`: `Document`, `Chunk`, `RetrievedPassage`, `Claim`, `EvidenceJudgement`, `VerificationResult`, `AnswerSentence`, `Answer`, `KGNode`, `KGEdge`, `KnowledgeGraph` + 5 protocols · Pranshu
- changed · Research design v0.1 → v1.0 (H3 exploratory; QASPER open-corpus dropped as primary; cluster bootstrap; pre-registered hybrid; H2 restricted to non-boolean answers) · Pranshu
- fixed · BM25 sparse-matrix construction (CSR arrays were passed to the CSC constructor); chunker dropped the abstract for documents without sections · Pranshu

## 2026-11-05 (Day 1, experimental sprint)
- added · Dense retrieval: persistent embedding cache (key includes HF revision + corpus digest), FAISS IndexFlatIP, length-sorted batching, query priming; SPECTER2 (proximity + adhoc_query adapters), e5-base-v2, SciBERT-mean encoders; `retrieval/registry.py`; `evaluation/reporting.py` (JSON + JSONL + CSV) · Pranshu
- added · SciFact loader + download script + audit (`data/scifact.py`), SciFact claim→abstract retrieval experiment · Pranshu (SciFact loader ownership → Pawan)
- added · Real NLI verifier (`verification/nli.py`, label map read from model config), TF-IDF+LR baseline, classification metrics with claim-clustered CIs, SciFact verifier experiment · Pranshu (ownership → Pawan)
- added · H2 pipeline: `LLMRAGGenerator` (open-weight LLM, cached generations), claim extraction, `VerifiedReviser` (prune / reassign / drop contradicted / drop unsupported / verify-only ablation), ALCE-style citation P/R with an independent judge, QASPER answer-F1 · Pranshu (ownership → Pawan)
- added · `build_pipeline()` with explicit fallbacks + `RevisingPipeline`; contracts unchanged (additive only) · Pranshu
- added · Section-agnostic window ablation (`--unit window`), H1 ablation runner · Pranshu
- fixed · Claim extraction skipped sentences of <3 words, so short claims went unverified; now only abstentions/wordless fragments are skipped · Pranshu
- note · The mock verifier is no longer used by any experiment; it remains only for offline UI development, flagged `is_mock=True` · Pranshu
