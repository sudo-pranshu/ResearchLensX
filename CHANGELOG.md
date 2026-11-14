# Changelog

Format: date · type (added / changed / interface / fixed / result) · description.

## 2026-11-05 (Day 1)
- fixed · after an external methodology review: gap text no longer claims other papers fail to address an item ("not checked"), review citations use the real section, related-work and citation-bearing sentences are no longer attributed to the paper, complete-linkage clusters with stored similarity, one sentence per gap, round-robin ranking across papers, missing-comparison uses unthresholded mentions, NLI edges gated on a shared extracted entity; 7 regression tests
- added · `analysis/gaps.py` (author-stated / collection-pattern / absence-in-collection gaps, evidence + template RQ/methodology, confidence capped at medium, no novelty claims), `analysis/review.py` (cited literature review, Markdown/HTML/PDF), app pages Research gaps and Literature review, `scripts/gap_review_sheet.py` for manual labelling; shared strict limitation rule
- added · `analysis/compare.py` comparative methodology table, shared/unique entities, Markdown export; app page Compare papers (same question asked of each paper)
- interface · schema 1.3 (additive): graph node types author/finding/limitation, edge types authored_by/uses_method/evaluates_on/uses_metric/reports_finding/has_limitation/similar_problem
- added · knowledge graph (`knowledge_graph/extract.py`, `builder.py`): provenance on every element, reference-matched `cites`, TF-IDF `similar_problem`, shared-entity edges, gated NLI supports/contradicts (candidates, mock never creates edges); app page with pyvis view and node inspector; EXP-KG-001 audit
- added · Streamlit app (`app/`): Library (upload, duplicates/failures reported), Ask (cited extractive answers, page-level references, abstention, claim-status pills), Search (retriever comparison), Verify a claim (NLI + explanation), Evaluation (committed results); `workspace.py` service layer with visible fallbacks and MOCK banner; `Document.abstract_page`; thread-safe `CorpusStore`; 29 new tests
- fixed · found by running the app: SQLite cross-thread error under Streamlit reruns; extractive generator preferred figure/table captions
- interface · schema 1.2 (additive): `VerificationResult.support` (direct/combined) and `.explanation`
- added · `qa/engine.py` MultiDocQA (cross-paper balancing, references with page/section, coverage abstention, claim status: supported/inferred/conflicting/contradicted/unsupported), `NLIVerifier(combine=, temperature=)` + deterministic explanations, `evaluation/calibration.py`; EXP-H1-007, EXP-QA-001, EXP-H2-003
- added · cross-encoder reranker (`retrieval/rerank.py`, spec `rerank:<spec>`), redundancy suppression and section filters (`retrieval/postprocess.py`), `ChunkIndex.search(where=, dedup_threshold=)`, `Pipeline.ask(where=)`, `build_default_pipeline(rerank=, dedup_threshold=)`; 8 offline tests
- added · `store/corpus.py` SQLite corpus store (dedupe by doc_id, batch ingest with per-file failure report, metadata filters on year/title/ids, schema-version guard) + 7 tests
- interface · schema 1.1 (additive): `Section.pages`, `Chunk.page` for page-level citations; `chunk_document` propagates the start page
- added · `PyMuPDFIngestor` (title, abstract, numbered/known headings, margin header/footer removal, de-hyphenation, cross-page paragraph merge, references, arXiv id/year) with 14 tests on self-generated PDFs; real-PDF fixes: rotated arXiv stamp ignored (read for id only), ligature normalisation, hanging-indent reference splitting, lettered appendix headings
- changed · project is solo; team docs removed, freeze moved to 2026-11-27
- added · Initial scaffold: contracts v1.0, `Pipeline`, mocks, fixtures, retrieval (BM25/TF-IDF/LSA/RRF/linear), evaluation (metrics, clustered stats), QASPER loader, chunker, demo CLI, CI · Pranshu
- interface · `contracts.SCHEMA_VERSION = "1.0"`: `Document`, `Chunk`, `RetrievedPassage`, `Claim`, `EvidenceJudgement`, `VerificationResult`, `AnswerSentence`, `Answer`, `KGNode`, `KGEdge`, `KnowledgeGraph` + 5 protocols · Pranshu
- changed · Research design v0.1 → v1.0 (H3 exploratory; QASPER open-corpus dropped as primary; cluster bootstrap; pre-registered hybrid; H2 restricted to non-boolean answers) · Pranshu
- fixed · BM25 sparse-matrix construction (CSR arrays were passed to the CSC constructor); chunker dropped the abstract for documents without sections · Pranshu

## 2026-11-05 (Day 1, experimental sprint)
- added · Dense retrieval: persistent embedding cache (key includes HF revision + corpus digest), FAISS IndexFlatIP, length-sorted batching, query priming; SPECTER2 (proximity + adhoc_query adapters), e5-base-v2, SciBERT-mean encoders; `retrieval/registry.py`; `evaluation/reporting.py` (JSON + JSONL + CSV) · Pranshu
- added · SciFact loader + download script + audit (`data/scifact.py`), SciFact claim→abstract retrieval experiment · Pranshu
- added · Real NLI verifier (`verification/nli.py`, label map read from model config), TF-IDF+LR baseline, classification metrics with claim-clustered CIs, SciFact verifier experiment · Pranshu
- added · H2 pipeline: `LLMRAGGenerator` (open-weight LLM, cached generations), claim extraction, `VerifiedReviser` (prune / reassign / drop contradicted / drop unsupported / verify-only ablation), ALCE-style citation P/R with an independent judge, QASPER answer-F1 · Pranshu
- added · `build_pipeline()` with explicit fallbacks + `RevisingPipeline`; contracts unchanged (additive only) · Pranshu
- added · Section-agnostic window ablation (`--unit window`), H1 ablation runner · Pranshu
- fixed · Claim extraction skipped sentences of <3 words, so short claims went unverified; now only abstentions/wordless fragments are skipped · Pranshu
- note · The mock verifier is no longer used by any experiment; it remains only for offline UI development, flagged `is_mock=True` · Pranshu
