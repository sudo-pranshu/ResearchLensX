# Changelog

Format: date · type (added / changed / interface / fixed / result) · description · owner.

## 2026-11-05 (Day 1)
- added · Initial scaffold: contracts v1.0, `Pipeline`, mocks, fixtures, retrieval (BM25/TF-IDF/LSA/RRF/linear), evaluation (metrics, clustered stats), QASPER loader, chunker, demo CLI, CI · Pranshu
- interface · `contracts.SCHEMA_VERSION = "1.0"`: `Document`, `Chunk`, `RetrievedPassage`, `Claim`, `EvidenceJudgement`, `VerificationResult`, `AnswerSentence`, `Answer`, `KGNode`, `KGEdge`, `KnowledgeGraph` + 5 protocols · Pranshu
- changed · Research design v0.1 → v1.0 (H3 exploratory; QASPER open-corpus dropped as primary; cluster bootstrap; pre-registered hybrid; H2 restricted to non-boolean answers) · Pranshu
- fixed · BM25 sparse-matrix construction (CSR arrays were passed to the CSC constructor); chunker dropped the abstract for documents without sections · Pranshu
