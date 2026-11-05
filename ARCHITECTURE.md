# Architecture

Authoritative summary: [PROJECT_BIBLE.md §C](PROJECT_BIBLE.md#c-architecture-summary). This file holds the detail.

## 1. Component diagram

```mermaid
flowchart LR
  subgraph G[Gaurav: document intelligence & app]
    PDF[(PDF)] --> ING[Ingestor<br/>PyMuPDF / GROBID*]
    ING --> DOC[Document]
    DOC --> CH[chunk_document]
    KGB[GraphBuilder] --> KG[KnowledgeGraph]
    APP[Streamlit app]
  end
  subgraph P[Pranshu: retrieval & integration]
    CH --> CI[ChunkIndex]
    CI --> BM[BM25 / TF-IDF / LSA]
    CI --> DN[SPECTER2 / e5 + FAISS]
    BM & DN --> RRF[RRF fusion]
    PIPE[Pipeline facade]
  end
  subgraph W[Pawan: verification & generation]
    GEN[Generator<br/>extractive / LLM-RAG]
    VER[Verifier<br/>NLI]
  end
  QASPER[(QASPER JSON)] -. load_qasper_documents .-> DOC
  RRF -->|RetrievedPassage[]| GEN
  GEN -->|Answer| VER
  VER -->|VerificationResult| KGB
  DOC --> KGB
  APP <-->|ask / compare| PIPE
  PIPE --- CI & GEN & VER
  KG --> APP
```
\* optional service with an automatic fallback.

## 2. Contracts (`src/researchlens/contracts.py`, owner Pranshu)

| Contract | Producer | Consumer | Key fields |
|---|---|---|---|
| `Document` | Ingestor (Gaurav), `load_qasper_documents` (Pranshu) | chunker, GraphBuilder, app | `doc_id, title, authors, year, abstract, sections[Section], references[Reference], source` |
| `Chunk` | `chunk_document` (Gaurav) | `ChunkIndex` | `chunk_id = "{doc_id}::s{i}::p{j}::w{k}", doc_id, section, text, order` |
| `RetrievedPassage` | `PassageRetriever` (Pranshu) | Generator, Verifier, app | `chunk, score, rank (1-based), retriever` |
| `Claim` | Generator / claim extractor (Pawan) | Verifier | `claim_id, text, source` |
| `VerificationResult` | Verifier (Pawan) | Pipeline, GraphBuilder, app | `label ∈ {SUPPORTS, CONTRADICTS, INSUFFICIENT}, confidence, evidence[EvidenceJudgement], model, is_mock` |
| `Answer` | Generator (Pawan), annotated by Pipeline | app, H2 evaluation | `sentences[AnswerSentence(text, citations, verification)], passages, mode, generator, is_mock, timings_ms` |
| `KnowledgeGraph` | GraphBuilder (Gaurav) | app | `nodes[KGNode], edges[KGEdge]` |

Interfaces (typing `Protocol`, checked by `isinstance` in tests):
`Ingestor.ingest(pdf_path) -> Document` · `PassageRetriever.search(query, k, doc_ids) -> list[RetrievedPassage]` · `Generator.generate(question, passages) -> Answer` · `Verifier.verify(claim, passages) -> VerificationResult` · `GraphBuilder.build(documents, verifications) -> KnowledgeGraph`.

**Change rule:** a PR that touches `contracts.py` must carry the `interface-change` label, update the fixtures in `data/sample/` and CHANGELOG, and get Pranshu's approval. Breaking changes bump `SCHEMA_VERSION`.

## 3. Integration facade (`pipeline.py`)

```python
pipe = build_default_pipeline(documents, retriever="bm25", verifier=None)   # CPU, offline, always works
ans  = pipe.ask(question, k=5, doc_ids=None, verify=True)                   # -> Answer
cmp_ = pipe.compare(question, doc_ids, k=3)                                 # -> {doc_id: Answer}
```
The app never imports `retrieval/`, `verification/` or `generation/` directly, so each module can change behind the facade.

## 4. Degradation matrix

| Component | Preferred | Fallback | Who detects | UI behaviour |
|---|---|---|---|---|
| PDF parsing | GROBID (if `GROBID_URL` reachable) | PyMuPDF | Ingestor | badge "parsed with PyMuPDF" |
| Dense retrieval | SPECTER2 + FAISS index from `artifacts/` | BM25 only | Pipeline builder | badge "lexical only" |
| Generator | LLM (Colab, or local Ollama if configured) | ExtractiveGenerator | Pipeline builder | badge "extractive mode" |
| Verifier | NLI model (CPU OK) | none, or MockVerifier in dev only | Pipeline builder | MOCK badge when `is_mock`; "not verified" when absent |

**No hard-coded demo outputs.** Every value shown in the app is produced at run time by a component whose name and `is_mock` flag are displayed.

## 5. Directory layout

```
src/researchlens/
  contracts.py  pipeline.py  mocks.py           # Pranshu (shared, PR-gated)
  retrieval/    evaluation/metrics.py stats.py  data/qasper.py   # Pranshu
  verification/ generation/  evaluation/attribution.py  data/scifact.py   # Pawan
  ingestion/    processing/  knowledge_graph/   # Gaurav
app/            # Gaurav (Streamlit + optional FastAPI)
experiments/    # h1_* Pranshu, h2_* Pawan
configs/        # YAML experiment configs, one per run family
data/sample/    # committed synthetic fixtures (contract examples)
data/raw/       # NOT committed (download scripts in scripts/)
results/        # run outputs; only summary/config/comparisons committed
notebooks/      # Colab drivers (thin: clone repo, run scripts, save to Drive)
```

## 6. Compute placement

| Job | Where | Owner | Output, cached at |
|---|---|---|---|
| BM25/TF-IDF/LSA, all tests, app | any laptop (CPU) | all | — |
| SPECTER2/e5/SciBERT encoding of QASPER + SciFact | Colab GPU | Pranshu | `artifacts/indexes/*.npy` (Drive) |
| NLI zero-shot on SciFact dev (~300 claims) | CPU acceptable | Pawan | `results/h2/` |
| Optional NLI fine-tune on SciFact train | Colab GPU | Pawan's script, run by Pranshu | Drive checkpoint + model card |
| LLM generation for H2 | Colab GPU | Pawan's script, run by Pranshu | `results/h2/generations_*.jsonl` (committed) |
