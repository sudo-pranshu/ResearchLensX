# Pranshu: Retrieval, H1, Integration

**You may edit:** `src/researchlens/{contracts.py,pipeline.py,mocks.py}`, `retrieval/**`, `evaluation/{metrics,stats}.py`, `data/qasper.py`, `experiments/h1_*`, `configs/h1_*`, `notebooks/**`, root docs, CI, dependency files.
**Do not edit:** Pawan's and Gaurav's paths (TEAM_RESPONSIBILITIES.md). Request changes through their issues and PRs.
**Validate:** `python -m pytest -q` locally + CI; every H1 run from a clean commit with a config.

## Day 1–2 (milestone "Day 2: setup & baselines")
| # | Task | Done when |
|---|---|---|
| P1 | Merge the scaffold; invite Pawan and Gaurav; create `develop` | CI green on `develop`; both collaborators accepted |
| P2 | Commit classical H1 dev runs with cluster stats (EXP-H1-001/002) | PR merged with config + summary + register rows |
| P3 | Colab: run `notebooks/colab_h1_dense.ipynb` on QASPER dev for SPECTER2, e5, SciBERT; cache embeddings to Drive | `results/h1_dense_dev/*` committed; D* chosen by R@5 and recorded as ADR |
| P4 | SciFact retrieval (claims → abstracts) in `experiments/h1_scifact.py`, using Pawan's `data/scifact.py` loader (or a local reader until it lands) | dev nDCG@10 for BM25 / D* / RRF committed |

## Day 3–6
- P5 `retrieval/faiss_index.py`: build/save/load a dense index for app corpora (`artifacts/indexes/<corpus>/{emb.npy,meta.json}`), plus a CPU query encoder. Test with LSA so CI stays offline.
- P6 `build_pipeline(config)`: hybrid retriever when an index exists, BM25 fallback otherwise. Load Pawan's verifier and generator when available, extractive fallback otherwise.
- P7 Run Pawan's H2 generation script on Colab (Day 6–7) and commit the JSONL.

## Day 7–8
- P8 Write `configs/h1_final_test.yaml` → run once on QASPER test; SciFact dev final. Register rows.
- (stretch) cross-encoder reranking as a separate factor.

## Claude session rules
Start each session by reading `CLAUDE.md` and this file. Tell Claude which issue you're on. Claude must not edit other members' paths or `contracts.py` without an explicit `interface-change` decision from you.
