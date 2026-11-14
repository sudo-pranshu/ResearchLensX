# CLAUDE.md: ResearchLens X master context

Read this first in every Claude session, then `docs/development/PROJECT_STATE.md` and `docs/production/PRODUCTION_ROADMAP.md`.

## Project
ResearchLens X is maintained by a single developer, Pranshu Kumar. The research specification and experiment register are in `docs/research/PROJECT_BIBLE.md`. Primary hypotheses: H1 hybrid retrieval, H2 claim-level verification. H3 is exploratory.

## Hard rules for Claude
1. **Authorship:** everything is in Pranshu Kumar's name. No Claude/Anthropic attribution, co-author trailers or "generated with" lines in commits, PRs, issues or docs.
2. **`contracts.py` changes** are allowed only when additive/versioned, with a CHANGELOG entry and tests (bump `SCHEMA_VERSION` for breaking changes).
3. **Never fabricate results.** Metrics come only from runs with a committed config. Don't write numbers into docs that aren't in `results/`. Mock outputs must keep `is_mock=True`.
4. **Dev for tuning, test once.** Don't run test-split evaluations unless the issue says it is the final test run.
5. **No paid APIs, no secrets in code.** Credentials go in `.env` (gitignored). Models are open-weight HF checkpoints pinned by id (+ revision when final).
6. **Tests:** every new component gets a unit test that runs offline on CPU. Tests that need PyMuPDF or Streamlit skip when those packages are missing; no test downloads a model.
7. **Don't duplicate work:** search the repo before adding a utility (`tokenize`, metrics, stats and sentence splitting already exist).
8. Before finishing: `python -m pytest -q` passes; update CHANGELOG for user-visible or interface changes.

## Map
- Contracts and data flow: `src/researchlens/contracts.py`, `ARCHITECTURE.md`
- Facade the app uses: `src/researchlens/pipeline.py` (`build_default_pipeline`, `Pipeline.ask/compare`)
- Existing utilities: `retrieval/lexical.tokenize`, `generation/extractive.split_sentences`, `evaluation/metrics`, `evaluation/stats.compare_many(groups=...)`
- Experiment configs: `configs/*.yaml` → `experiments/*.py --config ...`
- Commands: `python -m pytest -q` · `python scripts/demo.py "question"` · `python experiments/h1_retrieval.py --config configs/h1_classical_dev.yaml`
