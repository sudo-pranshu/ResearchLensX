# CLAUDE.md: ResearchLens X master context

Read this first in every Claude session, then `IMPLEMENTATION_ROADMAP.md` and `PROJECT_STATE.md`.

## Project
Final-year NLP project by a single developer, Pranshu Kumar (14-day plan, freeze **2026-11-27**). The single source of truth is `PROJECT_BIBLE.md`. Primary hypotheses: H1 hybrid retrieval, H2 claim-level verification. H3 is exploratory.

## Hard rules for Claude
1. **Authorship:** everything is in Pranshu Kumar's name. No Claude/Anthropic attribution, co-author trailers or "generated with" lines in commits, PRs, issues or docs.
2. **`contracts.py` changes** are allowed only when additive/versioned, with a CHANGELOG entry and tests (bump `SCHEMA_VERSION` for breaking changes).
3. **Never fabricate results.** Metrics come only from runs with a committed config. Don't write numbers into docs that aren't in `results/`. Mock outputs must keep `is_mock=True`.
4. **Dev for tuning, test once.** Don't run test-split evaluations unless the issue says it is the final test run.
5. **No paid APIs, no secrets in code.** Credentials go in `.env` (gitignored). Models are open-weight HF checkpoints pinned by id (+ revision when final).
6. **Tests:** every new component gets a unit test that runs offline on CPU. Model-dependent tests are skipped unless `RLX_RUN_MODEL_TESTS=1`.
7. **Don't duplicate work:** search the repo before adding a utility (`tokenize`, metrics, stats and sentence splitting already exist).
8. Before finishing: `python -m pytest -q` passes; update CHANGELOG for user-visible or interface changes.

## Map
- Contracts and data flow: `src/researchlens/contracts.py`, `ARCHITECTURE.md`
- Facade the app uses: `src/researchlens/pipeline.py` (`build_default_pipeline`, `Pipeline.ask/compare`)
- Existing utilities: `retrieval/lexical.tokenize`, `generation/extractive.split_sentences`, `evaluation/metrics`, `evaluation/stats.compare_many(groups=...)`
- Experiment configs: `configs/*.yaml` → `experiments/*.py --config ...`
- Commands: `python -m pytest -q` · `python scripts/demo.py "question"` · `python experiments/h1_retrieval.py --config configs/h1_classical_dev.yaml`
