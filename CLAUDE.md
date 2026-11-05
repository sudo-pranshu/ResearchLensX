# CLAUDE.md: ResearchLens X master context

Read this first in every Claude session. Then read your member workflow in `docs/development/<NAME>_WORKFLOW.md`.

## Project
14-day final-year NLP project (freeze **2026-11-18**). The single source of truth is `PROJECT_BIBLE.md`. Primary hypotheses: H1 hybrid retrieval, H2 claim-level verification. H3 is exploratory.

## Hard rules for Claude
1. **Stay in the owner's paths.** Ask which member you're working for and which issue. Edit only that member's paths (TEAM_RESPONSIBILITIES.md ownership matrix).
2. **Never change `src/researchlens/contracts.py`** unless Pranshu explicitly asks for an `interface-change`. If a contract seems insufficient, write the proposal in the issue instead.
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
