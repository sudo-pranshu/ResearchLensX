# Contributing to ResearchLens X

Thanks for your interest. ResearchLens X is maintained by Pranshu Kumar; contributions that keep the project honest, tested and reproducible are welcome.

## Principles that every change must respect
1. **No fabricated results.** Metrics in docs must come from a run with a committed config in `results/` and a row in the [experiment register](docs/research/PROJECT_BIBLE.md). Never edit numbers to look better; report negative results.
2. **Mocks are labelled.** Any placeholder component keeps `is_mock=True` and the UI shows the MOCK banner. Mocks never take part in evaluations.
3. **Traceability.** Answers and analyses point to a paper, section, page and quote. Do not add features that generate unsupported claims.
4. **Scope claims honestly.** Research gaps are collection-scoped; never describe a result as novel in the wider literature.
5. **Dev for tuning, test once.** Test-split runs are done once per final configuration and registered.
6. **No data in git.** No PDFs, datasets, model checkpoints, secrets or personal data.

## Setup
```bash
git clone https://github.com/sudo-pranshu/ResearchLensX.git && cd ResearchLensX
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dense.txt -r requirements-app.txt
pip install -e .
python -m pytest -q                # 133 tests, offline, CPU
python scripts/check_docs.py       # link and wording gate
```

## Workflow
- Branches: `main` holds stable releases, `develop` is integration, work happens on `feature/<topic>`, `fix/<topic>` or `docs/<topic>` branches opened as pull requests into `develop`. See [docs/development/RELEASE_PLAN.md](docs/development/RELEASE_PLAN.md).
- Commits: conventional style (`feat(retrieval): ...`, `fix(ingestion): ...`, `docs: ...`, `exp(h1): ...`), small and descriptive.
- Every new component needs a unit test that runs offline on CPU. Run the full suite before opening a pull request. CI runs the same suite and the docs gate.
- Changes to `src/researchlens/contracts.py` must be additive, bump `SCHEMA_VERSION`, update fixtures and add a changelog entry.
- Update [CHANGELOG.md](CHANGELOG.md) for user-visible changes and the relevant document under `docs/`.

## Reporting issues
Use the issue templates. For security issues follow [SECURITY.md](SECURITY.md). Scientific concerns (a wrong number, an overclaim, a flawed metric) are bugs: please open an issue with the evidence.

## Licence
By contributing you agree that your contributions are licensed under the project licence, [AGPL-3.0-or-later](LICENSE).
