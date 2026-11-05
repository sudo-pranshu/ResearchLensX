# Contributing

## Setup (everyone, CPU only)
```bash
git clone https://github.com/sudo-pranshu/ResearchLensX.git && cd ResearchLensX
python3 -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q                                          # must be green before you start
```
Optional extras: `pip install -r requirements-dense.txt` (models, ~2 GB), `pip install -r requirements-app.txt` (PyMuPDF, Streamlit, NetworkX, Plotly).

## Branches
- `main`: stable, integrated, tagged at milestones. **No direct commits.**
- `develop`: integration branch. All feature PRs target `develop`.
- `feature/pranshu-<topic>`, `feature/pawan-<topic>`, `feature/gaurav-<topic>`: one branch per issue, short-lived (≤2 days).

```bash
git switch develop && git pull
git switch -c feature/gaurav-pymupdf-ingestor
# ... work, commit ...
git push -u origin feature/gaurav-pymupdf-ingestor
gh pr create --base develop --fill        # or use the GitHub web UI
```

## Commits
Conventional commits, small and descriptive: `feat(ingestion): detect numbered section headings`, `fix(retrieval): …`, `test(verification): …`, `docs: …`, `exp(h1): dense dev run`. Reference the issue: `Closes #12`.

## Pull requests
1. CI green (pytest).
2. Tests cover the new behaviour; every component has at least one working validation test.
3. Reviewer: the owner of each touched path (TEAM_RESPONSIBILITIES.md). Pranshu reviews anything shared.
4. Interface change → `interface-change` label, update `data/sample/` fixtures + CHANGELOG, Pranshu approves.
5. Results PR → include config + summary + register row.

## Never commit
Secrets/API keys/tokens (use `.env`, see `.env.example`), raw datasets (`data/raw/`), model checkpoints or indexes (`checkpoints/`, `artifacts/`), virtual environments, PDFs, notebooks with large outputs (clear outputs first).

## Code style
Python ≥3.10, type hints on public functions, docstrings stating inputs/outputs, no hidden global state, deterministic by default. Keep functions testable without network or GPU. Gate any model download behind an explicit class/flag.

## Dependencies
Add a dependency only in the right `requirements*.txt` **and** `pyproject.toml`, in the same PR, with a lower-bound version. Core requirements must stay CPU-only and small.
