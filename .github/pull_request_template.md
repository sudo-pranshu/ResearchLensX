## What and why
<!-- Closes #  -->

## Checks
- [ ] `python -m pytest -q` passes locally
- [ ] `python scripts/check_docs.py` passes
- [ ] New behaviour has an offline CPU test
- [ ] No data, PDFs, checkpoints or secrets added
- [ ] Results changes: config committed, run from a clean commit, register row added, dev/test split stated
- [ ] `contracts.py` untouched, or the change is additive with a version bump and changelog entry
- [ ] Claims in docs match `results/` (no invented numbers; negative results reported)

## Notes for review
