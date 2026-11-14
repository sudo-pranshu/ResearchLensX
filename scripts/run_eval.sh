#!/usr/bin/env bash
# One-command reproducible evaluation suite.
#   ./scripts/run_eval.sh            # fast: tests + CPU-only experiments (a few minutes; needs QASPER dev, see scripts/download_qasper.sh)
#   ./scripts/run_eval.sh --full     # also the model-based dev runs (needs the dense extra; ~1 h on an 8 GB laptop)
# Results go to results/<EXP-ID>/ ; the register is docs/research/PROJECT_BIBLE.md section G. The test split is NEVER run here.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
echo "== 1/4 unit and integration tests"; $PY -m pytest -q
[ -f data/raw/qasper/qasper-dev-v0.3.json ] || { echo "QASPER dev missing: run scripts/download_qasper.sh"; exit 1; }
echo "== 2/4 classical retrieval baselines (EXP-H1-001)"
$PY experiments/h1_retrieval.py --config configs/h1_classical_dev.yaml --run-name EXP-H1-001_rerun --out results/_rerun
echo "== 3/4 abstention signal (EXP-QA-001)"
$PY experiments/qa_abstention.py --out results/_rerun/EXP-QA-001
echo "== 4/4 calibration from stored verifier probabilities (EXP-H2-003)"
$PY experiments/h2_calibration.py --out results/_rerun/EXP-H2-003
if [ "${1:-}" = "--full" ]; then
  echo "== dense retrieval + rerank dev (EXP-H1-003, EXP-H1-007)"
  $PY experiments/h1_retrieval.py --config configs/h1_dense_dev.yaml --run-name EXP-H1-003_rerun --out results/_rerun
  $PY experiments/h1_retrieval.py --config configs/h1_rerank_dev.yaml --run-name EXP-H1-007_rerun --out results/_rerun
  [ -d data/raw/scifact ] || bash scripts/download_scifact.sh
  $PY experiments/h2_scifact_verifier.py --config configs/h2_scifact_verifier.yaml --run-name EXP-H2-001_rerun --out results/_rerun
fi
echo "done. Compare results/_rerun/* with the committed results/ (numbers should match up to library versions)."
