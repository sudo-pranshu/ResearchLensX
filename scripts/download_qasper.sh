#!/usr/bin/env bash
# Download QASPER v0.3 (CC BY 4.0) from AllenAI's public S3 bucket. Small (tens of MB).
# Verify these URLs against https://allenai.org/data/qasper before first use.
set -euo pipefail
DEST="${1:-data/raw/qasper}"
mkdir -p "$DEST"
cd "$DEST"
for f in qasper-train-dev-v0.3.tgz qasper-test-and-evaluator-v0.3.tgz; do
  [ -f "$f" ] || curl -fL -o "$f" "https://qasper-dataset.s3.us-west-2.amazonaws.com/$f"
  tar -xzf "$f"
done
ls -la
