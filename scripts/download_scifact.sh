#!/usr/bin/env bash
# SciFact (Wadden et al., 2020), CC BY-NC 2.0. ~3 MB compressed. Source: AllenAI public S3 release.
set -euo pipefail
DEST="${1:-data/raw/scifact}"
mkdir -p "$DEST" && cd "$DEST"
[ -f data.tar.gz ] || curl -fL -o data.tar.gz "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
tar -xzf data.tar.gz
ls -la data
