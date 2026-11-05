#!/usr/bin/env bash
# EXP-H1-005: H1 secondary analyses on QASPER dev (no confirmatory claims). Usage: scripts/run_h1_ablations.sh <dense>
set -euo pipefail
D="${1:-specter2}"
CFG=configs/h1_classical_dev.yaml
S="bm25 $D rrf:bm25+$D"
run(){ python3 experiments/h1_retrieval.py --config $CFG --systems $S "$@"; }
# (a) section-aware vs section-agnostic units (captions excluded in all three arms for comparability)
run --no-floats --run-name EXP-H1-005a_units_paragraph_prefix
run --no-floats --no-section-prefix --run-name EXP-H1-005b_units_paragraph_noprefix
run --no-floats --unit window --window-words 128 --run-name EXP-H1-005c_units_window128
# (b) RRF k sensitivity
for k in 10 30 100; do run --rrf-k $k --systems "rrf:bm25+$D" --run-name EXP-H1-005d_rrf_k$k; done
# (c) title-conditioned open-corpus diagnostic with the dense retriever
run --setting open --title-query --run-name EXP-H1-005e_open_titleq
run --setting open --run-name EXP-H1-005f_open_bare
