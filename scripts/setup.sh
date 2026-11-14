#!/usr/bin/env bash
# Idempotent environment setup: Python venv, dependencies, configuration check.
#   scripts/setup.sh           full install (everything the app can do; PyTorch is the big download)
#   scripts/setup.sh --lite    no neural packages: BM25/TF-IDF search, extraction, graph, gaps, review (no NLI/e5/rerank)
#   scripts/setup.sh --force   reinstall even if nothing changed
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/_common.sh"
cd "$RLX_ROOT"

MODE=full; FORCE=0
for a in "$@"; do case "$a" in --lite) MODE=lite ;; --force) FORCE=1 ;; -h|--help) sed -n 2,6p "$0"; exit 0 ;; *) die "unknown option: $a" ;; esac; done

PY_BIN="$(find_python)" || die "No suitable Python found. ResearchLens X needs Python 3.10 to 3.13.
       macOS:  brew install python@3.12      Debian/Ubuntu:  sudo apt install python3 python3-venv"

# 1. virtual environment
if [ -x "$RLX_PY" ] && ! "$RLX_PY" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' 2>/dev/null; then
  warn "existing .venv uses an unsupported Python; recreating it"; rm -rf "$RLX_VENV"
fi
if [ ! -x "$RLX_PY" ]; then
  say "Creating virtual environment in .venv ($("$PY_BIN" --version 2>&1))"
  "$PY_BIN" -m venv "$RLX_VENV" 2>"$RLX_LOG_DIR/setup.log" || { cat "$RLX_LOG_DIR/setup.log" >&2
    die "could not create the virtual environment. On Debian/Ubuntu install it with: sudo apt install python3-venv"; }
else
  ok "virtual environment present ($("$RLX_PY" --version 2>&1))"
fi

# 2. dependencies (skipped when requirement files and mode are unchanged)
sha() { if command -v shasum >/dev/null 2>&1; then shasum -a 256; else sha256sum; fi; }
STAMP_FILE="$RLX_VENV/.rlx-stamp"
STAMP="$(cat requirements.txt requirements-app.txt requirements-dense.txt pyproject.toml | sha | cut -d' ' -f1)-$MODE"
if [ "$FORCE" = 0 ] && [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE")" = "$STAMP" ] && "$RLX_PY" -c 'import researchlens, fitz, streamlit' 2>/dev/null; then
  ok "dependencies up to date ($MODE)"
else
  if [ "$MODE" = full ]; then
    say "Installing dependencies (full). First run downloads roughly 1-2 GB, mostly PyTorch; later runs skip this."
    REQ="-r requirements.txt -r requirements-app.txt -r requirements-dense.txt"
  else
    say "Installing dependencies (lite: no PyTorch, no neural models)."
    REQ="-r requirements.txt -r requirements-app.txt"
  fi
  : > "$RLX_LOG_DIR/setup.log"
  # shellcheck disable=SC2086
  if ! { "$RLX_PY" -m pip install --disable-pip-version-check --progress-bar off $REQ >>"$RLX_LOG_DIR/setup.log" 2>&1 \
         && "$RLX_PY" -m pip install --disable-pip-version-check --progress-bar off --no-deps -e . >>"$RLX_LOG_DIR/setup.log" 2>&1; }; then
    tail -n 25 "$RLX_LOG_DIR/setup.log" >&2
    die "dependency installation failed (full log: .rlx/logs/setup.log). Check your network connection, or try ./run.sh --lite"
  fi
  printf '%s' "$STAMP" > "$STAMP_FILE"
  ok "dependencies installed ($MODE)"
fi

# 3. configuration
if [ -f .env ]; then
  set -a; . ./.env; set +a; ok "loaded .env"
else
  ok "no .env found: using defaults (copy .env.example to .env to change them)"
fi
DB="${RLX_DB:-data/processed/library.sqlite}"
case "$DB" in /*) DBPATH="$DB" ;; *) DBPATH="$RLX_ROOT/$DB" ;; esac
mkdir -p "$(dirname "$DBPATH")" 2>/dev/null && [ -w "$(dirname "$DBPATH")" ] || die "RLX_DB directory is not writable: $(dirname "$DBPATH")"
case "${RLX_PORT:-8501}" in ''|*[!0-9]*) die "RLX_PORT must be a number (got '${RLX_PORT}')" ;; esac
ok "library database: $DB"
