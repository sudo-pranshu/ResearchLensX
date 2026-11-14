#!/usr/bin/env bash
# ResearchLens X: one command to set up and launch everything locally.
#
#   ./run.sh                 set up if needed, start the app, health-check it, print the URL (Ctrl+C stops it)
#   ./run.sh --lite          skip PyTorch and neural models (BM25/TF-IDF search; no NLI, e5 or reranker)
#   ./run.sh --port 8600     preferred port (default 8501; the next free port is used if it is taken)
#   ./run.sh --open          open the app in your browser
#   ./run.sh --detach        start in the background and return (stop with scripts/stop.sh)
#   ./run.sh --restart       stop a running instance first
# Supported: macOS and Linux. Windows: use WSL2.
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
. scripts/_common.sh

SETUP_FLAGS=""; PORT_WANT=""; OPEN=0; DETACH=0; RESTART=0
while [ $# -gt 0 ]; do
  case "$1" in
    --lite) SETUP_FLAGS="--lite" ;;
    --port) shift; PORT_WANT="${1:-}" ;;
    --open) OPEN=1 ;;
    --detach) DETACH=1 ;;
    --restart) RESTART=1 ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    *) die "unknown option: $1 (see ./run.sh --help)" ;;
  esac; shift
done

case "$(uname -s)" in
  Darwin|Linux) ;;
  *) die "Native Windows is not supported. Install WSL2 and run ./run.sh inside it." ;;
esac

printf '%s\n' "${C_B}ResearchLens X${C_0}  Scientific Research Intelligence Platform"
[ "$RESTART" = 1 ] && scripts/stop.sh

# Already running? Do not start a duplicate.
OLD_PID="$(cat "$RLX_PID_FILE" 2>/dev/null || true)"; OLD_PORT="$(cat "$RLX_PORT_FILE" 2>/dev/null || true)"
if pid_alive "$OLD_PID"; then
  if [ -n "$OLD_PORT" ] && http_ok "http://127.0.0.1:$OLD_PORT/_stcore/health"; then
    ok "ResearchLens X is already running: http://localhost:$OLD_PORT"
    echo "    stop it with scripts/stop.sh, or restart with ./run.sh --restart"; exit 0
  fi
  warn "found a stale instance (pid $OLD_PID); replacing it"; scripts/stop.sh --quiet
fi

# 1. environment
# shellcheck disable=SC2086
scripts/setup.sh $SETUP_FLAGS
[ -f .env ] && { set -a; . ./.env; set +a; }

# 2. port
WANT="${PORT_WANT:-${RLX_PORT:-8501}}"
PORT="$(free_port "$WANT")" || die "no free port found near $WANT; pass one with --port"
[ "$PORT" = "$WANT" ] || warn "port $WANT is in use; using $PORT instead"

# 3. models: tell the user before anything large is downloaded
HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
if "$RLX_PY" -c 'import torch' 2>/dev/null; then
  say "Neural models are downloaded from the Hugging Face Hub the first time a feature needs them:"
  for m in "intfloat--e5-base-v2:e5 dense retrieval (~0.4 GB)" "cross-encoder--ms-marco-MiniLM-L-6-v2:reranker (~0.1 GB)" "MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli:NLI verification (~0.4 GB)"; do
    d="${m%%:*}"; label="${m#*:}"
    if [ -d "$HUB/models--$d" ]; then echo "    cached       $label"; else echo "    on first use $label"; fi
  done
  echo "    BM25 and TF-IDF search, extraction, comparison, graph, gaps and reviews need no model."
else
  say "Lite mode: no neural models. NLI verification, e5 and the reranker are unavailable (the app shows this)."
fi

# 4. start
LOG="$RLX_LOG_DIR/app.log"
say "Starting the app on port $PORT (log: .rlx/logs/app.log)"
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false nohup "$RLX_PY" -m streamlit run apps/streamlit/streamlit_app.py \
  --server.port "$PORT" --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false >>"$LOG" 2>&1 &
PID=$!
printf '%s' "$PID" > "$RLX_PID_FILE"; printf '%s' "$PORT" > "$RLX_PORT_FILE"

cleanup() { trap - INT TERM; echo; say "Shutting down"; scripts/stop.sh --quiet; ok "stopped"; exit 0; }
[ "$DETACH" = 1 ] || trap cleanup INT TERM

# 5. wait for health
URL="http://localhost:$PORT"
for i in $(seq 1 90); do
  if ! pid_alive "$PID"; then tail -n 25 "$LOG" >&2; rm -f "$RLX_PID_FILE" "$RLX_PORT_FILE"; die "the app exited during startup (log above, full log: .rlx/logs/app.log)"; fi
  http_ok "http://127.0.0.1:$PORT/_stcore/health" && break
  sleep 1
  if [ "$i" = 90 ]; then tail -n 25 "$LOG" >&2; scripts/stop.sh --quiet; die "the app did not become healthy within 90 s (log above)"; fi
done
scripts/healthcheck.sh --app || { scripts/stop.sh --quiet; die "health checks failed (see messages above)"; }

printf '\n%s\n' "${C_G}${C_B}ResearchLens X is running:  $URL${C_0}"
echo "  Library page: upload a research PDF to begin. Quick tour: docs/user-guide/DEMO_WORKFLOW.md"
if [ "$OPEN" = 1 ]; then { command -v open >/dev/null && open "$URL"; } || { command -v xdg-open >/dev/null && xdg-open "$URL"; } || true; fi
if [ "$DETACH" = 1 ]; then echo "  Running in the background. Stop it with: scripts/stop.sh"; exit 0; fi
echo "  Press Ctrl+C to stop."
wait "$PID" || true
if ! pid_alive "$PID"; then rm -f "$RLX_PID_FILE" "$RLX_PORT_FILE"; fi
