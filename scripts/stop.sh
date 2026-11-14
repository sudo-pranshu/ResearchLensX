#!/usr/bin/env bash
# Gracefully stop a running ResearchLens X app (started by ./run.sh, foreground or --detach).
set -uo pipefail
. "$(cd "$(dirname "$0")" && pwd)/_common.sh"
QUIET=0; [ "${1:-}" = "--quiet" ] && QUIET=1
PID="$(cat "$RLX_PID_FILE" 2>/dev/null || true)"
if ! pid_alive "$PID"; then
  rm -f "$RLX_PID_FILE" "$RLX_PORT_FILE"
  [ "$QUIET" = 1 ] || ok "nothing to stop (ResearchLens X is not running)"; exit 0
fi
[ "$QUIET" = 1 ] || say "Stopping ResearchLens X (pid $PID)"
pkill -TERM -P "$PID" 2>/dev/null || true
kill -TERM "$PID" 2>/dev/null || true
for _ in 1 2 3 4 5 6 7 8 9 10; do pid_alive "$PID" || break; sleep 1; done
if pid_alive "$PID"; then warn "did not exit within 10 s, forcing"; pkill -KILL -P "$PID" 2>/dev/null || true; kill -KILL "$PID" 2>/dev/null || true; fi
rm -f "$RLX_PID_FILE" "$RLX_PORT_FILE"
[ "$QUIET" = 1 ] || ok "stopped"
