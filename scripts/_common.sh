# Shared helpers, sourced by run.sh and scripts/*.sh. Not executed directly. Works with macOS bash 3.2 and Linux bash.
RLX_ROOT="${RLX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RLX_RUN_DIR="$RLX_ROOT/.rlx/run"
RLX_LOG_DIR="$RLX_ROOT/.rlx/logs"
RLX_VENV="$RLX_ROOT/.venv"
RLX_PY="$RLX_VENV/bin/python"
RLX_PID_FILE="$RLX_RUN_DIR/app.pid"
RLX_PORT_FILE="$RLX_RUN_DIR/app.port"
mkdir -p "$RLX_RUN_DIR" "$RLX_LOG_DIR"

if [ -t 1 ]; then C_B=$'\033[1m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_R=$'\033[31m'; C_D=$'\033[2m'; C_0=$'\033[0m'; else C_B=; C_G=; C_Y=; C_R=; C_D=; C_0=; fi
say()  { printf '%s\n' "${C_B}==>${C_0} $*"; }
ok()   { printf '%s\n' "${C_G} ok${C_0} $*"; }
warn() { printf '%s\n' "${C_Y}warn${C_0} $*" >&2; }
die()  { printf '%s\n' "${C_R}error${C_0} $*" >&2; exit 1; }

# pid_alive PID -> 0 if the process exists
pid_alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

# http_ok URL -> 0 if the Streamlit health endpoint answers "ok" (uses curl, else the venv python)
http_ok() {
  local url="$1" body=""
  if command -v curl >/dev/null 2>&1; then body="$(curl -fsS --max-time 3 "$url" 2>/dev/null || true)"
  elif [ -x "$RLX_PY" ]; then body="$("$RLX_PY" - "$url" <<'PY' 2>/dev/null || true
import sys, urllib.request
print(urllib.request.urlopen(sys.argv[1], timeout=3).read().decode())
PY
)"; fi
  [ "$body" = "ok" ]
}

# find_python -> prints a Python >= 3.10 and < 3.14 interpreter, or nothing
find_python() {
  local c
  for c in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' 2>/dev/null; then
      command -v "$c"; return 0
    fi
  done
  return 1
}

# free_port START -> first free TCP port >= START (tries 30 ports)
free_port() {
  "${RLX_PY:-python3}" - "$1" <<'PY'
import socket, sys
start = int(sys.argv[1])
for p in range(start, start + 30):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", p)); print(p); sys.exit(0)
        except OSError:
            pass
sys.exit(1)
PY
}
