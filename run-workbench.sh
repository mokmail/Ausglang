#!/bin/bash
# Start the SGLang Workbench (backend + UI) detached.
#
#   ./run-workbench.sh                 # http://0.0.0.0:8090
#   ./run-workbench.sh 8091            # custom port
#
# Stop: pkill -f workbench/app.py      Logs: workbench/data/workbench.log

set -uo pipefail
PORT="${1:-8090}"
ROOT=/Users/kmail/sglang
WB="$ROOT/workbench"
LOG="$WB/data/workbench.log"

mkdir -p "$WB/data"

if pgrep -f "workbench/app.py" >/dev/null; then
  echo "workbench already running:"; pgrep -fl "workbench/app.py" | cut -c1-110
  exit 0
fi

cd "$ROOT"
source .venv/bin/activate
nohup python "$WB/app.py" --port "$PORT" --host 0.0.0.0 >> "$LOG" 2>&1 &
echo "workbench pid=$! — waiting for /healthz …"

for _ in $(seq 1 30); do
  if curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/healthz"; then
    LANIP=$(ipconfig getifaddr en0 2>/dev/null || true)
    echo "READY: http://127.0.0.1:$PORT/"
    [ -n "${LANIP:-}" ] && echo "LAN:   http://$LANIP:$PORT/   (unauthenticated)"
    exit 0
  fi
  sleep 1
done

echo "did not start — last lines of $LOG:"
tail -20 "$LOG"
exit 1
