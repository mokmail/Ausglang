#!/bin/bash
# Start the host-side model loader (so a containerized Workbench can switch models).
#
#   ./run-loader.sh                    # 0.0.0.0:43441, no token
#   LOADER_TOKEN=secret ./run-loader.sh
#
# Then, on the container side:
#   WORKBENCH_LOADER_URL=http://<host-ip>:43441
#   WORKBENCH_LOADER_TOKEN=secret          (only if LOADER_TOKEN was set)
#
# macOS/Windows: Metal/CUDA cannot be driven from a Linux container, which is why
# this agent exists. Linux+NVIDIA can skip it and use the compose `gpu` profile.

set -uo pipefail
PORT="${1:-43441}"
ROOT=/Users/kmail/sglang
LOG="$ROOT/loader.log"

if pgrep -f "workbench/loader_agent.py" >/dev/null; then
  echo "loader already running:"; pgrep -fl "workbench/loader_agent.py" | cut -c1-110
  exit 0
fi

cd "$ROOT"
source .venv/bin/activate
nohup python workbench/loader_agent.py --port "$PORT" --host 0.0.0.0 >> "$LOG" 2>&1 &
echo "loader pid=$! — waiting for /healthz …"

for _ in $(seq 1 30); do
  if curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/healthz"; then
    LANIP=$(ipconfig getifaddr en0 2>/dev/null || true)
    echo "READY: http://127.0.0.1:$PORT/healthz"
    [ -n "${LANIP:-}" ] && echo "LAN:   http://$LANIP:$PORT   (set WORKBENCH_LOADER_URL to this)"
    [ -n "${LOADER_TOKEN:-}" ] && echo "token: LOADER_TOKEN is set — mirror it as WORKBENCH_LOADER_TOKEN"
    exit 0
  fi
  sleep 1
done

echo "loader did not start — last lines of $LOG:"; tail -20 "$LOG"; exit 1
