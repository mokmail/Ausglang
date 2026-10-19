#!/usr/bin/env bash
# Start the SGLang engine from the compose `gpu` profile and report progress.
#
#   ./start-gpu.sh
#
# The Workbench cannot see the Docker image pull or the container's warm-up, so
# this script writes <data>/engine-status.json — bind-mounted into the container
# at /data — and the UI renders it as "pulling image → starting engine → ready".
# Model download progress needs no help: it is read from the HF cache.
#
# Image pull progress is streamed from the Docker Engine API (no TTY parsing),
# so percentages are real even on the first few-GB pull.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DOCKER_SOCK="${DOCKER_SOCK:-/var/run/docker.sock}"
PULL_ONLY=0
[ "${1:-}" = "--pull-only" ] && PULL_ONLY=1

read_env() {  # read_env KEY DEFAULT
  local v=""
  [ -f "$HERE/.env" ] && v="$(grep -E "^$1=" "$HERE/.env" 2>/dev/null | tail -1 | cut -d= -f2- || true)"
  printf '%s' "${v:-$2}"
}

DATA_DIR="${WORKBENCH_DATA_DIR:-$(read_env WORKBENCH_DATA_DIR "$HERE/data")}"
case "$DATA_DIR" in /*) ;; *) DATA_DIR="$HERE/${DATA_DIR#./}" ;; esac
MODEL="${WORKBENCH_MODEL:-$(read_env MODEL Qwen/Qwen3-1.7B)}"
IMAGE="${SGLANG_IMAGE:-$(read_env SGLANG_IMAGE lmsysorg/sglang:latest)}"
PORT_ENGINE="${SGLANG_PORT:-$(read_env SGLANG_PORT 30000)}"
STATUS="$DATA_DIR/engine-status.json"

mkdir -p "$DATA_DIR"

status() {  # status PHASE PROGRESS MESSAGE HINT
  python3 - "$STATUS" "$1" "$2" "$3" "${4:-}" "$MODEL" <<'PY'
import json, os, sys, time
path, phase, progress, message, hint, model = sys.argv[1:7]
try:
    progress = None if progress == "" else float(progress)
except ValueError:
    progress = None
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump({"phase": phase, "progress": progress, "message": message,
               "hint": hint or None, "model": model or None,
               "updated": time.time()}, f)
os.replace(tmp, path)
PY
}

pull_with_progress() {
  python3 - "$IMAGE" "$STATUS" "$MODEL" "$DOCKER_SOCK" <<'PY'
import http.client, json, os, socket, sys, time, urllib.parse

image, path, model, sock_path = sys.argv[1:5]
repo, sep, tag = image.rpartition(":")
if not sep or "/" in tag:
    repo, tag = image, "latest"

class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=120)
        self.unix_path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(120)
        s.connect(self.unix_path)
        self.sock = s

layers = {}          # id -> [current, total]
last_emit = 0.0

def emit(phase="pulling", progress=None, message="pulling engine image"):
    global last_emit
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"phase": phase, "progress": progress, "message": message,
                   "hint": "first start only — the image is cached afterwards",
                   "model": model or None, "updated": time.time()}, f)
    os.replace(tmp, path)
    last_emit = time.time()

emit()

try:
    conn = UnixHTTPConnection(sock_path)
    q = urllib.parse.urlencode({"fromImage": repo, "tag": tag})
    conn.request("POST", "/images/create?" + q)
    resp = conn.getresponse()
    if resp.status >= 400:
        body = resp.read().decode("utf-8", "replace")[:300]
        emit("failed", None, f"image pull failed (HTTP {resp.status})")
        print(f"pull failed: HTTP {resp.status} {body}", file=sys.stderr)
        sys.exit(1)

    for raw in resp:
        try:
            msg = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            continue
        if msg.get("error"):
            emit("failed", None, "image pull failed")
            print(f"pull failed: {msg['error']}", file=sys.stderr)
            sys.exit(1)
        lid, detail = msg.get("id"), msg.get("progressDetail") or {}
        if lid and detail.get("total"):
            layers[lid] = [int(detail.get("current") or 0), int(detail.get("total") or 0)]
        elif lid and msg.get("status") in ("Download complete", "Pull complete"):
            if lid in layers:
                layers[lid][0] = layers[lid][1]
        now = time.time()
        if now - last_emit < 0.5:
            continue
        cur = sum(v[0] for v in layers.values())
        tot = sum(v[1] for v in layers.values())
        pct = round(100.0 * cur / tot, 1) if tot else None
        emit("pulling", pct if (pct is None or pct < 100) else None,
             f"pulling engine image · {len(layers)} layers")
    emit("pulling", None, "image ready")
    print("image ready")
except FileNotFoundError:
    print(f"note: {sock_path} not found — compose will pull the image itself", file=sys.stderr)
except PermissionError:
    print(f"note: no access to {sock_path} — compose will pull the image itself", file=sys.stderr)
except Exception as e:
    print(f"note: pull progress unavailable ({type(e).__name__}: {e})", file=sys.stderr)
PY
}

status pulling "" "pulling engine image" "first start only — the image is cached afterwards"
pull_ok=0
pull_with_progress && pull_ok=1 || true
if [ "$PULL_ONLY" = 1 ]; then
  [ "$pull_ok" = 1 ] && exit 0
  exit 1
fi

status starting "" "starting the engine" "loading $MODEL onto the GPU"
cd "$HERE"
MODEL="$MODEL" docker compose --profile gpu up -d

echo "waiting for http://127.0.0.1:$PORT_ENGINE/health …"
for i in $(seq 1 360); do          # up to 30 min: model download + warm-up
  code="$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT_ENGINE/health" 2>/dev/null || true)"
  if [ "$code" = "200" ]; then
    status ready 100 "engine ready" ""
    echo
    echo "engine     http://127.0.0.1:$PORT_ENGINE/v1"
    echo "workbench  http://127.0.0.1:${WORKBENCH_PORT:-8090}/"
    exit 0
  fi
  # keep the file fresh; the Workbench reads model-download progress from the
  # HF cache and prefers that over this "starting" message
  if [ $((i % 10)) -eq 0 ]; then
    status starting "" "starting the engine" "loading $MODEL onto the GPU"
  fi
  sleep 5
done

status failed "" "engine did not become healthy" \
  "check: docker compose --profile gpu logs sglang"
echo "engine did not become healthy — see: docker compose --profile gpu logs sglang" >&2
exit 1
