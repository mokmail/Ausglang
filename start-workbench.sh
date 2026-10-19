#!/usr/bin/env bash
# Start SGLang Workbench with docker compose, on macOS / Linux / Windows (WSL2).
#
#    ./start-workbench.sh              # engine (if needed) + container
#    ./start-workbench.sh --no-engine  # container only, engine runs elsewhere
#    ./start-workbench.sh down         # stop everything
#
# Why a script at all: the container must reach the *host*, and the documented name for
# that (host.docker.internal) resolves to the VM's own loopback on some runners, where the
# host's LAN address is the only route. The script writes the address it finds into .env,
# so a plain `docker compose up -d` keeps working afterwards.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PORT_APP="${PORT_APP:-8090}"
PORT_ENGINE="${PORT_ENGINE:-43440}"
PORT_LOADER="${PORT_LOADER:-43441}"
MODEL="${WORKBENCH_MODEL:-HuggingFaceTB/SmolLM3-3B}"
CTX="${WORKBENCH_CTX:-8192}"
START_ENGINE=1
COMPOSE_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --no-engine) START_ENGINE=0 ;;
    down|stop) docker compose -f "$HERE/docker-compose.yml" down 2>/dev/null || true; exit 0 ;;
    *) COMPOSE_ARGS+=("$arg") ;;
  esac
done

lan_ip() {
  case "$(uname -s)" in
    Darwin) for i in en0 en1 en2; do ip=$(ipconfig getifaddr "$i" 2>/dev/null || true); [ -n "$ip" ] && { echo "$ip"; return; }; done ;;
    Linux)  ip route get 1.1.1.1 2>/dev/null | grep -o 'src [0-9.]*' | awk '{print $2}' | head -1 ;;
  esac
  echo "127.0.0.1"
}

IP="$(lan_ip)"
[ "$IP" = "127.0.0.1" ] && echo "warning: no LAN address found, falling back to loopback" >&2

# ---- 1. docker -------------------------------------------------------------
if ! docker info >/dev/null 2>&1; then
  echo "starting Docker…"
  if [ "$(uname -s)" = "Darwin" ]; then
    open -a "Rancher Desktop" 2>/dev/null || open -a Docker 2>/dev/null || true
  fi
  for _ in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 3; done
  docker info >/dev/null 2>&1 || { echo "Docker is not running" >&2; exit 1; }
fi

# ---- 2. .env (the container's map back to this host) -----------------------
HF=""
for cand in "${HF_CACHE_DIR:-}" "$HOME/.cache/huggingface"; do
  [ -n "$cand" ] && [ -d "$cand" ] && { HF="$cand"; break; }
done

[ -f "$HERE/.env" ] || { [ -f "$HERE/.env.example" ] && cp "$HERE/.env.example" "$HERE/.env"; }

set_kv() {  # set_kv KEY VALUE — rewrite one line, keep the rest
  local k="$1" v="$2" f="$HERE/.env"
  touch "$f"
  if grep -q "^${k}=" "$f"; then
    awk -v k="$k" -v v="$v" 'BEGIN{FS=OFS="="} $1==k{print k"="v; next} {print}' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  else
    printf '%s=%s\n' "$k" "$v" >> "$f"
  fi
}
set_kv UPSTREAM "http://$IP:$PORT_ENGINE"
set_kv WORKBENCH_LOADER_URL "http://$IP:$PORT_LOADER"
[ -n "$HF" ] && set_kv HF_CACHE_DIR "$HF"
echo "host address: $IP  (written to .env)"

# ---- 3. engine (host side; the container has no GPU) -----------------------
if [ "$START_ENGINE" = "1" ] && [ -x "$HERE/run-loader.sh" ]; then
  if ! curl -fsS -m 3 "http://127.0.0.1:$PORT_LOADER/healthz" >/dev/null 2>&1; then
    echo "starting loader agent…"
    "$HERE/run-loader.sh" >/dev/null 2>&1 || true
    sleep 3
  fi
  if [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT_ENGINE/health")" != "200" ]; then
    echo "starting engine: $MODEL (ctx $CTX)"
    printf '{"model":"%s","context_length":%s}' "$MODEL" "$CTX" \
      | curl -fsS -m 20 -X POST "http://127.0.0.1:$PORT_LOADER/load" -H 'Content-Type: application/json' -d @- >/dev/null || true
    for _ in $(seq 1 60); do
      [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT_ENGINE/health")" = "200" ] && break
      sleep 5
    done
  fi
fi

# ---- 4. container ----------------------------------------------------------
cd "$HERE"
docker compose up -d --build ${COMPOSE_ARGS[@]+"${COMPOSE_ARGS[@]}"} 2>&1 | tail -3

for _ in $(seq 1 40); do
  [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT_APP/healthz")" = "200" ] && break
  sleep 3
done

echo
echo "workbench  http://127.0.0.1:$PORT_APP/    (LAN: http://$IP:$PORT_APP/)"
echo "api docs   http://127.0.0.1:$PORT_APP/docs"
echo "engine     http://$IP:$PORT_ENGINE/v1"
echo "diagnose   curl -s http://127.0.0.1:$PORT_APP/api/upstream"
