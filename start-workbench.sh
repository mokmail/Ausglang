#!/usr/bin/env bash
# Start the whole workbench: SGLang engine (if needed) + the Docker container.
#
# The container cannot reach the Mac over `host.docker.internal` on this VM setup — that
# name resolves to the VM's own loopback, and the engine is refused. The LAN address does
# work, but it changes with DHCP, so resolve it here instead of hard-coding it in compose.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PORT_APP="${PORT_APP:-8090}"
PORT_ENGINE="${PORT_ENGINE:-43440}"
PORT_LOADER="${PORT_LOADER:-43441}"

lan_ip() {
  for iface in en0 en1 en2; do
    ip=$(ipconfig getifaddr "$iface" 2>/dev/null || true)
    [ -n "$ip" ] && { echo "$ip"; return; }
  done
  echo "127.0.0.1"
}

IP="$(lan_ip)"
echo "host LAN address: $IP"

# 1. Docker daemon (Rancher Desktop / Docker Desktop may be asleep after a reboot)
if ! docker info >/dev/null 2>&1; then
  echo "starting Docker…"
  open -a "Rancher Desktop" 2>/dev/null || open -a Docker 2>/dev/null || true
  for _ in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 3; done
  docker info >/dev/null 2>&1 || { echo "Docker did not start" >&2; exit 1; }
fi

# 2. Loader agent (owns the engine lifecycle; the container proxies model switches to it)
if ! curl -fsS -m 3 "http://127.0.0.1:$PORT_LOADER/healthz" >/dev/null 2>&1; then
  echo "starting loader agent…"
  "$HERE/run-loader.sh" >/dev/null 2>&1 || true
  sleep 3
fi

# 3. Engine: reuse a healthy one, otherwise start the model in settings
if [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT_ENGINE/health")" != "200" ]; then
  MODEL="${WORKBENCH_MODEL:-HuggingFaceTB/SmolLM3-3B}"
  echo "starting engine: $MODEL"
  printf '{"model":"%s","context_length":%s}' "$MODEL" "${WORKBENCH_CTX:-8192}" \
    | curl -fsS -m 20 -X POST "http://127.0.0.1:$PORT_LOADER/load" -H 'Content-Type: application/json' -d @- >/dev/null
  for _ in $(seq 1 60); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT_ENGINE/health")" = "200" ] && break
    sleep 5
  done
fi

# 4. Container, pointed at the address that actually routes to the host
cd "$HERE"
UPSTREAM="http://$IP:$PORT_ENGINE" \
WORKBENCH_LOADER_URL="http://$IP:$PORT_LOADER" \
  docker compose up -d "$@" 2>&1 | tail -2

for _ in $(seq 1 40); do
  [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT_APP/healthz")" = "200" ] && break
  sleep 3
done

echo
echo "workbench  http://127.0.0.1:$PORT_APP/   (LAN: http://$IP:$PORT_APP/)"
echo "engine     http://$IP:$PORT_ENGINE/v1"
echo "api docs   http://127.0.0.1:$PORT_APP/docs"
