#!/bin/bash
# Verify the SGLang Workbench end-to-end: API, SSE streaming, tool harness,
# memory store, MCP client, persistence, UI rendering.
#
#   ./verify-workbench.sh          # ~2 min, needs the SGLang server + workbench up
#
# Ad-hoc check, not a test suite: there is no canonical lint/build for this app.
set -uo pipefail
WB=${WB:-http://127.0.0.1:8090}
UP=http://127.0.0.1:43440
PY=/Users/kmail/sglang/.venv/bin/python
fail=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; fail=1; fi; }
jget() { curl -s -m 15 "$WB$1"; }

echo "workbench = $WB   sglang = $UP"

# ---- backend + config ------------------------------------------------------
check "workbench /healthz"        "[ \$(curl -s -m 5 -o /dev/null -w '%{http_code}' $WB/healthz) = '200' ]"
check "sglang upstream healthy"   "[ \$(curl -s -m 5 -o /dev/null -w '%{http_code}' $UP/health) = '200' ]"
check "state reports server up"   "jget /api/state | $PY -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d[\"server\"][\"up\"] else 1)'"
check "context window configured" "jget /api/state | $PY -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d[\"settings\"][\"context_length\"]>=8192 else 1)'"
check "models list from cache"    "jget '/api/models?source=cache' | grep -q 'Qwen/Qwen3-0.6B'"
check "settings round-trip"       "curl -s -X POST -m 10 $WB/api/settings -H 'Content-Type: application/json' -d '{\"temperature\":0.55}' | grep -q '\"temperature\":0.55'"
curl -s -X POST -m 10 $WB/api/settings -H 'Content-Type: application/json' -d '{"temperature":0.7}' >/dev/null

# ---- chats + streaming -----------------------------------------------------
CID=$(curl -s -X POST -m 10 $WB/api/chats -H 'Content-Type: application/json' -d '{}' | $PY -c 'import json,sys; print(json.load(sys.stdin)["id"])')
check "chat created"               "[ -n '$CID' ]"
stream=$(curl -s -N -m 180 $WB/api/chat -H 'Content-Type: application/json' \
  -d "{\"chat_id\":\"$CID\",\"message\":\"Say hello in three words.\"}")
check "SSE emits deltas"           "echo \"\$stream\" | grep -q '^event: delta'"
check "SSE ends with done"         "echo \"\$stream\" | grep -q '^event: done'"
check "message persisted"          "jget /api/chats/$CID/messages | grep -q '\"role\":\"assistant\"'"
check "chat title auto-set"        "jget /api/chats | grep -q 'Say hello in three words'"

regen=$(curl -s -N -m 180 $WB/api/chat -H 'Content-Type: application/json' -d "{\"chat_id\":\"$CID\",\"regenerate\":true}")
check "regenerate streams answer"  "echo \"\$regen\" | grep -q '^event: done'"
check "export as markdown"         "curl -s -m 10 '$WB/api/chats/$CID/export?fmt=md' | grep -q '^# '"

# ---- memory ----------------------------------------------------------------
MTXT="The verification run lives at $(date +%H:%M:%S)"
MID=$(curl -s -X POST -m 10 $WB/api/memory -H 'Content-Type: application/json' -d "{\"text\":\"$MTXT\"}" | $PY -c 'import json,sys; print(json.load(sys.stdin)["id"])')
check "memory created"             "[ -n '$MID' ]"
check "memory listed"              "jget /api/memory | grep -q 'verification run lives at'"
check "memory deleted"             "curl -s -X DELETE -m 10 $WB/api/memory/$MID | grep -q deleted"

# ---- MCP + tool harness ----------------------------------------------------
# the bundled demo server is auto-registered by the backend with a path that exists
# in whichever deployment this is (host or container); just connect to it.
conn=$(curl -s -X POST -m 30 $WB/api/mcp/servers/demo/connect -H 'Content-Type: application/json' -d '{}')
check "MCP connect lists tools"    "echo '$conn' | grep -q 'disk_free'"
check "MCP tool call works"        "curl -s -X POST -m 30 $WB/api/mcp/tools/call -H 'Content-Type: application/json' -d '{\"server\":\"demo\",\"tool\":\"now\",\"arguments\":{}}' | grep -q 'T'"
check "tools exposed to model"     "jget /api/state | grep -q 'disk_free'"

TCID=$(curl -s -X POST -m 10 $WB/api/chats -H 'Content-Type: application/json' -d '{}' | $PY -c 'import json,sys; print(json.load(sys.stdin)["id"])')
toolstream=$(curl -s -N -m 240 $WB/api/chat -H 'Content-Type: application/json' \
  -d "{\"chat_id\":\"$TCID\",\"message\":\"Use the disk_free tool and report the free space.\"}")
check "harness emits tool events"  "echo \"\$toolstream\" | grep -q '^event: tool'"
check "harness reports rounds"     "echo \"\$toolstream\" | grep -q '\"rounds\": [1-9]'"

# ---- UI --------------------------------------------------------------------
check "index served"               "[ \$(curl -s -m 5 -o /dev/null -w '%{http_code}' $WB/) = '200' ]"
for f in app.js app.css; do check "static $f served" "[ \$(curl -s -m 5 -o /dev/null -w '%{http_code}' $WB/static/$f) = '200' ]"; done

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ -x "$CHROME" ]; then
  prof=$(mktemp -d)/prof
  dom=$("$CHROME" --headless=new --disable-gpu --no-first-run --user-data-dir="$prof" \
        --virtual-time-budget=25000 --dump-dom "$WB/?panel=memory" 2>/dev/null)
  check "UI renders chat list"      "echo \"\$dom\" | grep -q 'chat-item'"
  check "UI shows model chip"       "echo \"\$dom\" | grep -q 'id=\"modelChip\"'"
  check "UI opens memory drawer"    "echo \"\$dom\" | grep -q 'id=\"panel\" class=\"open\"'"
  check "UI composer present"       "echo \"\$dom\" | grep -q 'id=\"input\"'"
  rm -rf "$(dirname "$prof")"
else
  echo "SKIP  UI DOM checks (no Chrome)"
fi


# ---- harness components (c1..c8) -------------------------------------------
curl -s -m 10 $WB/api/harnesses > /tmp/wb_h.json
check "8 harness components exposed"      "grep -q plugins /tmp/wb_h.json && grep -q tool_guidance /tmp/wb_h.json"
check "default harness exists"            "grep -q Default /tmp/wb_h.json"

printf '%s' '{"name":"verify-harness","system_prompt":"Be terse.","guardrails":"No secrets.","skills":[{"name":"cite","body":"Name the source of every fact."}],"subagents":[{"name":"critic","system":"You critique drafts.","description":"Critique"}],"rules":{"blocked_input":["SSN[ ]*[:=]"],"max_prompt_tokens":6000}}' > /tmp/wb_harness.json
NH=$(curl -s -X POST -m 10 $WB/api/harnesses -H 'Content-Type: application/json' -d @/tmp/wb_harness.json)
HID=$(echo "$NH" | $PY -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
check "harness created"                   "[ -n '$HID' ]"
check "harness preview composes"          "curl -s -m 10 '$WB/api/harnesses/$HID/preview?q=hello' | grep -q '## Role'"
check "preview reports token budget"      "curl -s -m 10 '$WB/api/harnesses/$HID/preview?q=hello' | grep -q tokens"

HC=$(curl -s -X POST -m 10 $WB/api/chats -H 'Content-Type: application/json' -d '{}' | $PY -c 'import json,sys; print(json.load(sys.stdin)["id"])')
printf '%s' "{\"harness_id\":\"$HID\"}" > /tmp/wb_bind.json
curl -s -X POST -m 10 $WB/api/chats/$HC/harness -H 'Content-Type: application/json' -d @/tmp/wb_bind.json > /dev/null

printf '%s' "{\"chat_id\":\"$HC\",\"message\":\"my SSN: 123-45-6789\"}" > /tmp/wb_blocked.json
GR=$(curl -s -m 20 -o /dev/null -w '%{http_code}' -X POST $WB/api/chat -H 'Content-Type: application/json' -d @/tmp/wb_blocked.json)
check "guardrail blocks input (400)"      "[ '$GR' = '400' ]"

printf '%s' "{\"chat_id\":\"$HC\",\"message\":\"Reply with exactly: HARNESS OK\"}" > /tmp/wb_hchat.json
curl -s -N -m 180 $WB/api/chat -H 'Content-Type: application/json' -d @/tmp/wb_hchat.json > /tmp/wb_hchat.out
check "chat runs under the harness"       "grep -q 'HARNESS OK' /tmp/wb_hchat.out"
check "done carries the prompt report"    "grep -q prompt /tmp/wb_hchat.out"
check "done carries guardrail verdict"    "grep -q guardrails /tmp/wb_hchat.out"

printf '%s' "{\"chat_id\":\"$HC\",\"message\":\"Use the delegate tool to ask the critic subagent to critique: local models are slow.\"}" > /tmp/wb_dl.json
curl -s -N -m 240 $WB/api/chat -H 'Content-Type: application/json' -d @/tmp/wb_dl.json > /tmp/wb_dl.out
check "subagent tool invoked"             "grep -q '\"server\": \"harness\"' /tmp/wb_dl.out"
check "subagent returned a result"        "grep -q 'state.*done' /tmp/wb_dl.out"

printf '%s' "{\"chat_id\":\"$HC\",\"harness_id\":\"$HID\"}" > /tmp/wb_probe.json
PR=$(curl -s -m 240 -X POST $WB/api/harness/probe -H 'Content-Type: application/json' -d @/tmp/wb_probe.json)
check "probe returns candidate output"    "echo '$PR' | grep -q text"
check "probe reports timing"              "echo '$PR' | grep -q elapsed_s"

printf '%s' '{"lesson":"Check the context window before sending long prompts.","domain":"verify"}' > /tmp/wb_lesson.json
LID=$(curl -s -X POST -m 10 $WB/api/lessons -H 'Content-Type: application/json' -d @/tmp/wb_lesson.json | $PY -c 'import json,sys; print(json.load(sys.stdin).get("id",""))')
check "lesson stored"                     "[ -n '$LID' ] && curl -s -m 10 $WB/api/lessons | grep -q 'context window'"
printf '%s' "{\"chat_id\":\"$HC\"}" > /tmp/wb_reflect.json
curl -s -m 240 -X POST $WB/api/lessons/reflect -H 'Content-Type: application/json' -d @/tmp/wb_reflect.json > /tmp/wb_reflect.out
check "reflect produces a lesson"         "grep -q lesson /tmp/wb_reflect.out"
check "lesson delete"                     "curl -s -X DELETE -m 10 $WB/api/lessons/$LID | grep -q deleted"
check "harness delete"                    "curl -s -X DELETE -m 10 $WB/api/harnesses/$HID | grep -q deleted"

# ---- cleanup ---------------------------------------------------------------
curl -s -X DELETE -m 10 $WB/api/chats/$CID >/dev/null
curl -s -X DELETE -m 10 $WB/api/chats/$TCID >/dev/null
curl -s -X DELETE -m 10 $WB/api/chats/$HC >/dev/null

echo "---"; [ $fail -eq 0 ] && echo "RESULT: ad-hoc checks all passed" || echo "RESULT: FAILURES PRESENT"
exit $fail
