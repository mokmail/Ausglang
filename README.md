# SGLang Workbench

A local LLM workbench over an SGLang (OpenAI-compatible) server: professional chat UI,
long-term memory, MCP tools with an agent harness, model switching, attachments,
personas, prompt library and a documented FastAPI.

The workbench is **GPU-agnostic** — it speaks HTTP to SGLang, so the same image runs
on macOS, Windows and Linux while the Metal/CUDA stack stays where it works.

---

## Quick start

### 1. Start SGLang (the model server)

**macOS (Metal + MLX)**

```bash
~/sglang/start-sglang.sh                 # Qwen/Qwen3-0.6B on :43440
~/sglang/start-sglang.sh Qwen/Qwen3-1.7B 43440 0.0.0.0 0.30 4 12288
```

**Linux / Windows (WSL2) with NVIDIA**

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B \
  --host 0.0.0.0 --port 43440 --mem-fraction-static 0.85
```

Check it: `curl http://localhost:43440/health` → 200.

**Docker networking note.** Docker Desktop maps the host to
`host.docker.internal`. Rancher Desktop / colima can resolve that name to the VM
gateway and still refuse the connection — use the machine's LAN address instead
(`ipconfig getifaddr en0` on macOS) and check it with
`GET /api/upstream/check?url=http://<ip>:43440`.

### 2. Run the workbench

**Docker compose (recommended, any OS)**

```bash
cd ~/sglang/workbench
cp env.example .env          # adjust UPSTREAM / HF_CACHE_DIR if needed
docker compose up -d --build
open http://localhost:8090
```

**Docker, one-liner (macOS/Linux)**

```bash
docker run -d --name sglang-workbench -p 8090:8090 \
  -e UPSTREAM=http://host.docker.internal:43440 \
  --add-host host.docker.internal:host-gateway \
  -v sglang-workbench-data:/data \
  -v "$HOME/.cache/huggingface:/hf-cache:ro" \
  sglang-workbench:2.0.0
```

**Without Docker (uses the SGLang venv)**

```bash
~/sglang/workbench/run-workbench.sh        # http://localhost:8090
```

### Switching models

Click the model chip in the header — one click on any downloaded model restarts
the server on it. A banner under the header shows the live job (download %,
bytes, rate, ETA, elapsed, cancel), on every tab, and survives a page reload
because it is driven by `GET /api/server/job`, not by the click that started it.

Deep links: `/?switch=org/model`, `/?menu=models`, `/?panel=model`.

### Memory self-healing

Starting a model on Metal is a two-part bet: the weights must fit **and** the KV pool must be
big enough to be useful. Both are checked automatically:

1. crash during startup with `Insufficient Memory` → the loader halves the context and retries once;
2. started, but `/get_server_info` reports a KV pool below `min(4096, ctx)` → the loader stops it,
   raises `--mem-fraction-static` one notch (+0.15, capped 0.55), caps the context at 8192 and retries;
3. crash later under load with the same OOM signature → the supervisor restarts at the reduced
   context with `--max-running-requests 1` instead of looping on the same settings.

The job message and the server card report the resulting pool (`pool 32,974 tok`); a pool under
4096 tokens is shown as a warning, because a model that only fits 256 tokens answers nothing.

Example from this machine: `HuggingFaceTB/SmolLM3-3B` at ctx 12288 produced a 256-token pool,
was refitted to mem-fraction 0.45 / ctx 8192, and served 8,069-token prompts in ~11 s.

### Model loading from inside a container

A Linux container cannot drive Metal, so `POST /api/server/start` returns HTTP 501
there by design. To keep "Load model" working, run the host-side loader and point
the container at it:

```bash
./run-loader.sh                                  # host: 0.0.0.0:43441
WORKBENCH_LOADER_URL=http://<host-ip>:43441 docker compose up -d
```

The Workbench then forwards `start/stop/cancel/job/logs` to the loader, which
spawns SGLang on the host where the GPU actually is. Set `LOADER_TOKEN` on the
host and `WORKBENCH_LOADER_TOKEN` on the container if the loader is reachable
from other machines.

### 3. Linux + NVIDIA only: SGLang in its own container

```bash
MODEL=Qwen/Qwen3-1.7B docker compose --profile gpu up -d
UPSTREAM=http://sglang:30000 docker compose up -d workbench
```

Requires the NVIDIA container toolkit. Not possible on macOS/Windows: containers
there have no Metal and no CUDA.

---

## Platform matrix

| Host | SGLang runs | Workbench runs | Notes |
|---|---|---|---|
| macOS (Apple Silicon) | host, Metal/MLX | container **or** host | containers cannot reach Metal |
| Windows 11 | WSL2 host | container (Docker Desktop) | use `host.docker.internal` |
| Linux + NVIDIA | host or `--profile gpu` | container | GPU profile needs the NVIDIA toolkit |
| Linux CPU | host (`--device cpu` builds) | container | slow but works |

Model **loading** from inside the container is refused by design (HTTP 501) unless
`SGLANG_PYTHON` points at an interpreter that can run `sglang.launch_server`. Browse,
select and inspect models in the UI; start the server on the host.

---

## API

Interactive docs: **`/docs`** (Swagger UI), **`/redoc`**, raw spec at **`/openapi.json`**.

31 endpoints grouped by tag:

| Tag | Endpoints |
|---|---|
| `system` | `/api/state`, `/api/version`, `/api/settings`, `/healthz` |
| `chat` | `POST /api/chat` — SSE stream (`start`, `delta`, `reasoning`, `tool`, `done`) |
| `chats` | list/create/patch/delete, `/fork`, `/messages`, `/export`, `/autotitle` |
| `models` | `/api/models?source=cache\|curated\|mlx\|hf\|local`, `/api/models/detail` |
| `server` | `/start`, `/job`, `/cancel`, `/stop`, `/logs` |
| `memory` | long-term facts injected into the system prompt |
| `prompts` | reusable prompt library |
| `personas` | saved system-prompt + sampling profiles |
| `stats` | token/message accounting per day |

### Examples

```bash
# runtime snapshot
curl -s localhost:8090/api/state | jq '.server, .job.state'

# streaming chat
curl -N localhost:8090/api/chat -H 'Content-Type: application/json' \
  -d '{"chat_id":"<id>","message":"Explain chunked prefill in two lines"}'

# load a model (requires SGLANG_PYTHON; refused in containers)
curl -X POST localhost:8090/api/server/start -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","context_length":12288}'

# register and call an MCP tool
curl -X POST localhost:8090/api/mcp/servers -H 'Content-Type: application/json' \
  -d '{"name":"demo","command":"python","args":["mcp_demo_server.py"]}'
curl -X POST localhost:8090/api/mcp/servers/demo/connect
curl -X POST localhost:8090/api/mcp/tools/call -H 'Content-Type: application/json' \
  -d '{"server":"demo","tool":"now","arguments":{}}'
```

### Auth

Set `WORKBENCH_API_KEY` and every `/api/*` call must send `X-API-Key: <value>`.
`/docs` stays open so you can paste the key into Swagger's Authorize dialog.


---

## Harness layer (adapted from google-research/harness2)

[Harness²](https://github.com/google-research/harness2) defines a harness as the
eight editable components around a model and shows that refining them — not the
weights — is what turns a model into a working agent. This app implements the
component model and the guardrail/experience half of that idea; the recursive
search stays in the research code.

| Component | Here | What it does |
|---|---|---|
| c1 system prompt | `system_prompt` | base role instructions |
| c2 guardrails | `guardrails` + `rules` | rules text **and** machine checks: blocked-input regex, forbidden-output regex, `require_json`, `max_prompt_tokens` (auto-trim), tool allowlist |
| c3 memory | `memory` | preamble + stored facts + ranked lessons, injected with section headers |
| c4 tool guidance | `tool_guidance` | per-tool notes folded into the schema descriptions the model reads |
| c5 subagents | `subagents[]` | each one becomes a callable tool (`{"task": ...}`); runs as a nested, tool-free completion |
| c6 skills | `skills[]` | SKILL.md-style instruction files, applied when relevant |
| c7 scripts | `scripts[]` | named procedures the model is told it may follow |
| c8 plugins | `plugins[]` | behavioural hooks appended to every turn (tone, format, lint rules) |

Harnesses are stored in SQLite, bound per chat, and composable from the UI's
**Harness** tab, which also shows the token cost of each section, previews the
composed system prompt, and **probes** a candidate harness against the chat's last
turn so two variants can be compared before you bind one.

### Lessons (the experience library)

`cue / lesson / evidence / confidence` rows, keyword-ranked against each message
and injected into context. `POST /api/lessons/reflect` asks the model to extract a
transferable lesson from the last exchange — the useful half of Harness²'s
"harnessing-level recursion" without needing benchmarks.

### What is deliberately not here

- **Recursive Propose–Probe–Compose search** over candidate harnesses (`--k`,
  parallel/sequential) with an LLM judge and tree selection — that is a research
  loop over benchmark rollouts; a chat app has a human in the seat instead, so the
  probe is manual and the user composes.
- **Benchmark adapters** (JobBench / LAB / WorkBuddy) and grader-verdict leak
  checks — no graders exist here.
- **Executor substrates** (OpenCode / Codex) — the workbench *is* the executor and
  speaks to any OpenAI-compatible server.

### Endpoints

```
GET    /api/harnesses                     list + component schema
POST   /api/harnesses                     create
PATCH  /api/harnesses/{id}                update components
POST   /api/harnesses/{id}/duplicate
DELETE /api/harnesses/{id}
GET    /api/harnesses/{id}/preview?q=     composed system prompt + token report
POST   /api/chats/{id}/harness            bind a harness to a chat
POST   /api/harness/probe                 run the last user turn under a candidate
GET/POST/DELETE /api/lessons              experience library
POST   /api/lessons/reflect               extract a lesson from the last exchange
```

---

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `UPSTREAM` | `http://127.0.0.1:43440` (`host.docker.internal` in the image) | SGLang base URL |
| `WORKBENCH_PORT` | `8090` | published port (compose) |
| `WORKBENCH_DATA` | `./data` (`/data` in the image) | SQLite + settings volume |
| `WORKBENCH_API_KEY` | empty (off) | require `X-API-Key` on `/api/*` |
| `WORKBENCH_CORS_ORIGINS` | `*` | comma-separated allowed origins |
| `WORKBENCH_ROOT` | `/Users/kmail/sglang` | where `server.log` lives |
| `SGLANG_PYTHON` | `<root>/.venv/bin/python` | interpreter for model loading |
| `WORKBENCH_LOADER_URL` | empty | host loader for model switching from a container |
| `WORKBENCH_LOADER_TOKEN` | empty | `X-Loader-Token` sent to the loader |
| `LOADER_TOKEN` | empty | require `X-Loader-Token` on the loader |
| `HF_HOME` | `~/.cache/huggingface` | model cache root (mounted read-only in Docker) |

---

## Context window: the real ceiling

The UI exposes the context window, but this backend is OOM-sensitive on Apple
Silicon. Measured on an M4 Pro / 24 GB / MLX:

| Context | Prefill result |
|---|---|
| 8,192 | fine |
| 12,288 | fine (≈6.5 s for a 12k prompt) |
| 16,384 | `[METAL] Command buffer execution failed: Insufficient Memory` → server dies |

So the default is **12288** and the API rejects anything above **14336** with a 400.
On NVIDIA, raise it freely (`--mem-fraction-static 0.85`).

---

## Memory

`memory.py` measures the machine and sizes a model so it actually fits. Unified memory is
shared by macOS, Docker and SGLang, and both failure modes are decided at launch: too high a
`--mem-fraction-static` aborts the Metal allocator, too low leaves a KV pool of a few hundred
tokens. The plan is computed from real numbers — weight shards in the HF cache, KV cost per
token from the model's own `config.json`, and current available RAM:

```
GET /memory?model=Qwen/Qwen3-1.7B          # loader (host numbers)
GET /api/memory                            # workbench, proxied to the loader
```

The loader answers with `scope: "host"` (available RAM, swap, pressure, per-process RSS), the
live `served` block for the model that is actually running, and a plan only for a model you are
about to load; without a loader the workbench falls back to `scope: "container"`, which is the cgroup view
and worth knowing but is not what SGLang competes for. Example on this 24 GB M4 Pro:

```
3.78 GB weights + 10.50 GB KV of 14.88 GB allocated (mem-fraction 0.62)
pool ≈ 98,304 tokens, ctx 12,288
```

The plan is a starting point: the loader checks the real KV pool after the engine answers
`/health` and refits (mem-fraction, context, concurrency) if the pool is too small to serve a
conversation. A plan that looks generous against *total* memory is clamped by what is free right
now — that is how a 3B model at fraction 0.62 used to die with a Metal OOM.

## Layout and remembered state

The workspace is a real three-column layout, not a chat page with a drawer:

| viewport | layout |
|---|---|
| ≥ 1360 px | sidebar · chat · panel — the panel is a grid column with a drag handle (320–760 px) |
| 1200–1359 px | sidebar · chat, panel opens as an overlay with a dim scrim (click it to close) |
| < 1200 px | chat only, sidebar behind a hamburger, panel still overlays |

Drag either handle to resize; double-click a handle to reset it to its default. The density
button in the topbar switches comfortable/compact spacing.

Everything you change is remembered twice — `localStorage` for an instant restore on the next
paint, and `POST /api/ui-state` for durability across browsers, container rebuilds and other
machines on the LAN:

* open chat, scroll position per chat, composer draft per chat
* panel open/closed, active panel tab, sidebar width, panel width, density, theme
* model browser filters (source, task, query)

`GET /api/ui-state` returns the stored blob, so a fresh browser profile opens exactly where the
last one left off.

## Rendering

Answers are parsed with **marked** (CommonMark + GFM) and sanitised with **DOMPurify**;
code is highlighted by **highlight.js**. All three are vendored under `static/vendor/`,
so rendering works offline and inside the container — no CDN at runtime.

| content | how it renders |
|---|---|
| headings, lists, task lists, bold/italic/strike, links, images | native markdown |
| pipe tables | real `<table>` with alignment, zebra rows, sticky-free styling |
| fenced code | `hljs` tokens, language label, copy / save buttons |
| ```` ```mermaid ```` | rendered to inline SVG, sized from its viewBox, dark theme |
| ```` ```html ```` | sandboxed `<iframe sandbox="allow-scripts">` preview, auto-height via `postMessage` |
| reasoning | collapsible `details` block; `reasoning_content` deltas and inline ` thinking…</think>` both routed there |
| empty fences | skipped, so a lazy model cannot draw empty boxes |

Streaming re-renders are throttled to ~8 fps so parsing never blocks token delivery, and a
final render always runs when the stream ends.

Deep links: `/?chat=<id>` opens a specific chat, `/?menu=models` opens the model switcher,
`/?switch=org/model` starts a model switch, `/?ask=…` prefills the composer.

## What the UI gives you

- streaming chat with markdown + syntax highlighting, stop, regenerate, edit, fork, pin
- attachments (text/markdown/code, drag-and-drop or picker) folded into the prompt
- long-term memory store, keyword-ranked into the system prompt
- MCP servers (stdio) with tool discovery, enable/disable, and a model-driven tool loop
- model browser: downloaded / curated / mlx-community / HuggingFace search / local dirs
- load jobs with download progress, cancel, error surface with the log tail
- personas, prompt library, usage stats, server log viewer, command palette (⌘K)
- light/dark theme, deep links: `/?ask=…`, `/?panel=model|params|memory|prompts|mcp`

## Layout

```
workbench/
  app.py               FastAPI backend (API + SSE + MCP client + job runner)
  static/              index.html, app.css, app.js (no build step)
  Dockerfile           portable image (python:3.11-slim, non-root)
  docker-compose.yml   workbench + optional GPU sglang profile
  run-workbench.sh     host launcher
  loader_agent.py      host-side model loader (start/stop/job for containers)
  run-loader.sh        launcher for the loader
  verify-workbench.sh  end-to-end ad-hoc checks
  mcp_demo_server.py   tiny stdio MCP server used by the tests
```
