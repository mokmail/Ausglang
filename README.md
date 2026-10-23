# SGLang Workbench

A local LLM workbench over an SGLang (OpenAI-compatible) server: professional chat UI,
long-term memory, MCP tools with an agent harness, **recursive harness refinement**,
model switching, attachments, personas, prompt library and a documented FastAPI.

The workbench is **GPU-agnostic** — it speaks HTTP to SGLang, so the same image runs
on macOS, Windows and Linux while the Metal/CUDA stack stays where it works.

---

## Docker Compose

The image is GPU-agnostic: it talks to SGLang over the OpenAI API, so Metal/CUDA stays on the
host and the same container runs on macOS, Windows (WSL2) and Linux.

```bash
./start-workbench.sh          # writes .env for this host, starts engine + container
open http://localhost:8090
```

Or, once `.env` exists (copy `.env.example` to make one by hand):

```bash
docker compose up -d --build
docker compose logs -f workbench
docker compose down
```

**What the script does that plain compose cannot**: the container must reach the *host*, and
`host.docker.internal` resolves to the VM's own loopback on Rancher/Colima — the engine
connection is refused. The script detects the host's LAN address, writes `UPSTREAM` and
`WORKBENCH_LOADER_URL` into `.env`, and starts the engine when it is not already running.

| I want to… | command |
|---|---|
| start everything | `./start-workbench.sh` |
| start the container only (engine elsewhere) | `./start-workbench.sh --no-engine` |
| stop | `./start-workbench.sh down` |
| use a different port | `WORKBENCH_PORT=9000 ./start-workbench.sh` |
| SGLang in its own container (Linux + NVIDIA) | `docker compose --profile gpu up -d` |
| see what the container can reach | `curl -s localhost:8090/api/upstream` |

`/api/upstream` is the diagnostic to reach for when the UI says the server is down: it lists
every address probed, which one answered, whether the loader was found, and a hint when none
of them worked.

State lives in `./data` (bind mount, `WORKBENCH_DATA_DIR`): the SQLite database, artifacts and
settings. Copy that directory to back up or move the workspace; it is the only thing the
container writes. `HF_CACHE_DIR` is mounted read-only so the model browser can list models the
host already downloaded.

Images are built from source on start (`--build`), so a `git pull` followed by
`./start-workbench.sh` always runs the code you have.

## Quick start

```bash
./start-workbench.sh        # Docker + loader + engine + container, then prints the URLs
```

It resolves the host's LAN address before starting the container: on this VM setup
`host.docker.internal` resolves to the VM's own loopback and the engine connection is refused,
so the compose file's hostname default cannot be trusted. Re-run the script after a reboot or a
DHCP lease change.

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
./start-gpu.sh                     # recommended: reports pull/warm-up progress
# or:
MODEL=Qwen/Qwen3-1.7B docker compose --profile gpu up -d
UPSTREAM=http://sglang:30000 docker compose up -d workbench
```

`start-gpu.sh` streams image-pull progress from the Docker Engine API and writes
`data/engine-status.json`; the Workbench renders it as one banner across every tab:
**pulling image** (real %) → **downloading model** → **starting engine** → **ready**.
Model download progress is read from the HF cache, so it appears even when compose
was started by hand. The engine's lifecycle is owned by compose, so "Load model"
stays disabled (`MODEL=` in `.env` changes it); the banner still tells you what the
engine is doing.

The weights are ~8 GB: if the repo lives on a small partition, point
`HF_CACHE_DIR` in `.env` at a larger filesystem (for example
`/var/tmp/sglang-hf-cache`); both the engine and the Workbench mount it.

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
component model, the guardrail/experience half, and a local **Propose–Probe–Compose**
loop that runs against the current task instead of a benchmark rollout.

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

### Recursive refinement (Propose–Probe–Compose)

The **Recursive refinement** card in the Harness tab runs the task-level recursion
locally:

1. **Probe** — run the selected harness on the current task (last user turn, or a
   task you type) with tools disabled, so the harness is what changes.
2. **Propose** — ask the model for one or more harness edits as strict JSON:
   append/replace a section, add a skill/script/subagent/plugin, or add tool
   guidance. Edits that target the answer instead of the harness are rejected.
3. **Probe candidates** — run the same task under each candidate. A deterministic
   task-focus candidate is always included, so a weak model that writes prose
   instead of an edit still produces a usable contrast.
4. **Judge** — contrast the outputs without ground truth (fit, usability,
   guardrail flags) and pick a winner.
5. **Compose** — synthesize the next harness from the winning edits and the
   contrasts; invalid composer output falls back to the judged candidate.
6. **Reflect** — extract one transferable lesson into the experience library,
   closing the harnessing-level recursion for the next task.

`sequential` carries the refined harness and evidence through `k` steps;
`parallel` proposes `k` general and `k` domain-specific candidates, probes them
together, and composes once. Results (baseline, candidates, scores, lesson, and
the composed system prompt) are stored in `harness_runs` and can be adopted as a
new harness or adopted + bound to the chat. If the local model returns unusable
JSON, the loop records the fallback and still produces a validated result — it
never writes a malformed harness.

### Lessons (the experience library)

`cue / lesson / evidence / confidence` rows, keyword-ranked against each message
and injected into context. `POST /api/lessons/reflect` asks the model to extract a
transferable lesson from the last exchange; the recursive loop also writes one per
run.

### What is deliberately not here

- **Benchmark rollouts** (JobBench / LAB / WorkBuddy) and grader-verdict leak
  checks — the loop runs on your live chat task, not a benchmark with graders.
- **Executor substrates** (OpenCode / Codex) — the workbench *is* the executor and
  speaks to any OpenAI-compatible server.
- **Weight updates** — only the harness changes.

### Endpoints

```
GET    /api/harnesses                     list + component schema
POST   /api/harnesses                     create
PATCH  /api/harnesses/{id}                update components
POST   /api/harnesses/{id}/duplicate
DELETE /api/harnesses/{id}
GET    /api/harnesses/{id}/preview?q=     composed system prompt + token report
GET    /api/harnesses/{id}/composition    the mounted plugin tree (profile, rows, services)
GET    /api/harness/profiles              shipped profiles and their bundles
POST   /api/chats/{id}/harness            bind a harness to a chat
POST   /api/chats/{id}/compact            condense this chat's session now (/compact)
GET    /api/chats/{id}/session?tail=      durable session events (JSONL replay)
POST   /api/harness/probe                 run the last user turn under a candidate
POST   /api/harness/recursion             start Propose–Probe–Compose
GET    /api/harness/recursion/job         progress of the running refinement
POST   /api/harness/recursion/cancel      cancel the refinement
GET    /api/harness/runs                  recent refinement runs
GET    /api/harness/runs/{id}             one run: baseline, probes, verdict, result
POST   /api/harness/runs/{id}/adopt       save the refined harness (optional bind)
GET/POST/DELETE /api/lessons              experience library
POST   /api/lessons/reflect               extract a lesson from the last exchange
```

---

## PDF extraction (`pdfmcp`)

An MCP server that turns a PDF into **structured markdown plus a parallel
manifest**, and extracts the images into their own directory.

```bash
# register it with the running workbench
curl -X POST localhost:8090/api/mcp/servers -H 'Content-Type: application/json' \
  -d '{"name":"pdf","command":"python","args":["/abs/path/to/pdfmcp_server.py"]}'
curl -X POST localhost:8090/api/mcp/servers/pdf/connect
```

The server is registered automatically on startup when `pdfmcp/` is present.

### What you see in the composer

The input area carries a row of chips, one per registered MCP server, so the
answer to "what can this actually do?" is visible where you write the prompt:

```
● pdf 5 tools   ● demo 4 tools   manage
```

Click a chip to expand that server's tools with a one-line description and the
parameter signature (required parameters in plain brackets, optional ones in
square brackets):

```
extract_pdf   Extract text and images from a PDF into structured markdown + manifest.
              (pdf_path, [out_dir], [backend], [do_ocr], [extract_images], [name])
```

A green dot means connected, amber means registered but not connected, and a
dimmed chip means the server is switched off. The `tools N` button shows how
many tools are available; hovering it lists every server and tool, and says
plainly whether they are currently being sent to the model.

### What it produces

```
<out_dir>/<name>/
  document.md     structured markdown: headings, paragraphs, lists, tables
  manifest.json   the machine-readable twin of the markdown
  images/         one file per extracted image
```

`document.md` carries a `<!-- page: N -->` marker before each page and an
`<!-- id: t0042 | page: 7 | topic: 3.1 PDF backends -->` anchor before every
block, so any passage in the markdown can be traced back to the manifest.

`manifest.json` is the parallel index:

```jsonc
{
  "source": { "name": "paper.pdf", "sha256": "…", "page_count": 9 },
  "backend": "docling",
  "counts": { "text_blocks": 725, "images": 5, "topics": 25, "words": 7646 },
  "topics": [ { "topic": "3.1 PDF backends", "blocks": 20 } ],
  "texts": [
    {
      "id": "t0042",
      "page": 7,                    // 1-based source page
      "kind": "paragraph",
      "topic": "PDF backends",      // the section it belongs to
      "topic_path": ["Processing pipeline", "PDF backends"],
      "level": null,
      "bbox": [108.9, 719.5, 504.2, 706.1],
      "order": 42,                  // reading order
      "text": "…",
      "meta": { "undecodable": 0 }  // glyphs that could not be decoded
    }
  ],
  "images": [
    {
      "id": "i0002",
      "page": 3,                    // the page it came from
      "file": "i0002_p003.png",
      "file_relative": "images/i0002_p003.png",
      "width_px": 785, "height_px": 276,
      "bytes": 33545,
      "bbox": [109.75, 719.46, 502.24, 581.74],
      "topic": "Table Structure Recognition",
      "meta": { "dpi": 144, "caption": "…" }
    }
  ]
}
```

### Tools

| Tool | What it does |
|---|---|
| `extract_pdf` | full extraction: markdown + manifest + images |
| `extract_images` | images only, with the page each came from |
| `read_manifest` | read back and summarise a manifest |
| `list_extractions` | what has been extracted so far |
| `pdf_info` | metadata, page count, outline — writes nothing |

### Engines

| Backend | Structure | Needs |
|---|---|---|
| `docling` | best: heading hierarchy, tables, reading order, page provenance | `docling` (pulls torch) |
| `fallback` | text layout, PDF outline, embedded images | `pypdf` + `pypdfium2` only |

`auto` (the default) uses Docling when it is installed and falls back to
pypdf/pdfium otherwise — the fallback needs no ML and no model downloads, so
the server still works in a slim container. `fonttools` is required for
correct text decoding on PDFs with subsetted CFF fonts; without it those
documents come out as mojibake.

To enable the Docling backend, uncomment it in `requirements.txt` and rebuild.

### Transports

```bash
python pdfmcp_server.py             # stdio — what the workbench speaks
python pdfmcp_server.py --http      # streamable-http on 0.0.0.0:8500/mcp
```

The workbench talks to the server over stdio (it spawns it as a child
process), so no extra container is needed. For clients that expect an HTTP
MCP endpoint, the compose file ships an optional service:

```bash
docker compose --profile pdf up -d   # then: http://localhost:8510/mcp
```

Extractions are written to `PDFMCP_OUT` (default `data/pdf-extracts`), which
in the container is the `./data` bind mount — so they survive a rebuild.

---

## DeepSeek Harness (the loop that runs your chats)

The chat loop is a port of [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
(`dsh`), the plugin-composed agent harness: **everything is a plugin**, and a
composition is an ordered list of rows. The eight harness components above are
mounted *as plugins beside* the loop, so the harness you edit in the UI is the
actual composition the model ran under.

| dsh concept | Here |
|---|---|
| plugin kernel | `dsh/kernel.py` — services, typed events, reversible effects, deferred injection, scoped registration |
| profile / bundle / patch | `dsh/profiles.py` — `workbench`, `headless`, `sdk-minimal`; rows patched by id |
| session log | `dsh/session.py` — append-only JSONL events; `derive_messages()` is the model history |
| prompt assembly | `dsh/prompt.py` — ordered sections, `{{variables}}`, tool schemas, strict rendering |
| tool pipeline | `dsh/tools.py` — `tools/pre-execute` → guards → `tools/execute` → `tools/post-execute` → `tools/result` |
| agent loop | `dsh/loop.py` — turn/step machine, inbox, `agent/pre-step`, `agent/request`, streaming, barriers + parallel pool |
| compaction | `dsh/compaction.py` — pressure trigger, tool-result pruning, one durable summary, overflow recovery |
| components | `dsh/harness.py` — c1…c8, artifacts, MCP proxies, subagents, runtime context |
| bridge | `dsh/runtime.py` — harness → patch layers → mounted host → one turn |

A turn is the same shape as upstream:

```
turn/start → claim → assemble prompt + tool schemas → agent/pre-step
→ step/start → agent/request → system/message + user/message
→ derive + freeze → stream → assistant/message
→ tool/call → guarded pipeline → tool/result → step/end
→ (another step if tools owe one) → agent/turn-stopping → turn/end
```

Durable facts land in `data/sessions/<chat>.jsonl` (one session per chat, resumed
across turns), which is why `/api/chats/{id}/session` can replay exactly what the
model saw. Compaction writes a `compaction/replace` marker instead of editing
history, so the condensed prompt is reproducible.

The **Harness → Composition** card shows the mounted tree (row → state →
services); clicking a row prints its resolved config. `condense chat` runs
`/compact` immediately.

Tests (no network, scripted adapter):

```sh
python3 -m unittest discover -s tests -v
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
GET /api/memory/plan                       # workbench, proxied to the loader
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
  loader_core.py       download/spawn/OOM-retry loop shared by app.py and the loader
  static/              index.html, app.css, app.js (no build step)
  Dockerfile           portable image (python:3.11-slim, non-root)
  docker-compose.yml   workbench + optional GPU sglang profile
  run-workbench.sh     host launcher
  loader_agent.py      host-side model loader (start/stop/job for containers)
  run-loader.sh        launcher for the loader
  verify-workbench.sh  end-to-end ad-hoc checks
  mcp_demo_server.py   tiny stdio MCP server used by the tests
  pdfmcp/              PDF extraction engine + manifest writer + MCP server
  pdfmcp_server.py     launcher for the PDF MCP server (stdio or --http)
  tests/
    test_dsh.py
    test_pdfmcp.py
```
