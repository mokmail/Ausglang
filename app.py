#!/usr/bin/env python3
"""SGLang Workbench v2 — local LLM workbench over an SGLang/OpenAI-compatible server.

    python app.py --port 8090 [--host 0.0.0.0] [--upstream http://127.0.0.1:43440]

Capabilities
  chats      SQLite persistence, pinning, rename, delete, fork-at-message,
             regenerate, message editing, markdown/JSON export, full-text search
  models     catalog from HF / mlx-community / local dirs / HF cache with
             downloaded flags + sizes, background download with progress,
             validated load (job states: preparing → downloading → starting → ready/failed)
  server     start/stop with model + context window + mem fraction, live
             /get_server_info introspection, RSS, log tail
  memory     long-term facts ranked against each message and injected
  mcp        stdio JSON-RPC servers, tool discovery, manual + harness tool calls
  harness    model-driven tool loop with a JSON protocol any local model can follow,
             plus recursive Propose–Probe–Compose harness refinement
  prompts    reusable prompt library
  personas   saved system-prompt + sampling profiles
  settings   temperature/top_p/top_k/max_tokens/repeat penalty/seed/stop/json mode
  stats      per-day message and token accounting
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import harness2_lite as h2
from dsh import runtime as dsh_runtime
from dsh.llm import OpenAICompatibleAdapter
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("WORKBENCH_ROOT", "/Users/kmail/sglang"))
HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
DATA = Path(os.environ.get("WORKBENCH_DATA", str(HERE / "data")))
DATA.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA / "workbench.db"
SETTINGS_PATH = DATA / "settings.json"
MCP_PATH = DATA / "mcp_servers.json"
MODEL_CACHE = DATA / "models_cache.json"
DSH_SESSIONS = DATA / "sessions"
LEGACY_LOOP = os.environ.get("WORKBENCH_LEGACY_LOOP", "").strip().lower() in (
    "1", "true", "yes", "on")
SERVER_LOG = Path(os.environ.get("SGLANG_LOG", str(ROOT / "server.log")))

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:43440").rstrip("/")
UPSTREAM_CHOSEN: dict[str, Any] = {"url": UPSTREAM, "at": 0.0, "all_failed_at": 0.0}
PORT = int(os.environ.get("WORKBENCH_PORT", "8090"))
# Path to the python that can run `-m sglang.launch_server`. Empty/absent means
# model loading is disabled here (typical inside a container, where the GPU
# stack lives on the host) and the API returns a clear error instead of dying.
PY = os.environ.get("SGLANG_PYTHON") or str(ROOT / ".venv/bin/python")
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
# Per-chat sandbox where the model and the UI may write files.
ARTIFACTS_DIR = Path(os.environ.get("WORKBENCH_ARTIFACTS", str(DATA / "artifacts")))
# Optional host-side loader (loader_agent.py). Set when this app runs in a
# container: Metal cannot be driven from Linux, so start/stop/job are forwarded.
LOADER_URL = os.environ.get("WORKBENCH_LOADER_URL", "").rstrip("/")
LOADER_TOKEN = os.environ.get("WORKBENCH_LOADER_TOKEN", "")
API_VERSION = "2.0.0"
CTX_CEILING = 12288   # tested: 12k prefill passes, 16k Metal-OOMs on this box

DEFAULT_SETTINGS: dict[str, Any] = {
    "model": "Qwen/Qwen3-0.6B",
    "context_length": CTX_CEILING,
    "mem_fraction_static": 0.30,
    "max_running_requests": 4,
    "chunked_prefill_size": 1024,
    "mlx_cache_gb": 1.0,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 512,
    "repeat_penalty": 1.05,
    "seed": None,
    "stop": [],
    "json_mode": False,
    "system_prompt": "You are a helpful, precise assistant running locally on the user's Mac.",
    "enable_thinking": False,
    "memory_enabled": True,
    "tools_enabled": True,
    "max_tool_rounds": 3,
    "auto_title": False,
    "theme": "dark",
    "stream": True,
    "persona": None,
}

# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #



def ensure_demo_mcp() -> None:
    """Keep the bundled demo MCP server pointing at a command that exists HERE.

    The registry is shared between host and container deployments, so a host path
    like /Users/x/sglang/.venv/bin/python is meaningless inside the image. Repair
    it instead of shipping a dead example.
    """
    if not (HERE / "mcp_demo_server.py").exists():
        return
    servers = load_mcp()
    want_cmd, want_args = sys.executable, [str(HERE / "mcp_demo_server.py")]
    changed = False
    demo = next((x for x in servers if x.get("name") == "demo"), None)
    if demo is None:
        servers.append({"name": "demo", "command": want_cmd, "args": want_args, "env": {}, "enabled": True})
        changed = True
    else:
        cmd_ok = Path(demo.get("command") or "").exists()
        args_ok = all(Path(a).exists() for a in (demo.get("args") or []) if a.endswith(".py"))
        if not (cmd_ok and args_ok):
            demo["command"], demo["args"] = want_cmd, want_args
            changed = True
    if changed:
        save_mcp(servers)


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY, title TEXT, model TEXT,
                params TEXT, created REAL, updated REAL, pinned INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, chat_id TEXT, role TEXT, content TEXT,
                reasoning TEXT, tools TEXT, meta TEXT, created REAL,
                edited INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_msg_chat ON messages(chat_id, created);
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY, kind TEXT, text TEXT, created REAL
            );
            CREATE TABLE IF NOT EXISTS prompts (
                id TEXT PRIMARY KEY, title TEXT, body TEXT, created REAL
            );
            CREATE TABLE IF NOT EXISTS personas (
                id TEXT PRIMARY KEY, name TEXT, system_prompt TEXT,
                params TEXT, created REAL
            );
            CREATE TABLE IF NOT EXISTS harnesses (
                id TEXT PRIMARY KEY, name TEXT, components TEXT,
                created REAL, updated REAL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY, chat_id TEXT, name TEXT, path TEXT,
                size INTEGER, mime TEXT, created REAL
            );
            CREATE TABLE IF NOT EXISTS lessons (
                id TEXT PRIMARY KEY, domain TEXT, cue TEXT, lesson TEXT,
                evidence TEXT, confidence TEXT, created REAL
            );
            CREATE TABLE IF NOT EXISTS harness_runs (
                id TEXT PRIMARY KEY, harness_id TEXT, chat_id TEXT, task TEXT,
                mode TEXT, k INTEGER, state TEXT, step INTEGER, progress REAL,
                baseline TEXT, candidates TEXT, verdict TEXT, result TEXT,
                error TEXT, created REAL, finished REAL
            );
            """
        )
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(chats)")}
        if "pinned" not in cols:
            conn.execute("ALTER TABLE chats ADD COLUMN pinned INTEGER DEFAULT 0")
        mcols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
        if "edited" not in mcols:
            conn.execute("ALTER TABLE messages ADD COLUMN edited INTEGER DEFAULT 0")
        ccols = {r["name"] for r in conn.execute("PRAGMA table_info(chats)")}
        if "harness" not in ccols:
            conn.execute("ALTER TABLE chats ADD COLUMN harness TEXT")
        if not conn.execute("SELECT id FROM harnesses LIMIT 1").fetchone():
            hid = h2.new_id()
            conn.execute(
                "INSERT INTO harnesses (id,name,components,created,updated) VALUES (?,?,?,?,?)",
                (hid, h2.DEFAULT_HARNESS["name"], json.dumps(h2.DEFAULT_HARNESS), h2.now(), h2.now()),
            )


def load_settings() -> dict[str, Any]:
    s = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        try:
            s.update(json.loads(SETTINGS_PATH.read_text()))
        except Exception:
            pass
    return s


def save_settings(s: dict[str, Any]) -> None:
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2))
    tmp.replace(SETTINGS_PATH)


def load_mcp() -> list[dict[str, Any]]:
    if MCP_PATH.exists():
        try:
            return json.loads(MCP_PATH.read_text())
        except Exception:
            return []
    return []


def save_mcp(servers: list[dict[str, Any]]) -> None:
    MCP_PATH.write_text(json.dumps(servers, indent=2))


# --------------------------------------------------------------------------- #
# model catalog
# --------------------------------------------------------------------------- #

CURATED = [
    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-3-1b-it",
    "microsoft/Phi-3.5-mini-instruct", "HuggingFaceTB/SmolLM3-3B",
    "mlx-community/Qwen3-4B-4bit", "mlx-community/Llama-3.2-3B-Instruct-4bit",
]
HF_CACHE = HF_CACHE_DIR / "hub"


def cached_models() -> dict[str, float]:
    out: dict[str, float] = {}
    if not HF_CACHE.exists():
        return out
    for d in HF_CACHE.glob("models--*"):
        name = d.name.replace("models--", "").replace("--", "/")
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        out[name] = round(size / 2**30, 2)
    return out


def _cache_get(key: str, ttl: float = 600.0) -> Any | None:
    if not MODEL_CACHE.exists():
        return None
    try:
        data = json.loads(MODEL_CACHE.read_text())
    except Exception:
        return None
    hit = data.get(key)
    if hit and time.time() - hit.get("t", 0) < ttl:
        return hit.get("v")
    return None


def _cache_put(key: str, value: Any) -> None:
    data = {}
    if MODEL_CACHE.exists():
        try:
            data = json.loads(MODEL_CACHE.read_text())
        except Exception:
            data = {}
    data[key] = {"t": time.time(), "v": value}
    try:
        MODEL_CACHE.write_text(json.dumps(data))
    except Exception:
        pass


def hf_search(source: str, q: str, limit: int, task: str | None) -> list[dict[str, Any]]:
    """Blocking HF search — call via asyncio.to_thread."""
    from huggingface_hub import HfApi

    key = f"{source}|{q}|{limit}|{task}"
    hit = _cache_get(key)
    if hit is not None:
        return hit

    api = HfApi()
    kwargs: dict[str, Any] = {"limit": limit, "sort": "downloads", "direction": -1}
    if q:
        kwargs["search"] = q
    if source == "mlx":
        kwargs["author"] = "mlx-community"
    if task:
        kwargs["filter"] = task

    got = cached_models()
    out: list[dict[str, Any]] = []
    for m in api.list_models(**kwargs):
        tags = getattr(m, "tags", []) or []
        if source == "hf" and not any(
            t in tags for t in ("text-generation", "conversational", "mlx", "gguf")
        ):
            continue
        size = got.get(m.id)
        out.append({
            "id": m.id,
            "downloads": getattr(m, "downloads", 0) or 0,
            "likes": getattr(m, "likes", 0) or 0,
            "gated": bool(getattr(m, "gated", False)),
            "updated": str(getattr(m, "last_modified", "") or "")[:10],
            "downloaded": size is not None,
            "size_gb": size,
            "source": source,
        })
    _cache_put(key, out)
    return out


def model_detail(model_id: str) -> dict[str, Any]:
    """Size + gating for one repo, for the pre-load confirmation."""
    from huggingface_hub import HfApi

    key = f"detail|{model_id}"
    hit = _cache_get(key, ttl=900)
    if hit is not None:
        return hit
    info = HfApi().model_info(model_id, files_metadata=True)
    total = sum((getattr(s, "size", 0) or 0) for s in (info.siblings or []))
    detail = {
        "id": model_id,
        "gated": bool(getattr(info, "gated", False)),
        "size_gb": round(total / 2**30, 2),
        "downloaded": model_id in cached_models(),
        "tags": (getattr(info, "tags", []) or [])[:12],
        "updated": str(getattr(info, "last_modified", "") or "")[:10],
    }
    _cache_put(key, detail)
    return detail


# --------------------------------------------------------------------------- #
# MCP client (stdio, JSON-RPC 2.0)
# --------------------------------------------------------------------------- #


class McpStdioClient:
    def __init__(self, name: str, command: str, args: list[str], env: dict | None = None):
        self.name, self.command, self.args = name, command, args
        self.env = {**os.environ, **(env or {})}
        self.proc: subprocess.Popen | None = None
        self.tools: list[dict[str, Any]] = []
        self._id = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        self.proc = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.env, text=True, bufsize=1,
        )
        self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "sglang-workbench", "version": "2.0"},
        })
        self._notify("notifications/initialized", {})
        self.refresh_tools()

    def _send(self, obj: dict) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _notify(self, method: str, params: dict) -> None:
        with self._lock:
            self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _rpc(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        with self._lock:
            self._id += 1
            rid = self._id
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            deadline = time.time() + timeout
            assert self.proc and self.proc.stdout
            while time.time() < deadline:
                line = self.proc.stdout.readline()
                if not line:
                    err = ""
                    if self.proc.stderr:
                        try:
                            err = self.proc.stderr.read(400)
                        except Exception:
                            pass
                    raise RuntimeError(f"MCP server {self.name} closed stdout. {err}".strip())
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == rid:
                    if "error" in msg:
                        raise RuntimeError(msg["error"].get("message", "MCP error"))
                    return msg.get("result", {})
            raise TimeoutError(f"MCP {self.name} timed out on {method}")

    def refresh_tools(self) -> list[dict[str, Any]]:
        self.tools = self._rpc("tools/list", {}).get("tools", [])
        return self.tools

    def call_tool(self, tool: str, arguments: dict) -> Any:
        res = self._rpc("tools/call", {"name": tool, "arguments": arguments}, timeout=120)
        parts = [c.get("text", "") for c in res.get("content", [])
                 if isinstance(c, dict) and c.get("type") == "text"]
        return "\n".join(parts) if parts else json.dumps(res, ensure_ascii=False)

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass


MCP_CLIENTS: dict[str, McpStdioClient] = {}


def mcp_client(name: str) -> McpStdioClient:
    cfg = next((s for s in load_mcp() if s["name"] == name), None)
    if not cfg:
        raise HTTPException(404, f"unknown MCP server '{name}'")
    client = MCP_CLIENTS.get(name)
    if client is None:
        client = McpStdioClient(name, cfg["command"], cfg.get("args", []), cfg.get("env"))
        MCP_CLIENTS[name] = client
    if not client.tools:
        client.start()
    return client


def all_mcp_tools() -> list[dict[str, Any]]:
    out = []
    for cfg in load_mcp():
        if cfg.get("enabled") is False:
            continue
        try:
            for t in mcp_client(cfg["name"]).tools:
                out.append({"server": cfg["name"], **t})
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# server lifecycle (background job with progress)
# --------------------------------------------------------------------------- #


class LoadJob:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {
            "state": "idle", "model": None, "progress": 0.0,
            "message": "", "error": None, "started": None, "finished": None,
        }
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.cancel = threading.Event()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def _set(self, **kw: Any) -> None:
        with self.lock:
            self.state.update(kw)


JOB = LoadJob()


def _run(cmd: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess | None:
    """subprocess.run that survives slim images with no pgrep/ps/pkill."""
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


def loader_call(method: str, path: str, payload: dict | None = None) -> Any:
    headers = {"X-Loader-Token": LOADER_TOKEN} if LOADER_TOKEN else {}
    try:
        r = httpx.request(method, f"{LOADER_URL}{path}", json=payload, headers=headers, timeout=180)
    except Exception as e:
        raise HTTPException(502, f"loader unreachable at {LOADER_URL}: {type(e).__name__}: {e}")
    if r.status_code >= 400:
        detail = r.text[:400]
        raise HTTPException(r.status_code, detail)
    return r.json()


def _docker_gateway() -> str | None:
    """Default gateway inside the container — on plain Docker this is the host."""
    try:
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:
                fields = line.split()
                if len(fields) > 2 and fields[1] == "00000000":
                    ip = int(fields[2], 16)
                    return ".".join(str((ip >> shift) & 0xFF) for shift in (0, 8, 16, 24))
    except Exception:
        pass
    return None


def upstream_candidates() -> list[str]:
    """Every address the engine could plausibly live at, most explicit first.

    `host.docker.internal` is the documented answer on Docker Desktop, but it resolves to
    the VM's own loopback on some runners (Rancher, Colima), where the only route is the
    host's LAN address. Trying a list is what makes `docker compose up` work unedited.
    """
    urls: list[str] = [UPSTREAM]
    for extra in os.environ.get("WORKBENCH_UPSTREAMS", "").split(","):
        if extra.strip():
            urls.append(extra.strip().rstrip("/"))
    # compose always defines the variable (possibly empty), so empty counts as unset
    if not os.environ.get("WORKBENCH_UPSTREAMS", "").strip():
        urls.append("http://host.docker.internal:43440")
        gw = _docker_gateway()
        if gw:
            urls.append(f"http://{gw}:43440")
        urls.append("http://127.0.0.1:43440")
    seen: set[str] = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def _healthy(url: str, timeout: float = 1.5) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=timeout).status_code == 200
    except Exception:
        return False


def resolve_upstream(ttl: float = 20.0, force: bool = False) -> str:
    """Pick the first reachable candidate and remember it briefly."""
    global UPSTREAM
    now = time.time()
    if not force:
        if _healthy(UPSTREAM_CHOSEN["url"], 0.8) and now - UPSTREAM_CHOSEN["at"] < ttl * 3:
            return UPSTREAM_CHOSEN["url"]
        if now - UPSTREAM_CHOSEN["at"] < ttl:
            return UPSTREAM_CHOSEN["url"]
    for url in upstream_candidates():
        if _healthy(url):
            if url != UPSTREAM_CHOSEN["url"]:
                print(f"[upstream] using {url}", flush=True)
            UPSTREAM = UPSTREAM_CHOSEN["url"] = url
            UPSTREAM_CHOSEN["at"] = now
            return url
    UPSTREAM_CHOSEN["at"] = now
    return UPSTREAM_CHOSEN["url"]


def resolve_loader(ttl: float = 60.0, force: bool = False) -> str:
    """When no loader URL is configured, look for one next to the engine."""
    global LOADER_URL
    if LOADER_URL and not force:
        return LOADER_URL
    base = resolve_upstream()
    host = base.split("//", 1)[-1].rsplit(":", 1)[0]
    for candidate in (f"http://{host}:43441", "http://host.docker.internal:43441"):
        try:
            if httpx.get(f"{candidate}/healthz", timeout=1.5).status_code == 200:
                if candidate != LOADER_URL:
                    print(f"[loader] using {candidate}", flush=True)
                LOADER_URL = candidate
                return candidate
        except Exception:
            continue
    return LOADER_URL


def server_up() -> bool:
    return _healthy(resolve_upstream(), 1.2)


_INFO_CACHE: dict[str, Any] = {"t": 0.0, "v": {}}


def server_info(ttl: float = 2.0) -> dict[str, Any]:
    """Cheap snapshot: cached for `ttl` so a loading model cannot stall /api/state."""
    if time.time() - _INFO_CACHE["t"] < ttl and _INFO_CACHE["v"]:
        return dict(_INFO_CACHE["v"])
    base: dict[str, Any] = {"up": server_up(), "upstream": UPSTREAM}
    if not base["up"]:
        _INFO_CACHE.update(t=time.time(), v=dict(base))
        return base
    try:
        info = httpx.get(f"{UPSTREAM}/get_server_info", timeout=5).json()
        base.update(
            model=info.get("model_path"),
            context_length=info.get("context_length"),
            mem_fraction_static=info.get("mem_fraction_static"),
            max_total_num_tokens=info.get("max_total_num_tokens"),
            served_name=info.get("served_model_name"),
            engine_version=info.get("version"),
        )
    except Exception:
        pass
    pg = _run(["pgrep", "-f", "sglang.launch_server --model-path"])
    pid = (pg.stdout.split() if pg and pg.returncode == 0 else [])
    if pid:
        base["pid"] = int(pid[0])
        ps = _run(["ps", "-o", "rss=", "-p", pid[0]])
        rss = (ps.stdout.strip() if ps else "")
        if rss.isdigit():
            base["rss_gb"] = round(int(rss) / 2**20, 2)
    _INFO_CACHE.update(t=time.time(), v=dict(base))
    return base


# ---- engine provisioning --------------------------------------------------- #
# The engine may run outside this container (compose `gpu` profile, host loader,
# another box). There is no local job to poll then, so this synthesizes one from
# what is observable here: the upstream health check, partial blobs in the
# read-only HF cache, and an optional status file written by start-gpu.sh.

ENGINE_MODEL = os.environ.get("WORKBENCH_ENGINE_MODEL", "").strip()
ENGINE_STATUS_PATH = DATA / "engine-status.json"
_ENGINE_DOWN_SINCE: float | None = None
_HF_TOTAL: dict[str, tuple[float, int | None]] = {}
_HF_RATE: dict[str, tuple[float, int, float | None]] = {}


def _engine_file_status() -> dict[str, Any] | None:
    """Fresh status written by the host launcher (start-gpu.sh), if any."""
    try:
        raw = json.loads(ENGINE_STATUS_PATH.read_text())
    except Exception:
        return None
    try:
        updated = float(raw.get("updated") or 0)
    except (TypeError, ValueError):
        return None
    if time.time() - updated > 900:      # stale: no launcher is running
        return None
    raw["age_s"] = round(time.time() - updated, 1)
    return raw


def _hf_cache_bytes(model: str) -> dict[str, int] | None:
    """Bytes already in the HF cache for `model`, including `.incomplete` blobs."""
    base = HF_CACHE_DIR / "hub" / ("models--" + model.replace("/", "--")) / "blobs"
    if not base.is_dir():
        return None
    done = partial = files = 0
    newest = 0.0
    try:
        entries = list(base.iterdir())
    except OSError:
        return None
    for p in entries:
        try:
            if not p.is_file():
                continue
            st = p.stat()
        except OSError:
            continue
        files += 1
        done += st.st_size
        if p.name.endswith(".incomplete"):
            partial += st.st_size
            newest = max(newest, st.st_mtime)
    return {"done": done, "partial": partial, "files": files, "newest": newest}


def _hf_repo_total(model: str) -> int | None:
    """Sum of the repo's file sizes, so a partial download gets a real percent."""
    hit = _HF_TOTAL.get(model)
    if hit and time.time() - hit[0] < (600 if hit[1] else 60):
        return hit[1]
    total: int | None = None
    try:
        r = httpx.get(f"https://huggingface.co/api/models/{model}",
                      params={"blobs": "true"}, timeout=4.0)
        if r.status_code == 200:
            sizes = [int(s.get("size") or 0) for s in r.json().get("siblings") or []]
            if sum(sizes) > 0:
                total = sum(sizes)
    except Exception:
        pass
    _HF_TOTAL[model] = (time.time(), total)
    return total


def _hf_rate(model: str, done: int) -> float | None:
    """Bytes/second over the last sample, smoothed across polls."""
    now = time.time()
    prev = _HF_RATE.get(model)
    rate: float | None = None
    if prev and now - prev[0] >= 1.5:
        rate = (done - prev[1]) / (now - prev[0]) if done >= prev[1] else None
    elif prev:
        rate = prev[2]
    _HF_RATE[model] = (now, done, rate)
    return rate


def engine_status() -> dict[str, Any]:
    """One object the UI can render while the engine is being provisioned.

    Phases: ready → pulling → downloading → starting → waiting (and failed).
    `progress` is a real percentage only while downloading or when the host
    launcher reports one; the UI shows an indeterminate sweep otherwise.
    """
    global _ENGINE_DOWN_SINCE
    info = server_info()
    up = bool(info.get("up"))
    now = time.time()
    model = (ENGINE_MODEL or info.get("model") or load_settings().get("model") or "").strip()
    base: dict[str, Any] = {
        "url": UPSTREAM,
        "model": model or None,
        "reachable": up,
    }
    if up:
        _ENGINE_DOWN_SINCE = None
        _HF_RATE.pop(model, None)
        return {**base, "phase": "ready", "active": False, "progress": 100,
                "message": "engine ready", "elapsed_s": None, "hint": None}

    if _ENGINE_DOWN_SINCE is None:
        _ENGINE_DOWN_SINCE = now
    elapsed = round(now - _ENGINE_DOWN_SINCE, 1)
    cached = _hf_cache_bytes(model) if model else None
    launcher = _engine_file_status()

    # A partial blob is the most useful signal: real bytes and a real percent.
    # Ignore partials that stopped changing long ago (an aborted download).
    stale_partial = cached and cached["partial"] > 0 and \
        not launcher and now - cached["newest"] > 900
    if cached and cached["partial"] > 0 and not stale_partial:
        total = _hf_repo_total(model)
        pct = min(100.0, 100.0 * cached["done"] / total) if total else None
        return {**base, "phase": "downloading", "active": True, "progress": pct,
                "bytes_done": cached["done"], "bytes_total": total,
                "rate_bps": _hf_rate(model, cached["done"]),
                "message": f"downloading {model}", "elapsed_s": elapsed,
                "hint": "the model weights are being written to the HF cache"}

    if launcher and launcher.get("phase") in ("pulling", "starting", "failed"):
        phase = str(launcher["phase"])
        return {**base, "phase": phase, "active": phase != "failed",
                "progress": launcher.get("progress"),
                "message": launcher.get("message") or {
                    "pulling": "pulling engine image",
                    "starting": "starting the engine",
                }.get(phase, phase),
                "elapsed_s": elapsed, "hint": launcher.get("hint")}

    if cached and cached["files"]:
        return {**base, "phase": "starting", "active": True, "progress": None,
                "bytes_done": cached["done"],
                "message": "weights cached — warming up the engine",
                "elapsed_s": elapsed,
                "hint": "SGLang is loading the weights onto the GPU"}

    # An engine that is merely switched off is not a pull in progress. Keep the
    # banner for the first 15 minutes (the normal provisioning window), then fall
    # back to the status pill so it does not nag forever.
    if elapsed > 900:
        return {**base, "phase": "offline", "active": False, "progress": None,
                "message": "engine offline", "elapsed_s": elapsed,
                "hint": "start it with ./start-gpu.sh or point UPSTREAM at a running server"}

    return {**base, "phase": "waiting", "active": True, "progress": None,
            "message": "waiting for the engine", "elapsed_s": elapsed,
            "hint": ("the first start pulls the Docker image and model; "
                     "this can take several minutes")}


def _download(model: str) -> None:
    from huggingface_hub import snapshot_download

    detail = model_detail(model)
    expected = max(detail["size_gb"] * 2**30, 1)
    before = cached_models().get(model, 0.0) * 2**30
    JOB._set(state="downloading", message=f"fetching {model}", progress=0.0,
             bytes_total=int(expected), bytes_done=int(before))

    done: dict[str, Any] = {"err": None}

    def work() -> None:
        try:
            snapshot_download(repo_id=model, max_workers=4)
        except Exception as e:
            done["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while t.is_alive():
        if JOB.cancel.is_set():
            JOB._set(state="cancelled", message="cancelled by user")
            raise RuntimeError("cancelled")
        t.join(1.5)
        d = HF_CACHE / ("models--" + model.replace("/", "--"))
        now = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0.0
        pct = min(99.0, max(0.0, (now - before) / max(expected - before, 1) * 100))
        JOB._set(progress=round(pct, 1), message=f"downloading {pct:.0f}%",
                 bytes_done=int(now), bytes_total=int(expected))
    if done["err"]:
        raise RuntimeError(done["err"])
    JOB._set(progress=100.0, message="download complete")


def _spawn(model: str, cfg: dict[str, Any]) -> None:
    cmd = [
        PY, "-m", "sglang.launch_server",
        "--model-path", model,
        "--host", cfg.get("host", "0.0.0.0"),
        "--port", str(cfg.get("port", 43440)),
        "--mem-fraction-static", str(cfg["mem_fraction_static"]),
        "--max-running-requests", str(cfg["max_running_requests"]),
        "--context-length", str(cfg["context_length"]),
        "--chunked-prefill-size", str(cfg["chunked_prefill_size"]),
    ]
    env = {
        **os.environ,
        "SGLANG_USE_MLX": "1",
        "SGLANG_MLX_CACHE_LIMIT_GB": str(cfg.get("mlx_cache_gb", 1.0)),
        "SGLANG_MLX_CLEAR_CACHE_STEPS": "64",
    }
    log = open(SERVER_LOG, "ab")
    log.write(f"\n===== workbench load {time.strftime('%Y-%m-%d %H:%M:%S')} {model} =====\n".encode())
    log.flush()
    JOB.proc = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=str(ROOT), env=env)


def _log_tail(n: int = 25) -> str:
    if not SERVER_LOG.exists():
        return ""
    lines = SERVER_LOG.read_text(errors="replace").replace("\r", "\n").splitlines()
    return "\n".join(lines[-n:])


OOM_MARKERS = ("Insufficient Memory", "OutOfMemory", "out of memory")


def _oom_retry(payload: dict[str, Any], tail: str) -> dict[str, Any] | None:
    if not any(m.lower() in tail.lower() for m in OOM_MARKERS) or payload.get("_oom_retried"):
        return None
    ctx = int(payload.get("context_length") or CTX_CEILING)
    smaller = max(4096, ctx // 2)
    return {**payload, "context_length": smaller, "_oom_retried": True} if smaller < ctx else None


def load_worker(payload: dict[str, Any]) -> None:
    model = payload["model"]
    settings = load_settings()
    cfg = {**settings, **{k: v for k, v in payload.items() if v is not None}}
    JOB.cancel.clear()
    JOB._set(state="preparing", model=model, progress=0.0, message="checking model",
             error=None, started=time.time(), finished=None)
    try:
        detail = model_detail(model)
        if not detail["downloaded"]:
            _download(model)          # the old server keeps answering meanwhile
        else:
            JOB._set(progress=100.0, message="already in HF cache")

        if server_up():
            JOB._set(state="stopping", message="stopping the running server")
            subprocess.run(["pkill", "-f", "sglang.launch_server --model-path"], capture_output=True)
            for _ in range(45):
                if not server_up():
                    break
                time.sleep(1)
            subprocess.run(["pkill", "-9", "-f", "sglang.launch_server --model-path"],
                           capture_output=True)
            time.sleep(2)

        JOB._set(state="starting", message="starting sglang", progress=100.0)
        _spawn(model, cfg)
        deadline = time.time() + 600
        while time.time() < deadline:
            if JOB.cancel.is_set():
                subprocess.run(["pkill", "-9", "-f", "sglang.launch_server --model-path"],
                               capture_output=True)
                JOB._set(state="cancelled", message="cancelled by user")
                return
            if server_up():
                settings.update(
                    model=model,
                    context_length=cfg["context_length"],
                    mem_fraction_static=cfg["mem_fraction_static"],
                    max_running_requests=cfg["max_running_requests"],
                    mlx_cache_gb=cfg.get("mlx_cache_gb", 1.0),
                )
                save_settings(settings)
                _INFO_CACHE.update(t=0.0, v={})   # drop the stale pre-load snapshot
                JOB._set(state="ready", message=f"{model} ready", finished=time.time())
                return
            if JOB.proc and JOB.proc.poll() is not None:
                tail = _log_tail(40)
                retry = _oom_retry(payload, tail)
                if retry:
                    JOB._set(state="starting", message=f"out of memory — retrying with ctx {retry['context_length']}")
                    return load_worker(retry)
                raise RuntimeError("server exited during startup:\n" + tail)
            time.sleep(2)
        raise RuntimeError("timed out waiting for /health:\n" + _log_tail(20))
    except Exception as e:
        JOB._set(state="failed", error=str(e)[:4000], message="load failed", finished=time.time())


def start_load(payload: dict[str, Any]) -> dict[str, Any]:
    if LOADER_URL:
        return loader_call("POST", "/load", payload)
    if not Path(PY).exists():
        raise HTTPException(
            501,
            "model loading is disabled in this deployment: no SGLang interpreter at "
            f"{PY}. Run the host-side loader (loader_agent.py / run-loader.sh) and set "
            "WORKBENCH_LOADER_URL, or start SGLang yourself and point UPSTREAM at it.",
        )
    if JOB.snapshot()["state"] in ("preparing", "downloading", "starting", "stopping"):
        raise HTTPException(409, "a load is already in progress")
    if not payload.get("model"):
        raise HTTPException(400, "model required")
    ctx = int(payload.get("context_length") or load_settings()["context_length"])
    if ctx > 14336:
        raise HTTPException(
            400,
            f"context_length {ctx} is above the tested ceiling ({CTX_CEILING} works, 16384 dies with a "
            "Metal OOM on this Mac). Lower it before loading.",
        )
    JOB.thread = threading.Thread(target=load_worker, args=(payload,), daemon=True)
    JOB.thread.start()
    return {"accepted": True, "model": payload["model"]}


def stop_server() -> dict[str, Any]:
    JOB.cancel.set()
    if not shutil.which("pkill"):
        return {"stopped": False,
                "error": "this deployment cannot signal processes (no pkill); stop SGLang where it runs"}
    _run(["pkill", "-f", "sglang.launch_server --model-path"])
    for _ in range(40):
        if not server_up():
            break
        time.sleep(1)
    _run(["pkill", "-9", "-f", "sglang.launch_server --model-path"])
    JOB._set(state="idle", message="stopped", progress=0.0)
    return {"stopped": True}


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #


def memory_context(query: str, limit: int = 6) -> str:
    with db() as conn:
        rows = conn.execute("SELECT text FROM memory ORDER BY created DESC LIMIT 200").fetchall()
    if not rows:
        return ""
    words = {w for w in re.findall(r"[a-z0-9]{3,}", (query or "").lower())}
    scored = []
    for r in rows:
        tw = {w for w in re.findall(r"[a-z0-9]{3,}", r["text"].lower())}
        scored.append((len(words & tw), r["text"]))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    picked = [t for s, t in scored[:limit] if s > 0] or [t for _, t in scored[:3]]
    return "\n".join(f"- {p}" for p in picked) if picked else ""


# --------------------------------------------------------------------------- #
# tool harness
# --------------------------------------------------------------------------- #

TOOL_PROTOCOL = """
You can use tools. Available tools are listed below as JSON schemas.

To call a tool, reply with ONLY a fenced json block in this exact shape:
```json
{"tool": "<tool_name>", "arguments": { ... }}
```
Wait for the observation before writing the final answer. Never invent observations.
Do not wrap the JSON in prose. If no tool is needed, answer normally.
""".strip()


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": t["name"],
            "server": t["server"],
            "description": (t.get("description") or "")[:400],
            "input_schema": t.get("inputSchema") or {},
        }
        for t in all_mcp_tools()
    ]


def extract_tool_call(text: str, known: set[str]) -> dict[str, Any] | None:
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates += re.findall(r'(\{[^{}]*"tool"\s*:.*?\})', text, re.S)
    for blob in candidates:
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        name = obj.get("tool") or obj.get("name")
        if isinstance(name, str) and name in known:
            args = obj.get("arguments") or obj.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"input": args}
            return {"tool": name, "arguments": args if isinstance(args, dict) else {}}
    return None


async def run_delegate(harness: dict[str, Any], args: dict, params: dict[str, Any]) -> str:
    """c5: run one named subagent as a nested, tool-free completion."""
    name = str(args.get("name") or "")
    task = str(args.get("task") or "")
    sub = next((x for x in (harness.get("subagents") or []) if x.get("name") == name), None)
    if not sub:
        return f"no subagent named {name!r}; available: " + ", ".join(
            x.get("name", "?") for x in (harness.get("subagents") or []))
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)) as c:
            r = await c.post(f"{UPSTREAM}/v1/chat/completions", json={
                "model": params.get("model"),
                "messages": [
                    {"role": "system", "content": (sub.get("system") or "You are a focused subagent.") + "\n\nAnswer the delegated task concisely."},
                    {"role": "user", "content": task},
                ],
                "max_tokens": int(params.get("max_tokens", 512)),
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False},
            })
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"subagent error: {type(e).__name__}: {e}"


async def run_tool(server: str, tool: str, arguments: dict) -> str:
    try:
        return await asyncio.to_thread(mcp_client(server).call_tool, tool, arguments)
    except Exception as e:
        return f"tool error: {e}"


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# API schema (pydantic) — drives /docs, /redoc, /openapi.json
# --------------------------------------------------------------------------- #


class ChatAttachment(BaseModel):
    name: str = Field(examples=["notes.md"])
    text: str = Field(examples=["# Notes"])


class ChatRequest(BaseModel):
    chat_id: str = Field(examples=["a1b2c3d4e5f6"])
    message: str = Field(default="", description="User text (optional when regenerate=true)")
    regenerate: bool = Field(default=False, description="Re-run the last user turn")
    attachments: list[ChatAttachment] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class SettingsPatch(BaseModel):
    model_config = {"extra": "allow"}
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0, le=200)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    repeat_penalty: float | None = Field(default=None, ge=1, le=2)
    system_prompt: str | None = None
    memory_enabled: bool | None = None
    tools_enabled: bool | None = None
    enable_thinking: bool | None = None
    json_mode: bool | None = None
    seed: int | None = None
    stop: list[str] | None = None


class ServerStartRequest(BaseModel):
    model: str = Field(examples=["Qwen/Qwen3-0.6B"])
    context_length: int = Field(default=12288, ge=1024, le=14336,
                                description="Tested ceiling here is 12288; 16384 dies with a Metal OOM")
    mem_fraction_static: float = Field(default=0.30, ge=0.1, le=0.8)
    max_running_requests: int = Field(default=4, ge=1, le=64)
    chunked_prefill_size: int = Field(default=1024, ge=256, le=8192)
    mlx_cache_gb: float = Field(default=1.0, ge=0.25, le=16)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=43440)


class ChatNew(BaseModel):
    title: str | None = None
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ChatPatch(BaseModel):
    title: str | None = None
    pinned: bool | None = None
    params: dict[str, Any] | None = None


class ForkRequest(BaseModel):
    message_id: str | None = Field(default=None, description="Fork after this message; omit for a full copy")


class MessageEdit(BaseModel):
    content: str = Field(examples=["Corrected text"])


class MemoryCreate(BaseModel):
    text: str = Field(examples=["The user's NUC is at 192.168.1.57"])
    kind: str = Field(default="fact")


class PromptCreate(BaseModel):
    title: str = Field(examples=["Code review"])
    body: str = Field(examples=["Review this code for race conditions."])


class PersonaCreate(BaseModel):
    name: str = Field(examples=["Terse engineer"])
    system_prompt: str = Field(examples=["Answer in short, technical sentences."])
    params: dict[str, Any] = Field(default_factory=dict)


class McpServerCreate(BaseModel):
    name: str = Field(examples=["demo"])
    command: str = Field(examples=["/usr/bin/python3"])
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class McpToggle(BaseModel):
    enabled: bool = True


class McpToolCall(BaseModel):
    server: str = Field(examples=["demo"])
    tool: str = Field(examples=["now"])
    arguments: dict[str, Any] = Field(default_factory=dict)


TAGS = [
    {"name": "system", "description": "Health, version, runtime state, settings."},
    {"name": "chat", "description": "Streaming completions, memory injection, MCP tool harness."},
    {"name": "chats", "description": "Conversation storage: list, create, rename, pin, fork, export."},
    {"name": "models", "description": "Model catalog: HF, mlx-community, local folders, HF cache."},
    {"name": "server", "description": "SGLang process lifecycle with download/start progress."},
    {"name": "memory", "description": "Long-term facts injected into the system prompt."},
    {"name": "prompts", "description": "Reusable prompt library."},
    {"name": "personas", "description": "Saved system-prompt + sampling profiles."},
    {"name": "harness", "description": "Eight-component harness (c1..c8): system prompt, guardrails, memory, tool guidance, subagents, skills, scripts, plugins — plus probes and the lesson library."},
    {"name": "mcp", "description": "stdio MCP servers, tool discovery, manual invocation."},
    {"name": "artifacts", "description": "Files the model or user writes into a chat workspace."},
    {"name": "stats", "description": "Token and message accounting."},
]

app = FastAPI(
    title="SGLang Workbench API",
    version=API_VERSION,
    description=(
        "Local LLM workbench over an SGLang (OpenAI-compatible) server.\n\n"
        "**Streaming**: `POST /api/chat` returns `text/event-stream` with `start`, `delta`, "
        "`reasoning`, `tool`, `error`, `done` events.\n\n"
        "**Harness**: with MCP tools enabled the model gets a JSON tool protocol; calls are "
        "executed and observations fed back for up to `max_tool_rounds`.\n\n"
        "**Auth**: set `WORKBENCH_API_KEY` to require `X-API-Key` on `/api/*`."
    ),
    openapi_tags=TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("WORKBENCH_CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("WORKBENCH_API_KEY", "").strip()


@app.middleware("http")
async def api_key_guard(request, call_next):
    if API_KEY and request.url.path.startswith("/api/"):
        if request.headers.get("x-api-key") != API_KEY:
            from fastapi.responses import JSONResponse as _JSON
            return _JSON({"detail": "invalid or missing X-API-Key"}, status_code=401)
    return await call_next(request)


@app.get("/api/upstream/check", tags=["system"], summary="Probe an SGLang base URL for reachability")
def api_upstream_check(url: str = Query(..., description="e.g. http://host.docker.internal:43440")) -> dict[str, Any]:
    """Useful when the workbench runs in a container: Docker Desktop maps the host to
    ``host.docker.internal``, while Rancher/colima setups may only be reachable via a
    LAN address or the default gateway."""
    try:
        r = httpx.get(url.rstrip("/") + "/health", timeout=4.0)
        return {"url": url, "reachable": r.status_code == 200, "status": r.status_code}
    except Exception as e:
        return {"url": url, "reachable": False, "error": f"{type(e).__name__}: {e}"}


@app.get("/api/version", tags=["system"], summary="API, engine and upstream versions")
def api_version() -> dict[str, Any]:
    return {"api": API_VERSION, "engine": server_info().get("engine_version"), "upstream": UPSTREAM}


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def chat_artifact_dir(chat_id: str) -> Path:
    d = ARTIFACTS_DIR / re.sub(r"[^A-Za-z0-9_.-]", "_", chat_id or "scratch")
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_artifact_name(name: str) -> str:
    """No traversal, no absolute paths, no separators — a plain file name."""
    base = Path(str(name or "file.txt").replace("\\", "/")).name
    base = re.sub(r"[^\w.\-]+", "_", base).strip("._") or "file.txt"
    return base[:120]


def save_artifact(chat_id: str, name: str, content: str | bytes, mime: str = "text/plain") -> dict[str, Any]:
    d = chat_artifact_dir(chat_id)
    fname = safe_artifact_name(name)
    path = d / fname
    data = content.encode() if isinstance(content, str) else content
    path.write_bytes(data)
    aid = h2.new_id()
    guess = "text/plain"
    low = fname.lower()
    for ext, m in ((".pdf", "application/pdf"), (".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                   (".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
                   (".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                   (".py", "text/x-python"), (".js", "text/javascript"), (".json", "application/json"),
                   (".md", "text/markdown"), (".html", "text/html"), (".css", "text/css"),
                   (".csv", "text/csv"), (".sh", "text/x-shellscript"), (".txt", "text/plain")):
        if low.endswith(ext):
            guess = m
            break
    with db() as conn:
        conn.execute(
            "INSERT INTO artifacts (id,chat_id,name,path,size,mime,created) VALUES (?,?,?,?,?,?,?)",
            (aid, chat_id, fname, str(path), len(data), mime or guess, h2.now()),
        )
    return {"id": aid, "name": fname, "size": len(data), "mime": mime or guess,
            "chat_id": chat_id, "path": str(path)}


TEXTISH = (".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml",
           ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".sh", ".sql", ".rs", ".go",
           ".java", ".c", ".cpp", ".h", ".log", ".xml", ".ini", ".conf", ".env")


def _xml_text(xml: bytes) -> str:
    """Strip tags from OOXML parts, keeping paragraph/row boundaries as newlines."""
    raw = xml.decode("utf-8", errors="replace")
    raw = raw.replace("</w:p>", "\n").replace("</a:p>", "\n").replace("</w:tr>", "\n")
    raw = raw.replace("</si>", "\n").replace("</c>", "\t")
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = (raw.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
              .replace("&quot;", '"').replace("&apos;", "'"))
    return re.sub(r"\n{3,}", "\n\n", raw).strip()


def extract_document(name: str, data: bytes) -> tuple[str, dict[str, Any]]:
    """Return (text, meta) for PDF / DOCX / PPTX / XLSX / plain text. No network, no OCR."""
    low = (name or "").lower()
    meta: dict[str, Any] = {"kind": "text"}
    if low.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise HTTPException(500, "PDF support missing: pypdf is not installed in this deployment")
        reader = PdfReader(io.BytesIO(data))
        pages = [(pg.extract_text() or "") for pg in reader.pages]
        text = "\n\n".join(pages).strip()
        meta = {"kind": "pdf", "pages": len(pages)}
        if not text:
            raise HTTPException(422, "this PDF has no text layer (scanned image?) — no OCR here")
        return text, meta
    if low.endswith((".docx", ".pptx", ".xlsx")):
        import zipfile

        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            if low.endswith(".docx"):
                text = _xml_text(z.read("word/document.xml"))
                meta = {"kind": "docx", "parts": 1}
            elif low.endswith(".pptx"):
                slides = sorted(n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n))
                text = "\n\n".join(_xml_text(z.read(n)) for n in slides)
                meta = {"kind": "pptx", "slides": len(slides)}
            else:
                text = _xml_text(z.read("xl/sharedStrings.xml")) if "xl/sharedStrings.xml" in names else ""
                meta = {"kind": "xlsx", "sheets": len([n for n in names if n.startswith("xl/worksheets/")])}
        return text.strip(), meta
    if low.endswith(TEXTISH) or not low.count("."):
        return data.decode("utf-8", errors="replace"), {"kind": "text"}
    raise HTTPException(
        415,
        f"unsupported file type: {name}. Send PDF, DOCX, PPTX, XLSX or plain text "
        "(scanned PDFs need OCR, which this box does not run).",
    )


def artifact_rows(chat_id: str | None = None) -> list[dict[str, Any]]:
    with db() as conn:
        if chat_id:
            rows = conn.execute("SELECT * FROM artifacts WHERE chat_id=? ORDER BY created DESC", (chat_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM artifacts ORDER BY created DESC").fetchall()
    return [dict(r) for r in rows]


def run_artifact_tool(chat_id: str, tool: str, args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """Native tools: write_file / read_file / list_files inside the chat sandbox."""
    if tool == "write_file":
        name = args.get("name") or "file.txt"
        content = args.get("content")
        if content is None:
            return "error: write_file needs `content`", None
        art = save_artifact(chat_id, name, str(content))
        return f"wrote {art['name']} ({art['size']} bytes) in this chat's workspace", art
    if tool == "read_file":
        name = safe_artifact_name(args.get("name") or "")
        path = chat_artifact_dir(chat_id) / name
        if not path.exists():
            return f"error: no file named {name}", None
        return path.read_text(errors="replace")[:20000], None
    if tool == "list_files":
        files = [{"name": a["name"], "size": a["size"]} for a in artifact_rows(chat_id)]
        return json.dumps(files), None
    return f"error: unknown artifact tool {tool}", None


def list_harnesses() -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM harnesses ORDER BY created").fetchall()
    out = []
    for r in rows:
        comps = h2.normalize(json.loads(r["components"] or "{}"))
        out.append({**comps, "id": r["id"], "name": r["name"],
                    "created": r["created"], "updated": r["updated"]})
    return out


def get_harness(harness_id: str | None, chat_id: str | None = None) -> dict[str, Any]:
    """Resolve the harness for a chat: explicit id, the chat's binding, else the first one."""
    with db() as conn:
        if not harness_id and chat_id:
            row = conn.execute("SELECT harness FROM chats WHERE id=?", (chat_id,)).fetchone()
            harness_id = row["harness"] if row else None
        if harness_id:
            row = conn.execute("SELECT * FROM harnesses WHERE id=?", (harness_id,)).fetchone()
        else:
            row = conn.execute("SELECT * FROM harnesses ORDER BY created LIMIT 1").fetchone()
    if not row:
        return {"id": None, **h2.normalize(None)}
    comps = h2.normalize(json.loads(row["components"] or "{}"))
    return {**comps, "id": row["id"], "name": row["name"]}


def save_harness(harness_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    current = get_harness(harness_id)
    merged = h2.normalize({**current, **patch})
    with db() as conn:
        conn.execute("UPDATE harnesses SET name=?, components=?, updated=? WHERE id=?",
                     (merged.get("name") or "Harness", json.dumps(merged), h2.now(), harness_id))
    return {**merged, "id": harness_id}


def lessons_for(query: str) -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM lessons ORDER BY created DESC LIMIT 200").fetchall()
    return h2.lesson_context([dict(r) for r in rows], query)


# ---- recursive refinement: Propose–Probe–Compose --------------------------- #


class HarnessRunJob:
    """Small background job so the recursion survives a slow local model."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.state: dict[str, Any] = {
            "state": "idle", "run_id": None, "harness_id": None, "mode": None, "k": 0,
            "step": 0, "progress": 0.0, "message": "", "error": None,
            "started": None, "finished": None, "result": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def _set(self, **kw: Any) -> None:
        with self.lock:
            self.state.update(kw)


H2_JOB = HarnessRunJob()
H2_ACTIVE = ("proposing", "probing", "judging", "composing", "reflecting")


class H2Cancelled(Exception):
    pass


def _h2_check_cancel() -> None:
    if H2_JOB.cancel.is_set():
        raise H2Cancelled()


def _h2_row_update(run_id: str, **fields: Any) -> None:
    allowed = {"state", "step", "progress", "baseline", "candidates", "verdict",
               "result", "error", "finished"}
    clean = {k: v for k, v in fields.items() if k in allowed}
    if not clean:
        return
    for k, v in list(clean.items()):
        if isinstance(v, (dict, list)):
            clean[k] = json.dumps(v, ensure_ascii=False)
    cols = ", ".join(f"{k}=?" for k in clean)
    with db() as conn:
        conn.execute(f"UPDATE harness_runs SET {cols} WHERE id=?", (*clean.values(), run_id))


def _h2_job(state: str, *, step: int = 0, progress: float = 0.0,
            message: str = "", error: str | None = None,
            result: dict[str, Any] | None = None) -> None:
    H2_JOB._set(state=state, step=step, progress=round(progress, 1),
                message=message, error=error, result=result)


async def _h2_complete(messages: list[dict[str, str]], *,
                       max_tokens: int = 500, temperature: float = 0.25) -> str:
    """One tool-free completion used by the harness editor/judge/composer."""
    settings = load_settings()
    body = {
        "model": settings.get("model"),
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{resolve_upstream()}/v1/chat/completions", json=body)
    if r.status_code != 200:
        raise RuntimeError(f"upstream {r.status_code}: {r.text[:400]}")
    data = r.json()
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


async def _h2_probe(harness: dict[str, Any], task: str, params: dict[str, Any],
                    cid: str, name: str) -> dict[str, Any]:
    """Run the same task under one harness, with no tools (isolate the harness)."""
    mem = memory_context(task) if params.get("memory_enabled") else ""
    lessons = lessons_for(task) if params.get("memory_enabled") else []
    system, report = h2.compose_system(
        harness, lessons=lessons, memory_text=mem,
        json_mode=bool(params.get("json_mode")),
    )
    messages = ([{"role": "system", "content": system}] if system.strip() else [])
    messages.append({"role": "user", "content": task})
    started = time.time()
    text = await _h2_complete(messages, max_tokens=int(params.get("probe_tokens") or 420),
                              temperature=0.35)
    return {
        "id": cid,
        "name": name,
        "harness": harness,
        "text": text,
        "elapsed_s": round(time.time() - started, 2),
        "report": report,
        "guardrails": h2.check_output(harness, text),
    }


async def _h2_propose(harness: dict[str, Any], task: str, *,
                      angle: str, step: int, budget: int,
                      lessons: list[dict[str, Any]]) -> dict[str, Any]:
    raw = await _h2_complete(
        h2.proposal_messages(harness, task, lessons=lessons, angle=angle,
                             step=step, budget=budget),
        max_tokens=520, temperature=0.45,
    )
    parsed = h2.parse_json(raw)
    edits = h2.edits_from_payload(parsed)
    candidate, applied, skipped = h2.apply_edits(harness, edits, report=True)
    rationale = (parsed.get("rationale") if isinstance(parsed, dict) else "") or ""
    if not applied:
        edits = h2.heuristic_edits(task)
        candidate, applied, skipped = h2.apply_edits(harness, edits, report=True)
        rationale = (rationale + " · " if rationale else "") + \
            "model proposal was unusable; applied a deterministic task-focus edit"
    return {"angle": angle, "raw": raw, "rationale": rationale,
            "edits": edits, "applied": applied, "skipped": skipped,
            "harness": candidate}


def _h2_score_values(score: dict[str, Any]) -> float:
    def num(key: str) -> float:
        try:
            return max(0.0, min(10.0, float(score.get(key) or 0)))
        except (TypeError, ValueError):
            return 0.0
    # New schema is one score; the older three-axis schema still works.
    if score.get("score") is not None:
        return round(num("score"), 2)
    if score.get("total") is not None:
        return round(num("total"), 2)
    return round(num("fit") * 0.5 + num("actionability") * 0.3 + num("safety") * 0.2, 2)


def _h2_attach_scores(candidates: list[dict[str, Any]],
                      verdict: dict[str, Any]) -> str | None:
    scores = verdict.get("scores") if isinstance(verdict.get("scores"), list) else []
    best: tuple[float, str] | None = None
    for c in candidates:
        found = next((s for s in scores if str(s.get("id")) == str(c.get("id"))), {})
        c["score"] = _h2_score_values(found)
        c["note"] = str(found.get("note") or "")
        if best is None or c["score"] > best[0]:
            best = (c["score"], str(c.get("id")))
    winner = str(verdict.get("winner") or "")
    if winner.lower() in ("baseline", "base", "none"):
        verdict["winner"] = "baseline"
        return None
    if not any(winner == str(c.get("id")) for c in candidates):
        winner = best[1] if best else ""
    verdict["winner"] = winner
    return winner or None


async def _h2_judge(task: str, baseline: dict[str, Any],
                    candidates: list[dict[str, Any]]) -> dict[str, Any]:
    raw = await _h2_complete(h2.judge_messages(task, baseline, candidates),
                             max_tokens=450, temperature=0.15)
    parsed = h2.parse_json(raw)
    if isinstance(parsed, list):
        parsed = {"scores": parsed}
    if not isinstance(parsed, dict):
        parsed = {}

    raw_scores = parsed.get("scores")
    rows: list[dict[str, Any]] = []
    if isinstance(raw_scores, dict):
        for cid, val in raw_scores.items():
            if isinstance(val, dict):
                rows.append({"id": str(cid), **val})
            elif isinstance(val, (int, float)):
                rows.append({"id": str(cid), "score": val})
    elif isinstance(raw_scores, list):
        for s in raw_scores:
            if not isinstance(s, dict):
                continue
            cid = s.get("id") or s.get("candidate") or s.get("name")
            if cid is None:
                continue
            rows.append({
                "id": str(cid),
                "score": s.get("score", s.get("total")),
                "fit": s.get("fit"),
                "actionability": s.get("actionability"),
                "safety": s.get("safety"),
                "note": s.get("note") or s.get("rationale") or s.get("reason") or "",
            })
    # Some models return {"c1": 8, "c2": 6} at the top level.
    if not rows:
        for c in candidates:
            val = parsed.get(str(c.get("id")))
            if isinstance(val, (int, float)):
                rows.append({"id": str(c.get("id")), "score": val})
    known = {str(c.get("id")) for c in candidates} | {"baseline"}
    # Local models often copy the example id ("c1"). If none of the ids match,
    # preserve the judge's ordering and assign scores to the real candidates.
    if rows and not any(str(r.get("id")) in known for r in rows):
        for i, r in enumerate(rows):
            if i < len(candidates):
                r["id"] = str(candidates[i].get("id"))
    rows = [r for r in rows if str(r.get("id")) in known]
    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for r in rows:
        rid = str(r.get("id"))
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        deduped.append(r)
    rows = deduped
    if not rows:
        rows = [{"id": str(c.get("id")), "score": 6,
                 "note": "judge produced no parseable scores"} for c in candidates]
    for c in candidates:
        if not any(str(r.get("id")) == str(c.get("id")) for r in rows):
            rows.append({"id": str(c.get("id")), "score": 5,
                         "note": "not scored by the judge"})

    winner = str(parsed.get("winner") or parsed.get("best") or "").strip()
    normalized = {
        "scores": rows,
        "winner": winner,
        "rationale": str(parsed.get("rationale") or parsed.get("reason") or
                         "judge compared the probes without ground truth")[:800],
        "raw": raw,
    }
    return normalized


async def _h2_compose(base: dict[str, Any], task: str, baseline: dict[str, Any],
                      candidates: list[dict[str, Any]], verdict: dict[str, Any]) -> dict[str, Any]:
    if str(verdict.get("winner") or "").lower() in ("baseline", "base"):
        return {"harness": base, "edits": [], "applied": 0,
                "rationale": "baseline judged best; no edit is justified", "raw": ""}
    raw = await _h2_complete(
        h2.compose_messages(base, task, baseline, candidates, verdict),
        max_tokens=650, temperature=0.3,
    )
    parsed = h2.parse_json(raw)
    edits = h2.edits_from_payload(parsed)
    if edits:
        candidate, applied, _ = h2.apply_edits(base, edits, report=True)
        if applied:
            return {"harness": candidate, "edits": edits, "applied": applied,
                    "rationale": (parsed.get("rationale") if isinstance(parsed, dict) else "") or "",
                    "raw": raw}
    # No composed edits: fall back to the judged winner, not to the baseline.
    winner = str(verdict.get("winner") or "")
    chosen = next((c for c in candidates if str(c.get("id")) == winner), None)
    fallback = (chosen or {}).get("harness") or base
    fallback_edits = (chosen or {}).get("edits") or []
    if json.dumps(h2.normalize(fallback), sort_keys=True) == json.dumps(h2.normalize(base), sort_keys=True):
        fallback_edits = h2.heuristic_edits(task)
        fallback, applied, _ = h2.apply_edits(base, fallback_edits, report=True)
        if not applied:
            fallback_edits = []
    return {"harness": fallback, "edits": fallback_edits,
            "applied": len(fallback_edits),
            "rationale": "composer returned no valid edits; kept the strongest usable probe",
            "raw": raw}


async def _h2_reflect(task: str, before: dict[str, Any], after: dict[str, Any],
                      evidence: str, domain: str, *, changed: bool = True) -> dict[str, Any]:
    if not changed:
        lesson = {
            "cue": "tasks where the baseline already performs well",
            "lesson": ("Keep the harness unchanged unless a probe shows a concrete gain; "
                       "do not add task-focus boilerplate."),
            "evidence": "no candidate exceeded the baseline on this task",
            "confidence": "medium",
        }
        lid = h2.new_id()
        with db() as conn:
            conn.execute(
                "INSERT INTO lessons (id,domain,cue,lesson,evidence,confidence,created) "
                "VALUES (?,?,?,?,?,?,?)",
                (lid, domain, lesson["cue"], lesson["lesson"], lesson["evidence"],
                 lesson["confidence"], h2.now()),
            )
        lesson["id"] = lid
        return lesson
    raw = await _h2_complete(h2.reflect_messages(task, before, after, evidence),
                             max_tokens=240, temperature=0.2)
    parsed = h2.parse_json(raw)
    if not isinstance(parsed, dict):
        parsed = {}
    lesson_text = str(parsed.get("lesson") or "").strip()
    useful = bool(re.search(
        r"(?i)\b(task|deliverable|check|verify|restate|state|format|plan|assumption|output|step|brief)\b",
        lesson_text,
    ))
    if len(lesson_text) < 20 or not useful:
        lesson_text = ("Add a task-focus block that names the deliverable and the check "
                       "that proves it before answering.")
        parsed = {"cue": "tasks with a concrete deliverable",
                  "lesson": lesson_text,
                  "evidence": evidence[:400],
                  "confidence": "medium"}
    lesson = {
        "cue": str(parsed.get("cue") or "")[:300],
        "lesson": lesson_text[:500],
        "evidence": str(parsed.get("evidence") or evidence[:400])[:800],
        "confidence": str(parsed.get("confidence") or "medium")[:16],
    }
    if lesson["lesson"]:
        lid = h2.new_id()
        with db() as conn:
            conn.execute(
                "INSERT INTO lessons (id,domain,cue,lesson,evidence,confidence,created) "
                "VALUES (?,?,?,?,?,?,?)",
                (lid, domain, lesson["cue"], lesson["lesson"], lesson["evidence"],
                 lesson["confidence"], h2.now()),
            )
        lesson["id"] = lid
    return lesson


async def _h2_loop(run_id: str, payload: dict[str, Any]) -> None:
    task = str(payload["task"])
    mode = str(payload.get("mode") or "sequential")
    budget = int(payload.get("k") or 1)
    params = {**load_settings(), **(payload.get("params") or {})}
    params["tools_enabled"] = False          # isolate the harness, not MCP state
    initial = payload["harness"]
    base = initial
    lessons = lessons_for(task) if params.get("memory_enabled") else []
    domain = str(payload.get("domain") or "general")

    _h2_job("probing", step=0, progress=8, message="running the baseline")
    _h2_row_update(run_id, state="probing", step=0, progress=8)
    baseline = await _h2_probe(base, task, params, "baseline", "baseline")
    baseline0 = baseline
    _h2_check_cancel()
    _h2_job("probing", step=0, progress=20, message="baseline captured")
    _h2_row_update(run_id, baseline=baseline)

    all_candidates: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    verdict: dict[str, Any] = {}

    if mode == "parallel":
        angles = ["general"] * budget + ["domain"] * budget
        _h2_job("proposing", step=0, progress=26,
                message=f"proposing {len(angles)} harness candidates")
        proposals = []
        for i, angle in enumerate(angles):
            _h2_check_cancel()
            try:
                proposals.append(await _h2_propose(base, task, angle=angle, step=1,
                                                   budget=budget, lessons=lessons))
            except H2Cancelled:
                raise
            except Exception as e:
                proposals.append({"angle": angle, "edits": h2.heuristic_edits(task),
                                  "harness": h2.apply_edits(base, h2.heuristic_edits(task)),
                                  "rationale": f"proposal fallback: {e}"})
            _h2_job("proposing", step=i + 1, progress=26 + 8 * (i + 1) / len(angles),
                    message=f"candidate {i + 1}/{len(angles)}")
        _h2_job("probing", step=0, progress=44, message="probing candidates")
        # A deterministic task-focus candidate is always included: it keeps the
        # loop effective when a small model proposes prose instead of an edit.
        hedits = h2.heuristic_edits(task)
        hharness = h2.apply_edits(base, hedits)
        if not any(json.dumps(h2.normalize(p["harness"]), sort_keys=True) ==
                   json.dumps(h2.normalize(hharness), sort_keys=True) for p in proposals):
            proposals.append({"angle": "deterministic", "edits": hedits,
                              "harness": hharness,
                              "rationale": "task-focus fallback candidate"})
        for i, prop in enumerate(proposals):
            _h2_check_cancel()
            cid = f"c{i + 1}"
            probe = await _h2_probe(prop["harness"], task, params, cid,
                                    f"{prop.get('angle')} · {prop.get('rationale', '')[:40]}")
            probe["angle"] = prop.get("angle")
            probe["edits"] = prop.get("edits")
            all_candidates.append(probe)
            _h2_job("probing", step=i + 1, progress=44 + 14 * (i + 1) / len(proposals),
                    message=f"probe {i + 1}/{len(proposals)}")
        _h2_check_cancel()
        _h2_job("judging", step=0, progress=63, message="contrasting probes")
        verdict = await _h2_judge(task, baseline, all_candidates)
        winner = _h2_attach_scores(all_candidates, verdict)
        _h2_row_update(run_id, candidates=all_candidates, verdict=verdict)
        _h2_check_cancel()
        _h2_job("composing", step=0, progress=78, message="composing the refined harness")
        composed = await _h2_compose(base, task, baseline, all_candidates, verdict)
        final = composed["harness"]
        steps.append({"step": 1, "mode": "parallel", "candidates": [
            c["id"] for c in all_candidates], "winner": winner,
            "edits": composed.get("edits") or [],
            "rationale": composed.get("rationale") or ""})
    else:
        for step in range(1, budget + 1):
            _h2_check_cancel()
            angle = "domain" if step % 2 == 0 else "general"
            _h2_job("proposing", step=step,
                    progress=26 + 40 * (step - 1) / max(1, budget),
                    message=f"step {step}/{budget}: proposing edits")
            prop = await _h2_propose(base, task, angle=angle, step=step,
                                     budget=budget, lessons=lessons)
            _h2_check_cancel()
            _h2_job("probing", step=step,
                    progress=36 + 40 * (step - 1) / max(1, budget),
                    message=f"step {step}/{budget}: probing candidates")
            step_probes: list[dict[str, Any]] = []
            cid = f"s{step}"
            probe = await _h2_probe(prop["harness"], task, params, cid,
                                    f"{angle} · step {step}")
            probe["angle"] = angle
            probe["edits"] = prop.get("edits")
            step_probes.append(probe)
            hedits = h2.heuristic_edits(task)
            hharness = h2.apply_edits(base, hedits)
            if json.dumps(h2.normalize(hharness), sort_keys=True) != \
                    json.dumps(h2.normalize(prop["harness"]), sort_keys=True):
                _h2_job("probing", step=step,
                        progress=40 + 40 * (step - 1) / max(1, budget),
                        message=f"step {step}/{budget}: probing deterministic fallback")
                hprobe = await _h2_probe(hharness, task, params, f"s{step}h",
                                         "deterministic task focus")
                hprobe["angle"] = "deterministic"
                hprobe["edits"] = hedits
                step_probes.append(hprobe)
            all_candidates.extend(step_probes)
            _h2_job("judging", step=step,
                    progress=46 + 40 * (step - 1) / max(1, budget),
                    message=f"step {step}/{budget}: contrasting behaviours")
            verdict = await _h2_judge(task, baseline, step_probes)
            winner = _h2_attach_scores(step_probes, verdict)
            _h2_check_cancel()
            _h2_job("composing", step=step,
                    progress=58 + 40 * (step - 1) / max(1, budget),
                    message=f"step {step}/{budget}: composing")
            composed = await _h2_compose(base, task, baseline, step_probes, verdict)
            chosen = next((p for p in step_probes if p["id"] == winner), baseline if not winner else step_probes[0])
            steps.append({"step": step, "angle": angle,
                          "candidates": [p["id"] for p in step_probes],
                          "winner": winner, "score": chosen.get("score"),
                          "edits": composed.get("edits") or [],
                          "rationale": composed.get("rationale") or "",
                          "proposal": prop.get("rationale") or ""})
            base = composed["harness"]
            baseline = chosen             # next step compares against this behaviour
            _h2_row_update(run_id, step=step, progress=58 + 40 * step / max(1, budget),
                           candidates=all_candidates, verdict=verdict)
        final = base

    _h2_check_cancel()
    changed = json.dumps(h2.normalize(final), sort_keys=True) != \
        json.dumps(h2.normalize(initial), sort_keys=True)
    _h2_job("reflecting", step=budget, progress=93,
            message="extracting a transferable lesson")
    evidence = "\n\n".join(
        f"{c['id']}: {c.get('text', '')[:600]}" for c in all_candidates
    )
    lesson = await _h2_reflect(task, initial, final, evidence, domain, changed=changed)
    _h2_check_cancel()

    system_preview, report = h2.compose_system(final, lessons=lessons,
                                               memory_text=memory_context(task),
                                               json_mode=bool(params.get("json_mode")))
    result = {
        "harness": final,
        "base_harness": initial,
        "baseline": baseline0,
        "candidates": all_candidates,
        "verdict": verdict,
        "steps": steps,
        "lesson": lesson,
        "mode": mode,
        "k": budget,
        "task": task,
        "changed": changed,
        "system_preview": system_preview,
        "report": report,
    }
    _h2_row_update(run_id, state="ready", step=budget, progress=100,
                   baseline=baseline0, candidates=all_candidates, verdict=verdict,
                   result=result, finished=h2.now())
    _h2_job("ready", step=budget, progress=100, message="refined harness ready",
            result=result)
    H2_JOB._set(finished=time.time())


def _h2_worker(run_id: str, payload: dict[str, Any]) -> None:
    try:
        asyncio.run(_h2_loop(run_id, payload))
    except H2Cancelled:
        H2_JOB._set(state="cancelled", message="cancelled", finished=time.time())
        _h2_row_update(run_id, state="cancelled", finished=h2.now())
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        H2_JOB._set(state="failed", error=msg[:2000], message="refinement failed",
                    finished=time.time())
        _h2_row_update(run_id, state="failed", error=msg[:2000], finished=h2.now())


def effective_tools(harness: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    """MCP tools after the harness' allowlist + tool guidance, plus the delegate tool."""
    if not params.get("tools_enabled"):
        return []
    tools = h2.filter_tools(tool_schemas(), harness["rules"].get("tool_allowlist") or [])
    tools = h2.apply_tool_guidance(tools, harness.get("tool_guidance") or {})
    # native file tools, always available (they are what makes artifacts possible)
    tools.extend([
        {
            "name": "write_file",
            "server": "artifacts",
            "description": "Create or overwrite a file in this chat's workspace. Use for code, notes, data, any deliverable.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "file name, e.g. solution.py"},
                    "content": {"type": "string", "description": "full file contents"},
                },
                "required": ["name", "content"],
            },
        },
        {
            "name": "read_file",
            "server": "artifacts",
            "description": "Read back a file from this chat's workspace.",
            "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
        {
            "name": "list_files",
            "server": "artifacts",
            "description": "List the files already written in this chat.",
            "input_schema": {"type": "object", "properties": {}},
        },
    ])

    # c5: every subagent is callable by its own name (models naturally emit that),
    # plus a generic "delegate" escape hatch.
    subs = [x for x in (harness.get("subagents") or []) if x.get("name")]
    for sub in subs:
        if not sub.get("name"):
            continue
        tools.append({
            "name": sub["name"],
            "server": "harness",
            "description": sub.get("description") or f"Delegate a subtask to the {sub['name']} subagent.",
            "input_schema": {
                "type": "object",
                "properties": {"task": {"type": "string", "description": "What the subagent should do"}},
                "required": ["task"],
            },
        })
    if subs:
        # The system prompt also documents a generic delegate tool; keep it real so
        # models that follow that prose (rather than the per-name schemas) still work.
        tools.append({
            "name": "delegate",
            "server": "harness",
            "description": "Delegate a subtask to a named subagent and return its answer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": [x["name"] for x in subs]},
                    "task": {"type": "string"},
                },
                "required": ["name", "task"],
            },
        })
    return tools


def build_messages(chat_id: str, user_text: str, settings: dict[str, Any],
                   attachments: list[dict] | None = None,
                   harness: dict[str, Any] | None = None) -> tuple[list[dict], dict[str, Any]]:
    """Compose the prompt from the chat's harness (c1..c8) + memory + lessons."""
    harness = harness or get_harness(None, chat_id)
    mem = memory_context(user_text) if settings.get("memory_enabled") else ""
    lessons = lessons_for(user_text) if settings.get("memory_enabled") else []

    system, report = h2.compose_system(
        harness, lessons=lessons, memory_text=mem, json_mode=bool(settings.get("json_mode"))
    )

    tools = effective_tools(harness, settings)
    if tools:
        system += f"\n\n{h2_tool_protocol()}\n\nTools:\n{json.dumps(tools, ensure_ascii=False)}"

    with db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id=? AND role IN ('user','assistant') "
            "ORDER BY created",
            (chat_id,),
        ).fetchall()

    msgs = [{"role": "system", "content": system}] if system.strip() else []
    for r in rows:
        if r["content"]:
            msgs.append({"role": r["role"], "content": r["content"]})
    if attachments:
        for a in attachments:
            name = (a.get("name") or "attachment").strip()
            text = (a.get("text") or "")[:100_000]
            msgs.append({"role": "user", "content": f"[attached file: {name}]\n```\n{text}\n```"})

    max_tokens = int(harness["rules"].get("max_prompt_tokens") or 0)
    if max_tokens:
        msgs, trimmed = h2.budget_trim(msgs, max_tokens)
        report["trimmed_messages"] = trimmed
    report["harness_id"] = harness.get("id")
    report["harness_name"] = harness.get("name")
    report["tools"] = [t.get("name") for t in tools]
    return msgs, report


def h2_tool_protocol() -> str:
    return TOOL_PROTOCOL


# --------------------------------------------------------------------------- #
# DeepSeek Harness composition — everything below `/api/chat` is a plugin tree
# --------------------------------------------------------------------------- #


def dsh_session_root() -> Path:
    DSH_SESSIONS.mkdir(parents=True, exist_ok=True)
    return DSH_SESSIONS


def dsh_rows_for(chat_id: str, harness: dict[str, Any], params: dict[str, Any],
                 query: str, holder: dict[str, Any]) -> list[dict[str, Any]]:
    """Patch a workbench harness onto the `workbench` profile."""
    rules = harness.get("rules") or {}
    memory_on = bool(params.get("memory_enabled"))
    lessons = lessons_for(query) if (memory_on and rules.get("include_lessons", True)) else []
    mcp = tool_schemas() if params.get("tools_enabled") else []

    async def call_tool(server: str, tool: str, args: dict[str, Any]) -> str:
        return await run_tool(server, tool, args)

    def artifact_write(name: str, content: str) -> dict[str, Any]:
        _, art = run_artifact_tool(chat_id, "write_file", {"name": name, "content": content})
        return {"name": name, "artifact": art}

    def artifact_read(name: str) -> str:
        return run_artifact_tool(chat_id, "read_file", {"name": name})[0]

    def artifact_list() -> list[dict[str, Any]]:
        return [{"name": a["name"], "size": a.get("size")} for a in artifact_rows(chat_id)]

    async def subagent_runner(**kw: Any) -> str:
        host = holder.get("host")
        if host is None:
            return "error: the composition is not mounted"
        return await dsh_runtime.run_subagent(host, params=params, **kw)

    return dsh_runtime.build_rows(
        harness, params,
        chat_id=chat_id,
        session_root=dsh_session_root(),
        memory_text=memory_context(query) if memory_on else "",
        lessons=lessons,
        tool_schemas=mcp,
        tool_call=call_tool,
        artifact_write=artifact_write,
        artifact_read=artifact_read,
        artifact_list=artifact_list,
        subagent_runner=subagent_runner,
        model=params.get("model") or "",
        provider="sglang",
        workspace=str(DATA),
    )


def dsh_mount(rows: list[dict[str, Any]], holder: dict[str, Any]) -> Any:
    adapter = OpenAICompatibleAdapter(
        UPSTREAM,
        api_key=os.environ.get("WORKBENCH_UPSTREAM_KEY", ""),
        timeout=(10.0, 600.0),
    )
    host = dsh_runtime.build_host(rows, adapter=adapter)
    holder["host"] = host
    return host


@app.get("/api/state", tags=["system"], summary="Full runtime snapshot")
def api_state() -> dict[str, Any]:
    s = load_settings()
    with db() as conn:
        counts = {
            "chats": conn.execute("SELECT COUNT(*) c FROM chats").fetchone()["c"],
            "memory": conn.execute("SELECT COUNT(*) c FROM memory").fetchone()["c"],
            "prompts": conn.execute("SELECT COUNT(*) c FROM prompts").fetchone()["c"],
            "personas": conn.execute("SELECT COUNT(*) c FROM personas").fetchone()["c"],
            "artifacts": conn.execute("SELECT COUNT(*) c FROM artifacts").fetchone()["c"],
        }
    return {
        "server": server_info(),
        "job": JOB.snapshot(),
        "engine": engine_status(),
        "settings": s,
        "ctx_ceiling": CTX_CEILING,
        "can_load_models": bool(LOADER_URL) or Path(PY).exists(),
        "loader_url": LOADER_URL or None,
        "tools": tool_schemas() if s.get("tools_enabled") else [],
        "mcp": [
            {"name": c["name"],
             "enabled": c.get("enabled", True),
             "connected": bool(MCP_CLIENTS.get(c["name"]) and MCP_CLIENTS[c["name"]].tools)}
            for c in load_mcp()
        ],
        "counts": counts,
        "harnesses": [{"id": h["id"], "name": h["name"]} for h in list_harnesses()],
        "harness": {k: v for k, v in get_harness(None, None).items() if k != "id"},
        "harness_components": [{"key": k, "label": lbl} for k, lbl in h2.COMPONENTS],
        "loop": "legacy" if LEGACY_LOOP else "dsh",
        "profile": dsh_runtime.PROFILE_NAME,
    }


@app.get("/api/ui-state", tags=["system"], summary="UI state remembered by this workspace")
def api_ui_state() -> dict[str, Any]:
    """Layout, filters, drafts and the last opened chat.

    The browser keeps its own copy in localStorage (instant, no round-trip); this
    is the durable copy that survives a new browser profile, a container rebuild
    or a different machine on the LAN.
    """
    return {"state": load_settings().get("ui_state") or {}}


@app.post("/api/ui-state", tags=["system"], summary="Remember UI state")
def api_ui_state_save(payload: dict = Body(...)) -> dict[str, Any]:
    s = load_settings()
    current = dict(s.get("ui_state") or {})
    current.update({k: v for k, v in payload.items() if v is not None})
    s["ui_state"] = current
    save_settings(s)
    return {"saved": True, "keys": sorted(current)}


@app.get("/api/settings", tags=["system"], summary="Saved settings")
def api_settings() -> dict[str, Any]:
    return load_settings()


@app.post("/api/settings", tags=["system"], summary="Patch saved settings")
def api_save_settings(patch: SettingsPatch = Body(...)) -> dict[str, Any]:
    patch = patch.model_dump(exclude_none=True)
    s = load_settings()
    s.update(patch)
    save_settings(s)
    return s


# ---- models ---------------------------------------------------------------- #


@app.get("/api/models", tags=["models"], summary="Search models by source")
async def api_models(source: str = "cache", q: str = "", limit: int = 30,
                     task: str | None = None) -> list[dict[str, Any]]:
    got = cached_models()
    if source == "cache":
        rows = [{"id": k, "size_gb": v, "downloaded": True, "source": "cache"} for k, v in got.items()]
        rows = [r for r in rows if not q or q.lower() in r["id"].lower()]
        return sorted(rows, key=lambda x: -x["size_gb"])[:limit]
    if source == "local":
        out = []
        for base in (Path("/Users/kmail/models"), Path.home() / "models"):
            if base.exists():
                for d in base.iterdir():
                    if (d / "config.json").exists():
                        out.append({"id": str(d), "downloaded": True, "size_gb": 0, "source": "local"})
        return out[:limit]
    if source == "curated":
        return [
            {"id": mid, "downloaded": mid in got, "size_gb": got.get(mid), "source": "curated"}
            for mid in CURATED
        ]
    if source in ("hf", "mlx"):
        try:
            return await asyncio.to_thread(hf_search, source, q, min(limit, 50), task)
        except Exception as e:
            raise HTTPException(502, f"HF search failed: {type(e).__name__}: {e}")
    raise HTTPException(400, "unknown source")


@app.get("/api/models/detail", tags=["models"], summary="Size and gating for one repo")
async def api_model_detail(id: str = Query(...)) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(model_detail, id)
    except Exception as e:
        raise HTTPException(502, f"cannot inspect {id}: {type(e).__name__}: {e}")


# ---- server ---------------------------------------------------------------- #


@app.post("/api/server/start", tags=["server"], summary="Load a model in the background")
def api_server_start(req: ServerStartRequest = Body(...)) -> dict[str, Any]:
    payload = req.model_dump()
    return start_load(payload)


@app.get("/api/upstream", tags=["system"], summary="Where the engine is and why it is (not) reachable")
def api_upstream() -> dict[str, Any]:
    """One call that answers "is the container talking to the right host?"."""
    probe = []
    for url in upstream_candidates():
        probe.append({"url": url, "ok": _healthy(url, 1.2)})
    chosen = resolve_upstream()
    loader_probe = []
    host = chosen.split("//", 1)[-1].rsplit(":", 1)[0]
    for url in (f"http://{host}:43441", "http://host.docker.internal:43441"):
        try:
            ok = httpx.get(f"{url}/healthz", timeout=1.2).status_code == 200
        except Exception:
            ok = False
        loader_probe.append({"url": url, "ok": ok})
    return {
        "chosen": chosen,
        "reachable": _healthy(chosen, 1.2),
        "candidates": probe,
        "loader": {"configured": bool(os.environ.get("WORKBENCH_LOADER_URL")),
                   "resolved": LOADER_URL or None,
                   "candidates": loader_probe},
        "hint": ("set UPSTREAM in .env to the address of the machine running SGLang"
                 if not any(c["ok"] for c in probe) else None),
    }


@app.get("/api/memory", tags=["system"], summary="Memory picture and a model sizing plan")
def api_memory(model: str | None = None) -> dict[str, Any]:
    """Host numbers when a loader agent is configured; otherwise the container's own
    cgroup view, which is still useful but not what SGLang competes for."""
    if LOADER_URL:
        return loader_call("GET", "/memory" + (f"?model={model}" if model else ""))
    from memory import plan, snapshot

    target = model or server_info().get("model") or load_settings().get("model") or ""
    return {"scope": "container", "snapshot": snapshot(),
            "plan": plan(target) if target else None}


@app.get("/api/engine", tags=["system"],
         summary="Engine provisioning: image pull, model download, warm-up")
def api_engine() -> dict[str, Any]:
    """Cheap-to-poll view of an engine this container does not supervise."""
    return engine_status()


@app.get("/api/server/job", tags=["server"], summary="Current load/download job")
def api_server_job() -> dict[str, Any]:
    if LOADER_URL:
        return loader_call("GET", "/job")
    return JOB.snapshot()


@app.post("/api/server/cancel", tags=["server"], summary="Cancel the running load")
def api_server_cancel() -> dict[str, Any]:
    if LOADER_URL:
        return loader_call("POST", "/cancel")
    JOB.cancel.set()
    return {"cancelling": True}


@app.post("/api/server/stop", tags=["server"], summary="Stop the SGLang process")
def api_server_stop() -> dict[str, Any]:
    if LOADER_URL:
        return loader_call("POST", "/stop")
    return stop_server()


@app.get("/api/server/logs", tags=["server"], summary="Tail the SGLang log")
def api_server_logs(tail: int = 200) -> list[str]:
    if LOADER_URL:
        return loader_call("GET", f"/logs?tail={tail}")
    if not SERVER_LOG.exists():
        return []
    lines = SERVER_LOG.read_text(errors="replace").replace("\r", "\n").splitlines()
    return lines[-tail:]


# ---- chats ----------------------------------------------------------------- #


@app.get("/api/chats", tags=["chats"], summary="List or create conversations")
def api_chats(q: str = "") -> list[dict[str, Any]]:
    with db() as conn:
        if q:
            rows = conn.execute(
                """SELECT c.* FROM chats c WHERE c.title LIKE ?
                   OR EXISTS (SELECT 1 FROM messages m WHERE m.chat_id=c.id AND m.content LIKE ?)
                   ORDER BY c.pinned DESC, c.updated DESC""",
                (f"%{q}%", f"%{q}%"),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM chats ORDER BY pinned DESC, updated DESC").fetchall()
    return [dict(r) for r in rows]


@app.post("/api/chats", tags=["chats"], summary="List or create conversations")
def api_chat_new(req: ChatNew = Body(default_factory=ChatNew)) -> dict[str, Any]:
    payload = req.model_dump()
    cid = uuid.uuid4().hex[:12]
    now = time.time()
    title = payload.get("title") or "New chat"
    with db() as conn:
        conn.execute(
            "INSERT INTO chats (id,title,model,params,created,updated,pinned) VALUES (?,?,?,?,?,?,0)",
            (cid, title, payload.get("model"), json.dumps(payload.get("params", {})), now, now),
        )
    return {"id": cid, "title": title, "created": now, "updated": now}


@app.patch("/api/chats/{chat_id}", tags=["chats"], summary="Rename, pin or delete a conversation")
def api_chat_patch(chat_id: str, req: ChatPatch = Body(...)) -> dict[str, Any]:
    payload = req.model_dump()
    with db() as conn:
        if "title" in payload:
            conn.execute("UPDATE chats SET title=? WHERE id=?", (payload["title"], chat_id))
        if "pinned" in payload:
            conn.execute("UPDATE chats SET pinned=? WHERE id=?", (int(bool(payload["pinned"])), chat_id))
        if "params" in payload:
            conn.execute("UPDATE chats SET params=? WHERE id=?", (json.dumps(payload["params"]), chat_id))
    return {"ok": True}


@app.delete("/api/chats", tags=["chats"], summary="Delete every conversation")
def api_chats_clear(confirm: str = "") -> dict[str, Any]:
    """Wipe the chat history: every conversation, its messages and its artifacts.

    Requires ``?confirm=all`` so a stray request (or a mistyped curl) cannot destroy the
    workspace. Settings, memory entries, prompts, personas, harnesses and MCP servers are
    deliberately kept — this is a history reset, not a factory reset.
    """
    if confirm != "all":
        raise HTTPException(428, "refusing to delete every chat without confirm=all")
    with db() as conn:
        ids = [r["id"] for r in conn.execute("SELECT id FROM chats").fetchall()]
        messages = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
        artifacts = conn.execute("SELECT COUNT(*) c FROM artifacts").fetchone()["c"]
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM artifacts")
        conn.execute("DELETE FROM chats")
    files = 0
    for cid in ids:
        d = chat_artifact_dir(cid)
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    files += 1
            shutil.rmtree(d, ignore_errors=True)
    return {"deleted_chats": len(ids), "deleted_messages": messages,
            "deleted_artifacts": artifacts, "files_removed": files}


@app.delete("/api/chats/{chat_id}", tags=["chats"], summary="Rename, pin or delete a conversation")
def api_chat_delete(chat_id: str) -> dict[str, Any]:
    with db() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    return {"deleted": chat_id}


@app.post("/api/chats/{chat_id}/fork", tags=["chats"], summary="Fork a conversation at a message")
def api_chat_fork(chat_id: str, req: ForkRequest = Body(default_factory=ForkRequest)) -> dict[str, Any]:
    payload = req.model_dump()
    upto = payload.get("message_id")
    new_id = uuid.uuid4().hex[:12]
    now = time.time()
    with db() as conn:
        src = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
        if not src:
            raise HTTPException(404, "no such chat")
        rows = conn.execute("SELECT * FROM messages WHERE chat_id=? ORDER BY created", (chat_id,)).fetchall()
        keep = []
        for r in rows:
            keep.append(r)
            if upto and r["id"] == upto:
                break
        conn.execute(
            "INSERT INTO chats (id,title,model,params,created,updated,pinned) VALUES (?,?,?,?,?,?,0)",
            (new_id, (src["title"] or "Chat") + " (fork)", src["model"], src["params"], now, now),
        )
        for r in keep:
            conn.execute(
                "INSERT INTO messages (id,chat_id,role,content,reasoning,tools,meta,created,edited) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex[:12], new_id, r["role"], r["content"], r["reasoning"],
                 r["tools"], r["meta"], r["created"], r["edited"]),
            )
    return {"id": new_id, "messages": len(keep)}


@app.get("/api/chats/{chat_id}/messages", tags=["chats"], summary="Messages of one conversation")
def api_messages(chat_id: str) -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM messages WHERE chat_id=? ORDER BY created", (chat_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["tools"] = json.loads(d["tools"]) if d.get("tools") else None
        d["meta"] = json.loads(d["meta"]) if d.get("meta") else None
        out.append(d)
    return out


@app.patch("/api/messages/{message_id}", tags=["chats"], summary="Edit or delete one message")
def api_message_edit(message_id: str, req: MessageEdit = Body(...)) -> dict[str, Any]:
    payload = req.model_dump()
    text = (payload.get("content") or "").strip()
    if not text:
        raise HTTPException(400, "content required")
    with db() as conn:
        conn.execute("UPDATE messages SET content=?, edited=1 WHERE id=?", (text, message_id))
        row = conn.execute("SELECT chat_id FROM messages WHERE id=?", (message_id,)).fetchone()
        if row:
            conn.execute("UPDATE chats SET updated=? WHERE id=?", (time.time(), row["chat_id"]))
    return {"ok": True}


@app.delete("/api/messages/{message_id}", tags=["chats"], summary="Edit or delete one message")
def api_message_delete(message_id: str) -> dict[str, Any]:
    with db() as conn:
        conn.execute("DELETE FROM messages WHERE id=?", (message_id,))
    return {"deleted": message_id}


@app.get("/api/chats/{chat_id}/export", tags=["chats"], summary="Export a conversation")
def api_export(chat_id: str, fmt: str = "md") -> Any:
    with db() as conn:
        chat = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
        rows = conn.execute(
            "SELECT role, content, created, meta FROM messages WHERE chat_id=? ORDER BY created",
            (chat_id,),
        ).fetchall()
    if not chat:
        raise HTTPException(404, "no such chat")
    if fmt == "json":
        body = json.dumps({"chat": dict(chat), "messages": [dict(r) for r in rows]},
                          indent=2, ensure_ascii=False)
        media = "application/json"
    else:
        parts = [f"# {chat['title']}\n", f"_model: {chat['model'] or 'unknown'}_\n"]
        for r in rows:
            who = "You" if r["role"] == "user" else "Assistant"
            parts.append(f"## {who}\n\n{r['content']}\n")
        body = "\n".join(parts)
        media = "text/markdown"
    fname = re.sub(r"[^\w\-]+", "_", chat["title"] or "chat")[:60] or "chat"
    return StreamingResponse(
        iter([body]), media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{fname}.{fmt}"'},
    )


@app.post("/api/chats/{chat_id}/autotitle", tags=["chats"], summary="Generate a short title with the model")
async def api_autotitle(chat_id: str) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute(
            "SELECT content FROM messages WHERE chat_id=? AND role='user' ORDER BY created LIMIT 1",
            (chat_id,),
        ).fetchone()
    if not row:
        raise HTTPException(400, "empty chat")
    prompt = f"Give a 3-5 word title for a conversation starting with:\n{row['content'][:400]}"
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                f"{UPSTREAM}/v1/chat/completions",
                json={
                    "model": load_settings()["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 24, "temperature": 0.2,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        title = r.json()["choices"][0]["message"]["content"].strip().strip('"').splitlines()[0][:60]
    except Exception:
        raise HTTPException(502, "model unavailable for titling")
    with db() as conn:
        conn.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))
    return {"title": title}


# ---- memory / prompts / personas ------------------------------------------- #


@app.get("/api/memory", tags=["memory"], summary="List or add long-term facts")
def api_memory() -> list[dict[str, Any]]:
    with db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM memory ORDER BY created DESC")]


@app.post("/api/memory", tags=["memory"], summary="List or add long-term facts")
def api_memory_add(req: MemoryCreate = Body(...)) -> dict[str, Any]:
    payload = req.model_dump()
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text required")
    mid = uuid.uuid4().hex[:12]
    with db() as conn:
        conn.execute("INSERT INTO memory (id,kind,text,created) VALUES (?,?,?,?)",
                     (mid, payload.get("kind", "fact"), text, time.time()))
    return {"id": mid, "text": text}


@app.delete("/api/memory/{mid}", tags=["memory"], summary="Delete a fact")
def api_memory_delete(mid: str) -> dict[str, Any]:
    with db() as conn:
        conn.execute("DELETE FROM memory WHERE id=?", (mid,))
    return {"deleted": mid}


@app.get("/api/prompts", tags=["prompts"], summary="List or create prompts")
def api_prompts() -> list[dict[str, Any]]:
    with db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM prompts ORDER BY created DESC")]


@app.post("/api/prompts", tags=["prompts"], summary="List or create prompts")
def api_prompt_add(req: PromptCreate = Body(...)) -> dict[str, Any]:
    payload = req.model_dump()
    title = (payload.get("title") or "").strip()
    body = (payload.get("body") or "").strip()
    if not title or not body:
        raise HTTPException(400, "title and body required")
    pid = uuid.uuid4().hex[:12]
    with db() as conn:
        conn.execute("INSERT INTO prompts (id,title,body,created) VALUES (?,?,?,?)",
                     (pid, title, body, time.time()))
    return {"id": pid, "title": title, "body": body}


@app.delete("/api/prompts/{pid}", tags=["prompts"], summary="Delete a prompt")
def api_prompt_delete(pid: str) -> dict[str, Any]:
    with db() as conn:
        conn.execute("DELETE FROM prompts WHERE id=?", (pid,))
    return {"deleted": pid}


@app.get("/api/personas", tags=["personas"], summary="List or create personas")
def api_personas() -> list[dict[str, Any]]:
    with db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM personas ORDER BY created DESC")]


@app.post("/api/personas", tags=["personas"], summary="List or create personas")
def api_persona_add(req: PersonaCreate = Body(...)) -> dict[str, Any]:
    payload = req.model_dump()
    name = (payload.get("name") or "").strip()
    system = (payload.get("system_prompt") or "").strip()
    if not name or not system:
        raise HTTPException(400, "name and system_prompt required")
    pid = uuid.uuid4().hex[:12]
    with db() as conn:
        conn.execute(
            "INSERT INTO personas (id,name,system_prompt,params,created) VALUES (?,?,?,?,?)",
            (pid, name, system, json.dumps(payload.get("params") or {}), time.time()),
        )
    return {"id": pid, "name": name}


@app.delete("/api/personas/{pid}", tags=["personas"], summary="Delete a persona")
def api_persona_delete(pid: str) -> dict[str, Any]:
    with db() as conn:
        conn.execute("DELETE FROM personas WHERE id=?", (pid,))
    return {"deleted": pid}



# ---- harness (Harness² components) ----------------------------------------- #


@app.get("/api/harnesses", tags=["harness"], summary="List harnesses and their components")
def api_harnesses() -> dict[str, Any]:
    return {
        "components": [{"key": k, "label": lbl} for k, lbl in h2.COMPONENTS],
        "harnesses": list_harnesses(),
    }


@app.post("/api/harnesses", tags=["harness"], summary="Create a harness")
def api_harness_create(payload: dict = Body(default={})) -> dict[str, Any]:
    name = (payload.get("name") or "New harness").strip()
    base = h2.blank_harness(name)
    for k, v in (payload.get("components") or payload).items():
        if k in base and k != "name":
            base[k] = v
    hid = h2.new_id()
    with db() as conn:
        conn.execute("INSERT INTO harnesses (id,name,components,created,updated) VALUES (?,?,?,?,?)",
                     (hid, name, json.dumps(base), h2.now(), h2.now()))
    return {"id": hid, "name": name}


@app.patch("/api/harnesses/{hid}", tags=["harness"], summary="Update harness components")
def api_harness_patch(hid: str, payload: dict = Body(...)) -> dict[str, Any]:
    return save_harness(hid, payload)


@app.post("/api/harnesses/{hid}/duplicate", tags=["harness"], summary="Duplicate a harness")
def api_harness_duplicate(hid: str) -> dict[str, Any]:
    src = get_harness(hid)
    name = f"{src.get('name') or 'Harness'} copy"
    comps = {k: src[k] for k, _ in h2.COMPONENTS if k in src}
    comps["name"] = name
    new = h2.new_id()
    with db() as conn:
        conn.execute("INSERT INTO harnesses (id,name,components,created,updated) VALUES (?,?,?,?,?)",
                     (new, name, json.dumps(comps), h2.now(), h2.now()))
    return {"id": new, "name": name}


@app.delete("/api/harnesses/{hid}", tags=["harness"], summary="Delete a harness")
def api_harness_delete(hid: str) -> dict[str, Any]:
    with db() as conn:
        count = conn.execute("SELECT COUNT(*) c FROM harnesses").fetchone()["c"]
        if count <= 1:
            raise HTTPException(400, "cannot delete the last harness")
        conn.execute("DELETE FROM harnesses WHERE id=?", (hid,))
        conn.execute("UPDATE chats SET harness=NULL WHERE harness=?", (hid,))
    return {"deleted": hid}


@app.get("/api/harnesses/{hid}/preview", tags=["harness"],
         summary="Compose the system prompt without sending it")
async def api_harness_preview(hid: str, q: str = "") -> dict[str, Any]:
    """The prompt the mounted composition would actually send, section by section."""
    harness = get_harness(hid)
    params = {**load_settings(), "memory_enabled": bool(q)}
    holder: dict[str, Any] = {}
    rows = dsh_rows_for("", harness, params, q, holder)
    host = dsh_mount(rows, holder)
    try:
        prompt = host.root.require("systemPrompt")
        tools = host.root.require("tools")
        assembly = await prompt.assemble_async("agent", tools.schemas("agent"))
        text = prompt.render(assembly)
        report: dict[str, Any] = {}
        for section in assembly.report.get("sections", []):
            report[section["name"]] = {"chars": section["chars"],
                                       "tokens": section["tokens"]}
        report["total"] = assembly.report.get("total", {})
        report["tools"] = assembly.report.get("tools", [])
        return {
            "harness": harness.get("name"),
            "system": text,
            "report": report,
            "profile": dsh_runtime.PROFILE_NAME,
            "services": host.services(),
            "plugin_count": len(host.dump()["rows"]),
        }
    finally:
        host.dispose()


@app.get("/api/harness/profiles", tags=["harness"], summary="Shipped harness profiles")
def api_harness_profiles() -> list[dict[str, Any]]:
    from dsh.profiles import PROFILES

    return [p.dump() for p in PROFILES.values()]


@app.get("/api/harnesses/{hid}/composition", tags=["harness"],
         summary="The mounted plugin tree for one harness")
def api_harness_composition(hid: str) -> dict[str, Any]:
    harness = get_harness(hid)
    params = load_settings()
    holder: dict[str, Any] = {}
    rows = dsh_rows_for("", harness, params, "", holder)
    host = dsh_mount(rows, holder)
    try:
        report = dsh_runtime.composition_report(rows, host)
        report["harness"] = {"id": hid, "name": harness.get("name")}
        return report
    finally:
        host.dispose()


@app.get("/api/chats/{chat_id}/session", tags=["harness"],
         summary="Durable session events for a chat (JSONL replay)")
def api_chat_session(chat_id: str, tail: int = 200) -> dict[str, Any]:
    path = dsh_session_root() / f"{chat_id}.jsonl"
    events: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {"chat_id": chat_id, "exists": path.exists(), "count": len(events),
            "events": events[-max(1, min(tail, 2000)):]}


@app.post("/api/chats/{chat_id}/compact", tags=["harness"],
          summary="Condense this chat's session now (the /compact command)")
async def api_chat_compact(chat_id: str, payload: dict = Body(default_factory=dict)) -> dict[str, Any]:
    params = {**load_settings(), **(payload.get("params") or {})}
    harness = get_harness(None, chat_id)
    holder: dict[str, Any] = {}
    host = dsh_mount(dsh_rows_for(chat_id, harness, params, "", holder), holder)
    try:
        store = host.root.require("agentSessionStore")
        compaction = host.root.require("compaction")
        path = dsh_session_root() / f"{chat_id}.jsonl"
        if path.exists():
            session = store.restore(path, ctx=host.root)
            session.persist, session.path = True, path
        else:
            session = store.create(chat_id=chat_id, persist=True, ctx=host.root)
            with db() as conn:
                rows = conn.execute(
                    "SELECT role, content FROM messages WHERE chat_id=? AND "
                    "role IN ('user','assistant') ORDER BY created", (chat_id,)).fetchall()
            dsh_runtime.seed_history(session, [{"role": r["role"], "content": r["content"]}
                                               for r in rows if r["content"]])
        before = compaction.pressure(session)
        result = await compaction.condense(session, config={"model": params.get("model")},
                                           force=True)
        return {"chat_id": chat_id, "before_tokens": before,
                "after_tokens": compaction.pressure(session), **result}
    finally:
        host.dispose()


@app.post("/api/chats/{chat_id}/harness", tags=["harness"], summary="Bind a harness to a chat")
def api_chat_harness(chat_id: str, payload: dict = Body(...)) -> dict[str, Any]:
    hid = payload.get("harness_id")
    with db() as conn:
        conn.execute("UPDATE chats SET harness=? WHERE id=?", (hid, chat_id))
    return {"chat_id": chat_id, "harness_id": hid}


@app.post("/api/harness/probe", tags=["harness"],
          summary="Run the chat's last user turn under a candidate harness (Probe)")
async def api_harness_probe(payload: dict = Body(...)) -> dict[str, Any]:
    chat_id = payload.get("chat_id")
    hid = payload.get("harness_id")
    if not chat_id or not hid:
        raise HTTPException(400, "chat_id and harness_id required")
    with db() as conn:
        row = conn.execute(
            "SELECT content FROM messages WHERE chat_id=? AND role='user' ORDER BY created DESC LIMIT 1",
            (chat_id,),
        ).fetchone()
    if not row:
        raise HTTPException(400, "chat has no user turn yet")
    params = {**load_settings(), **(payload.get("params") or {})}
    harness = get_harness(hid)
    messages, report = build_messages(chat_id, row["content"], params, None, harness=harness)
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)) as c:
            r = await c.post(f"{UPSTREAM}/v1/chat/completions", json={
                "model": params.get("model"),
                "messages": messages,
                "max_tokens": int(params.get("max_tokens", 512)),
                "temperature": float(params.get("temperature", 0.7)),
                "chat_template_kwargs": {"enable_thinking": False},
            })
        data = r.json()
        text = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(502, f"probe failed: {type(e).__name__}: {e}")
    return {
        "harness": harness.get("name"),
        "text": text,
        "elapsed_s": round(time.time() - started, 2),
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
        "report": report,
        "guardrails": h2.check_output(harness, text),
    }


# ---- recursive refinement (Harness² Propose–Probe–Compose) ----------------- #

H2_RUNNING = ("queued", "probing", "proposing", "judging", "composing", "reflecting")


@app.post("/api/harness/recursion", tags=["harness"],
          summary="Refine the selected harness with Propose–Probe–Compose")
def api_h2_start(payload: dict = Body(default={})) -> dict[str, Any]:
    if H2_JOB.snapshot()["state"] in H2_RUNNING:
        raise HTTPException(409, "a harness refinement is already running")
    if not server_up():
        raise HTTPException(503, "the model server is down")

    chat_id = payload.get("chat_id")
    task = str(payload.get("task") or "").strip()
    if not task and chat_id:
        with db() as conn:
            row = conn.execute(
                "SELECT content FROM messages WHERE chat_id=? AND role='user' "
                "ORDER BY created DESC LIMIT 1", (chat_id,),
            ).fetchone()
        task = (row["content"] if row else "") or ""
    if not task:
        raise HTTPException(400, "a task (or a chat with a user turn) is required")

    harness = get_harness(payload.get("harness_id"), chat_id)
    if not harness.get("id"):
        raise HTTPException(400, "no harness selected")
    mode = str(payload.get("mode") or "sequential")
    if mode not in h2.RECURSION_MODES:
        raise HTTPException(400, f"mode must be one of {h2.RECURSION_MODES}")
    try:
        k = max(1, min(3, int(payload.get("k") or 1)))
    except (TypeError, ValueError):
        k = 1

    run_id = h2.new_id()
    started = h2.now()
    with db() as conn:
        conn.execute(
            "INSERT INTO harness_runs (id,harness_id,chat_id,task,mode,k,state,step,"
            "progress,created) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, harness.get("id"), chat_id, task[:8000], mode, k,
             "queued", 0, 0.0, started),
        )
    H2_JOB.cancel.clear()
    H2_JOB._set(state="queued", run_id=run_id, harness_id=harness.get("id"),
                mode=mode, k=k, step=0, progress=0.0, message="queued",
                error=None, started=started, finished=None, result=None)
    run_payload = {
        "task": task,
        "mode": mode,
        "k": k,
        "harness": harness,
        "params": payload.get("params") or {},
        "domain": payload.get("domain") or "general",
    }
    H2_JOB.thread = threading.Thread(target=_h2_worker, args=(run_id, run_payload), daemon=True)
    H2_JOB.thread.start()
    return {"accepted": True, "run_id": run_id, "mode": mode, "k": k,
            "harness": harness.get("name")}


@app.get("/api/harness/recursion/job", tags=["harness"],
         summary="Progress of the running Propose–Probe–Compose refinement")
def api_h2_job() -> dict[str, Any]:
    return H2_JOB.snapshot()


@app.post("/api/harness/recursion/cancel", tags=["harness"],
          summary="Cancel the current refinement")
def api_h2_cancel() -> dict[str, Any]:
    H2_JOB.cancel.set()
    return {"cancelling": True}


@app.get("/api/harness/runs", tags=["harness"], summary="Recent refinement runs")
def api_h2_runs(limit: int = 20) -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT id,harness_id,chat_id,task,mode,k,state,step,progress,error,"
            "created,finished FROM harness_runs ORDER BY created DESC LIMIT ?",
            (max(1, min(100, limit)),),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/harness/runs/{run_id}", tags=["harness"], summary="One refinement run")
def api_h2_run(run_id: str) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT * FROM harness_runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        raise HTTPException(404, "run not found")
    out = dict(row)
    for key in ("baseline", "candidates", "verdict", "result"):
        try:
            out[key] = json.loads(out[key]) if out.get(key) else None
        except Exception:
            out[key] = None
    return out


@app.post("/api/harness/runs/{run_id}/adopt", tags=["harness"],
          summary="Save the refined harness as a new harness")
def api_h2_adopt(run_id: str, payload: dict = Body(default={})) -> dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT * FROM harness_runs WHERE id=?", (run_id,)).fetchone()
    if not row or row["state"] != "ready":
        raise HTTPException(400, "run is not ready to adopt")
    try:
        result = json.loads(row["result"] or "{}")
    except Exception:
        raise HTTPException(400, "run has no result")
    components = h2.normalize(result.get("harness"))
    base_name = get_harness(row["harness_id"]).get("name") or "Harness"
    name = str(payload.get("name") or f"{base_name} · refined").strip()[:80]
    existing = {h["name"] for h in list_harnesses()}
    if name in existing:
        name = f"{name} {run_id[:4]}"
    components["name"] = name
    hid = h2.new_id()
    with db() as conn:
        conn.execute(
            "INSERT INTO harnesses (id,name,components,created,updated) VALUES (?,?,?,?,?)",
            (hid, name, json.dumps(components), h2.now(), h2.now()),
        )
    bound = False
    if payload.get("bind") and row["chat_id"]:
        with db() as conn:
            conn.execute("UPDATE chats SET harness=? WHERE id=?", (hid, row["chat_id"]))
        bound = True
    return {"id": hid, "name": name, "bound": bound, "harness": components}


# ---- lessons (experience library) ------------------------------------------ #


@app.get("/api/lessons", tags=["harness"], summary="List stored lessons")
def api_lessons() -> list[dict[str, Any]]:
    with db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM lessons ORDER BY created DESC")]


@app.post("/api/lessons", tags=["harness"], summary="Add a lesson")
def api_lesson_add(payload: dict = Body(...)) -> dict[str, Any]:
    lesson = (payload.get("lesson") or "").strip()
    if not lesson:
        raise HTTPException(400, "lesson required")
    lid = h2.new_id()
    with db() as conn:
        conn.execute(
            "INSERT INTO lessons (id,domain,cue,lesson,evidence,confidence,created) VALUES (?,?,?,?,?,?,?)",
            (lid, payload.get("domain") or "general", payload.get("cue") or "",
             lesson, payload.get("evidence") or "", payload.get("confidence") or "medium", h2.now()),
        )
    return {"id": lid, "lesson": lesson}


@app.delete("/api/lessons/{lid}", tags=["harness"], summary="Delete a lesson")
def api_lesson_delete(lid: str) -> dict[str, Any]:
    with db() as conn:
        conn.execute("DELETE FROM lessons WHERE id=?", (lid,))
    return {"deleted": lid}


@app.post("/api/lessons/reflect", tags=["harness"],
          summary="Ask the model for a transferable lesson from the last exchange")
async def api_lessons_reflect(payload: dict = Body(...)) -> dict[str, Any]:
    chat_id = payload.get("chat_id")
    if not chat_id:
        raise HTTPException(400, "chat_id required")
    with db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY created DESC LIMIT 4", (chat_id,)
        ).fetchall()
    if not rows:
        raise HTTPException(400, "nothing to reflect on")
    convo = "\n\n".join(f"{r['role'].upper()}: {r['content'][:1200]}" for r in reversed(rows))
    ask = (
        "Reflect on this exchange and extract ONE transferable lesson that would improve "
        "future work of the same kind. Reply with JSON only, keys: cue (when to apply it), "
        "lesson (imperative sentence), evidence (what in the exchange shows it), "
        "confidence (low|medium|high).\n\n" + convo
    )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)) as c:
            r = await c.post(f"{UPSTREAM}/v1/chat/completions", json={
                "model": load_settings().get("model"),
                "messages": [{"role": "user", "content": ask}],
                "max_tokens": 220, "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": False},
            })
        raw = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(502, f"reflect failed: {type(e).__name__}: {e}")
    blob = raw[raw.find("{"): raw.rfind("}") + 1] or "{}"
    try:
        parsed = json.loads(blob)
    except Exception:
        parsed = {"lesson": raw.strip()[:400], "cue": "", "evidence": "", "confidence": "low"}
    parsed.setdefault("lesson", raw.strip()[:400])   # stable contract even on bad JSON
    lid = h2.new_id()
    with db() as conn:
        conn.execute(
            "INSERT INTO lessons (id,domain,cue,lesson,evidence,confidence,created) VALUES (?,?,?,?,?,?,?)",
            (lid, payload.get("domain") or "general", str(parsed.get("cue") or "")[:300],
             str(parsed.get("lesson") or raw.strip()[:400]), str(parsed.get("evidence") or "")[:600],
             str(parsed.get("confidence") or "low"), h2.now()),
        )
    return {"id": lid, **parsed}



# ---- artifacts (files the model or the user creates) ----------------------- #


@app.get("/api/chats/{chat_id}/artifacts", tags=["artifacts"], summary="Files written in this chat")
def api_chat_artifacts(chat_id: str) -> list[dict[str, Any]]:
    return artifact_rows(chat_id)


@app.post("/api/chats/{chat_id}/artifacts", tags=["artifacts"], summary="Write a file into the chat workspace")
def api_artifact_create(chat_id: str, payload: dict = Body(...)) -> dict[str, Any]:
    name = payload.get("name") or "file.txt"
    content = payload.get("content")
    if content is None:
        raise HTTPException(400, "content required")
    return save_artifact(chat_id, name, str(content), payload.get("mime") or "text/plain")



@app.post("/api/extract", tags=["artifacts"], summary="Extract text from an uploaded document")
async def api_extract(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "file larger than 25 MB")
    text, meta = extract_document(file.filename or "upload", data)
    return {"name": file.filename, "size": len(data), "chars": len(text), **meta, "text": text}


@app.post("/api/chats/{chat_id}/uploads", tags=["artifacts"],
          summary="Upload a document into the chat: stores the original and returns extracted text")
async def api_upload(chat_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(400, "the uploaded file is empty")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "file larger than 25 MB")
    name = Path(file.filename or "upload").name
    try:
        text, meta = extract_document(name, data)
    except Exception as exc:  # a corrupt or unsupported document must not 500
        raise HTTPException(400, f"could not read {name}: {type(exc).__name__}: {str(exc)[:200]}") from exc
    art = save_artifact(chat_id, name, data, mime=(file.content_type or "application/octet-stream"))
    return {"artifact": art, "name": name, "size": len(data), "chars": len(text), **meta, "text": text}


@app.get("/api/artifacts", tags=["artifacts"], summary="Recent artifacts across chats")
def api_artifacts(limit: int = 50) -> list[dict[str, Any]]:
    return artifact_rows()[:limit]


@app.get("/api/artifacts/{aid}", tags=["artifacts"], summary="Download an artifact")
def api_artifact_download(aid: str) -> Any:
    row = next((a for a in artifact_rows() if a["id"] == aid), None)
    if not row:
        raise HTTPException(404, "no such artifact")
    path = Path(row["path"])
    if not path.exists():
        raise HTTPException(410, "artifact file is gone")
    return FileResponse(path, media_type=row["mime"] or "application/octet-stream", filename=row["name"])


@app.get("/api/artifacts/{aid}/preview", tags=["artifacts"], summary="Text preview of an artifact")
def api_artifact_preview(aid: str, max_bytes: int = 65536) -> dict[str, Any]:
    row = next((a for a in artifact_rows() if a["id"] == aid), None)
    if not row:
        raise HTTPException(404, "no such artifact")
    path = Path(row["path"])
    if not path.exists():
        raise HTTPException(410, "artifact file is gone")
    raw = path.read_bytes()[:max_bytes]
    try:
        text = raw.decode()
        binary = False
    except UnicodeDecodeError:
        text = raw.decode(errors="replace")
        binary = True
    return {"id": aid, "name": row["name"], "mime": row["mime"], "size": row["size"],
            "text": text, "truncated": row["size"] > max_bytes, "binary": binary}


@app.delete("/api/artifacts/{aid}", tags=["artifacts"], summary="Delete an artifact")
def api_artifact_delete(aid: str) -> dict[str, Any]:
    row = next((a for a in artifact_rows() if a["id"] == aid), None)
    if row:
        try:
            Path(row["path"]).unlink(missing_ok=True)
        except Exception:
            pass
        with db() as conn:
            conn.execute("DELETE FROM artifacts WHERE id=?", (aid,))
    return {"deleted": aid}


# ---- stats ----------------------------------------------------------------- #


@app.get("/api/stats", tags=["stats"], summary="Usage accounting")
def api_stats() -> dict[str, Any]:
    with db() as conn:
        rows = conn.execute("SELECT role, meta, created FROM messages").fetchall()
        n_chats = conn.execute("SELECT COUNT(*) c FROM chats").fetchone()["c"]
    days: dict[str, dict[str, Any]] = {}
    total_p = total_c = total_msgs = 0
    for r in rows:
        day = time.strftime("%Y-%m-%d", time.localtime(r["created"]))
        d = days.setdefault(day, {"date": day, "prompt": 0, "completion": 0, "messages": 0})
        d["messages"] += 1
        total_msgs += 1
        if r["meta"]:
            try:
                usage = json.loads(r["meta"]).get("usage") or {}
            except Exception:
                usage = {}
            d["prompt"] += usage.get("prompt_tokens", 0)
            d["completion"] += usage.get("completion_tokens", 0)
            total_p += usage.get("prompt_tokens", 0)
            total_c += usage.get("completion_tokens", 0)
    return {
        "chats": n_chats,
        "messages": total_msgs,
        "prompt_tokens": total_p,
        "completion_tokens": total_c,
        "days": sorted(days.values(), key=lambda x: x["date"]),
    }


# ---- MCP ------------------------------------------------------------------- #


@app.get("/api/mcp/servers", tags=["mcp"], summary="List or register MCP servers")
def api_mcp_list() -> list[dict[str, Any]]:
    out = []
    for cfg in load_mcp():
        c = MCP_CLIENTS.get(cfg["name"])
        out.append({**cfg, "connected": bool(c and c.tools),
                    "tools": [t["name"] for t in (c.tools if c else [])]})
    return out


@app.post("/api/mcp/servers", tags=["mcp"], summary="List or register MCP servers")
def api_mcp_add(req: McpServerCreate = Body(...)) -> dict[str, Any]:
    payload = req.model_dump()
    name = (payload.get("name") or "").strip()
    command = (payload.get("command") or "").strip()
    if not name or not command:
        raise HTTPException(400, "name and command required")
    servers = [s for s in load_mcp() if s["name"] != name]
    servers.append({"name": name, "command": command, "args": payload.get("args", []),
                    "env": payload.get("env") or {}, "enabled": payload.get("enabled", True)})
    save_mcp(servers)
    MCP_CLIENTS.pop(name, None)
    return {"ok": True}


@app.patch("/api/mcp/servers/{name}", tags=["mcp"], summary="Enable, disable or delete a server")
def api_mcp_toggle(name: str, req: McpToggle = Body(...)) -> dict[str, Any]:
    payload = req.model_dump()
    servers = load_mcp()
    for s in servers:
        if s["name"] == name:
            s["enabled"] = bool(payload.get("enabled", True))
    save_mcp(servers)
    return {"ok": True}


@app.delete("/api/mcp/servers/{name}", tags=["mcp"], summary="Enable, disable or delete a server")
def api_mcp_delete(name: str) -> dict[str, Any]:
    client = MCP_CLIENTS.pop(name, None)
    if client:
        client.close()
    save_mcp([s for s in load_mcp() if s["name"] != name])
    return {"deleted": name}


@app.post("/api/mcp/servers/{name}/connect", tags=["mcp"], summary="Connect and discover tools")
def api_mcp_connect(name: str) -> dict[str, Any]:
    client = mcp_client(name)
    try:
        return {"connected": True, "tools": [t["name"] for t in client.refresh_tools()]}
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")


@app.post("/api/mcp/tools/call", tags=["mcp"], summary="Invoke one MCP tool")
def api_mcp_call(req: McpToolCall = Body(...)) -> dict[str, Any]:
    payload = req.model_dump()
    server, tool = payload.get("server"), payload.get("tool")
    if not server or not tool:
        raise HTTPException(400, "server and tool required")
    return {"result": mcp_client(server).call_tool(tool, payload.get("arguments") or {})}


# ---- chat ------------------------------------------------------------------ #


@app.post("/api/chat", tags=["chat"], summary="Stream a completion (DeepSeek Harness loop)")
async def api_chat(req: ChatRequest = Body(...)) -> StreamingResponse:
    """One turn through the mounted plugin tree.

    The session log is the source of the request: the loop commits the prompt and
    the accepted users, derives and freezes history, streams, runs the guarded
    tool pipeline and settles every fact durably. Set WORKBENCH_LEGACY_LOOP=1 to
    fall back to the pre-harness loop.
    """
    if LEGACY_LOOP:
        return await _legacy_chat(req)
    payload = req.model_dump()
    chat_id = payload.get("chat_id")
    regenerate = bool(payload.get("regenerate"))
    text = (payload.get("message") or "").strip()
    attachments = payload.get("attachments") or []
    if not chat_id or (not text and not regenerate):
        raise HTTPException(400, "chat_id and message required (or regenerate=true)")

    if not server_up():
        state_now = JOB.snapshot()["state"]
        if state_now in ("preparing", "downloading", "starting", "stopping"):
            raise HTTPException(409, f"model is {state_now} — try again when it reports ready")
        raise HTTPException(503, "the model server is down — start one from the model switcher")

    settings = load_settings()
    params = {**settings, **(payload.get("params") or {})}
    harness = get_harness(None, chat_id)
    violations = h2.check_input(harness, text)
    if violations:
        raise HTTPException(400, "guardrail: " + "; ".join(violations))

    now = time.time()
    with db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id=? AND role IN ('user','assistant') "
            "ORDER BY created",
            (chat_id,),
        ).fetchall()
        history = [{"role": r["role"], "content": r["content"]} for r in rows if r["content"]]
        if regenerate:
            row = conn.execute(
                "SELECT content FROM messages WHERE chat_id=? AND role='user' ORDER BY created DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
            if not row:
                raise HTTPException(400, "nothing to regenerate")
            text = row["content"]
            for i in range(len(history) - 1, -1, -1):
                if history[i]["role"] == "user":
                    history = history[:i]
                    break
            conn.execute(
                "DELETE FROM messages WHERE chat_id=? AND created > (SELECT MAX(created) FROM messages "
                "WHERE chat_id=? AND role='user')",
                (chat_id, chat_id),
            )
            conn.execute("UPDATE chats SET updated=? WHERE id=?", (now, chat_id))
        else:
            conn.execute(
                "INSERT INTO messages (id,chat_id,role,content,created) VALUES (?,?,?,?,?)",
                (uuid.uuid4().hex[:12], chat_id, "user", text, now),
            )
            t = conn.execute("SELECT title FROM chats WHERE id=?", (chat_id,)).fetchone()
            if t and t["title"] in (None, "", "New chat"):
                conn.execute("UPDATE chats SET title=?, updated=? WHERE id=?", (text[:60], now, chat_id))
            else:
                conn.execute("UPDATE chats SET updated=? WHERE id=?", (now, chat_id))

    holder: dict[str, Any] = {}
    host = dsh_mount(dsh_rows_for(chat_id, harness, params, text, holder), holder)
    services = host.services()
    started = time.time()

    async def gen():
        collected: list[str] = []
        reasoning: list[str] = []
        tool_events: list[dict[str, Any]] = []
        result: dict[str, Any] = {}
        error: str | None = None
        finished = False
        yield sse("start", {"chat_id": chat_id, "model": params.get("model"), "loop": "dsh"})
        try:
            async for frame in dsh_runtime.run_turn(
                host, chat_id=chat_id, text=text, history=history,
                attachments=attachments, model=params.get("model") or "",
                params=params, holder=holder,
            ):
                kind = frame.get("type")
                if kind == "delta":
                    collected.append(frame["text"])
                    yield sse("delta", {"text": frame["text"]})
                elif kind == "reasoning":
                    reasoning.append(frame["text"])
                    yield sse("reasoning", {"text": frame["text"]})
                elif kind == "tool":
                    tool_events.append({k: frame.get(k) for k in
                                        ("tool", "server", "arguments", "result", "state", "status")})
                    yield sse("tool", {k: frame.get(k) for k in
                                       ("tool", "server", "arguments", "result", "state", "status")})
                elif kind == "artifact":
                    yield sse("artifact", frame.get("artifact") or {})
                elif kind == "done":
                    result = frame
                    finished = True
                elif kind == "rejected":
                    error = "guardrail: " + (frame.get("reason") or "rejected")
                    finished = True
                elif kind == "error":
                    error = frame.get("message") or "error"
                    finished = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            agent = holder.get("agent")
            if agent is not None and not finished:
                agent.cancel(keep_inbox=True)
            host.dispose()

        if error:
            yield sse("error", {"message": error})
            return

        final = (result.get("text") or "".join(collected)).strip()
        elapsed = result.get("elapsed_s") or round(time.time() - started, 2)
        usage = result.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0}
        problems = h2.check_output(harness, final)
        meta = {
            "usage": usage,
            "elapsed_s": elapsed,
            "model": params.get("model"),
            "guardrails": problems,
            "prompt": result.get("prompt") or {},
            "loop": "dsh",
            "session_id": result.get("session_id"),
            "compactions": result.get("compactions", 0),
            "plugins": len(services),
        }
        with db() as conn:
            conn.execute(
                "INSERT INTO messages (id,chat_id,role,content,reasoning,tools,meta,created) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex[:12], chat_id, "assistant", final,
                 "".join(reasoning) or None,
                 json.dumps({"rounds": result.get("rounds", 0), "events": tool_events})
                 if tool_events else None,
                 json.dumps(meta), time.time()),
            )
            conn.execute("UPDATE chats SET updated=? WHERE id=?", (time.time(), chat_id))
        yield sse("done", {"text": final, "usage": usage, "elapsed_s": elapsed,
                           "rounds": result.get("rounds", 0), "guardrails": problems,
                           "prompt": result.get("prompt") or {}, "loop": "dsh"})

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _legacy_chat(req: ChatRequest) -> StreamingResponse:
    payload = req.model_dump()
    chat_id = payload.get("chat_id")
    regenerate = bool(payload.get("regenerate"))
    text = (payload.get("message") or "").strip()
    attachments = payload.get("attachments") or []
    if not chat_id or (not text and not regenerate):
        raise HTTPException(400, "chat_id and message required (or regenerate=true)")

    if not server_up():
        state_now = JOB.snapshot()["state"]
        if state_now in ("preparing", "downloading", "starting", "stopping"):
            raise HTTPException(409, f"model is {state_now} — try again when it reports ready")
        raise HTTPException(503, "the model server is down — start one from the model switcher")

    settings = load_settings()
    params = {**settings, **(payload.get("params") or {})}
    harness = get_harness(None, chat_id)
    violations = h2.check_input(harness, text)
    if violations:
        raise HTTPException(400, "guardrail: " + "; ".join(violations))
    tools = effective_tools(harness, params)
    known = {t["name"] for t in tools}

    now = time.time()
    with db() as conn:
        if regenerate:
            row = conn.execute(
                "SELECT content FROM messages WHERE chat_id=? AND role='user' ORDER BY created DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
            if not row:
                raise HTTPException(400, "nothing to regenerate")
            text = row["content"]
            conn.execute(
                "DELETE FROM messages WHERE chat_id=? AND created > (SELECT MAX(created) FROM messages "
                "WHERE chat_id=? AND role='user')",
                (chat_id, chat_id),
            )
            conn.execute("UPDATE chats SET updated=? WHERE id=?", (now, chat_id))
        else:
            conn.execute(
                "INSERT INTO messages (id,chat_id,role,content,created) VALUES (?,?,?,?,?)",
                (uuid.uuid4().hex[:12], chat_id, "user", text, now),
            )
            t = conn.execute("SELECT title FROM chats WHERE id=?", (chat_id,)).fetchone()
            if t and t["title"] in (None, "", "New chat"):
                conn.execute("UPDATE chats SET title=?, updated=? WHERE id=?", (text[:60], now, chat_id))
            else:
                conn.execute("UPDATE chats SET updated=? WHERE id=?", (now, chat_id))

    messages, prompt_report = build_messages(chat_id, text, params, attachments)
    start = time.time()

    async def gen():
        collected: list[str] = []
        tool_events: list[dict[str, Any]] = []
        reasoning: list[str] = []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
        rounds = 0
        yield sse("start", {"chat_id": chat_id, "model": params.get("model")})

        while True:
            body: dict[str, Any] = {
                "model": params.get("model"),
                "messages": messages,
                "max_tokens": int(params.get("max_tokens", 512)),
                "temperature": float(params.get("temperature", 0.7)),
                "top_p": float(params.get("top_p", 0.95)),
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if params.get("top_k"):
                body["top_k"] = int(params["top_k"])
            if params.get("repeat_penalty"):
                body["repetition_penalty"] = float(params["repeat_penalty"])
            if params.get("seed") not in (None, ""):
                body["seed"] = int(params["seed"])
            if params.get("stop"):
                body["stop"] = params["stop"] if isinstance(params["stop"], list) else [params["stop"]]
            body["chat_template_kwargs"] = {"enable_thinking": bool(params.get("enable_thinking", False))}

            round_text: list[str] = []
            try:
                timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("POST", f"{UPSTREAM}/v1/chat/completions", json=body) as resp:
                        if resp.status_code != 200:
                            detail = (await resp.aread()).decode()[:800]
                            yield sse("error", {"message": f"upstream {resp.status_code}: {detail}"})
                            break
                        async for line in resp.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                obj = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            if obj.get("usage"):
                                usage_total["prompt_tokens"] = obj["usage"].get("prompt_tokens", 0)
                                usage_total["completion_tokens"] += obj["usage"].get("completion_tokens", 0)
                            choice = (obj.get("choices") or [{}])[0]
                            delta = choice.get("delta") or {}
                            if delta.get("reasoning_content"):
                                reasoning.append(delta["reasoning_content"])
                                yield sse("reasoning", {"text": delta["reasoning_content"]})
                            if delta.get("content"):
                                round_text.append(delta["content"])
                                yield sse("delta", {"text": delta["content"]})
            except Exception as e:
                yield sse("error", {"message": f"{type(e).__name__}: {e}"})
                break

            out = "".join(round_text)
            call = (extract_tool_call(out, known)
                    if (known and rounds < int(params.get("max_tool_rounds", 3))) else None)
            if not call:
                collected.append(out)
                break

            rounds += 1
            tool_name, args = call["tool"], call["arguments"]
            server = next((t["server"] for t in tools if t["name"] == tool_name), "harness")
            ev_running = {"tool": tool_name, "server": server, "arguments": args, "state": "running"}
            yield sse("tool", ev_running)
            subagent_names = {x.get("name") for x in (harness.get("subagents") or [])}
            if server == "artifacts":
                result, art = run_artifact_tool(chat_id, tool_name, args)
                if art:
                    yield sse("artifact", art)
            elif tool_name == "delegate" or tool_name in subagent_names:
                payload = {"name": args.get("name") or tool_name, "task": args.get("task") or args.get("input") or ""}
                result = await run_delegate(harness, payload, params)
            else:
                result = await run_tool(server, tool_name, args)
            tool_events.append({**ev_running, "result": result[:4000], "state": "done"})
            yield sse("tool", {"tool": tool_name, "server": server, "result": result[:4000], "state": "done"})
            messages.append({"role": "assistant", "content": out})
            messages.append({
                "role": "user",
                "content": f"Observation from {tool_name}:\n{result[:6000]}\n\nContinue. "
                           "Use another tool call if needed, otherwise answer.",
            })

        final = "".join(collected).strip()
        elapsed = round(time.time() - start, 2)
        with db() as conn:
            conn.execute(
                "INSERT INTO messages (id,chat_id,role,content,reasoning,tools,meta,created) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex[:12], chat_id, "assistant", final,
                 "".join(reasoning) or None,
                 json.dumps({"rounds": rounds, "events": tool_events}) if rounds else None,
                 json.dumps({"usage": usage_total, "elapsed_s": elapsed, "model": params.get("model"),
                             "guardrails": h2.check_output(harness, final), "prompt": prompt_report}),
                 time.time()),
            )
            conn.execute("UPDATE chats SET updated=? WHERE id=?", (time.time(), chat_id))
        problems = h2.check_output(harness, final)
        yield sse("done", {"text": final, "usage": usage_total, "elapsed_s": elapsed, "rounds": rounds,
                           "guardrails": problems, "prompt": prompt_report})

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---- static ---------------------------------------------------------------- #

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/healthz", tags=["system"], summary="Liveness probe")
def healthz() -> dict[str, Any]:
    return {"ok": True, "server": server_info(), "job": JOB.snapshot()}


def _startup_discovery() -> None:
    url = resolve_upstream(force=True)
    loader = resolve_loader(force=True)
    print(f"[startup] upstream={url} loader={loader or '(none)'}", flush=True)


def main() -> None:
    global UPSTREAM, PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--upstream", default=UPSTREAM)
    a = ap.parse_args()
    UPSTREAM = a.upstream.rstrip("/")
    PORT = a.port
    init_db()
    ensure_demo_mcp()
    if not SETTINGS_PATH.exists():
        save_settings(DEFAULT_SETTINGS)
    print(f"SGLang Workbench v2  http://{a.host}:{a.port}  -> upstream {UPSTREAM}")
    _startup_discovery()
    import uvicorn

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
