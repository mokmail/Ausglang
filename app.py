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
  harness    model-driven tool loop with a JSON protocol any local model can follow
  prompts    reusable prompt library
  personas   saved system-prompt + sampling profiles
  settings   temperature/top_p/top_k/max_tokens/repeat penalty/seed/stop/json mode
  stats      per-day message and token accounting
"""

from __future__ import annotations

import argparse
import asyncio
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
from fastapi import Body, FastAPI, HTTPException, Query
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
SERVER_LOG = Path(os.environ.get("SGLANG_LOG", str(ROOT / "server.log")))

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:43440")
PORT = int(os.environ.get("WORKBENCH_PORT", "8090"))
# Path to the python that can run `-m sglang.launch_server`. Empty/absent means
# model loading is disabled here (typical inside a container, where the GPU
# stack lives on the host) and the API returns a clear error instead of dying.
PY = os.environ.get("SGLANG_PYTHON") or str(ROOT / ".venv/bin/python")
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
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
            CREATE TABLE IF NOT EXISTS lessons (
                id TEXT PRIMARY KEY, domain TEXT, cue TEXT, lesson TEXT,
                evidence TEXT, confidence TEXT, created REAL
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


def server_up() -> bool:
    try:
        return httpx.get(f"{UPSTREAM}/health", timeout=1.2).status_code == 200
    except Exception:
        return False


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


def load_worker(payload: dict[str, Any]) -> None:
    model = payload["model"]
    settings = load_settings()
    cfg = {**settings, **{k: v for k, v in payload.items() if v is not None}}
    JOB.cancel.clear()
    JOB._set(state="preparing", model=model, progress=0.0, message="checking model",
             error=None, started=time.time(), finished=None)
    try:
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

        detail = model_detail(model)
        if not detail["downloaded"]:
            _download(model)
        else:
            JOB._set(progress=100.0, message="already in HF cache")

        JOB._set(state="starting", message="starting sglang", progress=100.0)
        _spawn(model, cfg)
        deadline = time.time() + 240
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
                JOB._set(state="ready", message=f"{model} ready", finished=time.time())
                return
            if JOB.proc and JOB.proc.poll() is not None:
                raise RuntimeError("server exited during startup:\n" + _log_tail(20))
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


def list_harnesses() -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM harnesses ORDER BY created").fetchall()
    out = []
    for r in rows:
        comps = h2.normalize(json.loads(r["components"] or "{}"))
        out.append({"id": r["id"], "name": r["name"], **comps,
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
    return {"id": row["id"], "name": row["name"], **comps}


def save_harness(harness_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    current = get_harness(harness_id)
    merged = h2.normalize({**current, **patch})
    with db() as conn:
        conn.execute("UPDATE harnesses SET name=?, components=?, updated=? WHERE id=?",
                     (merged.get("name") or "Harness", json.dumps(merged), h2.now(), harness_id))
    return {"id": harness_id, **merged}


def lessons_for(query: str) -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM lessons ORDER BY created DESC LIMIT 200").fetchall()
    return h2.lesson_context([dict(r) for r in rows], query)


def effective_tools(harness: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    """MCP tools after the harness' allowlist + tool guidance, plus the delegate tool."""
    if not params.get("tools_enabled"):
        return []
    tools = h2.filter_tools(tool_schemas(), harness["rules"].get("tool_allowlist") or [])
    tools = h2.apply_tool_guidance(tools, harness.get("tool_guidance") or {})
    # c5: every subagent is callable by its own name (models naturally emit that),
    # plus a generic "delegate" escape hatch.
    for sub in harness.get("subagents") or []:
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


@app.get("/api/state", tags=["system"], summary="Full runtime snapshot")
def api_state() -> dict[str, Any]:
    s = load_settings()
    with db() as conn:
        counts = {
            "chats": conn.execute("SELECT COUNT(*) c FROM chats").fetchone()["c"],
            "memory": conn.execute("SELECT COUNT(*) c FROM memory").fetchone()["c"],
            "prompts": conn.execute("SELECT COUNT(*) c FROM prompts").fetchone()["c"],
            "personas": conn.execute("SELECT COUNT(*) c FROM personas").fetchone()["c"],
        }
    return {
        "server": server_info(),
        "job": JOB.snapshot(),
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
    }


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


@app.get("/api/harnesses/{hid}/preview", tags=["harness"], summary="Compose the system prompt without sending it")
def api_harness_preview(hid: str, q: str = "") -> dict[str, Any]:
    harness = get_harness(hid)
    lessons = lessons_for(q) if q else []
    text, report = h2.compose_system(harness, lessons=lessons, memory_text=memory_context(q) if q else "")
    return {"harness": harness.get("name"), "system": text, "report": report}


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
    lid = h2.new_id()
    with db() as conn:
        conn.execute(
            "INSERT INTO lessons (id,domain,cue,lesson,evidence,confidence,created) VALUES (?,?,?,?,?,?,?)",
            (lid, payload.get("domain") or "general", str(parsed.get("cue") or "")[:300],
             str(parsed.get("lesson") or raw.strip()[:400]), str(parsed.get("evidence") or "")[:600],
             str(parsed.get("confidence") or "low"), h2.now()),
        )
    return {"id": lid, **parsed}


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


@app.post("/api/chat", tags=["chat"], summary="Stream a completion (tool harness)")
async def api_chat(req: ChatRequest = Body(...)) -> StreamingResponse:
    payload = req.model_dump()
    chat_id = payload.get("chat_id")
    regenerate = bool(payload.get("regenerate"))
    text = (payload.get("message") or "").strip()
    attachments = payload.get("attachments") or []
    if not chat_id or (not text and not regenerate):
        raise HTTPException(400, "chat_id and message required (or regenerate=true)")

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
            yield sse("tool", {"tool": tool_name, "server": server, "arguments": args, "state": "running"})
            subagent_names = {x.get("name") for x in (harness.get("subagents") or [])}
            if tool_name == "delegate" or tool_name in subagent_names:
                payload = {"name": args.get("name") or tool_name, "task": args.get("task") or args.get("input") or ""}
                result = await run_delegate(harness, payload, params)
            else:
                result = await run_tool(server, tool_name, args)
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
                 json.dumps({"rounds": rounds}) if rounds else None,
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
    import uvicorn

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
