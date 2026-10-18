#!/usr/bin/env python3
"""Host-side model loader for a containerized Workbench.

Why this exists: SGLang on macOS needs Metal and a macOS Python, so it cannot be
spawned from inside a Linux container. The Workbench therefore refuses model
loading there (HTTP 501). Run this agent on the host and point the container at
it — the Workbench then forwards /api/server/{start,stop,cancel,job} to it, and
"Load model" in the UI works exactly as it does natively.

    ./run-loader.sh                                  # 0.0.0.0:43441, token from env
    python loader_agent.py --port 43441 --host 127.0.0.1

Security: with no LOADER_TOKEN this agent lets anyone who can reach it start a
process on your machine. Leave it on loopback, or set LOADER_TOKEN when binding
to the LAN and mirror it as WORKBENCH_LOADER_TOKEN on the container.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory as meminfo  # noqa: E402  (host RAM accounting; local module)
import httpx
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

ROOT = Path(os.environ.get("WORKBENCH_ROOT", "/Users/kmail/sglang"))
PY = os.environ.get("SGLANG_PYTHON") or str(ROOT / ".venv/bin/python")
UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:43440")
SERVER_LOG = Path(os.environ.get("SGLANG_LOG", str(ROOT / "server.log")))
HF_CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
TOKEN = os.environ.get("LOADER_TOKEN", "").strip()
CTX_CEILING = 12288
ACTIVE = ("preparing", "downloading", "starting", "stopping")

app = FastAPI(
    title="SGLang Workbench loader",
    version="1.0.0",
    description=(
        "Host-side process supervisor for SGLang. The containerized Workbench forwards "
        "model start/stop/job calls here because Metal cannot run in a Linux container.\n\n"
        "Set `LOADER_TOKEN` to require `X-Loader-Token`."
    ),
)


@app.middleware("http")
async def token_guard(request: Request, call_next):
    open_paths = ("/healthz", "/docs", "/redoc", "/openapi.json")
    if TOKEN and request.url.path not in open_paths:
        if request.headers.get("x-loader-token") != TOKEN:
            return JSONResponse({"detail": "invalid or missing X-Loader-Token"}, status_code=401)
    return await call_next(request)


class Job:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {
            "state": "idle", "model": None, "progress": 0.0,
            "message": "", "error": None, "started": None, "finished": None,
        }
        self.proc: subprocess.Popen | None = None
        self.cancel = threading.Event()
        self.last_payload: dict[str, Any] | None = None

    def get(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def set(self, **kw: Any) -> None:
        with self.lock:
            self.state.update(kw)


JOB = Job()


def _run(cmd: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


def up() -> bool:
    try:
        return httpx.get(f"{UPSTREAM}/health", timeout=2.5).status_code == 200
    except Exception:
        return False


def log_tail(n: int = 20) -> str:
    if not SERVER_LOG.exists():
        return ""
    return "\n".join(SERVER_LOG.read_text(errors="replace").replace("\r", "\n").splitlines()[-n:])


def cached_size(model: str) -> float:
    d = HF_CACHE / ("models--" + model.replace("/", "--"))
    if not d.exists():
        return 0.0
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def expected_size(model: str) -> float:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model, files_metadata=True)
        return sum((getattr(s, "size", 0) or 0) for s in (info.siblings or [])) or 1.0
    except Exception:
        return 1.0


def download(model: str) -> None:
    from huggingface_hub import snapshot_download

    expected, before = expected_size(model), cached_size(model)
    JOB.set(state="downloading", message=f"fetching {model}", progress=0.0,
            bytes_total=int(expected), bytes_done=int(before))
    err: list[str] = []

    def work() -> None:
        try:
            snapshot_download(repo_id=model, max_workers=4)
        except Exception as e:  # noqa: BLE001
            err.append(f"{type(e).__name__}: {e}")

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while t.is_alive():
        if JOB.cancel.is_set():
            JOB.set(state="cancelled", message="cancelled by user")
            raise RuntimeError("cancelled")
        t.join(1.5)
        done_now = cached_size(model)
        pct = min(99.0, max(0.0, (done_now - before) / max(expected - before, 1) * 100))
        JOB.set(progress=round(pct, 1), message=f"downloading {pct:.0f}%",
                bytes_done=int(done_now), bytes_total=int(expected))
    if err:
        raise RuntimeError(err[0])
    JOB.set(progress=100.0, message="download complete")


OOM_MARKERS = ("Insufficient Memory", "OutOfMemory", "out of memory")


def _oom_retry(payload: dict[str, Any], tail: str) -> dict[str, Any] | None:
    """Halve the context once when Metal ran out of memory, else None."""
    if not any(m.lower() in tail.lower() for m in OOM_MARKERS):
        return None
    if payload.get("_oom_retried"):
        return None
    ctx = int(payload.get("context_length") or CTX_CEILING)
    smaller = max(4096, ctx // 2)
    if smaller >= ctx:
        return None
    return {**payload, "context_length": smaller, "_oom_retried": True}


def worker(payload: dict[str, Any]) -> None:
    model = payload["model"]
    JOB.cancel.clear()
    JOB.set(state="preparing", model=model, progress=0.0, message="checking model",
            error=None, started=time.time(), finished=None)
    try:
        # Download BEFORE touching the running server: switching to an uncached
        # model must not take the old one offline for the whole download.
        if cached_size(model) == 0:
            download(model)
        else:
            JOB.set(progress=100.0, message="already in HF cache")

        if up():
            JOB.set(state="stopping", message="stopping the running server")
            _run(["pkill", "-f", "sglang.launch_server --model-path"])
            for _ in range(45):
                if not up():
                    break
                time.sleep(1)
            _run(["pkill", "-9", "-f", "sglang.launch_server --model-path"])
            time.sleep(2)

        cfg = {
            "context_length": int(payload.get("context_length") or CTX_CEILING),
            "mem_fraction_static": float(payload.get("mem_fraction_static") or 0.30),
            "max_running_requests": int(payload.get("max_running_requests") or 4),
            "chunked_prefill_size": int(payload.get("chunked_prefill_size") or 1024),
            "mlx_cache_gb": float(payload.get("mlx_cache_gb") or 1.0),
            "host": payload.get("host") or "0.0.0.0",
            "port": int(payload.get("port") or 43440),
        }
        JOB.set(state="starting", message="starting sglang", progress=100.0)
        env = {
            **os.environ,
            "SGLANG_USE_MLX": "1",
            "SGLANG_MLX_CACHE_LIMIT_GB": str(cfg["mlx_cache_gb"]),
            "SGLANG_MLX_CLEAR_CACHE_STEPS": "64",
        }
        log = open(SERVER_LOG, "ab")
        log.write(f"\n===== loader {time.strftime('%Y-%m-%d %H:%M:%S')} {model} =====\n".encode())
        log.flush()
        JOB.proc = subprocess.Popen(
            [PY, "-m", "sglang.launch_server",
             "--model-path", model,
             "--host", cfg["host"], "--port", str(cfg["port"]),
             "--mem-fraction-static", str(cfg["mem_fraction_static"]),
             "--max-running-requests", str(cfg["max_running_requests"]),
             "--context-length", str(cfg["context_length"]),
             "--chunked-prefill-size", str(cfg["chunked_prefill_size"])],
            stdout=log, stderr=log, cwd=str(ROOT), env=env,
        )
        deadline = time.time() + 600   # cold weights + Metal warmup can be slow
        while time.time() < deadline:
            if JOB.cancel.is_set():
                _run(["pkill", "-9", "-f", "sglang.launch_server --model-path"])
                JOB.set(state="cancelled", message="cancelled by user")
                return
            if up():
                info: dict[str, Any] = {}
                try:
                    info = httpx.get(f"{UPSTREAM}/get_server_info", timeout=8).json()
                except Exception:
                    pass
                pool = int(info.get("max_total_num_tokens") or 0)
                ctx = int(payload.get("context_length") or CTX_CEILING)
                if 0 < pool < min(4096, ctx) and not payload.get("_pool_fitted"):
                    frac = min(0.55, float(payload.get("mem_fraction_static") or 0.30) + 0.15)
                    fitted = {**payload, "mem_fraction_static": frac,
                              "context_length": min(ctx, 8192), "_pool_fitted": True}
                    print(f"[loader] KV pool only {pool} tokens at ctx {ctx}; retrying with "
                          f"mem-fraction {frac:.2f} ctx {fitted['context_length']}", flush=True)
                    JOB.set(state="starting",
                            message=f"KV pool only {pool} tokens — refitting (mem {frac:.2f}, ctx {fitted['context_length']})")
                    _run(["pkill", "-f", "sglang.launch_server --model-path"])
                    for _ in range(30):
                        if not up():
                            break
                        time.sleep(1)
                    _run(["pkill", "-9", "-f", "sglang.launch_server --model-path"])
                    time.sleep(2)
                    return worker(fitted)
                JOB.last_payload = dict(payload)
                JOB.set(state="ready",
                        message=f"{model} ready" + (f" · pool {pool:,} tok" if pool else ""),
                        finished=time.time(), pool=pool, context_length=ctx)
                return
            if JOB.proc and JOB.proc.poll() is not None:
                tail = log_tail(40)
                retry = _oom_retry(payload, tail)
                if retry:
                    JOB.set(state="starting", message=f"out of memory at ctx {payload.get('context_length')} — retrying with {retry['context_length']}")
                    print(f"[loader] OOM at ctx {payload.get('context_length')}; retrying with {retry['context_length']}", flush=True)
                    return worker(retry)
                if any(m.lower() in tail.lower() for m in OOM_MARKERS):
                    raise RuntimeError(
                        "out of memory: Metal could not allocate this model at this context size. "
                        f"Retry with a smaller context (<= {max(4096, int(payload.get('context_length') or 12288) // 2)}), "
                        "a smaller model, or a larger machine.\n" + tail[-800:])
                raise RuntimeError("server exited during startup:\n" + tail)
            time.sleep(2)
        tail = log_tail(40)
        retry = _oom_retry(payload, tail)
        if retry:
            JOB.set(state="starting", message=f"out of memory — retrying with ctx {retry['context_length']}")
            return worker(retry)
        raise RuntimeError("timed out waiting for /health:\n" + tail)
    except Exception as e:  # noqa: BLE001
        JOB.set(state="failed", error=str(e)[:4000], message="load failed", finished=time.time())


@app.get("/healthz", tags=["system"], summary="Loader liveness")
def healthz() -> dict[str, Any]:
    return {"ok": True, "upstream": UPSTREAM, "engine_up": up(), "job": JOB.get(),
            "python": PY, "python_exists": Path(PY).exists()}


@app.post("/load", tags=["server"], summary="Start (and download if needed) a model")
def load(payload: dict = Body(...)) -> dict[str, Any]:
    if not Path(PY).exists():
        raise HTTPException(501, f"no SGLang interpreter at {PY}; set SGLANG_PYTHON")
    if JOB.get()["state"] in ACTIVE:
        raise HTTPException(409, "a load is already in progress")
    if not payload.get("model"):
        raise HTTPException(400, "model required")
    ctx = int(payload.get("context_length") or CTX_CEILING)
    if ctx > 14336:
        raise HTTPException(400, f"context_length {ctx} exceeds the tested ceiling ({CTX_CEILING} works, "
                                 "16384 dies with a Metal OOM)")
    threading.Thread(target=worker, args=(payload,), daemon=True).start()
    return {"accepted": True, "model": payload["model"]}


@app.get("/memory", tags=["system"], summary="Host memory, live engine numbers and a sizing plan")
def memory(model: str | None = None) -> dict[str, Any]:
    """Unified memory is shared by macOS, Docker and SGLang, so the numbers that
    matter (available RAM, swap, pressure, per-process RSS) only exist on the host.

    For the model that is actually serving, the live figures from /get_server_info are
    authoritative: a *plan* computed while an engine holds the RAM reads as "does not fit"
    for a model that is demonstrably running. A plan is only for a model you are about to load.
    """
    served: dict[str, Any] = {}
    if up():
        try:
            info = httpx.get(f"{UPSTREAM}/get_server_info", timeout=6).json()
            served = {
                "model": info.get("model_path"),
                "context_length": info.get("context_length"),
                "max_total_num_tokens": info.get("max_total_num_tokens"),
                "mem_fraction_static": info.get("mem_fraction_static"),
            }
        except Exception:
            served = {}

    target = model or served.get("model") or (JOB.last_payload or {}).get("model") or ""
    asking_about_live_model = bool(served) and (not model or model == served.get("model"))
    # planning a different model: the served one is stopped first, so its allocation is free
    reclaim = 0.0
    if served and not asking_about_live_model:
        snap_total = meminfo.snapshot()["total_gb"]
        reclaim = round(float(served.get("mem_fraction_static") or 0) * snap_total, 2)
    return {
        "scope": "host",
        "snapshot": meminfo.snapshot(),
        "served": served or None,
        "plan": None if asking_about_live_model else (
            meminfo.plan(target, reclaim_gb=reclaim) if target else None),
        "note": ("live engine figures — stop the model to plan a different one"
                 if asking_about_live_model else
                 "plan for the next load; the engine is stopped before it starts"),
    }


@app.get("/job", tags=["server"], summary="Current job state")
def job() -> dict[str, Any]:
    return JOB.get()


@app.post("/cancel", tags=["server"], summary="Cancel the running load")
def cancel() -> dict[str, Any]:
    JOB.cancel.set()
    return {"cancelling": True}


@app.post("/stop", tags=["server"], summary="Stop the SGLang server")
def stop() -> dict[str, Any]:
    JOB.cancel.set()
    JOB.last_payload = None
    _run(["pkill", "-f", "sglang.launch_server --model-path"])
    for _ in range(40):
        if not up():
            break
        time.sleep(1)
    _run(["pkill", "-9", "-f", "sglang.launch_server --model-path"])
    JOB.set(state="idle", message="stopped", progress=0.0)
    return {"stopped": True}


@app.get("/logs", tags=["server"], summary="Tail the SGLang log")
def logs(tail: int = 200) -> list[str]:
    if not SERVER_LOG.exists():
        return []
    return SERVER_LOG.read_text(errors="replace").replace("\r", "\n").splitlines()[-tail:]


def supervisor(interval: float = 15.0) -> None:
    """Restart the last loaded model if SGLang dies (the loader owns lifecycle)."""
    while True:
        time.sleep(interval)
        try:
            payload = JOB.last_payload
            if not payload or JOB.get()["state"] in ACTIVE:
                continue
            # something may still be warming (job already timed out on /health but the
            # process is alive): do not kill and restart it — that thrash never converges
            if _run(["pgrep", "-f", "sglang.launch_server --model-path"]) is not None:
                running = subprocess.run(["pgrep", "-f", "sglang.launch_server --model-path"],
                                         capture_output=True, text=True).stdout.strip()
                if running:
                    continue
            if not up():
                tail = log_tail(40)
                retry = _oom_retry(payload, tail)
                if retry:
                    retry["max_running_requests"] = 1
                    print(f"[supervisor] OOM crash — restarting {payload['model']} at ctx "
                          f"{retry['context_length']} with 1 concurrent request", flush=True)
                    JOB.last_payload = retry
                    JOB.cancel.clear()
                    worker(dict(retry))
                    continue
                print(f"[supervisor] {payload['model']} down — restarting", flush=True)
                JOB.cancel.clear()
                worker(dict(payload))
        except Exception as e:  # noqa: BLE001
            print(f"[supervisor] error: {type(e).__name__}: {e}", flush=True)


def adopt_running_engine() -> None:
    """A restarted loader must not orphan a running engine: adopt it so the supervisor
    keeps watching it and /memory can report live figures."""
    if not up():
        return
    try:
        info = httpx.get(f"{UPSTREAM}/get_server_info", timeout=8).json()
    except Exception:
        return
    model = info.get("model_path")
    if not model:
        return
    JOB.last_payload = {"model": model,
                        "context_length": info.get("context_length"),
                        "mem_fraction_static": info.get("mem_fraction_static")}
    JOB.set(state="ready", model=model, progress=100.0, finished=time.time(),
            message=f"{model} ready (adopted)")


def main() -> None:
    global UPSTREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=43441)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--upstream", default=UPSTREAM)
    a = ap.parse_args()
    UPSTREAM = a.upstream.rstrip("/")
    print(f"loader agent on http://{a.host}:{a.port}  python={PY}  upstream={UPSTREAM}")
    adopt_running_engine()
    threading.Thread(target=supervisor, daemon=True).start()
    import uvicorn

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
