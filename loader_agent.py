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

Downloading, spawning and the OOM/KV-pool retries are `loader_core.run_load`, the
same code the native workbench runs — the two deployments cannot drift apart.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402
from fastapi import Body, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

import memory as meminfo  # noqa: E402  (host RAM accounting; local module)
from loader_core import (CTX_CEILING, PROC_PATTERN, Job, cached_size, hf_size,  # noqa: E402
                         log_tail, oom_retry, run, run_load, stop_engine)

ROOT = Path(os.environ.get("WORKBENCH_ROOT", "/Users/kmail/sglang"))
PY = os.environ.get("SGLANG_PYTHON") or str(ROOT / ".venv/bin/python")
UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:43440")
SERVER_LOG = Path(os.environ.get("SGLANG_LOG", str(ROOT / "server.log")))
HF_CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
TOKEN = os.environ.get("LOADER_TOKEN", "").strip()

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


JOB = Job(pool=0, context_length=CTX_CEILING)


def up() -> bool:
    try:
        return httpx.get(f"{UPSTREAM}/health", timeout=2.5).status_code == 200
    except Exception:
        return False


def engine_info() -> dict[str, Any]:
    try:
        return httpx.get(f"{UPSTREAM}/get_server_info", timeout=8).json()
    except Exception:
        return {}


def _cfg(payload: dict[str, Any]) -> dict[str, Any]:
    """Launch parameters for this load: the payload, backed by loader defaults."""
    return {
        "context_length": int(payload.get("context_length") or CTX_CEILING),
        "mem_fraction_static": float(payload.get("mem_fraction_static") or 0.30),
        "max_running_requests": int(payload.get("max_running_requests") or 4),
        "chunked_prefill_size": int(payload.get("chunked_prefill_size") or 1024),
        "mlx_cache_gb": float(payload.get("mlx_cache_gb") or 1.0),
        "host": payload.get("host") or "0.0.0.0",
        "port": int(payload.get("port") or 43440),
    }


def _on_ready(cfg: dict[str, Any], model: str, info: dict[str, Any]) -> None:
    pool = int(info.get("max_total_num_tokens") or 0)
    # the supervisor restarts from this payload, so it has to carry the context
    # and memory fraction that actually produced a working engine
    JOB.last_payload = {**cfg, "model": model}
    JOB.set(state="ready",
            message=f"{model} ready" + (f" · pool {pool:,} tok" if pool else ""),
            finished=time.time(), pool=pool, context_length=cfg["context_length"])


def worker(payload: dict[str, Any]) -> None:
    run_load(
        JOB, payload,
        python=PY, hf_cache=HF_CACHE, log_path=SERVER_LOG, root=ROOT,
        probe_up=up,
        build_cfg=_cfg,
        is_cached=lambda m: cached_size(HF_CACHE, m) > 0,
        expected_bytes=hf_size,
        engine_info=engine_info,
        on_ready=_on_ready,
        fit_pool=True,        # an engine that started with a useless KV pool is not a load
        ctx_ceiling=CTX_CEILING,
        timeout=600.0,        # cold weights + Metal warmup can be slow
        tag="loader",
    )


@app.get("/healthz", tags=["system"], summary="Loader liveness")
def healthz() -> dict[str, Any]:
    return {"ok": True, "upstream": UPSTREAM, "engine_up": up(), "job": JOB.snapshot(),
            "python": PY, "python_exists": Path(PY).exists()}


@app.post("/load", tags=["server"], summary="Start (and download if needed) a model")
def load(payload: dict = Body(...)) -> dict[str, Any]:
    if not Path(PY).exists():
        raise HTTPException(501, f"no SGLang interpreter at {PY}; set SGLANG_PYTHON")
    if JOB.busy:
        raise HTTPException(409, "a load is already in progress")
    if not payload.get("model"):
        raise HTTPException(400, "model required")
    ctx = int(payload.get("context_length") or CTX_CEILING)
    if ctx > 14336:
        raise HTTPException(400, f"context_length {ctx} exceeds the tested ceiling "
                                 f"({CTX_CEILING} works, 16384 dies with a Metal OOM)")
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
        info = engine_info()
        if info:
            served = {
                "model": info.get("model_path"),
                "context_length": info.get("context_length"),
                "max_total_num_tokens": info.get("max_total_num_tokens"),
                "mem_fraction_static": info.get("mem_fraction_static"),
            }

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
    return JOB.snapshot()


@app.post("/cancel", tags=["server"], summary="Cancel the running load")
def cancel() -> dict[str, Any]:
    JOB.cancel.set()
    return {"cancelling": True}


@app.post("/stop", tags=["server"], summary="Stop the SGLang server")
def stop() -> dict[str, Any]:
    JOB.last_payload = None
    return stop_engine(JOB, up)


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
            if not payload or JOB.busy:
                continue
            # something may still be warming (the job already timed out on /health but
            # the process is alive): do not kill and restart it, that never converges
            pgrep = run(["pgrep", "-f", PROC_PATTERN])
            if pgrep is not None and pgrep.stdout.strip():
                continue
            if not up():
                tail = log_tail(SERVER_LOG, 40)
                retry = oom_retry(payload, tail, CTX_CEILING)
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
    info = engine_info()
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
