#!/usr/bin/env python3
"""Model-loading primitives shared by the workbench and the host-side loader agent.

`app.py` (native, on the Mac) and `loader_agent.py` (host supervisor for a
containerized workbench) both download a model and then start
`sglang.launch_server`, watching the job state and the log for the failure modes
that matter on Metal: running out of memory, and a KV pool too small to serve
the requested context. That logic used to be copied into both files and had
already drifted apart; it lives here once.

Nothing here imports FastAPI: the loader agent runs on the host with a bare
Python and needs only the stdlib plus the optional `huggingface_hub`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

CTX_CEILING = 12288   # tested: 12k prefill passes, 16k Metal-OOMs on the 16 GB box
ACTIVE_STATES = ("preparing", "downloading", "starting", "stopping")
OOM_MARKERS = ("Insufficient Memory", "OutOfMemory", "out of memory")
PROC_PATTERN = "sglang.launch_server --model-path"


class Job:
    """Thread-safe progress state for one background lifecycle worker."""

    def __init__(self, **defaults: Any) -> None:
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen | None = None
        self.last_payload: dict[str, Any] | None = None
        self.state: dict[str, Any] = {
            "state": "idle", "model": None, "progress": 0.0,
            "message": "", "error": None, "started": None, "finished": None,
            **defaults,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def get(self) -> dict[str, Any]:
        return self.snapshot()

    def set(self, **kw: Any) -> None:
        with self.lock:
            self.state.update(kw)

    # older call sites still spell it `_set`
    _set = set

    @property
    def busy(self) -> bool:
        return self.snapshot()["state"] in ACTIVE_STATES


def run(cmd: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess | None:
    """subprocess.run that survives slim images with no pgrep/ps/pkill."""
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None


def log_tail(path: Path, n: int = 25) -> str:
    try:
        if not path.exists():
            return ""
        lines = path.read_text(errors="replace").replace("\r", "\n").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def cached_size(hf_cache: Path, model: str) -> float:
    """Bytes of `model` already on disk in the Hugging Face cache (0 when absent)."""
    d = hf_cache / ("models--" + model.replace("/", "--"))
    if not d.exists():
        return 0.0
    try:
        return float(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()))
    except OSError:
        return 0.0


def hf_size(model: str) -> float:
    """Total bytes the repo will occupy, from the HF API (1.0 when unknown)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model, files_metadata=True)
        return float(sum((getattr(s, "size", 0) or 0) for s in (info.siblings or []))) or 1.0
    except Exception:
        return 1.0


def download(job: Job, hf_cache: Path, model: str, expected: float) -> None:
    """Blocking snapshot_download with progress and a cooperative cancel."""
    from huggingface_hub import snapshot_download

    before = cached_size(hf_cache, model)
    job.set(state="downloading", message=f"fetching {model}", progress=0.0,
            bytes_total=int(expected), bytes_done=int(before))
    err: list[str] = []

    def work() -> None:
        try:
            snapshot_download(repo_id=model, max_workers=4)
        except Exception as exc:  # noqa: BLE001
            err.append(f"{type(exc).__name__}: {exc}")

    t = threading.Thread(target=work, name=f"download-{model}", daemon=True)
    t.start()
    while t.is_alive():
        if job.cancel.is_set():
            job.set(state="cancelled", message="cancelled by user")
            raise RuntimeError("cancelled")
        t.join(1.5)
        done = cached_size(hf_cache, model)
        pct = min(99.0, max(0.0, (done - before) / max(expected - before, 1) * 100))
        job.set(progress=round(pct, 1), message=f"downloading {pct:.0f}%",
                bytes_done=int(done), bytes_total=int(expected))
    if err:
        raise RuntimeError(err[0])
    job.set(progress=100.0, message="download complete")


def oom_retry(payload: dict[str, Any], tail: str,
              ctx_ceiling: int = CTX_CEILING) -> dict[str, Any] | None:
    """Halve the context once when the log says Metal ran out of memory."""
    if payload.get("_oom_retried"):
        return None
    if not any(m.lower() in tail.lower() for m in OOM_MARKERS):
        return None
    ctx = int(payload.get("context_length") or ctx_ceiling)
    smaller = max(4096, ctx // 2)
    if smaller >= ctx:
        return None
    return {**payload, "context_length": smaller, "_oom_retried": True}


def pool_refit(payload: dict[str, Any], info: dict[str, Any],
               ctx_ceiling: int = CTX_CEILING) -> dict[str, Any] | None:
    """Retry with a wider KV pool when the engine came up but can barely serve.

    A model can start and still be useless: a small `mem-fraction-static` leaves
    a pool that cannot hold one useful prompt. Raising the fraction is what
    turns "it runs" into "it answers".
    """
    if payload.get("_pool_fitted"):
        return None
    pool = int(info.get("max_total_num_tokens") or 0)
    ctx = int(payload.get("context_length") or ctx_ceiling)
    if not (0 < pool < min(4096, ctx)):
        return None
    frac = min(0.55, float(payload.get("mem_fraction_static") or 0.30) + 0.15)
    return {**payload, "mem_fraction_static": frac,
            "context_length": min(ctx, 8192), "_pool_fitted": True}


def spawn(job: Job, *, python: str, model: str, cfg: dict[str, Any],
          log_path: Path, root: Path, tag: str = "workbench") -> None:
    """Start the engine with its output appended to the server log."""
    cmd = [
        python, "-m", "sglang.launch_server",
        "--model-path", model,
        "--host", str(cfg.get("host") or "0.0.0.0"),
        "--port", str(cfg.get("port") or 43440),
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    log.write(f"\n===== {tag} load {time.strftime('%Y-%m-%d %H:%M:%S')} {model} =====\n".encode())
    log.flush()
    job.proc = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=str(root), env=env)
    log.close()      # the child holds its own descriptor; do not leak ours


def stop_engine(job: Job, probe_up: Callable[[], bool], *,
                wait_s: int = 40) -> dict[str, Any]:
    """Signal the running engine and wait for it to let go of /health."""
    job.cancel.set()
    if not shutil.which("pkill"):
        return {"stopped": False,
                "error": "this deployment cannot signal processes (no pkill); "
                         "stop SGLang where it runs"}
    run(["pkill", "-f", PROC_PATTERN])
    for _ in range(wait_s):
        if not probe_up():
            break
        time.sleep(1)
    run(["pkill", "-9", "-f", PROC_PATTERN])
    job.set(state="idle", message="stopped", progress=0.0)
    return {"stopped": True}


def run_load(
    job: Job,
    payload: dict[str, Any],
    *,
    python: str,
    hf_cache: Path,
    log_path: Path,
    root: Path,
    probe_up: Callable[[], bool],
    build_cfg: Callable[[dict[str, Any]], dict[str, Any]],
    is_cached: Callable[[str], bool],
    expected_bytes: Callable[[str], float],
    engine_info: Callable[[], dict[str, Any]] | None = None,
    on_ready: Callable[[dict[str, Any], str, dict[str, Any]], None] | None = None,
    fit_pool: bool = False,
    ctx_ceiling: int = CTX_CEILING,
    timeout: float = 600.0,
    tag: str = "workbench",
) -> None:
    """Download if needed, (re)start the engine, wait for /health, report.

    Every outcome lands in `job.state`; this never raises. `on_ready` receives
    `(cfg, model, engine_info)` once the engine answers. An OOM or an undersized
    KV pool restarts once with a smaller context / wider pool.
    """
    retry_kwargs: dict[str, Any] = dict(python=python, hf_cache=hf_cache, log_path=log_path,
                        root=root, probe_up=probe_up, build_cfg=build_cfg,
                        is_cached=is_cached, expected_bytes=expected_bytes,
                        engine_info=engine_info, on_ready=on_ready,
                        fit_pool=fit_pool, ctx_ceiling=ctx_ceiling,
                        timeout=timeout, tag=tag)
    model = payload["model"]
    cfg = build_cfg(payload)
    job.cancel.clear()
    job.set(state="preparing", model=model, progress=0.0, message="checking model",
            error=None, started=time.time(), finished=None)
    try:
        if is_cached(model):
            job.set(progress=100.0, message="already in HF cache")
        else:
            # download BEFORE touching the running server: switching to an
            # uncached model must not take the old one offline for the download
            download(job, hf_cache, model, expected_bytes(model))

        if probe_up():
            job.set(state="stopping", message="stopping the running server")
            run(["pkill", "-f", PROC_PATTERN])
            for _ in range(45):
                if not probe_up():
                    break
                time.sleep(1)
            run(["pkill", "-9", "-f", PROC_PATTERN])
            time.sleep(2)

        job.set(state="starting", message="starting sglang", progress=100.0)
        spawn(job, python=python, model=model, cfg=cfg, log_path=log_path,
              root=root, tag=tag)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if job.cancel.is_set():
                run(["pkill", "-9", "-f", PROC_PATTERN])
                job.set(state="cancelled", message="cancelled by user")
                return
            if probe_up():
                info = engine_info() if engine_info else {}
                if fit_pool:
                    fitted = pool_refit(payload, info, ctx_ceiling)
                    if fitted:
                        print(f"[{tag}] KV pool only {info.get('max_total_num_tokens')} "
                              f"tokens; retrying with mem-fraction "
                              f"{fitted['mem_fraction_static']:.2f} ctx "
                              f"{fitted['context_length']}", flush=True)
                        job.set(state="starting",
                                message=(f"KV pool only {info.get('max_total_num_tokens')} "
                                         f"tokens — refitting (mem "
                                         f"{fitted['mem_fraction_static']:.2f}, ctx "
                                         f"{fitted['context_length']})"))
                        run(["pkill", "-f", PROC_PATTERN])
                        for _ in range(30):
                            if not probe_up():
                                break
                            time.sleep(1)
                        run(["pkill", "-9", "-f", PROC_PATTERN])
                        time.sleep(2)
                        return run_load(job, fitted, **retry_kwargs)
                if on_ready:
                    on_ready(cfg, model, info)
                return
            if job.proc and job.proc.poll() is not None:
                tail = log_tail(log_path, 40)
                retry = oom_retry(payload, tail, ctx_ceiling)
                if retry:
                    job.set(state="starting",
                            message=(f"out of memory — retrying with ctx "
                                     f"{retry['context_length']}"))
                    return run_load(job, retry, **retry_kwargs)
                if any(m.lower() in tail.lower() for m in OOM_MARKERS):
                    raise RuntimeError(
                        "out of memory: Metal could not allocate this model at this "
                        "context size. Retry with a smaller context "
                        f"(<= {max(4096, int(payload.get('context_length') or ctx_ceiling) // 2)}), "
                        "a smaller model, or a larger machine.\n" + tail[-800:])
                raise RuntimeError("server exited during startup:\n" + tail)
            time.sleep(2)
        tail = log_tail(log_path, 40)
        retry = oom_retry(payload, tail, ctx_ceiling)
        if retry:
            job.set(state="starting",
                    message=f"out of memory — retrying with ctx {retry['context_length']}")
            return run_load(job, retry, **retry_kwargs)
        raise RuntimeError("timed out waiting for /health:\n" + tail)
    except Exception as exc:  # noqa: BLE001
        job.set(state="failed", error=str(exc)[:4000], message="load failed",
                finished=time.time())
