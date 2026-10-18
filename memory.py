#!/usr/bin/env python3
"""Memory accounting and model sizing for Apple Silicon unified memory.

SGLang on Metal has one hard constraint: weights and the KV pool both come out of
the same 24 GB of unified RAM that macOS, Docker's VM and the browser are using.
Two failure modes follow, and both are decided at launch time:

  * ``--mem-fraction-static`` too high  → the Metal allocator aborts with
    "Command buffer execution failed: Insufficient Memory";
  * too low                             → the server starts but the KV pool is
    a few hundred tokens, so every real prompt is rejected.

This module measures the machine, reads the model's own config, and returns the
largest (context, mem-fraction) pair that still finishes the job. It is used by
``loader_agent.py`` on the host — a container cannot see host memory.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import psutil

HF_CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
DTYPE_BYTES = {
    "float16": 2, "bfloat16": 2, "half": 2, "bf16": 2, "fp16": 2,
    "float32": 4, "float": 4, "fp32": 4, "int8": 1, "uint8": 1,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
}
# bytes per GB that the MLX allocator actually spends (decimal vs binary noise)
GB = 1024 ** 3


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=4)
        return out.stdout.strip() or None
    except Exception:
        return None


def pressure() -> dict[str, Any]:
    """macOS memory-pressure level: 1 normal, 2 warning, 4 critical."""
    raw = _sysctl("kern.memorystatus_vm_pressure_level")
    level = int(raw) if raw and raw.isdigit() else 1
    return {
        "level": level,
        "label": {1: "normal", 2: "warning", 4: "critical"}.get(level, "unknown"),
    }


def _proc_rss(pattern: str) -> tuple[float, int]:
    """Total RSS in GB and process count for processes whose cmdline matches."""
    total = 0
    count = 0
    for p in psutil.process_iter(["cmdline", "memory_info"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if pattern in cmd:
                total += p.info["memory_info"].rss
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return round(total / GB, 2), count


def _engine_rss() -> tuple[float, int]:
    """RSS of every SGLang process — launcher, scheduler and detokenizer are separate,
    and the launcher alone is a rounding error against the weights it owns."""
    total = 0
    count = 0
    for p in psutil.process_iter(["cmdline", "memory_info"]):
        try:
            cmd = p.info["cmdline"] or []
            if any(a == "sglang.launch_server" or a.strip().startswith("sglang::") for a in cmd):
                total += p.info["memory_info"].rss
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return round(total / GB, 2), count


def snapshot() -> dict[str, Any]:
    """Live host memory picture, in GB, plus the processes that compete for it."""
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    sglang_gb, sglang_procs = _engine_rss()
    docker_gb, docker_procs = _proc_rss("Virtualization.VirtualMachine")
    engine_gb, engine_procs = _proc_rss("workbench/app.py")
    return {
        "total_gb": round(vm.total / GB, 2),
        "used_gb": round((vm.total - vm.available) / GB, 2),
        "available_gb": round(vm.available / GB, 2),
        "percent": vm.percent,
        "wired_gb": round(getattr(vm, "wired", 0) / GB, 2),
        "compressed_gb": round(getattr(vm, "compressed", 0) / GB, 2),
        "swap_total_gb": round(sw.total / GB, 2),
        "swap_used_gb": round(sw.used / GB, 2),
        "pressure": pressure(),
        "sglang_gb": sglang_gb,
        "sglang_procs": sglang_procs,
        "docker_vm_gb": docker_gb,
        "docker_vm_procs": docker_procs,
        "workbench_gb": engine_gb,
        "mlx_cache_limit_gb": float(os.environ.get("SGLANG_MLX_CACHE_LIMIT_GB") or 0) or None,
    }


def _snapshot_dir(model: str) -> Path | None:
    root = HF_CACHE / ("models--" + model.replace("/", "--"))
    if not root.exists():
        return None
    snapshots = [p for p in (root / "snapshots").glob("*") if p.is_dir()]
    if not snapshots:
        return None
    return max(snapshots, key=lambda p: p.stat().st_mtime)


def weights_bytes(model: str) -> int:
    """Size of the weight shards actually on disk (0 if the model is not cached)."""
    snap = _snapshot_dir(model)
    if not snap:
        return 0
    total = 0
    for f in snap.iterdir():
        if f.is_file() and f.suffix in {".safetensors", ".bin", ".gguf", ".npz"}:
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def kv_bytes_per_token(model: str) -> int:
    """KV cache cost of one token, from the model's own config.json."""
    snap = _snapshot_dir(model)
    if not snap:
        return 0
    cfg_path = snap / "config.json"
    if not cfg_path.exists():
        return 0
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return 0
    layers = cfg.get("num_hidden_layers") or cfg.get("num_layers") or cfg.get("n_layer") or 0
    heads = cfg.get("num_attention_heads") or cfg.get("n_head") or 0
    kv_heads = cfg.get("num_key_value_heads") or cfg.get("n_head_kv") or heads
    hidden = cfg.get("hidden_size") or cfg.get("n_embd") or 0
    head_dim = cfg.get("head_dim") or (hidden // heads if heads else 0)
    dtype = str(cfg.get("torch_dtype") or cfg.get("dtype") or "float16").lower()
    nbytes = DTYPE_BYTES.get(dtype, 2)
    if not (layers and kv_heads and head_dim):
        return 0
    return int(2 * layers * kv_heads * head_dim * nbytes)  # K and V


def _note(wgb: float, kv_gb: float, alloc_gb: float, frac: float, free: float,
          pool: int, ctx: int, bpt: int, snap_engine: bool = False) -> str:
    head = (f"{wgb:.2f} GB weights + {max(0.0, kv_gb):.2f} GB KV of {alloc_gb:.2f} GB "
            f"(mem-fraction {frac}, {free:.1f} GB free)")
    if not bpt:
        return head + " — KV cost unknown, model config not cached yet"
    if not pool:
        return head + " — the weights alone do not fit; lower the context, pick a smaller model"
    if snap_engine:
        head += " (engine running — RAM returns when it stops)"
    return f"{head}; pool ≈ {pool:,} tokens, ctx {ctx:,}"


def plan(model: str, ctx_ceiling: int = 12288, reserve_gb: float | None = None,
         reclaim_gb: float = 0.0) -> dict[str, Any]:
    """Pick mem-fraction and context length that fit this machine right now.

    The reserve covers macOS itself, Docker's VM, the browser and this agent.
    Everything above it is split between weights and KV cache. ``reclaim_gb`` is memory the
    caller knows will be released first — the running engine, which the loader stops.
    """
    snap = snapshot()
    total = snap["total_gb"]
    # Keep 4-8 GB for the OS and the rest of the desktop; never fewer than 4.
    reserve = reserve_gb if reserve_gb is not None else min(8.0, max(4.0, 0.17 * total))
    budget = max(2.0, total - reserve)

    wbytes = weights_bytes(model)
    wgb = round(wbytes / GB, 2) if wbytes else 0.0
    if not wgb:
        # Not cached yet: estimate from the parameter count in the repo name.
        import re

        m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model)
        wgb = round(float(m.group(1)) * 0.62, 2) if m else 2.0

    # the allocation must also fit what is free *right now* — SGLang cannot use RAM
    # that the rest of the desktop is holding, which is how a 3B model ends up OOMing
    # at a fraction that looks fine against total memory.
    # free RAM is the honest signal: Metal holds weights outside process RSS, so a
    # running engine looks small while its memory is simply gone. The loader stops the
    # engine before a load, and that is the moment these numbers describe.
    # a model that is about to be replaced gives its whole allocation back
    free = round(snap["available_gb"] + reclaim_gb, 2)
    headroom = max(0.22, (free - 1.5) / total)
    frac = round(min(0.62, max(0.22, budget / total), headroom), 3)
    alloc_gb = frac * total
    kv_gb = alloc_gb - wgb - 0.6            # 0.6 GB of runtime/activation overhead
    bpt = kv_bytes_per_token(model) or 0
    # Metal's allocator and SGLang's own buffers take roughly half of the theoretical
    # KV room (calibrated: a 3B at fraction 0.45 yields ~33k tokens, not the ~65k the
    # naive arithmetic suggests), so the estimate stays conservative.
    pool = int(max(0.0, kv_gb) * GB / bpt * 0.5) if bpt else 0

    ctx = ctx_ceiling
    if pool:
        ctx = min(ctx_ceiling, max(2048, int(pool * 0.8) // 1024 * 1024))
    fits = kv_gb > 0.5 and (pool == 0 or pool >= 2048)

    return {
        "model": model,
        "total_gb": total,
        "reserve_gb": round(reserve, 2),
        "budget_gb": round(budget, 2),
        "mem_fraction_static": frac,
        "allocated_gb": round(alloc_gb, 2),
        "weights_gb": wgb,
        "weights_measured": bool(wbytes),
        "kv_gb": round(kv_gb, 2),
        "kv_bytes_per_token": bpt,
        "pool_tokens": pool,
        "context_length": ctx,
        "feasible": fits,
        "note": _note(wgb, kv_gb, alloc_gb, frac, free, pool, ctx, bpt,
                      bool(snap["sglang_procs"])),
    }


if __name__ == "__main__":  # pragma: no cover - manual probe
    import sys

    mdl = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
    print(json.dumps({"snapshot": snapshot(), "plan": plan(mdl)}, indent=2))
