"""Tests for loader_core: the download/spawn/OOM-retry loop both loaders share.

Run with:  python3 -m unittest discover -s tests -v
No SGLang, no GPU: a shell script stands in for the interpreter and serves
/health from a throwaway HTTP server.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import loader_core as lc

FAKE_ENGINE = '''\
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = 43440
argv = sys.argv[1:]
for i, a in enumerate(argv):
    if a == "--port":
        port = int(argv[i + 1])

MODE = "{mode}"

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass
    def do_GET(self):
        if self.path == "/health":
            body = b'{{"ok":true}}'
        else:
            body = json.dumps({{"model_path": "fake/model", "context_length": 8192,
                               "max_total_num_tokens": 7000}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if MODE == "oom":
    print("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB", flush=True)
    sys.exit(3)

ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''

FAKE_PYTHON = """#!/bin/sh
exec "{python}" "{script}" "$@"
"""


class LoaderCoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="loader-test-"))
        self.log = self.tmp / "server.log"
        self.engine_script = self.tmp / "fake_engine.py"
        self.wrapper = self.tmp / "fake_python.sh"
        self.ready: list[tuple[dict, str, dict]] = []
        self.port = 43440
        self.jobs: list[lc.Job] = []

    def job(self) -> lc.Job:
        job = lc.Job()
        self.jobs.append(job)
        return job

    def _install_fake(self, mode: str = "up") -> None:
        self.engine_script.write_text(FAKE_ENGINE.format(mode=mode))
        self.wrapper.write_text(FAKE_PYTHON.format(python=sys.executable,
                                                   script=self.engine_script))
        self.wrapper.chmod(0o755)

    def _probe(self) -> bool:
        import httpx

        try:
            return httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=0.4).status_code == 200
        except Exception:
            return False

    def _run(self, job, payload, *, fit_pool=False, timeout=8.0):
        def on_ready(cfg, model, info):
            self.ready.append((cfg, model, info))
            job.set(state="ready", message=f"{model} ready", finished=time.time())

        return lc.run_load(
            job, payload,
            python=str(self.wrapper),
            hf_cache=self.tmp / "hf",
            log_path=self.log,
            root=self.tmp,
            probe_up=self._probe,
            build_cfg=lambda p: {
                "context_length": int(p.get("context_length") or lc.CTX_CEILING),
                "mem_fraction_static": float(p.get("mem_fraction_static") or 0.30),
                "max_running_requests": int(p.get("max_running_requests") or 4),
                "chunked_prefill_size": 1024,
                "mlx_cache_gb": 1.0,
                "host": "127.0.0.1",
                "port": self.port,
            },
            is_cached=lambda m: True,
            expected_bytes=lambda m: 1.0,
            engine_info=lambda: {"max_total_num_tokens": 7000},
            on_ready=on_ready,
            fit_pool=fit_pool,
            timeout=timeout,
            tag="test",
        )

    def tearDown(self):
        lc.run(["pkill", "-9", "-f", str(self.engine_script)])
        for job in self.jobs:
            if job.proc is not None:
                job.proc.terminate()
                try:
                    job.proc.wait(timeout=5)
                except Exception:
                    job.proc.kill()

    def test_cached_model_starts_and_reports_ready(self):
        self._install_fake("up")
        job = self.job()
        self._run(job, {"model": "fake/model", "context_length": 6144})

        self.assertEqual(job.snapshot()["state"], "ready", job.snapshot())
        self.assertEqual(len(self.ready), 1)
        cfg, model, info = self.ready[0]
        self.assertEqual(model, "fake/model")
        self.assertEqual(cfg["context_length"], 6144)
        self.assertEqual(info["max_total_num_tokens"], 7000)
        self.assertIn("test load", self.log.read_text())

    def test_oom_in_log_retries_once_with_half_context_then_fails(self):
        self._install_fake("oom")
        job = lc.Job()
        started = time.time()
        self._run(job, {"model": "fake/model", "context_length": 8192}, timeout=6)
        snap = job.snapshot()

        self.assertEqual(snap["state"], "failed", snap)
        self.assertIn("out of memory", (snap["error"] or "").lower(), snap)
        log = self.log.read_text()
        # spawn ran twice: the retry halves the context
        self.assertEqual(log.count("test load"), 2, log[:600])
        self.assertLess(time.time() - started, 20)

    def test_cancel_stops_the_engine(self):
        self._install_fake("up")
        job = self.job()
        # cancel while the loader is still waiting for /health
        threading.Timer(2.5, job.cancel.set).start()

        lc.run_load(
            job, {"model": "fake/model"},
            python=str(self.wrapper), hf_cache=self.tmp / "hf", log_path=self.log,
            root=self.tmp, probe_up=lambda: False,
            build_cfg=lambda p: {"context_length": 4096, "mem_fraction_static": 0.3,
                                 "max_running_requests": 1, "chunked_prefill_size": 512,
                                 "host": "127.0.0.1", "port": self.port},
            is_cached=lambda m: True, expected_bytes=lambda m: 1.0,
            timeout=8.0, tag="test",
        )
        self.assertEqual(job.snapshot()["state"], "cancelled", job.snapshot())


class HelpersTest(unittest.TestCase):
    def test_oom_retry_halves_once(self):
        retry = lc.oom_retry({"context_length": 8192}, "CUDA out of memory")
        assert retry is not None
        self.assertEqual(retry["context_length"], 4096)
        self.assertIsNone(lc.oom_retry({"context_length": 8192, "_oom_retried": True},
                                       "out of memory"))
        self.assertIsNone(lc.oom_retry({"context_length": 4096}, "out of memory"))  # floor
        self.assertIsNone(lc.oom_retry({"context_length": 8192}, "all good"))

    def test_pool_refit_only_for_a_useless_pool(self):
        fit = lc.pool_refit({"context_length": 8192, "mem_fraction_static": 0.30},
                            {"max_total_num_tokens": 300})
        self.assertIsNotNone(fit)
        self.assertGreater(fit["mem_fraction_static"], 0.30)
        self.assertLessEqual(fit["context_length"], 8192)
        self.assertIsNone(lc.pool_refit({"context_length": 8192}, {"max_total_num_tokens": 7000}))
        self.assertIsNone(lc.pool_refit({"context_length": 8192, "_pool_fitted": True},
                                        {"max_total_num_tokens": 300}))

    def test_cached_size_walks_the_hf_layout(self):
        tmp = Path(tempfile.mkdtemp(prefix="hf-"))
        hub = tmp / "hub"
        d = hub / "models--org--name" / "snapshots" / "abc"
        d.mkdir(parents=True)
        (d / "weights.bin").write_bytes(b"x" * 1000)
        self.assertEqual(lc.cached_size(hub, "org/name"), 1000)
        self.assertEqual(lc.cached_size(hub, "org/absent"), 0)

    def test_log_tail_reads_the_last_lines(self):
        tmp = Path(tempfile.mkdtemp(prefix="log-"))
        p = tmp / "s.log"
        p.write_text("\n".join(f"line {i}" for i in range(50)))
        self.assertEqual(lc.log_tail(p, 3), "line 47\nline 48\nline 49")
        self.assertEqual(lc.log_tail(tmp / "missing.log"), "")

    def test_job_busy_tracks_active_states(self):
        job = lc.Job()
        self.assertFalse(job.busy)
        for state in lc.ACTIVE_STATES:
            job.set(state=state)
            self.assertTrue(job.busy, state)
        job.set(state="ready")
        self.assertFalse(job.busy)


if __name__ == "__main__":
    unittest.main()
