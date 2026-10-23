"""Tests for the PDF upload -> model prompt -> tool flow.

These are the paths that broke "upload a PDF and ask to extract the text":
the upload bypassed the PDF engine, the model never learned the file path,
and the PDF tools were not offered when the tools switch was off.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Isolate the app from the shared data directory: the registry there points at
# the container's interpreter, which does not exist on the host, and writing to
# it as a different user breaks the container's SQLite WAL.
import os  # noqa: E402
import tempfile as _tempfile  # noqa: E402

_TEST_DATA = Path(_tempfile.mkdtemp(prefix="pdfmcp-appdata-"))
os.environ["WORKBENCH_DATA"] = str(_TEST_DATA)
os.environ.setdefault("WORKBENCH_ROOT", str(Path(__file__).resolve().parent.parent))

from pdfmcp import extract_for_prompt  # noqa: E402
from test_pdfmcp import _make_pdf  # noqa: E402


def _register_local_servers() -> None:
    """Point the app at MCP servers this interpreter can actually run."""
    root = Path(__file__).resolve().parent.parent
    servers = [
        {"name": "demo", "command": sys.executable, "args": [str(root / "mcp_demo_server.py")],
         "env": {}, "enabled": True, "bundled": True},
        {"name": "pdf", "command": sys.executable, "args": [str(root / "pdfmcp_server.py")],
         "env": {}, "enabled": True, "bundled": True},
    ]
    (_TEST_DATA / "mcp_servers.json").write_text(json.dumps(servers, indent=2), encoding="utf-8")


_register_local_servers()


def _load_app():
    """Import the app, or None when its dependencies are not installed."""
    try:
        import app as A  # noqa: PLC0415
    except Exception:
        return None
    return A


APP = _load_app()


def _app():
    """The imported app module, or skip the test."""
    if APP is None:
        raise unittest.SkipTest("the app's dependencies are not installed")
    return APP


class UploadExtractionTest(unittest.TestCase):
    """The upload endpoint must use the PDF engine, not naive pypdf text."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pdfmcp-up-"))
        self.pdf = self.tmp / "paper.pdf"
        _make_pdf(self.pdf)
        self.out = self.tmp / "out"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_upload_extracts_structured_markdown(self) -> None:
        text, meta = extract_for_prompt(self.pdf)
        self.assertIn("Alpine Geology", text)
        self.assertEqual(meta["kind"], "pdf")
        self.assertEqual(meta["pages"], 1)
        self.assertGreaterEqual(meta["topics"], 1)

    def test_upload_writes_manifest_and_images(self) -> None:
        text, meta = extract_for_prompt(self.pdf, out_dir=self.out)
        manifest = Path(meta["manifest"])
        self.assertTrue(manifest.exists())
        m = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertTrue(m["texts"])
        for entry in m["texts"]:
            self.assertIn("page", entry)
            self.assertIn("topic", entry)
        self.assertTrue(Path(meta["markdown"]).exists())
        self.assertTrue(Path(meta["images_dir"]).is_dir())

    def test_prompt_copy_has_no_block_id_noise(self) -> None:
        """The copy in the prompt keeps page markers but drops the per-block
        id comments -- they are noise to a model."""
        text, _ = extract_for_prompt(self.pdf)
        self.assertNotIn("<!-- id:", text)
        self.assertIn("<!-- page:", text)

    def test_scanned_pdf_is_reported_not_raised(self) -> None:
        """A PDF with no text layer must not blow up the upload."""
        scanned = self.tmp / "scan.pdf"
        shutil.copyfile(self.pdf, scanned)
        # strip the text from the content stream -> an image-only page
        raw = scanned.read_bytes()
        raw = raw.replace(b"BT\n", b"XX\n").replace(b" Tj", b" xx")
        scanned.write_bytes(raw)
        try:
            text, meta = extract_for_prompt(scanned)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"a scanned PDF must not raise: {type(exc).__name__}: {exc}")
        if "Alpine Geology" not in text:
            self.assertIn("warning", meta)
            self.assertIn("scan", meta["warning"].lower())


@unittest.skipUnless(APP is not None, "the app's dependencies are not installed")
class ToolAvailabilityTest(unittest.TestCase):
    """A PDF attachment must bring the PDF tools, even with tools switched off.

    These exercise app-level behaviour, so they need the app's own
    dependencies (httpx and friends) -- not just the PDF engines.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pdfmcp-tools-"))
        self.pdf = self.tmp / "paper.pdf"
        _make_pdf(self.pdf)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _effective_tools(self, attachments, tools_enabled):
        A = _app()
        A.init_db()   # the isolated data dir starts empty
        harness = A.get_harness(None, None)
        return A.effective_tools(harness, {"tools_enabled": tools_enabled}, attachments)

    def test_pdf_attachment_brings_the_pdf_tools(self) -> None:
        att = [{"name": "paper.pdf", "text": "x", "path": "/data/artifacts/a/paper.pdf"}]
        names = {t["name"] for t in self._effective_tools(att, False)}
        self.assertIn("extract_pdf", names)
        self.assertIn("extract_images", names)

    def test_no_attachment_keeps_tools_off(self) -> None:
        names = {t["name"] for t in self._effective_tools([], False)}
        self.assertNotIn("extract_pdf", names)

    def test_non_pdf_attachment_does_not_bring_pdf_tools(self) -> None:
        att = [{"name": "notes.md", "text": "x", "path": "/data/artifacts/a/notes.md"}]
        names = {t["name"] for t in self._effective_tools(att, False)}
        self.assertNotIn("extract_pdf", names)

    def test_attachment_path_reaches_the_prompt(self) -> None:
        A = _app()
        A.init_db()
        att = [{"name": "paper.pdf", "text": "the text", "path": "/data/artifacts/a/paper.pdf"}]
        msgs, _ = A.build_messages("nonexistent-chat", "extract it", {"tools_enabled": False}, att)
        block = next((m["content"] for m in msgs if "attached file" in (m.get("content") or "")), "")
        self.assertIn("/data/artifacts/a/paper.pdf", block)
        self.assertIn("extract_pdf", block)


@unittest.skipUnless(APP is not None, "the app's dependencies are not installed")
class ContextBudgetTest(unittest.TestCase):
    """A big PDF must not blow the context window.

    The uploaded text used to be pasted in whole, which killed the request
    with ContextOverflow before the model ever saw the file.
    """

    def test_big_pdf_is_trimmed_to_fit(self) -> None:
        A = _app()
        settings = {"context_length": 12288}
        att = [{"name": "paper.pdf", "text": "x" * 200_000, "path": "/data/a/paper.pdf"}]
        out = A.fit_attachments(att, settings, used_chars=6000)
        self.assertLess(len(out[0]["text"]), 40_000)
        self.assertIn("truncated", out[0]["text"])

    def test_trimmed_pdf_points_at_the_tool(self) -> None:
        A = _app()
        A.init_db()
        settings = {"context_length": 12288}
        att = [{"name": "paper.pdf", "text": "x" * 200_000, "path": "/data/a/paper.pdf"}]
        msgs, _ = A.build_messages("nonexistent-chat", "extract it", settings, att)
        block = next((m["content"] for m in msgs if "attached file" in (m.get("content") or "")), "")
        # the path is in the header, the tool hint in the text
        self.assertIn("/data/a/paper.pdf", block)
        self.assertIn("extract_pdf", block)
        self.assertIn("truncated", block)

    def test_small_attachment_is_left_alone(self) -> None:
        A = _app()
        settings = {"context_length": 12288}
        att = [{"name": "notes.md", "text": "hello", "path": "/data/a/notes.md"}]
        out = A.fit_attachments(att, settings, used_chars=6000)
        self.assertEqual(out[0]["text"], "hello")

    def test_no_attachments_is_fine(self) -> None:
        A = _app()
        self.assertEqual(A.fit_attachments(None, {"context_length": 12288}), [])
        self.assertEqual(A.fit_attachments([], {"context_length": 12288}), [])

    def test_prompt_with_a_big_pdf_stays_within_the_window(self) -> None:
        A = _app()
        A.init_db()
        att = [{"name": "paper.pdf", "text": "x" * 200_000, "path": "/data/a/paper.pdf"}]
        msgs, _ = A.build_messages("nonexistent-chat", "extract it", {"tools_enabled": False}, att)
        total = sum(len(m.get("content") or "") for m in msgs)
        # 12288 tokens at ~3 chars/token, with room for the answer
        self.assertLess(total, 12288 * 3)


if __name__ == "__main__":
    unittest.main()
