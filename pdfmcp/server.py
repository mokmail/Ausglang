"""MCP server: extract PDF text into structured markdown + a parallel manifest.

Transport
---------
``stdio`` (default) speaks plain JSON-RPC on stdin/stdout, which is what the
SGLang Workbench's own MCP client speaks -- register it with::

    POST /api/mcp/servers  {"name": "pdf", "command": "python", "args": ["pdfmcp/server.py"]}

``http`` speaks streamable-http on ``/mcp`` for clients that expect that
(Open-WebUI and friends)::

    python pdfmcp/server.py --http --port 8500

The tools are identical in both modes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from .engines import extract
from .writer import write_extraction

# The workbench passes the repo root in the environment; fall back to the
# package's parent so the server also works when started by hand.
REPO_ROOT = Path(os.environ.get("WORKBENCH_ROOT") or Path(__file__).resolve().parent.parent)
# WORKBENCH_DATA is the persistent volume in the container; writing there
# means the extractions survive a rebuild.
DATA_ROOT = Path(os.environ.get("WORKBENCH_DATA") or (REPO_ROOT / "data"))
DEFAULT_OUT = Path(os.environ.get("PDFMCP_OUT") or (DATA_ROOT / "pdf-extracts"))

mcp = FastMCP(
    "pdfmcp",
    instructions=(
        "Extracts text and images from PDF files into a structured markdown "
        "document plus a parallel manifest.json that records, for every text "
        "block, its page number, section topic and reading order, and for "
        "every image, its file and source page."
    ),
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _out_dir(out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir).expanduser().resolve()
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUT.resolve()


def _bar(value: float, width: int = 10) -> str:
    v = max(0.0, min(1.0, float(value)))
    filled = round(v * width)
    return "█" * filled + "░" * (width - filled)


def _pct(value: float) -> str:
    return f"{float(value) * 100:.0f}%"


def _err(exc: Exception, what: str) -> dict[str, Any]:
    """A clean, human-readable error -- never a raw traceback."""
    return {
        "ok": False,
        "error": f"{what}: {exc}",
        "error_type": type(exc).__name__,
        "visualization": {
            "markdown": f"### ⚠️ {what} failed\n\n"
                        f"`{type(exc).__name__}` — {exc}\n\n"
                        f"Check the path and that the file is a readable PDF."
        },
    }


def _payload(data: dict[str, Any], markdown: str) -> dict[str, Any]:
    """Return the JSON payload with a ready-to-render markdown block."""
    data = dict(data)
    data.setdefault("ok", True)
    data["visualization"] = {"markdown": markdown}
    # duplicated for front-ends that look for a different key
    data["_markdown"] = markdown
    data["content"] = markdown
    return data


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #


@mcp.tool
def extract_pdf(
    pdf_path: str,
    out_dir: str | None = None,
    backend: str = "auto",
    do_ocr: bool = False,
    extract_images: bool = True,
    name: str | None = None,
) -> dict[str, Any]:
    """Extract text and images from a PDF into structured markdown + manifest.

    Writes three things into ``<out_dir>/<name>/``:

    * ``document.md`` — the text as structured markdown (headings, paragraphs,
      lists, tables), with a ``<!-- page: N -->`` marker and an
      ``<!-- id: ... -->`` anchor before every block.
    * ``manifest.json`` — the parallel machine-readable twin. Every entry in
      ``texts`` carries ``page``, ``topic`` (the section it belongs to),
      ``topic_path``, ``level`` and its reading ``order``; every entry in
      ``images`` carries the file on disk and the page it came from.
    * ``images/`` — one file per extracted image.

    Args:
        pdf_path: Path to the PDF file.
        out_dir: Directory to write into (default: the workbench data dir).
        backend: ``auto`` (best available), ``docling`` (best structure, needs
            the docling package) or ``fallback`` (pypdf + pdfium, no ML).
        do_ocr: OCR scanned/image-only pages (docling backend only).
        extract_images: Also extract the embedded images.
        name: Name for the output folder (default: the PDF's file name).
    """
    try:
        doc = extract(pdf_path, backend=backend, do_ocr=do_ocr)
    except Exception as exc:  # noqa: BLE001
        return _err(exc, "PDF extraction")

    # A PDF with no text layer is a scan. Try again with OCR before giving up
    # -- the user asked for the text, and OCR is the only way to get it.
    if not doc.blocks and not do_ocr:
        try:
            retry = extract(pdf_path, backend="docling", do_ocr=True)
            if retry.blocks:
                doc = retry
                doc.warnings.append("no text layer found: extracted with OCR instead")
        except Exception as exc:  # noqa: BLE001
            doc.warnings.append(
                f"no text layer (this looks like a scan) and OCR is unavailable: "
                f"{type(exc).__name__}: {exc}")
            if not doc.warnings[0].startswith("no text layer"):
                doc.warnings.insert(0, "no text layer found (this PDF looks like a scan)")
    elif not doc.blocks:
        doc.warnings.append("no text layer found (this PDF looks like a scan)")

    if not extract_images:
        doc.images = []

    try:
        result = write_extraction(doc, _out_dir(out_dir), name=name)
    except Exception as exc:  # noqa: BLE001
        return _err(exc, "Writing the output files")

    manifest = result.manifest
    counts = manifest["counts"]
    pages_with_text = len({b["page"] for b in manifest["texts"] if b["page"]})
    density = (pages_with_text / result.document.page_count) if result.document.page_count else 0.0

    top_topics = manifest["topics"][:8]
    topic_rows = "\n".join(
        f"| {t['topic']} | `{_bar(t['blocks'] / max(1, top_topics[0]['blocks']))}` "
        f"{_pct(t['blocks'] / max(1, top_topics[0]['blocks']))} | {t['blocks']} |"
        for t in top_topics
    ) or "| — | — | — |"

    warn = ""
    if manifest.get("warnings"):
        warn = "\n\n### ⚠️ Extraction warnings\n" + "\n".join(
            f"- {w}" for w in manifest["warnings"]
        )

    md = f"""### 📄 Extracted — `{result.source.name}`

> # ✅ Extraction complete
> {counts['text_blocks']} text blocks and {counts['images']} images from {counts['pages']} pages.

| Output | Path |
|---|---|
| Markdown | `{result.markdown_path}` |
| Manifest | `{result.manifest_path}` |
| Images | `{result.images_dir}` |

| Metric | Value |
|---|---|
| Backend | `{result.document.backend}` |
| Pages | {counts['pages']} |
| Text blocks | {counts['text_blocks']} |
| Words | {counts['words']} |
| Images | {counts['images']} |
| Topics | {counts['topics']} |
| Pages with text | `{_bar(density)}` {_pct(density)} |

**Topics by size** (scaled to the largest):

| Topic | Share | Blocks |
|---|---|---|
{topic_rows}
{warn}
"""
    return _payload({
        "markdown_path": str(result.markdown_path),
        "manifest_path": str(result.manifest_path),
        "images_dir": str(result.images_dir),
        "source": manifest["source"],
        "backend": result.document.backend,
        "counts": counts,
        "topics": manifest["topics"],
        "warnings": manifest.get("warnings", []),
    }, md)


@mcp.tool
def extract_images(
    pdf_path: str,
    out_dir: str | None = None,
    min_width: int = 16,
    min_height: int = 16,
    name: str | None = None,
) -> dict[str, Any]:
    """Extract only the images of a PDF, with the page each one came from.

    Returns one record per image: file name, page number, pixel size, the
    position on the page and the section topic it falls under. Images smaller
    than ``min_width`` x ``min_height`` pixels are skipped (they are usually
    bullets, rules or logos).

    Args:
        pdf_path: Path to the PDF file.
        out_dir: Directory to write into (default: the workbench data dir).
        min_width: Skip images narrower than this.
        min_height: Skip images shorter than this.
        name: Name for the output folder (default: the PDF's file name).
    """
    try:
        doc = extract(pdf_path, backend="auto")
    except Exception as exc:  # noqa: BLE001
        return _err(exc, "PDF extraction")

    kept = [i for i in doc.images if i.width >= min_width and i.height >= min_height]
    dropped = len(doc.images) - len(kept)
    doc.images = kept

    try:
        result = write_extraction(doc, _out_dir(out_dir), name=name)
    except Exception as exc:  # noqa: BLE001
        return _err(exc, "Writing the image files")

    images = result.manifest["images"]
    by_page: dict[int, int] = {}
    for img in images:
        by_page[img["page"]] = by_page.get(img["page"], 0) + 1

    rows = "\n".join(
        f"| `{img['file']}` | {img['page']} | {img['width_px']}×{img['height_px']} "
        f"| {img.get('topic') or '—'} |"
        for img in images[:40]
    ) or "| — | — | — | — |"

    page_list = ", ".join(f"p.{p} ({n})" for p, n in sorted(by_page.items())) or "none"

    md = f"""### 🖼️ Extracted {len(images)} images — `{result.source.name}`

> # ✅ Image extraction complete
> Images written to `{result.images_dir}`.

| Output | Path |
|---|---|
| Images | `{result.images_dir}` |
| Manifest | `{result.manifest_path}` |

**By page:** {page_list}

| File | Page | Size (px) | Topic |
|---|---|---|---|
{rows}
"""
    payload = {
        "images_dir": str(result.images_dir),
        "manifest_path": str(result.manifest_path),
        "count": len(images),
        "skipped_small": dropped,
        "by_page": {str(p): n for p, n in sorted(by_page.items())},
        "images": images,
    }
    return _payload(payload, md)


@mcp.tool
def read_manifest(manifest_path: str) -> dict[str, Any]:
    """Read a manifest.json written by ``extract_pdf`` and summarise it.

    Args:
        manifest_path: Path to the manifest.json file.
    """
    p = Path(manifest_path).expanduser()
    if not p.exists():
        return _err(FileNotFoundError(f"no such manifest: {p}"), "Reading the manifest")
    try:
        manifest = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return _err(exc, "Parsing the manifest")

    counts = manifest.get("counts", {})
    topics = manifest.get("topics", [])[:10]
    rows = "\n".join(
        f"| {t.get('topic')} | {t.get('blocks')} |" for t in topics
    ) or "| — | — |"

    md = f"""### 📋 Manifest — `{manifest.get('source', {}).get('name', '?')}`

> # ✅ Manifest read
> {counts.get('text_blocks', 0)} text blocks, {counts.get('images', 0)} images.

| Field | Value |
|---|---|
| Backend | `{manifest.get('backend')}` |
| Pages | {counts.get('pages')} |
| Words | {counts.get('words')} |
| Topics | {counts.get('topics')} |

| Topic | Blocks |
|---|---|
{rows}
"""
    return _payload({
        "source": manifest.get("source"),
        "backend": manifest.get("backend"),
        "counts": counts,
        "topics": manifest.get("topics", []),
        "outline": manifest.get("outline", []),
        "warnings": manifest.get("warnings", []),
    }, md)


@mcp.tool
def list_extractions(out_dir: str | None = None) -> dict[str, Any]:
    """List the PDF extractions that were written so far.

    Args:
        out_dir: Directory to look in (default: the workbench data dir).
    """
    root = _out_dir(out_dir)
    found: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        counts = m.get("counts", {})
        found.append({
            "name": manifest_path.parent.name,
            "source": m.get("source", {}).get("name"),
            "backend": m.get("backend"),
            "pages": counts.get("pages"),
            "text_blocks": counts.get("text_blocks"),
            "images": counts.get("images"),
            "markdown": str(manifest_path.parent / "document.md"),
            "manifest": str(manifest_path),
        })

    rows = "\n".join(
        f"| `{f['name']}` | {f['source']} | {f['pages']} | {f['text_blocks']} | {f['images']} |"
        for f in found
    ) or "| — | — | — | — | — |"

    md = f"""### 📂 {len(found)} extraction(s) in `{root}`

> # ✅ {'Extractions found' if found else 'Nothing extracted yet'}

| Name | Source | Pages | Blocks | Images |
|---|---|---|---|---|
{rows}
"""
    return _payload({"out_dir": str(root), "count": len(found), "extractions": found}, md)


@mcp.tool
def pdf_info(pdf_path: str) -> dict[str, Any]:
    """Report what is inside a PDF without writing anything.

    Gives the page count, the document metadata, the bookmark outline and
    whether the file has an extractable text layer (a scanned PDF has none).

    Args:
        pdf_path: Path to the PDF file.
    """
    try:
        doc = extract(pdf_path, backend="auto")
    except Exception as exc:  # noqa: BLE001
        return _err(exc, "Reading the PDF")

    pages_with_text = sorted({b.page for b in doc.blocks if b.page})
    has_text = bool(doc.blocks)
    outline_rows = "\n".join(
        f"| {o.get('title')} | {o.get('page', '—')} |" for o in doc.outline[:15]
    ) or "| — | — |"

    md = f"""### 📖 `{doc.source.name}`

> # {'✅ Text layer found' if has_text else '⚠️ No text layer — this PDF is probably scanned'}

| Field | Value |
|---|---|
| Pages | {doc.page_count} |
| Backend | `{doc.backend}` |
| Text blocks | {len(doc.blocks)} |
| Words | {sum(b.word_count for b in doc.blocks)} |
| Images | {len(doc.images)} |
| Size | {doc.source.stat().st_size} bytes |

**Pages with text:** {', '.join(str(p) for p in pages_with_text[:30]) or 'none'}

| Outline entry | Page |
|---|---|
{outline_rows}
"""
    return _payload({
        "path": str(doc.source),
        "name": doc.source.name,
        "page_count": doc.page_count,
        "has_text_layer": has_text,
        "text_blocks": len(doc.blocks),
        "words": sum(b.word_count for b in doc.blocks),
        "images": len(doc.images),
        "metadata": doc.metadata,
        "outline": doc.outline,
        "pages_with_text": pages_with_text,
    }, md)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF extraction MCP server")
    ap.add_argument("--http", action="store_true", help="serve streamable-http instead of stdio")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--path", default="/mcp")
    args = ap.parse_args()

    if args.http:
        # In-container clients are rejected by DNS-rebinding protection unless
        # it is cleared for a 0.0.0.0 bind.
        try:
            mcp.settings.transport_security = None
        except Exception:
            pass
        mcp.run(transport="http", host=args.host, port=args.port, path=args.path)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
