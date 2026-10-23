"""Turn an extracted :class:`Document` into a markdown file + a parallel manifest.

The manifest is the machine-readable twin of the markdown: for every text
block it records *where* the text came from (page, section topic, position in
the reading order), and for every image it records the file on disk and the
page it was taken from. The two files are written side by side and share the
same block/image ids, so a reader can walk the markdown and look up the
provenance of any passage in the manifest.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engines import Document, ImageAsset, _clean, _slug, strip_numbering


@dataclass
class ExtractionResult:
    source: Path
    markdown_path: Path
    manifest_path: Path
    images_dir: Path
    document: Document
    manifest: dict[str, Any]

    @property
    def text_count(self) -> int:
        return len(self.document.blocks)

    @property
    def image_count(self) -> int:
        return len(self.document.images)


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #


def _heading_prefix(block_level: int | None) -> str:
    lvl = max(1, min(int(block_level or 1), 6))
    return "#" * lvl


def _md_table(text: str) -> str:
    """Docling already returns markdown for tables; make sure it is fenced."""
    if "|" in text and "---" in text:
        return text
    return text


def render_markdown(doc: Document, *, image_files: dict[str, str] | None = None,
                    annotate: bool = True) -> str:
    """Render the document as structured markdown.

    Each block is preceded by an HTML comment carrying its id and page, so the
    markdown and the manifest can be joined again without guessing. Pass
    ``annotate=False`` for a copy meant for a model prompt: the per-block id
    comments are noise there, but the page markers stay.
    """
    image_files = image_files or {}
    out: list[str] = []

    title = (doc.title or "").strip() or doc.source.stem.replace("_", " ").replace("-", " ").strip()
    out.append(f"# {title}")
    out.append("")

    if doc.metadata:
        rows = ["| field | value |", "|---|---|"]
        for k, v in doc.metadata.items():
            if not v:
                continue
            # escape the pipe outside the f-string: a backslash inside an
            # f-string expression is a syntax error before Python 3.12
            cell = str(v).replace("|", "\\|")
            rows.append(f"| {k} | {cell} |")
        if len(rows) > 2:
            out.append("<!-- source metadata -->")
            out.extend(rows)
            out.append("")

    if doc.outline:
        out.append("## Contents")
        out.append("")
        for entry in doc.outline:
            page = entry.get("page")
            suffix = f" — p. {page}" if page else ""
            out.append(f"- {entry.get('title', '')}{suffix}")
        out.append("")

    # An image is placed after the last text block of its page, so it appears
    # once -- not after every paragraph on that page.
    by_page: dict[int, list[str]] = {}
    for img in doc.images:
        fname = image_files.get(img.id)
        if fname:
            by_page.setdefault(img.page, []).append(img.id)
    img_by_id = {img.id: img for img in doc.images}

    last_block_on_page: dict[int, str] = {}
    for b in doc.blocks:
        if b.page:
            last_block_on_page[b.page] = b.id

    last_page: int | None = None
    for block in doc.blocks:
        if block.page and block.page != last_page:
            last_page = block.page
            out.append(f"<!-- page: {block.page} -->")

        if annotate:
            out.append(f"<!-- id: {block.id} | page: {block.page} | topic: {block.topic} -->")

        if block.kind == "heading":
            prefix = _heading_prefix(block.level)
            out.append(f"{prefix} {strip_numbering(block.text)}")
            out.append("")
        elif block.kind == "list_item":
            out.append(f"- {block.text}")
            out.append("")
        elif block.kind == "table":
            out.append(_md_table(block.text))
            out.append("")
        elif block.kind == "code":
            out.append("```")
            out.append(block.text)
            out.append("```")
            out.append("")
        else:
            out.append(block.text)
            out.append("")

        # the images of this page, emitted once after its last block
        if last_block_on_page.get(block.page) == block.id:
            for img_id in by_page.get(block.page, []):
                img = img_by_id[img_id]
                fname = image_files.get(img.id)
                caption = (img.meta or {}).get("caption") or ""
                alt = caption or f"Image {img.id} (page {img.page})"
                out.append(f"![{alt}]({fname})")
                out.append(f"<!-- image: {img.id} | page: {img.page} -->")
                out.append("")

    # pages that have images but no text block at all
    for page, img_ids in by_page.items():
        if page in last_block_on_page:
            continue
        out.append(f"<!-- page: {page} -->")
        for img_id in img_ids:
            img = img_by_id[img_id]
            fname = image_files.get(img.id)
            caption = (img.meta or {}).get("caption") or ""
            alt = caption or f"Image {img.id} (page {img.page})"
            out.append(f"![{alt}]({fname})")
            out.append(f"<!-- image: {img.id} | page: {img.page} -->")
            out.append("")

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


def build_manifest(doc: Document, *, image_files: dict[str, str],
                    markdown_path: Path, images_dir: Path) -> dict[str, Any]:
    """Build the manifest that sits next to the markdown."""
    topics: dict[str, int] = {}
    for b in doc.blocks:
        if b.topic:
            topics[b.topic] = topics.get(b.topic, 0) + 1

    blocks = []
    for b in doc.blocks:
        d = b.to_dict()
        d["markdown_anchor"] = f"<!-- id: {b.id} -->"
        blocks.append(d)

    images = []
    for img in doc.images:
        d = img.to_dict()
        d.pop("meta", None)
        meta = dict(img.meta or {})
        meta.pop("_pil", None)
        meta.pop("_bytes", None)
        d["file"] = image_files.get(img.id, "")
        d["file_relative"] = str(Path(images_dir.name) / d["file"]) if d["file"] else ""
        written = images_dir / d["file"] if d["file"] else None
        d["bytes"] = written.stat().st_size if written and written.exists() else None
        d["meta"] = meta
        images.append(d)

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "source": {
            "path": str(doc.source),
            "name": doc.source.name,
            "stem": doc.source.stem,
            "size_bytes": doc.source.stat().st_size,
            "sha256": doc.sha256,
            "page_count": doc.page_count,
        },
        "backend": doc.backend,
        "title": doc.title,
        "metadata": doc.metadata,
        "outline": doc.outline,
        "files": {
            "markdown": markdown_path.name,
            "images_dir": images_dir.name,
        },
        "counts": {
            "text_blocks": len(doc.blocks),
            "images": len(doc.images),
            "topics": len(topics),
            "pages": doc.page_count,
            "words": sum(b.word_count for b in doc.blocks),
        },
        "topics": [
            {"topic": t, "blocks": n}
            for t, n in sorted(topics.items(), key=lambda kv: -kv[1])
        ],
        "texts": blocks,
        "images": images,
    }
    if doc.warnings:
        manifest["warnings"] = doc.warnings
    return manifest


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #


def write_extraction(
    doc: Document,
    out_dir: str | os.PathLike[str],
    *,
    name: str | None = None,
) -> ExtractionResult:
    """Write ``<out_dir>/<name>/document.md``, ``manifest.json`` and ``images/``."""
    root = Path(out_dir).expanduser().resolve()
    stem = _slug(name or doc.source.stem, "document")
    target = root / stem
    images_dir = target / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # --- write the images -------------------------------------------------- #
    image_files: dict[str, str] = {}
    for img in doc.images:
        meta = img.meta or {}
        fname = f"{img.id}_p{img.page:03d}.png"
        target_file = images_dir / fname
        raw = meta.get("_bytes")
        pil = meta.get("_pil")
        try:
            if raw:
                target_file.write_bytes(raw)
            elif pil is not None:
                pil.save(target_file)
            else:
                continue
        except Exception:
            continue
        image_files[img.id] = fname
        img.filename = fname

    # --- markdown + manifest ----------------------------------------------- #
    markdown = render_markdown(doc, image_files=image_files)
    markdown_path = target / "document.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    manifest = build_manifest(
        doc,
        image_files=image_files,
        markdown_path=markdown_path,
        images_dir=images_dir,
    )
    manifest_path = target / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return ExtractionResult(
        source=doc.source,
        markdown_path=markdown_path,
        manifest_path=manifest_path,
        images_dir=images_dir,
        document=doc,
        manifest=manifest,
    )
