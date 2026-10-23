"""PDF extraction engine for the MCP server.

Two backends, same output contract:

* ``docling``  -- IBM Docling (MIT). Best structure: heading hierarchy, tables,
  reading order, per-item page provenance. Heavier (torch) but optional.
* ``fallback`` -- pypdf + pypdfium2 (BSD/Apache). No ML, no model downloads.
  Text layout, outline/ToC and embedded-image geometry only.

Both produce a :class:`Document` whose ``blocks`` carry a page number and a
section topic, so the manifest is identical in shape regardless of backend.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #


@dataclass
class Block:
    """One text unit of the document.

    ``topic`` is the section the block belongs to (nearest enclosing heading),
    ``page`` is 1-based. Both are the fields the manifest is built around.
    """

    id: str
    page: int
    kind: str  # heading | paragraph | list_item | caption | code | table | text
    text: str
    topic: str = ""
    topic_path: list[str] = field(default_factory=list)
    level: int | None = None
    bbox: list[float] | None = None
    order: int = 0
    parent_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["word_count"] = self.word_count
        d["char_count"] = len(self.text)
        return d


@dataclass
class ImageAsset:
    """An extracted image, with the page it came from."""

    id: str
    page: int
    filename: str
    width: int
    height: int
    topic: str = ""
    topic_path: list[str] = field(default_factory=list)
    bbox: list[float] | None = None
    order: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["width_px"] = d.pop("width")
        d["height_px"] = d.pop("height")
        return d


@dataclass
class Document:
    source: Path
    backend: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    outline: list[dict[str, Any]] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    images: list[ImageAsset] = field(default_factory=list)
    page_count: int = 0
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        h = hashlib.sha256()
        with open(self.source, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

_NUM_PREFIX = re.compile(r"^\s*((?:\d+|[IVXLC]+)(?:\.\d+)*)[.)\s]+")


def strip_numbering(text: str) -> str:
    """Drop a leading ``3.1`` / ``IV.`` style numbering from a heading."""
    return _NUM_PREFIX.sub("", text).strip() or text.strip()


def _clean(text: str) -> str:
    """Normalise whitespace and repair the classic soft-hyphen break.

    Undecodable glyphs (``U+FFFD``) are dropped here rather than in the
    renderer: a PDF with a subsetted font whose encoding cannot be resolved
    yields replacement characters, and they must not reach the markdown or
    the manifest. The count is reported per block in ``meta.undecodable``.
    """
    text = text.replace("\xad", "")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = text.replace("\ufffd", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _slug(text: str, fallback: str = "doc") -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return s[:80] or fallback


def _pdf_date(value: Any) -> str:
    """pypdf gives ``D:20241210022326Z``; make it ISO-ish."""
    if not value:
        return ""
    m = re.match(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?", str(value))
    if not m:
        return str(value)
    y, mo, d, hh, mm, ss = (m.group(i) or "" for i in range(1, 7))
    return f"{y}-{mo or '01'}-{d or '01'}T{hh or '00'}:{mm or '00'}:{ss or '00'}"


# --------------------------------------------------------------------------- #
# backend: docling
# --------------------------------------------------------------------------- #


def docling_available() -> bool:
    try:
        import docling  # noqa: F401
    except Exception:
        return False
    return True


def _docling_outline(result: Any) -> list[dict[str, Any]]:
    """Read the PDF bookmark tree that Docling surfaces on the result."""
    out: list[dict[str, Any]] = []
    raw = getattr(result, "_pdf_outline", None)
    if not raw:
        return out

    def walk(items: Iterable[Any], depth: int) -> None:
        for item in items or []:
            if isinstance(item, (list, tuple)):
                walk(item, depth + 1)
                continue
            title = getattr(item, "title", None)
            if title is None and isinstance(item, dict):
                title = item.get("title")
            if not title:
                continue
            page = getattr(item, "page", None)
            if page is None and isinstance(item, dict):
                page = item.get("page")
            out.append({"title": str(title).strip(), "level": depth + 1, "page": page})

    walk(raw, 0)
    return out


def extract_with_docling(
    path: Path,
    *,
    do_ocr: bool = False,
    do_tables: bool = True,
    device: str = "CPU",
    extract_pictures: bool = True,
) -> Document:
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = do_ocr
    opts.do_table_structure = do_tables
    opts.generate_picture_images = extract_pictures
    opts.images_scale = 2.0
    opts.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CPU if str(device).upper() == "CPU" else AcceleratorDevice.AUTO
    )
    # Assign real heading levels (Docling flattens everything to level 1
    # without this step).
    try:
        opts.heading_hierarchy_options.enabled = True
    except Exception:
        pass

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    result = converter.convert(str(path))
    doc = result.document

    blocks: list[Block] = []
    images: list[ImageAsset] = []
    warnings: list[str] = []

    # --- text, in reading order ------------------------------------------- #
    # ``traverse_pictures=True`` matters: the layout model often classifies a
    # text-heavy region as a picture and nests the words underneath it. Without
    # this, those paragraphs never appear in the body traversal and the
    # extracted text silently loses most of the document.
    stack: list[tuple[int, str]] = []  # (level, heading text)
    counter = 0
    for item, _level in doc.iterate_items(traverse_pictures=True):
        label = str(getattr(item, "label", "") or "")
        if label == "picture":
            continue
        text = getattr(item, "text", None)
        if text is None and label == "table":
            try:
                text = item.export_to_markdown(doc)
            except Exception:
                text = None
        if not text or not str(text).strip():
            continue
        raw_text = str(text)
        undecodable = raw_text.count("\ufffd")
        text = _clean(raw_text)
        if not text:
            continue

        prov = getattr(item, "prov", None)
        page = int(prov[0].page_no) if prov else 0
        bbox = None
        if prov:
            bb = prov[0].bbox
            try:
                bbox = [round(float(bb.l), 2), round(float(bb.t), 2),
                        round(float(bb.r), 2), round(float(bb.b), 2)]
            except Exception:
                bbox = None

        kind = {
            "section_header": "heading",
            "title": "heading",
            "list_item": "list_item",
            "caption": "caption",
            "code": "code",
            "formula": "text",
            "footnote": "text",
            "page_header": "text",
            "page_footer": "text",
            "table": "table",
        }.get(label, "paragraph")

        level = getattr(item, "level", None)
        if kind == "heading":
            level = int(level) if isinstance(level, int) else 1
            # keep the stack consistent with the heading's depth
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, strip_numbering(text)))
            topic_path = [t for _, t in stack]
            topic = topic_path[-1] if topic_path else ""
        else:
            topic_path = [t for _, t in stack]
            topic = topic_path[-1] if topic_path else ""

        counter += 1
        blocks.append(
            Block(
                id=f"t{counter:04d}",
                page=page,
                kind=kind,
                text=text,
                topic=topic,
                topic_path=topic_path,
                level=level if kind == "heading" else None,
                bbox=bbox,
                order=counter,
                meta={"label": label, "undecodable": undecodable} if label
                else {"undecodable": undecodable},
            )
        )

    # --- pictures ---------------------------------------------------------- #
    # A picture belongs to the section that is open at its position in the
    # reading order, so pick the nearest preceding heading on the same page
    # and fall back to the last heading seen before that page.
    def topic_at(page: int, order_hint: float | None = None) -> tuple[str, list[str]]:
        same_page = [b for b in blocks if b.kind == "heading" and b.page == page]
        if order_hint is not None and same_page:
            # pick the heading closest above the picture
            above = [b for b in same_page if (b.bbox or [0, 0, 0, 0])[1] >= order_hint]
            if above:
                best = max(above, key=lambda b: (b.bbox or [0, 0, 0, 0])[1])
                return (best.topic_path[-1] if best.topic_path else best.text,
                        list(best.topic_path))
            if same_page:
                best = min(same_page, key=lambda b: (b.bbox or [0, 0, 0, 0])[1])
                return (best.topic_path[-1] if best.topic_path else best.text,
                        list(best.topic_path))
        earlier = [b for b in blocks if b.kind == "heading" and b.page < page]
        if earlier:
            best = earlier[-1]
            return (best.topic_path[-1] if best.topic_path else best.text, list(best.topic_path))
        return ("", [])

    if extract_pictures:
        pic_no = 0
        for pic in getattr(doc, "pictures", []) or []:
            prov = getattr(pic, "prov", None)
            if not prov:
                continue
            page = int(prov[0].page_no)
            ref = getattr(pic, "image", None)
            if ref is None or not getattr(ref, "uri", None):
                continue
            try:
                raw = base64_bytes(str(ref.uri))
            except Exception:
                continue
            if not raw:
                continue
            pic_no += 1
            size = getattr(ref, "size", None)
            width = int(getattr(size, "width", 0) or 0)
            height = int(getattr(size, "height", 0) or 0)
            bbox = None
            try:
                bb = prov[0].bbox
                bbox = [round(float(bb.l), 2), round(float(bb.t), 2),
                        round(float(bb.r), 2), round(float(bb.b), 2)]
            except Exception:
                bbox = None
            topic, topic_path = topic_at(page, bbox[1] if bbox else None)
            images.append(
                ImageAsset(
                    id=f"i{pic_no:04d}",
                    page=page,
                    filename="",  # filled in by the writer
                    width=width,
                    height=height,
                    topic=topic,
                    topic_path=list(topic_path),
                    bbox=bbox,
                    order=pic_no,
                    meta={
                        "dpi": int(getattr(ref, "dpi", 0) or 0),
                        "caption": _first_caption(pic),
                        "_bytes": raw,
                    },
                )
            )

    if not blocks:
        warnings.append("no text extracted -- the PDF may be a scanned image (try do_ocr=True)")

    meta = getattr(doc, "name", "") or ""
    return Document(
        source=path,
        backend="docling",
        title="",
        metadata={},
        outline=_docling_outline(result),
        blocks=blocks,
        images=images,
        page_count=len(getattr(doc, "pages", {}) or {}),
        warnings=warnings,
        meta={"docling_name": meta},
    )


def base64_bytes(data_uri: str) -> bytes:
    import base64

    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    return base64.b64decode(data_uri)


def _first_caption(pic: Any) -> str:
    for cap in getattr(pic, "captions", []) or []:
        text = getattr(cap, "text", "") or ""
        if text.strip():
            return _clean(text)
    return ""


# --------------------------------------------------------------------------- #
# backend: fallback (pypdf + pypdfium2)
# --------------------------------------------------------------------------- #

_HEADING_SIZE_FACTOR = 1.15


def extract_with_fallback(path: Path) -> Document:
    """Structure-light extraction: no ML, no model downloads."""
    import pypdf
    import pypdfium2 as pdfium

    reader = pypdf.PdfReader(str(path))
    blocks: list[Block] = []
    warnings: list[str] = []
    counter = 0

    # --- PDF outline (bookmarks) as the topic source ---------------------- #
    outline: list[dict[str, Any]] = []
    try:
        for item in reader.outline:
            if isinstance(item, list):
                continue
            try:
                page = int(reader.get_destination_page_number(item)) + 1
            except Exception:
                page = 0
            outline.append({"title": str(getattr(item, "title", "") or "").strip(),
                            "level": 1, "page": page})
    except Exception:
        pass

    # --- per page text, with font sizes so headings can be spotted -------- #
    for pno, page in enumerate(reader.pages, start=1):
        spans: list[dict[str, Any]] = []

        def visitor(text, _cm, tm, _font_dict, font_size, _spans=spans):
            if text and text.strip():
                _spans.append({
                    "t": text,
                    "size": round(float(font_size or 0), 1),
                    "x": float(tm[4]),
                    "y": float(tm[5]),
                })

        try:
            page.extract_text(visitor_text=visitor)
        except Exception as exc:
            warnings.append(f"page {pno}: text extraction failed ({exc})")
            continue

        if not spans:
            continue

        # group spans into lines by baseline
        rows: dict[float, list[dict[str, Any]]] = {}
        for s in spans:
            rows.setdefault(round(s["y"]), []).append(s)

        lines: list[dict[str, Any]] = []
        for y in sorted(rows, reverse=True):
            parts = sorted(rows[y], key=lambda s: s["x"])
            text = "".join(p["t"] for p in parts)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            lines.append({
                "y": y,
                "x0": min(p["x"] for p in parts),
                "size": max(p["size"] for p in parts),
                "text": text,
            })

        if not lines:
            continue

        # body font size = the most common size weighted by characters
        weighted: dict[float, int] = {}
        for s in spans:
            weighted[s["size"]] = weighted.get(s["size"], 0) + len(s["t"])
        body = max(weighted, key=lambda k: weighted[k]) if weighted else 10.0
        threshold = body * _HEADING_SIZE_FACTOR

        # outline entries that start on this page define the topic
        page_topics = [o["title"] for o in outline if o.get("page") == pno]
        stack: list[tuple[int, str]] = [
            (o.get("level", 1), o["title"]) for o in outline if o.get("page") == pno
        ]

        for line in lines:
            is_heading = line["size"] >= threshold and len(line["text"]) < 200
            if is_heading:
                # deeper outline level wins when the sizes are close
                lvl = 1
                if stack:
                    lvl = stack[-1][0]
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                stack.append((lvl, strip_numbering(line["text"])))
                kind = "heading"
                level = lvl
            else:
                kind = "list_item" if line["text"][:1] in "•-–" else "paragraph"
                level = None
            topic_path = [t for _, t in stack]
            topic = topic_path[-1] if topic_path else ""

            counter += 1
            blocks.append(Block(
                id=f"t{counter:04d}",
                page=pno,
                kind=kind,
                text=_clean(line["text"]),
                topic=topic,
                topic_path=topic_path,
                level=level,
                bbox=None,
                order=counter,
                meta={"font_size": line["size"], "x0": round(line["x0"], 2),
                      "y": round(line["y"], 2)},
            ))

    # --- images via pdfium ------------------------------------------------ #
    images: list[ImageAsset] = []
    try:
        from .png import bitmap_to_png

        pdf = pdfium.PdfDocument(str(path))
        pic_no = 0
        failed = 0
        tiny = 0
        for pno in range(len(pdf)):
            page = pdf[pno]
            for obj in page.get_objects():
                if obj.type != pdfium.raw.FPDF_PAGEOBJ_IMAGE:
                    continue
                try:
                    bitmap = obj.get_bitmap()
                    png = bitmap_to_png(bitmap)
                    width, height = int(bitmap.width), int(bitmap.height)
                except Exception:
                    # never swallow this: a silently missing image is worse
                    # than a reported failure
                    failed += 1
                    continue
                if width < 8 or height < 8:
                    # rules, bullets and separators -- count them so the
                    # number is visible rather than silently lost
                    tiny += 1
                    continue
                pic_no += 1
                try:
                    bounds = obj.get_bounds()
                    bbox = [round(float(v), 2) for v in bounds]
                except Exception:
                    bbox = None
                topic = ""
                topic_path: list[str] = []
                for b in reversed(blocks):
                    if b.page <= pno + 1:
                        topic = b.topic
                        topic_path = list(b.topic_path)
                        break
                images.append(ImageAsset(
                    id=f"i{pic_no:04d}",
                    page=pno + 1,
                    filename="",
                    width=width,
                    height=height,
                    topic=topic,
                    topic_path=topic_path,
                    bbox=bbox,
                    order=pic_no,
                    meta={"format": "png", "_bytes": png},
                ))
        if failed:
            warnings.append(f"{failed} embedded image(s) could not be decoded")
        if tiny:
            warnings.append(f"{tiny} embedded image(s) were smaller than 8x8 pixels "
                            f"and were skipped")
    except Exception as exc:
        warnings.append(f"image extraction failed ({exc})")

    if not blocks:
        warnings.append("no text extracted -- the PDF may be a scanned image")

    meta = {}
    try:
        raw = reader.metadata or {}
        meta = {str(k).lstrip("/"): _pdf_date(v) if "Date" in str(k) else str(v)
                for k, v in raw.items()}
    except Exception:
        pass

    return Document(
        source=path,
        backend="fallback",
        title=meta.get("Title", ""),
        metadata=meta,
        outline=outline,
        blocks=blocks,
        images=images,
        page_count=len(reader.pages),
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #


def extract(
    path: str | os.PathLike[str],
    *,
    backend: str = "auto",
    do_ocr: bool = False,
    do_tables: bool = True,
    device: str = "CPU",
) -> Document:
    """Extract a PDF into blocks + image assets.

    ``backend`` is ``auto`` (docling when importable, else fallback),
    ``docling`` (force) or ``fallback`` (force).
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")
    if p.suffix.lower() != ".pdf":
        raise ValueError(f"not a PDF file: {p.name}")

    if backend == "fallback":
        return extract_with_fallback(p)
    if backend == "docling":
        return extract_with_docling(p, do_ocr=do_ocr, do_tables=do_tables, device=device)
    if backend != "auto":
        raise ValueError(f"unknown backend {backend!r} (use auto, docling or fallback)")

    if docling_available():
        try:
            return extract_with_docling(p, do_ocr=do_ocr, do_tables=do_tables, device=device)
        except Exception as exc:
            try:
                doc = extract_with_fallback(p)
            except Exception as exc2:
                raise ValueError(
                    f"the PDF could not be read: docling said {exc.__class__.__name__}: {exc}; "
                    f"the fallback engine said {exc2.__class__.__name__}: {exc2}"
                ) from exc2
            doc.warnings.insert(0, f"docling failed ({exc.__class__.__name__}: {exc}); "
                                  f"fell back to the pypdf/pdfium engine")
            doc.backend = "fallback"
            return doc
    return extract_with_fallback(p)
