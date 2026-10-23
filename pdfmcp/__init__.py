"""PDF extraction MCP server.

Public API::

    from pdfmcp import extract, write_extraction

    doc = extract("paper.pdf")
    result = write_extraction(doc, "out/")
    result.markdown_path, result.manifest_path
"""

from .engines import Block, Document, ImageAsset, extract
from .writer import ExtractionResult, build_manifest, render_markdown, write_extraction

__all__ = [
    "Block",
    "Document",
    "ImageAsset",
    "ExtractionResult",
    "extract",
    "write_extraction",
    "render_markdown",
    "build_manifest",
    "extract_for_prompt",
]

__version__ = "1.0.0"


def extract_for_prompt(path, *, out_dir=None, do_ocr: bool = True):
    """Extract a PDF for a model prompt.

    Returns ``(text, meta)`` where ``text`` is the document as markdown
    (headings, lists, tables, page markers) and ``meta`` describes it.

    When ``out_dir`` is given, the full extraction -- ``document.md``, the
    parallel ``manifest.json`` and the images -- is written there, and the
    paths come back in ``meta`` so the caller can point a tool at them.

    A PDF with no text layer (a scan) is retried with OCR when the docling
    backend is available; if that is not possible the text comes back empty
    and ``meta["warning"]`` says why, rather than raising.
    """
    from pathlib import Path

    from .engines import extract as _extract
    from .writer import write_extraction as _write

    p = Path(path)
    doc = _extract(p, backend="auto", do_ocr=False)
    ocr_tried = False
    if not doc.blocks and do_ocr:
        # no text layer: try again with OCR (docling only)
        ocr_tried = True
        try:
            doc = _extract(p, backend="docling", do_ocr=True)
        except Exception as exc:
            doc.warnings.append(f"OCR unavailable: {type(exc).__name__}: {exc}")

    text = render_markdown(doc, annotate=False)
    meta: dict = {
        "kind": "pdf",
        "pages": doc.page_count,
        "backend": doc.backend,
        "topics": len({b.topic for b in doc.blocks if b.topic}),
        "ocr": ocr_tried,
    }
    if out_dir is not None:
        result = _write(doc, out_dir, name=p.stem)
        meta["manifest"] = str(result.manifest_path)
        meta["markdown"] = str(result.markdown_path)
        meta["images_dir"] = str(result.images_dir)
        meta["images"] = len(result.document.images)
    if doc.warnings:
        meta["warnings"] = doc.warnings
    # a scan has no *content* -- the document title alone is not text
    has_content = any(b.text.strip() for b in doc.blocks)
    if not has_content:
        ocr_note = ""
        if ocr_tried:
            ocr_note = (" OCR was tried and is not available in this deployment "
                         "(it needs the docling package).")
        meta["warning"] = (
            f"this PDF has no extractable text -- it looks like a scanned image."
            f"{ocr_note} The file itself is saved, so it can be OCR'd later."
        )
    return text, meta
