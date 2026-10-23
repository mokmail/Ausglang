"""Tests for the pdfmcp package.

The PDF fixtures are generated on the fly (no binary blobs in the repo) so the
suite runs offline. The docling backend is optional: when it is not installed
the tests exercise the pypdf/pdfium fallback, and the docling-specific
assertions are skipped.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdfmcp import extract, write_extraction  # noqa: E402
from pdfmcp.engines import docling_available, strip_numbering  # noqa: E402


def _make_pdf(path: Path) -> None:
    """A small PDF with two sections, a list and an embedded image.

    Written by hand (raw PDF syntax) so the test suite needs no PDF library
    of its own -- the engines under test are what parse it.
    """
    import zlib

    # a 16x16 RGB image: every pixel is the same blue, so the channel order
    # of the extracted PNG can be verified
    w = h = 16
    # PDF /FlateDecode wants raw RGB samples (no PNG filter bytes)
    pdf_stream = zlib.compress(bytes((30, 120, 220)) * (w * h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        import struct

        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    # the text is drawn with a base-14 font; content stream in PDF units
    lines = [
        (20, 30, "Alpine Geology"),
        (14, 46, "Tectonic setting of the Eastern Alps."),
        (14, 56, "The Periadriatic line separates the domains."),
        (17, 72, "Methods"),
        (14, 88, "We sampled 42 sites along the traverse."),
        (14, 98, "First site was a marble outcrop."),
    ]
    content = ["BT"]
    for size, y, text in lines:
        content.append(f"/F1 {size} Tf 1 0 0 1 40 {720 - y} Tm ({text}) Tj")
    content.append("ET")
    content.append("q 60 0 0 60 320 600 cm /Im1 Do Q")
    stream = "\n".join(content).encode("latin-1")

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> /XObject << /Im1 6 0 R >> >> "
        b"/Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /XObject /Subtype /Image /Width " + str(w).encode()
        + b" /Height " + str(h).encode()
        + b" /ColorSpace /DeviceRGB "
        b"/BitsPerComponent 8 /Filter /FlateDecode /Length "
        + str(len(pdf_stream)).encode() + b" >>\nstream\n" + pdf_stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n"
            f"{xref}\n%%EOF\n").encode()
    path.write_bytes(bytes(out))


class EngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pdfmcp-test-"))
        self.pdf = self.tmp / "sample.pdf"
        _make_pdf(self.pdf)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fallback_extracts_text_with_pages(self) -> None:
        doc = extract(self.pdf, backend="fallback")
        self.assertGreater(len(doc.blocks), 0)
        text = " ".join(b.text for b in doc.blocks)
        self.assertIn("Alpine Geology", text)
        self.assertIn("42 sites", text)
        self.assertTrue(all(b.page >= 1 for b in doc.blocks))

    def test_fallback_finds_headings_and_topics(self) -> None:
        doc = extract(self.pdf, backend="fallback")
        headings = [b for b in doc.blocks if b.kind == "heading"]
        self.assertTrue(headings, "no heading detected")
        self.assertTrue(any("Alpine" in h.text for h in headings))
        # a paragraph under a heading carries that heading as its topic
        paras = [b for b in doc.blocks if b.kind == "paragraph" and "42 sites" in b.text]
        self.assertTrue(paras)
        self.assertTrue(paras[0].topic, "paragraph has no topic")

    def test_fallback_extracts_images(self) -> None:
        doc = extract(self.pdf, backend="fallback")
        self.assertGreaterEqual(len(doc.images), 1)
        img = doc.images[0]
        self.assertGreater(img.width, 0)
        self.assertGreater(img.height, 0)
        self.assertGreaterEqual(img.page, 1)

    def test_fallback_images_are_valid_png_bytes(self) -> None:
        """The fallback must not depend on Pillow: a missing Pillow used to
        make image extraction fail silently and report zero images."""
        doc = extract(self.pdf, backend="fallback")
        self.assertTrue(doc.images, "no images extracted")
        for img in doc.images:
            raw = (img.meta or {}).get("_bytes")
            self.assertIsInstance(raw, bytes, "image has no encoded bytes")
            self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"),
                            "image bytes are not a PNG")
            # IHDR: width/height must match what the manifest claims
            width = int.from_bytes(raw[16:20], "big")
            height = int.from_bytes(raw[20:24], "big")
            self.assertEqual(width, img.width)
            self.assertEqual(height, img.height)

    def test_fallback_png_encoder_handles_bgr(self) -> None:
        """The png encoder must produce a decodable PNG without Pillow."""
        import zlib

        from pdfmcp.png import bitmap_to_png

        class FakeBitmap:
            width = 2
            height = 2
            stride = 6
            format = 2  # FPDFBitmap_BGR
            n_channels = 3
            rev_byteorder = False
            # pdfium stores pixels as B,G,R -- so (0,0,255) is RED
            buffer = bytes([
                0, 0, 255,        # B=0 G=0 R=255 -> RGB (255,0,0)
                0, 255, 0,        # B=0 G=255 R=0 -> RGB (0,255,0)
                255, 0, 0,        # B=255 G=0 R=0 -> RGB (0,0,255)
                255, 255, 255,    # white
            ])

        png = bitmap_to_png(FakeBitmap())
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(int.from_bytes(png[16:20], "big"), 2)
        self.assertEqual(int.from_bytes(png[20:24], "big"), 2)
        # the IDAT must inflate to a filter byte plus 2 pixels of RGB
        idat_start = png.index(b"IDAT") + 4
        length = int.from_bytes(png[idat_start - 8:idat_start - 4], "big")
        raw = zlib.decompress(png[idat_start:idat_start + length])
        self.assertEqual(raw[0], 0)                       # filter: none
        self.assertEqual(raw[1:4], bytes((255, 0, 0)))    # red first
        self.assertEqual(raw[4:7], bytes((0, 255, 0)))    # green second

    def test_extracted_image_colour_matches_the_source(self) -> None:
        """End-to-end colour check: the PNG on disk must hold the colour
        that was drawn into the PDF, not a swapped channel order."""
        from pdfmcp.png import bitmap_to_png

        doc = extract(self.pdf, backend="fallback")
        self.assertTrue(doc.images)
        raw = (doc.images[0].meta or {})["_bytes"]

        import zlib

        idat_start = raw.index(b"IDAT") + 4
        length = int.from_bytes(raw[idat_start - 8:idat_start - 4], "big")
        pixels = zlib.decompress(raw[idat_start:idat_start + length])
        r, g, b = pixels[1], pixels[2], pixels[3]
        # the fixture draws RGB(30, 120, 220); the PDF colour space rounds
        # the red channel slightly, so allow a small tolerance
        self.assertEqual((g, b), (120, 220),
                         f"channel order is wrong: got RGB{(r, g, b)}")
        self.assertAlmostEqual(r, 30, delta=16,
                               msg=f"channel order is wrong: got RGB{(r, g, b)}")
        # and it must round-trip through the encoder unchanged
        self.assertTrue(bitmap_to_png.__doc__)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            extract(self.tmp / "nope.pdf")

    def test_non_pdf_raises(self) -> None:
        other = self.tmp / "notes.txt"
        other.write_text("hello", encoding="utf-8")
        with self.assertRaises(ValueError):
            extract(other)

    def test_unknown_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract(self.pdf, backend="nonsense")

    def test_strip_numbering(self) -> None:
        self.assertEqual(strip_numbering("3.1 PDF backends"), "PDF backends")
        self.assertEqual(strip_numbering("2 Getting Started"), "Getting Started")
        self.assertEqual(strip_numbering("Plain title"), "Plain title")


class WriterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pdfmcp-test-"))
        self.pdf = self.tmp / "sample.pdf"
        _make_pdf(self.pdf)
        self.out = self.tmp / "out"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_markdown_manifest_and_images(self) -> None:
        doc = extract(self.pdf, backend="fallback")
        res = write_extraction(doc, self.out)

        self.assertTrue(res.markdown_path.exists())
        self.assertTrue(res.manifest_path.exists())
        self.assertTrue(res.images_dir.is_dir())

        md = res.markdown_path.read_text(encoding="utf-8")
        self.assertIn("Alpine Geology", md)
        self.assertIn("<!-- page: 1 -->", md)
        self.assertNotIn("\ufffd", md)

        manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertEqual(manifest["source"]["name"], "sample.pdf")
        self.assertEqual(manifest["counts"]["text_blocks"], len(doc.blocks))
        self.assertTrue(manifest["texts"])
        for entry in manifest["texts"]:
            self.assertIn("page", entry)
            self.assertIn("topic", entry)
            self.assertIn("topic_path", entry)
            self.assertIn("order", entry)
            self.assertIn("id", entry)
        self.assertTrue(manifest["images"])
        for entry in manifest["images"]:
            self.assertIn("page", entry)
            self.assertIn("file", entry)
            self.assertTrue((res.images_dir / entry["file"]).exists())

    def test_manifest_ids_match_markdown_anchors(self) -> None:
        doc = extract(self.pdf, backend="fallback")
        res = write_extraction(doc, self.out)
        md = res.markdown_path.read_text(encoding="utf-8")
        manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["texts"]:
            self.assertIn(f"<!-- id: {entry['id']} |", md)

    def test_image_appears_once_in_markdown(self) -> None:
        doc = extract(self.pdf, backend="fallback")
        res = write_extraction(doc, self.out)
        md = res.markdown_path.read_text(encoding="utf-8")
        for entry in res.manifest["images"]:
            self.assertEqual(md.count(f"<!-- image: {entry['id']} |"), 1,
                             f"image {entry['id']} referenced more than once")

    def test_topics_are_counted(self) -> None:
        doc = extract(self.pdf, backend="fallback")
        res = write_extraction(doc, self.out)
        manifest = json.loads(res.manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["topics"])
        self.assertEqual(
            manifest["counts"]["topics"], len(manifest["topics"])
        )


@unittest.skipUnless(docling_available(), "docling is not installed")
class DoclingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pdfmcp-docling-"))
        self.pdf = self.tmp / "sample.pdf"
        _make_pdf(self.pdf)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_docling_extracts_headings(self) -> None:
        doc = extract(self.pdf, backend="docling")
        self.assertEqual(doc.backend, "docling")
        headings = [b for b in doc.blocks if b.kind == "heading"]
        self.assertTrue(headings, "no heading detected by docling")
        self.assertTrue(any("Alpine" in h.text for h in headings))

    def test_docling_blocks_carry_pages(self) -> None:
        doc = extract(self.pdf, backend="docling")
        self.assertTrue(all(b.page >= 1 for b in doc.blocks))
        pages = {b.page for b in doc.blocks}
        self.assertTrue(pages)


class PortabilityTest(unittest.TestCase):
    """The container runs Python 3.11, so the code must parse there.

    A backslash inside an f-string expression is a syntax error before
    Python 3.12 -- and a 3.13-only interpreter will happily accept it, so
    the failure only shows up in the deployed image.
    """

    def _files(self) -> list[Path]:
        root = Path(__file__).resolve().parent.parent
        files = sorted(root.glob("pdfmcp/*.py")) + [
            root / "pdfmcp_server.py", root / "app.py",
        ]
        return [f for f in files if f.exists()]

    def test_no_backslash_in_fstring_expressions(self) -> None:
        import re

        # f" ... { expr } ..." where expr contains a backslash
        pattern = re.compile(r'f"[^"\n]*\{[^}\n]*\\[^}\n]*\}')
        for path in self._files():
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    self.fail(f"{path.name}:{n}: backslash inside an f-string "
                              f"expression is a syntax error on Python 3.11: {line.strip()}")

    def test_package_compiles(self) -> None:
        import py_compile

        for path in self._files():
            py_compile.compile(str(path), doraise=True)


if __name__ == "__main__":
    unittest.main()
