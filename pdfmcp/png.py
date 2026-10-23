"""Write PNG bytes from a pdfium bitmap without requiring Pillow.

``PdfBitmap.to_pil()`` needs Pillow, which is not a dependency of the
fallback engine -- and a missing Pillow used to make image extraction fail
silently and report zero images. This encodes the bitmap buffer directly, so
the fallback path has no image-library dependency at all.
"""

from __future__ import annotations

import struct
import zlib
from typing import Any

# pdfium bitmap formats (pypdfium2.raw)
_FPDFBitmap_Gray = 1
_FPDFBitmap_BGR = 2
_FPDFBitmap_BGRx = 3
_FPDFBitmap_BGRA = 4
_FPDFBitmap_BGRA_Premul = 5


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def bitmap_to_png(bitmap: Any) -> bytes:
    """Encode a ``pypdfium2.PdfBitmap`` as PNG bytes.

    Handles the grayscale, BGR, BGRx, BGRA and premultiplied-BGRA formats
    that pdfium produces, in both byte orders.
    """
    width = int(bitmap.width)
    height = int(bitmap.height)
    stride = int(bitmap.stride)
    fmt = int(bitmap.format)
    n_channels = int(bitmap.n_channels)
    reverse = bool(bitmap.rev_byteorder)
    buf = bytes(bitmap.buffer)

    if fmt == _FPDFBitmap_Gray or n_channels == 1:
        color_type = 0  # grayscale
        channels = 1
    elif n_channels >= 4:
        color_type = 6  # RGBA
        channels = 4
    else:
        color_type = 2  # RGB
        channels = 3

    raw = bytearray()
    for y in range(height):
        row = buf[y * stride:y * stride + width * n_channels]
        raw.append(0)  # filter type 0 (None)
        if channels == 1:
            raw += row[:width]
        elif channels == 3:
            if reverse:
                for i in range(0, width * 3, 3):
                    raw += row[i:i + 3]
            else:
                # pdfium gives BGR; PNG wants RGB
                for i in range(0, width * 3, 3):
                    raw += bytes((row[i + 2], row[i + 1], row[i]))
        else:
            premul = fmt == _FPDFBitmap_BGRA_Premul
            for i in range(0, width * 4, 4):
                b, g, r = row[i], row[i + 1], row[i + 2]
                a = row[i + 3]
                if premul and a:
                    # undo the premultiplication so the colour is right
                    b = min(255, b * 255 // a)
                    g = min(255, g * 255 // a)
                    r = min(255, r * 255 // a)
                if reverse:
                    raw += bytes((r, g, b, a))
                else:
                    raw += bytes((r, g, b, a))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + _chunk(b"IEND", b""))
