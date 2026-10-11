"""Image format helpers shared by figure analysis and publishing."""

from __future__ import annotations

import io

import resvg_py
from PIL import Image

# Render scale for vector figures: 3x the SVG's CSS size gives a sharp PNG on
# high-density screens without huge files.
SVG_ZOOM = 3
# Largest rendered side in pixels (a huge SVG at 3x would need gigabytes).
MAX_RENDER_SIDE = 4096


def is_svg(data: bytes) -> bool:
    """SVG by content, tolerating a UTF-8 BOM, an XML declaration, a doctype or
    a leading comment."""
    head = data[:4096].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    return head.startswith((b"<svg", b"<?xml", b"<!--", b"<!doctype")) and (
        b"<svg" in head
    )


def svg_to_png(data: bytes) -> bytes:
    """Rasterize an SVG figure on white (no system libraries needed).

    Raises ``ValueError`` when the SVG cannot be decoded or rendered.
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    png = bytes(
        resvg_py.svg_to_bytes(
            svg_string=text, dpi=96.0, zoom=SVG_ZOOM, background="white"
        )
    )
    try:
        with Image.open(io.BytesIO(png)) as image:
            if max(image.size) <= MAX_RENDER_SIDE:
                return png
            image.thumbnail((MAX_RENDER_SIDE, MAX_RENDER_SIDE))
            out = io.BytesIO()
            image.save(out, format="PNG")
            return out.getvalue()
    except Image.DecompressionBombError as e:
        raise ValueError(f"SVG renders too large: {e}") from e
