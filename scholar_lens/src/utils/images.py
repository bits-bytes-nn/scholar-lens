"""Image format helpers shared by figure analysis and publishing."""

from __future__ import annotations

import base64
import binascii
import io
import math
import re
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET

from PIL import Image

# Render scale for vector figures: 3x the SVG's CSS size gives a sharp PNG on
# high-density screens without huge files.
SVG_ZOOM = 3
# Largest rendered side in pixels (a huge SVG at 3x would need gigabytes).
MAX_RENDER_SIDE = 4096
# resvg runs in a child process: some features (huge pattern tiles, filter
# regions, feMorphology radii) allocate gigabytes or burn CPU for hours inside
# a tiny canvas, and a native call cannot be interrupted in-process.
RENDER_TIMEOUT_SECONDS = 60.0
# Address-space cap for the child (Linux; macOS ignores RLIMIT_AS, leaving the
# timeout as the only bound there).
RENDER_MEMORY_BYTES = 2 * 1024**3
# Renders allowed at once: each child may use RENDER_MEMORY_BYTES, so the
# total stays within a worker's memory and CPU however many figures a paper
# has (figure analysis fans out over a thread pool).
MAX_CONCURRENT_RENDERS = 2
_RENDER_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_RENDERS)
# Largest raster a figure may embed as a data: URL (resvg decodes it whole).
MAX_EMBEDDED_PIXELS = 50_000_000

# Read the SVG from stdin, write the PNG to stdout. Isolated mode (-I) keeps
# the child from importing anything from the working directory.
_RENDER_CHILD = """
import resource, sys
memory, zoom = int(sys.argv[1]), float(sys.argv[2])
try:
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
except (ValueError, OSError):
    pass  # not enforceable on this platform (macOS)
import resvg_py
svg = sys.stdin.buffer.read().decode("utf-8")
png = resvg_py.svg_to_bytes(svg_string=svg, dpi=96.0, zoom=zoom, background="white")
sys.stdout.buffer.write(bytes(png))
"""


def is_svg(data: bytes) -> bool:
    """SVG by content, tolerating a UTF-8 BOM, an XML declaration, a doctype or
    a leading comment."""
    head = data[:4096].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    return head.startswith((b"<svg", b"<?xml", b"<!--", b"<!doctype")) and (
        b"<svg" in head
    )


_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"
# Only in-document (#id) and inline-image references may stay: resvg follows
# any other href or CSS url() to the local filesystem (an untrusted figure
# could embed /etc/... or stream /dev/zero).
# (A nested data:image/svg+xml could carry its own file references.)
_SAFE_REF = re.compile(r"\s*(#|data:image/(png|jpe?g|gif|webp)[;,])", re.IGNORECASE)
_CSS_URL = re.compile(r"url\(\s*(?!['\"]?\s*#)[^)]*\)", re.IGNORECASE)
_PLAIN_DOCTYPE = re.compile(r"<!DOCTYPE[^\[>]*>", re.IGNORECASE)
_CSS_IMPORT = re.compile(r"@import[^;]*;?", re.IGNORECASE)


def _clean_css(css: str) -> str:
    return _CSS_URL.sub("none", _CSS_IMPORT.sub("", css))


def sanitize_svg(text: str) -> str:
    """The SVG with every external reference removed.

    Drops ``href``/``xlink:href`` values other than ``#id`` and
    raster ``data:image/…``, and CSS ``url(…)``/``@import`` pointing outside the
    document. Raises ``ValueError`` for a DTD internal subset (entities) or
    malformed XML.
    """
    # A plain public-DTD doctype (matplotlib writes one) is harmless and
    # dropped; an internal subset can declare entities (billion laughs).
    text = _PLAIN_DOCTYPE.sub("", text, count=1)
    if re.search(r"<!(DOCTYPE|ENTITY)", text, re.IGNORECASE):
        raise ValueError("SVG with a DTD internal subset is not rendered")
    ET.register_namespace("", _SVG_NS)
    ET.register_namespace("xlink", _XLINK_NS)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise ValueError(f"malformed SVG: {e}") from e
    for element in root.iter():
        for name in list(element.attrib):
            value = element.attrib[name]
            if name.rsplit("}", 1)[-1] == "href":
                if not _SAFE_REF.match(value):
                    del element.attrib[name]
                elif not value.lstrip().startswith("#"):
                    _check_embedded_image(value)
            elif "url(" in value.lower() or "@import" in value.lower():
                element.attrib[name] = _clean_css(value)
        if element.text and element.tag.rsplit("}", 1)[-1] == "style":
            element.text = _clean_css(element.text)
    return ET.tostring(root, encoding="unicode")


def _check_embedded_image(data_url: str) -> None:
    """Reject a ``data:image/…`` raster larger than :data:`MAX_EMBEDDED_PIXELS`
    (its size is read from the header, without decoding the pixels)."""
    header, _, payload = data_url.partition(",")
    if ";base64" not in header.lower():
        return  # percent-encoded rasters are rare and small
    try:
        raw = base64.b64decode("".join(payload.split()), validate=False)
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
    except (binascii.Error, ValueError, OSError):
        return  # not a readable image: resvg skips it
    except Image.DecompressionBombError as e:
        raise ValueError(f"SVG embeds an oversized image: {e}") from e
    if width * height > MAX_EMBEDDED_PIXELS:
        raise ValueError(f"SVG embeds a {width}x{height} image")


# CSS pixels per unit at resvg's 96 dpi (em/ex use its default 12px font).
_UNIT_PX = {
    "": 1.0,
    "px": 1.0,
    "pt": 96 / 72,
    "pc": 16.0,
    "in": 96.0,
    "cm": 96 / 2.54,
    "mm": 96 / 25.4,
    "em": 12.0,
    "ex": 6.0,
}
_LENGTH = re.compile(
    r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)\s*([a-z%]*)\s*$", re.I
)
# resvg's size when an SVG gives neither width/height nor a viewBox.
_DEFAULT_SIDE = 100.0
# Larger declared sides are not figures (and overflow float zoom maths).
_MAX_DECLARED_SIDE = 1e6


def _length_px(value: str | None, reference: float) -> float | None:
    """A root width/height in CSS pixels (``%`` of ``reference``); ``None``
    when absent or ``auto`` (SVG 2). Raises ``ValueError`` for a value that is
    not a length."""
    if value is None or value.strip().lower() in ("", "auto"):
        return None
    match = _LENGTH.match(value)
    unit = match.group(2).lower() if match else ""
    if not match or (unit != "%" and unit not in _UNIT_PX):
        raise ValueError(f"unsupported SVG length {value!r}")
    number = float(match.group(1))
    return number * reference / 100 if unit == "%" else number * _UNIT_PX[unit]


def svg_canvas_size(text: str) -> tuple[float, float]:
    """The CSS-pixel size resvg would render ``text`` at (zoom 1).

    Raises ``ValueError`` for a non-finite, non-positive or absurd size.
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise ValueError(f"malformed SVG: {e}") from e
    box = (root.get("viewBox") or "").replace(",", " ").split()
    try:
        box_w, box_h = (float(box[2]), float(box[3])) if len(box) == 4 else (0, 0)
    except ValueError:
        box_w = box_h = 0.0
    width = _length_px(root.get("width"), box_w or _DEFAULT_SIDE)
    height = _length_px(root.get("height"), box_h or _DEFAULT_SIDE)
    if width is None and height is None:
        width, height = (
            (box_w, box_h)
            if box_w > 0 and box_h > 0
            else (
                _DEFAULT_SIDE,
                _DEFAULT_SIDE,
            )
        )
    elif width is None or height is None:
        # One side given: the other follows the viewBox aspect ratio.
        ratio = box_h / box_w if box_w > 0 and box_h > 0 else 1.0
        width = width if width is not None else height / ratio  # type: ignore[operator]
        height = height if height is not None else width * ratio
    for side in (width, height):
        if not math.isfinite(side) or not 0 < side <= _MAX_DECLARED_SIDE:
            raise ValueError(f"SVG size {width}x{height} is not renderable")
    return width, height


def pin_canvas(text: str) -> tuple[str, float, float]:
    """``text`` with its root sized explicitly (absolute ``width``/``height``,
    and a ``viewBox`` when it has none), plus that size in CSS pixels.

    Without both, resvg sizes the canvas from the drawn content, so a tiny
    SVG with a huge shape would escape the zoom bound; a browser clips such an
    SVG to its box the same way.
    """
    width, height = svg_canvas_size(text)
    root = ET.fromstring(text)
    box = (root.get("viewBox") or "").replace(",", " ").split()
    try:
        has_box = len(box) == 4 and float(box[2]) > 0 and float(box[3]) > 0
    except ValueError:
        has_box = False
    if has_box and all(_is_absolute(root.get(side)) for side in ("width", "height")):
        return text, width, height  # already pinned: render it untouched
    root.set("width", format(width, ".10g"))
    root.set("height", format(height, ".10g"))
    if not has_box:
        root.set("viewBox", f"0 0 {root.get('width')} {root.get('height')}")
    return ET.tostring(root, encoding="unicode"), width, height


def _is_absolute(value: str | None) -> bool:
    match = _LENGTH.match(value or "")
    return bool(match) and match.group(2).lower() in _UNIT_PX  # type: ignore[union-attr]


def svg_to_png(data: bytes) -> bytes:
    """Rasterize an SVG figure on white (no system libraries needed); external
    references are removed first (:func:`sanitize_svg`).

    Rendering runs in a child process bounded by
    :data:`RENDER_TIMEOUT_SECONDS` and (on Linux) :data:`RENDER_MEMORY_BYTES`.
    Raises ``ValueError`` when the SVG cannot be decoded or rendered within
    those bounds.
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    text, width, height = pin_canvas(sanitize_svg(text))
    # Bound the canvas before resvg allocates it (a 100-byte SVG can declare a
    # 100000px side); the thumbnail below only trims rounding.
    zoom = min(SVG_ZOOM, MAX_RENDER_SIDE / max(width, height))
    png = _render_isolated(text, zoom)
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


def _render_isolated(text: str, zoom: float) -> bytes:
    """resvg in a bounded child process; any failure is ``ValueError``."""
    try:
        with _RENDER_SLOTS:
            done = _run_child(text, zoom)
    except subprocess.TimeoutExpired as e:
        raise ValueError(f"SVG rendering exceeded {RENDER_TIMEOUT_SECONDS:g} s") from e
    if done.returncode != 0 or not done.stdout:
        lines = done.stderr.decode("utf-8", "replace").strip().splitlines()
        reason = next(
            (line for line in reversed(lines) if not line.startswith("note:")),
            "no output",
        )
        raise ValueError(
            f"SVG rendering failed (exit {done.returncode}: over the memory "
            f"cap, killed, or unrenderable): {reason[:200]}"
        )
    return done.stdout


def _run_child(text: str, zoom: float) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _RENDER_CHILD,
            str(RENDER_MEMORY_BYTES),
            repr(zoom),
        ],
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=RENDER_TIMEOUT_SECONDS,
        check=False,
    )
