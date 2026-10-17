"""SVG rasterization must not follow references out of the document."""

from __future__ import annotations

import base64
import io
import tarfile
import time
from pathlib import Path

import pytest
from PIL import Image

from scholar_lens.src.utils import images
from scholar_lens.src.utils.images import (
    MAX_RENDER_SIDE,
    is_ar5iv_placeholder,
    raster_to_png,
    sanitize_svg,
    source_figure,
    svg_canvas_size,
    svg_to_png,
)

SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">{}</svg>'


def _red_png(path: Path) -> Path:
    Image.new("RGB", (20, 20), (255, 0, 0)).save(path)
    return path


def _pixel(png: bytes) -> tuple[int, int, int]:
    with Image.open(io.BytesIO(png)) as image:
        return image.convert("RGB").getpixel((30, 30))  # type: ignore[return-value]


@pytest.mark.parametrize("attr", ["href", "xlink:href"])
def test_local_file_image_is_not_rendered(tmp_path: Path, attr: str) -> None:
    red = _red_png(tmp_path / "red.png")
    svg = SVG.format(f'<image {attr}="{red}" width="20" height="20"/>')
    if attr.startswith("xlink"):
        svg = svg.replace("<svg ", '<svg xmlns:xlink="http://www.w3.org/1999/xlink" ')
    assert _pixel(svg_to_png(svg.encode())) == (255, 255, 255)


def test_dev_zero_and_css_urls_are_removed() -> None:
    svg = SVG.format(
        "<style>@import url(/etc/x.css); rect { fill: url(/dev/zero) }</style>"
        '<image href="/dev/zero"/><use href="file:///etc/passwd"/>'
        '<rect width="5" height="5" style="fill: url(/dev/zero)"/>'
        '<rect width="5" height="5" fill="url(#g)"/>'
    )
    clean = sanitize_svg(svg)
    assert "/dev/zero" not in clean and "/etc/" not in clean and "@import" not in clean
    assert "url(#g)" in clean  # in-document references stay
    svg_to_png(svg.encode())  # renders promptly


def test_in_document_and_raster_data_references_stay() -> None:
    svg = SVG.format(
        '<image href="data:image/png;base64,AAAA"/><use href="#a"/>'
        '<image href="data:image/svg+xml;base64,AAAA"/>'
    )
    clean = sanitize_svg(svg)
    assert "data:image/png" in clean and 'href="#a"' in clean
    assert "svg+xml" not in clean  # a nested SVG could carry its own references


def test_entity_declarations_are_rejected() -> None:
    svg = '<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY a "aaaa">]>' + SVG.format(
        "<text>&a;</text>"
    )
    with pytest.raises(ValueError, match="DTD"):
        svg_to_png(svg.encode())


def test_public_doctype_is_tolerated() -> None:
    svg = (
        '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n'
        '  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
        + SVG.format('<rect width="20" height="20" fill="red"/>')
    )
    assert _pixel(svg_to_png(svg.encode())) == (255, 0, 0)


def test_malformed_svg_raises_value_error() -> None:
    with pytest.raises(ValueError):
        svg_to_png(b"<svg xmlns='http://www.w3.org/2000/svg'><rect></svg>")


def _svg(attrs: str) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" {attrs}><rect width="10" height="10"/></svg>'


@pytest.mark.parametrize(
    "attrs,size",
    [
        ('width="72pt" height="36pt" viewBox="0 0 72 36"', (96.0, 48.0)),
        ('width="1in" height="2.54cm"', (96.0, 96.0)),
        ('viewBox="0 0 300 150"', (300.0, 150.0)),
        ('width="50%" viewBox="0 0 400 200"', (200.0, 100.0)),
        ("", (100.0, 100.0)),
        ('width="auto" height="auto" viewBox="0 0 200 100"', (200.0, 100.0)),
    ],
)
def test_canvas_size_follows_units_and_viewbox(
    attrs: str, size: tuple[float, float]
) -> None:
    assert svg_canvas_size(_svg(attrs)) == pytest.approx(size)


def test_huge_declared_canvas_is_bounded_before_rendering() -> None:
    # A ~100-byte SVG declaring a 100000px side used to allocate gigabytes.
    png = svg_to_png(
        _svg('width="100000" height="100000" viewBox="0 0 10 10"').encode()
    )
    with Image.open(io.BytesIO(png)) as image:
        assert max(image.size) <= MAX_RENDER_SIDE


@pytest.mark.parametrize(
    "attrs",
    [
        'width="1e309" height="10"',
        'width="-5" height="10"',
        'width="1e7"',
        'width="10furlong"',
    ],
)
def test_unrenderable_sizes_raise_value_error(attrs: str) -> None:
    with pytest.raises(ValueError):
        svg_to_png(_svg(attrs).encode())


@pytest.mark.parametrize(
    "attrs",
    ["", 'width="auto" height="auto"', 'width="50%" height="50%"', 'width="200"'],
)
def test_unsized_svg_with_huge_content_stays_within_its_box(attrs: str) -> None:
    # Without a viewBox and two absolute sides resvg sizes the canvas from the
    # content (a 30000px rect used to allocate gigabytes); it is now clipped
    # to the computed box like a browser would.
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" {attrs}>'
        '<rect width="30000" height="30000" fill="red"/></svg>'
    )
    with Image.open(io.BytesIO(svg_to_png(svg.encode()))) as image:
        assert max(image.size) <= 3 * 200


def _data_png(width: int, height: int) -> str:
    buffer = io.BytesIO()
    Image.new("1", (width, height)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_oversized_embedded_raster_is_rejected_before_rendering() -> None:
    svg = SVG.format(f'<image width="20" height="20" href="{_data_png(8000, 8000)}"/>')
    with pytest.raises(ValueError, match="embeds a 8000x8000 image"):
        svg_to_png(svg.encode())


def test_normal_embedded_raster_still_renders() -> None:
    # matplotlib heatmaps embed modest PNGs like this one.
    svg = SVG.format(f'<image width="20" height="20" href="{_data_png(640, 480)}"/>')
    assert svg_to_png(svg.encode()).startswith(b"\x89PNG")


@pytest.mark.parametrize(
    "svg",
    [
        # Both ran for minutes (or allocated gigabytes) on a tiny canvas; the
        # child process is stopped at the timeout (or by its memory cap).
        '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">'
        '<defs><filter id="f"><feMorphology radius="50" operator="dilate"/>'
        '</filter></defs><rect width="300" height="300" filter="url(#f)"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><defs>'
        '<pattern id="p" width="1000000" height="1000000" '
        'patternUnits="userSpaceOnUse"><rect width="1000000" height="1000000" '
        'fill="red"/></pattern></defs><rect width="100" height="100" '
        'fill="url(#p)"/></svg>',
    ],
)
def test_runaway_render_is_stopped(svg: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images, "RENDER_TIMEOUT_SECONDS", 0.5)
    started = time.monotonic()
    with pytest.raises(ValueError, match="exceeded|failed"):
        svg_to_png(svg.encode())
    assert time.monotonic() - started < 10


def test_child_failure_becomes_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images, "_RENDER_CHILD", "import sys; sys.exit(3)")
    with pytest.raises(ValueError, match="exit 3"):
        svg_to_png(SVG.format("").encode())


def test_renders_run_at_most_max_concurrent(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import subprocess
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from scholar_lens.src.utils import images

    active, peak, lock = 0, 0, threading.Lock()
    # Renders meet in pairs, so the cap is reached on any runner; a third
    # concurrent render would only show up when the semaphore lets it in.
    together = threading.Barrier(images.MAX_CONCURRENT_RENDERS, timeout=5)

    def fake_child(text: str, zoom: float) -> subprocess.CompletedProcess[bytes]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        together.wait()
        time.sleep(0.01)
        with lock:
            active -= 1
        return subprocess.CompletedProcess([], 0, stdout=b"png", stderr=b"")

    monkeypatch.setattr(images, "_run_child", fake_child)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: images._render_isolated("<svg/>", 1.0), range(8)))
    assert peak == images.MAX_CONCURRENT_RENDERS


def test_raster_to_png_converts_and_bounds_size() -> None:
    small = io.BytesIO()
    Image.new("RGB", (5000, 10), "blue").save(small, format="WEBP")
    with Image.open(io.BytesIO(raster_to_png(small.getvalue()))) as image:
        assert image.format == "PNG" and max(image.size) == MAX_RENDER_SIDE
    huge = io.BytesIO()
    Image.new("1", (8000, 7000)).save(huge, format="BMP")
    with pytest.raises(ValueError, match="too large"):
        raster_to_png(huge.getvalue())
    with pytest.raises(ValueError, match="not a decodable image"):
        raster_to_png(b"<html>404</html>")


@pytest.mark.parametrize("fmt", ["TIFF", "BMP", "AVIF"])
def test_raster_to_png_reads_the_converted_formats(fmt: str) -> None:
    Image.init()
    if fmt not in Image.OPEN:
        pytest.skip(f"Pillow has no {fmt} plugin here")
    data = io.BytesIO()
    Image.new("RGB", (4, 3), "red").save(data, format=fmt)
    with Image.open(io.BytesIO(raster_to_png(data.getvalue()))) as image:
        assert (image.format, image.size) == ("PNG", (4, 3))


def test_raster_to_png_refuses_other_formats_with_a_clean_reason() -> None:
    # Pillow reads ICO, but it is not a figure format the publisher converts.
    ico = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(ico, format="ICO")
    with pytest.raises(ValueError, match="expected one of PNG, JPEG") as info:
        raster_to_png(ico.getvalue())
    assert "BytesIO" not in str(info.value)


def test_raster_to_png_converts_cmyk() -> None:
    cmyk = io.BytesIO()
    Image.new("CMYK", (4, 4), (0, 255, 255, 0)).save(cmyk, format="JPEG")
    with Image.open(io.BytesIO(raster_to_png(cmyk.getvalue()))) as image:
        assert image.format == "PNG" and image.mode == "RGBA"


PLACEHOLDER = Path(__file__).parent / "fixtures" / "ar5iv_placeholder.png"


def test_ar5iv_placeholder_is_recognized_by_its_bytes() -> None:
    data = PLACEHOLDER.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (325, 400)
    assert is_ar5iv_placeholder(data)
    assert not is_ar5iv_placeholder(data + b"\0")


def _png(color: str = "red") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (6, 4), color).save(out, format="PNG")
    return out.getvalue()


def _pdf() -> bytes:
    import pymupdf

    with pymupdf.open() as doc:
        page = doc.new_page(width=100, height=50)
        page.draw_rect(pymupdf.Rect(10, 10, 90, 40), color=(0, 0, 1), fill=(0, 0, 1))
        return doc.tobytes()


def _archive(files: dict[str, bytes], compress: bool = True) -> bytes:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz" if compress else "w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


class TestSourceFigure:
    """A figure ar5iv lost is taken from the paper's arXiv source archive."""

    def test_same_path_wins_over_a_copy_elsewhere(self) -> None:
        archive = _archive(
            {
                "main.tex": b"\\documentclass{article}",
                "Media/overview.png": _png("red"),
                "arxiv_v1/Media/overview.png": _png("blue"),
            }
        )
        assert source_figure(archive, "Media/overview.png") == _png("red")

    def test_unique_stem_in_another_directory(self) -> None:
        archive = _archive({"./figs/overview.jpg": b"\xff\xd8\xffjpeg"}, False)
        assert source_figure(archive, "overview.png") == b"\xff\xd8\xffjpeg"

    def test_pdf_figure_is_rendered(self) -> None:
        png = source_figure(_archive({"figs/arch.pdf": _pdf()}), "figs/arch.png")
        with Image.open(io.BytesIO(png)) as image:
            assert image.format == "PNG" and image.size == (200, 100)
            assert image.convert("RGB").getpixel((100, 50)) == (0, 0, 255)

    def test_png_is_preferred_over_pdf(self) -> None:
        archive = _archive({"f/a.pdf": _pdf(), "f/a.png": _png()})
        assert source_figure(archive, "f/a.png") == _png()

    @pytest.mark.parametrize(
        "files,path,reason",
        [
            ({"main.tex": b"x"}, "Media/x1.png", "no figure file"),
            (
                {"a/fig.png": _png("red"), "b/fig.png": _png("blue")},
                "fig.png",
                "several figure files",
            ),
            ({"figs/plot.eps": b"%!PS"}, "figs/plot.png", "EPS"),
            ({"figs/plot.pdf": b"not a pdf"}, "figs/plot.png", "cannot be rendered"),
        ],
    )
    def test_unusable_source_raises(
        self, files: dict[str, bytes], path: str, reason: str
    ) -> None:
        with pytest.raises(ValueError, match=reason):
            source_figure(_archive(files), path)

    def test_single_gzipped_tex_is_not_an_archive(self) -> None:
        import gzip

        with pytest.raises(ValueError, match="not a source archive"):
            source_figure(gzip.compress(b"\\documentclass{article}"), "x.png")
