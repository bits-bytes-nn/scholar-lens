"""Tests for the pure / offline pieces of the PDF figure parser.

These tests deliberately avoid the Bedrock and Upstage code paths. Only the
pydantic models and the local ``fitz``-based figure extraction are exercised;
no AWS or network calls are made.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from scholar_lens.src.parser import (
    Content,
    Figure,
    Region,
    _is_valid_region,
    extract_figures_from_pdf,
)

# A simple rectangle covering most of the page (normalised corner coords).
SQUARE_COORDS: list[dict[str, float]] = [
    {"x": 0.1, "y": 0.1},
    {"x": 0.5, "y": 0.1},
    {"x": 0.5, "y": 0.5},
    {"x": 0.1, "y": 0.5},
]


@pytest.fixture
def one_page_pdf(tmp_path: Path) -> Path:
    """Build a real single-page PDF so ``get_pixmap`` always succeeds."""
    pdf_path = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Hello figure")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class TestContentModel:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("  padded  ", "padded"),
            ("\n\ttabbed\n", "tabbed"),
            ("clean", "clean"),
        ],
    )
    def test_validate_text_strips_whitespace(self, raw: str, expected: str) -> None:
        assert Content(text=raw).text == expected

    def test_non_string_coerced_to_empty(self) -> None:
        assert Content(text=None).text == ""  # type: ignore[arg-type]

    def test_str_truncates_to_50_chars(self) -> None:
        content = Content(text="x" * 200)
        rendered = str(content)
        assert rendered == f"Content(text='{'x' * 50}...')"


class TestFigureModel:
    def test_validate_text_fields_strip_caption_and_analysis(self) -> None:
        fig = Figure(
            figure_id="f1",
            path="/tmp/f.png",
            caption="  a caption  ",
            analysis="\n analysis \t",
        )
        assert fig.caption == "a caption"
        assert fig.analysis == "analysis"

    def test_none_text_fields_preserved(self) -> None:
        fig = Figure(figure_id="f1", path="/tmp/f.png")
        assert fig.caption is None
        assert fig.analysis is None

    def test_resize_if_needed_returns_small_bytes_unchanged(self) -> None:
        small = b"not really an image but well under 5MB"
        assert Figure._resize_if_needed(small) is small


class TestRegionModel:
    def test_valid_region_accepted(self) -> None:
        region = Region(page=1, coordinates=SQUARE_COORDS)
        assert region.page == 1
        assert len(region.coordinates) == 4

    def test_fewer_than_four_coordinates_rejected(self) -> None:
        with pytest.raises(ValueError):
            Region(page=1, coordinates=SQUARE_COORDS[:3])

    @pytest.mark.parametrize("page", [0, -1])
    def test_non_positive_page_rejected(self, page: int) -> None:
        with pytest.raises(ValueError):
            Region(page=page, coordinates=SQUARE_COORDS)


class TestIsValidRegion:
    def test_in_range_page_is_valid(self, one_page_pdf: Path) -> None:
        doc = fitz.open(str(one_page_pdf))
        with doc:
            region = Region(page=1, coordinates=SQUARE_COORDS)
            assert _is_valid_region(region, doc) is True

    def test_page_beyond_document_is_invalid(self, one_page_pdf: Path) -> None:
        doc = fitz.open(str(one_page_pdf))
        with doc:
            region = Region(page=2, coordinates=SQUARE_COORDS)
            assert _is_valid_region(region, doc) is False


class TestExtractFiguresFromPdf:
    def test_produces_existing_png(self, one_page_pdf: Path, tmp_path: Path) -> None:
        figures_dir = tmp_path / "figures"
        region = Region(page=1, coordinates=SQUARE_COORDS)

        paths = extract_figures_from_pdf(one_page_pdf, figures_dir, [region])

        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".png"

    def test_out_of_range_region_is_skipped(
        self, one_page_pdf: Path, tmp_path: Path
    ) -> None:
        figures_dir = tmp_path / "figures"
        region = Region(page=99, coordinates=SQUARE_COORDS)

        paths = extract_figures_from_pdf(one_page_pdf, figures_dir, [region])

        assert paths == []

    def test_missing_pdf_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_figures_from_pdf(
                tmp_path / "ghost.pdf",
                tmp_path / "figures",
                [Region(page=1, coordinates=SQUARE_COORDS)],
            )
