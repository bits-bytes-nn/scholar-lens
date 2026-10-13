"""save_pdf: size cap and PDF header check, with no partial file left."""

from __future__ import annotations

from pathlib import Path

import pytest

from scholar_lens.src import pdf_download
from scholar_lens.src.pdf_download import (
    NotPdfContentError,
    PdfTooLargeError,
    save_pdf,
)


def test_header_after_leading_bytes_is_a_pdf(tmp_path: Path) -> None:
    path = save_pdf(
        [b"\xef\xbb\xbf\n", b"%PDF-1.7 body"], tmp_path / "a.pdf", source="x"
    )
    assert path.read_bytes().endswith(b"%PDF-1.7 body")


def test_html_is_not_a_pdf_and_leaves_no_file(tmp_path: Path) -> None:
    with pytest.raises(NotPdfContentError):
        save_pdf([b"<html>" + b" " * 2048], tmp_path / "a.pdf", source="x")
    assert not (tmp_path / "a.pdf").exists()


def test_size_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pdf_download, "MAX_PDF_BYTES", 8)
    with pytest.raises(PdfTooLargeError):
        save_pdf([b"%PDF-", b"123456"], tmp_path / "a.pdf", source="x")
    assert not (tmp_path / "a.pdf").exists()
    with pytest.raises(PdfTooLargeError):
        save_pdf([b"%PDF-"], tmp_path / "a.pdf", source="x", declared_length="99")
