"""save_pdf: size cap and PDF header check, with no partial file left."""

from __future__ import annotations

from pathlib import Path

import pytest

from scholar_lens.adapters.scholarly import pdf as pdf_download
from scholar_lens.adapters.scholarly.pdf import (
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


def test_a_download_past_its_deadline_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter([0.0, 1.0, 400.0])
    monkeypatch.setattr(pdf_download, "_clock", lambda: next(ticks))
    path = tmp_path / "p.pdf"
    with pytest.raises(pdf_download.PdfDownloadTooSlowError):
        pdf_download.save_pdf([b"%PDF-1.7 a", b"b"], path, source="https://x.org/p")
    assert not path.exists()


def test_a_trickling_server_with_a_content_length_hits_the_deadline(
    tmp_path: Path,
) -> None:
    import socket
    import threading
    import time

    import requests

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve() -> None:
        conn, _ = server.accept()
        conn.recv(4096)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n%PDF-1.7 ")
        try:
            for _ in range(60):  # one byte every 0.05 s: never a full chunk
                time.sleep(0.05)
                conn.sendall(b"x")
        except OSError:
            pass
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    started = time.monotonic()
    with requests.get(f"http://127.0.0.1:{port}/p.pdf", stream=True, timeout=5) as r:
        with pytest.raises(pdf_download.PdfDownloadTooSlowError):
            pdf_download.save_pdf(
                pdf_download.arriving_chunks(r),
                tmp_path / "p.pdf",
                source="local",
                deadline_seconds=0.5,
            )
    assert time.monotonic() - started < 2.5
    server.close()
