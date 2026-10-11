"""Shared pytest fixtures for the Scholar-Lens test suite.

All AWS and network access is mocked — these tests never make real calls and
incur zero cost. AWS is faked with ``moto``; HTTP with ``responses``.
"""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

import boto3
import pytest

# Ensure AWS SDK never picks up real credentials/region during tests.
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# Test output must not land in the repository's logs/ directory. The app's file
# handler opens its file lazily, so dropping it here leaves no file behind.
from scholar_lens.src.logger import logger as _app_logger  # noqa: E402

for _handler in [h for h in _app_logger.handlers if isinstance(h, logging.FileHandler)]:
    _app_logger.removeHandler(_handler)
    _handler.close()

# A minimal valid PDF: header + EOF marker. ``%PDF-`` magic bytes make the
# PdfUrlSource validation pass; PyMuPDF can open it.
MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n"
    b"trailer<</Size 4/Root 1 0 R>>\n"
    b"startxref\n0\n%%EOF\n"
)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly on any real outbound socket connection.

    Defense-in-depth: ``responses``/``moto`` patch at the HTTP/boto layer, so
    legitimately-mocked calls never reach here. Any UNmocked network attempt
    (a leak) raises instead of silently hitting the internet, keeping the suite
    deterministic and free.
    """
    real_connect = socket.socket.connect

    def guarded_connect(self, address):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else address
        # Allow loopback (moto's local endpoints / responses internals).
        if host in ("127.0.0.1", "::1", "localhost"):
            return real_connect(self, address)
        raise RuntimeError(
            f"Blocked unexpected network connection to {address!r} during tests. "
            "Mock it with `responses` or `moto`."
        )

    def guarded_connect_ex(self, address):  # type: ignore[no-untyped-def]
        guarded_connect(self, address)
        return 0

    real_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Resolve every public name to a documentation-range address (RFC 5737
        # TEST-NET-3 is not "global", so use a fixed public one) without real
        # DNS: tests stay independent of the runner's resolver, and the
        # connect guard above still blocks any actual connection.
        if host in (None, "localhost", "127.0.0.1", "::1"):
            return real_getaddrinfo(host, port, *args, **kwargs)
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 0))
        ]

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    return MINIMAL_PDF


@pytest.fixture
def aws_region() -> str:
    return "us-east-1"


@pytest.fixture
def boto_session(aws_region: str) -> boto3.Session:
    return boto3.Session(region_name=aws_region)


@pytest.fixture
def tmp_papers_dir(tmp_path: Path) -> Path:
    d = tmp_path / "papers"
    d.mkdir()
    return d
