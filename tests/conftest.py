"""Shared pytest fixtures for the Scholar-Lens test suite.

All AWS and network access is mocked — these tests never make real calls and
incur zero cost. AWS is faked with ``moto``; HTTP with ``responses``.
"""

from __future__ import annotations

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

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


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
