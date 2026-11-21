"""Shared pytest fixtures for the Scholar-Lens test suite.

All AWS and network access is mocked — these tests never make real calls and
incur zero cost. AWS is faked with ``moto``; HTTP with ``responses``.
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest

# The AWS SDK never sees the developer's credentials, profile or region:
# fake keys win over anything exported, and the config and credentials files
# are empty (a session built with an explicit profile name then fails).
for _name in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_REGION"):
    os.environ.pop(_name, None)
os.environ["AWS_CONFIG_FILE"] = os.devnull
os.environ["AWS_SHARED_CREDENTIALS_FILE"] = os.devnull
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
# Child processes (git, sandboxed programs) do not inherit the socket guard
# below: git may use only local repositories, and HTTP from a child goes to a
# closed local port instead of the network.
os.environ["GIT_ALLOW_PROTOCOL"] = "file"
for _name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
    os.environ[_name] = "http://127.0.0.1:9"
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost,::1"

# Test output must not land in the repository's logs/ directory. The app's file
# handler opens its file lazily, so dropping it here leaves no file behind.
from scholar_lens.entrypoints.logs import configure_logging  # noqa: E402

_app_logger = configure_logging()

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
def _block_network(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[None]:
    """Fail loudly on any real outbound socket connection.

    Defense-in-depth: ``responses``/``moto`` patch at the HTTP/boto layer, so
    legitimately-mocked calls never reach here. Any UNmocked network attempt
    (a leak) raises instead of silently hitting the internet, keeping the suite
    deterministic and free. The test also fails when the code under test
    swallowed that error (a broad ``except`` turning it into a fallback),
    unless it is marked ``blocked_network`` because it exercises that path.
    """
    real_connect = socket.socket.connect
    blocked: list[object] = []

    def guarded_connect(self, address):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else address
        # Allow loopback (moto's local endpoints / responses internals).
        if host in ("127.0.0.1", "::1", "localhost"):
            return real_connect(self, address)
        blocked.append(address)
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
    yield
    if blocked and request.node.get_closest_marker("blocked_network") is None:
        pytest.fail(
            f"The test tried the network ({blocked!r}) and the code under test "
            "swallowed the guard's error: give it a fake, or mark the test "
            "`blocked_network` when that failure is what it checks."
        )


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


@pytest.fixture(autouse=True)
def _no_arxiv_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The process-wide arXiv limiter (1 request per 3 s) would make every test
    that touches the arXiv client wait in real time."""
    from scholar_lens.adapters.scholarly import arxiv as arxiv_handler

    from scholar_lens.adapters.http.rate_limit import (
        ARXIV_ASSET_LIMITER,
        ARXIV_RATE_LIMITER,
    )

    monkeypatch.setattr(arxiv_handler._ARXIV_RATE_LIMITER, "acquire", lambda: None)
    # A penalty one test sets (a 429's Retry-After) never carries into the next.
    for limiter in (ARXIV_RATE_LIMITER, ARXIV_ASSET_LIMITER):
        monkeypatch.setattr(limiter, "_blocked_until", 0.0)


@pytest.fixture(autouse=True)
def _no_developer_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests see no API key, token or alternate config from the developer's
    shell or .env (a key changes which providers a pipeline calls)."""
    from scholar_lens.configs.config import CONFIG_PATH_ENV
    from scholar_lens.domain.constants import EnvVars

    for name in [*(v.value for v in EnvVars), CONFIG_PATH_ENV]:
        if name not in ("LOG_LEVEL", "AWS_PROFILE_NAME"):
            monkeypatch.delenv(name, raising=False)
