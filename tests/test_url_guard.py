"""SSRF url_guard and the IP-pinned transports (no real network)."""

from __future__ import annotations

import pytest

from scholar_lens.src.url_guard import (
    UnsafeUrlError,
    assert_url_is_public,
    is_url_public,
)

# --------------------------------------------------------------------------- #
# SSRF url_guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://127.0.0.1:8080/",  # loopback
        "http://localhost/x",  # loopback by name
        "http://10.0.0.5/internal",  # RFC1918
        "http://192.168.1.1/admin",  # RFC1918
        "http://[::1]/x",  # IPv6 loopback
        "ftp://example.com/x",  # disallowed scheme
        "file:///etc/passwd",  # disallowed scheme
        "http://0.0.0.0/",  # unspecified
    ],
)
def test_blocks_unsafe_urls(url: str) -> None:
    assert is_url_public(url) is False
    with pytest.raises(UnsafeUrlError):
        assert_url_is_public(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://1.1.1.1/x",  # public IP literal — no DNS needed
        "http://8.8.8.8/paper.pdf",
        "https://[2606:4700:4700::1111]/x",  # public IPv6 literal
    ],
)
def test_allows_public_ip_literals(url: str) -> None:
    # IP literals avoid DNS so the test stays hermetic under the network block.
    assert is_url_public(url) is True


def test_allows_public_hostname_via_mocked_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock DNS so we exercise the hostname path without real network.
    import scholar_lens.src.url_guard as guard

    monkeypatch.setattr(
        guard.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    assert is_url_public("https://example.com/paper.pdf") is True


def test_ipv4_mapped_ipv6_loopback_blocked() -> None:
    # ::ffff:127.0.0.1 must be unwrapped and rejected.
    assert is_url_public("http://[::ffff:127.0.0.1]/") is False


# --------------------------------------------------------------------------- #
# resolve_validated_ip + PinnedHTTPTransport (DNS-rebinding TOCTOU defence)
# --------------------------------------------------------------------------- #


def test_resolve_validated_ip_returns_validated_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scholar_lens.src.url_guard as guard
    from scholar_lens.src.url_guard import resolve_validated_ip

    monkeypatch.setattr(
        guard.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    assert resolve_validated_ip("https://example.com/x") == "93.184.216.34"


def test_resolve_validated_ip_none_for_literal() -> None:
    from scholar_lens.src.url_guard import resolve_validated_ip

    # An IP literal is already the connection target — no rewrite needed.
    assert resolve_validated_ip("https://1.1.1.1/x") is None


def test_resolve_validated_ip_blocks_private_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scholar_lens.src.url_guard as guard
    from scholar_lens.src.url_guard import resolve_validated_ip

    # A hostname that resolves to a private IP (the rebinding payload) is rejected.
    monkeypatch.setattr(
        guard.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("169.254.169.254", 0))],
    )
    with pytest.raises(UnsafeUrlError):
        resolve_validated_ip("https://evil.example.com/")


def test_pinned_transport_rewrites_host_to_validated_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The transport pins the connection to the validated IP while preserving the
    # hostname for TLS SNI / Host — verified without opening a socket.
    import httpx

    import scholar_lens.src.pinned_transport as pt

    monkeypatch.setattr(pt, "resolve_validated_ip", lambda url: "93.184.216.34")
    request = httpx.Request("GET", "https://example.com/page")
    pt._pin(request)
    assert request.url.host == "93.184.216.34"
    assert request.extensions["sni_hostname"] == "example.com"
    assert request.headers["host"] == "example.com"  # Host header unchanged

    # An IP literal (resolve returns None) is left untouched.
    monkeypatch.setattr(pt, "resolve_validated_ip", lambda url: None)
    literal_req = httpx.Request("GET", "https://1.1.1.1/x")
    pt._pin(literal_req)
    assert literal_req.url.host == "1.1.1.1"


@pytest.mark.parametrize(
    "url",
    [
        "http://100.100.100.200/latest/meta-data/",  # Alibaba metadata (CGNAT)
        "http://100.64.0.1/",  # shared address space
        "http://198.18.0.1/",  # benchmarking range
    ],
)
def test_blocks_non_global_ranges(url: str) -> None:
    assert is_url_public(url) is False


def test_pinned_transport_restores_hostname_on_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The connection goes to the pinned IP, but the response (and so every link
    # resolved against response.url) must keep the hostname.
    import httpx

    import scholar_lens.src.pinned_transport as pt

    monkeypatch.setattr(pt, "resolve_validated_ip", lambda url: "93.184.216.34")
    seen: list[str] = []

    def fake_send(self, request):  # type: ignore[no-untyped-def]
        seen.append(request.url.host)
        return httpx.Response(200, request=request, text="ok")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_send)
    with httpx.Client(transport=pt.PinnedHTTPTransport()) as client:
        response = client.get("https://docs.example.com/guide/")
    assert seen == ["93.184.216.34"]
    assert response.url.host == "docs.example.com"


async def test_pinned_async_transport_restores_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    import scholar_lens.src.pinned_transport as pt

    monkeypatch.setattr(pt, "resolve_validated_ip", lambda url: "93.184.216.34")

    async def fake_send(self, request):  # type: ignore[no-untyped-def]
        return httpx.Response(200, request=request, text="ok")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fake_send)
    async with httpx.AsyncClient(transport=pt.PinnedAsyncHTTPTransport()) as client:
        response = await client.get("https://docs.example.com/guide/")
    assert str(response.url) == "https://docs.example.com/guide/"


def test_pinned_requests_adapter_sni_and_url(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    from requests.adapters import HTTPAdapter

    import scholar_lens.src.pinned_transport as pt

    monkeypatch.setattr(pt, "resolve_validated_ip", lambda url: "93.184.216.34")
    adapter = pt.PinnedRequestsAdapter()
    captured: dict = {}

    def fake_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
        captured["url"] = request.url
        captured["host"] = request.headers.get("Host")
        captured["sni"] = getattr(self._local, "server_hostname", None)
        response = requests.Response()
        response.status_code = 200
        response.url = request.url
        return response

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    prepared = requests.Request("GET", "https://papers.example.com/a.pdf").prepare()
    response = adapter.send(prepared)
    assert captured["url"] == "https://93.184.216.34/a.pdf"
    assert captured["host"] == "papers.example.com"
    assert captured["sni"] == "papers.example.com"
    assert response.url == "https://papers.example.com/a.pdf"
    assert prepared.url == "https://papers.example.com/a.pdf"
    assert adapter._local.server_hostname is None
