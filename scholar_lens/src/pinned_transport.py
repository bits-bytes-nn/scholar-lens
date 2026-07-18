"""httpx transports that pin every request to an SSRF-validated IP.

A plain SSRF guard resolves the hostname, then the HTTP client resolves it
*again* independently to connect — a DNS-rebinding TOCTOU window where the second
lookup can return a private/metadata IP the guard never saw. These transports
close that window: each request (including every redirect hop, since each
re-enters ``handle_request``) is validated via :func:`resolve_validated_ip`, and
the connection is pinned to the exact IP that was validated. The original
hostname is preserved as the TLS SNI / ``Host`` header so certificate
verification and virtual-hosting still work.

A request to a non-public target raises :class:`UnsafeUrlError`, which the
fetchers already treat as "skip this URL".
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import requests
from requests.adapters import HTTPAdapter

from .url_guard import resolve_validated_ip


def _pin(request: httpx.Request) -> None:
    """Validate the request URL and rewrite its host to the validated IP.

    No-op for an IP literal (already validated, nothing to rewrite). Raises
    :class:`UnsafeUrlError` for a non-public target.
    """
    host = request.url.host
    pinned_ip = resolve_validated_ip(str(request.url))
    if pinned_ip is None:
        return  # IP literal — connect as-is
    # Connect to the validated IP, but keep the real hostname for TLS SNI and the
    # Host header so cert verification / vhosting still succeed.
    request.url = request.url.copy_with(host=pinned_ip)
    request.extensions = dict(request.extensions)
    request.extensions["sni_hostname"] = host


class PinnedHTTPTransport(httpx.HTTPTransport):
    """Sync transport that pins each request to an SSRF-validated IP."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        _pin(request)
        return super().handle_request(request)


class PinnedAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    """Async transport that pins each request to an SSRF-validated IP."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        _pin(request)
        return await super().handle_async_request(request)


class PinnedRequestsAdapter(HTTPAdapter):
    """``requests`` adapter that pins each request to an SSRF-validated IP.

    The ``requests``-based fetchers (e.g. the arbitrary-PDF-URL source) validate
    a URL, then let urllib3 resolve the host *again* to connect — the same
    DNS-rebinding TOCTOU window :class:`PinnedHTTPTransport` closes for httpx.
    This adapter re-validates and pins on every ``send`` (each redirect hop, when
    redirects are followed, re-enters ``send``): it rewrites the URL host to the
    validated IP and sets the ``Host`` header + TLS ``server_hostname`` to the
    original hostname so certificate verification and virtual-hosting still work.

    A non-public target raises :class:`~scholar_lens.src.url_guard.UnsafeUrlError`.
    """

    def send(
        self, request: requests.PreparedRequest, **kwargs: Any
    ) -> requests.Response:
        url = request.url or ""
        parts = urlsplit(url)
        host = parts.hostname or ""
        pinned_ip = resolve_validated_ip(url)  # raises UnsafeUrlError if not public
        if pinned_ip is None:
            return super().send(request, **kwargs)  # IP literal — connect as-is
        # Preserve the Host header so the server routes/certifies by hostname.
        request.headers.setdefault("Host", parts.netloc)
        # Rewrite the connection target to the validated IP (bracket IPv6).
        connect_host = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
        netloc = connect_host + (f":{parts.port}" if parts.port else "")
        request.url = urlunsplit(
            (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
        )
        # Pin TLS SNI / cert verification to the real hostname, not the IP.
        self._pinned_server_hostname = host
        return super().send(request, **kwargs)

    def get_connection_with_tls_context(
        self,
        request: requests.PreparedRequest,
        verify: bool | str | None,
        proxies: Any = None,
        cert: Any = None,
    ) -> Any:
        conn = super().get_connection_with_tls_context(
            request, verify, proxies=proxies, cert=cert
        )
        host = getattr(self, "_pinned_server_hostname", None)
        if host and hasattr(conn, "conn_kw"):
            # Send the real hostname as TLS SNI so certificate verification and
            # virtual-hosting succeed even though we connect to the pinned IP.
            # urllib3 verifies the cert against server_hostname, so setting it
            # alone is sufficient (do NOT also pass assert_hostname — the pool
            # supplies its own and a duplicate kwarg raises TypeError).
            conn.conn_kw["server_hostname"] = host
        return conn
