"""httpx transports that pin every request to an SSRF-validated IP.

A plain SSRF guard resolves the hostname, then the HTTP client resolves it
*again* independently to connect — a DNS-rebinding TOCTOU window where the second
lookup can return a private/metadata IP the guard never saw. These transports
close that window: each request (including every redirect hop, since each
re-enters ``handle_request``) is validated via :func:`resolve_validated_ip`, and
the connection is pinned to the exact IP that was validated. The original
hostname is preserved as the TLS SNI / ``Host`` header so certificate
verification and virtual-hosting still work. Every hop is also upgraded from
``http`` to ``https`` (:func:`upgrade_to_https`): the Batch security group only
allows 443 egress.

A request to a non-public target raises :class:`UnsafeUrlError`, which the
fetchers already treat as "skip this URL".
"""

from __future__ import annotations

import threading
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import requests
from requests.adapters import HTTPAdapter

from .url_guard import resolve_validated_ip, upgrade_to_https


def _pin(request: httpx.Request) -> httpx.URL:
    """Upgrade the request URL to https, validate it and pin its host to the
    validated IP.

    Returns the (upgraded) hostname URL so the caller can restore it after
    sending (the response, redirects and link resolution must see the
    hostname, never the pinned IP). No IP rewrite for an IP literal. Raises
    :class:`UnsafeUrlError` for a non-public target.
    """
    original = httpx.URL(upgrade_to_https(str(request.url)))
    request.url = original
    pinned_ip = resolve_validated_ip(str(original))
    if pinned_ip is None:
        return original  # IP literal — connect as-is
    # Connect to the validated IP, but keep the real hostname for TLS SNI and the
    # Host header so cert verification / vhosting still succeed.
    request.url = original.copy_with(host=pinned_ip)
    request.extensions = dict(request.extensions)
    request.extensions["sni_hostname"] = original.host
    return original


class PinnedHTTPTransport(httpx.HTTPTransport):
    """Sync transport that pins each request to an SSRF-validated IP."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        original = _pin(request)
        try:
            return super().handle_request(request)
        finally:
            request.url = original


class PinnedAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    """Async transport that pins each request to an SSRF-validated IP."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        original = _pin(request)
        try:
            return await super().handle_async_request(request)
        finally:
            request.url = original


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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._local = threading.local()

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: Any = None,
        verify: Any = True,
        cert: Any = None,
        proxies: dict[str, str] | None = None,
    ) -> requests.Response:
        kwargs: dict[str, Any] = {
            "stream": stream,
            "timeout": timeout,
            "verify": verify,
            "cert": cert,
            "proxies": proxies,
        }
        url = upgrade_to_https(request.url or "")
        request.url = url
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
        # Per-thread, so concurrent sends through one adapter never swap SNI.
        self._local.server_hostname = host
        try:
            response = super().send(request, **kwargs)
        finally:
            self._local.server_hostname = None
            request.url = url
        response.url = url
        return response

    def get_connection_with_tls_context(
        self,
        request: requests.PreparedRequest,
        verify: Any,
        proxies: Any = None,
        cert: Any = None,
    ) -> Any:
        conn = super().get_connection_with_tls_context(
            request, verify, proxies=proxies, cert=cert
        )
        host = getattr(self._local, "server_hostname", None)
        if host and hasattr(conn, "conn_kw"):
            # Send the real hostname as TLS SNI so certificate verification and
            # virtual-hosting succeed even though we connect to the pinned IP.
            # urllib3 verifies the cert against server_hostname, so setting it
            # alone is sufficient (do NOT also pass assert_hostname — the pool
            # supplies its own and a duplicate kwarg raises TypeError).
            conn.conn_kw["server_hostname"] = host
        return conn
