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

import httpx

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
