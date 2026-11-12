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

import asyncio
import threading
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import requests
from requests.adapters import HTTPAdapter

from scholar_lens.adapters.http.url_guard import resolve_validated_ip, upgrade_to_https

_PER_HOST_LIMITS = httpx.Limits(max_connections=10, max_keepalive_connections=2)


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


class PinnedHTTPTransport(httpx.BaseTransport):
    """Sync transport that pins each request to an SSRF-validated IP.

    Connections are pooled per hostname, not per pinned IP: two sites behind
    one CDN address must never share a TLS connection opened with the other
    one's SNI (the second site answers 403, and its certificate is never
    checked)."""

    def __init__(self, **kwargs: Any) -> None:
        # A crawl touches many hosts once: few idle sockets kept per host.
        kwargs.setdefault("limits", _PER_HOST_LIMITS)
        self._kwargs = kwargs
        self._pools: dict[str, httpx.HTTPTransport] = {}
        self._lock = threading.Lock()

    def _pool(self, host: str) -> httpx.HTTPTransport:
        with self._lock:
            if host not in self._pools:
                self._pools[host] = httpx.HTTPTransport(**self._kwargs)
            return self._pools[host]

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        original = _pin(request)
        try:
            return self._pool(original.host).handle_request(request)
        finally:
            request.url = original

    def close(self) -> None:
        with self._lock:
            pools, self._pools = list(self._pools.values()), {}
        for pool in pools:
            pool.close()


# A name lookup longer than this is a dead resolver, not a slow page.
_RESOLVE_TIMEOUT_S = 15.0


class PinnedAsyncHTTPTransport(httpx.AsyncBaseTransport):
    """Async transport that pins each request to an SSRF-validated IP, with
    connections pooled per hostname (see :class:`PinnedHTTPTransport`)."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("limits", _PER_HOST_LIMITS)
        self._kwargs = kwargs
        self._pools: dict[str, httpx.AsyncHTTPTransport] = {}

    def _pool(self, host: str) -> httpx.AsyncHTTPTransport:
        # One event loop: no await between the check and the insert.
        if host not in self._pools:
            self._pools[host] = httpx.AsyncHTTPTransport(**self._kwargs)
        return self._pools[host]

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # The name lookup blocks: off the event loop, under its own deadline
        # (httpx's timeouts start at the connection).
        try:
            original = await asyncio.wait_for(
                asyncio.to_thread(_pin, request), _RESOLVE_TIMEOUT_S
            )
        except TimeoutError as e:
            raise httpx.ConnectTimeout(
                f"resolving {request.url.host} took over {_RESOLVE_TIMEOUT_S}s",
                request=request,
            ) from e
        try:
            return await self._pool(original.host).handle_async_request(request)
        finally:
            request.url = original

    async def aclose(self) -> None:
        pools, self._pools = list(self._pools.values()), {}
        for pool in pools:
            await pool.aclose()


class PinnedRequestsAdapter(HTTPAdapter):
    """``requests`` adapter that pins each request to an SSRF-validated IP.

    The ``requests``-based fetchers (e.g. the arbitrary-PDF-URL source) validate
    a URL, then let urllib3 resolve the host *again* to connect — the same
    DNS-rebinding TOCTOU window :class:`PinnedHTTPTransport` closes for httpx.
    This adapter re-validates and pins on every ``send`` (each redirect hop, when
    redirects are followed, re-enters ``send``): it rewrites the URL host to the
    validated IP and sets the ``Host`` header + TLS ``server_hostname`` to the
    original hostname so certificate verification and virtual-hosting still work.

    A non-public target raises :class:`~scholar_lens.adapters.http.url_guard.UnsafeUrlError`.
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
        # The Host header routes by hostname. Set for this hop and removed
        # after it: a redirect copies the request, and a stale Host would
        # reach the next host's pinned IP.
        request.headers["Host"] = parts.netloc
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
            request.headers.pop("Host", None)
        response.url = url
        return response

    def build_connection_pool_key_attributes(
        self, request: requests.PreparedRequest, verify: Any, cert: Any = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        host_params, pool_kwargs = super().build_connection_pool_key_attributes(
            request, verify, cert
        )
        host = getattr(self._local, "server_hostname", None)
        if host:
            # The real hostname as TLS SNI (certificate verification and
            # virtual hosting) and as part of the pool key: hosts sharing a
            # pinned IP never reuse each other's connections.
            pool_kwargs["server_hostname"] = host
        return host_params, pool_kwargs
