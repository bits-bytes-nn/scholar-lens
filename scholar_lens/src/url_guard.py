"""SSRF guard for user-supplied URLs.

The pipeline fetches URLs that originate from untrusted input — an arbitrary
paper PDF URL, documentation URLs for the tech-guide, and links/images
discovered while crawling those pages. Without a check, a caller could point any
of these at an internal address (``127.0.0.1``, RFC-1918 ranges) or, most
dangerously, the cloud instance-metadata endpoint ``169.254.169.254`` and have
the server fetch it on their behalf.

:func:`assert_url_is_public` validates scheme/host and resolves the hostname,
rejecting any address that is loopback, private, link-local, or otherwise not a
normal public unicast address. It raises :class:`UnsafeUrlError` on rejection so
callers can convert it into a domain-specific error.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

from .logger import logger

_ALLOWED_SCHEMES = {"http", "https"}
# The cloud metadata service. Covered by is_link_local already, but called out
# explicitly so the intent (and the IPv6 form) is unmistakable.
_METADATA_ADDRESSES = {"169.254.169.254", "fd00:ec2::254"}


class UnsafeUrlError(ValueError):
    """Raised when a URL is rejected by the SSRF guard."""


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or str(ip) in _METADATA_ADDRESSES
        # IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) must be unwrapped and checked.
        or (
            isinstance(ip, ipaddress.IPv6Address)
            and ip.ipv4_mapped is not None
            and _is_blocked_ip(ip.ipv4_mapped)
        )
    )


def assert_url_scheme_and_literal(url: str) -> None:
    """Cheap, offline SSRF pre-check: scheme, host presence, and IP literals.

    Does NOT perform DNS resolution, so it is safe to call at object-construction
    time (no network I/O). Use :func:`assert_url_is_public` at fetch time for the
    full resolving check.
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeUrlError(
            f"URL scheme '{parts.scheme}' is not allowed (only http/https)."
        )
    host = parts.hostname
    if not host:
        raise UnsafeUrlError(f"URL '{url}' has no host.")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        return  # hostname, not a literal — DNS check happens at fetch time
    if _is_blocked_ip(literal):
        raise UnsafeUrlError(f"URL '{url}' targets a non-public address ({host}).")


def assert_url_is_public(url: str) -> None:
    """Reject non-public URLs to prevent SSRF.

    Validates the scheme is http(s), the host is present, and every IP the host
    resolves to is a public unicast address. Raises :class:`UnsafeUrlError`
    otherwise. DNS resolution failures are also treated as unsafe (fail closed).
    Performs DNS resolution — call at fetch time, not in hot constructors.
    """
    assert_url_scheme_and_literal(url)
    parts = urlsplit(url)
    host = parts.hostname
    # If the host is an IP literal, the pre-check already validated it.
    try:
        ipaddress.ip_address(host or "")
        return
    except ValueError:
        pass

    # Hostname: resolve and check every returned address (fail closed on error).
    try:
        infos = socket.getaddrinfo(host, parts.port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise UnsafeUrlError(f"Could not resolve host '{host}': {e}") from e
    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise UnsafeUrlError(f"Host '{host}' resolved to no addresses.")
    for addr in addresses:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            raise UnsafeUrlError(
                f"URL '{url}' resolves to a non-public address ({addr})."
            )


def is_url_public(url: str) -> bool:
    """Non-raising form of :func:`assert_url_is_public`."""
    try:
        assert_url_is_public(url)
        return True
    except UnsafeUrlError as e:
        logger.debug("Blocked non-public URL '%s': %s", url, e)
        return False


def resolve_validated_ip(url: str) -> str | None:
    """Validate ``url`` and return ONE public IP to connect to (closing TOCTOU).

    :func:`assert_url_is_public` resolves and validates the host, but a plain
    client then resolves *again* independently — a DNS-rebinding window where the
    second lookup can return a private IP. To eliminate that, a fetcher should
    connect to the exact IP validated here (pinning the host to this address and
    sending the original hostname as TLS SNI / Host header).

    Returns the validated IP for a hostname; ``None`` for an IP literal (the
    literal is itself the connection target and is already validated, so no
    rewrite is needed). Raises :class:`UnsafeUrlError` exactly like
    :func:`assert_url_is_public`.
    """
    assert_url_scheme_and_literal(url)
    parts = urlsplit(url)
    host = parts.hostname or ""
    try:
        ipaddress.ip_address(host)
        return None  # literal: already validated, connect as-is
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, parts.port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise UnsafeUrlError(f"Could not resolve host '{host}': {e}") from e
    addresses = [str(info[4][0]) for info in infos]
    if not addresses:
        raise UnsafeUrlError(f"Host '{host}' resolved to no addresses.")
    # Every resolved address must be public (an attacker could otherwise round-
    # robin a private one in); pin the first.
    for addr in addresses:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            raise UnsafeUrlError(
                f"URL '{url}' resolves to a non-public address ({addr})."
            )
    return addresses[0]
