"""PyPI's JSON API: a package's latest version and the extras it provides,
so a guide's install commands can be checked against the index."""

from __future__ import annotations

import logging
import re

import httpx

logger = logging.getLogger(__name__)

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$")


def package_extras(
    name: str, *, client: httpx.Client | None = None, timeout: float = 10.0
) -> tuple[str, frozenset[str]] | None:
    """(latest version, extras) of ``name`` on PyPI, or None when the index
    does not know it or cannot be reached (the check is skipped, never
    failed)."""
    if not _NAME.match(name):
        return None
    owned = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        response = http.get(PYPI_JSON_URL.format(name=name))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        info = response.json().get("info") or {}
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("PyPI lookup of %s failed: %s", name, e)
        return None
    finally:
        if owned:
            http.close()
    extras = frozenset(e.lower() for e in info.get("provides_extra") or [])
    return str(info.get("version") or ""), extras
