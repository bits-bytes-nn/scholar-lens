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


# A package's code is read, never run: the archive is opened in memory and
# only its .py members are kept.
MAX_ARCHIVE_BYTES = 80_000_000
MAX_SOURCE_CHARS = 12_000_000
_SKIP_PART = re.compile(r"(?:^|/)(?:tests?|testing|benchmarks?|examples?|docs?)/")


def package_source(
    name: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 60.0,
    max_bytes: int = MAX_ARCHIVE_BYTES,
) -> tuple[str, dict[str, str]] | None:
    """(version, {path in the archive: Python source}) of ``name``'s latest
    release on PyPI, from its smallest distribution (a pure wheel, else the
    sdist). None when the index does not know it, the archive is larger
    than ``max_bytes`` or cannot be read: the code check is skipped, never
    failed."""
    import io
    import tarfile
    import zipfile

    if not _NAME.match(name):
        return None
    owned = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        response = http.get(PYPI_JSON_URL.format(name=name))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        version = str((data.get("info") or {}).get("version") or "")
        files = [
            f
            for f in data.get("urls") or []
            if f.get("filename", "").endswith((".whl", ".tar.gz", ".zip"))
            and int(f.get("size") or 0) <= max_bytes
        ]
        if not files:
            logger.info("No readable distribution of %s within the size cap.", name)
            return None
        chosen = min(files, key=lambda f: int(f.get("size") or 0))
        archive = http.get(chosen["url"])
        archive.raise_for_status()
        blob = archive.content[: max_bytes + 1]
    except (httpx.HTTPError, ValueError, KeyError) as e:
        logger.warning("PyPI source of %s could not be read: %s", name, e)
        return None
    finally:
        if owned:
            http.close()
    if len(blob) > max_bytes:
        return None
    sources: dict[str, str] = {}
    total = 0

    def keep(path: str, raw: bytes) -> bool:
        nonlocal total
        if not path.endswith(".py") or _SKIP_PART.search(path):
            return True
        text = raw.decode("utf-8", errors="replace")
        total += len(text)
        if total > MAX_SOURCE_CHARS:
            return False
        sources[path] = text
        return True

    try:
        if chosen["filename"].endswith((".whl", ".zip")):
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                for info in zf.infolist():
                    if not info.is_dir() and not keep(info.filename, zf.read(info)):
                        break
        else:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    handle = tf.extractfile(member)
                    if handle is not None and not keep(member.name, handle.read()):
                        break
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as e:
        logger.warning("Archive of %s could not be read: %s", name, e)
        return None
    return version, sources
