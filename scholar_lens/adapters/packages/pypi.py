"""PyPI's JSON API: a package's release (the run's target, else the latest)
and the extras it provides, so a guide's install commands can be checked
against the index."""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Callable
from typing import Any

import httpx

from scholar_lens.domain.guide.sdk_index import PackageSource
from scholar_lens.domain.sources import (
    IndexUnreachable,
    ReleaseUnavailable,
    newest_release,
    release_number,
    same_release,
)

logger = logging.getLogger(__name__)

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
PYPI_RELEASE_JSON_URL = "https://pypi.org/pypi/{name}/{version}/json"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$")


def _release_json(
    http: httpx.Client, name: str, release: str | None
) -> dict[str, Any] | None:
    """PyPI's JSON of ``name`` at ``release`` (its latest when None): a
    release the index has by that number, else the newest final release of
    the series it names ("0.6" docs: 0.6.3). None when the index does not
    know the package.

    Raises:
        ReleaseUnavailable: The package has no such release.
        httpx.HTTPError: PyPI could not answer.
    """
    if release:
        version = release_number(release) or release
        response = http.get(PYPI_RELEASE_JSON_URL.format(name=name, version=version))
        if response.status_code != 404:
            response.raise_for_status()
            return dict(response.json())
    project = http.get(PYPI_JSON_URL.format(name=name))
    if project.status_code == 404:
        return None
    project.raise_for_status()
    data = dict(project.json())
    if not release:
        return data
    series = [v for v in data.get("releases") or {} if same_release(v, version)]
    if (found := newest_release(series)) is None:
        raise ReleaseUnavailable(f"PyPI has no {name} release {release}")
    response = http.get(PYPI_RELEASE_JSON_URL.format(name=name, version=found))
    response.raise_for_status()
    return dict(response.json())


def package_extras(
    name: str,
    *,
    release: str | None = None,
    client: httpx.Client | None = None,
    timeout: float = 10.0,
) -> tuple[str, frozenset[str]] | None:
    """(version, extras) of ``name`` at ``release`` on PyPI (its latest
    release when None), or None when the index does not know it (or the
    name is not a package name).

    Raises:
        IndexUnreachable: PyPI could not answer (a timeout, a connection
            or server error, an unreadable answer): unlike an unknown
            package, the answer is missing.
        ReleaseUnavailable: PyPI knows the package but not ``release``.
    """
    if not _NAME.match(name):
        return None
    owned = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        data = _release_json(http, name, release)
        if data is None:
            return None
        info = data.get("info") or {}
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("PyPI lookup of %s failed: %s", name, e)
        raise IndexUnreachable(f"PyPI lookup of {name} failed: {e}") from e
    finally:
        if owned:
            http.close()
    extras = frozenset(e.lower() for e in info.get("provides_extra") or [])
    return str(info.get("version") or ""), extras


# A package's code is read, never run: the archive is opened in memory and
# only its .py members are kept.
MAX_ARCHIVE_BYTES = 80_000_000
MAX_SOURCE_CHARS = 12_000_000
# A project's own tests, docs, examples and benchmarks, beside its packages:
# at a wheel's root, or at an sdist's project root (or its src/). Inside an
# importable package (pkg/testing/) such a folder is public API and kept.
_SKIP_DIRS = r"(?:tests?|testing|benchmarks?|examples?|docs?)/"
_SKIP_IN_WHEEL = re.compile(rf"^{_SKIP_DIRS}")
_SKIP_IN_SDIST = re.compile(rf"^[^/]+/(?:src/)?{_SKIP_DIRS}")
# The top-level import names a distribution's metadata lists.
_TOP_LEVEL = re.compile(r"(?:^|/)[^/]+\.(?:dist|egg)-info/top_level\.txt$")
_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")


def package_source(
    name: str,
    *,
    release: str | None = None,
    client: httpx.Client | None = None,
    timeout: float = 60.0,
    max_bytes: int = MAX_ARCHIVE_BYTES,
    max_chars: int = MAX_SOURCE_CHARS,
) -> PackageSource | None:
    """The Python code of ``name`` at ``release`` on PyPI (its latest
    release when None), from its smallest distribution (a pure wheel, else
    the sdist). None when the index does not know it, the archive is larger
    than ``max_bytes`` or cannot be read: the code check is skipped, never
    failed. More than ``max_chars`` of code is cut, and the result says it
    is partial.

    Raises:
        ReleaseUnavailable: PyPI knows the package but not ``release``, or
            has no distribution of it within ``max_bytes``.
    """
    import io
    import tarfile
    import zipfile

    if not _NAME.match(name):
        return None
    owned = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        data = _release_json(http, name, release)
        if data is None:
            return None
        version = str((data.get("info") or {}).get("version") or "")
        files = [
            f
            for f in data.get("urls") or []
            if f.get("filename", "").endswith((".whl", ".tar.gz", ".zip"))
            and int(f.get("size") or 0) <= max_bytes
        ]
        if not files:
            logger.info("No readable distribution of %s within the size cap.", name)
            if release:
                raise ReleaseUnavailable(
                    f"PyPI has no readable distribution of {name} {version} "
                    "within the size cap"
                )
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
    wheel = chosen["filename"].endswith(".whl")
    sources: dict[str, str] = {}
    tops: set[str] = set()
    total = 0
    complete = True

    def keep(path: str, read: Callable[[], bytes], size: int) -> None:
        nonlocal total, complete
        if size > max_chars:
            # Never decompress a member larger than the whole budget (an
            # archive bomb); a real source file is far smaller.
            if path.endswith(".py"):
                complete = False
            return
        if _TOP_LEVEL.match(path):
            text = read().decode("utf-8", errors="replace")
            tops.update(t for t in text.split() if _IDENTIFIER.match(t))
            return
        skip = _SKIP_IN_WHEEL if wheel else _SKIP_IN_SDIST
        if not path.endswith(".py") or skip.match(path):
            return
        root = path.split("/", 1)[0]
        if wheel and "/" in path and not root.endswith((".dist-info", ".data")):
            tops.add(root)
        if not complete:
            return
        text = read().decode("utf-8", errors="replace")
        total += len(text)
        if total > max_chars:
            complete = False
            return
        sources[path] = text

    try:
        if chosen["filename"].endswith((".whl", ".zip")):
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                for info in zf.infolist():
                    if not info.is_dir():
                        keep(
                            info.filename,
                            functools.partial(zf.read, info),
                            info.file_size,
                        )
        else:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    handle = tf.extractfile(member)
                    if handle is not None:
                        keep(member.name, handle.read, member.size)
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as e:
        logger.warning("Archive of %s could not be read: %s", name, e)
        return None
    if not complete:
        logger.warning(
            "Source of %s %s is over %d characters: the index is partial "
            "(%d files read).",
            name,
            version,
            max_chars,
            len(sources),
        )
    return PackageSource(version, sources, complete, frozenset(tops))


class PyPIIndex:
    """The :class:`~scholar_lens.application.ports.PackageIndex` on PyPI."""

    def extras(
        self, name: str, release: str | None = None
    ) -> tuple[str, frozenset[str]] | None:
        """The protocol's answer: None when unknown or unreachable (the
        extras check of an install command is skipped, never failed)."""
        try:
            return package_extras(name, release=release)
        except IndexUnreachable:
            return None

    def source(self, name: str, release: str | None = None) -> PackageSource | None:
        return package_source(name, release=release)
