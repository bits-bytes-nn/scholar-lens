"""Abstractions for resolving a paper from heterogeneous sources.

Historically Scholar-Lens only accepted arXiv IDs. This module decouples the
pipeline from arXiv by introducing a small :class:`PaperSource` interface with
two concrete implementations:

* :class:`ArxivSource` — wraps :class:`ArxivHandler`; rich metadata via the
  arXiv API.
* :class:`PdfUrlSource` — accepts an arbitrary URL that must serve a PDF. It
  derives a stable identifier from the URL and downloads the file; descriptive
  metadata (title, category, keywords) is left for the LLM attribute-extraction
  stage, exactly as it already happens for arXiv papers.

:func:`resolve_paper_source` chooses the right implementation for a raw
``--source`` argument (an arXiv ID or a URL) and is the single entry point the
pipeline should use.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from pydantic import HttpUrl

from .arxiv_handler import ArxivHandler, ArxivMetadata
from .logger import logger
from .url_guard import (
    UnsafeUrlError,
    assert_url_is_public,
    assert_url_scheme_and_literal,
)

# arXiv IDs come in two shapes: new style "2401.06066" (optionally "v3") and the
# legacy "math.GT/0309136" form. We only need to recognise the modern style plus
# the obvious "arxiv.org/abs/..." URLs; everything else is treated as a generic
# URL source.
_ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_HOST_PATTERN = re.compile(r"(^|\.)arxiv\.org$", re.IGNORECASE)

_PDF_DOWNLOAD_TIMEOUT_SECONDS = 30
_PDF_PROBE_TIMEOUT_SECONDS = 15
_PDF_MAGIC_BYTES = b"%PDF-"
# Cap PDF downloads so a hostile/oversized URL cannot exhaust disk or memory.
_MAX_PDF_BYTES = 100 * 1024 * 1024  # 100 MB


class PaperSourceError(Exception):
    """Base error for paper-source resolution and download."""


class NotAPdfError(PaperSourceError):
    """Raised when a supplied URL does not resolve to a PDF document."""


class PaperSource(ABC):
    """A resolvable paper source.

    The ``source_id`` is a filesystem- and S3-safe slug used for directory
    naming; ``pdf_url`` is the canonical link shown in references.
    """

    @property
    @abstractmethod
    def source_id(self) -> str: ...

    @property
    @abstractmethod
    def pdf_url(self) -> HttpUrl: ...

    @property
    def arxiv_html_id(self) -> str | None:
        """arXiv identifier for HTML rendering, or ``None`` if not arXiv-backed.

        Sources that expose an arXiv HTML rendering return the id the arXiv
        HTML parser expects; all others return ``None``, signalling that only
        PDF parsing is available.
        """
        return None

    @abstractmethod
    def fetch_metadata(self) -> ArxivMetadata: ...

    @abstractmethod
    def download_pdf(self, papers_dir: Path) -> Path: ...

    def close(self) -> None:  # noqa: B027 - intentional optional no-op hook
        """Release any held resources (e.g. HTTP sessions). Default no-op."""


class ArxivSource(PaperSource):
    """Paper backed by the arXiv API."""

    def __init__(self, arxiv_id: str, handler: ArxivHandler | None = None) -> None:
        self._arxiv_id = arxiv_id
        self._handler = handler or ArxivHandler()

    @property
    def source_id(self) -> str:
        return self._arxiv_id.replace(".", "_")

    @property
    def arxiv_html_id(self) -> str | None:
        """The raw arXiv identifier (with dots), as accepted by the arXiv API."""
        return self._arxiv_id

    @property
    def pdf_url(self) -> HttpUrl:
        metadata = self.fetch_metadata()
        if metadata.pdf_url is None:
            raise PaperSourceError(f"arXiv paper '{self._arxiv_id}' has no PDF URL.")
        return metadata.pdf_url

    def fetch_metadata(self) -> ArxivMetadata:
        return self._handler.fetch_metadata(self._arxiv_id)

    def download_pdf(self, papers_dir: Path) -> Path:
        return self._handler.download_paper(self._arxiv_id, papers_dir)


class PdfUrlSource(PaperSource):
    """Paper backed by an arbitrary URL that must serve a PDF.

    The URL is validated to actually return a PDF (via a ``HEAD``/streamed
    ``GET`` content-type and magic-byte check). Non-PDF URLs raise
    :class:`NotAPdfError`, satisfying the "reject if the URL is not a paper PDF"
    requirement.
    """

    def __init__(self, url: str, session: requests.Session | None = None) -> None:
        self._url = HttpUrl(url)
        self._raw_url = url
        # SSRF guard (cheap, offline): reject bad schemes and internal IP
        # literals at construction. The full DNS-resolving check runs at fetch
        # time in _assert_is_pdf so constructing a source does no network I/O.
        try:
            assert_url_scheme_and_literal(url)
        except UnsafeUrlError as e:
            raise PaperSourceError(str(e)) from e
        self._session = session or requests.Session()
        self._discovered_title: str | None = None

    @property
    def source_id(self) -> str:
        """Stable slug derived from the URL.

        Uses the PDF filename stem when present, disambiguated by a short hash
        of the full URL so distinct URLs never collide.
        """
        parsed = urlparse(self._raw_url)
        stem = Path(unquote(parsed.path)).stem
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()
        digest = hashlib.sha1(self._raw_url.encode("utf-8")).hexdigest()[:8]
        return f"{slug}-{digest}" if slug else f"paper-{digest}"

    @property
    def pdf_url(self) -> HttpUrl:
        return self._url

    def _fallback_title(self) -> str:
        """Human-readable title from the URL stem when no better one exists."""
        parsed = urlparse(self._raw_url)
        stem = Path(unquote(parsed.path)).stem
        words = re.sub(r"[^a-zA-Z0-9]+", " ", stem).strip()
        return words.title() if words else self.source_id

    def fetch_metadata(self) -> ArxivMetadata:
        """Minimal metadata.

        The title is taken from the PDF's embedded document metadata when it was
        discovered during :meth:`download_pdf`; otherwise it falls back to a
        human-readable form of the URL stem. Other descriptive fields are
        enriched by the downstream LLM attribute-extraction stage.
        """
        now = datetime.now(UTC)
        return ArxivMetadata(
            arxiv_id=self.source_id,
            title=self._discovered_title or self._fallback_title(),
            authors=["Unknown"],
            published=now,
            updated=now,
            pdf_url=self._url,
        )

    def download_pdf(self, papers_dir: Path) -> Path:
        self._assert_is_pdf()
        paper_dir = papers_dir / self.source_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = paper_dir / f"{self.source_id}.pdf"
        try:
            with self._session.get(
                self._raw_url, timeout=_PDF_DOWNLOAD_TIMEOUT_SECONDS, stream=True
            ) as response:
                response.raise_for_status()
                # Reject oversized downloads up front when the server declares a
                # length, and enforce the cap while streaming regardless.
                declared = response.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > _MAX_PDF_BYTES:
                    raise PaperSourceError(
                        f"PDF at '{self._raw_url}' exceeds the {_MAX_PDF_BYTES}-byte "
                        f"limit (Content-Length: {declared})."
                    )
                written = 0
                with pdf_path.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > _MAX_PDF_BYTES:
                            fh.close()
                            pdf_path.unlink(missing_ok=True)
                            raise PaperSourceError(
                                f"PDF at '{self._raw_url}' exceeds the "
                                f"{_MAX_PDF_BYTES}-byte limit."
                            )
                        fh.write(chunk)
        except requests.RequestException as e:
            raise PaperSourceError(
                f"Failed to download PDF from '{self._raw_url}': {e}"
            ) from e
        if pdf_path.stat().st_size == 0 or not self._has_pdf_magic(pdf_path):
            pdf_path.unlink(missing_ok=True)
            raise NotAPdfError(
                f"Downloaded content from '{self._raw_url}' is not a valid PDF."
            )
        self._discovered_title = self._extract_pdf_title(pdf_path)
        logger.info("Successfully downloaded PDF to: '%s'", pdf_path)
        return pdf_path

    @staticmethod
    def _extract_pdf_title(pdf_path: Path) -> str | None:
        """Best-effort title from the PDF's embedded document metadata."""
        try:
            import fitz  # PyMuPDF, already a project dependency

            with fitz.open(pdf_path) as doc:
                title = (doc.metadata or {}).get("title", "")
            title = " ".join(title.split())
            # Reject empty or junk titles (e.g. "untitled", a bare filename).
            if title and title.lower() not in {"untitled", pdf_path.stem.lower()}:
                return title
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Could not read PDF title from '%s': %s", pdf_path, e)
        return None

    def _assert_is_pdf(self) -> None:
        """Probe the URL and reject anything that is not a PDF.

        A ``.pdf`` suffix is only a weak hint — servers routinely return HTML
        login/landing pages from ``.pdf`` URLs — so we never trust it alone.
        We accept an explicit ``application/pdf`` content-type or confirmed PDF
        magic bytes; we reject an explicit HTML content-type; and when the
        content-type is missing/ambiguous we fall back to a ranged magic-byte
        probe. Only if every signal is inconclusive do we defer to the
        post-download magic-byte check in :meth:`download_pdf`.
        """
        # Full SSRF check (resolves DNS) right before the first real request.
        try:
            assert_url_is_public(self._raw_url)
        except UnsafeUrlError as e:
            raise PaperSourceError(str(e)) from e
        try:
            head = self._session.head(
                self._raw_url,
                timeout=_PDF_PROBE_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
            content_type = head.headers.get("Content-Type", "").lower()
            if "application/pdf" in content_type:
                return
            if content_type and "html" in content_type:
                raise NotAPdfError(
                    f"URL '{self._raw_url}' returned HTML, not a PDF "
                    f"(Content-Type: {content_type})."
                )
        except requests.RequestException as e:
            logger.debug("HEAD probe for '%s' failed: %s", self._raw_url, e)
        # Content-type was absent or ambiguous: confirm via magic bytes.
        if not self._probe_magic_bytes():
            raise NotAPdfError(f"URL '{self._raw_url}' does not appear to serve a PDF.")

    def _probe_magic_bytes(self) -> bool:
        try:
            with self._session.get(
                self._raw_url,
                timeout=_PDF_PROBE_TIMEOUT_SECONDS,
                stream=True,
                headers={"Range": "bytes=0-7"},
            ) as response:
                response.raise_for_status()
                prefix = next(response.iter_content(chunk_size=8), b"")
                return prefix.startswith(_PDF_MAGIC_BYTES)
        except requests.RequestException as e:
            logger.debug("Magic-byte probe for '%s' failed: %s", self._raw_url, e)
            return False

    @staticmethod
    def _has_pdf_magic(pdf_path: Path) -> bool:
        with pdf_path.open("rb") as fh:
            return fh.read(len(_PDF_MAGIC_BYTES)).startswith(_PDF_MAGIC_BYTES)

    def close(self) -> None:
        """Close the owned requests session (releases the connection pool)."""
        self._session.close()


def is_arxiv_id(value: str) -> bool:
    """True if ``value`` looks like a modern arXiv identifier (not a URL)."""
    return bool(_ARXIV_ID_PATTERN.match(value.strip()))


def extract_arxiv_id_from_url(url: str) -> str | None:
    """Return the arXiv ID embedded in an arxiv.org URL, else ``None``."""
    parsed = urlparse(url)
    if not _ARXIV_HOST_PATTERN.search(parsed.netloc):
        return None
    # Matches /abs/2401.06066, /pdf/2401.06066v2, /pdf/2401.06066v2.pdf
    match = re.search(r"/(?:abs|pdf|html)/(\d{4}\.\d{4,5}(?:v\d+)?)", parsed.path)
    return match.group(1) if match else None


def resolve_paper_source(
    source: str, *, arxiv_handler: ArxivHandler | None = None
) -> PaperSource:
    """Resolve a raw ``--source`` argument to a concrete :class:`PaperSource`.

    Accepts a bare arXiv ID, an ``arxiv.org`` URL (any of abs/pdf/html), or an
    arbitrary URL pointing at a paper PDF. Arbitrary non-PDF URLs are rejected
    later, at download time, via :class:`NotAPdfError`.
    """
    source = source.strip()
    if is_arxiv_id(source):
        return ArxivSource(source, handler=arxiv_handler)
    if source.lower().startswith(("http://", "https://")):
        if arxiv_id := extract_arxiv_id_from_url(source):
            logger.info("Detected arXiv ID '%s' from URL.", arxiv_id)
            return ArxivSource(arxiv_id, handler=arxiv_handler)
        return PdfUrlSource(source)
    raise PaperSourceError(
        f"Could not resolve '{source}' to an arXiv ID or a URL. Provide a valid "
        "arXiv ID (e.g. 2401.06066) or a URL to a paper PDF."
    )
