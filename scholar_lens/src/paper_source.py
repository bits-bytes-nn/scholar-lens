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
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from pydantic import HttpUrl

from .arxiv_handler import ArxivHandler, ArxivMetadata
from .constants import arxiv_id_from_url, normalize_arxiv_id
from .logger import logger
from .pdf_download import (
    CHUNK_BYTES,
    NotPdfContentError,
    PdfTooLargeError,
    save_pdf,
)
from .pinned_transport import PinnedRequestsAdapter
from .url_guard import (
    UnsafeUrlError,
    assert_url_is_public,
    assert_url_scheme_and_literal,
    is_temporary_dns_failure,
)

# "arxiv.org/abs/2106.09685" or "openreview.net/pdf?id=…" typed without a scheme.
_SCHEMELESS_URL = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}/", re.IGNORECASE)

_PDF_DOWNLOAD_TIMEOUT_SECONDS = 30
# Max redirect hops to follow manually (each re-validated against the SSRF guard).
_MAX_REDIRECTS = 5


class PaperSourceError(Exception):
    """Base error for paper-source resolution and download: the request
    itself is unusable (bad URL, not a paper)."""


class NotAPdfError(PaperSourceError):
    """Raised when a supplied URL does not resolve to a PDF document."""


class PaperFetchError(PaperSourceError):
    """The paper could not be fetched for transport reasons (timeout, refused
    connection, 429, 5xx): an outage worth retrying, not a bad request."""


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

    @property
    def title_is_fallback(self) -> bool:
        """Whether the metadata title is a best-effort guess that a title parsed
        from the paper text should replace."""
        return False

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
        return re.sub(r"[./]", "_", self._arxiv_id)

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
        # Into this source's own folder (an unversioned id would otherwise land
        # in "<id>vN/" while the rest of the paper lives in "<id>/").
        return self._handler.download_paper(
            self._arxiv_id, papers_dir, folder=self.source_id
        )


class PdfUrlSource(PaperSource):
    """Paper backed by an arbitrary URL that must serve a PDF.

    The download is one streamed ``GET`` checked by content type and magic
    bytes. A URL that answers with something other than a PDF raises
    :class:`NotAPdfError`; a transport error or 429/5xx raises
    :class:`PaperFetchError`.
    """

    def __init__(self, url: str, session: requests.Session | None = None) -> None:
        self._url = HttpUrl(url)
        self._raw_url = url
        # SSRF guard (cheap, offline): reject bad schemes and internal IP
        # literals at construction. The full DNS-resolving check runs when the
        # PDF is fetched, so constructing a source does no network I/O.
        try:
            assert_url_scheme_and_literal(url)
        except UnsafeUrlError as e:
            raise PaperSourceError(str(e)) from e
        self._session = session or requests.Session()
        # Pin every hop to the SSRF-validated IP so a low-TTL DNS name can't
        # rebind to an internal/metadata address between the guard check and the
        # actual connect (the TOCTOU window a plain requests.Session leaves open).
        # Skipped when the caller injected its own session (tests use `responses`,
        # which stubs the transport).
        if session is None:
            pinned_adapter = PinnedRequestsAdapter()
            self._session.mount("https://", pinned_adapter)
            self._session.mount("http://", pinned_adapter)
        self._discovered_title: str | None = None

    @property
    def source_id(self) -> str:
        """Stable slug derived from the URL.

        Uses the PDF filename stem when present, disambiguated by a short hash
        of the full URL so distinct URLs never collide.
        """
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", self._url_stem()).strip("-").lower()
        digest = hashlib.sha1(self._raw_url.encode("utf-8")).hexdigest()[:8]
        return f"{slug}-{digest}" if slug else f"paper-{digest}"

    def _url_stem(self) -> str:
        """The paper's name in its URL: the file stem, or the ``id`` query value
        for endpoints such as OpenReview's ``/pdf?id=nZeVKeeFYf9``."""
        parsed = urlparse(self._raw_url)
        paper_ids = parse_qs(parsed.query).get("id")
        return paper_ids[0] if paper_ids else Path(unquote(parsed.path)).stem

    @property
    def pdf_url(self) -> HttpUrl:
        return self._url

    @property
    def title_is_fallback(self) -> bool:
        # Embedded PDF titles are often junk ("Microsoft Word - draft.docx") and
        # the URL stem is a guess; the title read from the first page wins.
        return True

    def _fallback_title(self) -> str:
        """Human-readable title from the URL stem when no better one exists."""
        words = re.sub(r"[^a-zA-Z0-9]+", " ", self._url_stem()).strip()
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

    def _safe_request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Issue a request that re-validates every URL against the SSRF guard.

        Auto-redirects are disabled and followed manually so each hop's target is
        DNS-resolved and checked (``assert_url_is_public``) before we connect to
        it — closing both the follow-a-3xx-to-internal-host hole and the
        validate-then-reconnect (TOCTOU) window a plain ``requests`` session
        leaves open. ``stream`` is forced on; callers read/close the response.
        """
        kwargs.pop("allow_redirects", None)
        current = url
        for _ in range(_MAX_REDIRECTS + 1):
            try:
                assert_url_is_public(current)
            except UnsafeUrlError as e:
                if is_temporary_dns_failure(e):
                    raise PaperFetchError(str(e)) from e
                raise PaperSourceError(str(e)) from e
            try:
                response = self._session.request(
                    method, current, allow_redirects=False, stream=True, **kwargs
                )
            except UnsafeUrlError as e:
                # The pinned adapter resolves again at connect time.
                if is_temporary_dns_failure(e):
                    raise PaperFetchError(str(e)) from e
                raise PaperSourceError(str(e)) from e
            if response.is_redirect and response.next is not None:
                location = response.headers.get("Location", "")
                response.close()
                if not location:
                    raise PaperSourceError(f"Redirect from '{current}' had no target.")
                current = urljoin(current, location)
                continue
            return response
        raise PaperSourceError(
            f"Too many redirects (> {_MAX_REDIRECTS}) fetching '{url}'."
        )

    def download_pdf(self, papers_dir: Path) -> Path:
        """Download the PDF in one streamed GET: an HTML content type or a body
        without PDF magic bytes is :class:`NotAPdfError` (a ``.pdf`` suffix is
        only a hint; servers return login pages from such URLs), a transport
        failure or 429/5xx is :class:`PaperFetchError`."""
        paper_dir = papers_dir / self.source_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = paper_dir / f"{self.source_id}.pdf"
        try:
            with self._safe_request(
                "GET", self._raw_url, timeout=_PDF_DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                status = response.status_code
                if status == 429 or status >= 500:
                    raise PaperFetchError(
                        f"'{self._raw_url}' is unavailable right now (HTTP {status})."
                    )
                if not response.ok:
                    raise PaperSourceError(
                        f"'{self._raw_url}' returned HTTP {status}, not a PDF."
                    )
                content_type = response.headers.get("Content-Type", "").lower()
                if "html" in content_type:
                    raise NotAPdfError(
                        f"URL '{self._raw_url}' returned HTML, not a PDF "
                        f"(Content-Type: {content_type})."
                    )
                save_pdf(
                    response.iter_content(chunk_size=CHUNK_BYTES),
                    pdf_path,
                    source=self._raw_url,
                    declared_length=response.headers.get("Content-Length"),
                )
        except NotPdfContentError as e:
            raise NotAPdfError(str(e)) from e
        except PdfTooLargeError as e:
            raise PaperSourceError(str(e)) from e
        except requests.RequestException as e:
            raise PaperFetchError(
                f"Failed to download PDF from '{self._raw_url}': {e}"
            ) from e
        self._discovered_title = self._extract_pdf_title(pdf_path)
        logger.info("Successfully downloaded PDF to: '%s'", pdf_path)
        return pdf_path

    @staticmethod
    def _extract_pdf_title(pdf_path: Path) -> str | None:
        """Best-effort title from the PDF's embedded document metadata."""
        try:
            import pymupdf

            with pymupdf.open(pdf_path) as doc:
                title = (doc.metadata or {}).get("title", "")
            title = " ".join(title.split())
            # Reject empty or junk titles (e.g. "untitled", a bare filename).
            if title and title.lower() not in {"untitled", pdf_path.stem.lower()}:
                return title
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Could not read PDF title from '%s': %s", pdf_path, e)
        return None

    def close(self) -> None:
        """Close the owned requests session (releases the connection pool)."""
        self._session.close()


def resolve_paper_source(
    source: str, *, arxiv_handler: ArxivHandler | None = None
) -> PaperSource:
    """Resolve a raw ``--source`` argument to a concrete :class:`PaperSource`.

    Accepts a bare arXiv ID, an arXiv (or ar5iv/alphaXiv) URL, or an arbitrary
    URL pointing at a paper PDF, with or without the ``https://`` scheme.
    Arbitrary non-PDF URLs are rejected later, at download time, via
    :class:`NotAPdfError`.
    """
    source = source.strip()
    if arxiv_id := normalize_arxiv_id(source):
        return ArxivSource(arxiv_id, handler=arxiv_handler)
    if _SCHEMELESS_URL.match(source):
        source = f"https://{source}"
    if source.lower().startswith(("http://", "https://")):
        try:
            if arxiv_id := arxiv_id_from_url(source):
                logger.info("Detected arXiv ID '%s' from URL.", arxiv_id)
                return ArxivSource(arxiv_id, handler=arxiv_handler)
            return PdfUrlSource(source)
        # A pydantic ValidationError (e.g. an invalid internationalized host) is
        # a ValueError, as is urlparse's "Invalid IPv6 URL".
        except ValueError as e:
            raise PaperSourceError(f"'{source}' is not a valid URL: {e}") from e
    raise PaperSourceError(
        f"Could not resolve '{source}' to an arXiv ID or a URL. Provide a valid "
        "arXiv ID (e.g. 2401.06066) or a URL to a paper PDF."
    )
