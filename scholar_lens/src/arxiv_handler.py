import re
from datetime import datetime
from pathlib import Path

import arxiv
import requests
import tenacity
from pydantic import BaseModel, Field, HttpUrl, ValidationInfo, field_validator

from .citation_metadata import title_similarity, titles_match
from .constants import normalize_arxiv_id
from .logger import logger
from .pdf_download import CHUNK_BYTES, save_pdf
from .rate_limiter import RateLimiter

# The only retry layer for arXiv API calls (the arxiv client itself makes one
# attempt): jittered backoff so a momentary 429 does not abort a whole job.
_ARXIV_FETCH_MAX_ATTEMPTS: int = 4
_ARXIV_FETCH_INITIAL_WAIT: int = 5
_ARXIV_FETCH_MAX_WAIT: int = 60
# Fallback penalty when a 429 carries no Retry-After header (arXiv usually omits
# it). Named so the assumption is visible alongside the other retry constants.
_ARXIV_429_FALLBACK_RETRY_SECONDS: float = 30.0
# (connect, read) timeouts for arXiv API and PDF requests; without them a stalled
# connection would hang a Batch job until its 3-hour limit.
_ARXIV_TIMEOUT: tuple[float, float] = (10.0, 60.0)


class _TimeoutAdapter(requests.adapters.HTTPAdapter):
    """Applies a default timeout to requests made by the arxiv client."""

    def send(self, request, **kwargs):  # type: ignore[no-untyped-def,override]
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = _ARXIV_TIMEOUT
        return super().send(request, **kwargs)


# A single process-wide limiter shared by ALL ArxivHandler instances, so the
# concurrent citation stage cannot fan out into an arXiv 429 storm. arXiv asks
# for ~1 request / 3s; we honour that globally.
_ARXIV_RATE_LIMITER = RateLimiter(rate=1.0, per=3.0, name="arxiv")


def _retry_after_seconds(error: Exception) -> float:
    """Best-effort extraction of a Retry-After hint from an arXiv/HTTP error."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0
    # arXiv 429s rarely include Retry-After; apply a conservative default.
    if "429" in str(error):
        return _ARXIV_429_FALLBACK_RETRY_SECONDS
    return 0.0


class ArxivPaperError(Exception):
    pass


class ArxivDownloadError(ArxivPaperError):
    pass


class ArxivNotFoundError(ArxivPaperError):
    pass


class ArxivMetadata(BaseModel):
    arxiv_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    published: datetime
    updated: datetime
    abstract: str | None = None
    journal_ref: str | None = None
    primary_category: str | None = None
    categories: list[str] = Field(default_factory=list)
    doi: str | None = None
    pdf_url: HttpUrl | None = None

    def __repr__(self) -> str:
        return f"ArxivMetadata(arxiv_id='{self.arxiv_id}', title='{self.title}')"

    @field_validator("abstract", mode="before")
    @classmethod
    def clean_abstract(cls, v: str | None) -> str | None:
        if v:
            return " ".join(v.strip().split())
        return v

    @field_validator("doi", mode="before")
    @classmethod
    def generate_and_validate_doi(
        cls, v: str | None, info: ValidationInfo
    ) -> str | None:
        if v:
            return v

        arxiv_id = info.data.get("arxiv_id")
        # Only synthesise an arXiv DOI for genuine arXiv identifiers; generic
        # PDF-URL sources use a derived slug for which this is meaningless.
        # The arXiv DOI is deterministic (10.48550/arXiv.<id>), so we construct
        # it directly. A previous version made a doi.org round-trip "to validate"
        # on every metadata construction — wasteful and a 429/hang risk inside a
        # pydantic validator. The format is stable; no network call is needed.
        if normalize_arxiv_id(arxiv_id):
            clean_arxiv_id = re.sub(r"v\d+$", "", str(arxiv_id))
            return f"10.48550/arXiv.{clean_arxiv_id}"

        return None

    @field_validator("updated", mode="before")
    @classmethod
    def set_updated_if_missing(
        cls, v: datetime | None, info: ValidationInfo
    ) -> datetime:
        if v is not None:
            return v
        published = info.data.get("published")
        if published is not None:
            return published
        raise ValueError("Both 'updated' and 'published' are None")


class ArxivHandler:
    DEFAULT_PAGE_SIZE: int = 100

    def __init__(self) -> None:
        # Pacing is the process-wide _ARXIV_RATE_LIMITER's job; the client's
        # own per-instance delay would only add to it.
        self.client = arxiv.Client(
            page_size=self.DEFAULT_PAGE_SIZE, delay_seconds=0, num_retries=0
        )
        session = getattr(self.client, "_session", None)
        if isinstance(session, requests.Session):
            adapter = _TimeoutAdapter()
            session.mount("https://", adapter)
            session.mount("http://", adapter)

    def download_paper(
        self, arxiv_id: str, papers_dir: Path, *, folder: str | None = None
    ) -> Path:
        """Download the PDF into ``papers_dir/<folder>`` (default: the resolved,
        versioned id)."""
        try:
            paper = self._fetch_single_paper(arxiv_id)
            safe_id = folder or re.sub(r"[./]", "_", paper.get_short_id())
            paper_dir = papers_dir / safe_id
            paper_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = paper_dir / f"{safe_id}.pdf"
            self._download(str(paper.pdf_url), pdf_path)
            logger.info("Successfully downloaded PDF to: '%s'", pdf_path)
            return pdf_path
        except ArxivNotFoundError:
            raise
        except Exception as e:
            raise ArxivDownloadError(
                f"Failed to download paper '{arxiv_id}': {str(e)}"
            ) from e

    @staticmethod
    def _download(url: str, pdf_path: Path) -> None:
        """Stream the PDF with timeouts and a size cap (``urlretrieve``, used
        by the arxiv library, has neither)."""
        with requests.get(url, stream=True, timeout=_ARXIV_TIMEOUT) as response:
            response.raise_for_status()
            save_pdf(
                response.iter_content(chunk_size=CHUNK_BYTES),
                pdf_path,
                source=url,
                declared_length=response.headers.get("Content-Length"),
            )

    def fetch_metadata(self, arxiv_id: str) -> ArxivMetadata:
        try:
            paper = self._fetch_single_paper(arxiv_id)
            metadata = self._paper_to_metadata(paper)
            logger.info("Successfully fetched metadata for arXiv ID: '%s'", arxiv_id)
            return metadata
        except (ArxivNotFoundError, ArxivPaperError):
            raise
        except Exception as e:
            logger.error(
                "Failed to process metadata for arXiv ID: '%s': %s", arxiv_id, str(e)
            )
            raise ArxivPaperError(f"Could not process metadata for '{arxiv_id}'") from e

    def _fetch_single_paper(self, arxiv_id: str) -> arxiv.Result:
        # Retry transient API errors (HTTP 429 etc.) with jittered backoff, but
        # never retry a genuine "not found" — that is terminal.
        retryer = tenacity.Retrying(
            wait=tenacity.wait_exponential_jitter(
                initial=_ARXIV_FETCH_INITIAL_WAIT, max=_ARXIV_FETCH_MAX_WAIT
            ),
            stop=tenacity.stop_after_attempt(_ARXIV_FETCH_MAX_ATTEMPTS),
            # Retry transient API errors but NOT a genuine "not found"
            # (ArxivNotFoundError subclasses ArxivPaperError, so exclude it).
            retry=tenacity.retry_if_exception(
                lambda e: isinstance(e, ArxivPaperError)
                and not isinstance(e, ArxivNotFoundError)
            ),
            before_sleep=lambda s: logger.warning(
                "arXiv fetch for '%s' failed (attempt %d); retrying...",
                arxiv_id,
                s.attempt_number,
            ),
            reraise=True,
        )
        return retryer(self._fetch_single_paper_once, arxiv_id)

    def _fetch_single_paper_once(self, arxiv_id: str) -> arxiv.Result:
        _ARXIV_RATE_LIMITER.acquire()
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            return next(self.client.results(search))
        except StopIteration as e:
            raise ArxivNotFoundError(
                f"Paper not found for arXiv ID: '{arxiv_id}'"
            ) from e
        except Exception as e:
            _ARXIV_RATE_LIMITER.penalize(_retry_after_seconds(e))
            raise ArxivPaperError(
                f"An unexpected error occurred while fetching '{arxiv_id}': {e}"
            ) from e

    @staticmethod
    def _paper_to_metadata(paper: arxiv.Result) -> ArxivMetadata:
        author_names = [author.name.strip() for author in paper.authors if author.name]

        return ArxivMetadata(
            arxiv_id=paper.get_short_id(),
            title=paper.title,
            authors=author_names,
            published=paper.published,
            updated=paper.updated,
            abstract=paper.summary,
            journal_ref=paper.journal_ref,
            primary_category=paper.primary_category,
            categories=paper.categories or [],
            doi=paper.doi,
            pdf_url=HttpUrl(paper.pdf_url) if paper.pdf_url else None,
        )

    def search_by_title(self, title: str, max_results: int = 10) -> str | None:
        queries_to_try = [f'ti:"{title}"']
        if "-" in title:
            queries_to_try.append(f'ti:"{title.replace("-", " ")}"')

        for query in queries_to_try:
            try:
                _ARXIV_RATE_LIMITER.acquire()
                search = arxiv.Search(query=query, max_results=max_results)
                # The closest matching title, not the first: a survey or a
                # "Revisited" follow-up may be listed before the paper itself.
                best: tuple[float, str] | None = None
                for paper in self.client.results(search):
                    if titles_match(title, paper.title):
                        similarity = title_similarity(title, paper.title)
                        if similarity == 1.0:
                            return paper.get_short_id()
                        if best is None or similarity > best[0]:
                            best = (similarity, paper.get_short_id())
                if best is not None:
                    return best[1]
            except Exception as e:
                _ARXIV_RATE_LIMITER.penalize(_retry_after_seconds(e))
                logger.warning("Search query '%s' failed: %s", query, e)
                continue

        logger.warning("No confident title match found for: '%s'", title)
        return None
