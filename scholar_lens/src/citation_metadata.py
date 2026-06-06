"""Reference metadata resolution from lenient, non-arXiv sources.

The citation stage used to hit the arXiv API up to three times per reference
(title search + content fetch + title/author fetch), which is both slow and the
direct cause of arXiv 429 storms. This module resolves a citation's title,
authors, abstract, and URL from APIs with generous rate limits — Crossref first
(no key, abstract often present), Semantic Scholar second — and exposes the
result so the summariser can work from an *abstract* instead of downloading the
full paper. arXiv stays available as a fallback in the summariser.

Every provider is best-effort and never raises: a lookup miss returns ``None`` so
the caller can fall back. Network access goes through a shared rate limiter.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from .logger import logger
from .rate_limiter import RateLimiter

_TIMEOUT = 15.0
_CROSSREF_URL = "https://api.crossref.org/works"
_SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Generous shared limiters (these APIs tolerate more than arXiv, but politeness
# still avoids 429s under the concurrent citation fan-out).
_CROSSREF_LIMITER = RateLimiter(rate=5.0, per=1.0, name="crossref")
_SEMANTIC_SCHOLAR_LIMITER = RateLimiter(rate=1.0, per=1.0, name="semantic-scholar")


@dataclass
class ReferenceMetadata:
    title: str
    authors: list[str]
    abstract: str | None = None
    url: str | None = None

    @property
    def author_str(self) -> str:
        return ", ".join(self.authors)

    @property
    def has_abstract(self) -> bool:
        return bool(self.abstract and self.abstract.strip())


class MetadataProvider(ABC):
    """Resolves a free-text reference title to structured metadata."""

    @abstractmethod
    def lookup(self, title: str) -> ReferenceMetadata | None: ...


class CrossrefProvider(MetadataProvider):
    """Crossref REST API — no key, broad coverage, abstracts when available."""

    def __init__(self, *, timeout: float = _TIMEOUT, mailto: str | None = None) -> None:
        self.timeout = timeout
        self.mailto = mailto

    def lookup(self, title: str) -> ReferenceMetadata | None:
        params = {
            "query.bibliographic": title,
            "rows": "1",
            "select": "title,author,abstract,URL",
        }
        if self.mailto:
            params["mailto"] = self.mailto
        try:
            _CROSSREF_LIMITER.acquire()
            resp = httpx.get(_CROSSREF_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])
        except (httpx.HTTPError, ValueError) as e:
            logger.debug("Crossref lookup failed for '%s': %s", title, e)
            return None
        if not items:
            return None
        item = items[0]
        found_title = " ".join(item.get("title") or []) or title
        authors = [
            " ".join(filter(None, [a.get("given"), a.get("family")]))
            for a in item.get("author", [])
        ]
        abstract = _strip_jats(item.get("abstract"))
        return ReferenceMetadata(
            title=found_title,
            authors=[a for a in authors if a],
            abstract=abstract,
            url=item.get("URL"),
        )


class SemanticScholarProvider(MetadataProvider):
    """Semantic Scholar graph search — good abstract coverage for CS/ML."""

    def __init__(self, *, timeout: float = _TIMEOUT) -> None:
        self.timeout = timeout

    def lookup(self, title: str) -> ReferenceMetadata | None:
        params = {"query": title, "limit": "1", "fields": "title,abstract,authors,url"}
        try:
            _SEMANTIC_SCHOLAR_LIMITER.acquire()
            resp = httpx.get(_SEMANTIC_SCHOLAR_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except (httpx.HTTPError, ValueError) as e:
            logger.debug("Semantic Scholar lookup failed for '%s': %s", title, e)
            return None
        if not data:
            return None
        item = data[0]
        return ReferenceMetadata(
            title=item.get("title") or title,
            authors=[
                a.get("name", "") for a in item.get("authors", []) if a.get("name")
            ],
            abstract=item.get("abstract"),
            url=item.get("url"),
        )


class ChainedMetadataResolver:
    """Tries each provider in order, returning the first usable hit.

    "Usable" prefers a result *with* an abstract (so the summariser can work from
    it); if no provider yields an abstract, the first non-empty hit is returned so
    at least title/authors/url are available.
    """

    def __init__(self, providers: list[MetadataProvider] | None = None) -> None:
        self.providers = (
            providers
            if providers is not None
            else [CrossrefProvider(), SemanticScholarProvider()]
        )

    def resolve(self, title: str) -> ReferenceMetadata | None:
        first_hit: ReferenceMetadata | None = None
        for provider in self.providers:
            try:
                result = provider.lookup(title)
            except Exception as e:  # noqa: BLE001 - resolution must never raise
                logger.debug("Provider %s raised for '%s': %s", provider, title, e)
                continue
            if result is None:
                continue
            if result.has_abstract:
                return result
            first_hit = first_hit or result
        return first_hit


def _strip_jats(abstract: str | None) -> str | None:
    """Crossref abstracts are JATS XML; strip the tags to plain text."""
    if not abstract:
        return None
    text = re.sub(r"<[^>]+>", " ", abstract)
    text = " ".join(text.split())
    return text or None
