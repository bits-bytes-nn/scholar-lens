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

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from .logger import logger
from .rate_limiter import RateLimiter

_TIMEOUT = 15.0
_CROSSREF_URL = "https://api.crossref.org/works"
_SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Generous shared limiters (these APIs tolerate more than arXiv, but politeness
# still avoids 429s under the concurrent citation fan-out).
_CROSSREF_LIMITER = RateLimiter(rate=5.0, per=1.0, name="crossref")
_SEMANTIC_SCHOLAR_LIMITER = RateLimiter(rate=1.0, per=1.0, name="semantic-scholar")

# Crossref work types whose URL we trust as the primary publication of a cited
# paper. Book chapters / encyclopedia entries / monographs are EXCLUDED: for an
# AI/ML reference they are almost always a different work that merely cites the
# original, so their URL would mis-attribute the citation.
_CROSSREF_PRIMARY_TYPES = frozenset(
    {
        "journal-article",
        "proceedings-article",
        "posted-content",  # preprints (arXiv, bioRxiv, …)
    }
)


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
            "select": "title,author,abstract,URL,type",
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
        # Crossref's top fuzzy hit for an AI/ML paper is frequently a book
        # chapter / encyclopedia entry that merely CITES the work (e.g. a Springer
        # chapter matching "Attention Is All You Need"). Linking to it
        # mis-attributes the citation, so only trust the URL for primary
        # article-like types; otherwise keep title/abstract but drop the link.
        url = item.get("URL") if item.get("type") in _CROSSREF_PRIMARY_TYPES else None
        return ReferenceMetadata(
            title=found_title,
            authors=[a for a in authors if a],
            abstract=abstract,
            url=url,
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
        hits: list[ReferenceMetadata] = []
        for provider in self.providers:
            try:
                result = provider.lookup(title)
            except Exception as e:  # noqa: BLE001 - resolution must never raise
                logger.debug("Provider %s raised for '%s': %s", provider, title, e)
                continue
            if result is not None:
                hits.append(result)
        if not hits:
            return None

        # Prefer a single coherent hit that has BOTH an abstract (for
        # summarising) and its OWN URL (for linking); then any hit with an
        # abstract; then the first hit. We never graft a URL from a different
        # provider's result onto another's metadata — that could surface a URL
        # for a different paper. Final link trust is still gated by title
        # similarity in the summariser.
        for h in hits:
            if h.has_abstract and h.url:
                return h
        return next((h for h in hits if h.has_abstract), hits[0])


def _strip_jats(abstract: str | None) -> str | None:
    """Crossref abstracts are JATS XML; extract plain text.

    Parse with an XML/HTML parser rather than a ``<[^>]+>`` regex: many real
    Crossref records contain raw inequality signs in prose ("for all x < y and
    a > b"), which a greedy tag-strip would treat as a tag and delete, silently
    dropping the clause between them. The parser also decodes entities
    (``&amp;`` -> ``&``) so the summarizer sees clean text.
    """
    if not abstract:
        return None
    text = BeautifulSoup(abstract, "html.parser").get_text(separator=" ")
    text = " ".join(text.split())
    return text or None
