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

import os
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import httpx
from bs4 import BeautifulSoup

from .constants import EnvVars
from .logger import logger
from .rate_limiter import RateLimiter

_TIMEOUT = 15.0
_CROSSREF_URL = "https://api.crossref.org/works"
_SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Shared limiters sized to each API's published limits: Crossref's public pool
# allows 1 request per second, one at a time (x-rate-limit-limit: 1,
# x-concurrency-limit: 1); Semantic Scholar's unauthenticated pool is shared
# and throttles bursts.
_CROSSREF_LIMITER = RateLimiter(rate=1.0, per=1.0, name="crossref")
_CROSSREF_CONCURRENCY = threading.Lock()
_SEMANTIC_SCHOLAR_LIMITER = RateLimiter(rate=1.0, per=1.0, name="semantic-scholar")
# A 429 without Retry-After waits this long before the single retry.
_DEFAULT_RETRY_AFTER = 5.0
_MAX_RETRY_AFTER = 60.0

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


# Minimum similarity between a queried and a returned title before the record
# is accepted. Lower values (or substring containment) accept "LoRA+" for
# "LoRA" or a survey for the paper it surveys.
TITLE_MATCH_THRESHOLD: float = 0.9


def titles_match(queried: str | None, resolved: str | None) -> bool:
    """Whether two titles name the same paper (case/punctuation-insensitive)."""
    if not queried or not resolved:
        return False

    def norm(value: str) -> str:
        return " ".join(re.sub(r"[^\w ]+", " ", value.casefold()).split())

    q, r = norm(queried), norm(resolved)
    if not (q and r) or SequenceMatcher(None, q, r).ratio() < TITLE_MATCH_THRESHOLD:
        return False
    # Version-like tokens tell sibling papers apart even when the rest of the
    # title is identical (GPT-4 vs GPT-4o, Qwen2 vs Qwen2.5, LoRA vs LoRA+).
    return _distinguishing_tokens(queried) == _distinguishing_tokens(resolved)


_TOKEN = re.compile(r"[\w.+]+")


def _distinguishing_tokens(title: str) -> set[str]:
    tokens = {t.strip(".").casefold() for t in _TOKEN.findall(title)}
    return {t for t in tokens if any(ch.isdigit() for ch in t) or "+" in t}


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


def _retry_after(response: httpx.Response) -> float:
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        seconds = _DEFAULT_RETRY_AFTER
    return min(max(seconds, 0.0), _MAX_RETRY_AFTER)


def _get_json(
    url: str,
    *,
    params: dict[str, str],
    limiter: RateLimiter,
    timeout: float,
    headers: dict[str, str] | None = None,
    serial: threading.Lock | None = None,
) -> Any:
    """GET JSON at the API's pace. A 429 pauses every caller of the API for
    its ``Retry-After`` and is retried once; any other failure raises."""
    for attempt in range(2):
        limiter.acquire()
        if serial is not None:
            with serial:
                response = httpx.get(
                    url, params=params, headers=headers, timeout=timeout
                )
        else:
            response = httpx.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code == 429 and attempt == 0:
            limiter.penalize(_retry_after(response))
            continue
        response.raise_for_status()
        return response.json()
    raise AssertionError("unreachable")  # pragma: no cover


class CrossrefProvider(MetadataProvider):
    """Crossref REST API — no key, broad coverage, abstracts when available."""

    def __init__(self, *, timeout: float = _TIMEOUT, mailto: str | None = None) -> None:
        self.timeout = timeout
        self.mailto = mailto or os.environ.get(EnvVars.CROSSREF_MAILTO.value)

    def lookup(self, title: str) -> ReferenceMetadata | None:
        params = {
            "query.bibliographic": title,
            "rows": "1",
            "select": "title,author,abstract,URL,type",
        }
        if self.mailto:
            params["mailto"] = self.mailto
        try:
            data = _get_json(
                _CROSSREF_URL,
                params=params,
                limiter=_CROSSREF_LIMITER,
                timeout=self.timeout,
                serial=_CROSSREF_CONCURRENCY,
            )
            items = data.get("message", {}).get("items", [])
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("Crossref lookup failed for '%s': %s", title, e)
            return None
        if not items:
            return None
        item = items[0]
        found_title = " ".join(item.get("title") or []).strip()
        if not found_title:
            return None  # a record without a title can't pass the title gate honestly
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

    def __init__(
        self, *, timeout: float = _TIMEOUT, api_key: str | None = None
    ) -> None:
        self.timeout = timeout
        self.api_key = api_key or os.environ.get(EnvVars.SEMANTIC_SCHOLAR_API_KEY.value)

    def lookup(self, title: str) -> ReferenceMetadata | None:
        params = {"query": title, "limit": "1", "fields": "title,abstract,authors,url"}
        try:
            data = _get_json(
                _SEMANTIC_SCHOLAR_URL,
                params=params,
                limiter=_SEMANTIC_SCHOLAR_LIMITER,
                timeout=self.timeout,
                headers={"x-api-key": self.api_key} if self.api_key else None,
            ).get("data", [])
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("Semantic Scholar lookup failed for '%s': %s", title, e)
            return None
        if not data:
            return None
        item = data[0]
        if not (item.get("title") or "").strip():
            return None
        return ReferenceMetadata(
            title=item["title"],
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
        """Return the best record whose title matches ``title``.

        Stops at the first matching record that has both an abstract and its
        own URL (later providers are not queried); otherwise prefers a matching
        record with an abstract, then any matching record. Records for other
        papers are discarded, and URLs are never grafted across records.
        """
        hits: list[ReferenceMetadata] = []
        for provider in self.providers:
            try:
                result = provider.lookup(title)
            except Exception as e:  # noqa: BLE001 - resolution must never raise
                logger.warning("Provider %s raised for '%s': %s", provider, title, e)
                continue
            if result is None or not titles_match(title, result.title):
                continue
            if result.has_abstract and result.url:
                return result
            hits.append(result)
        if not hits:
            return None
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
