"""Reference metadata resolution from lenient, non-arXiv sources.

The citation stage used to hit the arXiv API up to three times per reference
(title search + content fetch + title/author fetch), which is both slow and the
direct cause of arXiv 429 storms. This module resolves a citation's title,
authors, abstract, and URL from APIs with generous rate limits — Crossref first
(no key, abstract often present), OpenAlex second (optional key, broad
abstract coverage), Semantic Scholar last and only with an API key (its keyless pool is
shared and mostly answers 429) — and exposes the result so the summariser can work from an *abstract* instead of downloading the
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

from .constants import EnvVars, normalize_arxiv_id
from .logger import logger
from .rate_limiter import RateLimiter

_TIMEOUT = 15.0
_CROSSREF_URL = "https://api.crossref.org/works"
_OPENALEX_URL = "https://api.openalex.org/works"
_SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Shared limiters sized to each API's published limits: Crossref's public pool
# allows 1 request per second, one at a time (x-rate-limit-limit: 1,
# x-concurrency-limit: 1); OpenAlex allows 10 per second (kept well below);
# Semantic Scholar's pool throttles bursts.
_CROSSREF_LIMITER = RateLimiter(rate=1.0, per=1.0, name="crossref")
_CROSSREF_CONCURRENCY = threading.Lock()
_OPENALEX_LIMITER = RateLimiter(rate=2.0, per=1.0, name="openalex")
# Search results scanned for a matching record (one request either way).
_SEARCH_CANDIDATES = 5
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


def _norm_title(value: str) -> str:
    return " ".join(re.sub(r"[^\w ]+", " ", value.casefold()).split())


def title_similarity(queried: str, resolved: str) -> float:
    """Similarity of two titles (case/punctuation-insensitive); 1.0 when
    they are the same title."""
    return SequenceMatcher(None, _norm_title(queried), _norm_title(resolved)).ratio()


def titles_match(queried: str | None, resolved: str | None) -> bool:
    """Whether two titles name the same paper (case/punctuation-insensitive)."""
    if not queried or not resolved:
        return False

    q, r = _norm_title(queried), _norm_title(resolved)
    if not (q and r) or not _similar(q, r):
        return False
    # Version-like tokens tell sibling papers apart even when the rest of the
    # title is identical (GPT-4 vs GPT-4o, Qwen2 vs Qwen2.5, LoRA vs LoRA+).
    if _distinguishing_tokens(queried) != _distinguishing_tokens(resolved):
        return False
    # A swapped word names another paper even at high similarity ("...Large
    # Language Models" vs "...Protein Language Models"); an added or dropped
    # word alone is left to the similarity ratio.
    # Spelling variants are not swaps: hyphenation ("Few-Shot" / "Few Shot"),
    # British spelling ("Visualising") or a plural.
    if _squash(q) == _squash(r):
        return True
    q_only = _title_words(queried) - _title_words(resolved)
    r_only = _title_words(resolved) - _title_words(queried)
    swapped = bool(r_only) and any(
        not any(_spelling_variant(a, b) for b in r_only) for a in q_only
    )
    return not swapped


def _similar(q: str, r: str) -> bool:
    """``title_similarity`` reaches the threshold; the cheap upper bounds of
    the ratio rule most pairs out first (pools compare hundreds of titles)."""
    matcher = SequenceMatcher(None, q, r)
    return (
        matcher.real_quick_ratio() >= TITLE_MATCH_THRESHOLD
        and matcher.quick_ratio() >= TITLE_MATCH_THRESHOLD
        and matcher.ratio() >= TITLE_MATCH_THRESHOLD
    )


# Two differing title words this similar, with the same start, are one word
# spelled two ways; a different start is a prefix ("Unsupervised", "Decoder").
_VARIANT_RATIO = 0.8


def _spelling_variant(a: str, b: str) -> bool:
    return a[:3] == b[:3] and SequenceMatcher(None, a, b).ratio() >= _VARIANT_RATIO


_DASHES = re.compile(r"[-\u2010-\u2015\u2212]")


def _squash(title: str) -> str:
    return re.sub(r"[\W_]+", "", title)


def _title_words(title: str) -> set[str]:
    return {w.removesuffix("s") for w in _content_words(_DASHES.sub("", title))}


_TOKEN = re.compile(r"[\w.+]+")


def _distinguishing_tokens(title: str) -> set[str]:
    tokens = {t.strip(".").casefold() for t in _TOKEN.findall(title)}
    return {t for t in tokens if any(ch.isdigit() for ch in t) or "+" in t}


_WORD = re.compile(r"[^\W_]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from how in into is it of on or our over the "
    "to towards via what when where which why with without you your".split()
)


def _content_words(text: str) -> set[str]:
    words = set(_WORD.findall(text.casefold()))
    return {w for w in words if w not in _STOPWORDS and not w.isdigit()}


def abstract_fits_title(title: str, abstract: str | None) -> bool:
    """Whether ``abstract`` plausibly belongs to ``title``: it shares at least
    one of the title's content words (a missing abstract contradicts
    nothing). Index records sometimes pair the right title with another
    work's abstract; such an abstract shares none."""
    words = _content_words(title)
    return not (words and abstract) or bool(words & _content_words(abstract))


# arXiv DOIs (10.48550/arXiv.<id>), bare or as a doi.org URL.
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(\S+)$", re.IGNORECASE)


def arxiv_id_from_doi(doi: str | None) -> str | None:
    """The arXiv id behind an arXiv DOI, or None for any other value."""
    match = _ARXIV_DOI.search((doi or "").strip())
    return normalize_arxiv_id(match.group(1)) if match else None


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


class QuotaExhaustedError(RuntimeError):
    """The API asked for a wait longer than ``_MAX_RETRY_AFTER`` (a spent daily
    quota): waiting is pointless, so the provider is skipped for the run."""


def _retry_after(response: httpx.Response) -> float:
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        seconds = _DEFAULT_RETRY_AFTER
    seconds = max(seconds, 0.0)
    if seconds > _MAX_RETRY_AFTER:
        raise QuotaExhaustedError(
            f"{response.url.host} asks to retry in {seconds:.0f}s"
        )
    return seconds


def _get_json(
    url: str,
    *,
    params: dict[str, str],
    limiter: RateLimiter,
    timeout: float,
    headers: dict[str, str] | None = None,
    serial: threading.Lock | None = None,
    json_body: Any = None,
) -> Any:
    """GET JSON at the API's pace (POST ``json_body`` when given). A 429
    pauses every caller of the API for its ``Retry-After`` and is retried
    once; a wait beyond ``_MAX_RETRY_AFTER`` raises ``QuotaExhaustedError``,
    any other failure ``httpx.HTTPError``."""

    def send() -> httpx.Response:
        if json_body is not None:
            return httpx.post(
                url, params=params, json=json_body, headers=headers, timeout=timeout
            )
        return httpx.get(url, params=params, headers=headers, timeout=timeout)

    for attempt in range(2):
        limiter.acquire()
        if serial is not None:
            with serial:
                response = send()
        else:
            response = send()
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
            "rows": str(_SEARCH_CANDIDATES),
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
        # The top fuzzy hit is often another paper (a survey of ResNet ranks
        # above ResNet itself); take the matching record whose title is
        # closest to the queried one, the earliest on a tie.
        matches = [
            (" ".join(item.get("title") or []).strip(), item)
            for item in items
            if titles_match(title, " ".join(item.get("title") or []))
        ]
        if not matches:
            return None
        found_title, item = max(matches, key=lambda m: title_similarity(title, m[0]))
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


def _inverted_index_text(index: Any) -> str | None:
    """OpenAlex ships abstracts as ``{word: [positions]}``; put the words back
    in position order."""
    if not isinstance(index, dict):
        return None
    placed = sorted(
        (pos, word)
        for word, positions in index.items()
        if isinstance(positions, list)
        for pos in positions
        if isinstance(pos, int) and not isinstance(pos, bool)
    )
    text = " ".join(word for _, word in placed)
    return text or None


class OpenAlexProvider(MetadataProvider):
    """OpenAlex works search — broad coverage, abstracts for most papers.
    Works without a key on a small daily budget per IP; an optional API key
    (sent as a Bearer token, never in the URL) raises it. ``mailto`` (the
    Crossref contact) identifies the caller."""

    def __init__(
        self,
        *,
        timeout: float = _TIMEOUT,
        mailto: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.mailto = mailto or os.environ.get(EnvVars.CROSSREF_MAILTO.value)
        self.api_key = api_key or os.environ.get(EnvVars.OPENALEX_API_KEY.value)

    def lookup(self, title: str) -> ReferenceMetadata | None:
        # Commas and colons are filter syntax in OpenAlex's search parameter.
        params = {
            "search": " ".join(re.sub(r"[,:]", " ", title).split()),
            "per-page": str(_SEARCH_CANDIDATES),
            "select": "title,authorships,abstract_inverted_index,doi,id",
        }
        if self.mailto:
            params["mailto"] = self.mailto
        try:
            results = _get_json(
                _OPENALEX_URL,
                params=params,
                limiter=_OPENALEX_LIMITER,
                timeout=self.timeout,
                headers=(
                    {"Authorization": f"Bearer {self.api_key}"}
                    if self.api_key
                    else None
                ),
            ).get("results", [])
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("OpenAlex lookup failed for '%s': %s", title, e)
            return None
        # The top hit is often a sibling paper ("QA-LoRA" for "LoRA") or a
        # record pairing the right title with another work's abstract and DOI;
        # take the first whose title matches and whose abstract fits it.
        for item in results:
            found_title = (item.get("title") or "").strip()
            abstract = _inverted_index_text(item.get("abstract_inverted_index"))
            if not titles_match(title, found_title) or not abstract_fits_title(
                title, abstract
            ):
                continue
            url = item.get("doi") or item.get("id")
            if arxiv_id_from_doi(url):
                # OpenAlex's abstracts for arXiv records are unreliable (one
                # paper's DOI carried another's abstract); arXiv supplies them.
                abstract = None
            authors = [
                (a.get("author") or {}).get("display_name") or ""
                for a in item.get("authorships") or []
            ]
            return ReferenceMetadata(
                title=found_title,
                # Drop placeholder names without letters (e.g. ":").
                authors=[a for a in authors if re.search(r"[^\W\d_]", a)],
                abstract=abstract,
                url=url,
            )
        return None


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
        if providers is None:
            providers = [CrossrefProvider(), OpenAlexProvider()]
            # Without a key every call (and every 429 pause) is wasted time.
            if os.environ.get(EnvVars.SEMANTIC_SCHOLAR_API_KEY.value):
                providers.append(SemanticScholarProvider())
        self.providers = providers
        self._exhausted: set[int] = set()

    def resolve(self, title: str) -> ReferenceMetadata | None:
        """Return the best record whose title matches ``title``.

        Stops at the first matching record that has both an abstract and its
        own URL (later providers are not queried); otherwise prefers a matching
        record with an abstract, then one with a URL (an arXiv DOI lets the
        caller fetch the paper by id), then any matching record. Records for
        other papers are discarded, an abstract that shares no content word
        with the title is dropped as another work's, and URLs are never
        grafted across records.
        """
        hits: list[ReferenceMetadata] = []
        for provider in self.providers:
            if id(provider) in self._exhausted:
                continue
            try:
                result = provider.lookup(title)
            except QuotaExhaustedError as e:
                self._exhausted.add(id(provider))
                logger.warning("Skipping %s for the rest of the run: %s", provider, e)
                continue
            except Exception as e:  # noqa: BLE001 - resolution must never raise
                logger.warning("Provider %s raised for '%s': %s", provider, title, e)
                continue
            if result is None or not titles_match(title, result.title):
                continue
            if not abstract_fits_title(title, result.abstract):
                result.abstract = None  # another work's abstract
            if result.has_abstract and result.url:
                return result
            hits.append(result)
        if not hits:
            return None
        return next(
            (h for h in hits if h.has_abstract),
            next((h for h in hits if h.url), hits[0]),
        )


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
