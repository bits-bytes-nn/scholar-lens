"""Web research primitives for the technical-guide generator.

Gathers source material for a tech guide/tutorial from a user-supplied URL
list: it fetches each page, optionally discovers in-scope sub-URLs (same-site
links), extracts readable text and image references, and can augment the corpus
with web-search results via a pluggable :class:`WebSearchProvider`.

The search provider is an abstraction so deployments can choose their backend
(Brave Search API by default, or none) without changing the agent.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import socket
import time
from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from itertools import pairwise
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag, UnicodeDammit

from ..adapters.http.pinned_transport import PinnedHTTPTransport
from ..adapters.http.rate_limit import ARXIV_RATE_LIMITER, arxiv_retry_after_seconds
from ..adapters.http.url_guard import (
    UnsafeUrlError,
    assert_url_is_public,
    is_temporary_dns_failure,
    is_url_public,
)
from ..domain.markdown.lint import FIGURE_MARKER, close_open_fence
from ..domain.markdown.math import segments
from .constants import EnvVars, arxiv_id_from_url, arxiv_paper_key, is_arxiv_host
from .logger import logger
from .parser import (
    MIN_PARSED_TEXT_CHARS,
    _readable_chars,
    _real_text,
    figure_caption,
)
from .prompt_safety import neutralize_prompt_tags
from .prompts.tech_guide import (
    OFFICIAL_LABEL,
    OLDER_DOCS_LABEL,
    REFERENCE_LABEL,
    THIRD_PARTY_LABEL,
)
from .source_set import (
    OFFICIAL_AUTHORITIES,
    SHARED_HOSTS,
    OfficialDocs,
    SourceSet,
    docs_version,
    is_official,
    iso_date,
    normalize_url,
)
from .utils import extract_text_from_html, parse_lines
from .utils.helpers import is_page_furniture

# Re-exported: the URL rules moved to :mod:`source_set`.
__all__ = ["SHARED_HOSTS", "OfficialDocs", "docs_version", "normalize_url"]

DEFAULT_TIMEOUT: int = 30
DEFAULT_MAX_SUBPAGES: int = 8
DEFAULT_MAX_IMAGES_PER_PAGE: int = 12
# Pages larger than this are skipped (a docs page is far smaller); the cap is
# enforced while streaming so an oversized body is never fully downloaded.
MAX_PAGE_BYTES: int = 5 * 1024 * 1024
# A <meta refresh> is followed only on a stub page this small (a real docs page
# with an incidental refresh tag keeps its own content), and at most this often.
_META_REFRESH_MAX_BYTES = 16 * 1024
MAX_META_REFRESH_HOPS = 3
# A page joins a prompt's sources only with more room than this left.
_MIN_SOURCE_ROOM = 2_000
_META_REFRESH_URL = re.compile(
    r"^\s*\d+(?:\.\d*)?\s*[;,]\s*(?:url\s*=\s*)?['\"]?([^'\"\s]+)", re.I
)
# Official reference pages fetched to check a guide's flags and metric names
# (engine arguments, CLI, configuration, metrics): at most this many searches
# and pages per guide.
MAX_REFERENCE_SEARCHES = 3
MAX_REFERENCE_PAGES = 6
_REFERENCE_LINK = re.compile(
    r"(?:^|[/_.-])(?:cli|args?|arguments?|engine[_-]?args|config(?:uration)?|"
    r"metrics?|optimi[sz]ation|reference|parameters?|options?|settings?|env(?:ironment)?)"
    r"(?:$|[/_.-])",
    re.I,
)
_ARGUMENT_WORDS = frozenset(
    {"cli", "arg", "args", "arguments", "options", "parameters"}
)
_METRIC_WORDS = frozenset({"metric", "metrics"})
# Prometheus naming-convention suffixes that mark a metric name
# ("..._requests_total"); guide_authority's metric pattern uses the same list.
METRIC_SUFFIXES = ("total", "seconds", "bytes", "count", "bucket", "sum")


def _path_words(url: str) -> set[str]:
    return {w for w in re.split(r"[/_.-]+", urlparse(url).path.lower()) if w}


def _reference_words(identifiers: list[str]) -> set[str]:
    """Path words of a reference page that settles ``identifiers``: the kind
    of page (arguments for flags, metrics for metric names) and the
    identifiers' own words (``--gpu-memory-utilization`` → "memory")."""
    words: set[str] = set()
    for identifier in identifiers:
        is_metric = ":" in identifier or identifier.endswith(
            tuple(f"_{suffix}" for suffix in METRIC_SUFFIXES)
        )
        words |= _METRIC_WORDS if is_metric else _ARGUMENT_WORDS
        words |= {w for w in re.split(r"[-_:=\s]+", identifier.lower()) if len(w) > 2}
    return words


_BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
# Tavily's answers for a spent plan quota or key: no later query succeeds.
# A rejected key; Tavily also answers a spent plan or credit quota with
# 432 or 433.
_AUTH_FAILED_STATUSES = frozenset({401, 403})
_TAVILY_EXHAUSTED_STATUSES = _AUTH_FAILED_STATUSES | {432, 433}


def arxiv_paper_urls(url: str) -> tuple[str, str] | None:
    """For a link to an arXiv paper (abs, pdf or html, on arXiv or a mirror),
    the paper's HTML rendering (the full text; the abstract page holds only
    the abstract and a PDF is not read) and its abstract page, the fallback;
    None for any other URL."""
    arxiv_id = arxiv_id_from_url(url)
    if arxiv_id is None:
        return None
    return f"https://arxiv.org/html/{arxiv_id}", f"https://arxiv.org/abs/{arxiv_id}"


def page_key(url: str) -> str:
    """Dedup key of a page: one per arXiv paper (its abs, pdf and html links
    and every version), else :func:`normalize_url`."""
    paper = arxiv_paper_key(url)
    return f"arxiv:{paper}" if paper else normalize_url(url)


def product_names(proposed: str, pages: Collection[PageContent]) -> list[str]:
    """The command and package names the relevance gate read off the docs
    (``proposed``, comma- or line-separated), kept only when the seed
    ``pages`` show them as a word. No name shown means none: a product name
    is never guessed from the site's host."""
    text = " ".join(page.text.lower() for page in pages)
    names: list[str] = []
    # A "none" answer names nothing (any Python page shows "None").
    for raw in parse_lines("\n".join(re.split(r"[,\n]", proposed))):
        name = raw.strip("`'\"").lower()
        if (
            name
            and name not in names
            and re.search(rf"(?<![\w/.-]){re.escape(name)}(?![\w-])", text)
        ):
            names.append(name)
    return names


class SourceFetchError(Exception):
    """Every requested page failed for transport reasons (timeouts, refused
    connections, 429/5xx): a retryable outage, not a bad request."""


def _is_resolver_error(error: BaseException) -> bool:
    """A DNS lookup failure (any ``socket.gaierror``) in the cause chain."""
    seen: BaseException | None = error
    while seen is not None:
        if isinstance(seen, socket.gaierror):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def _is_transport_failure(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status == 429 or status >= 500
    return isinstance(error, httpx.TransportError) or is_temporary_dns_failure(error)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    description: str = ""
    # Cleaned, extracted page body when the search backend provides it (Tavily).
    # Lets the corpus use the result directly instead of re-fetching the URL.
    content: str = ""
    # When the backend dates the page (Brave's ``page_age``, Tavily's
    # ``published_date``): ``YYYY-MM-DD``, else "".
    published: str = ""


@dataclass
class PageContent:
    url: str
    title: str
    text: str
    image_urls: list[str] = field(default_factory=list)
    # Each image's caption (its figure's caption, else its alt text; ""
    # for arXiv's "Refer to caption" placeholder).
    image_captions: dict[str, str] = field(default_factory=dict)
    links: list[str] = field(default_factory=list)
    # The links inside the page's single <article>/<main> (all of ``links``
    # when it has none): what its text cites, without its site navigation.
    content_links: list[str] = field(default_factory=list)
    # Where the page actually lives after HTTP and <meta refresh> redirects
    # (sub-page scope is taken from it); empty when it is ``url`` itself.
    final_url: str = ""
    # Explanatory material the user passed with the request (a paper, design
    # doc or deep-dive post): labelled reference material, never official.
    user_reference: bool = False
    # The source record (:mod:`source_set`), decided once at gather time by
    # :meth:`SourceSet.record`; "" until then. ``published`` is the page's
    # own date (its metadata, else the search backend's), ``YYYY-MM-DD``.
    authority: str = ""
    version: str = ""
    published: str = ""
    language: str = ""


@dataclass
class ResearchCorpus:
    """All gathered material for a guide."""

    pages: list[PageContent] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)

    @property
    def image_urls(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for page in self.pages:
            for url in page.image_urls:
                if url not in seen:
                    seen.add(url)
                    ordered.append(url)
        return ordered

    def relevant_text(
        self,
        query: str,
        *,
        max_chars: int,
        labelled: bool = False,
        prefer: Collection[str] = (),
    ) -> str:
        """The pages most relevant to ``query`` (word overlap with title and
        body), most relevant first, within ``max_chars``; a page over its room
        keeps its most relevant parts (:func:`_excerpt`). The pages whose URL
        is in ``prefer`` (the section plan's) come first, in that order, each
        with an equal share of ``max_chars`` so a long paper cannot push an
        official page out; the room they leave goes to the other pages by
        rank, a cut preferred page first. Each section of a guide reads only
        the material it needs instead of the whole corpus.
        With ``labelled``, each source is labelled by its record (official,
        another docs version, reference material or third-party) so the
        writer knows which one decides a flag or default."""
        words = {w for w in re.findall(r"[a-z0-9][a-z0-9_.-]{2,}", query.lower())}

        def score(page: PageContent) -> float:
            title = page.title.lower()
            body = page.text.lower()
            hits = sum(3 * (w in title) + min(body.count(w), 20) for w in words)
            return hits / (1 + len(body) / 20_000)  # don't favour sheer length

        by_url = {page.url: page for page in reversed(self.pages)}
        preferred = [by_url[url] for url in dict.fromkeys(prefer) if url in by_url]
        chosen = {id(page) for page in preferred}
        others = sorted(
            (p for p in self.pages if id(p) not in chosen), key=score, reverse=True
        )
        ordered = preferred + others
        blocks = [_source_block(page, labelled) for page in ordered]
        share = max_chars // len(preferred) if preferred else 0
        cuts = [_excerpt(blocks[i], share, words) for i in range(len(preferred))]
        cuts += [""] * len(others)
        used = sum(map(len, cuts))
        for i, block in enumerate(blocks):
            if max_chars - used <= _MIN_SOURCE_ROOM:
                break
            if len(cuts[i]) < len(block):
                longer = _excerpt(block, max_chars - used + len(cuts[i]), words)
                if len(longer) > len(cuts[i]):
                    used += len(longer) - len(cuts[i])
                    cuts[i] = longer
        return "\n\n---\n\n".join(cut for cut in cuts if cut)


_HEADING_LINE = re.compile(r"^#{1,6}[ \t]", re.MULTILINE)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_ELISION = "\n\n…\n\n"


def _parts(text: str, limit: int) -> list[tuple[str, int | None]]:
    """``text`` split at its heading lines, a part still over ``limit`` at its
    paragraph breaks; never inside a code fence or a ``$$`` span. Each part
    comes with the index of its heading section's first part (the one with
    the heading line), ``None`` before the first heading."""
    protected = [
        (seg.start, seg.end)
        for seg in segments(text)
        if seg.kind in ("fence", "display")
    ]
    starts = [start for start, _ in protected]

    def free(pos: int) -> bool:
        i = bisect_right(starts, pos) - 1
        return i < 0 or pos >= protected[i][1] or pos == protected[i][0]

    def spans(start: int, end: int, cuts: list[int]) -> list[tuple[int, int]]:
        bounds = [start, *(c for c in cuts if start < c < end), end]
        return list(pairwise(bounds))

    headings = [m.start() for m in _HEADING_LINE.finditer(text) if free(m.start())]
    breaks = [m.end() for m in _PARAGRAPH_BREAK.finditer(text) if free(m.end())]
    heading_starts = set(headings)
    parts: list[tuple[str, int | None]] = []
    for start, end in spans(0, len(text), headings):
        pieces = spans(start, end, breaks) if end - start > limit else [(start, end)]
        first = len(parts) if start in heading_starts else None
        parts += [(text[a:b], first) for a, b in pieces if text[a:b].strip()]
    return parts


def _excerpt(block: str, limit: int, words: Collection[str]) -> str:
    """A source block within ``limit``: whole when it fits, else its header
    (the source and title lines) and the parts of its text (:func:`_parts`)
    that hold the most of ``words``, each weighted by how few parts hold it
    (inverse document frequency), in page order, each gap marked by a "…"
    line; a kept piece of a long section brings the section's heading piece
    ("Deprecated options (removed in v0.10)") along. A reference page lists
    the flags a section needs anywhere on it, so keeping only its start
    would drop them."""
    if len(block) <= limit:
        return block
    header, sep, text = block.partition("\n\n")
    # Room for the parts, each with a join and a "…" line, and a last "…".
    room = limit - len(header) - len(sep) - len(_ELISION)
    if not sep or room <= len(_ELISION):
        return _cut(block, limit)
    pieces = _parts(text, room)
    parts = [part for part, _ in pieces]
    lowered = [part.lower() for part in parts]
    # A word in every part (the product's name, "and") picks none of them; a
    # word in few parts (a flag's) picks those.
    weight = {
        w: math.log(len(parts) / df)
        for w in words
        if (df := sum(w in part for part in lowered))
    }

    def score(i: int) -> float:
        return sum(v for w, v in weight.items() if w in lowered[i])

    ranked = sorted(range(len(parts)), key=lambda i: (-score(i), i))
    kept: dict[int, str] = {}
    for i in ranked:
        if i in kept:
            continue
        first = pieces[i][1]
        wanted = [j for j in (first, i) if j is not None and j not in kept]
        cost = sum(len(parts[j]) + len(_ELISION) for j in set(wanted))
        if cost <= room:
            kept.update((j, parts[j]) for j in wanted)
            room -= cost
        elif not kept:  # the best part alone is too long: its start
            kept[i] = _cut(parts[i], room - len(_ELISION))
            room = 0
    out: list[str] = []
    previous = -1
    for i in sorted(kept):
        if i != previous + 1:
            out.append(_ELISION.strip("\n"))
        out.append(kept[i].strip("\n"))
        previous = i
    if previous != len(parts) - 1:
        out.append(_ELISION.strip("\n"))
    return header + sep + "\n\n".join(part for part in out if part)


def _source_block(page: PageContent, labelled: bool) -> str:
    """One page for a prompt: URL, title and text, defanged (all three are
    page-controlled), labelled by :func:`source_label` when ``labelled``."""
    label = f" ({source_label(page)})" if labelled else ""
    return (
        f"## Source{label}: {neutralize_prompt_tags(page.url)}\n"
        f"# {neutralize_prompt_tags(page.title)}\n\n"
        f"{neutralize_prompt_tags(page.text)}"
    )


# How a prompt names each record authority.
_LABELS: dict[str, str] = {
    "official": OFFICIAL_LABEL,
    "official_repo": OFFICIAL_LABEL,
    "reference": REFERENCE_LABEL,
    "third_party": THIRD_PARTY_LABEL,
    "forum": THIRD_PARTY_LABEL,
    "older_docs": OLDER_DOCS_LABEL,
    "dev_docs": OLDER_DOCS_LABEL,
}


def source_label(page: PageContent) -> str:
    """How a prompt labels ``page``, read off its record (third-party until
    the page is recorded): its docs or repository are official, another docs
    version is an older docs version."""
    return _LABELS.get(page.authority, THIRD_PARTY_LABEL)


class WebSearchProvider(ABC):
    """Pluggable web-search backend."""

    # Whether this backend can actually return results. Callers gate expensive
    # research planning on this so they don't generate queries for a no-op.
    supports_search: bool = True
    # Searches that failed (429, 5xx, timeout, malformed answer) and so
    # returned nothing; not retried, reported as a run note instead.
    failed_searches: int = 0
    # True once the backend reports its quota or key spent.
    exhausted: bool = False

    @abstractmethod
    def search(
        self, query: str, *, count: int = 5, domains: Collection[str] = ()
    ) -> list[SearchResult]:
        """Results for ``query``; ``domains`` restricts them to those hosts."""


class FallbackSearchProvider(WebSearchProvider):
    """``primary``, and ``secondary`` for a query ``primary`` answers with
    nothing because it failed; once ``primary`` reports its quota spent it is
    skipped. ``failed_searches`` counts queries both failed."""

    def __init__(self, primary: WebSearchProvider, secondary: WebSearchProvider):
        self.primary = primary
        self.secondary = secondary
        self.failed_searches = 0

    def search(
        self, query: str, *, count: int = 5, domains: Collection[str] = ()
    ) -> list[SearchResult]:
        if not self.primary.exhausted:
            before = self.primary.failed_searches
            results = self.primary.search(query, count=count, domains=domains)
            if results or self.primary.failed_searches == before:
                return results
        before = self.secondary.failed_searches
        results = self.secondary.search(query, count=count, domains=domains)
        if self.secondary.failed_searches > before:
            self.failed_searches += 1
        return results


class NullSearchProvider(WebSearchProvider):
    """No-op provider: research relies solely on the supplied URLs."""

    supports_search: bool = False

    def search(
        self, query: str, *, count: int = 5, domains: Collection[str] = ()
    ) -> list[SearchResult]:
        return []


class BraveSearchProvider(WebSearchProvider):
    """Brave Search API provider (https://api.search.brave.com)."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        *,
        client: httpx.Client | None = None,
    ):
        """No request is made until :meth:`search` is called.

        Args:
            api_key: The Brave key (default: from the environment).
            timeout: Seconds per search request.
            client: The HTTP client (default: a one-off request per search).
        """
        resolved = api_key or os.environ.get(EnvVars.BRAVE_API_KEY.value)
        if not resolved:
            raise ValueError(
                f"{EnvVars.BRAVE_API_KEY.value} must be set to use Brave Search."
            )
        self.api_key: str = resolved
        self.timeout = timeout
        self._client = client

    def search(
        self, query: str, *, count: int = 5, domains: Collection[str] = ()
    ) -> list[SearchResult]:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        if domains:
            query = " OR ".join(f"site:{d}" for d in domains) + f" {query}"
        params: dict[str, str | int] = {"q": query, "count": count}
        try:
            get = httpx.get if self._client is None else self._client.get
            response = get(
                _BRAVE_WEB_SEARCH_URL,
                headers=headers,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as e:
            # ValueError covers JSONDecodeError on a malformed/truncated response.
            logger.warning("Brave search failed for '%s': %s", query, e)
            self.failed_searches += 1
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in _AUTH_FAILED_STATUSES:
                # A rejected key fails every later query the same way.
                self.exhausted = True
            return []
        web = data.get("web") if isinstance(data, dict) else None
        results = web.get("results") if isinstance(web, dict) else None
        if not isinstance(results, list):
            return []
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                description=r.get("description", ""),
                published=iso_date(str(r.get("page_age") or "")),
            )
            for r in results
            if isinstance(r, dict) and r.get("url")
        ]


class TavilySearchProvider(WebSearchProvider):
    """Tavily Search API provider (https://tavily.com).

    Built for LLM/agent retrieval: it returns cleaned, extracted page content
    alongside each result, so the corpus can use it directly without re-fetching
    the URL (no extra SSRF surface) and with better signal than raw snippets.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        *,
        client: httpx.Client | None = None,
    ):
        """No request is made until :meth:`search` is called.

        Args:
            api_key: The Tavily key (default: from the environment).
            timeout: Seconds per search request.
            client: The HTTP client (default: a one-off request per search).
        """
        resolved = api_key or os.environ.get(EnvVars.TAVILY_API_KEY.value)
        if not resolved:
            raise ValueError(
                f"{EnvVars.TAVILY_API_KEY.value} must be set to use Tavily Search."
            )
        self.api_key: str = resolved
        self.timeout = timeout
        self._client = client

    def search(
        self, query: str, *, count: int = 5, domains: Collection[str] = ()
    ) -> list[SearchResult]:
        payload: dict[str, object] = {
            "query": query,
            "max_results": count,
            "search_depth": "advanced",
            "include_raw_content": True,
        }
        if domains:
            payload["include_domains"] = list(domains)
        try:
            post = httpx.post if self._client is None else self._client.post
            response = post(
                _TAVILY_SEARCH_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as e:
            # ValueError covers JSONDecodeError on a malformed/truncated response.
            logger.warning("Tavily search failed for '%s': %s", query, e)
            self.failed_searches += 1
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in _TAVILY_EXHAUSTED_STATUSES:
                # The plan's quota or key is spent: later queries fail too.
                self.exhausted = True
            return []
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return []
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                description=r.get("content", ""),
                # raw_content is the cleaned full page; fall back to the snippet.
                content=absolute_markdown_links(
                    r.get("raw_content") or r.get("content") or "", r["url"]
                ),
                published=iso_date(str(r.get("published_date") or "")),
            )
            for r in results
            if isinstance(r, dict) and r.get("url")
        ]


_MARKDOWN_TARGET = re.compile(r"(\]\()([^)\s]+)")


def absolute_markdown_links(markdown: str, base_url: str) -> str:
    """Resolve the relative link and image targets of a fetched page's
    Markdown against the page URL, so a writer never copies a path such as
    ``../_images/x.svg`` that only works on the source site."""

    def repl(m: re.Match[str]) -> str:
        target = m.group(2)
        if target.startswith("#") or urlparse(target).scheme:
            return m.group(0)
        return m.group(1) + urljoin(base_url, target)

    return _MARKDOWN_TARGET.sub(repl, markdown)


# A non-arXiv site answering 429/503 is asked again this many times, after
# its Retry-After or PAGE_RETRY_BASE_SECONDS x 1, 3, 9.
PAGE_RATE_LIMIT_RETRIES = 3
PAGE_RETRY_BASE_SECONDS = 5.0


# A page entry of an llms.txt index: "- [Title](url): description".
_LLMS_LINK = re.compile(
    r"^\s*[-*]\s*\[([^\]]+)\]\((https?://[^)\s]+)\)(?::\s*(.*))?$", re.M
)
# llms.txt entries a guide needs first: how the tool works, then how to run it.
_INDEX_TIERS = (
    re.compile(r"architect|design|internal|how .{0,20}works|concept|overview", re.I),
    re.compile(
        r"production|deploy|scal|performance|monitor|observab|config|troubleshoot|best practice",
        re.I,
    ),
)


def _page_url(url: str) -> str:
    """The HTML page an llms.txt entry stands for: some sites list the
    Markdown source (``…/agent-loop/index.md``) of the page at ``…/agent-loop/``."""
    if url.endswith("/index.md"):
        return url[: -len("index.md")]
    if url.endswith(".md"):
        return url[: -len(".md")] + "/"
    return url


def _index_rank(label: str) -> int:
    return next((n for n, tier in enumerate(_INDEX_TIERS) if tier.search(label)), 2)


class WebResearcher:
    """Fetches pages, discovers in-scope sub-URLs, and extracts text + images."""

    def __init__(
        self,
        search_provider: WebSearchProvider | None = None,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        max_subpages: int = DEFAULT_MAX_SUBPAGES,
        max_images_per_page: int = DEFAULT_MAX_IMAGES_PER_PAGE,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.search_provider = search_provider or NullSearchProvider()
        # Injected so tests do not wait out a rate limit.
        self._sleep = sleep
        # Host -> its llms.txt entries (empty when it has none).
        self._llms_cache: dict[str, list[tuple[str, str]]] = {}
        self.timeout = timeout
        self.max_subpages = max_subpages
        self.max_images_per_page = max_images_per_page
        self._client = client

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            # PinnedHTTPTransport closes the SSRF DNS-rebinding TOCTOU: each
            # request (and redirect hop) is validated and connected to the exact
            # validated IP rather than re-resolved.
            self._client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                transport=PinnedHTTPTransport(),
            )
        return self._client

    def search_notes(self) -> list[str]:
        """A run note when web searches failed (their results are missing
        from the research), else nothing."""
        failed = self.search_provider.failed_searches
        if not failed:
            return []
        return [
            f"{failed} web search(es) failed (rate limit, server error or "
            "timeout) and were not retried: the research is missing their results."
        ]

    def close(self) -> None:
        """Close the lazily-created HTTP client (releases its connection pool)."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> WebResearcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def research(
        self,
        urls: list[str],
        *,
        discover_subpages: bool = True,
    ) -> ResearchCorpus:
        corpus = ResearchCorpus()
        visited: set[str] = set()
        failures: list[BaseException] = []

        for url in urls:
            page = self._fetch_page(url, failures=failures)
            if page is None:
                continue
            corpus.pages.append(page)
            visited.add(normalize_url(url))

            if discover_subpages:
                for sub_url in self._select_subpages(page, page.final_url or url):
                    if normalize_url(sub_url) in visited:
                        continue
                    sub_page = self._fetch_page(sub_url)
                    if sub_page is not None:
                        corpus.pages.append(sub_page)
                        visited.add(normalize_url(sub_url))

        if urls and not corpus.pages and len(failures) == len(urls):
            if all(_is_transport_failure(e) for e in failures):
                raise SourceFetchError(
                    f"Could not reach any of the requested pages: {failures[0]}"
                ) from failures[0]

        logger.info(
            "Researched %d pages, %d unique images.",
            len(corpus.pages),
            len(corpus.image_urls),
        )
        return corpus

    def run_searches(
        self, corpus: ResearchCorpus, queries: list[str], *, fetch_top: int = 0
    ) -> None:
        """Append web-search results (and optionally their pages) to a corpus.

        Separate from :meth:`research` so the deep-research planner can fetch
        the seed pages once, derive queries from them, then broaden the corpus
        without re-fetching the seed URLs.

        ``fetch_top`` > 0 also brings that many not-yet-visited results, taken
        round-robin across the queries, into ``corpus.pages`` so the gathered material actually
        reaches the writer (a result list alone is title+snippet only). A result
        that already carries cleaned ``content`` (Tavily) is used directly — no
        re-fetch, so no extra SSRF surface; otherwise the URL is fetched through
        the same SSRF guard as every other fetch (Brave). Deduped against pages
        already in the corpus.
        """
        per_query = [self.search_provider.search(query) for query in queries]
        for results in per_query:
            corpus.search_results.extend(results)

        if fetch_top <= 0:
            return

        # Take result i of every query before result i + 1 of any, so no single
        # query (e.g. an internals one) fills the whole fetch budget.
        interleaved = [
            results[i]
            for i in range(max((len(r) for r in per_query), default=0))
            for results in per_query
            if i < len(results)
        ]
        visited = {page_key(u) for p in corpus.pages for u in (p.url, p.final_url) if u}
        added = 0
        for result in interleaved:
            if added >= fetch_top:
                break
            key = page_key(result.url)
            if not urlparse(result.url).hostname or key in visited:
                continue
            visited.add(key)
            # An arXiv link is read from the paper's HTML rendering: the
            # backend's content of an abstract page is the abstract alone.
            page = (
                self._fetch_page(result.url)
                if arxiv_paper_urls(result.url) or not result.content
                else None
            )
            if page is None and result.content:
                # Backend already returned cleaned page content; use it directly.
                page = PageContent(
                    url=result.url,
                    title=result.title or result.url,
                    text=result.content,
                )
            if page is not None and not page.published:
                page.published = result.published
            if page is not None and page.final_url:
                # A redirect to a page the corpus already holds.
                landed = page_key(page.final_url)
                if landed != key and landed in visited:
                    continue
                visited.add(landed)
            if page is not None:
                corpus.pages.append(page)
                added += 1
        logger.info(
            "Added %d/%d search-result pages into the corpus.", added, fetch_top
        )

    def add_official_reference(
        self,
        corpus: ResearchCorpus,
        sources: SourceSet,
        identifiers: list[str],
        *,
        product: str = "",
        max_searches: int = MAX_REFERENCE_SEARCHES,
        max_pages: int = MAX_REFERENCE_PAGES,
        settles: Callable[[str], bool] | None = None,
    ) -> list[PageContent]:
        """Fetch reference pages of the official docs that can settle the
        given flags and metric names, add them to ``corpus`` and return them.

        Candidates are search results restricted to those hosts (one query per
        batch of identifiers, at most ``max_searches``), then reference-like links (CLI, arguments, configuration,
        metrics …) of the official pages already fetched, those whose path
        fits the identifiers first. With ``settles``, a page counts only when
        it settles an identifier (``settles(page.text)``); one that does not
        (an index of the CLI pages) is left out, and its reference-like links join the end of the candidates, ranked the
        same way. At most ``max_pages`` are added and ``2 * max_pages``
        fetched, each through the same SSRF-guarded fetch. What is official is
        read off each page's record (``sources`` records the fetched ones).
        """
        hosts = sorted(sources.official.hosts)
        sources.record_all(corpus.pages)  # a page that has a record keeps it

        def covers(url: str) -> bool:
            return sources.classify(url).authority in OFFICIAL_AUTHORITIES

        if not hosts or not identifiers or max_pages <= 0:
            return []
        visited = {normalize_url(p.final_url or p.url) for p in corpus.pages}
        visited |= {normalize_url(p.url) for p in corpus.pages}
        results: list[SearchResult] = []
        if self.search_provider.supports_search and max_searches > 0:
            searches = min(max_searches, len(identifiers))
            size = -(-len(identifiers) // searches)
            for i in range(searches):
                batch = " ".join(identifiers[i * size : (i + 1) * size])
                query = f"{product} {batch}".strip()
                results += self.search_provider.search(query, domains=hosts)
        wanted = _reference_words(identifiers)
        links: list[str] = []

        def queue_links(pages: list[PageContent]) -> None:
            """Queue the pages' reference-like links after those already
            queued, so a followed page's links never jump ahead of the links
            of the pages before it."""
            batch = [
                link
                for page in pages
                if is_official(page)
                for link in page.links
                if covers(link)
                and _REFERENCE_LINK.search(urlparse(link).path)
                and normalize_url(link) not in visited
            ]
            # Navigation order lists "metrics" before "engine_args"; a stable
            # sort by fit keeps that order only among equally fitting pages.
            batch.sort(key=lambda link: -len(_path_words(link) & wanted))
            links.extend(batch)

        queue_links(corpus.pages)
        added: list[PageContent] = []
        fetches = 0
        max_fetches = 2 * max_pages
        while (results or links) and len(added) < max_pages:
            result = results.pop(0) if results else SearchResult("", links.pop(0))
            norm = normalize_url(result.url)
            if not norm or norm in visited:
                continue
            if not covers(result.url):
                continue  # off-host (a backend ignoring the filter) or an old release
            visited.add(norm)
            page: PageContent | None
            if result.content:
                page = PageContent(
                    url=result.url,
                    title=result.title or result.url,
                    text=result.content,
                )
            elif fetches >= max_fetches:
                continue
            else:
                fetches += 1
                page = self._fetch_page(result.url)
            if page is None:
                continue
            sources.record(page)
            if not is_official(page):
                continue
            if settles is None or settles(page.text):
                added.append(page)
            else:
                logger.info(
                    "Reference candidate '%s' mentions none of the identifiers.",
                    page.final_url or page.url,
                )
                queue_links([page])
        corpus.pages.extend(added)
        logger.info(
            "Added %d official reference page(s) for %d identifier(s) "
            "(%d page(s) fetched).",
            len(added),
            len(identifiers),
            fetches,
        )
        return added

    def _fetch_page(
        self, url: str, *, failures: list[BaseException] | None = None
    ) -> PageContent | None:
        # SSRF guard, two layers: a cheap pre-check here for a fast skip + clear
        # log without opening a connection, and the authoritative check in
        # PinnedHTTPTransport, which validates and pins EVERY request — including
        # each redirect hop — to the exact validated IP (no DNS-rebinding window,
        # no unchecked redirect target). The transport raises UnsafeUrlError if a
        # hop targets a non-public address.
        if not is_url_public(url):
            try:
                # Again, for the reason. A lookup that now succeeds was a
                # resolver hiccup: go on and fetch (the transport re-checks).
                assert_url_is_public(url)
            except UnsafeUrlError as e:
                if is_temporary_dns_failure(e):  # an outage, not an unsafe URL
                    logger.warning("Failed to resolve '%s': %s", url, e)
                    if failures is not None:
                        failures.append(e)
                    return None
                if _is_resolver_error(e):
                    logger.warning(
                        "Skipping '%s': the host could not be resolved.", url
                    )
                else:
                    logger.warning(
                        "Skipping non-public URL '%s' (SSRF guard): %s", url, e
                    )
                return None
        try:
            fetched = self._get_html(url)
        except UnsafeUrlError as e:
            if is_temporary_dns_failure(e):  # at connect time: an outage
                logger.warning("Failed to resolve '%s' while fetching: %s", url, e)
                if failures is not None:
                    failures.append(e)
            elif _is_resolver_error(e):
                logger.warning(
                    "Skipping '%s': a host could not be resolved: %s", url, e
                )
            else:
                logger.warning(
                    "Skipping '%s' — it or a redirect targets a non-public address: %s",
                    url,
                    e,
                )
            return None
        except httpx.HTTPError as e:
            logger.warning("Failed to fetch '%s': %s", url, e)
            if failures is not None:
                failures.append(e)
            return None
        if fetched is None:
            return None
        html, final_url = fetched

        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else url
        # Relative links and image markers resolve against where the page
        # actually lives (after redirects), not the URL we asked for.
        main = _main_content(soup)
        text = extract_text_from_html(
            main or html, base_url=final_url, skip_page_furniture=True
        )
        # Images from the same scope as the text, so sidebar logos and badges
        # never become figures the writer may cite.
        # An arXiv abstract page has no figures, only site icons.
        content = BeautifulSoup(main, "html.parser") if main else soup
        images = (
            {}
            if is_arxiv_host(final_url) and "/abs/" in urlparse(final_url).path
            else self._extract_images(content, final_url)
        )
        image_urls = list(images)
        # The writer cites only listed images: drop the markers of the others
        # (over the per-page cap, or furniture).
        kept = set(image_urls)
        text = FIGURE_MARKER.sub(
            lambda m: m.group(0) if m.group("src") in kept else "", text
        )
        links = self._extract_links(soup, final_url)
        return PageContent(
            url=url,
            title=title,
            text=text,
            published=published_date(soup),
            image_urls=image_urls,
            image_captions=images,
            links=links,
            content_links=(
                self._extract_links(content, final_url) if main else list(links)
            ),
            final_url=final_url if final_url != url else "",
        )

    def _get_html(self, url: str) -> tuple[str, str] | None:
        """Fetch ``url`` like :meth:`_get_page`, also following a small
        stub page's ``<meta http-equiv="refresh">`` (docs sites such as
        ``/stable/`` aliases use it instead of an HTTP redirect) for at most
        :data:`MAX_META_REFRESH_HOPS` hops. An arXiv abstract or PDF link is
        read from the paper's HTML rendering, else its abstract page
        (:func:`arxiv_paper_urls`). Every hop goes through the same pinned,
        SSRF-guarded client."""
        paper = arxiv_paper_urls(url)
        fetched = self._get_paper(*paper) if paper else self._get_page(url)
        for _ in range(MAX_META_REFRESH_HOPS):
            if fetched is None:
                return None
            target = _meta_refresh_target(*fetched)
            if target is None:
                break
            if not is_url_public(target):
                raise UnsafeUrlError(f"meta refresh to non-public '{target}'")
            logger.info("Following meta refresh '%s' -> '%s'", fetched[1], target)
            fetched = self._get_page(target)
        return fetched

    def _get_paper(self, html_url: str, abs_url: str) -> tuple[str, str] | None:
        """An arXiv paper's HTML rendering, else its abstract page: when the
        paper has no rendering (404) or the rendering is a failed conversion's
        stub ("See pages 1-last of main.pdf", under the parser's
        :data:`MIN_PARSED_TEXT_CHARS`). Any other error (429, 5xx, a timeout)
        propagates, so the caller records it as a fetch failure."""
        try:
            fetched = self._get_page(html_url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
            logger.info("No HTML rendering at '%s'; reading '%s'.", html_url, abs_url)
            return self._get_page(abs_url)
        if fetched is not None and _readable_chars(fetched[0]) >= MIN_PARSED_TEXT_CHARS:
            return fetched
        logger.info("'%s' has almost no text; reading '%s'.", html_url, abs_url)
        return self._get_page(abs_url)

    def _get_page(self, url: str) -> tuple[str, str] | None:
        """:meth:`_get_page_once`, with a site that answers 429 or 503 (a
        docs site behind a bot shield) asked again up to
        :data:`PAGE_RATE_LIMIT_RETRIES` times after its Retry-After (capped)
        or a growing wait. arXiv is not retried here: its limiter holds every
        later request back and the job fails retryably."""
        for attempt in range(PAGE_RATE_LIMIT_RETRIES + 1):
            try:
                return self._get_page_once(url)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if (
                    status not in (429, 503)
                    or is_arxiv_host(url)
                    or attempt == PAGE_RATE_LIMIT_RETRIES
                ):
                    raise
                hinted = "retry-after" in e.response.headers
                wait = (
                    arxiv_retry_after_seconds(e)
                    if hinted
                    else PAGE_RETRY_BASE_SECONDS * 3**attempt
                ) or PAGE_RETRY_BASE_SECONDS
                wait *= 1 + random.uniform(0, 0.25)
                logger.info(
                    "'%s' answered %d; asking again in %.0f s.", url, status, wait
                )
                self._sleep(wait)
        raise AssertionError("unreachable")

    def _get_page_once(self, url: str) -> tuple[str, str] | None:
        """Stream ``url``; return (decoded HTML, final URL), or None when the
        resource is not HTML or exceeds :data:`MAX_PAGE_BYTES`. A request to
        arXiv waits for the process-wide arXiv rate limiter, and an arXiv 429
        holds every later arXiv request back."""
        on_arxiv = is_arxiv_host(url)
        if on_arxiv:
            ARXIV_RATE_LIMITER.acquire()
        with self.client.stream("GET", url) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if on_arxiv and response.status_code == 429:
                    ARXIV_RATE_LIMITER.penalize(arxiv_retry_after_seconds(e))
                raise
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" not in content_type and "xml" not in content_type:
                logger.debug("Skipping non-HTML resource '%s' (%s)", url, content_type)
                return None
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > MAX_PAGE_BYTES:
                    logger.warning(
                        "Skipping '%s': larger than %d bytes.", url, MAX_PAGE_BYTES
                    )
                    return None
            final_url = str(response.url)
            charset = response.charset_encoding
        raw = bytes(body)
        html = None
        if charset:
            try:
                html = raw.decode(charset, errors="replace")
            except LookupError:  # unknown charset label: sniff instead
                logger.debug("Unknown charset '%s' for '%s'", charset, url)
        if html is None:
            # No charset header: honour <meta charset> / byte sniffing instead
            # of assuming UTF-8 (EUC-KR, Shift_JIS docs otherwise turn to noise).
            html = UnicodeDammit(raw, is_html=True).unicode_markup or ""
        return html, final_url

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> dict[str, str]:
        """Content images, deduped, each with its caption: ``<img>`` and the
        SVG ``<object>`` figures arXiv renders (the elements the text's
        figure markers come from); tracking pixels and the logos in page
        chrome (header, navigation, footer, ``<noscript>``) are skipped."""
        images: dict[str, str] = {}
        for img in soup.find_all(("img", "object")):
            if not isinstance(img, Tag) or is_page_furniture(img):
                continue
            if img.name == "object":
                if "svg" not in str(img.get("type", "")):
                    continue
                src = img.get("data")
            else:
                src = img.get("src") or img.get("data-src")
            if isinstance(src, str) and src.strip():
                absolute = urljoin(base_url, src.strip())
                if (
                    absolute.lower().startswith(("http://", "https://"))
                    and absolute not in images
                ):
                    images[absolute] = _real_text(figure_caption(img))
            if len(images) >= self.max_images_per_page:
                break
        return images

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Absolute links, those in the main content area first (navigation
        menus would otherwise fill the sub-page budget)."""
        links: list[str] = []
        main = soup.find("main") or soup.find("article")
        anchors = list(main.find_all("a")) if isinstance(main, Tag) else []
        anchors += [a for a in soup.find_all("a") if a not in anchors]
        for anchor in anchors:
            if not isinstance(anchor, Tag):
                continue
            href = anchor.get("href")
            if isinstance(href, str) and href.strip():
                absolute = urljoin(base_url, href.strip().split("#")[0])
                if absolute.lower().startswith(("http://", "https://")):
                    links.append(absolute)
        return links

    def _select_subpages(self, page: PageContent, base_url: str) -> list[str]:
        """In-scope sub-URLs: same host and under the seed's path, deduped. A
        leaf page without an extension (``/docs/peft/guides/lora``) has no
        children, so its siblings are in scope instead — only when they share
        a topic directory (``/en/latest/`` alone is the whole docs version).
        When the site publishes an ``llms.txt`` index, its in-scope pages come
        first, the architecture and concept pages ahead of the rest (a page's
        navigation order buries them); the page's own links fill up."""
        base = urlparse(base_url)
        scope = _scope_prefix(base.path)
        indexed = self._llms_index(base_url)
        selected = self._indexed_under(indexed, base_url, scope)
        selected += [
            u for u in self._links_under(page, base_url, scope) if u not in selected
        ]
        parent = scope.rstrip("/").rsplit("/", 1)[0] + "/"
        if not selected and _is_topic_directory(parent):
            selected = self._indexed_under(indexed, base_url, parent)
            selected += [
                u
                for u in self._links_under(page, base_url, parent)
                if u not in selected
            ]
        return selected[: self.max_subpages]

    def _llms_index(self, base_url: str) -> list[tuple[str, str]]:
        """(url, title and description) of each page the site's ``/llms.txt``
        lists, in its order; empty when the site has none. Fetched once per
        host."""
        host = urlparse(base_url).netloc
        if host in self._llms_cache:
            return self._llms_cache[host]
        entries: list[tuple[str, str]] = []
        index_url = f"{urlparse(base_url).scheme}://{host}/llms.txt"
        try:
            response = self.client.get(index_url)
            if response.status_code == 200 and response.text.lstrip().startswith("#"):
                entries = [
                    (_page_url(m.group(2)), f"{m.group(1)} {m.group(3) or ''}".strip())
                    for m in _LLMS_LINK.finditer(response.text)
                ]
        except httpx.HTTPError as e:
            logger.info("No llms.txt at %s: %s", index_url, e)
        if entries:
            logger.info("Using %d pages listed in %s.", len(entries), index_url)
        self._llms_cache[host] = entries
        return entries

    def _indexed_under(
        self, indexed: list[tuple[str, str]], base_url: str, scope: str
    ) -> list[str]:
        base = urlparse(base_url)
        found: list[tuple[int, int, int, int, str]] = []
        seen = {normalize_url(base_url)}
        for order, (url, label) in enumerate(indexed):
            parsed = urlparse(url)
            norm = normalize_url(url)
            if (
                parsed.netloc != base.netloc
                or not (parsed.path.rstrip("/") + "/").startswith(scope)
                or norm in seen
            ):
                continue
            seen.add(norm)
            depth = parsed.path[len(scope) :].strip("/").count("/")
            # Pages right under the scope first (the tool's own architecture
            # page before a sub-product's), by tier, then the rest.
            found.append((min(depth, 1), _index_rank(label), depth, order, url))
        return [entry[-1] for entry in sorted(found)]

    def _links_under(self, page: PageContent, base_url: str, scope: str) -> list[str]:
        base = urlparse(base_url)
        selected: list[str] = []
        seen: set[str] = set()
        for link in page.links:
            parsed = urlparse(link)
            if parsed.netloc != base.netloc:
                continue
            if not (parsed.path.rstrip("/") + "/").startswith(scope):
                continue
            norm = normalize_url(link)
            if norm == normalize_url(base_url) or norm in seen:
                continue
            seen.add(norm)
            selected.append(link)
            if len(selected) >= self.max_subpages:
                break
        return selected


# Page metadata that dates a page: Open Graph article and common meta names.
_DATE_META = (
    ("property", "article:published_time"),
    ("name", "article:published_time"),
    ("itemprop", "datePublished"),
    ("name", "date"),
    ("name", "publish-date"),
    ("name", "pubdate"),
    ("name", "dc.date"),
    ("name", "dc.date.issued"),
    ("name", "citation_publication_date"),
)


def published_date(soup: BeautifulSoup) -> str:
    """The page's publication date (``YYYY-MM-DD``) from its metadata:
    ``article:published_time`` and similar meta tags, else a JSON-LD
    ``datePublished``; "" when it states none."""
    metas = [m for m in soup.find_all("meta") if isinstance(m, Tag)]
    for attribute, name in _DATE_META:
        for meta in metas:
            if str(meta.get(attribute, "")).strip().lower() == name.lower():
                if found := iso_date(str(meta.get("content", ""))):
                    return found
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.get_text() or "null")
        except ValueError:
            continue
        if found := _json_ld_date(data):
            return found
    return ""


def _json_ld_date(data: object) -> str:
    """The first ``datePublished`` in a JSON-LD value (objects, lists and
    ``@graph`` nodes)."""
    if isinstance(data, list):
        return next((d for item in data if (d := _json_ld_date(item))), "")
    if not isinstance(data, dict):
        return ""
    if found := iso_date(str(data.get("datePublished") or "")):
        return found
    return _json_ld_date(data.get("@graph") or [])


def _main_content(soup: BeautifulSoup) -> str:
    """The page's single ``<article>`` (else its single ``<main>``) as HTML,
    so the sidebars, version pickers and "rate this page" widgets around it
    stay out of the text; "" when the page has no such landmark."""
    for name in ("article", "main"):
        found = soup.find_all(name)
        if len(found) == 1 and found[0].get_text(strip=True):
            return str(found[0])
    return ""


def _cut(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` at a paragraph break, never inside a code
    fence (a fence left open would make the rest of the prompt read as code)."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    if (paragraph := head.rfind("\n\n")) > limit // 2:
        head = head[:paragraph]
    # Cut inside a math span: drop the open one. Code is masked first, since a
    # shell ``$$`` (the PID) is not math and would otherwise cut the page.
    masked = list(head)
    for seg in segments(head):
        if seg.kind in ("fence", "icode"):
            masked[seg.start : seg.end] = " " * (seg.end - seg.start)
    prose = "".join(masked)
    if prose.count("$$") % 2:
        head = head[: prose.rfind("$$")].rstrip()
        prose = prose[: len(head)]
    # A cut inside inline code leaves an odd backtick, inside "$$" a lone "$".
    if prose.count("`") % 2:
        head = head[: prose.rfind("`")].rstrip()
        prose = prose[: len(head)]
    # Only a "$$" split by the cut leaves a lone "$"; "$x$" ends correctly.
    if re.search(r"(?<!\$)\$$", prose) and text[len(head) : len(head) + 1] == "$":
        head = head[:-1].rstrip()
    return close_open_fence(head)


def _meta_refresh_target(html: str, base_url: str) -> str | None:
    """Absolute http(s) target of a stub page's ``<meta http-equiv=refresh>``,
    or None (no tag, a page with real content, or a self-refresh)."""
    if len(html) > _META_REFRESH_MAX_BYTES:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for meta in soup.find_all("meta"):
        if not isinstance(meta, Tag):
            continue
        if str(meta.get("http-equiv", "")).strip().lower() != "refresh":
            continue
        match = _META_REFRESH_URL.match(str(meta.get("content", "")))
        if not match:
            continue
        target = urljoin(base_url, match.group(1).strip())
        if target.lower().startswith(("http://", "https://")) and (
            normalize_url(target) != normalize_url(base_url)
        ):
            return target
    return None


def _scope_prefix(path: str) -> str:
    """Directory a seed URL scopes sub-page discovery to, with a trailing slash
    so ``/guide/`` never matches ``/guidelines``. ``/docs/intro.html`` scopes
    to ``/docs/``; ``/guide`` (a section index) scopes to ``/guide/``."""
    last = path.rsplit("/", 1)[-1]
    directory = path.rsplit("/", 1)[0] if "." in last else path.rstrip("/")
    return directory + "/"


def _is_topic_directory(path: str) -> bool:
    """Whether ``path`` names a section below the site root: at least two
    segments, and at least one after a docs version (``/en/latest/`` and
    ``/v2/`` are a whole release, ``/docs/peft/`` is a section)."""
    segments = [s for s in path.split("/") if s]
    if len(segments) < 2:
        return False
    version = docs_version(path)
    if version is None:
        return True
    index = [s.lower() for s in segments].index(version)
    return index < len(segments) - 1
