"""Web research primitives for the technical-guide generator.

Gathers source material for a tech guide/tutorial from a user-supplied URL
list: it fetches each page, optionally discovers in-scope sub-URLs (same-site
links), extracts readable text and image references, and can augment the corpus
with web-search results via a pluggable :class:`WebSearchProvider`.

The search provider is an abstraction so deployments can choose their backend
(Brave Search API by default, or none) without changing the agent.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from .constants import EnvVars
from .logger import logger
from .pinned_transport import PinnedHTTPTransport
from .url_guard import UnsafeUrlError, is_url_public
from .utils import extract_text_from_html

DEFAULT_TIMEOUT: int = 30
DEFAULT_MAX_SUBPAGES: int = 8
DEFAULT_MAX_IMAGES_PER_PAGE: int = 12
_BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Fetched page/search text is interpolated into prompt XML fences (<sources>,
# <search_results>, ...). A malicious page body containing a literal closing tag
# could break out of the data fence and inject instructions, so the prompts'
# "treat as untrusted DATA" framing is only sound if the delimiter itself can't
# appear in the data. Neutralise any "</tag>" / "<tag>" fence sequence by
# inserting a zero-width space after "<", which renders identically but no longer
# matches the delimiter. Only the prompt fence tags are touched, so legitimate
# markup/code in the source is left intact.
_PROMPT_FENCE_TAGS = (
    # Tech-guide fences.
    "sources",
    "source",
    "search_results",
    "available_images",
    "previously_written_sections",
    "section_to_write",
    "section",
    "depth_directive",
    # Review / summary / extraction fences wrapping untrusted paper content, so
    # the same defang protects those pipelines (a paper body is attacker-supplied
    # via the Slack bot just like a web page is).
    "paper",
    "paper_content",
    "current_content",
    "current_explanation",
    "previous_explanation",
    "citation_summaries",
    "codebase_summary",
    "official_codebase_summary",
    "numbered_sentences",
    "citations",
    "existing_citations",
    "existing_keywords",
)
_FENCE_TAG_RE = re.compile(
    r"<(/?)(" + "|".join(_PROMPT_FENCE_TAGS) + r")\s*>", re.IGNORECASE
)


def neutralize_prompt_tags(text: str) -> str:
    """Defang prompt-fence tags in untrusted text so it can't break the fence."""
    return _FENCE_TAG_RE.sub(lambda m: f"<​{m.group(1)}{m.group(2)}>", text)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    description: str = ""
    # Cleaned, extracted page body when the search backend provides it (Tavily).
    # Lets the corpus use the result directly instead of re-fetching the URL.
    content: str = ""


@dataclass
class PageContent:
    url: str
    title: str
    text: str
    image_urls: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)


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

    def combined_text(self, max_chars: int | None = None) -> str:
        blocks = [
            f"## Source: {p.url}\n# {p.title}\n\n{neutralize_prompt_tags(p.text)}"
            for p in self.pages
        ]
        text = "\n\n---\n\n".join(blocks)
        return text[:max_chars] if max_chars else text


class WebSearchProvider(ABC):
    """Pluggable web-search backend."""

    # Whether this backend can actually return results. Callers gate expensive
    # research planning on this so they don't generate queries for a no-op.
    supports_search: bool = True

    @abstractmethod
    def search(self, query: str, *, count: int = 5) -> list[SearchResult]: ...


class NullSearchProvider(WebSearchProvider):
    """No-op provider: research relies solely on the supplied URLs."""

    supports_search: bool = False

    def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        return []


class BraveSearchProvider(WebSearchProvider):
    """Brave Search API provider (https://api.search.brave.com)."""

    def __init__(self, api_key: str | None = None, timeout: int = DEFAULT_TIMEOUT):
        resolved = api_key or os.environ.get(EnvVars.BRAVE_API_KEY.value)
        if not resolved:
            raise ValueError(
                f"{EnvVars.BRAVE_API_KEY.value} must be set to use Brave Search."
            )
        self.api_key: str = resolved
        self.timeout = timeout

    def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params: dict[str, str | int] = {"q": query, "count": count}
        try:
            response = httpx.get(
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
            return []
        results = data.get("web", {}).get("results") or []
        if not isinstance(results, list):
            return []
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                description=r.get("description", ""),
            )
            for r in results
            if r.get("url")
        ]


class TavilySearchProvider(WebSearchProvider):
    """Tavily Search API provider (https://tavily.com).

    Built for LLM/agent retrieval: it returns cleaned, extracted page content
    alongside each result, so the corpus can use it directly without re-fetching
    the URL (no extra SSRF surface) and with better signal than raw snippets.
    """

    def __init__(self, api_key: str | None = None, timeout: int = DEFAULT_TIMEOUT):
        resolved = api_key or os.environ.get(EnvVars.TAVILY_API_KEY.value)
        if not resolved:
            raise ValueError(
                f"{EnvVars.TAVILY_API_KEY.value} must be set to use Tavily Search."
            )
        self.api_key: str = resolved
        self.timeout = timeout

    def search(self, query: str, *, count: int = 5) -> list[SearchResult]:
        payload: dict[str, object] = {
            "query": query,
            "max_results": count,
            "search_depth": "advanced",
            "include_raw_content": True,
        }
        try:
            response = httpx.post(
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
            return []
        results = data.get("results") or []
        if not isinstance(results, list):
            return []
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                description=r.get("content", ""),
                # raw_content is the cleaned full page; fall back to the snippet.
                content=(r.get("raw_content") or r.get("content") or ""),
            )
            for r in results
            if r.get("url")
        ]


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
    ) -> None:
        self.search_provider = search_provider or NullSearchProvider()
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
        search_queries: list[str] | None = None,
    ) -> ResearchCorpus:
        corpus = ResearchCorpus()
        visited: set[str] = set()

        for url in urls:
            page = self._fetch_page(url)
            if page is None:
                continue
            corpus.pages.append(page)
            visited.add(_normalize_url(url))

            if discover_subpages:
                for sub_url in self._select_subpages(page, url):
                    if _normalize_url(sub_url) in visited:
                        continue
                    sub_page = self._fetch_page(sub_url)
                    if sub_page is not None:
                        corpus.pages.append(sub_page)
                        visited.add(_normalize_url(sub_url))

        self.run_searches(corpus, search_queries or [])

        logger.info(
            "Researched %d pages, %d search results, %d unique images.",
            len(corpus.pages),
            len(corpus.search_results),
            len(corpus.image_urls),
        )
        return corpus

    def run_searches(
        self, corpus: ResearchCorpus, queries: list[str], *, fetch_top: int = 0
    ) -> None:
        """Append web-search results (and optionally their pages) to a corpus.

        Split out from :meth:`research` so the deep-research planner can fetch
        the seed pages once, derive queries from them, then broaden the corpus
        without re-fetching the seed URLs.

        ``fetch_top`` > 0 also brings that many of the highest-ranked, not-yet-
        visited results into ``corpus.pages`` so the gathered material actually
        reaches the writer (a result list alone is title+snippet only). A result
        that already carries cleaned ``content`` (Tavily) is used directly — no
        re-fetch, so no extra SSRF surface; otherwise the URL is fetched through
        the same SSRF guard as every other fetch (Brave). Deduped against pages
        already in the corpus.
        """
        for query in queries:
            corpus.search_results.extend(self.search_provider.search(query))

        if fetch_top <= 0:
            return

        visited = {_normalize_url(p.url) for p in corpus.pages}
        added = 0
        for result in corpus.search_results:
            if added >= fetch_top:
                break
            norm = _normalize_url(result.url)
            if not norm or norm in visited:
                continue
            visited.add(norm)
            if result.content:
                # Backend already returned cleaned page content; use it directly.
                corpus.pages.append(
                    PageContent(
                        url=result.url,
                        title=result.title or result.url,
                        text=result.content,
                    )
                )
                added += 1
                continue
            page = self._fetch_page(result.url)
            if page is not None:
                corpus.pages.append(page)
                added += 1
        logger.info(
            "Added %d/%d search-result pages into the corpus.", added, fetch_top
        )

    def _fetch_page(self, url: str) -> PageContent | None:
        # SSRF guard, two layers: a cheap pre-check here for a fast skip + clear
        # log without opening a connection, and the authoritative check in
        # PinnedHTTPTransport, which validates and pins EVERY request — including
        # each redirect hop — to the exact validated IP (no DNS-rebinding window,
        # no unchecked redirect target). The transport raises UnsafeUrlError if a
        # hop targets a non-public address.
        if not is_url_public(url):
            logger.warning("Skipping non-public URL '%s' (SSRF guard).", url)
            return None
        try:
            response = self.client.get(url)
            response.raise_for_status()
        except UnsafeUrlError as e:
            logger.warning(
                "Skipping '%s' — redirected to non-public target: %s", url, e
            )
            return None
        except httpx.HTTPError as e:
            logger.warning("Failed to fetch '%s': %s", url, e)
            return None

        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type and "xml" not in content_type:
            logger.debug("Skipping non-HTML resource '%s' (%s)", url, content_type)
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else url
        text = extract_text_from_html(response.text)
        image_urls = self._extract_images(soup, url)
        links = self._extract_links(soup, url)
        return PageContent(
            url=url,
            title=title,
            text=text,
            image_urls=image_urls,
            links=links,
        )

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        images: list[str] = []
        for img in soup.find_all("img"):
            if not isinstance(img, Tag):
                continue
            src = img.get("src") or img.get("data-src")
            if isinstance(src, str) and src.strip():
                absolute = urljoin(base_url, src.strip())
                if absolute.lower().startswith(("http://", "https://")):
                    images.append(absolute)
            if len(images) >= self.max_images_per_page:
                break
        return images

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        links: list[str] = []
        for anchor in soup.find_all("a"):
            if not isinstance(anchor, Tag):
                continue
            href = anchor.get("href")
            if isinstance(href, str) and href.strip():
                absolute = urljoin(base_url, href.strip().split("#")[0])
                if absolute.lower().startswith(("http://", "https://")):
                    links.append(absolute)
        return links

    def _select_subpages(self, page: PageContent, base_url: str) -> list[str]:
        """In-scope sub-URLs: same host and under the base path, deduped."""
        base = urlparse(base_url)
        base_dir = base.path.rsplit("/", 1)[0]
        selected: list[str] = []
        seen: set[str] = set()
        for link in page.links:
            parsed = urlparse(link)
            if parsed.netloc != base.netloc:
                continue
            if not parsed.path.startswith(base_dir):
                continue
            norm = _normalize_url(link)
            if norm == _normalize_url(base_url) or norm in seen:
                continue
            seen.add(norm)
            selected.append(link)
            if len(selected) >= self.max_subpages:
                break
        return selected


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/").lower()
