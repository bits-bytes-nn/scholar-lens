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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from .constants import EnvVars
from .logger import logger
from .url_guard import is_url_public
from .utils import extract_text_from_html

DEFAULT_TIMEOUT: int = 30
DEFAULT_MAX_SUBPAGES: int = 8
DEFAULT_MAX_IMAGES_PER_PAGE: int = 12
_BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    description: str = ""


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
        blocks = [f"## Source: {p.url}\n# {p.title}\n\n{p.text}" for p in self.pages]
        text = "\n\n---\n\n".join(blocks)
        return text[:max_chars] if max_chars else text


class WebSearchProvider(ABC):
    """Pluggable web-search backend."""

    @abstractmethod
    def search(self, query: str, *, count: int = 5) -> list[SearchResult]: ...


class NullSearchProvider(WebSearchProvider):
    """No-op provider: research relies solely on the supplied URLs."""

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
        results = data.get("web", {}).get("results", [])
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                description=r.get("description", ""),
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
            self._client = httpx.Client(timeout=self.timeout, follow_redirects=True)
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

    def run_searches(self, corpus: ResearchCorpus, queries: list[str]) -> None:
        """Append web-search results for each query onto an existing corpus.

        Split out from :meth:`research` so the deep-research planner can fetch
        the seed pages once, derive queries from them, then add search results
        without re-fetching the seed URLs.
        """
        for query in queries:
            corpus.search_results.extend(self.search_provider.search(query))

    def _fetch_page(self, url: str) -> PageContent | None:
        # SSRF guard: every fetched URL (initial, discovered sub-page, or link)
        # flows through here, so block internal/metadata targets at this choke
        # point. follow_redirects is on, but redirects re-enter the httpx client
        # without re-checking — acceptable for now as the host is validated and a
        # public host redirecting to a private one is an uncommon, lower-risk path.
        if not is_url_public(url):
            logger.warning("Skipping non-public URL '%s' (SSRF guard).", url)
            return None
        try:
            response = self.client.get(url)
            response.raise_for_status()
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
