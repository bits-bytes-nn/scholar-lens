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
from collections.abc import Collection
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag, UnicodeDammit

from .constants import EnvVars
from .logger import logger
from .pinned_transport import PinnedHTTPTransport
from .prompt_safety import neutralize_prompt_tags
from .url_guard import UnsafeUrlError, is_url_public
from .utils import extract_text_from_html

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
_META_REFRESH_URL = re.compile(
    r"^\s*\d+(?:\.\d*)?\s*[;,]\s*(?:url\s*=\s*)?['\"]?([^'\"\s]+)", re.I
)
_BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_TAVILY_SEARCH_URL = "https://api.tavily.com/search"


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
    # Where the page actually lives after HTTP and <meta refresh> redirects
    # (sub-page scope is taken from it); empty when it is ``url`` itself.
    final_url: str = ""


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
        official_hosts: Collection[str] | None = None,
    ) -> str:
        """The pages most relevant to ``query`` (word overlap with title and
        body), most relevant first, within ``max_chars``. Each section of a
        guide reads only the material it needs instead of the whole corpus.
        With ``official_hosts``, each source is labelled official or
        third-party so the writer knows which one decides a flag or default."""
        words = {w for w in re.findall(r"[a-z0-9][a-z0-9_.-]{2,}", query.lower())}

        def score(page: PageContent) -> float:
            title = page.title.lower()
            body = page.text.lower()
            hits = sum(3 * (w in title) + min(body.count(w), 20) for w in words)
            return hits / (1 + len(body) / 20_000)  # don't favour sheer length

        ranked = sorted(self.pages, key=score, reverse=True)
        blocks, used = [], 0
        for page in ranked:
            label = ""
            if official_hosts is not None:
                host = urlparse(page.final_url or page.url).hostname or ""
                label = (
                    " (official documentation)"
                    if host in official_hosts
                    else " (third-party)"
                )
            block = (
                f"## Source{label}: {neutralize_prompt_tags(page.url)}\n"
                f"# {neutralize_prompt_tags(page.title)}\n\n"
                f"{neutralize_prompt_tags(page.text)}"
            )
            room = max_chars - used
            if room <= 2_000:
                break
            blocks.append(block[:room])
            used += len(blocks[-1])
        return "\n\n---\n\n".join(blocks)

    def combined_text(self, max_chars: int | None = None) -> str:
        # URL and title are page-controlled too, so they are defanged as well.
        blocks = [
            f"## Source: {neutralize_prompt_tags(p.url)}\n"
            f"# {neutralize_prompt_tags(p.title)}\n\n{neutralize_prompt_tags(p.text)}"
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
            visited.add(normalize_url(url))

            if discover_subpages:
                for sub_url in self._select_subpages(page, page.final_url or url):
                    if normalize_url(sub_url) in visited:
                        continue
                    sub_page = self._fetch_page(sub_url)
                    if sub_page is not None:
                        corpus.pages.append(sub_page)
                        visited.add(normalize_url(sub_url))

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

        visited = {normalize_url(p.url) for p in corpus.pages}
        added = 0
        for result in corpus.search_results:
            if added >= fetch_top:
                break
            norm = normalize_url(result.url)
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
            fetched = self._get_html(url)
        except UnsafeUrlError as e:
            logger.warning(
                "Skipping '%s' — redirected to non-public target: %s", url, e
            )
            return None
        except httpx.HTTPError as e:
            logger.warning("Failed to fetch '%s': %s", url, e)
            return None
        if fetched is None:
            return None
        html, final_url = fetched

        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else url
        # Relative links and image markers resolve against where the page
        # actually lives (after redirects), not the URL we asked for.
        text = extract_text_from_html(html, base_url=final_url)
        image_urls = self._extract_images(soup, final_url)
        links = self._extract_links(soup, final_url)
        return PageContent(
            url=url,
            title=title,
            text=text,
            image_urls=image_urls,
            links=links,
            final_url=final_url if final_url != url else "",
        )

    def _get_html(self, url: str) -> tuple[str, str] | None:
        """Fetch ``url`` like :meth:`_get_page`, also following a small
        stub page's ``<meta http-equiv="refresh">`` (docs sites such as
        ``/stable/`` aliases use it instead of an HTTP redirect) for at most
        :data:`MAX_META_REFRESH_HOPS` hops. Every hop goes through the same
        pinned, SSRF-guarded client."""
        fetched = self._get_page(url)
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

    def _get_page(self, url: str) -> tuple[str, str] | None:
        """Stream ``url``; return (decoded HTML, final URL), or None when the
        resource is not HTML or exceeds :data:`MAX_PAGE_BYTES`."""
        with self.client.stream("GET", url) as response:
            response.raise_for_status()
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
        """In-scope sub-URLs: same host and under the seed's path, deduped."""
        base = urlparse(base_url)
        scope = _scope_prefix(base.path)
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


def normalize_url(url: str) -> str:
    """Dedup key: scheme and host are case-insensitive, the path is not."""
    parsed = urlparse(url)
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}".rstrip("/")
