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
import socket
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag, UnicodeDammit

from .constants import EnvVars
from .logger import logger
from .markdown_lint import close_open_fence
from .markdown_math import segments
from .pinned_transport import PinnedHTTPTransport
from .prompt_safety import neutralize_prompt_tags
from .url_guard import (
    UnsafeUrlError,
    assert_url_is_public,
    is_temporary_dns_failure,
    is_url_public,
)
from .utils import extract_text_from_html
from .utils.helpers import is_page_furniture

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
# A docs version in a URL path: a rolling alias or a pinned release such as
# "v0.6.0", "0.6" or "v2" (a bare number like a year is not a version). Aliases
# and dotted releases count at any depth (/deeplearning/nemo/user-guide/docs/en/
# v1.23.0/); a bare "vN" only right after the root or a language, since deeper
# it is an API version ("/reference/rest/v1/"), not a docs release.
_ROLLING_DOCS_VERSIONS = frozenset({"latest", "stable", "main", "dev", "nightly"})
_DOTTED_DOCS_VERSION = re.compile(r"v?\d+(?:\.\d+)+", re.I)
_BARE_DOCS_VERSION = re.compile(r"v\d+", re.I)
_BARE_VERSION_SEGMENTS = 2  # "v2" only right after the root or a language
# A product named by a docs path: huggingface.co/docs/<product>/…
_DOCS_PRODUCT_PATH = re.compile(r"^/docs/([a-z][a-z0-9_-]*)/", re.I)
# The command of a shell line in fenced code that takes long options
# ("vllm serve --x"). Lines are split at "&&", "||", ";" and "|"; "\\" line
# continuations are joined; environment assignments, options, numbers and
# generic tools or launchers (sudo, timeout, srun, bash …) are skipped, and
# "kubectl exec pod -- vllm …" reads what follows the "--".
_COMMAND_SEPARATOR = re.compile(r"&&|\|\||;|\|")
_ENV_ASSIGNMENT = re.compile(
    r"(?<![\w-])[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"\n]*\"|'[^'\n]*'|\S*)"
)
_PROMPT = re.compile(r"^\s*(?:\([\w.-]+\)\s*)?\$\s")
_COMMAND_NAME = re.compile(r"[a-z][a-z0-9_-]{1,30}", re.I)
_LONG_OPTION = re.compile(r"\s--[a-z]", re.I)
_LINE_CONTINUATION = re.compile(r"\\\r?\n\s*")
# Tools whose subcommands are not products ("pip install", "git clone").
_GENERIC_COMMANDS = frozenset(
    {
        "apt", "apt-get", "brew", "cat", "conda", "curl", "docker", "echo",
        "export", "git", "helm", "kubectl", "npm", "pip", "pip3", "printf",
        "python", "python3", "uv", "wget",
    }
)  # fmt: skip
# Launchers and shells: what they run is the command.
_LAUNCHERS = frozenset(
    {
        "bash", "cd", "env", "exec", "nice", "nohup", "numactl", "sh", "source",
        "srun", "stdbuf", "sudo", "taskset", "time", "timeout", "watch", "xargs",
        "zsh",
    }
)  # fmt: skip


def _command_of(part: str) -> str | None:
    """The tool a shell command line runs, or None (a generic tool's own
    subcommand is not one)."""
    rest = _PROMPT.sub("", part)
    if " -- " in rest:
        rest = rest.rsplit(" -- ", 1)[1]
    for token in _ENV_ASSIGNMENT.sub(" ", rest).split():
        word = token.strip("'\"")
        if word[:1] == "-" or word.isdigit() or word.lower() in _LAUNCHERS:
            continue
        if word.lower() in _GENERIC_COMMANDS or not _COMMAND_NAME.fullmatch(word):
            return None
        return word
    return None


def _commands_with_flags(code: str) -> list[str]:
    """The tools the shell lines in ``code`` that take long options run."""
    found: list[str] = []
    for line in _LINE_CONTINUATION.sub(" ", code).splitlines():
        for part in _COMMAND_SEPARATOR.split(line):
            if _LONG_OPTION.search(part) and (command := _command_of(part)):
                found.append(command.lower())
    return found


_FENCED_CODE = re.compile(r"^```[^\n]*\n(.*?)^```", re.S | re.M)
# Host labels that do not name a product (docs.vllm.ai → vllm).
HOST_NOISE = frozenset({"docs", "www", "doc", "api", "readthedocs", "github"})
# Images that are page furniture, not content: tracking pixels and chrome.
_BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def docs_version(url: str) -> str | None:
    """The first path segment that names a docs version (``latest``,
    ``stable``, ``v0.6.0``, ``0.6`` …), lower-cased; ``None`` when there is none."""
    segments = [s for s in urlparse(url).path.split("/") if s]
    for index, segment in enumerate(segments):
        lowered = segment.lower()
        if lowered in _ROLLING_DOCS_VERSIONS or _DOTTED_DOCS_VERSION.fullmatch(lowered):
            return lowered
        if index < _BARE_VERSION_SEGMENTS and _BARE_DOCS_VERSION.fullmatch(lowered):
            return lowered
    return None


def _is_current(version: str | None) -> bool:
    return version is None or version in _ROLLING_DOCS_VERSIONS


@dataclass(frozen=True)
class OfficialDocs:
    """The documentation the user asked for: its hosts and docs versions. A page
    on those hosts is official only for a compatible version, so an old pinned
    release (``/en/v0.6.0/``) does not override the current docs."""

    hosts: frozenset[str]
    versions: frozenset[str | None]

    @classmethod
    def from_urls(cls, urls: Collection[str]) -> OfficialDocs:
        return cls(
            hosts=frozenset(h for u in urls if (h := urlparse(u).hostname)),
            versions=frozenset(docs_version(u) for u in urls),
        )

    @classmethod
    def from_research(
        cls, urls: Collection[str], pages: Collection[PageContent]
    ) -> OfficialDocs:
        """Like :meth:`from_urls`, plus where the requested pages actually
        landed: a docs site that redirects (``pytorch.org/docs/stable`` →
        ``docs.pytorch.org/docs/2.14``) keeps its own pages official, and the
        pinned release a rolling or unversioned link resolves to is current."""
        base = cls.from_urls(urls)
        requested = {normalize_url(u) for u in urls}
        hosts, versions = set(base.hosts), set(base.versions)
        for page in pages:
            if not page.final_url or normalize_url(page.url) not in requested:
                continue
            if host := urlparse(page.final_url).hostname:
                hosts.add(host)
            landed = docs_version(page.final_url)
            if _is_current(docs_version(page.url)) or landed in base.versions:
                versions.add(landed)
        return cls(hosts=frozenset(hosts), versions=frozenset(versions))

    def on_host(self, url: str) -> bool:
        return (urlparse(url).hostname or "") in self.hosts

    def covers(self, url: str) -> bool:
        if not self.on_host(url):
            return False
        version = docs_version(url)
        if version is None or version in self.versions:
            return True
        if version in _ROLLING_DOCS_VERSIONS:
            # Rolling aliases describe the current release, like an
            # unversioned input does.
            return not self.versions or any(_is_current(v) for v in self.versions)
        return False  # a pinned release none of the inputs asked for


def product_names(urls: Collection[str], pages: Collection[PageContent]) -> list[str]:
    """Names the requested docs' commands go by, most specific first: the
    product in a ``/docs/<product>/`` path (when the pages use that name), the command the requested pages'
    fenced code runs most often with long options (at least twice, so one
    stray example does not count), then the site's host label
    (``docs.vllm.ai`` → ``vllm``). Generic tools are skipped, except as the
    docs' own host (``helm.sh`` → ``helm``)."""
    names: list[str] = []

    def add(name: str, *, generic_ok: bool = False) -> bool:
        name = name.lower()
        if not name or name in names:
            return False
        if name in _GENERIC_COMMANDS and not generic_ok:
            return False
        names.append(name)
        return True

    requested = {normalize_url(u) for u in urls}
    own_text = " ".join(  # code only: a title says "Quick Reference" too
        " ".join(_FENCED_CODE.findall(page.text)).lower()
        for page in pages
        if normalize_url(page.url) in requested
    )
    for url in urls:
        # A /docs/<x>/ segment names a product only when the pages' code uses it;
        # on single-product sites it is a section ("reference", "setup").
        if match := _DOCS_PRODUCT_PATH.match(urlparse(url).path):
            section = match.group(1).lower()
            if docs_version("/" + section) is None and re.search(
                rf"(?<![\w-]){re.escape(section)}(?![\w])", own_text
            ):
                add(section)
    counts = Counter(
        command
        for page in pages
        if normalize_url(page.url) in requested
        for block in _FENCED_CODE.findall(page.text)
        for command in _commands_with_flags(block)
    )
    if counts:
        command, count = counts.most_common(1)[0]
        if count >= 2:
            add(command)
    for url in urls:
        labels = (urlparse(url).hostname or "").split(".")[:-1]
        for label in labels:
            # The docs' own domain names its tool even when it is a generic one
            # elsewhere (helm.sh → helm), but not a subdomain (python.langchain.com).
            if label not in HOST_NOISE and add(label, generic_ok=label == labels[-1]):
                break
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
        official: OfficialDocs | None = None,
    ) -> str:
        """The pages most relevant to ``query`` (word overlap with title and
        body), most relevant first, within ``max_chars``. Each section of a
        guide reads only the material it needs instead of the whole corpus.
        With ``official``, each source is labelled official, older docs version
        or third-party so the writer knows which one decides a flag or default."""
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
            if official is not None:
                url = page.final_url or page.url
                if official.covers(url):
                    label = " (official documentation)"
                elif official.on_host(url):
                    label = " (older docs version)"
                else:
                    label = " (third-party)"
            block = (
                f"## Source{label}: {neutralize_prompt_tags(page.url)}\n"
                f"# {neutralize_prompt_tags(page.title)}\n\n"
                f"{neutralize_prompt_tags(page.text)}"
            )
            room = max_chars - used
            if room <= 2_000:
                break
            blocks.append(_cut(block, room))
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
    def search(
        self, query: str, *, count: int = 5, domains: Collection[str] = ()
    ) -> list[SearchResult]:
        """Results for ``query``; ``domains`` restricts them to those hosts."""


class NullSearchProvider(WebSearchProvider):
    """No-op provider: research relies solely on the supplied URLs."""

    supports_search: bool = False

    def search(
        self, query: str, *, count: int = 5, domains: Collection[str] = ()
    ) -> list[SearchResult]:
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
        visited = {normalize_url(p.url) for p in corpus.pages}
        added = 0
        for result in interleaved:
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

    def add_official_reference(
        self,
        corpus: ResearchCorpus,
        official: OfficialDocs,
        identifiers: list[str],
        *,
        product: str = "",
    ) -> list[PageContent]:
        """Fetch reference pages of the ``official`` docs that can settle the
        given flags and metric names, add them to ``corpus`` and return them.

        Candidates are search results restricted to those hosts (one query per
        batch of identifiers, at most :data:`MAX_REFERENCE_SEARCHES`), then
        reference-like links (CLI, arguments, configuration, metrics …) of the
        official pages already fetched. At most :data:`MAX_REFERENCE_PAGES` are
        added, each through the same SSRF-guarded fetch.
        """
        hosts = sorted(official.hosts)
        if not hosts or not identifiers:
            return []
        visited = {normalize_url(p.final_url or p.url) for p in corpus.pages}
        visited |= {normalize_url(p.url) for p in corpus.pages}
        candidates: list[SearchResult] = []
        if self.search_provider.supports_search:
            searches = min(MAX_REFERENCE_SEARCHES, len(identifiers))
            size = -(-len(identifiers) // searches)
            for i in range(searches):
                batch = " ".join(identifiers[i * size : (i + 1) * size])
                query = f"{product} {batch}".strip()
                candidates += self.search_provider.search(query, domains=hosts)
        for seen in corpus.pages:
            if not official.covers(seen.final_url or seen.url):
                continue
            candidates += [
                SearchResult(title="", url=link)
                for link in seen.links
                if official.covers(link) and _REFERENCE_LINK.search(urlparse(link).path)
            ]
        added: list[PageContent] = []
        for result in candidates:
            if len(added) >= MAX_REFERENCE_PAGES:
                break
            norm = normalize_url(result.url)
            if not norm or norm in visited:
                continue
            if not official.covers(result.url):
                continue  # off-host (a backend ignoring the filter) or an old release
            visited.add(norm)
            page: PageContent | None
            if result.content:
                page = PageContent(
                    url=result.url,
                    title=result.title or result.url,
                    text=result.content,
                )
            else:
                page = self._fetch_page(result.url)
            if page is not None and official.covers(page.final_url or page.url):
                added.append(page)
        corpus.pages.extend(added)
        logger.info(
            "Added %d official reference page(s) for %d identifier(s).",
            len(added),
            len(identifiers),
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
                    "Skipping '%s' — it or a redirect targets a non-public "
                    "address: %s",
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
        text = extract_text_from_html(
            _main_content(soup) or html, base_url=final_url, skip_page_furniture=True
        )
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
        """Content images, deduped: tracking pixels and the logos in page
        chrome (header, navigation, footer, ``<noscript>``) are skipped."""
        images: list[str] = []
        for img in soup.find_all("img"):
            if not isinstance(img, Tag) or is_page_furniture(img):
                continue
            src = img.get("src") or img.get("data-src")
            if isinstance(src, str) and src.strip():
                absolute = urljoin(base_url, src.strip())
                if (
                    absolute.lower().startswith(("http://", "https://"))
                    and absolute not in images
                ):
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
        """In-scope sub-URLs: same host and under the seed's path, deduped. A
        leaf page without an extension (``/docs/peft/guides/lora``) has no
        children, so its siblings are in scope instead."""
        base = urlparse(base_url)
        scope = _scope_prefix(base.path)
        selected = self._links_under(page, base_url, scope)
        parent = scope.rstrip("/").rsplit("/", 1)[0] + "/"
        if not selected and parent.count("/") > 2:
            selected = self._links_under(page, base_url, parent)
        return selected

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


def normalize_url(url: str) -> str:
    """Dedup key: scheme and host are case-insensitive, the path is not."""
    parsed = urlparse(url)
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}".rstrip("/")
