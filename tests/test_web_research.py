"""Tests for web research primitives (no real network)."""

from __future__ import annotations

import httpx
import pytest

from scholar_lens.src.web_research import (
    BraveSearchProvider,
    NullSearchProvider,
    PageContent,
    ResearchCorpus,
    SearchResult,
    WebResearcher,
    _normalize_url,
)

DOCS_HTML = """
<html><head><title>FastAPI Docs</title></head>
<body>
  <h1>Getting Started</h1>
  <p>FastAPI is a modern web framework.</p>
  <img src="/img/logo.png">
  <img src="https://cdn.example.com/diagram.svg">
  <a href="/docs/tutorial/">Tutorial</a>
  <a href="https://external.com/other">External</a>
  <script>analytics();</script>
</body></html>
"""

SUBPAGE_HTML = """
<html><head><title>Tutorial</title></head>
<body><p>Step by step tutorial content.</p></body></html>
"""


def _transport(routes: dict[str, tuple[int, str, str]]) -> httpx.MockTransport:
    """Build a MockTransport from {url: (status, content_type, body)}."""

    def handler(request: httpx.Request) -> httpx.Response:
        key = str(request.url)
        if key in routes:
            status, content_type, body = routes[key]
            return httpx.Response(
                status, headers={"Content-Type": content_type}, text=body
            )
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


def _researcher(routes: dict[str, tuple[int, str, str]], **kwargs) -> WebResearcher:
    client = httpx.Client(transport=_transport(routes))
    return WebResearcher(client=client, **kwargs)


class TestNormalizeUrl:
    def test_strips_scheme_case_and_trailing_slash(self) -> None:
        assert _normalize_url("HTTPS://Ex.com/Path/") == "https://ex.com/path"

    def test_drops_query_and_fragment(self) -> None:
        assert _normalize_url("https://x.com/a?b=1#c") == "https://x.com/a"


class TestNullSearchProvider:
    def test_returns_empty(self) -> None:
        assert NullSearchProvider().search("anything") == []


class TestResearchCorpus:
    def test_image_urls_are_unique_and_ordered(self) -> None:
        corpus = ResearchCorpus(
            pages=[
                PageContent("u1", "t1", "x", image_urls=["a", "b"]),
                PageContent("u2", "t2", "y", image_urls=["b", "c"]),
            ]
        )
        assert corpus.image_urls == ["a", "b", "c"]

    def test_combined_text_truncates(self) -> None:
        corpus = ResearchCorpus(pages=[PageContent("u", "t", "x" * 1000)])
        assert len(corpus.combined_text(max_chars=100)) == 100


class TestWebResearcherFetch:
    def test_fetches_page_text_and_images(self) -> None:
        url = "https://fastapi.tiangolo.com/"
        researcher = _researcher({url: (200, "text/html", DOCS_HTML)})
        corpus = researcher.research([url], discover_subpages=False)
        assert len(corpus.pages) == 1
        page = corpus.pages[0]
        assert page.title == "FastAPI Docs"
        assert "modern web framework" in page.text
        assert "analytics" not in page.text  # script stripped
        # Relative image resolved to absolute; external kept.
        assert "https://fastapi.tiangolo.com/img/logo.png" in page.image_urls
        assert "https://cdn.example.com/diagram.svg" in page.image_urls

    def test_non_html_resource_skipped(self) -> None:
        url = "https://x.com/file.pdf"
        researcher = _researcher({url: (200, "application/pdf", "%PDF-1.4")})
        corpus = researcher.research([url], discover_subpages=False)
        assert corpus.pages == []

    def test_failed_fetch_is_skipped(self) -> None:
        researcher = _researcher({})  # everything 404s
        corpus = researcher.research(["https://x.com/missing"], discover_subpages=False)
        assert corpus.pages == []

    def test_subpage_discovery_in_scope_only(self) -> None:
        base = "https://fastapi.tiangolo.com/docs/"
        sub = "https://fastapi.tiangolo.com/docs/tutorial/"
        researcher = _researcher(
            {
                base: (200, "text/html", DOCS_HTML),
                sub: (200, "text/html", SUBPAGE_HTML),
            }
        )
        corpus = researcher.research([base], discover_subpages=True)
        urls = {p.url for p in corpus.pages}
        assert base in urls
        assert sub in urls
        # The external link must NOT be fetched/added.
        assert all("external.com" not in u for u in urls)


class TestBraveSearchProvider:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        with pytest.raises(ValueError):
            BraveSearchProvider(api_key=None)

    def test_parses_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "web": {
                "results": [
                    {"title": "T1", "url": "https://a.com", "description": "d1"},
                    {"title": "T2", "url": "https://b.com", "description": "d2"},
                    {"title": "NoUrl", "description": "skipped"},
                ]
            }
        }
        provider = BraveSearchProvider(api_key="key")

        def fake_get(url, **kwargs):  # type: ignore[no-untyped-def]
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", fake_get)
        results = provider.search("query")
        assert [r.url for r in results] == ["https://a.com", "https://b.com"]
        assert isinstance(results[0], SearchResult)

    def test_http_error_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = BraveSearchProvider(api_key="key")

        def boom(url, **kwargs):  # type: ignore[no-untyped-def]
            raise httpx.ConnectError("down")

        monkeypatch.setattr(httpx, "get", boom)
        assert provider.search("q") == []
