"""Search ranking, the third-party cap, demoted domain classes and search
exclusions (docs/engine.md 3.1; synthetic URLs, no network)."""

from __future__ import annotations

import json
from collections.abc import Collection

import httpx
import pytest

from scholar_lens.adapters.web.research import (
    BraveSearchProvider,
    FallbackSearchProvider,
    PageContent,
    ResearchCorpus,
    SearchResult,
    TavilySearchProvider,
    WebResearcher,
    WebSearchProvider,
)
from scholar_lens.domain.sources import DomainClasses, SourceSet, search_exclusions

DOCS = "https://docs.example.dev/en/latest/"
DEMOTED = DomainClasses.from_config(
    {
        "forum": ["qa.example.net", "discuss.*"],
        "social": ["*.social.example"],
        "aggregator": ["links.example.org"],
    }
)


class _Search(WebSearchProvider):
    def __init__(self, per_query: dict[str, list[SearchResult]]) -> None:
        self.per_query = per_query
        self.excluded: list[tuple[str, ...]] = []

    def search(
        self,
        query: str,
        *,
        count: int = 5,
        domains: Collection[str] = (),
        exclude: Collection[str] = (),
    ) -> list[SearchResult]:
        self.excluded.append(tuple(exclude))
        return list(self.per_query.get(query, []))


def _result(url: str, published: str = "") -> SearchResult:
    return SearchResult(
        title=url, url=url, content=f"body of {url}", published=published
    )


def _run(
    per_query: dict[str, list[SearchResult]],
    *,
    fetch_top: int = 10,
    max_third_party: int | None = None,
    exclude: Collection[str] = (),
) -> tuple[list[PageContent], _Search]:
    search = _Search(per_query)
    researcher = WebResearcher(search)
    corpus = ResearchCorpus()
    sources = SourceSet.from_urls([DOCS], demoted=DEMOTED)
    researcher.run_searches(
        corpus,
        list(per_query),
        fetch_top=fetch_top,
        sources=sources,
        max_third_party=max_third_party,
        exclude=exclude,
    )
    return corpus.pages, search


def test_results_rank_by_authority_then_recency_then_search_rank() -> None:
    pages, _ = _run(
        {
            "q1": [
                _result("https://blog-a.example.org/old", "2021-05-01"),
                _result("https://qa.example.net/questions/1", "2026-01-01"),
                _result("https://blog-b.example.org/undated"),
            ],
            "q2": [
                _result("https://blog-c.example.org/new", "2025-09-09"),
                _result("https://docs.example.dev/en/latest/guide"),
                _result("https://docs.example.dev/en/v0.1.0/guide"),
            ],
        }
    )
    assert [p.url for p in pages] == [
        "https://docs.example.dev/en/latest/guide",  # official
        "https://blog-c.example.org/new",  # third-party, newest year
        "https://blog-a.example.org/old",
        "https://blog-b.example.org/undated",  # undated after dated
        "https://docs.example.dev/en/v0.1.0/guide",  # another docs version
        "https://qa.example.net/questions/1",  # demoted class, last
    ]
    assert [p.authority for p in pages] == [
        "official",
        "third_party",
        "third_party",
        "third_party",
        "older_docs",
        "forum",
    ]
    assert pages[1].published == "2025-09-09"


def test_equal_results_keep_the_round_robin_search_rank() -> None:
    pages, _ = _run(
        {
            "q1": [_result(f"https://a{i}.example.org/") for i in range(3)],
            "q2": [_result(f"https://b{i}.example.org/") for i in range(3)],
        },
        fetch_top=4,
    )
    assert [p.url for p in pages] == [
        "https://a0.example.org/",
        "https://b0.example.org/",
        "https://a1.example.org/",
        "https://b1.example.org/",
    ]


def test_third_party_pages_are_capped_and_leave_room_for_the_docs() -> None:
    pages, _ = _run(
        {
            "q": [
                *(_result(f"https://blog{i}.example.org/p") for i in range(5)),
                _result("https://discuss.example.com/t/1"),
                _result("https://docs.example.dev/en/latest/a"),
                _result("https://docs.example.dev/en/latest/b"),
            ]
        },
        fetch_top=6,
        max_third_party=2,
    )
    authorities = [p.authority for p in pages]
    assert authorities == ["official", "official", "third_party", "third_party"]


def test_a_result_that_lands_on_a_third_party_page_counts_against_the_cap() -> None:
    search = _Search(
        {
            "q": [
                SearchResult("a", "https://blog1.example.org/a", content="a"),
                SearchResult("b", "https://docs.example.dev/en/latest/moved"),
            ]
        }
    )
    researcher = WebResearcher(search)
    researcher._fetch_page = lambda url: PageContent(  # type: ignore[method-assign]
        url=url, title="", text="moved", final_url="https://blog2.example.org/b"
    )
    corpus = ResearchCorpus()
    sources = SourceSet.from_urls([DOCS])
    researcher.run_searches(
        corpus, ["q"], fetch_top=5, sources=sources, max_third_party=1
    )
    # The docs link ranks first but lands on a blog: that page fills the cap
    # and the other blog is left out.
    assert [(p.url, p.authority) for p in corpus.pages] == [
        ("https://docs.example.dev/en/latest/moved", "third_party")
    ]


@pytest.mark.parametrize(
    "url,cls",
    [
        ("https://qa.example.net/x", "forum"),
        ("https://sub.qa.example.net/x", "forum"),
        ("https://discuss.example.com/t/1", "forum"),
        ("https://www.social.example/p", "social"),
        ("https://social.example/p", "social"),
        ("https://links.example.org/top", "aggregator"),
        ("https://notqa.example.net/x", ""),
        ("https://blog.example.org/discuss/x", ""),
    ],
)
def test_demoted_classes_come_from_configured_domain_patterns(
    url: str, cls: str
) -> None:
    assert DEMOTED.match(url) == cls
    record = SourceSet.from_urls([DOCS], demoted=DEMOTED).classify(url)
    assert record.authority == ("forum" if cls else "third_party")


def test_nothing_is_demoted_without_configured_classes() -> None:
    record = SourceSet.from_urls([DOCS]).classify("https://qa.example.net/x")
    assert record.authority == "third_party"


def test_the_users_reference_material_is_never_demoted() -> None:
    sources = SourceSet.from_urls([DOCS], demoted=DEMOTED)
    record = sources.classify("https://qa.example.net/x", user_reference=True)
    assert record.authority == "reference"


def test_excluded_domains_reach_the_search_backends() -> None:
    assert search_exclusions(["*.a.example", "b.example", "forum.*", " "]) == [
        "a.example",
        "b.example",
    ]
    _, search = _run({"q": []}, exclude=["a.example"])
    assert search.excluded == [("a.example",)]

    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.host == "api.tavily.com":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(200, json={"web": {"results": []}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    TavilySearchProvider(api_key="k", client=client).search("q", exclude=["a.example"])
    BraveSearchProvider(api_key="k", client=client).search("q", exclude=["a.example"])
    assert json.loads(sent[0].content)["exclude_domains"] == ["a.example"]
    assert sent[1].url.params["q"] == "q -site:a.example"
    # A provider without the parameter still works when nothing is excluded.
    fallback = FallbackSearchProvider(_Search({}), _Search({}))
    assert fallback.search("q") == []


def test_backends_date_their_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.tavily.com":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://a.example/x",
                            "content": "c",
                            "published_date": "Tue, 2025-03-04",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "url": "https://b.example/y",
                            "page_age": "2024-02-01T00:00:00",
                        }
                    ]
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    (tavily,) = TavilySearchProvider(api_key="k", client=client).search("q")
    (brave,) = BraveSearchProvider(api_key="k", client=client).search("q")
    assert (tavily.published, brave.published) == ("2025-03-04", "2024-02-01")
