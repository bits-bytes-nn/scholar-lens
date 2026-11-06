"""Official-reference lookup and verification hedges for tech guides (no
LLM or network calls)."""

from __future__ import annotations

from collections.abc import Collection
from unittest.mock import MagicMock

import pytest

from scholar_lens.domain.markdown.lint import writing_issues
from scholar_lens.src.web_research import (
    MAX_REFERENCE_PAGES,
    MAX_REFERENCE_SEARCHES,
    OfficialDocs,
    PageContent,
    ResearchCorpus,
    SearchResult,
    WebResearcher,
    WebSearchProvider,
)
from tests.test_tech_guide import _make_generator

HOST = "docs.example.ai"
QUICKSTART = f"https://{HOST}/en/latest/quickstart.html"
ENGINE_ARGS = f"https://{HOST}/en/latest/configuration/engine_args.html"
OFFICIAL = OfficialDocs.from_urls([QUICKSTART])


def _hedges(text: str) -> list[str]:
    return [i for i in writing_issues(text) if i.startswith("verification hedge")]


@pytest.mark.parametrize(
    "text",
    [
        "이 플래그는 서드파티 자료로만 확인된 내용입니다(미검증).",
        "두 플래그 모두 제공된 공식 문서에서는 확인되지 않았습니다.",
        "연결 방법은 참고 자료에 나와 있지 않습니다.",
        "The flag is (unverified) in this release.",
    ],
)
def test_reader_facing_verification_hedges_are_flagged(text: str) -> None:
    assert _hedges(text)


@pytest.mark.parametrize(
    "text",
    [
        "## 참고 자료\n\n- [공식 문서](https://docs.example.ai)",
        "전체 옵션 목록은 공식 문서의 엔진 인자 페이지에 정리되어 있습니다.",
        "`--verified-only` 플래그로 검증된 결과만 남깁니다.",
    ],
)
def test_ordinary_prose_is_not_a_hedge(text: str) -> None:
    assert not _hedges(text)


class _FakeSearch(WebSearchProvider):
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def search(
        self, query: str, *, count: int = 5, domains: Collection[str] = ()
    ) -> list[SearchResult]:
        self.calls.append((query, tuple(domains)))
        return list(self.results)


def _researcher(results: list[SearchResult]) -> tuple[WebResearcher, _FakeSearch]:
    search = _FakeSearch(results)
    researcher = WebResearcher(search)
    fetched: list[str] = []

    def fetch(url: str) -> PageContent:
        fetched.append(url)
        return PageContent(url=url, title=url, text=f"page {url}")

    researcher._fetch_page = fetch  # type: ignore[method-assign]
    researcher.fetched = fetched  # type: ignore[attr-defined]
    return researcher, search


def test_reference_lookup_is_capped_and_stays_on_the_official_host() -> None:
    results = [
        SearchResult(title="args", url=ENGINE_ARGS, content="--speculative-config"),
        SearchResult(title="blog", url="https://blog.example.com/vllm-tuning"),
        *[
            SearchResult(title=f"p{i}", url=f"https://{HOST}/en/latest/p{i}.html")
            for i in range(10)
        ],
    ]
    researcher, search = _researcher(results)
    corpus = ResearchCorpus(
        pages=[
            PageContent(
                url=QUICKSTART,
                title="Quickstart",
                text="vllm serve",
                links=[f"https://{HOST}/en/latest/usage/metrics.html"],
            )
        ]
    )
    ids = [f"--flag-{i}" for i in range(9)]
    added = researcher.add_official_reference(corpus, OFFICIAL, ids, product="vllm")
    assert len(search.calls) == MAX_REFERENCE_SEARCHES
    assert all(domains == (HOST,) for _, domains in search.calls)
    assert len(added) == MAX_REFERENCE_PAGES
    assert added[0].url == ENGINE_ARGS and "--speculative-config" in added[0].text
    assert all(HOST in p.url for p in added)
    assert corpus.pages[-len(added) :] == added


def test_reference_lookup_uses_reference_links_without_search() -> None:
    researcher = WebResearcher()  # no search backend
    researcher._fetch_page = lambda url: PageContent(  # type: ignore[method-assign]
        url=url, title=url, text="engine args"
    )
    corpus = ResearchCorpus(
        pages=[
            PageContent(
                url=QUICKSTART,
                title="Quickstart",
                text="vllm serve",
                links=[ENGINE_ARGS, f"https://{HOST}/en/latest/blog/news.html"],
            )
        ]
    )
    added = researcher.add_official_reference(corpus, OFFICIAL, ["--x"], product="vllm")
    assert [p.url for p in added] == [ENGINE_ARGS]


def test_reference_lookup_skips_an_older_docs_version() -> None:
    old = f"https://{HOST}/en/v0.6.0/configuration/engine_args.html"
    researcher, _ = _researcher(
        [
            SearchResult(title="old args", url=old, content="--preemption-mode"),
            SearchResult(title="args", url=ENGINE_ARGS, content="--speculative-config"),
        ]
    )
    corpus = ResearchCorpus(
        pages=[PageContent(url=QUICKSTART, title="Quickstart", text="vllm serve")]
    )
    added = researcher.add_official_reference(corpus, OFFICIAL, ["--x"], product="vllm")
    assert [p.url for p in added] == [ENGINE_ARGS]


def test_python_arguments_count_as_identifiers() -> None:
    from scholar_lens.src.guide_authority import unverified_identifiers

    official = "Pass --gpu-memory-utilization to vllm serve."
    body = (
        "```python\nfrom vllm import LLM\n"
        "llm = LLM(model='m', gpu_memory_utilization=0.8, max_num_seqs=64)\n```\n\n"
        "Set `max_model_len=4096` too; `json.dumps(x, ensure_ascii=False)` too."
    )
    found = unverified_identifiers(body, official, ["vllm"])
    assert "gpu_memory_utilization" not in found  # its flag spelling is official
    assert "max_num_seqs" in found and "max_model_len" in found
    assert "ensure_ascii" not in found  # inline code that is not a bare argument


def test_third_party_flags_seed_the_reference_lookup() -> None:
    from scholar_lens.src.guide_authority import third_party_identifiers

    corpus = ResearchCorpus(
        pages=[
            PageContent(
                QUICKSTART,
                "Q",
                "```bash\nvllm serve m --port 8000\nvllm serve m --host h\n```",
            ),
            PageContent(
                "https://blog.example.com/tuning",
                "Blog",
                "```bash\nvllm serve m --max-model-len 4096 --port 8000\n```",
            ),
        ]
    )
    assert third_party_identifiers(corpus, OFFICIAL, ["vllm"]) == ["--max-model-len"]


def test_reference_links_that_fit_the_identifiers_come_first() -> None:
    researcher = WebResearcher()  # no search backend
    researcher._fetch_page = lambda url: PageContent(  # type: ignore[method-assign]
        url=url, title=url, text="reference"
    )
    metrics = f"https://{HOST}/en/latest/usage/metrics.html"
    memory = f"https://{HOST}/en/latest/configuration/conserving_memory.html"
    corpus = ResearchCorpus(
        pages=[
            PageContent(
                url=QUICKSTART,
                title="Quickstart",
                text="vllm serve",
                links=[metrics, memory, ENGINE_ARGS],
            )
        ]
    )
    added = researcher.add_official_reference(
        corpus, OFFICIAL, ["--gpu-memory-utilization"], max_pages=2
    )
    assert [p.url for p in added] == [memory, ENGINE_ARGS]


def _pages_researcher(pages: dict[str, PageContent]) -> WebResearcher:
    """A researcher without search whose fetches return ``pages`` (by URL),
    recording each fetched URL in ``researcher.fetched``."""
    researcher = WebResearcher()
    fetched: list[str] = []

    def fetch(url: str) -> PageContent | None:
        fetched.append(url)
        return pages.get(url)

    researcher._fetch_page = fetch  # type: ignore[method-assign]
    researcher.fetched = fetched  # type: ignore[attr-defined]
    return researcher


def _settles(identifiers: list[str]):  # type: ignore[no-untyped-def]
    from scholar_lens.src.guide_authority import mentions_any

    return lambda text: mentions_any(text, identifiers)


def test_reference_lookup_follows_an_index_page_to_the_flags() -> None:
    serve_args = f"https://{HOST}/en/latest/configuration/serve_args.html"
    cli_serve = f"https://{HOST}/en/latest/cli/serve.html"
    old_cli = f"https://{HOST}/en/v0.6.0/cli/serve.html"
    pages = {
        serve_args: PageContent(
            serve_args,
            "Server Arguments",
            "See the CLI reference.",
            links=[old_cli, "https://blog.example.com/cli/x", cli_serve],
        ),
        cli_serve: PageContent(
            cli_serve, "vllm serve", "--gpu-memory-utilization\n--kv-cache-dtype"
        ),
    }
    researcher = _pages_researcher(pages)
    corpus = ResearchCorpus(
        pages=[PageContent(QUICKSTART, "Quickstart", "vllm serve", links=[serve_args])]
    )
    ids = ["--gpu-memory-utilization", "--quantization"]
    added = researcher.add_official_reference(
        corpus, OFFICIAL, ids, settles=_settles(ids)
    )
    # The index page is left out; its in-scope link that has a flag counts.
    assert [p.url for p in added] == [cli_serve]
    assert researcher.fetched == [serve_args, cli_serve]  # type: ignore[attr-defined]
    assert [p.url for p in corpus.pages] == [QUICKSTART, cli_serve]


def test_reference_lookup_bounds_a_chain_of_index_pages() -> None:
    def url(i: int) -> str:
        return f"https://{HOST}/en/latest/reference/r{i}.html"

    # Each index page links the next; none mentions the flag.
    pages = {
        url(i): PageContent(url(i), f"r{i}", "An index.", links=[url(i + 1)])
        for i in range(50)
    }
    researcher = _pages_researcher(pages)
    corpus = ResearchCorpus(
        pages=[PageContent(QUICKSTART, "Quickstart", "vllm serve", links=[url(0)])]
    )
    added = researcher.add_official_reference(
        corpus, OFFICIAL, ["--x-y"], max_pages=3, settles=_settles(["--x-y"])
    )
    assert added == [] and len(corpus.pages) == 1
    assert len(researcher.fetched) == 2 * 3  # type: ignore[attr-defined]


def test_reference_lookup_drops_pages_that_settle_nothing() -> None:
    hit = f"https://{HOST}/en/latest/cli/serve.html"
    miss = f"https://{HOST}/en/latest/configuration/serve_args.html"
    researcher, _ = _researcher(
        [
            SearchResult(title="index", url=miss, content="See the CLI pages."),
            SearchResult(title="cli", url=hit, content="gpu_memory_utilization: 0.9"),
        ]
    )
    corpus = ResearchCorpus(pages=[PageContent(QUICKSTART, "Q", "vllm serve")])
    ids = ["--gpu-memory-utilization"]
    added = researcher.add_official_reference(
        corpus, OFFICIAL, ids, settles=_settles(ids)
    )
    # The API-argument spelling settles the flag, as the unverified check reads it.
    assert [p.url for p in added] == [hit]


async def test_an_identifier_lookup_requires_a_settling_page() -> None:
    gen, corpus = _budget_generator([])
    lookup = MagicMock(return_value=[])
    gen.researcher.add_official_reference = lookup
    await gen._add_reference(corpus, ["--max-model-len"])
    settles = lookup.call_args.kwargs["settles"]
    assert settles("max_model_len=4096") and not settles("--max-model-length")


async def test_reference_lookups_share_the_per_guide_caps() -> None:
    gen = _make_generator()
    gen._reference_pages, gen._reference_ids = [], []
    gen._official = OFFICIAL
    calls: list[dict] = []

    def add_reference(corpus, official, identifiers, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"ids": identifiers, **kwargs})
        return [PageContent(url=f"{ENGINE_ARGS}?{len(calls)}", title="", text="")] * 4

    gen.researcher.add_official_reference = MagicMock(side_effect=add_reference)
    gen.researcher.search_provider.supports_search = True
    corpus = ResearchCorpus()
    # The pre-writing lookup stops short of the caps, leaving grounding room.
    await gen._add_reference(corpus, ["--a"], max_pages=4, max_searches=2)
    await gen._add_reference(corpus, ["--z"], max_pages=4, max_searches=2)
    await gen._add_reference(corpus, ["--a", "--b"])
    await gen._add_reference(corpus, ["--c"])  # page cap reached: no lookup
    assert [c["ids"] for c in calls] == [["--a"], ["--b"]]
    assert (calls[0]["max_pages"], calls[0]["max_searches"]) == (4, 2)
    assert calls[1]["max_pages"] == MAX_REFERENCE_PAGES - 4
    assert calls[1]["max_searches"] == MAX_REFERENCE_SEARCHES - 1


def test_the_identifier_lookup_leaves_part_of_the_budget() -> None:
    from scholar_lens.src import tech_guide

    assert 0 < tech_guide._IDENTIFIER_REFERENCE_PAGES < MAX_REFERENCE_PAGES
    assert 0 < tech_guide._IDENTIFIER_REFERENCE_SEARCHES < MAX_REFERENCE_SEARCHES


def _budget_generator(search: list[SearchResult] | None):  # type: ignore[no-untyped-def]
    """A generator whose researcher has the given search results (None: no
    search backend) and a seed page linking reference-like pages."""
    from scholar_lens.src.web_research import NullSearchProvider

    links = [
        f"https://{HOST}/en/latest/{p}.html"
        for p in ("cli/serve", "configuration/engine_args", "usage/metrics")
        + ("configuration/env_vars", "api/reference", "serving/settings")
        + ("cli/bench", "configuration/optimization")
    ]
    seed = PageContent(QUICKSTART, "Quickstart", "vllm serve m", links=links)
    researcher = (
        WebResearcher(NullSearchProvider())
        if search is None
        else _researcher(search)[0]
    )
    # Every fetched page settles the flags these tests look up.
    researcher._fetch_page = lambda url: PageContent(  # type: ignore[method-assign]
        url, url, "--gpu-memory-utilization --max-model-len --tensor-parallel-size"
    )
    gen = _make_generator()
    gen.researcher = researcher
    gen._official = OFFICIAL
    gen._products = ("vllm",)
    gen._reference_pages, gen._reference_ids = [], []
    gen._reference_searches = 0
    return gen, ResearchCorpus(pages=[seed])


async def test_without_a_search_backend_the_lookup_follows_the_seed_links() -> None:
    gen, corpus = _budget_generator(None)
    await gen._add_reference(corpus, ["--gpu-memory-utilization"])
    assert gen._reference_pages
    assert all(p.url.startswith(f"https://{HOST}/") for p in gen._reference_pages)


async def test_the_identifier_lookup_stays_within_its_share() -> None:
    from scholar_lens.src.tech_guide import (
        _IDENTIFIER_REFERENCE_PAGES,
        _IDENTIFIER_REFERENCE_SEARCHES,
    )

    results = [
        SearchResult(title=f"p{i}", url=f"https://{HOST}/en/latest/p{i}.html")
        for i in range(10)
    ]
    gen, corpus = _budget_generator(results)
    # The pre-writing identifier lookup, as gather() runs it.
    await gen._add_reference(
        corpus,
        ["--max-model-len", "--tensor-parallel-size"],
        max_pages=_IDENTIFIER_REFERENCE_PAGES,
        max_searches=_IDENTIFIER_REFERENCE_SEARCHES,
    )
    assert len(gen._reference_pages) == _IDENTIFIER_REFERENCE_PAGES
    # A later lookup may use the rest of the per-guide caps.
    await gen._add_reference(corpus, ["--gpu-memory-utilization"])
    assert len(gen._reference_pages) == MAX_REFERENCE_PAGES


def test_arguments_the_guides_own_code_defines_are_not_unverified() -> None:
    from scholar_lens.src.guide_authority import unverified_identifiers

    body = (
        "```python\nfrom vllm import LLM\n\n"
        "def book(user_id: str, room_size: int = 2) -> str:\n    return user_id\n\n"
        "llm = LLM(model='m', gpu_memory_utilization=0.9)\n"
        "book(user_id='u', room_size=3)\nLLM(made_up_flag=1)\n```\n"
    )
    official = "LLM(model, gpu_memory_utilization)"
    assert unverified_identifiers(body, official, ["vllm"]) == ["made_up_flag"]
