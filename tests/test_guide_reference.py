"""Official-reference lookup, verification hedges and the input outline for
tech guides (no LLM or network calls)."""

from __future__ import annotations

from collections.abc import Collection
from unittest.mock import AsyncMock, MagicMock

import pytest

from scholar_lens.src.markdown_lint import writing_issues
from scholar_lens.src.tech_guide import input_outline
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


async def test_stale_flag_pulls_the_official_reference_into_grounding() -> None:
    gen = _make_generator(verify_grounding=True)
    quickstart = PageContent(
        url=QUICKSTART, title="Quickstart", text="## Installation\n\nvllm serve MODEL"
    )
    corpus = ResearchCorpus(pages=[quickstart])
    gen.researcher.research = MagicMock(return_value=corpus)
    engine_args = PageContent(
        url=ENGINE_ARGS,
        title="Engine Arguments",
        text="--speculative-config JSON speculative decoding settings",
    )

    def add_reference(corpus, hosts, identifiers, *, product=""):  # type: ignore[no-untyped-def]
        assert "--preemption-mode" in identifiers and product == "example"
        corpus.pages.append(engine_args)
        return [engine_args]

    gen.researcher.add_official_reference = MagicMock(side_effect=add_reference)
    gen.relevance_chain.ainvoke = AsyncMock(
        return_value={"is_relevant": "yes", "topic": "Using X", "reason": "docs"}
    )
    gen.synopsis_chain.ainvoke = AsyncMock(
        return_value={"synopsis": "1. Serving - run the server"}
    )
    gen.section_chain.ainvoke = AsyncMock(
        return_value={
            "section_markdown": "## Serving\n\n```bash\nexample serve m --preemption-mode swap\n```"
        }
    )
    gen.grounding_chain.ainvoke = AsyncMock(
        return_value={
            "grounded_markdown": "## Serving\n\n```bash\nexample serve m\n```"
        }
    )
    guide = await gen.generate([QUICKSTART], discover_subpages=False)
    sources = gen.grounding_chain.ainvoke.await_args.args[0]["sources"]
    assert "Engine Arguments" in sources and "(official documentation)" in sources
    assert "--preemption-mode" not in guide.body
    assert not any("--preemption-mode" in note for note in guide.notes)
    synopsis_inputs = gen.synopsis_chain.ainvoke.await_args.args[0]
    assert "- Installation" in synopsis_inputs["input_outline"]


def test_input_outline_lists_only_the_requested_pages() -> None:
    corpus = ResearchCorpus(
        pages=[
            PageContent(
                QUICKSTART, "Q", "## Installation\n\nx\n\n## On Attention Backends"
            ),
            PageContent("https://blog.example.com/a", "B", "## Unrelated"),
        ]
    )
    outline = input_outline(corpus, [QUICKSTART])
    assert outline == "- Installation\n- On Attention Backends"
    assert input_outline(ResearchCorpus(), [QUICKSTART]) == "(none)"


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
