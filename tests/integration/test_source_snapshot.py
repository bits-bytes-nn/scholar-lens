"""The gathered corpus is saved under a hash of its inputs and reused within
the TTL (docs/engine.md 3.1; no network or model calls)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import tenacity
from botocore.exceptions import ClientError

from scholar_lens.adapters.scholarly.survey_documents import EvidenceTextStore
from scholar_lens.adapters.web.guide_research import TechGuideGenerator
from scholar_lens.adapters.web.research import (
    BraveSearchProvider,
    FallbackSearchProvider,
    NullSearchProvider,
    TavilySearchProvider,
    search_chain,
)
from scholar_lens.adapters.web.snapshot import (
    SNAPSHOT_FORMAT,
    SnapshotPolicy,
    load_snapshot,
    snapshot_key,
)
from scholar_lens.domain.research import PageContent, ResearchCorpus, SearchResult
from scholar_lens.domain.sources import DomainClasses
from tests.integration.test_tech_guide import _make_generator, _relevant

DOCS = "https://docs.example.dev/en/latest/"


class Clock:
    def __init__(self) -> None:
        self.at = datetime(2026, 11, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.at


def _generator():  # type: ignore[no-untyped-def]
    gen = _make_generator(auto_research=True)
    page = PageContent(url=DOCS, title="Docs", text="docs text", published="2026-01-02")
    gen.researcher.research = MagicMock(return_value=ResearchCorpus(pages=[page]))
    gen.researcher.search_provider.supports_search = True
    gen.researcher.search_notes = MagicMock(return_value=["1 web search(es) failed"])

    def run_searches(corpus, queries, **kwargs):  # type: ignore[no-untyped-def]
        corpus.search_results.append(SearchResult("r", "https://blog.example.org/a"))
        corpus.pages.append(
            PageContent(url="https://blog.example.org/a", title="A", text="blog")
        )

    gen.researcher.run_searches = MagicMock(side_effect=run_searches)
    gen.relevance_chain.ainvoke = _relevant()
    gen.research_plan_chain.ainvoke = _plan()
    return gen


def _plan():  # type: ignore[no-untyped-def]
    async def plan(_payload):  # type: ignore[no-untyped-def]
        return {"topic": "Planned topic", "queries": "example internals"}

    return MagicMock(side_effect=plan)


@pytest.fixture
def policy(tmp_path):  # type: ignore[no-untyped-def]
    store = EvidenceTextStore(tmp_path / "guide-sources", None, "")
    return SnapshotPolicy(store, ttl_days=7, now=Clock())


async def test_a_rerun_with_the_same_inputs_reuses_the_corpus(policy) -> None:  # type: ignore[no-untyped-def]
    first = _generator()
    gathered = await first.gather([DOCS], discover_subpages=False, snapshots=policy)
    assert gathered.queries == ["example internals"]

    again = _generator()
    reused = await again.gather([DOCS], discover_subpages=False, snapshots=policy)
    # Nothing is fetched, searched or asked of a model again.
    again.researcher.research.assert_not_called()
    again.researcher.run_searches.assert_not_called()
    again.relevance_chain.ainvoke.assert_not_called()
    assert reused.topic == gathered.topic == "Planned topic"
    assert reused.queries == ["example internals"]
    assert [(p.url, p.authority, p.published) for p in reused.corpus.pages] == [
        (DOCS, "official", "2026-01-02"),
        ("https://blog.example.org/a", "third_party", ""),
    ]
    assert [r.url for r in reused.corpus.search_results] == [
        "https://blog.example.org/a"
    ]
    assert again.official.hosts == first.official.hosts
    assert again.sources.target == first.sources.target
    assert "1 web search(es) failed" in again.notes
    assert any("--fresh-sources" in n for n in again.notes)


async def test_fresh_sources_gather_again_and_replace_the_snapshot(policy) -> None:  # type: ignore[no-untyped-def]
    await _generator().gather([DOCS], discover_subpages=False, snapshots=policy)
    fresh = SnapshotPolicy(policy.store, ttl_days=7, fresh=True, now=policy.now)
    again = _generator()
    await again.gather([DOCS], discover_subpages=False, snapshots=fresh)
    again.researcher.research.assert_called_once()
    assert not any("--fresh-sources" in n for n in again.notes)


async def test_a_snapshot_past_the_ttl_is_gathered_again(policy) -> None:  # type: ignore[no-untyped-def]
    await _generator().gather([DOCS], discover_subpages=False, snapshots=policy)
    policy.now.at += timedelta(days=8)
    again = _generator()
    await again.gather([DOCS], discover_subpages=False, snapshots=policy)
    again.researcher.research.assert_called_once()


async def test_other_inputs_or_settings_gather_again(policy) -> None:  # type: ignore[no-untyped-def]
    await _generator().gather([DOCS], discover_subpages=False, snapshots=policy)
    other_query = _generator()
    await other_query.gather(
        [DOCS], discover_subpages=False, search_queries=["q"], snapshots=policy
    )
    other_query.researcher.research.assert_called_once()
    capped = _generator()
    capped.max_third_party_pages = 1
    await capped.gather([DOCS], discover_subpages=False, snapshots=policy)
    capped.researcher.research.assert_called_once()


def test_the_key_hashes_the_inputs_canonically() -> None:
    a = snapshot_key({"urls": [DOCS], "fetch_top_results": 12, "x": {"b": 1, "a": 2}})
    b = snapshot_key({"x": {"a": 2, "b": 1}, "fetch_top_results": 12, "urls": [DOCS]})
    assert a == b and a.startswith("sources/") and a.endswith(".json")
    assert a != snapshot_key({"urls": [DOCS], "fetch_top_results": 6})


def test_an_unreadable_or_older_snapshot_is_ignored() -> None:
    demoted = DomainClasses()
    assert load_snapshot("not json", demoted=demoted, repo_tags=None) is None
    older = f'{{"format": {SNAPSHOT_FORMAT - 1}}}'
    assert load_snapshot(older, demoted=demoted, repo_tags=None) is None


# ---- only a complete gather is saved, and its key holds every input ------------


@pytest.fixture
def no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    for call in (TechGuideGenerator._invoke_plan, TechGuideGenerator._invoke_relevance):
        monkeypatch.setattr(call.retry, "wait", tenacity.wait_none())  # type: ignore[attr-defined]


def _throttled() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ThrottlingException"}, "ResponseMetadata": {}}, "Converse"
    )


def _failing_then(answer: dict[str, str], failures: int) -> MagicMock:
    left = [failures]

    async def call(_payload):  # type: ignore[no-untyped-def]
        if left[0] > 0:
            left[0] -= 1
            raise _throttled()
        return answer

    return MagicMock(side_effect=call)


async def test_planning_and_the_gate_go_through_the_retry_policy(
    policy, no_retry_wait  # type: ignore[no-untyped-def]
) -> None:
    gen = _generator()
    gen.research_plan_chain.ainvoke = _failing_then(
        {"topic": "Planned topic", "queries": "example internals"}, 2
    )
    gen.relevance_chain.ainvoke = _failing_then(
        {"is_relevant": "yes", "topic": "Using X", "reason": "ok"}, 1
    )
    material = await gen.gather([DOCS], discover_subpages=False, snapshots=policy)
    assert gen.research_plan_chain.ainvoke.call_count == 3
    assert gen.relevance_chain.ainvoke.call_count == 2
    assert material.queries == ["example internals"]
    assert not any("planning failed" in n for n in gen.notes)


@pytest.mark.parametrize(
    "error", [RuntimeError("bad answer"), "throttled"], ids=["terminal", "exhausted"]
)
async def test_failed_planning_is_noted_and_its_corpus_not_saved(
    policy, no_retry_wait, error: object  # type: ignore[no-untyped-def]
) -> None:
    gen = _generator()

    async def fail(_payload):  # type: ignore[no-untyped-def]
        raise _throttled() if error == "throttled" else error  # type: ignore[misc]

    gen.research_plan_chain.ainvoke = MagicMock(side_effect=fail)
    material = await gen.gather([DOCS], discover_subpages=False, snapshots=policy)
    assert material.queries == []
    assert any(
        "Research planning failed" in n and "seed pages only" in n for n in gen.notes
    )
    again = _generator()
    await again.gather([DOCS], discover_subpages=False, snapshots=policy)
    again.researcher.research.assert_called_once()


async def test_a_refused_permission_in_planning_still_raises(policy) -> None:  # type: ignore[no-untyped-def]
    gen = _generator()

    async def refuse(_payload):  # type: ignore[no-untyped-def]
        raise ClientError(
            {"Error": {"Code": "AccessDeniedException"}, "ResponseMetadata": {}},
            "Converse",
        )

    gen.research_plan_chain.ainvoke = MagicMock(side_effect=refuse)
    with pytest.raises(ClientError):
        await gen.gather([DOCS], discover_subpages=False, snapshots=policy)


@pytest.mark.parametrize("gap", ["docs_page", "reference", "search"])
async def test_an_incomplete_gather_is_not_saved(policy, gap: str) -> None:  # type: ignore[no-untyped-def]
    other = "https://docs.example.dev/en/latest/other/"
    reference = "https://blog.example.org/deep-dive"
    gen = _generator()
    urls = [DOCS, other] if gap == "docs_page" else [DOCS]
    references = [reference] if gap == "reference" else []
    page = PageContent(url=DOCS, title="Docs", text="docs text")
    # Only the first docs page reads; a reference reads nothing.
    gen.researcher.research = MagicMock(
        side_effect=lambda u, **_: ResearchCorpus(pages=[page] if DOCS in u else [])
    )
    if gap == "search":

        def failing_search(corpus, queries, **kwargs):  # type: ignore[no-untyped-def]
            gen.researcher.search_provider.failed_searches += 1

        gen.researcher.run_searches = MagicMock(side_effect=failing_search)
    await gen.gather(
        urls, reference_urls=references, discover_subpages=False, snapshots=policy
    )
    if gap == "docs_page":
        assert any("could not be read" in n and other in n for n in gen.notes)
    if gap == "reference":
        assert any("could not be read" in n and reference in n for n in gen.notes)
    again = _generator()
    again.researcher.research = gen.researcher.research
    again.researcher.research.reset_mock()
    await again.gather(
        urls, reference_urls=references, discover_subpages=False, snapshots=policy
    )
    assert again.researcher.research.call_count >= 1


def test_the_key_holds_the_whole_search_chain() -> None:
    tavily, brave = TavilySearchProvider("k"), BraveSearchProvider("k")
    chains = {
        search_chain(FallbackSearchProvider(tavily, brave)),
        search_chain(FallbackSearchProvider(brave, tavily)),
        search_chain(tavily),
        search_chain(NullSearchProvider()),
    }
    assert len(chains) == 4
    gen = _generator()
    keys = set()
    for provider in (
        FallbackSearchProvider(tavily, brave),
        FallbackSearchProvider(brave, tavily),
    ):
        gen.researcher.search_provider = provider
        keys.add(snapshot_key(gen._gather_inputs([DOCS], [], False, [])))
    assert len(keys) == 2


@pytest.mark.parametrize("latest", ["2.1.0", "2.2.0"])
async def test_a_snapshot_holds_only_while_its_target_release_is_the_latest(
    policy, latest: str  # type: ignore[no-untyped-def]
) -> None:
    def generator(release: str):  # type: ignore[no-untyped-def]
        gen = _generator()
        gen.release_lookup = lambda name: release if name == "example" else None
        gen.relevance_chain.ainvoke = AsyncMock(
            return_value={
                "is_relevant": "yes",
                "topic": "Using Example",
                "reason": "ok",
                "commands": "example",
            }
        )
        page = PageContent(
            url=DOCS, title="Docs", text="Install it:\n\n    pip install example\n"
        )
        gen.researcher.research = MagicMock(return_value=ResearchCorpus(pages=[page]))
        return gen

    first = generator("2.1.0")
    await first.gather([DOCS], discover_subpages=False, snapshots=policy)
    assert first.sources.target.release == "2.1.0"
    again = generator(latest)
    await again.gather([DOCS], discover_subpages=False, snapshots=policy)
    assert again.researcher.research.called is (latest != "2.1.0")
    assert again.sources.target.release == latest


async def test_a_failed_tag_lookup_is_noted_and_its_corpus_not_saved(policy) -> None:  # type: ignore[no-untyped-def]
    def generator():  # type: ignore[no-untyped-def]
        gen = _generator()
        gen.researcher.tag_lookup_failures = []
        page = PageContent(url=DOCS, title="Docs", text="docs text")

        def research(urls, **_):  # type: ignore[no-untyped-def]
            gen.researcher.tag_lookup_failures.append("github.com/example/proj")
            return ResearchCorpus(pages=[page])

        gen.researcher.research = MagicMock(side_effect=research)
        return gen

    first = generator()
    await first.gather([DOCS], discover_subpages=False, snapshots=policy)
    assert any(
        "github.com/example/proj" in n and "development code" in n for n in first.notes
    )
    again = generator()
    await again.gather([DOCS], discover_subpages=False, snapshots=policy)
    again.researcher.research.assert_called_once()
