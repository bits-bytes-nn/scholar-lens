"""The gathered corpus is saved under a hash of its inputs and reused within
the TTL (docs/engine.md 3.1; no network or model calls)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from scholar_lens.src.source_set import DomainClasses
from scholar_lens.src.source_snapshot import (
    SNAPSHOT_FORMAT,
    SnapshotPolicy,
    load_snapshot,
    snapshot_key,
)
from scholar_lens.src.survey_documents import EvidenceTextStore
from scholar_lens.src.web_research import PageContent, ResearchCorpus, SearchResult
from tests.test_tech_guide import _make_generator, _relevant

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
