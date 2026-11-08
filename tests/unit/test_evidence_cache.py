"""The engine's evidence cache, per kind (docs/engine.md, 4.9): a reader
with the same inputs is not run again, another language or reader tag is
read again, unreadable cached evidence is read again, and a spec that
rechecks its cache (the guide's) drops a cached item today's policy
rejects."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest

from scholar_lens.application.guide_engine import EvidenceGuideEngine, GuideSettings
from scholar_lens.application.paper_engine import EvidencePaperEngine, PaperSettings
from scholar_lens.application.ports import AgentOutcome
from scholar_lens.application.survey_engine import (
    EngineSettings,
    EvidenceSurveyEngine,
)
from scholar_lens.domain.evidence import Evidence
from tests.unit.test_evidence_engine import FakeDocuments, _plan
from tests.unit.test_guide_engine import _blog, _official
from tests.unit.test_paper_engine import _paper


class Store:
    def __init__(self) -> None:
        self.blobs: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.blobs.get(key)

    def put(self, key: str, text: str) -> None:
        self.blobs[key] = text


class Reader:
    """Records the first line of the source's first section as a finding;
    counts its reads (a follow-up is not a read)."""

    def __init__(self) -> None:
        self.reads = 0

    async def run_agent(self, *, prompt: str, tools: Any, **_: Any) -> AgentOutcome:
        if "<already_recorded>" not in prompt:
            self.reads += 1
            named = {t.__name__: t for t in tools}
            text = named["read_section"]("S1").split("\n\n", 1)[1]
            line = next(ln for ln in text.split("\n") if ln.strip())
            quote = line[:100].rsplit(" ", 1)[0] if len(line) > 100 else line
            result = named["record_evidence"]("finding", "요청을 묶습니다", quote, "S1")
            assert result.startswith("recorded"), result
        return AgentOutcome(text="", stop_reason="end_turn")


@dataclass
class Kind:
    """Runs one kind's reading with a store, a language and a reader tag."""

    name: str
    read: Callable[[Reader, Store, str, str], Awaitable[Any]]


async def _paper_read(model: Reader, store: Store, language: str, tag: str) -> Any:
    engine = EvidencePaperEngine(
        model,  # type: ignore[arg-type]
        PaperSettings(language=language, post="review", reader_tag=tag),
        store=store,
    )
    return await engine.read("DPO", _paper(), [])


async def _guide_read(model: Reader, store: Store, language: str, tag: str) -> Any:
    engine = EvidenceGuideEngine(
        model,  # type: ignore[arg-type]
        GuideSettings(language=language, title_max_length=70, reader_tag=tag),
        store,
    )
    return await engine.read("Serve batching", [_official(), _blog()])


async def _survey_read(model: Reader, store: Store, language: str, tag: str) -> Any:
    engine = EvidenceSurveyEngine(
        model,  # type: ignore[arg-type]
        FakeDocuments(),
        EngineSettings(language=language, title_max_length=70, reader_tag=tag),
        store,
    )
    ledger, _ = await engine.read(_plan())
    return ledger


KINDS = [
    Kind("review", _paper_read),
    Kind("guide", _guide_read),
    Kind("survey", _survey_read),
]


def _run(kind: Kind, model: Reader, store: Store, **settings: str) -> Any:
    language = settings.get("language", "Korean")
    tag = settings.get("tag", "sonnet")
    return asyncio.run(kind.read(model, store, language, tag))


@pytest.mark.parametrize("kind", KINDS, ids=lambda k: k.name)
def test_a_language_or_reader_tag_change_reads_again(kind: Kind) -> None:
    store, model = Store(), Reader()
    _run(kind, model, store)
    sources = model.reads
    assert sources >= 1
    _run(kind, model, store)
    assert model.reads == sources  # the same inputs: cached
    _run(kind, model, store, language="English")
    assert model.reads == 2 * sources
    _run(kind, model, store, tag="opus")
    assert model.reads == 3 * sources
    # Each setting has its own cache entries, all reused afterwards.
    for settings in ({}, {"language": "English"}, {"tag": "opus"}):
        _run(kind, model, store, **settings)
    assert model.reads == 3 * sources


@pytest.mark.parametrize("kind", KINDS, ids=lambda k: k.name)
@pytest.mark.parametrize(
    "broken", ["not json", "[{]", '[{"id": 1}]', "{}", "[]", '"text"', "null"]
)
def test_unreadable_cached_evidence_is_read_again(kind: Kind, broken: str) -> None:
    store, model = Store(), Reader()
    first = _run(kind, model, store)
    sources = model.reads
    for key in store.blobs:
        store.blobs[key] = broken
    again = _run(kind, model, store)
    assert model.reads == 2 * sources
    assert [e.claim for e in again] == [e.claim for e in first]
    # And saved again in a form the next run reads.
    for text in store.blobs.values():
        assert [Evidence.model_validate(e) for e in json.loads(text)]


def _policy_breaking(item: dict[str, Any]) -> dict[str, Any]:
    """A cached third-party record of a kind today's guide policy rejects
    when the corpus has official docs (settings come from them only)."""
    return {
        **item,
        "id": "E99",
        "kind": "setting",
        "claim": "묶음 크기 설정",
        "scope": "@serve.batch",
    }


def test_the_guide_rechecks_its_cache_against_todays_policy() -> None:
    store, model = Store(), Reader()
    first = _run(KINDS[1], model, store)
    reads = model.reads
    blog = _blog()
    (key,) = [k for k, v in store.blobs.items() if blog.url in v or "W1" in v]
    items = json.loads(store.blobs[key])
    store.blobs[key] = json.dumps([*items, _policy_breaking(items[0])])
    again = _run(KINDS[1], model, store)
    # The rejected item is dropped, the rest of the cache is used as is.
    assert model.reads == reads
    assert [e.kind for e in again] == [e.kind for e in first]
    # Only the rejected items cached: nothing passes, so the page is read
    # again.
    store.blobs[key] = json.dumps([_policy_breaking(items[0])])
    _run(KINDS[1], model, store)
    assert model.reads == reads + 1


@pytest.mark.parametrize("kind", [KINDS[0], KINDS[2]], ids=lambda k: k.name)
def test_a_spec_that_does_not_recheck_keeps_what_its_policy_would_reject(
    kind: Kind,
) -> None:
    """Only the guide rechecks its cache against the policy; the paper and
    survey specs use it as recorded (their records pass the generic
    checks, which every kind reruns)."""
    store, model = Store(), Reader()
    first = _run(kind, model, store)
    reads = model.reads
    key = next(iter(store.blobs))
    items = json.loads(store.blobs[key])
    store.blobs[key] = json.dumps([*items, _policy_breaking(items[0])])
    again = _run(kind, model, store)
    assert model.reads == reads
    assert len(again) == len(first) + 1
