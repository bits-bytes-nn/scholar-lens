"""The engine's loop numbers are spec limits (docs/engine.md, 2.6 and 4.0):
the reader's follow-up, its ``finish`` refusals and section cap, the short
draft retry, the earlier evidence a fix sees and the evidence counts a
reader is asked for."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest

from scholar_lens.application.evidence_writer import EvidenceWriter
from scholar_lens.application.ports import AgentLimits, AgentOutcome
from scholar_lens.application.survey_engine import SURVEY_PROMPTS
from scholar_lens.domain.evidence import Evidence, SourceDocument
from scholar_lens.domain.post_spec import Limits
from scholar_lens.domain.specs import (
    GUIDE_SPEC,
    PAPER_REVIEW_SPEC,
    PAPER_SUMMARY_SPEC,
    SURVEY_SPEC,
)

TEXT = "## Overview\n\n" + "Serve batches requests on each replica. " * 40 + "\n"


def _document() -> SourceDocument:
    return SourceDocument.from_markdown("D1", "Batching", "full_text", TEXT)


class _Agent:
    """Reads the first section, calls ``finish`` until it is done, and
    records nothing (so every required kind stays missing)."""

    def __init__(self) -> None:
        self.turns: list[int] = []
        self.section = ""
        self.finishes: list[str] = []

    async def run_agent(self, *, tools: Any, limits: AgentLimits, **_: Any) -> Any:
        named = {t.__name__: t for t in tools}
        self.turns.append(limits.turns)
        self.section = named["read_section"]("S1")
        for _ in range(5):
            self.finishes.append(named["finish"]("all read"))
            if self.finishes[-1] == "done":
                break
        return AgentOutcome(text="done", stop_reason="end_turn")


@pytest.mark.parametrize(
    "limits",
    [
        Limits(),
        Limits(follow_up_turns=3, required_kind_asks=0, read_section_chars=50),
        Limits(follow_up_turns=12, required_kind_asks=4, read_section_chars=500),
    ],
)
def test_the_reader_follows_the_spec_limits(limits: Limits) -> None:
    agent = _Agent()
    writer = EvidenceWriter(agent, SURVEY_PROMPTS, max_fix_rounds=1, limits=limits)  # type: ignore[arg-type]
    asyncio.run(
        writer.read_document(
            _document(),
            prompt="read",
            limits=AgentLimits(turns=20),
            required=("method",),
        )
    )
    assert agent.turns == [20, limits.follow_up_turns]
    body = agent.section.split("\n\n", 1)[1]
    assert len(body) <= limits.read_section_chars
    # Each run of the agent: refusals up to the limit (shared), then done.
    assert agent.finishes.count("done") == 2
    assert len(agent.finishes) - 2 == limits.required_kind_asks


class _Drafts:
    def __init__(self, length: int) -> None:
        self.length = length
        self.calls = 0

    async def text(self, **_: Any) -> str:
        self.calls += 1
        return "## 절\n\n" + "가" * self.length + "입니다.\n"


@pytest.mark.parametrize("share", [0.1, 0.3, 0.6])
def test_a_short_draft_is_asked_again_below_the_spec_share(share: float) -> None:
    target = 1_000
    model = _Drafts(length=400)
    writer = EvidenceWriter(
        model,  # type: ignore[arg-type]
        SURVEY_PROMPTS,
        max_fix_rounds=1,
        limits=Limits(short_draft_share=share),
    )
    own = [_evidence("E1")]
    asyncio.run(writer.draft("<evidence>\nx\n</evidence>", own, target, retry_hint=""))
    short = 400 < share * target
    assert model.calls == (3 if short else 1)


def _evidence(i: str) -> Evidence:
    return Evidence(
        id=i, paper_id="P", kind="finding", claim=f"사실 {i}", quote="q", anchor="S1"
    )


@pytest.mark.parametrize("cap", [0, 5, 30])
@pytest.mark.parametrize("cites", [0, 3, 40])
def test_a_fix_sees_the_earlier_evidence_its_section_cites(
    cap: int, cites: int
) -> None:
    earlier = [_evidence(f"E{k}") for k in range(2, 80)]
    own = [_evidence("E1")]
    # The section cites the oldest earlier items: the ones a last-N window
    # would leave out.
    cited = earlier[:cites]
    draft = "## 절\n\n" + " ".join(f"사실입니다 [[{e.id}]]." for e in cited) + "\n"
    writer = EvidenceWriter(
        None,  # type: ignore[arg-type]
        SURVEY_PROMPTS,
        max_fix_rounds=1,
        limits=Limits(fix_earlier_evidence=cap),
    )
    seen = writer.fix_evidence(draft, own, earlier)
    shown = seen[len(own) :]
    assert seen[: len(own)] == own and len(shown) == min(cap, len(earlier))
    assert {e.id for e in cited[-cap:] if cap} <= {e.id for e in shown}
    # In their order, without repeats.
    assert shown == sorted(shown, key=earlier.index) and len(
        set(map(id, shown))
    ) == len(shown)


def test_the_limit_defaults_are_the_former_numbers() -> None:
    limits = Limits()
    assert (
        limits.fix_earlier_evidence,
        limits.short_draft_share,
        limits.follow_up_turns,
        limits.required_kind_asks,
        limits.read_section_chars,
    ) == (30, 0.3, 8, 2, 30_000)
    counts = {
        spec.name: (spec.limits.main_source_evidence, spec.limits.minor_source_evidence)
        for spec in (GUIDE_SPEC, SURVEY_SPEC, PAPER_REVIEW_SPEC, PAPER_SUMMARY_SPEC)
    }
    assert counts["guide"] == ((10, 25), (3, 10))
    assert counts["survey"] == ((12, 25), (4, 8))
    assert counts["review"][0] == (10, 25) and counts["summary"][0] == (6, 12)


@pytest.mark.parametrize("field", ["main_source_evidence", "minor_source_evidence"])
def test_every_count_range_is_ordered(field: str) -> None:
    for spec in (GUIDE_SPEC, SURVEY_SPEC, PAPER_REVIEW_SPEC, PAPER_SUMMARY_SPEC):
        low, high = getattr(spec.limits, field)
        assert 0 < low <= high
    assert field in {f.name for f in dataclasses.fields(Limits)}
