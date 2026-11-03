"""The evidence engine: checked records, the outline's ownership rules, the
draft checks, and one scripted run through every stage (no model, no
network)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from scholar_lens.application.ports import AgentLimits, AgentOutcome
from scholar_lens.application.survey_engine import (
    ClaimCheck,
    EngineSettings,
    EvidenceSurveyEngine,
    SectionReview,
    SentenceVerdict,
)
from scholar_lens.domain.evidence import (
    EvidenceLedger,
    SourceDocument,
    evidence_problems,
)
from scholar_lens.domain.survey.evidence_post import (
    EvidenceOutline,
    cite_papers,
    comparison_table,
    draft_problems,
    outline_problems,
    structure_lost,
)
from scholar_lens.domain.survey.plan import SurveyPlan

PAPER_TEXT = """## Abstract

We propose DPO, which optimizes preferences without a reward model.

## 4 Method

The DPO loss is $$\\mathcal{L}=-\\log\\sigma(\\beta h)$$ for a margin h.

## 6 Experiments

DPO reaches a 61% win rate against PPO on TL;DR summarization with GPT-4 as judge.
Training is less stable when beta is small.
"""


def _doc(pid: str = "2305.18290") -> SourceDocument:
    return SourceDocument.from_markdown(
        pid,
        "DPO",
        "full_text",
        PAPER_TEXT,
        figures=[
            ("https://arxiv.org/html/2305.18290v3/x1.png", "Figure 1: DPO vs RLHF.")
        ],
    )


def _plan() -> SurveyPlan:
    return SurveyPlan.model_validate(
        {
            "created_at": datetime(2026, 11, 4, tzinfo=UTC).isoformat(),
            "topic": "Preference optimization",
            "scope": "From RLHF to DPO.",
            "time_range": {"since": None, "until": None},
            "lineages": [
                {"id": "offline", "name": "Offline", "description": "Without RL."}
            ],
            "papers": [
                {
                    "id": "2305.18290",
                    "title": "Direct Preference Optimization",
                    "authors": "Rafailov et al.",
                    "year": 2023,
                    "url": "https://arxiv.org/abs/2305.18290",
                    "lineage": "offline",
                    "role": "core",
                    "reason": "the method",
                }
            ],
        }
    )


class TestEvidenceRecords:
    def test_a_verbatim_quote_with_its_numbers_and_scope_is_accepted(self) -> None:
        assert not evidence_problems(
            _doc(),
            kind="number",
            claim="PPO 대비 승률 61%",
            quote="DPO reaches a 61% win rate against PPO",
            anchor="S3",
            scope="TL;DR summarization, GPT-4 judge",
            cell="승률 61%",
        )

    @pytest.mark.parametrize(
        ("fields", "fragment"),
        [
            ({"quote": "DPO reaches a 71% win rate against PPO"}, "not verbatim"),
            ({"claim": "PPO 대비 승률 71%"}, "numbers the quote does not"),
            ({"scope": ""}, "needs its scope"),
            ({"anchor": "S9"}, "unknown anchor"),
        ],
    )
    def test_rejections_say_what_to_fix(
        self, fields: dict[str, str], fragment: str
    ) -> None:
        record = {
            "kind": "number",
            "claim": "PPO 대비 승률 61%",
            "quote": "DPO reaches a 61% win rate against PPO",
            "anchor": "S3",
            "scope": "TL;DR, GPT-4 judge",
            **fields,
        }
        problems = evidence_problems(_doc(), **record)
        assert any(fragment in p for p in problems), problems

    def test_equations_must_be_the_papers_latex_and_figures_known(self) -> None:
        doc = _doc()
        ok = dict(kind="equation", claim="손실", quote="The DPO loss is", anchor="S2")
        assert not evidence_problems(
            doc, **ok, latex="\\mathcal{L}=-\\log\\sigma(\\beta h)"
        )
        assert evidence_problems(doc, **ok, latex="\\mathcal{L}=\\log h")
        fig = dict(kind="figure", claim="비교 그림", quote="", anchor="S1")
        assert not evidence_problems(doc, **fig, figure_id="F1")
        assert evidence_problems(doc, **fig, figure_id="F7")

    def test_document_outline_and_search(self) -> None:
        doc = _doc()
        assert [s.anchor for s in doc.sections] == ["S1", "S2", "S3"]
        assert "F1 | figure" in doc.outline()
        assert doc.search("win rate PPO")[0][0] == "S3"


def _ledger() -> EvidenceLedger:
    doc = _doc()
    ledger = EvidenceLedger()
    ledger.add(
        doc,
        kind="method",
        claim="보상 모델 없이 선호를 최적화",
        quote="optimizes preferences without a reward model",
        anchor="S1",
    )
    ledger.add(
        doc,
        kind="number",
        claim="PPO 대비 승률 61%",
        quote="DPO reaches a 61% win rate against PPO",
        anchor="S3",
        scope="TL;DR, GPT-4 judge",
        cell="승률 61%",
    )
    ledger.add(
        doc,
        kind="limitation",
        claim="beta가 작으면 학습이 불안정",
        quote="Training is less stable when beta is small.",
        anchor="S3",
        cell="작은 beta에서 불안정",
    )
    ledger.add(
        doc,
        kind="figure",
        claim="DPO와 RLHF 비교",
        quote="",
        anchor="S1",
        figure_id="F1",
    )
    return ledger


def _outline(**changes: Any) -> EvidenceOutline:
    data: dict[str, Any] = {
        "title": "Preference Optimization Without RL",
        "short_names": {"2305.18290": "DPO"},
        "lineage_names": {"offline": "오프라인"},
        "ideas": {"2305.18290": "보상 모델 없는 선호 최적화"},
        "table": [{"paper": "2305.18290", "results": ["E2"], "limitation": "E3"}],
        "sections": [
            {
                "kind": "introduction",
                "title": "선호 최적화는 RL 없이도 됩니다",
                "thesis": "t",
                "evidence": ["E1"],
            },
            {
                "kind": "lineage",
                "title": "DPO는 보상 모델을 없앱니다",
                "thesis": "t",
                "lineage": "offline",
                "papers": ["2305.18290"],
                "evidence": ["E2", "E4"],
            },
            {"kind": "comparison", "title": "비교", "thesis": "t"},
            {"kind": "insights", "title": "통찰", "thesis": "t"},
        ],
    }
    data.update(changes)
    return EvidenceOutline.model_validate(data)


class TestOutline:
    def test_a_valid_outline_passes(self) -> None:
        assert not outline_problems(_outline(), _plan(), _ledger(), title_max_length=70)

    def test_evidence_given_twice_and_wrong_order_are_problems(self) -> None:
        sections = _outline().model_dump()["sections"]
        sections[2]["evidence"] = ["E2"]
        sections.reverse()
        problems = outline_problems(
            _outline(sections=sections), _plan(), _ledger(), title_max_length=70
        )
        assert any("given to both" in p for p in problems)
        assert any("first section must be the introduction" in p for p in problems)

    def test_comparison_table_cells(self) -> None:
        table = comparison_table(_plan(), _outline(), _ledger(), korean=True)
        assert (
            "| DPO | 오프라인 | 승률 61% (TL;DR, GPT-4 judge) | 작은 beta에서 불안정 |"
            in table
        )
        empty = _outline(table=[{"paper": "2305.18290"}])
        assert "정량 수치 없음" in comparison_table(
            _plan(), empty, _ledger(), korean=True
        )


class TestDraftChecks:
    URL = "https://arxiv.org/html/2305.18290v3/x1.png"

    def _check(self, markdown: str, numbers: dict[str, str] | None = None) -> list[str]:
        return draft_problems(
            markdown,
            title="DPO는 보상 모델을 없앱니다",
            own=["E2", "E4"],
            earlier=["E1"],
            ledger=_ledger(),
            earlier_numbers=numbers or {},
            figure_urls=[self.URL],
            free_numbers="2023 1 2 3",
        )

    def test_a_cited_sentence_with_its_numbers_passes(self) -> None:
        md = f"## DPO는 보상 모델을 없앱니다\n\nDPO는 PPO 대비 61% 승률을 보입니다 [[E2]].\n\n![비교]({self.URL})\n"
        assert self._check(md) == []

    @pytest.mark.parametrize(
        ("sentence", "fragment"),
        [
            ("DPO는 71% 승률을 보입니다 [[E2]].", "not in the cited evidence"),
            ("DPO는 61% 승률을 보입니다.", "without evidence"),
            ("DPO는 불안정합니다 [[E3]].", "belongs to another section"),
            ("초록에 따르면 DPO는 간단합니다 [[E2]].", "post's inputs"),
        ],
    )
    def test_problems(self, sentence: str, fragment: str) -> None:
        md = f"## DPO는 보상 모델을 없앱니다\n\n{sentence}\n\n![비교]({self.URL})\n"
        assert any(fragment in p for p in self._check(md)), self._check(md)

    def test_a_number_stated_earlier_is_a_repeat(self) -> None:
        md = f"## DPO는 보상 모델을 없앱니다\n\nDPO는 61% 승률입니다 [[E2]].\n\n![비교]({self.URL})\n"
        assert any("already stated" in p for p in self._check(md, {"61": "도입"}))

    def test_markers_become_paper_citations(self) -> None:
        assert (
            cite_papers("승률 61% [[E2, E1]].", _ledger()) == "승률 61% [@2305.18290]."
        )

    def test_structure_loss(self) -> None:
        before = f"## A\n\n문장 하나입니다.\n\n![x]({self.URL})\n\n$$a=b$$"
        assert structure_lost(before, "## A\n\n문장 하나입니다.") == "a figure was lost"
        assert structure_lost(before, before) == ""


class ScriptedModel:
    """Answers each port call from a script; the reader calls the tools."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run_agent(
        self, *, role, system, prompt, tools, limits: AgentLimits
    ) -> AgentOutcome:
        self.calls.append("read")
        by_name = {t.__name__: t for t in tools}
        record = by_name["record_evidence"]
        assert "S3 | 6 Experiments" in prompt
        assert record(
            "number",
            "PPO 대비 승률 71%",
            "DPO reaches a 61% win rate against PPO",
            "S3",
            "TL;DR",
        ).startswith("REJECTED")
        record(
            "method",
            "보상 모델 없이 선호를 최적화",
            "optimizes preferences without a reward model",
            "S1",
        )
        record(
            "number",
            "PPO 대비 승률 61%",
            "DPO reaches a 61% win rate against PPO",
            "S3",
            "TL;DR, GPT-4 judge",
            "승률 61%",
        )
        record(
            "limitation",
            "beta가 작으면 학습이 불안정",
            "Training is less stable when beta is small.",
            "S3",
            "",
            "작은 beta에서 불안정",
        )
        record("figure", "DPO와 RLHF 비교", "", "S1", "", "", "F1")
        return AgentOutcome(text="done", stop_reason="end_turn")

    async def structured(self, *, role, system, prompt, schema):
        self.calls.append(schema.__name__)
        if schema is EvidenceOutline:
            return _outline()
        if schema is ClaimCheck:
            bad = "71%" in prompt
            return ClaimCheck(
                verdicts=(
                    [SentenceVerdict(n=1, verdict="wrong", reason="61%")] if bad else []
                )
            )
        if schema is SectionReview:
            return SectionReview()
        raise AssertionError(schema)

    async def text(self, *, role, system, prompt) -> str:
        self.calls.append(f"text:{role}")
        if "<problems>" in prompt:  # the fix: the wrong number corrected
            title = prompt.split("## ", 1)[1].split("\n", 1)[0]
            return f"## {title}\n\nDPO는 PPO 대비 61% 승률을 보입니다 [[E2]].\n\n![비교](https://arxiv.org/html/2305.18290v3/x1.png)\n"
        title = prompt.split("title: ", 1)[1].split("\n", 1)[0]
        if title.startswith("DPO"):
            return f"## {title}\n\nDPO는 PPO 대비 71% 승률을 보입니다 [[E2]].\n\n![비교](https://arxiv.org/html/2305.18290v3/x1.png)\n"
        if title.startswith("선호"):
            return f"## {title}\n\nDPO는 보상 모델 없이 선호를 최적화합니다 [[E1]].\n"
        return f"## {title}\n\n두 방법은 서로 다른 장단점을 가집니다.\n"


class FakeDocuments:
    async def document(self, paper):
        return _doc(paper.id)


class MemoryStore:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def put(self, key: str, text: str) -> None:
        self.data[key] = text


def test_a_scripted_run_reads_outlines_writes_fixes_and_assembles() -> None:
    model = ScriptedModel()
    store = MemoryStore()
    engine = EvidenceSurveyEngine(
        model,
        FakeDocuments(),
        EngineSettings(language="Korean", title_max_length=70, reader_tag="sonnet"),
        store,
    )
    survey = asyncio.run(engine.write(_plan()))
    assert len(survey.ledger) == 4
    body = survey.body
    assert (
        body.index("## 선호 최적화는")
        < body.index("| 연도 | 논문 |")
        < body.index("## DPO는")
    )
    assert "61% 승률" in body and "71%" not in body
    assert "[@2305.18290]" in body and "[[E" not in body
    assert "| DPO | 오프라인 | 승률 61%" in body
    assert model.calls.count("read") == 1 and "text:write" in model.calls
    # The paper's evidence is cached: a second run does not read it again.
    (key,) = store.data
    assert len(json.loads(store.data[key])) == 4
    again = ScriptedModel()
    asyncio.run(
        EvidenceSurveyEngine(
            again,
            FakeDocuments(),
            EngineSettings(language="Korean", title_max_length=70, reader_tag="sonnet"),
            store,
        ).write(_plan())
    )
    assert "read" not in again.calls
