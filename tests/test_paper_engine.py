"""The evidence-based paper review and summary: reading groups, the outline's
order and fixed parts, the TL;DR shapes and a scripted run to the assembled
review and summary."""

from __future__ import annotations

import asyncio
from typing import Any

from scholar_lens.application.evidence_writer import (
    ClaimCheck,
    Edit,
    EvidenceCheck,
    FixedSection,
    PostReview,
    SectionEdits,
    SectionReview,
)
from scholar_lens.application.paper_engine import (
    EvidencePaperEngine,
    FinalText,
    PaperSettings,
    reading_groups,
)
from scholar_lens.application.ports import AgentLimits, AgentOutcome
from scholar_lens.domain.evidence import EvidenceLedger, SourceDocument
from scholar_lens.domain.paper.evidence_paper import (
    REVIEW_TLDR_QUESTIONS,
    PaperOutline,
    assemble_review,
    paper_outline_problems,
    repair_paper_outline,
    tldr_problems,
)

PAPER = """## Abstract

We propose DPO, which optimizes preferences without a reward model.

## 1 Introduction

RLHF fits a reward model and then runs PPO, which is unstable.

## 4 Method

The DPO loss is $$\\mathcal{L}=-\\log\\sigma(\\beta h)$$ for a margin h.

## 6 Experiments

DPO reaches a 61% win rate against PPO on TL;DR summarization with GPT-4 as judge.
Training is less stable when beta is small.
"""


def _paper() -> SourceDocument:
    return SourceDocument.from_markdown("2305.18290", "DPO", "full_text", PAPER)


def _ledger() -> EvidenceLedger:
    ledger = EvidenceLedger()
    doc = _paper()
    ledger.add(
        doc,
        kind="finding",
        claim="RLHF는 보상 모델과 PPO가 필요하고 불안정합니다",
        quote="RLHF fits a reward model and then runs PPO, which is unstable.",
        anchor="S2",
    )
    ledger.add(
        doc,
        kind="method",
        claim="보상 모델 없이 선호를 최적화합니다",
        quote="We propose DPO, which optimizes preferences without a reward model.",
        anchor="S1",
    )
    ledger.add(
        doc,
        kind="number",
        claim="PPO 대비 승률 61%",
        quote="DPO reaches a 61% win rate against PPO on TL;DR summarization with GPT-4 as judge.",
        anchor="S4",
        scope="TL;DR 요약, GPT-4 판정",
        cell="승률 61%",
    )
    ledger.add(
        doc,
        kind="limitation",
        claim="beta가 작으면 학습이 불안정합니다",
        quote="Training is less stable when beta is small.",
        anchor="S4",
        cell="작은 beta에서 불안정",
    )
    return ledger


def _outline(**changes: Any) -> PaperOutline:
    data: dict[str, Any] = {
        "sections": [
            {"kind": "tldr", "title": "요약", "thesis": "t"},
            {
                "kind": "problem",
                "title": "보상 모델이 만드는 불안정",
                "thesis": "t",
                "evidence": ["E1"],
            },
            {
                "kind": "method",
                "title": "보상 모델 없이 푸는 선호 최적화",
                "thesis": "t",
                "depth": "deep",
                "evidence": ["E2"],
            },
            {
                "kind": "experiments",
                "title": "요약 과제에서 앞서는 승률",
                "thesis": "t",
                "evidence": ["E3"],
            },
            {"kind": "closing", "title": "끝", "thesis": "t", "evidence": ["E4"]},
        ]
    }
    data.update(changes)
    return PaperOutline.model_validate(data)


def test_reading_groups_split_the_paper_by_size() -> None:
    groups = reading_groups(_paper(), chars=80)
    assert [[s.anchor for s in g.sections] for g in groups] == [
        ["S1"],
        ["S2"],
        ["S3"],
        ["S4"],
    ]
    assert all(g.paper_id == "2305.18290" for g in groups)
    assert [s.anchor for s in reading_groups(_paper(), 10_000)[0].sections] == [
        "S1",
        "S2",
        "S3",
        "S4",
    ]


def test_a_review_outline_gets_fixed_titles_and_passes() -> None:
    outline, notes = repair_paper_outline(_outline(), _ledger(), "review", "Korean")
    titles = [s.title for s in outline.sections]
    assert titles[0] == "TL;DR" and titles[-1] == "한계와 시사점"
    assert notes == []
    assert paper_outline_problems(outline, _ledger(), "review", max_sections=12) == []


def test_a_review_without_method_or_experiments_is_a_problem() -> None:
    sections = [s.model_dump() for s in _outline().sections]
    outline = _outline(sections=[sections[0], sections[1], sections[4]])
    problems = paper_outline_problems(outline, _ledger(), "review", max_sections=12)
    assert any("method" in p for p in problems)
    assert any("experiments" in p for p in problems)


def test_a_summary_gets_its_five_fixed_sections() -> None:
    sections = [s.model_dump() for s in _outline().sections]
    sections.insert(
        3, {"kind": "implementation", "title": "x", "thesis": "t", "evidence": []}
    )
    sections.insert(
        3, {"kind": "analysis", "title": "y", "thesis": "t", "evidence": []}
    )
    outline, notes = repair_paper_outline(
        _outline(sections=sections), _ledger(), "summary", "Korean"
    )
    assert [s.title for s in outline.sections] == [
        "TL;DR",
        "연구 배경",
        "제안 방법",
        "구현 방식",
        "실험 결과",
        "의의와 한계",
    ]
    assert paper_outline_problems(outline, _ledger(), "summary", max_sections=6) == []
    assert any("merged" in n for n in notes)


def test_tldr_shapes() -> None:
    questions = REVIEW_TLDR_QUESTIONS["Korean"]
    good = "## TL;DR\n\n" + "".join(f"### {q}\n\n답입니다.\n\n" for q in questions)
    assert tldr_problems(good, "review", "Korean") == []
    assert tldr_problems("## TL;DR\n\n### 다른 질문\n\n답.\n", "review", "Korean")
    assert (
        tldr_problems(
            "## TL;DR\n\nDPO는 보상 모델 없이 학습합니다.\n", "summary", "Korean"
        )
        == []
    )
    assert tldr_problems("## TL;DR\n\n- 목록\n", "summary", "Korean")


def test_assemble_review_splits_tldr_body_and_closing() -> None:
    key, body, closing = assemble_review(
        [
            ("tldr", "## TL;DR\n\n### Q\n\nA."),
            ("problem", "## P\n\nx."),
            ("closing", "## 한계와 시사점\n\ny."),
        ]
    )
    assert (key, body, closing) == (
        "### Q\n\nA.",
        "## P\n\nx.",
        "## 한계와 시사점\n\ny.",
    )


class ScriptedModel:
    def __init__(self, post: str) -> None:
        self.post = post
        self.calls: list[str] = []

    async def run_agent(
        self, *, role, system, prompt, tools, limits: AgentLimits
    ) -> AgentOutcome:
        self.calls.append("read")
        record = {t.__name__: t for t in tools}["record_evidence"]
        record(
            "finding",
            "RLHF는 보상 모델과 PPO가 필요하고 불안정합니다",
            "RLHF fits a reward model and then runs PPO, which is unstable.",
            "S2",
        )
        record(
            "method",
            "보상 모델 없이 선호를 최적화합니다",
            "We propose DPO, which optimizes preferences without a reward model.",
            "S1",
        )
        record(
            "number",
            "PPO 대비 승률 61%",
            "DPO reaches a 61% win rate against PPO on TL;DR summarization with GPT-4 as judge.",
            "S4",
            "TL;DR 요약, GPT-4 판정",
            "승률 61%",
        )
        record(
            "limitation",
            "beta가 작으면 학습이 불안정합니다",
            "Training is less stable when beta is small.",
            "S4",
        )
        return AgentOutcome(text="done", stop_reason="end_turn")

    async def structured(self, *, role, system, prompt, schema, context=""):
        if schema is SectionEdits:  # a fix as one edit: the old section for the new
            old = prompt.split("<section>\n", 1)[1].split("\n</section>", 1)[0]
            new = await self.text(
                role=role, system=system, prompt=prompt, context=context
            )
            return SectionEdits(edits=[Edit(find=old.strip(), replace=new.strip())])
        if schema is FixedSection:  # a fix answers like a text call, structured
            section = await self.text(
                role=role, system=system, prompt=prompt, context=context
            )
            return FixedSection(section=section, notes="고친 내용: 수치를 바로잡음")
        prompt = f"{context}\n\n{prompt}"
        self.calls.append(schema.__name__)
        if schema is PaperOutline:
            sections = [s.model_dump() for s in _outline().sections]
            if self.post == "summary":
                sections.insert(
                    3,
                    {
                        "kind": "implementation",
                        "title": "x",
                        "thesis": "t",
                        "evidence": [],
                    },
                )
            return PaperOutline.model_validate({"sections": sections})
        if schema is ClaimCheck:
            return ClaimCheck()
        if schema is SectionReview:
            return SectionReview()
        if schema is EvidenceCheck:
            return EvidenceCheck()
        if schema is PostReview:
            return PostReview()
        if schema is FinalText:
            return FinalText(
                subtitle="직접 선호 최적화",
                description="DPO는 보상 모델 없이 언어 모델을 사람의 선호에 맞추는 방법으로, "
                "TL;DR 요약 과제에서 PPO 대비 승률 61%를 기록했습니다. 손실 함수와 "
                "학습 안정성, 실험 설정과 한계, 실무에서 쓸 때 주의할 점까지 함께 정리합니다.",
            )
        raise AssertionError(schema)

    async def text(self, *, role, system, prompt, context="") -> str:
        prompt = f"{context}\n\n{prompt}"
        self.calls.append("text")
        if "Write the review's TL;DR" in prompt:
            questions = REVIEW_TLDR_QUESTIONS["Korean"]
            return "## TL;DR\n\n" + "".join(
                f"### {q}\n\nDPO는 PPO 대비 61% 승률을 기록했습니다 [[E3]].\n\n"
                for q in questions
            )
        if "Write the summary's TL;DR" in prompt:
            return "## TL;DR\n\nDPO는 보상 모델 없이 선호를 최적화합니다 [[E2]].\n"
        title = prompt.split("title: ", 1)[1].split("\n", 1)[0]
        kind = prompt.split("kind: ", 1)[1].split("\n", 1)[0]
        lines = {
            "problem": "RLHF는 보상 모델과 PPO가 필요해 불안정합니다 [[E1]].",
            "method": "DPO는 보상 모델 없이 선호를 최적화합니다 [[E2]].\n\n$$\\mathcal{L}=-\\log\\sigma(\\beta h)$$\n\n| 구분 | 값 |\n| --- | --- |\n| 손실 | 로그 시그모이드 |",
            "implementation": "구현은 손실 하나로 끝납니다 [[E2]].",
            "experiments": "DPO는 PPO 대비 61% 승률을 기록했습니다 [[E3]].",
            "closing": "beta가 작으면 학습이 불안정합니다 [[E4]].",
        }
        return f"## {title}\n\n{lines.get(kind, '내용입니다.')}\n"


def _run(post: str):  # type: ignore[no-untyped-def]
    model = ScriptedModel(post)
    engine = EvidencePaperEngine(
        model,
        PaperSettings(
            language="Korean",
            post=post,  # type: ignore[arg-type]
            reader_tag="sonnet",
            min_total_chars=1_000,
            max_total_chars=2_000,
        ),
    )
    return model, asyncio.run(engine.write("DPO", _paper(), abstract="We propose DPO."))


def test_a_scripted_review_run() -> None:
    model, result = _run("review")
    kinds = [k for k, _ in result.sections]
    assert kinds == ["tldr", "problem", "method", "experiments", "closing"]
    key, body, closing = assemble_review(result.sections)
    assert key.startswith("### 이 연구를 시작하게 된 배경과 동기는 무엇입니까?")
    assert closing.startswith("## 한계와 시사점") and "[[E" not in body
    assert result.subtitle == "직접 선호 최적화" and result.description.startswith(
        "DPO는"
    )
    assert model.calls.count("read") == 1


def test_a_scripted_summary_run() -> None:
    _, result = _run("summary")
    headings = [md.split("\n", 1)[0] for _, md in result.sections]
    assert headings == [
        "## TL;DR",
        "## 연구 배경",
        "## 제안 방법",
        "## 구현 방식",
        "## 실험 결과",
        "## 의의와 한계",
    ]


def test_numbers_may_come_from_the_section_the_evidence_is_quoted_from() -> None:
    from scholar_lens.domain.evidence_draft import draft_problems

    md = "## 결과\n\nGPT-4 판정에서 DPO는 61%, 다른 설정에서는 58%였습니다 [[E3]].\n"
    kwargs: dict[str, Any] = {
        "title": "결과",
        "own": ["E3"],
        "earlier": (),
        "ledger": _ledger(),
        "earlier_numbers": {},
        "figure_urls": (),
        "free_numbers": "",
    }
    assert any("58" in p for p in draft_problems(md, **kwargs))
    passages = {("2305.18290", "S4"): "61% ... a 58% win rate in the other setting"}
    assert not any("58" in p for p in draft_problems(md, passages=passages, **kwargs))


def test_a_conditional_bar_inside_a_nested_call_becomes_mid() -> None:
    from scholar_lens.domain.markdown.math import normalize_math

    assert normalize_math(r"$$\pi_{\theta}(y_{\tau(k)}|x)$$") == (
        r"$$\pi_{\theta}(y_{\tau(k)}\mid x)$$"
    )


def test_blank_table_cells_are_problems() -> None:
    from scholar_lens.domain.evidence_draft import draft_problems

    md = (
        "## 결과\n\n결과는 아래와 같습니다 [[E3]].\n\n| 모델 | 점수 |\n| --- | --- |\n"
        "| DPO | 61% [[E3]] |\n| Qwen3 | — |\n| AIME | 논문은 지표를 밝히지 않습니다 |\n"
    )
    problems = draft_problems(
        md,
        title="결과",
        own=["E3"],
        earlier=(),
        ledger=_ledger(),
        earlier_numbers={},
        figure_urls=(),
        free_numbers="",
    )
    assert any("blank" in p for p in problems)


def test_a_language_without_fixed_wording_keeps_the_outlines_titles() -> None:
    from scholar_lens.domain.paper.evidence_paper import fixed_title

    assert fixed_title("closing", "review", "Japanese") is None
    assert fixed_title("closing", "review", "English") == "Limitations and Takeaways"
    four = "## TL;DR\n\n" + "".join(f"### 質問{i}\n\n答え。\n\n" for i in range(4))
    assert tldr_problems(four, "review", "Japanese") == []
    assert tldr_problems("## TL;DR\n\n### 質問\n\n答え。\n", "review", "Japanese")


def test_a_cell_holding_only_markers_is_blank_once_they_go() -> None:
    from scholar_lens.domain.evidence_draft import draft_problems

    md = (
        "## 결과\n\n결과는 아래와 같습니다 [[E3]].\n\n| 모델 | 점수 | 근거 |\n"
        "| --- | --- | --- |\n| DPO | 61% | [[E3]] |\n"
    )
    problems = draft_problems(
        md,
        title="결과",
        own=["E3"],
        earlier=(),
        ledger=_ledger(),
        earlier_numbers={},
        figure_urls=(),
        free_numbers="",
    )
    assert any("only evidence markers" in p for p in problems)
    ok = md.replace("| DPO | 61% | [[E3]] |", "| DPO | 61% [[E3]] | 승률 |")
    assert not any(
        "blank" in p
        for p in draft_problems(
            ok,
            title="결과",
            own=["E3"],
            earlier=(),
            ledger=_ledger(),
            earlier_numbers={},
            figure_urls=(),
            free_numbers="",
        )
    )


def test_inline_math_inside_text_in_a_display_is_unnested() -> None:
    from scholar_lens.domain.markdown.math import normalize_math

    md = "$$\n\\pi_{\\text{bo$n$}} = \\text{if $k$ wins}\n$$\n"
    assert (
        normalize_math(md).strip() + "\n"
        == "$$\n\\pi_{\\text{bo}n} = \\text{if }k\\text{ wins}\n$$\n"
    )
    code = "```python\nx = '$$\\text{a$b$}$$'\n```\n"
    assert normalize_math(code).strip() == code.strip()
