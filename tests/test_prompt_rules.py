"""Invariants of the generation prompts (rendered text, not LLM output)."""

from __future__ import annotations

import pytest

from scholar_lens.src.prompts import (
    AttributesExtractionPrompt,
    PaperEditPrompt,
    PaperFinalizationPrompt,
    PaperPlanningPrompt,
    PaperSectionCheckPrompt,
    PaperSectionPrompt,
    PaperSummaryPrompt,
    TechGuideSectionPrompt,
)
from scholar_lens.src.prompts.base import BasePrompt
from scholar_lens.src.prompts.extraction import (
    DEFAULT_CATEGORY_OPTIONS,
    DEFAULT_CONTROLLED_TAGS,
)

WRITERS = [
    PaperSectionPrompt,
    PaperFinalizationPrompt,
    PaperSummaryPrompt,
    TechGuideSectionPrompt,
]


def _render(prompt: type[BasePrompt]) -> str:
    template = prompt.get_prompt()
    messages = template.format_messages(**{v: "X" for v in template.input_variables})
    return "\n".join(str(m.content) for m in messages)


@pytest.mark.parametrize("prompt", WRITERS + [PaperSectionCheckPrompt, PaperEditPrompt])
def test_math_rule_is_double_dollar_only(prompt: type[BasePrompt]) -> None:
    text = _render(prompt)
    assert "Write ALL math, inline and display, between $$ and $$" in text
    # The blog does not render \( \); no prompt may tell the model to use it.
    assert "Inline math: \\(" not in text
    assert "\\\\(" not in text


@pytest.mark.parametrize("prompt", WRITERS + [PaperSectionCheckPrompt])
def test_no_h1_rule(prompt: type[BasePrompt]) -> None:
    text = _render(prompt)
    assert "# Paper Title" not in text
    if prompt is not PaperFinalizationPrompt:
        assert "never `#`" in text or "Never write an H1" in text


def test_section_writer_carries_plan_revision_and_continuation_inputs() -> None:
    for var in (
        "section_brief",
        "plan_outline",
        "glossary",
        "owned_elsewhere",
        "previous_sections",
        "draft_to_revise",
        "current_explanation",
        "improvement_feedback",
    ):
        assert var in PaperSectionPrompt.input_variables


@pytest.mark.parametrize(
    "prompt", [PaperPlanningPrompt, PaperSectionPrompt, PaperSectionCheckPrompt]
)
def test_review_prompts_are_compact(prompt: type[BasePrompt]) -> None:
    # Guard against the old 47-80KB prompts creeping back.
    assert len(_render(prompt)) < 12_000


def test_reader_profile_is_shared() -> None:
    for prompt in (PaperPlanningPrompt, PaperSectionPrompt, PaperSectionCheckPrompt):
        assert "general data scientist or ML engineer" in _render(prompt)


def test_checker_outputs_only_what_is_parsed() -> None:
    assert PaperSectionCheckPrompt.output_variables == [
        "quality_score",
        "missing_items",
        "improvement_feedback",
    ]


def test_planner_asks_for_json_plan_with_ownership_and_figures() -> None:
    text = _render(PaperPlanningPrompt)
    for key in ('"must_cover"', '"owned_concepts"', '"figures"', '"sentence_ranges"'):
        assert key in text


def test_finalizer_outputs_tldr_closing_and_subtitle() -> None:
    assert set(PaperFinalizationPrompt.output_variables) == {
        "subtitle",
        "key_takeaways",
        "closing",
    }
    text = _render(PaperFinalizationPrompt)
    assert "### 이 연구를 시작하게 된 배경과 동기는 무엇입니까?" in text
    assert "## 한계와" in text


def test_attributes_prompt_defaults_vocabularies() -> None:
    text = _render(AttributesExtractionPrompt)
    for value in DEFAULT_CATEGORY_OPTIONS + DEFAULT_CONTROLLED_TAGS:
        assert value in text


def test_attributes_prompt_vocabularies_overridable() -> None:
    messages = AttributesExtractionPrompt.get_prompt().format_messages(
        text="t",
        existing_keywords="k",
        category_options="Only-Topic",
        controlled_tags="Only-Tag",
    )
    text = "\n".join(str(m.content) for m in messages)
    assert "Only-Topic" in text and "Only-Tag" in text
    assert "Finetuning" not in text


def test_summary_register_and_structure() -> None:
    text = _render(PaperSummaryPrompt)
    for heading in (
        "## 🔍 이 연구는 왜 필요합니까?",
        "## 💡 어떤 해결책을 제시합니까?",
        "## ⚙️ 제안한 방법은 어떻게 구현되었습니까?",
        "## 📊 핵심 실험 결과는 무엇입니까?",
        "## 🔮 이 연구의 의의와 한계는 무엇입니까?",
    ):
        assert heading in text
    assert "> **요약:**" in text
