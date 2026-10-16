"""Invariants of the generation prompts (rendered text, not LLM output)."""

from __future__ import annotations

import pytest

from scholar_lens.src.markdown_lint import style_hits, writing_issues
from scholar_lens.src.prompts import (
    AttributesExtractionPrompt,
    PaperEditPrompt,
    PaperFinalizationPrompt,
    PaperPlanningPrompt,
    PaperSectionCheckPrompt,
    PaperSectionPrompt,
    PaperSummaryPrompt,
    PaperSummaryShortenPrompt,
    SurveyCardPrompt,
    SurveyEvaluationPrompt,
    SurveyFinalizationPrompt,
    SurveyGroundingPrompt,
    SurveySectionPrompt,
    TechGuideEvaluationPrompt,
    TechGuideGroundingPrompt,
    TechGuideSectionPrompt,
)
from scholar_lens.src.prompts.base import (
    KOREAN_STYLE,
    NATURALNESS_CHECK,
    NUMBER_RULES,
    PAPER_TABLE_RULES,
    TABLE_SIZE_RULE,
    TERMINOLOGY_RULES,
    BasePrompt,
)
from scholar_lens.src.prompts.extraction import (
    DEFAULT_CATEGORY_OPTIONS,
    DEFAULT_CONTROLLED_TAGS,
)
from scholar_lens.src.prompts.tech_guide import TechGuideTaxonomyPrompt

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


@pytest.mark.parametrize("prompt", WRITERS + [PaperEditPrompt])
def test_math_rule_is_double_dollar_only(prompt: type[BasePrompt]) -> None:
    text = _render(prompt)
    assert "Write all math, inline and display, between $$ and $$" in text
    # The blog does not render \( \); no prompt may tell the model to use it.
    assert "Inline math: \\(" not in text
    assert "\\\\(" not in text


def test_checker_leaves_mechanical_rules_to_code() -> None:
    text = _render(PaperSectionCheckPrompt)
    assert "checked in code" in text
    assert "Write all math" not in text


@pytest.mark.parametrize("prompt", WRITERS)
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
        "must_fix",
        "improvement_feedback",
    ]


def test_planner_asks_for_json_plan_with_ownership_and_figures() -> None:
    text = _render(PaperPlanningPrompt)
    for key in ('"must_cover"', '"owned_concepts"', '"figures"', '"sentence_ranges"'):
        assert key in text


def test_finalizer_outputs_tldr_closing_and_subtitle() -> None:
    assert set(PaperFinalizationPrompt.output_variables) == {
        "subtitle",
        "description",
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
        "## 연구 배경",
        "## 제안 방법",
        "## 구현 방식",
        "## 실험 결과",
        "## 의의와 한계",
    ):
        assert heading in text


@pytest.mark.parametrize(
    "prompt",
    [
        PaperPlanningPrompt,
        PaperSectionPrompt,
        PaperSectionCheckPrompt,
        PaperEditPrompt,
        PaperFinalizationPrompt,
        PaperSummaryPrompt,
        TechGuideSectionPrompt,
        TechGuideEvaluationPrompt,
    ],
)
def test_one_terminology_rule_is_shared(prompt: type[BasePrompt]) -> None:
    text = _render(prompt)
    assert text.count(TERMINOLOGY_RULES.strip()) == 1


def test_terminology_rule_examples_agree_with_the_stem_lint() -> None:
    # The rule's bad example is what the lint flags; its fixes are not.
    assert writing_issues("결과를 cache합니다.")
    assert not writing_issues("결과를 캐시에 저장합니다.")


def test_translation_guideline_wins_over_the_general_term_rule() -> None:
    # The guideline's curated renderings (some transliterated) take precedence.
    assert "translation guideline lists takes its rendering" in TERMINOLOGY_RULES


# Every prompt that writes or rewrites Korean prose carries the one Korean
# style block; the checkers also carry the naturalness check that makes its
# worst breaks blocking.
KOREAN_WRITERS = [
    PaperSectionPrompt,
    PaperEditPrompt,
    PaperFinalizationPrompt,
    PaperSummaryPrompt,
    SurveyCardPrompt,
    SurveySectionPrompt,
    SurveyGroundingPrompt,
    TechGuideSectionPrompt,
    TechGuideGroundingPrompt,
]
CHECKERS = [PaperSectionCheckPrompt, SurveyEvaluationPrompt, TechGuideEvaluationPrompt]


@pytest.mark.parametrize("prompt", KOREAN_WRITERS + CHECKERS)
def test_one_korean_style_block_is_shared(prompt: type[BasePrompt]) -> None:
    assert _render(prompt).count(KOREAN_STYLE.strip()) == 1


@pytest.mark.parametrize("prompt", CHECKERS)
def test_checkers_carry_the_naturalness_check(prompt: type[BasePrompt]) -> None:
    assert _render(prompt).count(NATURALNESS_CHECK.strip()) == 1


@pytest.mark.parametrize("prompt", [SurveyEvaluationPrompt, TechGuideEvaluationPrompt])
def test_guide_and_survey_evaluators_list_wording_apart_from_facts(
    prompt: type[BasePrompt],
) -> None:
    # Wording issues are ranked after the score, blocking issues before it.
    text = _render(prompt)
    assert prompt.output_variables == [
        "quality_score",
        "blocking_issues",
        "wording_issues",
        "improvement_feedback",
    ]
    assert "blocking wording issues go in <wording_issues>" in " ".join(text.split())
    assert "<wording_issues>one blocking wording issue per line" in text


def test_review_checker_lists_blocking_wording_in_must_fix() -> None:
    # The review loop ranks every must-fix item after the score.
    assert "- a blocking wording issue of the naturalness check." in _render(
        PaperSectionCheckPrompt
    )


def test_naturalness_blocks_only_meaning_and_plain_da_form() -> None:
    check = " ".join(NATURALNESS_CHECK.split())
    assert "grammar or agent changes or obscures its meaning" in check
    assert "plain 다-form prose sentence outside quotations and code" in check
    assert "Everything else is feedback" in check and "at most" not in check
    # A table cell is a noun phrase or an 입니다 sentence, one form per table.
    assert "one form per table" in " ".join(KOREAN_STYLE.split())


def test_figure_introductions_carry_no_template_sentence() -> None:
    from scholar_lens.src.prompts.base import VISUAL_RULES

    rules = " ".join(VISUAL_RULES.split())
    assert "아래 그림" not in rules
    assert "never open two figure introductions the same way" in rules
    assert "never describe a panel it lacks" in rules


def test_shared_style_examples_are_topic_neutral() -> None:
    # Examples from one post's topic (a survey of RL, a vLLM guide) pull
    # every post toward its wording.
    from scholar_lens.src.prompts import SurveyOutlinePrompt

    shared = KOREAN_STYLE + TERMINOLOGY_RULES
    for topical in ("Gao", "정책", "가중치", "prefill"):
        assert topical not in shared
    assert "preference optimization" not in _render(SurveyOutlinePrompt)
    for clause in ("~다는 점/사실", "에게", "셈입니다"):
        assert clause in KOREAN_STYLE


def test_cards_get_the_translation_guideline() -> None:
    assert "<translation_guideline>" in _render(SurveyCardPrompt)


@pytest.mark.parametrize(
    "prompt", [SurveyFinalizationPrompt, PaperSummaryShortenPrompt]
)
def test_short_rewrites_ask_for_natural_prose(prompt: type[BasePrompt]) -> None:
    assert "translation" in _render(prompt)


def test_korean_style_examples_agree_with_the_lint() -> None:
    # The block's double passive, calque and 다-colon are what the blog's
    # prose lint flags; their rewrites are not.
    assert [
        h.rule for h in style_hits("값이 되어진다.\n\n학습에 있어서는 씁니다.")
    ] == [
        "translationese",
        "translationese",
    ]
    assert [h.rule for h in style_hits("단계는 셋입니다:\n")] == ["colon"]
    assert not style_hits("값이 된다. 학습에서 씁니다. 단계는 셋입니다.\n")
    for example in ("되어진다", "에 있어서", '"셋입니다:"', '"셋입니다."'):
        assert example in KOREAN_STYLE


@pytest.mark.parametrize("prompt", [SurveyEvaluationPrompt, TechGuideEvaluationPrompt])
def test_survey_and_guide_evaluators_check_the_number_rules(
    prompt: type[BasePrompt],
) -> None:
    assert NUMBER_RULES in _render(prompt)


def test_the_table_size_cap_has_one_source() -> None:
    assert PAPER_TABLE_RULES.startswith(TABLE_SIZE_RULE)
    assert _render(PaperSectionPrompt).count("8 rows and 7 columns") == 1


def test_survey_evaluator_blocks_re_explaining_earlier_sections() -> None:
    blocking = _render(SurveyEvaluationPrompt).split("BLOCKING ISSUES", 1)[1]
    assert "re-explaining what <previously_written_sections>" in blocking


def test_card_limitations_are_the_papers_not_the_inputs() -> None:
    text = _render(SurveyCardPrompt)
    assert "never of the text you were given" in text
    assert "leave limitations empty" in text


def test_guide_taxonomy_returns_a_description() -> None:
    assert "description" in TechGuideTaxonomyPrompt.output_variables
    assert "120-160 characters" in _render(TechGuideTaxonomyPrompt)


def test_survey_citation_example_uses_no_real_paper_key() -> None:
    assert '"[@key]"' in _render(SurveySectionPrompt)
