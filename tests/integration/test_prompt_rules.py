"""Invariants of the generation prompts (rendered text, not LLM output)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

from scholar_lens.adapters.bedrock.prompts.base import BasePrompt
from scholar_lens.adapters.bedrock.prompts.extraction import (
    AttributesExtractionPrompt,
)
from scholar_lens.adapters.bedrock.prompts.survey import SurveyFinalizationPrompt
from scholar_lens.adapters.bedrock.prompts.tech_guide import TechGuideTaxonomyPrompt
from scholar_lens.adapters.conventions_file import load_local_conventions
from scholar_lens.domain.markdown.lint import style_hits, writing_issues
from scholar_lens.domain.prompts.rules import (
    DESCRIPTION_CHARS,
    KOREAN_STYLE,
    MATH_RULES,
    NATURALNESS_CHECK,
    PAPER_TABLE_RULES,
    READER_PROFILE,
    TABLE_SIZE_RULE,
    TERMINOLOGY_RULES,
)
from scholar_lens.sections_engine.prompts.review import (
    PaperEditPrompt,
    PaperFinalizationPrompt,
    PaperPlanningPrompt,
    PaperSectionCheckPrompt,
    PaperSectionPrompt,
)
from scholar_lens.sections_engine.prompts.summary import (
    PaperSummaryPrompt,
    PaperSummaryShortenPrompt,
)

WRITERS = [
    PaperSectionPrompt,
    PaperFinalizationPrompt,
    PaperSummaryPrompt,
]


def _render(prompt: type[BasePrompt]) -> str:
    template = prompt.get_prompt()
    messages = template.format_messages(**{v: "X" for v in template.input_variables})
    return "\n".join(str(m.content) for m in messages)


@pytest.mark.parametrize("prompt", WRITERS + [PaperEditPrompt])
def test_math_rule_is_double_dollar_only(prompt: type[BasePrompt]) -> None:
    text = _render(prompt)
    assert MATH_RULES.strip().splitlines()[0] in text
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
        assert READER_PROFILE in _render(prompt)


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
    bundled = load_local_conventions()
    for value in [*bundled.topics, *bundled.tags.controlled]:
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
]
CHECKERS = [PaperSectionCheckPrompt]


@pytest.mark.parametrize("prompt", KOREAN_WRITERS + CHECKERS)
def test_one_korean_style_block_is_shared(prompt: type[BasePrompt]) -> None:
    assert _render(prompt).count(KOREAN_STYLE.strip()) == 1


@pytest.mark.parametrize("prompt", CHECKERS)
def test_checkers_carry_the_naturalness_check(prompt: type[BasePrompt]) -> None:
    assert _render(prompt).count(NATURALNESS_CHECK.strip()) == 1


def test_review_checker_lists_blocking_wording_in_must_fix() -> None:
    # The review loop ranks every must-fix item after the score.
    assert "- a blocking wording issue of the naturalness check." in _render(
        PaperSectionCheckPrompt
    )


def test_figure_introductions_carry_no_template_sentence() -> None:
    from scholar_lens.domain.prompts.rules import VISUAL_RULES

    rules = " ".join(VISUAL_RULES.split())
    assert "아래 그림" not in rules
    assert "never open two figure introductions the same way" in rules
    assert "never describe a panel it lacks" in rules


def test_shared_style_examples_are_topic_neutral() -> None:
    # Examples from one post's topic (a survey of RL, a vLLM guide) pull
    # every post toward its wording.
    shared = KOREAN_STYLE + TERMINOLOGY_RULES
    for topical in ("Gao", "정책", "가중치", "prefill"):
        assert topical not in shared
    for clause in ("~다는 점/사실", "에게", "셈입니다"):
        assert clause in KOREAN_STYLE


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


def test_the_table_size_cap_has_one_source() -> None:
    assert PAPER_TABLE_RULES.startswith(TABLE_SIZE_RULE)
    assert _render(PaperSectionPrompt).count("8 rows and 7 columns") == 1


def test_guide_taxonomy_returns_a_description() -> None:
    assert "description" in TechGuideTaxonomyPrompt.output_variables
    assert DESCRIPTION_CHARS in _render(TechGuideTaxonomyPrompt)


# ---- prompt examples stay off the evaluation set ------------------------------

_ROOT = Path(__file__).resolve().parents[2]
_PROMPT_DIRS = (
    "scholar_lens/domain/prompts",
    "scholar_lens/sections_engine/prompts",
    "scholar_lens/adapters/bedrock/prompts",
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def eval_item_names() -> set[str]:
    """The names of the evaluation set's subjects, read from
    eval/sets/baseline.yaml: an item key's subject (``guide-vllm`` → vllm)
    and its docs host (``docs.ray.io`` → ray); for a survey, the plan's
    paper names (a title's head before its colon, up to three words) and
    the key's subject when the plan's topic writes it as a name (an
    acronym, not a common noun such as "security")."""
    items = yaml.safe_load((_ROOT / "eval/sets/baseline.yaml").read_text())["items"]
    names: set[str] = set()
    for key, item in items.items():
        subject = key.split("-", 1)[1]
        if plan_file := item.get("plan"):
            plan = json.loads((_ROOT / plan_file).read_text())
            topic_words = re.findall(r"[A-Za-z0-9]+", plan["topic"])
            if any(w.lower() == subject and w != w.lower() for w in topic_words):
                names.add(subject)
            for paper in plan["papers"]:
                head, colon, _ = paper["title"].partition(":")
                if colon and len(head.split()) <= 3:
                    names.add(head)
        else:
            names.add(subject)
        for url in item.get("urls", []):
            names.add((urlparse(url).hostname or "").split(".")[-2])
    return names


def _mentions(words: list[str], name: str) -> bool:
    """Whether ``name`` occurs in ``words`` as whole words, or spelled
    together ("rayserve" for "Ray Serve")."""
    wanted = _words(name)
    joined = "".join(wanted)
    return any(words[i : i + len(wanted)] == wanted for i in range(len(words))) or any(
        "".join(words[i : i + k]) == joined
        for k in (2, 3)
        for i in range(len(words) - k + 1)
    )


def test_the_eval_item_names_are_read_from_the_set() -> None:
    names = {n.lower() for n in eval_item_names()}
    assert {"vllm", "dpo", "lora", "ray", "rlhf", "struq"} <= names
    assert "security" not in names


@pytest.mark.parametrize(
    "module",
    sorted(p for d in _PROMPT_DIRS for p in (_ROOT / d).glob("*.py")),
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_prompt_modules_name_no_eval_item(module: Path) -> None:
    """Prompt examples are invented (docs/engine.md 2.2): a prompt that
    names an evaluation subject pulls every post toward it and makes the
    evaluation measure the example."""
    names = eval_item_names()
    hits = [
        f"{n}: line {number}"
        for number, line in enumerate(module.read_text().splitlines(), 1)
        for n in sorted(names)
        if _mentions(_words(line), n)
    ]
    assert not hits, hits
