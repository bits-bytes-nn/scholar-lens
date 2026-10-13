"""Deterministic writing-rule detectors, Korean alt text and review budgets."""

from __future__ import annotations

import json

import pytest

from scholar_lens.src.markdown_lint import (
    NUMBERED_REF,
    fill_image_alts,
    limit_callouts,
    lint_markdown,
    mostly_hangul,
    writing_issues,
)
from scholar_lens.src.post_conventions import load_local_conventions
from scholar_lens.src.post_validator import validate_post
from scholar_lens.src.review_plan import (
    HARD_OVERRUN,
    MIN_HARD_SCALE,
    MIN_SECTION_WORDS,
    REVIEW_WORD_BUDGET,
    CoverItem,
    Edit,
    apply_edits,
    article_ledger,
    normalize_plan,
    parse_plan,
    review_word_budget,
    target_words,
)


@pytest.mark.parametrize(
    "text,flagged",
    [
        ("그림 3을 보면", True),
        ("3.2절에서", True),
        ("2절의 결과", True),
        ("Table 2 shows", True),
        ("4.1장에서", True),
        ("GPU 2장으로 학습합니다", False),
        ("이미지 4장을 씁니다", False),
        ("평가 지표 2개", False),
        ("2절반", False),
        ("수식 2개를 씁니다", False),
        ("그림 3장을 붙입니다", False),
        ("최대 2/3 절감합니다", False),
        ("§3.2에서", True),
        ("3.2장에서", True),
        ("Ubuntu 22.04 장기 지원", False),
        ("CUDA 12.1 장치 드라이버", False),
        ("vLLM 0.6.3 장애 사례", False),
        ("1.5 장점이 있습니다", False),
        ("유휴 GPU를 1 절전 모드로 전환합니다", False),
        ("배치 크기를 16 절삭 없이 유지합니다", False),
        ("그림 2번째 행을 보세요", False),
        ("표 3줄을 추가합니다", False),
        ("3절", True),
        ("3절과 4절", True),
        ("2.1절까지", True),
        ("그림 2", True),
        ("그림 2a에서", True),
        ("표 1, 표 2", True),
        ("수식 (3)을 풀면", True),
        ("표 2를 보면", True),
        ("그림 1보다 작습니다", True),
    ],
)
def test_numbered_ref_ignores_counters(text: str, flagged: bool) -> None:
    assert bool(NUMBERED_REF.search(text)) is flagged


class TestWritingIssues:
    def test_each_rule_is_detected(self) -> None:
        md = (
            "## 방법\n\n이 절에서는 모델을 fine-tune합니다. 그림 3과 같이 "
            "$x_1$ 값을 씁니다.\n\n![Low-rank update](/assets/images/a-1.png)\n"
        )
        issues = "\n".join(writing_issues(md))
        assert "self-reference '이 절'" in issues
        assert "'fine-tune합니다'" in issues
        assert "numbered reference '그림 3'" in issues
        assert "single-$ math" in issues
        assert "Low-rank update" not in issues  # alts are fixed downstream

    @pytest.mark.parametrize(
        "text,needle",
        [
            ("이 섹션에서 봅니다.", "self-reference '이 섹션'"),
            ("저장소를 fork할 수 있습니다.", "'fork할'"),
            ("기저가 공간을 span하는 방식입니다.", "'span하는'"),
            ("자기 토큰에만 attention하도록 합니다.", "'attention하도록'"),
            ("일부 토큰만 prefill하면서 진행합니다.", "'prefill하면서'"),
            ("요청은 swap-in되어 돌아옵니다.", "'swap-in되어'"),
            ("첫 step에 일부만 prefill되고 끝납니다.", "'prefill되고'"),
            ("같은 접두사를 prefill하므로 느립니다.", "'prefill하므로'"),
        ],
    )
    def test_more_slips_are_caught(self, text: str, needle: str) -> None:
        assert needle in "\n".join(writing_issues(f"## 방법\n\n{text}"))

    @pytest.mark.parametrize(
        "text",
        [
            "API를 호출하고 vLLM이 응답합니다.",
            "Claude Code한테 맡깁니다.",
            "`prefill`하면서 `swap-in`되어 돌아옵니다.",
            "[문서](https://docs.vllm.ai/prefill하면)를 봅니다.",
            "<https://example.com/a하는> 링크입니다.",
            "행렬을 SVD하면 됩니다.",
        ],
    )
    def test_stem_check_leaves_proper_usage_alone(self, text: str) -> None:
        assert writing_issues(f"## 방법\n\n{text}") == []

    def test_an_acronym_used_as_a_verb_is_fine(self) -> None:
        assert writing_issues("행렬을 SVD하고 비교합니다.") == []

    def test_escaped_dollar_is_not_math(self) -> None:
        assert writing_issues("매출은 \\$1.2B이고 비용은 \\$3_000입니다.") == []

    def test_code_math_and_clean_prose_are_ignored(self) -> None:
        md = (
            "## 방법\n\n이 장점은 분명합니다. LoRA를 적용합니다.\n\n"
            "```python\n# fine-tune합니다, 그림 3\n```\n\n`merge`하는 단계와 "
            "$$W_0$$를 봅니다. 비용은 $5입니다.\n\n![저랭크 갱신](/a.png)\n"
        )
        assert writing_issues(md) == []

    def test_callouts_over_the_cap_become_paragraphs(self) -> None:
        callouts = "\n\n".join(
            f"> **핵심:** 요점 {i}입니다.\n> 이어지는 줄 {i}." for i in range(5)
        )
        out = limit_callouts(f"본문입니다.\n\n{callouts}\n\n끝.")
        assert out.count("> **핵심:**") == 3
        assert "\n\n요점 3입니다.\n이어지는 줄 3.\n\n요점 4" in out
        assert out.endswith("끝.")
        assert lint_markdown(out) == lint_markdown(lint_markdown(out))
        fenced = "```\n" + callouts + "\n```"
        assert limit_callouts(fenced) == fenced

    def test_a_quote_opening_with_bold_text_is_not_a_callout(self) -> None:
        labelled = "\n\n".join(f"> **핵심:** 요점 {i}입니다." for i in range(3))
        quotes = "> **PagedAttention**은 KV cache를 나눕니다.\n\n> **에이전트 = 모델 + 하네스**"
        md = f"{labelled}\n\n{quotes}"
        assert limit_callouts(md) == md

    def test_alt_text_is_not_a_writer_slip(self) -> None:
        # fill_image_alts owns alt text; a writer can't be blocked on it.
        assert (
            writing_issues("본문입니다.\n\n![Figure 3: 구조](/assets/images/x-1.png)\n")
            == []
        )

    def test_link_targets_are_not_reworded(self) -> None:
        md = (
            "그림입니다.\n\n![전체 구조도](/p/figures/Figure1.png)\n\n"
            "[공식 문서](https://x.org/a#Section3)와 <https://x.org/Table2>, "
            "https://x.org/Figure_2 참고."
        )
        assert writing_issues(md) == []
        assert writing_issues("Theorem 2와 Lemma 1에 따르면") != []

    def test_english_post_alt_is_not_flagged(self) -> None:
        assert mostly_hangul("LoRA는 Transformer weight를 freeze") is False
        assert writing_issues("Body text.\n\n![Low-rank update](/a.png)") == []


class TestKoreanAlts:
    CAPTIONS = {"a-1.png": "Figure 1: Our reparametrization."}

    def test_question_heading_beats_an_english_caption(self) -> None:
        md = (
            "## 💡 어떤 해결책을 제시합니까?\n\n본문은 한국어로 씁니다.\n\n"
            "![](/assets/images/a-1.png)\n"
        )
        out = fill_image_alts(md, self.CAPTIONS, "LoRA")
        assert "![어떤 해결책을 제시합니까](/assets/images/a-1.png)" in out

    def test_heading_alt_drops_emoji(self) -> None:
        md = "## 📊 저랭크 분해\n\n본문은 한국어로 씁니다.\n\n![Refer](/a.png)\n"
        assert "![저랭크 분해](/a.png)" in fill_image_alts(md, {}, "LoRA")

    def test_english_alt_falls_back_to_the_heading(self) -> None:
        md = (
            "## 저랭크 분해\n\n본문은 한국어로 씁니다.\n\n![Refer to caption](/a.png)\n"
        )
        assert "![저랭크 분해](/a.png)" in fill_image_alts(md, {}, "LoRA")

    def test_korean_alt_and_english_posts_keep_their_alt(self) -> None:
        md = "## 방법\n\n본문입니다.\n\n![저랭크 구조](/a.png)"
        assert fill_image_alts(md, {}, "x") == md
        english = "## Method\n\nBody.\n\n![](/assets/images/a-1.png)"
        assert "![Our reparametrization.]" in fill_image_alts(
            english, self.CAPTIONS, "x"
        )

    def test_validator_warns_on_english_alt_in_a_korean_post(self) -> None:
        post = (
            '---\nlayout: post\ntitle: "T"\n---\n## TL;DR\n\n한국어 본문입니다. '
            "충분히 긴 한국어 문장입니다.\n\n![Low-rank update](/assets/images/x-1.png)\n"
        )
        found = validate_post(post, load_local_conventions(), "review")
        assert any(v.rule == "image-alt" and v.level == "warning" for v in found)


def _plan(sections: list[dict[str, object]]) -> str:
    plan = {"sections": sections}
    return f"<plan>{json.dumps(plan, ensure_ascii=False)}</plan>"


def _normalized(raw: str):  # type: ignore[no-untyped-def]
    return normalize_plan(
        parse_plan(raw), sentence_count=50, figure_ids={}, max_sections=20
    )


class TestReviewBudget:
    def _sections(self, difficulty: str, n: int = 8) -> list[dict[str, object]]:
        return [
            {
                "title": f"s{i}",
                "importance": "core",
                "must_cover": [
                    {"item": f"i{i}-{k}", "difficulty": difficulty} for k in range(4)
                ],
            }
            for i in range(n)
        ]

    def test_targets_scaled_to_the_budget_with_the_floor_kept(self) -> None:
        plan = _normalized(_plan(self._sections("medium")))
        total = sum(s.target_words for s in plan.sections)
        assert total <= review_word_budget(plan)
        assert all(s.target_words >= MIN_SECTION_WORDS for s in plan.sections)

    def test_mostly_hard_papers_get_more_room(self) -> None:
        medium = _normalized(_plan(self._sections("medium")))
        hard = _normalized(_plan(self._sections("hard")))
        total = lambda p: sum(s.target_words for s in p.sections)  # noqa: E731
        assert total(hard) > total(medium)

    def test_hard_items_are_kept_within_the_overrun(self) -> None:
        sections = self._sections("medium") + self._sections("hard", n=2)
        plan = _normalized(_plan(sections))
        assert plan.sections[0].length_scale < 1.0
        hard = plan.sections[-1]
        assert hard.target_words == target_words(hard.must_cover, "core")

    def test_hard_items_shrink_a_bounded_amount_far_over_budget(self) -> None:
        plan = _normalized(_plan(self._sections("hard", n=12)))
        total = sum(s.target_words for s in plan.sections)
        assert all(s.hard_scale == pytest.approx(MIN_HARD_SCALE) for s in plan.sections)
        assert total > HARD_OVERRUN * review_word_budget(plan)  # bounded shrink
        mild = _normalized(_plan(self._sections("hard", n=7)))
        assert MIN_HARD_SCALE < mild.sections[0].hard_scale < 1.0
        mild_total = sum(s.target_words for s in mild.sections)
        assert mild_total <= HARD_OVERRUN * review_word_budget(mild)

    def test_budget_grows_smoothly_with_the_hard_share(self) -> None:
        def budget(hard: int) -> int:
            items = [
                {"item": f"i{k}", "difficulty": "hard" if k < hard else "medium"}
                for k in range(100)
            ]
            raw = _plan([{"title": "s", "must_cover": items}])
            return review_word_budget(parse_plan(raw))

        assert budget(0) == REVIEW_WORD_BUDGET
        assert 0 < budget(51) - budget(49) < 200
        assert budget(100) > budget(51)

    def test_scale_has_a_floor(self) -> None:
        easy = [CoverItem(item=f"e{k}", difficulty="easy") for k in range(10)]
        assert target_words(easy, "core", 0.5) == round(10 * 60 * 0.5 * 1.3)

    def test_small_plans_are_not_scaled(self) -> None:
        plan = _normalized(_plan(self._sections("easy", n=2)))
        assert all(s.length_scale == 1.0 for s in plan.sections)


class TestEditsAndLedger:
    ARTICLE = (
        "## 결과\n\n결과 문장입니다.\n\n### 신뢰도 주의\n\n주의 문장입니다.\n\n"
        "## 다음\n\n다음 문장입니다."
    )

    def test_subsections_can_be_removed_but_top_level_headings_cannot(self) -> None:
        drop_sub = Edit(find="### 신뢰도 주의\n\n주의 문장입니다.\n\n", replace="")
        text, applied, _ = apply_edits(self.ARTICLE, [drop_sub], max_edits=5)
        assert applied == 1 and "### 신뢰도 주의" not in text
        drop_top = Edit(find="## 다음\n\n다음 문장입니다.", replace="다음 문장입니다.")
        _, applied, skipped = apply_edits(self.ARTICLE, [drop_top], max_edits=5)
        assert applied == 0 and "heading" in skipped[0]

    def test_ledger_lists_code_blocks_by_first_line(self) -> None:
        section = "## 구현\n\n```python\n\nimport torch\nx = 1\n```\n\n```\nb\n```\n"
        ledger = article_ledger([section])
        assert "- code: import torch" in ledger and "- code: b" in ledger

    def test_ledger_lists_callouts(self) -> None:
        ledger = article_ledger(["## A\n\n> **핵심:** 요점입니다.\n\n> 인용문입니다."])
        assert "- callout: **핵심:** 요점입니다." in ledger
        assert "인용문" not in ledger


class TestEditGuards:
    ARTICLE = (
        "## 방법\n\n설명입니다.\n\n$$\nh = W_0 x\n$$\n\n![구조](/assets/images/x-1.png)\n\n"
        "### 세부\n\n#### 더 세부\n\n내용입니다.\n\n"
        "| 방법 | 점수 |\n| --- | --- |\n| LoRA | 1 |\n\n## 결과\n\n끝입니다."
    )

    @pytest.mark.parametrize(
        "find,reason",
        [
            ("$$\nh = W_0 x\n$$\n\n", "only equation"),
            ("![구조](/assets/images/x-1.png)\n\n", "only figure"),
            ("| 방법 | 점수 |\n| --- | --- |\n| LoRA | 1 |\n\n", "only table"),
        ],
    )
    def test_structure_is_protected(self, find: str, reason: str) -> None:
        text, applied, skipped = apply_edits(
            self.ARTICLE, [Edit(find=find, replace="")], max_edits=5
        )
        assert applied == 0 and text == self.ARTICLE and reason in skipped[0]

    def test_a_repeated_equation_may_be_dropped(self) -> None:
        article = self.ARTICLE.replace("끝입니다.", "$$\nh = W_0 x\n$$\n\n끝입니다.")
        find = "## 결과\n\n$$\nh = W_0 x\n$$\n\n"
        _, applied, _ = apply_edits(
            article, [Edit(find=find, replace="## 결과\n\n")], max_edits=5
        )
        assert applied == 1

    @pytest.mark.parametrize(
        "replace", ["### 결과\n\n끝입니다.", "## 결과와 해석\n\n끝입니다."]
    )
    def test_a_top_heading_cannot_be_demoted_or_renamed(self, replace: str) -> None:
        edit = Edit(find="## 결과\n\n끝입니다.", replace=replace)
        _, applied, skipped = apply_edits(self.ARTICLE, [edit], max_edits=5)
        assert applied == 0 and "heading" in skipped[0]

    @pytest.mark.parametrize(
        "find,replace",
        [
            ("끝입니다.", "끝입니다.\n\n## 새 절\n\n추가입니다."),  # added
            ("방법\n\n설명입니다.", "방법론\n\n설명입니다."),  # renamed mid-line
        ],
    )
    def test_top_headings_cannot_be_added_or_renamed_mid_line(
        self, find: str, replace: str
    ) -> None:
        edit = Edit(find=find, replace=replace)
        _, applied, skipped = apply_edits(self.ARTICLE, [edit], max_edits=5)
        assert applied == 0 and "heading" in skipped[0]

    def test_a_whole_equation_may_be_fixed_in_place(self) -> None:
        fix = Edit(find="$$\nh = W_0 x\n$$", replace="$$\nh = W_0 x + BAx\n$$")
        text, applied, _ = apply_edits(self.ARTICLE, [fix], max_edits=5)
        assert applied == 1 and "BAx" in text

    def test_a_table_with_aligned_separator_is_protected(self) -> None:
        article = self.ARTICLE.replace("| --- | --- |", "|:-|-:|")
        find = "| 방법 | 점수 |\n|:-|-:|\n| LoRA | 1 |\n\n"
        _, applied, skipped = apply_edits(
            article, [Edit(find=find, replace="")], max_edits=5
        )
        assert applied == 0 and "only table" in skipped[0]


@pytest.mark.parametrize("raw", ["none.", "None", "없음.", "- none"])
def test_parse_lines_none_with_punctuation(raw: str) -> None:
    from scholar_lens.src.review_plan import parse_lines

    assert parse_lines(raw) == []


@pytest.mark.parametrize("answer", ["**없음**", "*None*", "_none_"])
def test_emphasised_none_answers_are_empty(answer: str) -> None:
    from scholar_lens.src.review_plan import parse_lines

    assert parse_lines(answer) == []


def test_config_file_sections_are_not_self_references() -> None:
    from scholar_lens.src.markdown_lint import writing_issues

    text = "`config.yaml`에 다음 섹션을 추가합니다.\n\n다음 섹션에서 설치를 다룹니다."
    hits = [h for h in writing_issues(text) if "self" in h.lower()]
    assert len(hits) == 1


def test_html_attribute_text_is_not_prose() -> None:
    from scholar_lens.src.markdown_lint import writing_issues

    html = (
        '<a data-glightbox="title: 구조 (출처: 논문, Figure 1)" href="/x.png">그림</a>'
    )
    assert writing_issues(html) == []


def test_edit_that_adds_a_rule_slip_is_skipped() -> None:
    from scholar_lens.src.review_plan import Edit, apply_edits

    text = "## 방법\n\n결과를 정리합니다."
    out, applied, skipped = apply_edits(
        text,
        [Edit(find="결과를 정리합니다.", replace="Table 2에서 결과를 정리합니다.")],
        max_edits=3,
    )
    assert applied == 0 and out == text and "slip" in skipped[0]


def test_skipped_slip_edit_names_the_new_slip() -> None:
    from scholar_lens.src.review_plan import Edit, apply_edits

    text = "## 방법\n\n결과를 정리합니다."
    _, _, skipped = apply_edits(
        text,
        [Edit(find="결과를 정리합니다.", replace="이 절에서 결과를 정리합니다.")],
        max_edits=1,
    )
    assert "self-reference '이 절'" in skipped[0]


def test_copied_figure_marker_is_one_clear_issue() -> None:
    from scholar_lens.src.markdown_lint import writing_issues

    text = (
        '본문입니다.\n[Image: alt="Figure 3: curve", src="https://arxiv.org/html/1/x3.png", '
        'caption="Loss curve (Table 2)."]'
    )
    issues = writing_issues(text)
    assert len(issues) == 1 and "write a Markdown image" in issues[0]


@pytest.mark.parametrize(
    "text,flagged",
    [
        ("이 절을 마치며 정리합니다.", True),
        ("이 절이 다루는 내용입니다.", True),
        ("이 섹션에서 설명합니다.", True),
        ("`config.yaml`에 다음 섹션을 추가합니다.", False),
        ("이 장을 넘기면 됩니다.", False),
    ],
)
def test_self_reference_endings(text: str, flagged: bool) -> None:
    from scholar_lens.src.markdown_lint import writing_issues

    assert any("self-reference" in i for i in writing_issues(text)) is flagged


@pytest.mark.parametrize(
    "text,flagged",
    [
        ("수식 (3)에서 보듯이 손실이 줄어듭니다.", True),
        ("식 (3)을 보면 됩니다.", True),
        ("식 3을 정리합니다.", True),
        ("식 3개를 비교합니다.", False),
        ("공식 3을 씁니다.", False),
        ("그림 3장을 붙였습니다.", False),
    ],
)
def test_korean_equation_references(text: str, flagged: bool) -> None:
    from scholar_lens.src.markdown_lint import writing_issues

    assert any("numbered reference" in i for i in writing_issues(text)) is flagged


@pytest.mark.parametrize(
    "answer",
    [
        "1. None",
        "- None —",
        "none (all covered)",
        "2) 없음",
        "None — all items are covered",
        "없음: 모두 반영됨",
        "Nothing missing.",
        "N/A",
        "해당 없음",
        "없습니다",
        "- 없음 (모두 반영됨)",
        "none;",
    ],
)
def test_numbered_and_annotated_none_answers_are_empty(answer: str) -> None:
    from scholar_lens.src.review_plan import parse_lines

    assert parse_lines(answer) == []


def test_real_items_with_parentheticals_are_kept() -> None:
    from scholar_lens.src.review_plan import parse_lines

    assert parse_lines("- ablation (Table 3)") == ["ablation (Table 3)"]


def test_writing_notes_count_the_hits_past_the_cap() -> None:
    from scholar_lens.src.markdown_lint import MAX_WRITING_NOTES, writing_notes

    text = "\n\n".join(f"Table {i}에서 보듯 좋습니다." for i in range(12))
    text += "\n\n이 절에서 정리합니다."
    notes = writing_notes(text)
    assert len(notes) == MAX_WRITING_NOTES + 1
    assert notes[-1].startswith("…and 3 more")
    assert "self-reference" in notes[-1] and "numbered reference" in notes[-1]


def test_edit_that_trades_one_slip_for_another_is_skipped() -> None:
    from scholar_lens.src.review_plan import Edit, apply_edits

    text = "## 방법\n\nTable 2에서 결과를 정리합니다."
    edit = Edit(find="Table 2에서 결과를", replace="이 절에서 결과를")
    out, applied, skipped = apply_edits(text, [edit], max_edits=3)
    assert applied == 0 and out == text and "self-reference" in skipped[0]


def test_config_section_is_not_a_self_reference() -> None:
    from scholar_lens.src.markdown_lint import writing_issues

    assert writing_issues("설정 파일의 다음 섹션에 키를 추가합니다.") == []
    assert writing_issues("이 섹션에서는 설치를 다룹니다.")


def test_callout_lines_ignore_code() -> None:
    from scholar_lens.src.markdown_lint import callout_lines

    text = "> **핵심:** a\n\n```\n> **핵심:** b\n```\n"
    assert callout_lines(text) == ["**핵심:** a"]


@pytest.mark.parametrize(
    "text,closed",
    [
        ("## A\n\n```python\nx = 1\n", True),
        ("## A\n\n```python\nx = 1\n```\n", False),
        ("## A\n\n````md\n```\ninner\n```\n", True),
        ("## A\n\n~~~\n```\n~~~\n", False),
        ("## A\n\n`` ```inline`` text\n", False),
    ],
)
def test_close_open_fence(text: str, closed: bool) -> None:
    from scholar_lens.src.markdown_lint import close_open_fence

    out = close_open_fence(text)
    assert (out != text) is closed
    if closed:
        assert close_open_fence(out) == out


def test_section_with_open_fence_does_not_swallow_the_next_heading() -> None:
    from scholar_lens.src.review_plan import ensure_section_heading

    section = ensure_section_heading("```python\nprint(1)\n", "예제")
    joined = section + "\n\n## 다음 절\n\n본문입니다."
    assert joined.count("```") == 2


def test_edit_that_drops_a_closing_fence_is_skipped() -> None:
    from scholar_lens.src.review_plan import Edit, apply_edits

    text = "## A\n\n```python\nx = 1\n```\n\n본문입니다.\n\n## B\n\n끝입니다."
    edit = Edit(find="x = 1\n```\n", replace="x = 1\n")
    out, applied, skipped = apply_edits(text, [edit], max_edits=3)
    assert applied == 0 and out == text and skipped


@pytest.mark.parametrize(
    "find,replace",
    [("본문.", "본문.\n\n```"), ("ls\n```", "ls")],
)
def test_edit_that_unbalances_a_fence_is_skipped(find: str, replace: str) -> None:
    from scholar_lens.src.review_plan import Edit, apply_edits

    text = "## A\n\n본문.\n\n```bash\nls\n```\n\n## B\n\n끝."
    out, applied, skipped = apply_edits(
        text, [Edit(find=find, replace=replace)], max_edits=3
    )
    assert applied == 0 and out == text and "fence" in skipped[0]


def test_edit_inside_a_whole_code_block_is_applied() -> None:
    from scholar_lens.src.review_plan import Edit, apply_edits

    text = "## A\n\n```bash\nls\n```\n"
    out, applied, _ = apply_edits(
        text,
        [Edit(find="```bash\nls\n```", replace="```bash\nls -la\n```")],
        max_edits=3,
    )
    assert applied == 1 and "ls -la" in out


@pytest.mark.parametrize(
    "text,find,replace,expected",
    [
        (  # change the info string
            "## A\n\n```py\nx = 1\n```\n\n본문.",
            "```py\n",
            "```python\n",
            "```python\nx = 1\n```",
        ),
        (  # a prose edit anchored on the closing fence
            "## A\n\n```bash\nls\n```\n\n다음 문단입니다.",
            "```\n\n다음 문단입니다.",
            "```\n\n고친 문단입니다.",
            "```\n\n고친 문단입니다.",
        ),
        (  # move a misplaced closing fence
            "## A\n\n```bash\nls\n```\n설명이 코드 밖에 있어야 합니다.\npwd\n\n## B\n\n끝.",
            "ls\n```\n설명이 코드 밖에 있어야 합니다.\npwd\n",
            "ls\npwd\n```\n\n설명이 코드 밖에 있어야 합니다.\n",
            "ls\npwd\n```\n\n설명이",
        ),
        (  # close the article's one unclosed fence
            "## A\n\n```bash\nls\n\n본문이 코드에 갇혔습니다.",
            "ls\n\n본문이",
            "ls\n```\n\n본문이",
            "ls\n```\n\n본문이",
        ),
    ],
)
def test_edits_that_repair_a_fence_are_applied(
    text: str, find: str, replace: str, expected: str
) -> None:
    from scholar_lens.src.review_plan import Edit, apply_edits

    out, applied, skipped = apply_edits(
        text, [Edit(find=find, replace=replace)], max_edits=3
    )
    assert applied == 1, skipped
    assert expected in out


def test_closing_an_unclosed_fence_may_bring_back_a_swallowed_heading() -> None:
    from scholar_lens.src.review_plan import Edit, apply_edits

    text = "## A\n\n```python\nx = 1\n\n## B\n\n끝."
    out, applied, skipped = apply_edits(
        text, [Edit(find="x = 1\n\n## B", replace="x = 1\n```\n\n## B")], max_edits=3
    )
    assert applied == 1, skipped
    assert "x = 1\n```\n\n## B" in out


@pytest.mark.parametrize(
    "replace",
    [
        "x = 1\n```\n\n## New\n\n## B",  # adds a heading
        "x = 1\n```\n\n## B\n\n## A",  # duplicates one
        "x = 1\n```\n\n## B 바꿈",  # renames the swallowed one
        "```\n\n## B",  # drops the code with the closing
    ],
)
def test_closing_an_unclosed_fence_changes_nothing_else(replace: str) -> None:
    from scholar_lens.src.review_plan import Edit, apply_edits

    text = "## A\n\n```python\nx = 1\n\n## B\n\n끝."
    out, applied, _ = apply_edits(
        text, [Edit(find="x = 1\n\n## B", replace=replace)], max_edits=3
    )
    assert applied == 0 and out == text
