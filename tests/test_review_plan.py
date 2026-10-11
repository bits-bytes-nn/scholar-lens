"""Deterministic plan handling for the review workflow (no LLM calls)."""

from __future__ import annotations

import json

import pytest

from scholar_lens.src.review_plan import (
    MIN_SECTION_WORDS,
    CoverItem,
    Edit,
    PlanError,
    PlannedSection,
    SourceText,
    apply_edits,
    approx_words,
    article_ledger,
    check_plan_shape,
    concepts_owned_elsewhere,
    ensure_section_heading,
    figure_catalog,
    normalize_plan,
    parse_edits,
    parse_lines,
    parse_plan,
    previous_context,
    render_brief,
    section_source,
    target_words,
)

MARKER_A = '[Image: alt="arch", src="/p/figures/0.png", caption="Architecture."]'
MARKER_B = (
    '[Image: alt="curve", src="https://arxiv.org/html/1/x2.png", caption="Loss."]'
)


def _plan_json(**overrides: object) -> str:
    plan = {
        "glossary": {"low-rank adaptation": "저랭크 적응"},
        "excluded_ranges": [[18, 19]],
        "sections": [
            {
                "title": "문제",
                "sentence_ranges": [[0, 4]],
                "must_cover": ["full fine-tuning cost"],
                "difficulty": "easy",
                "importance": "context",
                "owned_concepts": ["fine-tuning cost"],
                "figures": ["F1"],
            },
            {
                "title": "방법",
                "sentence_ranges": [[10, 17]],
                "must_cover": ["update rule", "scaling"],
                "difficulty": "hard",
                "importance": "core",
                "owned_concepts": ["Fine-tuning cost", "rank"],
                "figures": ["F1", "F2", "F9"],
                "prerequisites": [{"concept": "SVD", "reference": "R2"}],
                "needs_code": True,
            },
        ],
    }
    plan.update(overrides)
    return f"<plan>\n```json\n{json.dumps(plan, ensure_ascii=False)}\n```\n</plan>"


FIGURE_IDS = {"F1": "/p/figures/0.png", "F2": "https://arxiv.org/html/1/x2.png"}


def _normalized(raw: str | None = None, *, n: int = 20, max_sections: int = 10):  # type: ignore[no-untyped-def]
    return normalize_plan(
        parse_plan(raw or _plan_json()),
        sentence_count=n,
        figure_ids=FIGURE_IDS,
        max_sections=max_sections,
    )


class TestParsePlan:
    def test_fenced_json_inside_tags(self) -> None:
        plan = parse_plan(_plan_json())
        assert [s.title for s in plan.sections] == ["문제", "방법"]
        assert plan.glossary == {"low-rank adaptation": "저랭크 적응"}

    @pytest.mark.parametrize("raw", ["no json here", "<plan>{not json}</plan>"])
    def test_unusable_output_raises(self, raw: str) -> None:
        with pytest.raises(PlanError):
            parse_plan(raw)

    def test_section_without_must_cover_is_invalid(self) -> None:
        raw = _plan_json(sections=[{"title": "x", "must_cover": []}])
        with pytest.raises(PlanError):
            parse_plan(raw)

    def test_range_shapes_are_coerced(self) -> None:
        raw = _plan_json(
            sections=[
                {"title": "a", "sentence_ranges": [3, 9], "must_cover": ["x"]},
                {"title": "b", "sentence_ranges": ["10-12"], "must_cover": ["y"]},
                {
                    "title": "c",
                    "sentence_ranges": [{"start": 13, "end": 14}],
                    "must_cover": ["z"],
                },
            ]
        )
        plan = parse_plan(raw)
        assert [s.sentence_ranges for s in plan.sections] == [
            [(3, 9)],
            [(10, 12)],
            [(13, 14)],
        ]

    def test_glossary_list_form(self) -> None:
        raw = _plan_json(glossary=[{"term": "rank", "rendering": "랭크"}])
        assert parse_plan(raw).glossary == {"rank": "랭크"}


class TestNormalizePlan:
    def test_figures_resolved_and_assigned_once(self) -> None:
        plan = _normalized()
        assert plan.sections[0].figures == ["/p/figures/0.png"]
        # F1 already used; F9 unknown → only F2 remains.
        assert plan.sections[1].figures == ["https://arxiv.org/html/1/x2.png"]

    def test_concept_has_one_owner(self) -> None:
        plan = _normalized()
        assert plan.sections[0].owned_concepts == ["fine-tuning cost"]
        # The SVD prerequisite is explained (and owned) where first needed.
        assert plan.sections[1].owned_concepts == ["rank", "SVD"]

    def test_ranges_clamped_and_gaps_attached(self) -> None:
        raw = _plan_json(
            sections=[
                {"title": "a", "sentence_ranges": [[-3, 4]], "must_cover": ["x"]},
                {"title": "b", "sentence_ranges": [[40, 10]], "must_cover": ["y"]},
            ],
            excluded_ranges=[],
        )
        plan = _normalized(raw, n=20)
        # [-3,4] → [0,4]; [40,10] → [10,19]; the 5..9 gap goes to section a.
        assert plan.sections[0].sentence_ranges == [(0, 4), (5, 9)]
        assert plan.sections[1].sentence_ranges == [(10, 19)]

    def test_short_gaps_and_excluded_ranges_left_alone(self) -> None:
        plan = _normalized(n=20)
        # 5..9 (5 sentences) is attached; excluded 18..19 is not.
        assert (5, 9) in plan.sections[0].sentence_ranges
        assert all(end < 18 for s in plan.sections for _, end in s.sentence_ranges)

    def test_overflow_sections_merged_into_last(self) -> None:
        sections = [
            {"title": f"s{i}", "sentence_ranges": [[i, i]], "must_cover": [f"m{i}"]}
            for i in range(5)
        ]
        plan = _normalized(
            _plan_json(sections=sections, excluded_ranges=[]), n=5, max_sections=3
        )
        assert [s.title for s in plan.sections] == ["s0", "s1", "s2"]
        assert plan.sections[2].items == ["m2", "m3", "m4"]
        assert plan.sections[2].sentence_ranges == [(2, 2), (3, 3), (4, 4)]


class TestSectionInputs:
    def test_source_keeps_only_assigned_figures(self) -> None:
        sentences = [f"s{i}." for i in range(5)]
        sentences[1] = f"See {MARKER_A}."
        sentences[2] = f"And {MARKER_B}."
        plan = _normalized(n=5)
        section = plan.sections[1]  # owns only F2 (MARKER_B), anchors clamp to 4
        markers = {
            "/p/figures/0.png": MARKER_A,
            "https://arxiv.org/html/1/x2.png": MARKER_B,
        }
        section.sentence_ranges = [(0, 4)]
        text = section_source(section, sentences, markers)
        assert MARKER_B in text and MARKER_A not in text

    def test_assigned_figure_outside_anchors_is_appended(self) -> None:
        plan = _normalized(n=20)
        section = plan.sections[0]
        markers = {"/p/figures/0.png": MARKER_A}
        text = section_source(section, [f"s{i}." for i in range(20)], markers)
        assert text.endswith(MARKER_A)

    def test_brief_lists_targets_and_ownership(self) -> None:
        plan = _normalized()
        brief = render_brief(plan, 1, {})
        assert "update rule" in brief and "scaling" in brief
        assert f"about {plan.sections[1].target_words} English words" in brief
        assert "[hard] update rule" in brief
        assert "SVD" in brief
        assert "earlier, '문제': fine-tuning cost" in concepts_owned_elsewhere(plan, 1)

    def test_target_words_sum_item_budgets(self) -> None:
        easy = [CoverItem(item="a", difficulty="easy")]
        hard = [CoverItem(item=f"h{i}", difficulty="hard") for i in range(3)]
        assert target_words(easy, "context") == MIN_SECTION_WORDS  # floor
        assert target_words(hard, "core") > target_words(hard, "supporting")
        assert target_words(hard + easy, "core") > target_words(hard, "core")

    def test_figure_catalog_dedupes_in_order(self) -> None:
        text = f"{MARKER_B} then {MARKER_A} and {MARKER_B}"
        assert [fid for fid, _ in figure_catalog(text)] == ["F1", "F2"]
        assert figure_catalog(text)[0][1] == MARKER_B


class TestEdits:
    ARTICLE = "## 방법\n\nLoRA는 저랭크 행렬을 씁니다.\n\nLoRA는 저랭크 행렬을 씁니다.\n\n끝입니다."

    def test_exact_once_rule(self) -> None:
        edits = [
            Edit(find="LoRA는 저랭크 행렬을 씁니다.", replace="X"),  # twice → skip
            Edit(find="끝입니다.", replace="마지막입니다."),
            Edit(find="없는 문장", replace="Y"),
        ]
        text, applied, skipped = apply_edits(self.ARTICLE, edits, max_edits=10)
        assert applied == 1
        assert "마지막입니다." in text and "X" not in text
        assert len(skipped) == 2

    def test_headings_cannot_be_removed(self) -> None:
        text, applied, _ = apply_edits(
            self.ARTICLE, [Edit(find="## 방법\n\nLoRA", replace="LoRA")], max_edits=5
        )
        assert applied == 0 and text == self.ARTICLE

    def test_cap(self) -> None:
        edits = [Edit(find="끝입니다.", replace="끝.")] * 3
        _, applied, skipped = apply_edits(self.ARTICLE, edits, max_edits=1)
        assert applied == 1 and any("over the cap" in s for s in skipped)

    def test_parse_edits_keeps_latex_and_ampersands_verbatim(self) -> None:
        raw = (
            "<edit>\n<find>$$\\frac{a}{b}$$ & $$\\lVert W \\rVert$$</find>\n"
            "<replace>$$\\text{x\\_y}$$ &amp; done</replace>\n<reason>r</reason>\n</edit>"
            "<edit><find>a</find><replace>b</replace></edit>"
        )
        edits = parse_edits(raw)
        assert edits[0].find == "$$\\frac{a}{b}$$ & $$\\lVert W \\rVert$$"
        assert edits[0].replace == "$$\\text{x\\_y}$$ &amp; done"
        assert edits[1] == Edit(find="a", replace="b")
        assert parse_edits("no edits here") == []


class TestHelpers:
    def test_parse_lines(self) -> None:
        assert parse_lines("- a\n* b\nnone\n\n") == ["a", "b"]
        assert parse_lines("none") == []
        for none in ("해당 없음", "없습니다.", "(none)", '"none"', "None."):
            assert parse_lines(none) == [], none

    def test_heading_enforced(self) -> None:
        assert ensure_section_heading("본문입니다.", "방법").startswith("## 방법\n\n")
        assert ensure_section_heading("# 잘못\n\n본문", "방법") == "## 방법\n\n본문"
        assert ensure_section_heading("## 원래\n본문", "방법").startswith("## 원래")

    def test_approx_words_counts_korean_and_ignores_code(self) -> None:
        assert approx_words("LoRA uses low rank") == 4
        assert approx_words("가" * 25) == 10
        assert approx_words("```\ncode code code\n```") == 0


def _naive_sentences(chunk: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for i, ch in enumerate(chunk):
        if ch == "." and (i + 1 == len(chunk) or chunk[i + 1] == " "):
            spans.append((start, i + 1))
            start = i + 2
    if start < len(chunk):
        spans.append((start, len(chunk)))
    return spans


class TestSourceText:
    TEXT = (
        "## 2 Method\n\nWe freeze W. We train BA.\n\n"
        "| Model | Acc |\n| --- | --- |\n| LoRA | 87.5 |\n\n"
        "```python\nx = 1\n\ny = 2\n```\n\n$$\nh = W_0 x\n$$\n\nDone here."
    )

    def test_units_keep_blocks_whole_and_slices_keep_structure(self) -> None:
        source = SourceText.from_text(self.TEXT, _naive_sentences)
        units = source.sentences
        assert units[0] == "## 2 Method"
        assert units[1:3] == ["We freeze W.", "We train BA."]
        assert units[3] == "| Model | Acc |\n| --- | --- |\n| LoRA | 87.5 |"
        assert units[4] == "```python\nx = 1\n\ny = 2\n```"
        assert units[5] == "$$\nh = W_0 x\n$$"
        assert source.slice(0, 3).count("\n") >= 4  # line breaks survive

    def test_padding_respects_exclusions(self) -> None:
        sentences = [f"s{i}." for i in range(10)]
        section = PlannedSection(title="t", must_cover=["x"], sentence_ranges=[(4, 5)])
        text = section_source(section, sentences, {}, excluded=[(8, 9)], pad=3)
        assert text == " ".join(sentences[1:8])


class TestPlanQuality:
    def test_prerequisites_explained_once(self) -> None:
        sections = [
            {
                "title": "a",
                "must_cover": ["x"],
                "prerequisites": [{"concept": "SVD"}],
            },
            {
                "title": "b",
                "must_cover": ["y"],
                "prerequisites": [{"concept": "svd"}, {"concept": "Adam"}],
                "owned_concepts": ["Adam"],
            },
        ]
        plan = _normalized(_plan_json(sections=sections, excluded_ranges=[]), n=5)
        assert [p.concept for p in plan.sections[0].prerequisites] == ["SVD"]
        assert "SVD" in plan.sections[0].owned_concepts
        assert [p.concept for p in plan.sections[1].prerequisites] == ["Adam"]

    def test_duplicate_items_dropped_and_difficulty_inherited(self) -> None:
        sections = [
            {"title": "a", "must_cover": ["Rank 4", "x"], "difficulty": "hard"},
            {
                "title": "b",
                "must_cover": ["rank  4", {"item": "y", "difficulty": "easy"}],
            },
        ]
        plan = _normalized(_plan_json(sections=sections, excluded_ranges=[]), n=5)
        assert plan.sections[0].must_cover[0].difficulty == "hard"
        assert plan.sections[1].items == ["y"]
        assert plan.sections[1].must_cover[0].difficulty == "easy"

    def test_shape_checks(self) -> None:
        plan = parse_plan(_plan_json())
        with pytest.raises(PlanError, match="at least"):
            check_plan_shape(plan, sentence_count=500)
        lopsided = parse_plan(
            _plan_json(
                sections=[
                    {"title": "a", "must_cover": [f"i{k}" for k in range(9)]},
                    {"title": "b", "must_cover": ["j"]},
                ]
            )
        )
        with pytest.raises(PlanError, match="split it"):
            check_plan_shape(lopsided, sentence_count=50)
        check_plan_shape(plan, sentence_count=50)


class TestArticleContext:
    SECTIONS = [
        "## 문제\n\n첫 문장입니다. 둘째 문장입니다.\n\n$$L = -\\log p$$\n",
        "## 방법\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n![arch](/assets/images/x-1.png)",
        "## 결과\n\n결과 문장입니다.",
    ]

    def test_ledger_lists_what_is_shown(self) -> None:
        ledger = article_ledger(self.SECTIONS)
        assert "heading: ## 방법" in ledger
        assert "equation: $$L = -\\log p$$" in ledger
        assert "table: | A | B |" in ledger
        assert "figure: /assets/images/x-1.png" in ledger

    def test_previous_context_digests_older_sections(self) -> None:
        context = previous_context(self.SECTIONS, full=2)
        assert "Earlier sections (digest)" in context
        assert "첫 문장입니다." in context and "둘째 문장입니다." not in context
        assert self.SECTIONS[1] in context and self.SECTIONS[2] in context
        assert previous_context([]) == "(this is the first section)"


def test_ledger_ignores_prose_that_starts_with_inline_math() -> None:
    from scholar_lens.src.review_plan import article_ledger

    section = (
        "## 방법\n\n$$W_0$$는 동결되고 $$A$$와 $$B$$만 학습합니다.\n\n"
        "$$\nh = W_0 x + BAx\n$$\n\n설명 문단입니다.\n\n$$E = mc^2$$\n"
    )
    ledger = article_ledger([section])
    assert "h = W_0 x + BAx" in ledger and "E = mc^2" in ledger
    assert "동결되고" not in ledger


def test_paper_conclusion_goes_to_the_closing_only() -> None:
    from scholar_lens.src.review_plan import SourceText

    text = "## 1 Method\n\nA. B.\n\n## 7 Conclusion and Future Work\n\nC. D.\n\n## References\n\nE."
    source = SourceText.from_text(text, lambda chunk: [(0, len(chunk))])
    ranges = source.closing_ranges()
    assert len(ranges) == 1
    a, b = ranges[0]
    assert "Conclusion" in source.slice(a, b) and "References" not in source.slice(a, b)


def test_conclusion_ranges_leave_body_sections() -> None:
    plan = normalize_plan(
        parse_plan(_plan_json()),
        sentence_count=20,
        figure_ids=FIGURE_IDS,
        max_sections=10,
        closing_ranges=[(15, 17)],
    )
    method = next(s for s in plan.sections if s.title == "방법")
    covered = {i for a, b in method.sentence_ranges for i in range(a, b + 1)}
    assert covered.isdisjoint({15, 16, 17})
    assert all(
        not ({15, 16, 17} & set(range(a, b + 1)))
        for s in plan.sections
        for a, b in s.sentence_ranges
    )


@pytest.mark.parametrize(
    "heading,closing",
    [
        ("8 Conclusion and Future Work", True),
        ("5 Conclusion, Limitations, and Future Work", True),
        ("Limitations.", True),
        ("Appendix A Limitations", True),
        ("결론 및 향후 연구", True),
        ("6 Conclusion, Limitation, and Future Work", True),
        ("Discussion and Conclusions", True),
        ("VI. Conclusion", True),
        ("On the Limitations of Compute Thresholds", False),
        ("4.0.1 Limitations of scaling laws.", False),
        ("2.2 Limitations of Existing Methods", False),
        ("Jumping to Conclusions: X", False),
        ("7 CONCLUSION AND FUTURE W ORK", True),
        ("B Related Work", False),
    ],
)
def test_closing_heading_must_be_the_whole_title(heading: str, closing: bool) -> None:
    from scholar_lens.src.review_plan import SourceText

    text = f"## {heading}\n\nA sentence.\n\n## Next\n\nMore."
    source = SourceText.from_text(text, lambda chunk: [(0, len(chunk))])
    assert bool(source.closing_ranges()) is closing
