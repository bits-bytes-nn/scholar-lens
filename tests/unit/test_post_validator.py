"""Post validator: blog defects are errors, the hand-fixed versions pass.

Fixtures in ``tests/fixtures/blog_defects`` are blocks cut from the blog's
``main`` posts (``*.before.md``, as generated) and the same blocks after the
manual fixes on the ``ui-visual-polish`` branch (``*.after.md``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scholar_lens.domain.validation import Violation, errors, validate_post
from tests.unit.conventions import bundled_conventions

FIXTURES = Path(__file__).parents[1] / "fixtures" / "blog_defects"
SPEC = bundled_conventions()
# A description as the blog's hand-written ones read (PR #88): plain text,
# 120-160 characters, opening with the model's name.
DESCRIPTION = (
    "DeepSeek-V3는 총 671B, 토큰당 37B 파라미터가 활성화되는 MoE 모델로, MLA와 보조 "
    "손실 없는 부하 분산, FP8 혼합 정밀도 학습을 결합해 2.788M H800 GPU 시간만으로 "
    "학습을 마쳤습니다. 아키텍처를 정리합니다."
)


def make_post(
    body: str = "Multi-Token Prediction을 다룹니다.",
    *,
    kind: str = "review",
    front: dict[str, str] | None = None,
    drop: tuple[str, ...] = (),
) -> str:
    use_math = "true" if "$$" in body else "false"
    fields = {
        "layout": "post",
        "title": '"DeepSeek-V3 Technical Report"',
        "subtitle": '"DeepSeek-V3 기술 보고서"',
        "date": "2024-12-27 00:00:00",
        "paper_author": '"DeepSeek AI"',
        "categories": '["Paper Reviews", "Language-Models"]',
        "tags": '["Multi-Token-Prediction", "DeepSeek"]',
        "cover": "/assets/images/language-models.jpg",
        "use_math": use_math,
        "description": f'"{DESCRIPTION}"',
    }
    if kind == "guide":
        fields.pop("subtitle")
        fields.pop("paper_author")
        fields["categories"] = '["Tech Guides", "Agentic-AI"]'
        fields["cover"] = "/assets/images/agentic-ai.jpg"
        fields["tags"] = '["Agentic-AI"]'
    fields.update(front or {})
    for key in drop:
        fields.pop(key, None)
    header = "\n".join(f"{k}: {v}" for k, v in fields.items())
    opening = (
        "" if kind == "guide" else "## TL;DR\n### 왜 필요한가?\n짧은 답입니다.\n\n"
    )
    return f"---\n{header}\n---\n{opening}## 본문\n\n{body}\n"


def warnings(found: list[Violation]) -> list[Violation]:
    return [v for v in found if v.level == "warning"]


def rules(markdown: str, kind: str = "review", level: str = "error") -> set[str]:
    found = validate_post(markdown, SPEC, kind)  # type: ignore[arg-type]
    chosen = errors(found) if level == "error" else warnings(found)
    return {v.rule for v in chosen}


class TestBlogDefectFixtures:
    @pytest.mark.parametrize(
        "case,rule",
        [
            ("v3-currency-table", "math"),
            ("v3-currency-prose", "math"),
            ("v3-short-header", "table"),
            ("r1-subheader", "table"),
            ("qlora-multirow-header", "table"),
            ("zep-escaped-underscore", "math"),
        ],
    )
    def test_original_fails_fixed_passes(self, case: str, rule: str) -> None:
        before = (FIXTURES / f"{case}.before.md").read_text(encoding="utf-8")
        after = (FIXTURES / f"{case}.after.md").read_text(encoding="utf-8")
        assert rule in rules(make_post(before))
        assert rules(make_post(after)) == set()

    def test_clean_post_has_no_violations(self) -> None:
        assert validate_post(make_post(), SPEC, "review") == []


class TestFrontMatter:
    def test_author_key_forbidden(self) -> None:
        assert "forbidden-key" in rules(make_post(front={"author": '"Jonas"'}))

    @pytest.mark.parametrize(
        "author",
        [
            '"Preston Rasmussen et al."',
            '""',
        ],
    )
    def test_paper_author_must_be_an_organisation(self, author: str) -> None:
        assert "paper-author" in rules(make_post(front={"paper_author": author}))

    @pytest.mark.parametrize(
        "author",
        [
            '"Zep AI"',
            '"Google DeepMind"',
            '"Meta AI and New York University"',
            '"Carnegie Mellon, Johns Hopkins"',
            '"Hugging Face and Mila Quebec"',
        ],
    )
    def test_organisations_pass(self, author: str) -> None:
        assert "paper-author" not in rules(make_post(front={"paper_author": author}))

    def test_subtitle_required_and_distinct(self) -> None:
        assert "subtitle" in rules(make_post(drop=("subtitle",)))
        assert "subtitle" in rules(
            make_post(front={"subtitle": '"DeepSeek-V3 Technical Report"'})
        )

    def test_unknown_topic_and_wrong_cover(self) -> None:
        assert "categories" in rules(
            make_post(front={"categories": '["Paper Reviews", "Training-&-Inference"]'})
        )
        assert "cover" in rules(
            make_post(front={"cover": "/assets/images/default.jpg"})
        )

    def test_type_must_match_kind(self) -> None:
        assert "categories" in rules(make_post(), kind="guide")

    def test_long_title_needs_seo_title(self) -> None:
        long_title = (
            '"'
            + "DeepSeek-Coder: When the Large Language Model Meets Programming" * 2
            + '"'
        )
        assert "seo-title" in rules(make_post(front={"title": long_title}))
        assert "seo-title" not in rules(
            make_post(
                front={
                    "title": long_title,
                    "seo_title": '"DeepSeek-Coder: The Rise of Code Intelligence"',
                }
            )
        )

    def test_use_math_must_match_content(self) -> None:
        assert "use-math" in rules(make_post(front={"use_math": "true"}))

    def test_guide_front_matter(self) -> None:
        assert rules(make_post(kind="guide"), kind="guide") == set()
        assert "paper-author" in rules(
            make_post(kind="guide", front={"paper_author": '"X Lab"'}), kind="guide"
        )


class TestTags:
    def test_a_controlled_tag_is_optional(self) -> None:
        # None fits: better no related reading than wrong related reading.
        assert "tags" not in rules(
            make_post(front={"tags": '["Multi-Token-Prediction"]'})
        )

    def test_retired_alias_rejected(self) -> None:
        assert "tags" in rules(
            make_post(front={"tags": '["DeepSeekMoE-Architecture", "DeepSeek"]'})
        )

    def test_unnormalized_tag_rejected(self) -> None:
        assert "tags" in rules(make_post(front={"tags": '["Pan-&-Scan", "DeepSeek"]'}))


class TestBody:
    def test_h1_and_heading_jumps(self) -> None:
        assert "headings" in rules(make_post("# 제목\n\n본문"))
        assert "headings" in rules(make_post("#### 깊은 제목\n\n본문"))

    def test_paper_post_starts_with_tldr(self) -> None:
        post = make_post().replace("## TL;DR\n", "## 요약\n")
        assert "headings" in rules(post)

    def test_empty_tldr_is_an_error(self) -> None:
        empty = make_post().replace("### 왜 필요한가?\n짧은 답입니다.\n", "")
        assert "headings" in rules(empty)
        assert "headings" in rules(empty, kind="summarize")
        no_questions = make_post().replace("### 왜 필요한가?\n", "")
        assert "headings" in rules(no_questions)
        assert "headings" not in rules(no_questions, kind="summarize")
        assert "headings" not in rules(make_post())

    def test_display_math_needs_blank_lines(self) -> None:
        assert "math" in rules(make_post("문장입니다.\n$$ x^2 $$\n다음 문장"))
        assert "math" not in rules(make_post("문장입니다.\n\n$$ x^2 $$\n\n다음 문장"))

    def test_legacy_inline_delimiters(self) -> None:
        # The Markdown-escaped form reaches MathJax; a single backslash is a
        # literal parenthesis (normalize_math converts \(x\) math first).
        assert "math" in rules(make_post(r"변수 \\(x\\)를 씁니다."))
        assert "math" not in rules(make_post(r"**LongMemEval \(s\)** 데이터셋"))

    def test_bare_pipe_in_inline_math(self) -> None:
        assert "math" in rules(make_post("값 $$|x|$$ 와 $$y$$ 입니다."))
        # A single span on a one-line paragraph still becomes a GFM table.
        assert "math" in rules(make_post("본문 $$p(y|x)$$ 입니다."))
        assert "math" not in rules(make_post(r"본문 $$p(y \mid x)$$ 입니다."))
        assert "math" not in rules(make_post("$$\n|x| = 1\n$$"))  # display
        assert "math" not in rules(
            make_post(r"값 $$\lvert x \rvert$$ 와 $$y$$ 입니다.")
        )

    def test_bare_pipe_in_a_wrapped_inline_span(self) -> None:
        # Regression: a span that wraps onto the next line was never checked.
        assert "math" in rules(make_post("크기는 $$|S| +\n|T|$$ 입니다."))
        assert "math" not in rules(
            make_post("크기는 $$\\vert S\\vert +\n\\vert T\\vert$$ 입니다.")
        )

    def test_unconverted_bracket_display_in_prose(self) -> None:
        # Markdown shows "\[ y = f(x) \]" as "[ y = f(x) ]" (here in a quote).
        assert "math" in rules(make_post("> \\[ y = f(x) \\]"))
        assert "math" not in rules(make_post("본문 \\[1\\] 인용."))
        for prose in ("옵션 \\[--max_len\\] 참고.", "각주 \\[^1\\] 입니다."):
            assert "math" not in rules(make_post(prose))
        assert "math" in rules(make_post("> \\[ \\frac{a}{b} \\]"))

    def test_underscore_escape_allowed_inside_text(self) -> None:
        assert "math" not in rules(make_post(r"값 $$\text{batch\_size}$$ 입니다."))

    def test_images(self) -> None:
        assert "image" in rules(
            make_post("![a](https://arxiv.org/html/2412.19437v1/x1.png)")
        )
        assert "image" in rules(make_post("![a](/assets/2412/x1.png)"))
        assert rules(make_post("![a](/assets/images/deepseek-v3-1.png)")) == set()
        # A third-party image stays a link (copyright): a warning, not an error.
        assert "image" in rules(
            make_post("![a](https://docs.x.dev/a.png)"), level="warning"
        )
        assert "image" not in rules(make_post("![a](https://docs.x.dev/a.png)"))

    def test_data_url_image_is_an_error(self) -> None:
        post = make_post("![a](data:image/png;base64,iVBORw0KGgo=)")
        found = errors(validate_post(post, SPEC, "review"))
        assert [v.message for v in found if v.rule == "image"] == [
            "inline data: image; store it as a file in /assets/images/"
        ]

    def test_envelope_tag_leak(self) -> None:
        assert "envelope" in rules(make_post("본문</explanation>"))

    def test_code_blocks_are_ignored(self) -> None:
        body = "```python\n# comment\n$$ x\\_1 $$\n| a |\n```"
        assert rules(make_post(body, front={"use_math": "false"})) == set()


class TestDescription:
    @staticmethod
    def _found(post: str, **kwargs) -> list[Violation]:  # type: ignore[no-untyped-def]
        return [
            v
            for v in validate_post(post, SPEC, "review", **kwargs)
            if v.rule == "description"
        ]

    def test_the_blogs_kind_of_description_passes(self) -> None:
        assert self._found(make_post()) == []

    def test_no_longer_than_the_page_title_is_an_error(self) -> None:
        # <title> is "DeepSeek-V3 Technical Report 논문 리뷰" (34 characters).
        post = make_post(front={"description": '"DeepSeek-V3를 정리합니다."'})
        assert [v.level for v in self._found(post)] == ["error"]

    @pytest.mark.parametrize(
        ("lang", "levels"),
        [
            # A Korean paper post's <title> is its seo_title plus the suffix.
            ({}, ["warning"]),
            # Any other post's <title> is its seo_title too (head.html line 47).
            ({"lang": "en", "translation_id": "deepseek-v3"}, ["warning"]),
        ],
    )
    def test_seo_title_names_every_posts_page(
        self, lang: dict[str, str], levels: list[str]
    ) -> None:
        front = {
            "title": '"DeepSeek-V3 Technical Report on Mixture of Experts at Scale"',
            "seo_title": '"DeepSeek-V3"',
            "description": f'"{"DeepSeek-V3는 MoE 모델입니다. " * 2}"',  # 46
            **lang,
        }
        assert [v.level for v in self._found(make_post(front=front))] == levels

    def test_outside_the_usual_length_warns(self) -> None:
        text = "DeepSeek-V3는 MoE 모델입니다. " * 2  # 46 characters
        found = self._found(make_post(front={"description": f'"{text}"'}))
        assert [v.level for v in found] == ["warning"]

    def test_missing_on_a_paper_post_warns(self) -> None:
        found = self._found(make_post(drop=("description",)))
        assert [v.level for v in found] == ["warning"]

    def test_another_posts_description_is_an_error(self) -> None:
        from scholar_lens.domain.conventions import BlogPost

        def site(path: str) -> list[BlogPost]:
            return [BlogPost(path=path, description=f"  {DESCRIPTION}\n")]

        name = "2026-10-13-deepseek-v3-technical-report.md"
        other = self._found(
            make_post(), file_name=name, site_posts=site("_posts/2025-01-01-other.md")
        )
        assert [v.level for v in other] == ["error"]
        # The post this one replaces is not another post.
        replaced = self._found(
            make_post(),
            file_name=name,
            site_posts=site("_posts/2025-10-07-deepseek-v3-technical-report.md"),
        )
        assert replaced == []

    def test_math_in_first_paragraph_warns_without_a_description(self) -> None:
        post = make_post(drop=("description",)).replace(
            "짧은 답입니다.", "변화량 $$\\Delta W$$를 학습합니다."
        )
        assert any("contains math" in v.message for v in self._found(post))
        # With a description the blog never reads the first paragraph.
        with_one = make_post().replace(
            "짧은 답입니다.", "변화량 $$\\Delta W$$를 학습합니다."
        )
        assert self._found(with_one) == []


def test_leaked_figure_marker_is_an_error() -> None:
    post = make_post('[Image: alt="x", src="/assets/images/a-1.png", caption="c"]')
    assert "envelope" in rules(post)


@pytest.mark.parametrize(
    "date,ok",
    [
        ("2024-12-27 00:00:00", True),
        ("2024-12-27", True),  # a date without a time is checked too
        ("2024-12-28", False),
        ("2024-12-28 08:00:00 +0900", True),  # 2024-12-27 in UTC
    ],
)
def test_date_is_not_after_the_file_name(date: str, ok: bool) -> None:
    found = validate_post(
        make_post(front={"date": date}), SPEC, "review", file_name="2024-12-27-x.md"
    )
    assert ("date" not in {v.rule for v in errors(found)}) is ok


class TestBodyErrors:
    def test_table_and_inline_pipe_errors_are_exposed_for_writers(self) -> None:
        from scholar_lens.domain.validation import body_errors

        body = (
            "| A | B |\n| --- | --- |\n| 1 | 2 | 3 |\n\n"
            "$$a|b$$ and $$c$$ in one line.\n"
        )
        issues = body_errors(body)
        assert any(i.startswith("table:") and "3 cells" in i for i in issues)
        assert any(i.startswith("math:") and "bare |" in i for i in issues)
        assert body_errors("## 제목\n\n본문입니다.\n") == []
        # The blog lint's style rules block publishing, so the loops see them.
        assert any(
            i.startswith("style:")
            for i in body_errors("## 제목\n\n구성은 다음과 같습니다: 첫째.\n")
        )

    def test_unclosed_fence_is_an_error_not_silent_code(self) -> None:
        from scholar_lens.domain.validation import body_errors

        assert any("never closed" in i for i in body_errors("```py\nx = 1\n"))


class TestValidatorDetails:
    def _post(self, body: str) -> str:
        return (
            '---\nlayout: post\ntitle: "T"\nsubtitle: "부제"\n'
            "date: 2021-01-01 00:00:00\n"
            'categories: ["Tech Guides", "Language-Models"]\n'
            'tags: ["Efficient-Attention"]\n'
            "cover: /assets/images/language-models.jpg\nuse_math: true\n---\n" + body
        )

    def test_unclosed_fence_reported_at_its_line(self) -> None:
        md = self._post("## A\n\n본문 $$x$$입니다.\n\n```python\nx = 1\n")
        codes = [
            v
            for v in validate_post(md, bundled_conventions(), "guide")
            if v.rule == "code"
        ]
        assert codes and codes[0].line == 15  # the ``` line, not the front matter

    def test_attribute_list_may_follow_a_display(self) -> None:
        body = "## A\n\n본문입니다.\n\n$$x = 1$$\n{: .center}\n\n끝입니다.\n"
        assert not [
            v
            for v in validate_post(self._post(body), bundled_conventions(), "guide")
            if v.rule == "math"
        ]


class TestUnsourcedNumbers:
    SOURCE = (
        "Mistral 7B outperforms Llama 2 13B. Models from 0.6B parameters, an "
        "Elo of 1,338 and 87.3% on MMLU; 2 trillion tokens with a 128K context. "
        "M_opt ∝ C^0.5243 and 2.788M GPU hours; a 400M encoder; 1.4T tokens; "
        "LongMemEval improves by 15.2% (55.4% to 63.8%)."
    )

    @pytest.mark.parametrize(
        "text,missing",
        [
            # B -> 억 is x10: 7B is 70억, never 7억; 0.6B is 6억, not 0.6억.
            ("70억 파라미터의 Mistral 7B", []),
            ("7억 파라미터", ["7억"]),
            ("6억에서 시작", []),
            ("0.6억에서 시작", ["0.6억"]),
            # M -> 만 / 백만, K -> 천, T -> 조; thousands separators.
            ("278.8만 GPU 시간, 4억 인코더, 1.4조 토큰과 2조 토큰", []),
            # Combined Korean units add up: 12만 8천 is 128,000 (128K).
            ("128K 문맥, 12만 8천은 합성 수", []),
            ("12만8천 토큰", []),
            ("12만 9천 토큰", ["12만 9천"]),
            ("278만 8천 GPU 시간", []),
            # A larger unit after a smaller one starts a new number.
            ("128K와 8천 2조", ["8천"]),
            ("Elo 1338점, MMLU 87.3", []),
            # An exponent read as a percentage of the budget.
            ("예산의 52.43%를 모델에", ["52.43"]),
            ("15.2% 개선(55.4%에서 63.8%)", []),
            # Names are not numbers; a list keeps its items.
            ("Qwen3와 H800, 7B,13B", []),
        ],
    )
    def test_numbers_match_the_source(self, text: str, missing: list[str]) -> None:
        from scholar_lens.domain.validation import unsourced_numbers

        assert unsourced_numbers(text, self.SOURCE) == missing

    @pytest.mark.parametrize(
        ("text", "source", "missing"),
        [
            # A lone 만 is the unit 10^4 or the particle "only": the number is
            # sourced when the source has either reading.
            ("상위 5만 사용", "the top 5 layers", []),
            ("5만 명", "50,000 annotators", []),
            ("10만 tokens", "100K tokens", []),
            ("10만큼 늘었다", "grows by 10", []),
            ("상위 5만 사용", "the top 7 layers", ["5만"]),
            # A combined Korean number is one quantity, never "only".
            ("12만 8천", "128K context", []),
            ("12만 8천", "12 layers and 8 heads", ["12만 8천"]),
            ("1억 2000만 사용", "120M parameters", []),
        ],
    )
    def test_man_is_sourced_as_either_reading(
        self, text: str, source: str, missing: list[str]
    ) -> None:
        from scholar_lens.domain.validation import unsourced_numbers

        assert unsourced_numbers(text, source) == missing

    def test_description_problems(self) -> None:
        from scholar_lens.domain.validation import description_problems

        tldr = "DeepSeek-V3는 671B MoE 모델로 2.788M GPU 시간에 학습했습니다."
        assert description_problems(DESCRIPTION, tldr + " 37B H800") == []
        problems = description_problems("`DeepSeek-V3`는 99점입니다.", tldr)
        assert len(problems) == 3  # markup, length, an unsourced number
        assert "99" in problems[2]
        assert description_problems("", tldr) == ["description is empty"]


def errors_of(markdown: str, rule: str, kind: str = "review") -> list[str]:
    found = errors(validate_post(markdown, SPEC, kind))  # type: ignore[arg-type]
    return [v.message for v in found if v.rule == rule]


class TestBlogLintParity:
    """The blog lint's own cases (tech-blog test/test_lint_posts.rb): a post
    fails here exactly when it fails there."""

    @pytest.mark.parametrize(
        ("author", "ok"),
        [  # test_lint_posts.rb lines 34-40 and 271-274
            ('"Rasmussen et al."', False),
            ('"Rasmussen ET AL"', False),  # /et al/i, case-blind
            ('"Wet Algorithms Lab"', False),  # anywhere, as the blog reads it
            ('["A", "B"]', False),  # one string, not a list
            ('"Google DeepMind, Google Research, Stanford"', True),
        ],
    )
    def test_paper_author(self, author: str, ok: bool) -> None:
        found = errors_of(make_post(front={"paper_author": author}), "paper-author")
        assert (found == []) is ok

    @pytest.mark.parametrize(
        ("doi", "message"),
        [  # test_lint_posts.rb lines 290-297
            ('"10.48550/arXiv.2310.06825"', None),
            ('"10.18653/v1/2024.acl-long.1"', None),
            ('"10.1145/3292500.3330701v2"', None),  # only arXiv's version is cut
            ('"https://doi.org/10.48550/arXiv.2310.06825"', "bare DOI"),
            ('"arXiv:2310.06825"', "bare DOI"),
            ('""', "bare DOI"),
            ("", "bare DOI"),  # set but empty
            ("10.48550", "bare DOI"),
            ('"10.1000/a?b"', "bare DOI"),
            ('"10.1000/a#b"', "bare DOI"),
            ('"10.48550/arXiv.2310.06825v2"', "arXiv version"),
        ],
    )
    def test_paper_doi(self, doi: str, message: str | None) -> None:
        found = errors_of(make_post(front={"paper_doi": doi}), "paper-doi")
        assert found == [] if message is None else message in " ".join(found)

    @pytest.mark.parametrize(
        ("tag", "message"),
        [  # test_lint_posts.rb lines 60-66, 180-191
            ("Opt-In", None),  # a last "In"/"On" is a particle
            ("Leiden-Algorithm-for-Text", None),
            ("2024", None),
            ("pytorch", None),
            ("Attention-On-Demand", "should have on"),
            ("Community-Detection-For-RAG", "should have for"),
            ("Public-Dataset-Only Training", "joined by hyphens"),
            (".NET", "joined by hyphens"),
            ("+Plus", "joined by hyphens"),
            ("Sparse-Mixture-of-Experts", "retired"),
        ],
    )
    def test_tag_form(self, tag: str, message: str | None) -> None:
        post = make_post(front={"tags": f'["{tag}", "DeepSeek"]'})
        found = errors_of(post, "tags")
        assert found == [] if message is None else message in " ".join(found)

    @pytest.mark.parametrize(
        ("table", "message"),
        [  # test_lint_posts.rb lines 84-101
            ("| | A | B |\n|---|---|---|\n| x | 1 | 2 |", "first cell is empty"),
            ("| 모델 | A | |\n|---|---|---|\n| x | 1 | 2 |", "empty header cell"),
            # A cell holding only \| is not empty (tech-blog 71c4c51).
            ("| \\| | A |\n|---|---|\n| x | 1 |", None),
            ("| 모델 | A |\n|---|---|\n| | 85.2 |", None),
            ("| 모델 | A |\n|---|---|\n| | 85.2±0.3 |", "sub-header"),
        ],
    )
    def test_table_headers(self, table: str, message: str | None) -> None:
        found = errors_of(make_post(table), "table")
        assert found == [] if message is None else message in " ".join(found)

    @pytest.mark.parametrize(
        "body",
        [  # test_lint_posts.rb lines 277-281
            "그림에서 보여지는 곡선입니다.",
            "추론에 있어서는 그렇습니다.",
            "학습함에 있어 중요합니다.",
            "다음과 같이 작동합니다:\n\n1. 하나",
            # An indented code block is prose to the blog lint.
            "앞 문단.\n\n    보여지는 것: x",
            "### References\n\n* 논문\n\n### 정리\n\n좋다: 끝",
        ],
    )
    def test_style_rules_reject(self, body: str) -> None:
        assert errors_of(make_post(body), "style")

    @pytest.mark.parametrize(
        "body",
        [  # test_lint_posts.rb lines 282-287
            "이 점에 있어서 중요합니다.",
            '### 정리합니다: 요약\n\n> 그는 말했다: 아니다\n\n"이렇다: 저렇다"라고 '
            "합니다.\n\n`a다:b`",
            "| 바다: 깊이 | 1 |\n|---|---|\n| 2 | 3 |",
            "- 항목\n\n  | 같습니다: | 2 |\n  |---|---|\n  | 3 | 4 |\n\n"
            "![정리합니다: 그림](/assets/images/a.png)",
            "### References\n\n* 보여지는 것들: 논문",
            "```text\n다음과 같습니다:\n```",
            "Intro.\n\n$$\n같습니다:\n$$\n\nTail.",
        ],
    )
    def test_style_rules_exempt(self, body: str) -> None:
        assert errors_of(make_post(body), "style") == []

    @pytest.mark.parametrize(
        ("body", "insecure"),
        [  # htmlproofer's enforce_https: a/link href, img src/srcset
            ("[링크](http://example.com)를 봅니다.", True),
            ("![그림](http://example.com/a.png)", True),
            ("<http://example.com>", True),
            ('<a href="http://a.b">x</a>', True),
            (
                '<img src="https://a.b/x.png" srcset="http://a.b/y.png 2x" alt="x">',
                True,
            ),
            ("[x](//a.b/c)", True),
            ("[a]: http://a.b", True),
            ("[링크](https://example.com)와 `http://localhost:8000`", False),
            ("```bash\ncurl http://localhost:8000\n```", False),
            ("앞 문단.\n\n    curl http://localhost:8000", False),
            ("주소 http://localhost:8000 는 링크가 아닙니다.", False),
        ],
    )
    def test_http_links(self, body: str, insecure: bool) -> None:
        assert bool(errors_of(make_post(body), "link")) is insecure


def test_an_overlong_description_is_cut_at_its_last_fitting_sentence() -> None:
    from scholar_lens.domain.validation import DESCRIPTION_LENGTH, fit_description

    first = "가" * 130 + "입니다."
    long = first + " " + "나" * 40 + "입니다."
    assert len(long) > DESCRIPTION_LENGTH.stop - 1
    assert fit_description(long) == first
    # No sentence end in range: left as is (the validator warns).
    unbroken = "다" * 170
    assert fit_description(unbroken) == unbroken
    assert fit_description("짧은 설명입니다.") == "짧은 설명입니다."


def test_a_leaked_evidence_marker_blocks_the_post() -> None:
    from scholar_lens.domain.validation import _EVIDENCE_LEFTOVER

    assert _EVIDENCE_LEFTOVER.search("관점 [[E120], [E128]]도 설득력이 있습니다.")
    assert not _EVIDENCE_LEFTOVER.search("배열 [[1, 2]]와 E12 규격")


def test_math_and_lists_in_brackets_are_no_leftover_marker() -> None:
    from scholar_lens.domain.validation import _EVIDENCE_LEFTOVER

    for text in (
        "기저 벡터 [e1, e2, e3]",
        "$[e1, e2]$",
        "[E5, E6] 등급",
        "x = [[e0, e1], [e2, e3]]",
    ):
        assert not _EVIDENCE_LEFTOVER.search(text)


@pytest.mark.parametrize("key", ["title", "subtitle", "paper_author"])
def test_a_tag_in_a_page_header_field_is_an_error(key: str) -> None:
    post = make_post(front={key: '"<think> 토큰 분석"'})
    found = [v for v in validate_post(post, SPEC, "review") if v.rule == "front-matter"]
    assert found and key in found[0].message


def test_number_sources_name_the_table_and_heading_a_number_comes_from() -> None:
    from scholar_lens.domain.validation import number_sources

    paper = (
        "## 5 Results\n\nOn GPT-4.1 Mini the prompt is 7.5x shorter.\n\n"
        "### Qwen3 8B\n\n| Method | Length |\n| --- | --- |\n| GEPA | 9.2x |\n"
    )
    found = number_sources("GPT-4.1 Mini에서 프롬프트가 최대 9.2배 짧았습니다.", paper)
    assert "- 9.2:" in found and "[Qwen3 8B / | Method | Length |]" in found
    assert "- 7.5:" not in found  # only the numbers the checked text states
    assert number_sources("숫자가 없습니다.", paper) == ""


def test_number_sources_past_the_cap_are_counted_not_listed() -> None:
    from scholar_lens.domain.validation import number_sources

    paper = "\n".join(f"Row {n} scores {n}.5 on the task." for n in range(10, 40))
    text = " ".join(f"{n}.5" for n in range(10, 40))
    found = number_sources(text, paper, max_chars=200)
    assert "more number(s) not listed" in found and found.count("\n- ") < 29


def test_number_sources_say_how_many_passages_they_leave_out() -> None:
    from scholar_lens.domain.validation import number_sources

    paper = "\n".join(f"Setting {k}: the model scores 73.7 here." for k in range(5))
    found = number_sources("점수는 73.7입니다.", paper, per_number=3)
    assert (
        found.count("scores 73.7") == 3 and "(+2 more passage(s) not listed)" in found
    )


def test_number_sources_give_every_number_a_passage_first() -> None:
    from scholar_lens.domain.validation import number_sources

    paper = "\n".join(
        [f"Intro mentions {n}.5 in passing." for n in range(10, 40)]
        + [f"On MMLU the model scores {n}.5." for n in range(10, 40)]
    )
    text = " ".join(f"MMLU에서 {n}.5" for n in range(10, 40))
    found = number_sources(text, paper, per_number=2, max_chars=3_500)
    # All numbers get one passage before any gets a second (the cap leaves
    # room for every first, not every second), and the one that shares the
    # sentence's words (MMLU, with its particle) comes first.
    assert "not listed: not being listed" not in found
    assert found.count("On MMLU") == 30
    assert found.count("Intro mentions") < 30


def test_a_figure_reading_is_marked_and_ranked_after_the_papers_text() -> None:
    from scholar_lens.domain.validation import number_sources

    paper = (
        "Figure 3: Loss curves. (Figure analysis, approximate: the loss falls to 0.42.)\n"
        "Table 2 reports a final loss of 0.42 on WikiText."
    )
    found = number_sources("최종 손실은 0.42입니다.", paper)
    lines = found.splitlines()
    assert "Table 2 reports" in lines[1]
    assert "figure reading, not the paper's text" in lines[2]


def test_a_tag_in_the_papers_title_is_found_before_writing() -> None:
    from scholar_lens.domain.validation import paper_metadata_errors

    front = make_post(front={"title": '"Thinking with <think> Tokens"'}).split(
        "## TL;DR"
    )[0]
    assert any("has a tag" in e for e in paper_metadata_errors(front, SPEC, "review"))


def test_a_thin_seo_title_falls_back_to_the_title_cut_at_a_word() -> None:
    from scholar_lens.domain.post_format import shorten_title

    title = "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"
    # The bare name "GEPA" says too little: the title cut at a word, name first.
    assert shorten_title(title, 70) == (
        "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement"
    )
    # A cut never ends on a function word.
    assert (
        shorten_title("X: " + "Reasoning via " * 10, 30) == "X: Reasoning via Reasoning"
    )
    # A head of exactly a third of the limit is used; one shorter is not.
    assert shorten_title("H" * 23 + ": " + "word " * 20, 70) == "H" * 23
    assert shorten_title("H" * 22 + ": " + "word " * 20, 70) != "H" * 22
    # A substantial name before the colon is still used.
    long_head = "Direct Preference Optimization Explained: " + "x " * 40
    assert shorten_title(long_head, 70) == "Direct Preference Optimization Explained"


@pytest.mark.parametrize("extra", [0, 1])
def test_a_title_over_the_limit_needs_a_seo_title(extra: int) -> None:
    title = "T" * (SPEC.seo_title_max_length + extra)
    found = rules(make_post(front={"title": f'"{title}"'}, drop=("seo_title",)))
    assert ("seo-title" in found) == bool(extra)


@pytest.mark.parametrize("extra", [0, 1])
def test_seo_title_limit_is_exact(extra: int) -> None:
    limit = SPEC.seo_title_max_length
    seo = "S" * (limit + extra)
    found = rules(make_post(front={"seo_title": f'"{seo}"'}))
    assert ("seo-title" in found) == bool(extra)


@pytest.mark.parametrize("extra", [0, 1])
def test_specific_tag_count_is_exact(extra: int) -> None:
    count = SPEC.tags.max_specific + extra
    tags = ", ".join(f'"Specific-Tag-{i}"' for i in range(count))
    found = rules(make_post(front={"tags": f"[{tags}]"}))
    assert ("tags" in found) == bool(extra)


def test_a_translated_post_needs_its_translation_id() -> None:
    lang = SPEC.languages[0]
    assert "translation" in rules(make_post(front={"lang": lang}))
    assert "translation" not in rules(
        make_post(front={"lang": lang, "translation_id": "dsv3"})
    )


def test_a_thin_model_seo_title_is_replaced_but_a_short_korean_one_kept() -> None:
    from scholar_lens.domain.post_format import build_post_meta

    title = "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"

    def seo(proposed: str) -> str | None:
        return build_post_meta(
            title=title,
            category_type="Paper Reviews",
            proposed_topic="Language-Models",
            specific_tags=[],
            controlled_tags=[],
            proposed_seo_title=proposed,
            spec=SPEC,
        ).seo_title

    assert (
        seo("GEPA") == "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement"
    )
    # The model's candidate needs 12 characters.
    assert seo("A" * 12) == "A" * 12
    assert seo("A" * 11) != "A" * 11
    assert (
        seo("반성적 프롬프트 진화로 강화학습 넘기")
        == "반성적 프롬프트 진화로 강화학습 넘기"
    )


def test_cut_titles_stay_within_the_limit_and_keep_acronyms() -> None:
    from scholar_lens.domain.post_format import cut_title

    assert cut_title("A" * 80, 70) == "A" * 70  # one long word: a hard cut
    assert cut_title(": " * 40, 70)  # never empty
    # A trailing acronym or label letter is no function word...
    assert cut_title("Planning Models " * 4 + "in OR and More Words", 70).endswith(
        " OR"
    )
    assert cut_title(
        "Ranking Approaches " * 3 + "With Model A vs. Others Today", 70
    ).endswith("Model A")
    # ... but "vs." and "&" are.
    assert not cut_title(
        "Ranking Approaches " * 3 + "Long Name vs. Others", 70
    ).endswith("vs.")
    assert cut_title("Taskss " * 9 + "Tas & More Words Here", 70).endswith(" Tas")


def test_a_colon_inside_a_word_is_no_subtitle() -> None:
    from scholar_lens.domain.post_format import shorten_title

    title = "Deep Dive Into Amazon Bedrock Agents and Guardrails at AWS re:Invent 2026 Las Vegas"
    assert not shorten_title(title, 70).endswith(" re")  # type: ignore[union-attr]


def test_an_image_shown_twice_is_flagged_for_the_pr() -> None:
    figure = "![손실 곡선](/assets/images/post-1.png)"
    other = "![다른 그림](/assets/images/post-2.png)"
    twice = make_post(f"{figure}\n\n본문입니다.\n\n{other}\n\n{figure}")
    assert "image-repeat" in rules(twice, level="warning")
    assert "image-repeat" not in rules(twice)  # a warning, not a rejection
    once = make_post(f"{figure}\n\n본문입니다.\n\n{other}")
    assert "image-repeat" not in rules(once, level="warning")


def test_a_title_at_the_limit_once_normalized_is_not_cut() -> None:
    from scholar_lens.domain.post_format import cut_title

    for title in ["Word " * 13 + "Tail", "A  B " * 14, "Name: " + "x" * 63 + " :"]:
        whole = " ".join(title.split()).strip(" :")
        assert cut_title(title, len(whole)) == whole
        cut = cut_title(title, len(whole) - 1)
        assert cut and len(cut) <= len(whole) - 1


@pytest.mark.parametrize(
    ("seo_title", "warned"),
    [
        ("Optimizing Instructions and Demonstrations for Multi-Stage Language", True),
        ("Optimizing Instructions", False),  # the part before the colon
        ("프롬프트 최적화로 다단계 LM 프로그램 개선", False),  # written, not cut
    ],
)
def test_a_seo_title_cut_from_the_title_is_flagged(
    seo_title: str, warned: bool
) -> None:
    title = (
        "Optimizing Instructions: Demonstrations for Multi-Stage Language Model "
        "Programs"
        if not warned
        else "Optimizing Instructions and Demonstrations for Multi-Stage Language "
        "Model Programs"
    )
    post = make_post(front={"title": f'"{title}"', "seo_title": f'"{seo_title}"'})
    assert ("seo-title" in rules(post, level="warning")) == warned
    assert "seo-title" not in rules(post)
