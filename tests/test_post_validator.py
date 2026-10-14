"""Post validator: blog defects are errors, the hand-fixed versions pass.

Fixtures in ``tests/fixtures/blog_defects`` are blocks cut from the blog's
``main`` posts (``*.before.md``, as generated) and the same blocks after the
manual fixes on the ``ui-visual-polish`` branch (``*.after.md``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scholar_lens.src.post_conventions import load_local_conventions
from scholar_lens.src.post_validator import Violation, errors, validate_post

FIXTURES = Path(__file__).parent / "fixtures" / "blog_defects"
SPEC = load_local_conventions()
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
        fields["cover"] = "/assets/images/default.jpg"
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

    @pytest.mark.parametrize(
        ("doi", "ok"),
        [
            ('"10.48550/arXiv.2305.18290"', True),
            ('"10.52202/075280-2338"', True),
            ('"https://doi.org/10.48550/arXiv.2305.18290"', False),
            ('"10.48550/arXiv.2305.18290?v=2"', False),
            ('"10.1/x"', False),
        ],
    )
    def test_paper_doi_is_a_bare_doi(self, doi: str, ok: bool) -> None:
        assert ("paper-doi" not in rules(make_post(front={"paper_doi": doi}))) is ok

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
    def test_controlled_tag_required(self) -> None:
        assert "tags" in rules(make_post(front={"tags": '["Multi-Token-Prediction"]'}))

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
            # Any other post's <title> is its plain title (head.html).
            ({"lang": "en", "translation_id": "deepseek-v3"}, ["error"]),
        ],
    )
    def test_seo_title_names_only_a_korean_paper_posts_page(
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
        from scholar_lens.src.post_conventions import BlogPost

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
        from scholar_lens.src.post_validator import body_errors

        body = (
            "| A | B |\n| --- | --- |\n| 1 | 2 | 3 |\n\n"
            "$$a|b$$ and $$c$$ in one line.\n"
        )
        issues = body_errors(body)
        assert any(i.startswith("table:") and "3 cells" in i for i in issues)
        assert any(i.startswith("math:") and "bare |" in i for i in issues)
        assert body_errors("## 제목\n\n본문입니다.\n") == []

    def test_unclosed_fence_is_an_error_not_silent_code(self) -> None:
        from scholar_lens.src.post_validator import body_errors

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
            for v in validate_post(md, load_local_conventions(), "guide")
            if v.rule == "code"
        ]
        assert codes and codes[0].line == 15  # the ``` line, not the front matter

    def test_attribute_list_may_follow_a_display(self) -> None:
        body = "## A\n\n본문입니다.\n\n$$x = 1$$\n{: .center}\n\n끝입니다.\n"
        assert not [
            v
            for v in validate_post(self._post(body), load_local_conventions(), "guide")
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
        from scholar_lens.src.post_validator import unsourced_numbers

        assert unsourced_numbers(text, self.SOURCE) == missing

    @pytest.mark.parametrize(
        ("text", "read"),
        [
            # 만 as the particle "only" or in 만큼 is no unit.
            ("상위 5만 사용", [("5", "5")]),
            ("10만큼 늘었다", [("10", "10")]),
            ("32만을 썼다", [("32", "32")]),
            ("5만 개선", [("5", "5")]),
            # 만 is 10^4 before a number, a counter, a non-Hangul word, a
            # particle "only" never follows, punctuation or the end.
            ("5만 명", [("5만", "50000")]),
            ("10만 개의 문서", [("10만", "100000")]),
            ("10만 tokens", [("10만", "100000")]),
            ("100만에서 200만으로", [("100만", "1000000"), ("200만", "2000000")]),
            ("12만 8천", [("12만 8천", "128000")]),
            ("1억 2000만 사용", [("1억 2000만", "120000000")]),
            ("약 5만.", [("5만", "50000")]),
        ],
    )
    def test_man_is_a_unit_only_by_the_closed_rule(
        self, text: str, read: list[tuple[str, str]]
    ) -> None:
        from decimal import Decimal

        from scholar_lens.src.post_validator import _numbers

        assert [(n.text, n.absolute) for n in _numbers(text)] == [
            (t, Decimal(v)) for t, v in read
        ]

    def test_description_problems(self) -> None:
        from scholar_lens.src.post_validator import description_problems

        tldr = "DeepSeek-V3는 671B MoE 모델로 2.788M GPU 시간에 학습했습니다."
        assert description_problems(DESCRIPTION, tldr + " 37B H800") == []
        problems = description_problems("`DeepSeek-V3`는 99점입니다.", tldr)
        assert len(problems) == 3  # markup, length, an unsourced number
        assert "99" in problems[2]
        assert description_problems("", tldr) == ["description is empty"]
