"""Post validator: blog defects are errors, the hand-fixed versions pass.

Fixtures in ``tests/fixtures/blog_defects`` are blocks cut from the blog's
``main`` posts (``*.before.md``, as generated) and the same blocks after the
manual fixes on the ``ui-visual-polish`` branch (``*.after.md``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scholar_lens.src.post_conventions import load_local_conventions
from scholar_lens.src.post_validator import errors, validate_post, warnings

FIXTURES = Path(__file__).parent / "fixtures" / "blog_defects"
SPEC = load_local_conventions()


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
            '"Edward Hu, Yelong Shen, Phillip Wallis"',
            '""',
        ],
    )
    def test_paper_author_must_be_an_organisation(self, author: str) -> None:
        assert "paper-author" in rules(make_post(front={"paper_author": author}))

    @pytest.mark.parametrize(
        "author", ['"Zep AI"', '"Google DeepMind"', '"Meta AI and New York University"']
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
        assert "math" in rules(make_post(r"변수 \(x\)를 씁니다."))

    def test_bare_pipe_in_crowded_inline_math(self) -> None:
        assert "math" in rules(make_post("값 $$|x|$$ 와 $$y$$ 입니다."))
        assert "math" not in rules(
            make_post(r"값 $$\lvert x \rvert$$ 와 $$y$$ 입니다.")
        )

    def test_underscore_escape_allowed_inside_text(self) -> None:
        assert "math" not in rules(make_post(r"값 $$\text{batch\_size}$$ 입니다."))

    def test_images(self) -> None:
        assert "image" in rules(
            make_post("![a](https://arxiv.org/html/2412.19437v1/x1.png)")
        )
        assert "image" in rules(make_post("![a](/assets/2412/x1.png)"))
        assert rules(make_post("![a](/assets/images/deepseek-v3-1.png)")) == set()
        assert "image" in rules(
            make_post("![a](https://docs.x.dev/a.png)"), level="warning"
        )

    def test_envelope_tag_leak(self) -> None:
        assert "envelope" in rules(make_post("본문</explanation>"))

    def test_code_blocks_are_ignored(self) -> None:
        body = "```python\n# comment\n$$ x\\_1 $$\n| a |\n```"
        assert rules(make_post(body, front={"use_math": "false"})) == set()


class TestDescription:
    def test_math_in_first_paragraph_warns(self) -> None:
        post = make_post().replace(
            "짧은 답입니다.", "변화량 $$\\Delta W$$를 학습합니다."
        )
        assert "description" in rules(post, level="warning")
        assert "description" not in rules(make_post(), level="warning")


def test_leaked_figure_marker_is_an_error() -> None:
    post = make_post('[Image: alt="x", src="/assets/images/a-1.png", caption="c"]')
    assert "envelope" in rules(post)


def test_future_date_is_an_error() -> None:
    assert "date" in rules(make_post(front={"date": "2999-01-01 00:00:00"}))
    assert "date" not in rules(make_post())
