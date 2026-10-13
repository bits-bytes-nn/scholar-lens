"""Markup rules shared with the blog's ``script/lint-posts.rb`` (PR #89): the
cases are the blog's own lint tests, plus the auto-fixes ``publish_body``
applies so a generated post meets them."""

from __future__ import annotations

import pytest

from scholar_lens.src.markdown_lint import publish_body
from scholar_lens.src.post_validator import errors, validate_post
from tests.test_post_validator import SPEC, make_post


def messages(body: str, **kwargs) -> list[str]:  # type: ignore[no-untyped-def]
    return [
        v.message
        for v in errors(validate_post(make_post(body), SPEC, "review", **kwargs))
    ]


@pytest.mark.parametrize(
    "body,expected",
    [
        ("답변은 <think> 태그로 감쌉니다.", "<think> is read as HTML"),
        ("문장 앞에 <s>를 붙입니다.", "<s> is read as HTML"),
        ("질문은 <input>에 넣습니다.", "<input> is read as HTML"),
        ("이미지 자리에 <img> 토큰이 들어갑니다.", "<img> is read as HTML"),
        ('```python title="a.py"\nx\n```', "info has a space"),
        ("| a | b |\n|---|---|\n| $$p(y|x)$$ | 2 |", "inside $$ in a table"),
        ("| a | b |\n|---|---|\n| $$\\|\\theta\\|$$ | 2 |", "inside $$ in a table"),
        ("```text\nQuestion: {{ question }}\n```", "Liquid tag"),
        ("`{% raw %}{{ q }}{% endraw %}`와 `{{ context }}`", "Liquid tag"),
    ],
)
def test_blog_lint_findings(body: str, expected: str) -> None:
    assert any(expected in m for m in messages(body))


@pytest.mark.parametrize(
    "body",
    [
        "모델은 \\<think> 로 시작합니다.",
        '<svg width="10"><rect width="1"/><path d="M0"/></svg>',
        "For $$i<j \\le n$$ and x<y z we sum.",
        "답변은 `<think>` 또는 \\<think\\> 태그, <br> 줄바꿈, <https://a.b> 링크.",
        "| Syntax | Example |\n|---|---|\n| `$$` | $$\\lvert x \\rvert$$ |",
        "{% raw %}\n```text\nQuestion: {{ question }}\n```\n{% endraw %}",
        "`{%- raw -%}{{ q }}{%endraw%}`",
    ],
)
def test_blog_lint_passes(body: str) -> None:
    rules = ("HTML", "info has a space", "inside $$ in a table", "Liquid")
    assert not [m for m in messages(body) if any(r in m for r in rules)]


def test_description_must_be_plain() -> None:
    post = make_post(front={"description": '"R1은 <think> 블록을 씁니다"'})
    found = errors(validate_post(post, SPEC, "review"))
    assert any(v.rule == "description" for v in found)
    plain = make_post(front={"description": '"Mixtral 8*7B와 C++ [1]"'})
    assert not [
        v
        for v in errors(validate_post(plain, SPEC, "review"))
        if v.rule == "description"
    ]


def test_translation_needs_lang() -> None:
    post = make_post(front={"translation_id": "x"})
    assert any(
        "must set lang" in v.message
        for v in errors(validate_post(post, SPEC, "review"))
    )


@pytest.mark.parametrize(
    "name,ok",
    [
        ("2024-12-27-deepseek-v3.md", True),
        ("2024-12-26-deepseek-v3.md", False),  # date: 2024-12-27 is later
        ("deepseek-v3.md", False),
    ],
)
def test_file_name_date(name: str, ok: bool) -> None:
    found = [m for m in messages("본문입니다.", file_name=name) if "file name" in m]
    assert (not found) is ok


def test_publish_body_meets_the_markup_rules() -> None:
    body = (
        "본문 `{{ q }}`와 {% if %}, 토큰 <think>와 <img>, 줄바꿈 <br>.\n\n"
        '```python title="a.py"\nprompt = "{{ question }}"\n```\n\n'
        "| a | b |\n|---|---|\n| $$\\|x\\|_2$$ | $$p(y|x)$$ |\n"
    )
    out = publish_body(body, warn=False)
    assert publish_body(out, warn=False) == out
    assert "```python\n" in out and "{% raw %}\n```python" in out
    assert "&lt;think&gt;" in out and "&lt;img&gt;" in out and "<br>" in out
    rules = ("HTML", "info has a space", "inside $$ in a table", "Liquid")
    assert not [m for m in messages(out) if any(r in m for r in rules)]
