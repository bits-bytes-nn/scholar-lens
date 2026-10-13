"""Markup rules shared with the blog's ``script/lint-posts.rb`` (PR #89): the
cases are the blog's own lint tests, plus the auto-fixes ``publish_body``
applies so a generated post meets them."""

from __future__ import annotations

import pytest

from scholar_lens.src.markdown_lint import publish_body
from scholar_lens.src.post_validator import errors, validate_post
from tests.test_post_validator import DESCRIPTION, SPEC, make_post


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
    plain = make_post(front={"description": f'"Mixtral 8*7B와 C++ [1]. {DESCRIPTION}"'})
    assert not [
        v
        for v in errors(validate_post(plain, SPEC, "review"))
        if v.rule == "description"
    ]


@pytest.mark.parametrize(
    "name,front,ok",
    [
        ("2026-04-05-x.md", {"translation_id": "x", "lang": "ko"}, True),
        ("2026-04-05-x-en.md", {"translation_id": "x", "lang": "en"}, True),
        ("2026-04-05-y.md", {"translation_id": "x", "lang": "ko"}, False),
        ("2026-04-05-x-en.md", {}, False),  # an -en post sets translation_id
    ],
)
def test_translation_file_name(name: str, front: dict[str, str], ok: bool) -> None:
    found = validate_post(make_post(front=front), SPEC, "review", file_name=name)
    assert (not [v for v in errors(found) if v.rule == "translation"]) is ok


def test_a_published_translation_keeps_its_pair() -> None:
    from scholar_lens.src.post_conventions import BlogPost

    twin = BlogPost(path="_posts/2026-04-06-x-en.md", lang="en", translation_id="x")
    name = "2026-10-13-x.md"
    unpaired = validate_post(
        make_post(), SPEC, "review", file_name=name, site_posts=[twin]
    )
    assert any("is this post's translation" in v.message for v in errors(unpaired))
    paired = make_post(front={"translation_id": "x", "lang": "ko"})
    found = validate_post(paired, SPEC, "review", file_name=name, site_posts=[twin])
    assert not [v for v in errors(found) if v.rule == "translation"]
    clash = make_post(front={"translation_id": "x", "lang": "en"})
    found = validate_post(clash, SPEC, "review", file_name=name, site_posts=[twin])
    assert any("already has lang en" in v.message for v in errors(found))


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


class TestAr5ivBan:
    """ar5iv serves a "NO IMAGE AVAILABLE" PNG with HTTP 200 for a figure it
    lost (59 blog figures went blank), so the blog lint bans its links."""

    @pytest.mark.parametrize(
        "body",
        [
            "![구조](https://ar5iv.labs.arxiv.org/html/2106.09685/assets/x1.png)",
            "![구조](https://ar5iv.org/html/2106.09685/assets/x1.png)",
            "[논문](https://ar5iv.org/abs/2106.09685)을 참고합니다.",
            "> 출처: https://ar5iv.labs.arxiv.org/html/2106.09685",
        ],
    )
    def test_ar5iv_link_is_an_error(self, body: str) -> None:
        assert any("ar5iv link" in m for m in messages(body))

    @pytest.mark.parametrize(
        "body",
        [
            "주소는 `ar5iv.org` 형식입니다.",
            "```text\nhttps://ar5iv.org/html/2106.09685\n```",
            "![구조](https://arxiv.org/html/2106.09685v2/x1.png)",
        ],
    )
    def test_code_and_arxiv_html_pass(self, body: str) -> None:
        assert not [m for m in messages(body) if "ar5iv" in m]


class TestLightbox:
    """A glightbox link opens its href; a page there loads as a blank iframe."""

    def test_page_href_is_an_error(self) -> None:
        body = (
            '본문입니다.\n\n<a href="https://docs.example.com/guide" class="glightbox" '
            'data-glightbox="title: 구조 -> 흐름">\n'
            '<img src="/assets/images/guide-1.png" alt="구조"></a>'
        )
        found = [
            v
            for v in errors(validate_post(make_post(body), SPEC, "review"))
            if v.rule == "lightbox"
        ]
        assert len(found) == 1
        assert "set href to /assets/images/guide-1.png" in found[0].message
        # The line of the <a> tag, as the blog lint reports it.
        assert (
            found[0].line == make_post(body).split("\n").index(body.split("\n")[2]) + 1
        )

    @pytest.mark.parametrize(
        "body",
        [
            '<a href="/assets/images/g-1.png" class="glightbox">'
            '<img src="/assets/images/g-1.png" alt="구조"></a>',
            # Not a lightbox: an ordinary link around an image.
            '<a href="https://docs.example.com"><img src="/assets/images/g-1.png" '
            'alt="구조"></a>',
            '```html\n<a href="https://x.dev" class="glightbox"><img src="/a.png"></a>\n```',
        ],
    )
    def test_matching_href_and_code_pass(self, body: str) -> None:
        assert not [m for m in messages(body) if "glightbox" in m]
