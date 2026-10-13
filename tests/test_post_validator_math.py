"""Math, table and date rules shared with the blog's ``script/lint-posts.rb``
(PR #86): the cases are the blog's own lint tests."""

from __future__ import annotations

import pytest

from scholar_lens.src.markdown_math import normalize_math
from scholar_lens.src.post_validator import (
    errors,
    unsourced_table_numbers,
    validate_post,
)
from tests.test_post_validator import SPEC, make_post


def math_errors(body: str) -> list[str]:
    found = errors(validate_post(make_post(body), SPEC, "review"))
    return [v.message for v in found if v.rule == "math"]


@pytest.mark.parametrize(
    "body",
    [
        "Intro.\n\n$$y = \\begin{cases}\n1 & x > 0 \\\\\n0\n\\end{cases}$$\n\nTail.",
        "식 $$a +\n\\frac{1}{b}$$ 입니다.",
        "> 인용\n>\n> $$\n> a = b\n> $$\n>\n> 끝",
        "Intro\n$$x$$는 $$y$$보다 큽니다.",
        "Intro.\n\n$$a \\\\[4pt] b$$\n\nTail.",
        "It cost $10.2B and `\\\\(x\\\\)` is how not to write it.",
        "Intro.\n\n$$\nx_i\n$$\n\nTail.",
        "**LongMemEval \\(s\\)** 데이터셋",
    ],
)
def test_blog_valid_math_passes(body: str) -> None:
    assert math_errors(body) == []


@pytest.mark.parametrize(
    "body,message",
    [
        ("Intro\n$$\nx\n$$\n\nTail.", "blank line before"),
        ("Intro.\n\n$$\nx\n$$\nTail.", "blank line after"),
        ("Intro.\n\n$$y = \\begin{cases}\ne\\_i\n\\end{cases}$$\n\nTail.", r"\_"),
        ("가격은 $$5입니다.\n\n다음 문단.", "unbalanced"),
        ("> 문장\n> $$x=1$$\n> 문장", "blank lines around"),
        ("Intro.\n\n$$a$$ and $$b\n# Title again", "never closed"),
        ("별표(\\\\(*\\\\))는 \\\\(\\phi^{*}\\\\)입니다.", r"\( or \["),
        ("Intro.\n\n\\\\[ y = x \\\\]\n\nTail.", r"\( or \["),
        ("/\\pi_{\\text{ref}}$를 초기화합니다.", "single `$`"),
    ],
)
def test_blog_math_defects_are_errors(body: str, message: str) -> None:
    assert any(message in m for m in math_errors(body)), math_errors(body)


def test_normalize_fixes_what_it_can_before_validation() -> None:
    body = normalize_math("인라인 \\(x_i\\)와\n\n\\[y = x\\]\n\n끝입니다.")
    assert math_errors(body) == []


@pytest.mark.parametrize(
    "before,after",
    [
        (
            r"$$r=\frac{\pi_{\theta}(o_{t} \Vert q)}{\pi_{\mathrm{old}}(o_{t} \Vert q)}$$",
            r"$$r=\frac{\pi_{\theta}(o_{t} \mid q)}{\pi_{\mathrm{old}}(o_{t} \mid q)}$$",
        ),
        (
            r"$$\frac{1}{\lVert o_{i} \rVert}\sum_{t=1}^{\lVert o_{i} \rVert} a$$",
            r"$$\frac{1}{\lvert o_{i} \rvert}\sum_{t=1}^{\lvert o_{i} \rvert} a$$",
        ),
        (r"$$p(x \| y)$$", r"$$p(x \mid y)$$"),
    ],
)
def test_conditional_and_length_bars_are_normalised(before: str, after: str) -> None:
    assert normalize_math(before) == after


@pytest.mark.parametrize(
    "math",
    [
        r"$$\mathbb{D}_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\mathrm{ref}})$$",
        r"$$D_{KL}(p\|q)$$",
        r"$$\lVert W \rVert_F + \|x\|_2$$",
    ],
)
def test_norms_and_divergences_keep_their_bars(math: str) -> None:
    assert normalize_math(math) == math


def test_last_modified_at_is_never_written() -> None:
    post = make_post(front={"last_modified_at": "2026-10-12"})
    found = errors(validate_post(post, SPEC, "review"))
    assert any("last_modified_at" in v.message for v in found)


def test_adjacent_identical_table_rows_are_an_error() -> None:
    table = "| A | B |\n| --- | --- |\n| x | 1 |\n| x | 1 |\n| y | 2 |"
    found = errors(validate_post(make_post(table), SPEC, "review"))
    assert any("repeats the row above" in v.message for v in found)


def test_table_rows_must_come_from_the_paper() -> None:
    source = "Table 4: respondents 272 122 199\nwin rate 0.36 and 0.26"
    good = make_post(
        "| 비교 | A | B | C |\n| --- | --- | --- | --- |\n| 응답자 수 | 272 | 122 | 199 |"
    )
    bad = make_post(
        "| 비교 | A | B | C |\n| --- | --- | --- | --- |\n| 응답자 수 | 27 | 21 | 22 |"
    )
    assert unsourced_table_numbers(good, source) == []
    assert unsourced_table_numbers(bad, source) == ["응답자 수: 27 / 21 / 22"]
    regrouped = make_post(
        "| 온도 | 값 | 값 |\n| --- | --- | --- |\n| 0 | 0.36 | 0.26 |"
    )
    assert unsourced_table_numbers(regrouped, source) == []


@pytest.mark.parametrize(
    "body",
    [
        "값은 $$x +\ny$$ 이고 $$z$$ 입니다.",  # the line closing a wrap
        "앞 문장입니다\n$$x +\ny$$ 입니다.",  # a wrap that starts a line
    ],
)
def test_wrapped_inline_span_closing_line_is_checked_after_the_close(
    body: str,
) -> None:
    assert math_errors(body) == []


def test_wrapped_inline_span_with_korean_is_still_reported() -> None:
    assert any("Korean" in m for m in math_errors("값은 $$x + 가\ny$$ 입니다."))


@pytest.mark.parametrize(
    "body",
    [
        "손실은 다음과 같습니다. \\[\nL = \\sum_i \\log p(y_i \\mid x_i)\n\\]",
        "\\[\nL = \\sum_i x_i\n\\] 여기서 x는 입력입니다.",
        "- 손실 \\[\n  L = x_1\n  \\]",
    ],
)
def test_bracket_display_glued_to_prose_is_an_error(body: str) -> None:
    assert any("\\[" in e for e in math_errors(normalize_math(body)))


def test_indented_code_is_not_checked_as_math() -> None:
    assert math_errors("앞 문단.\n\n    $$|x|$$ code\n\n뒤") == []


@pytest.mark.parametrize(
    "body", ["배열은 a\\[\ni\\] 처럼 씁니다.", "옵션 --max-len \\[\n기본값\\]"]
)
def test_bracket_over_lines_without_math_is_not_math(body: str) -> None:
    assert math_errors(body) == []


def test_unbalanced_dollars_across_sibling_items_are_reported() -> None:
    found = math_errors("- A100: $$2/h\n- H100: $$4/h")
    assert sum("unbalanced" in m for m in found) == 2


def test_ordered_list_restarting_after_a_display_warns() -> None:
    from tests.test_post_validator import warnings

    post = make_post("1. 첫째\n\n$$x$$\n\n2. 둘째")
    found = [
        v for v in warnings(validate_post(post, SPEC, "review")) if v.rule == "list"
    ]
    assert found and "restarts at 2" in found[0].message
    indented = make_post("1. 첫째\n\n   $$x$$\n\n2. 둘째")
    assert not [
        v for v in warnings(validate_post(indented, SPEC, "review")) if v.rule == "list"
    ]


@pytest.mark.parametrize(
    "body",
    [
        "1. 첫째\n\n$$x$$\n\n여기서 x는 입력입니다.\n\n2. 둘째",  # display + paragraph
        "  1. a\n  $$x$$\n  2. b",  # glued, indented less than the item's content
    ],
)
def test_list_restart_after_a_display_and_paragraph_warns(body: str) -> None:
    from tests.test_post_validator import warnings

    found = [
        v
        for v in warnings(validate_post(make_post(body), SPEC, "review"))
        if v.rule == "list"
    ]
    assert found and "restarts at 2" in found[0].message


def test_inline_math_starting_a_continuation_line_is_no_display() -> None:
    from tests.test_post_validator import warnings

    post = make_post("1. 첫째\n$$x$$는 값입니다\n2. 둘째")
    assert not [
        v for v in warnings(validate_post(post, SPEC, "review")) if v.rule == "list"
    ]


def test_code_that_kramdown_renders_as_code_is_not_math() -> None:
    # The blog lint also reads these as prose; its own CI on the blog PR
    # judges them, so the validator only checks what kramdown shows as text.
    for body in ("> ```bash\n> echo $$\n> ```", "앞 문단.\n\n    echo $$ > pid"):
        assert math_errors(body) == []


def test_fenced_code_and_our_own_rules_are_not_applied_to_code() -> None:
    assert math_errors("```bash\necho $$\n```") == []
    # Bare | is our rule, not the blog lint's: indented code is left alone.
    assert math_errors("앞 문단.\n\n    $$|x|$$ code\n\n뒤") == []


def test_bare_pipe_on_a_continuation_line_renders_and_passes() -> None:
    assert math_errors("앞 문장입니다.\n특수 토큰 $$\\text{<|end|>}$$ 입니다.") == []
    assert any(
        "bare |" in m for m in math_errors("특수 토큰 $$\\text{<|end|>}$$ 입니다.")
    )


def test_callout_inside_a_numbered_step_is_no_list_restart() -> None:
    from tests.test_post_validator import warnings

    post = make_post("1. 첫\n   > **핵심:** 식입니다.\n   > $$x$$\n2. 둘째")
    assert not [
        v for v in warnings(validate_post(post, SPEC, "review")) if v.rule == "list"
    ]


@pytest.mark.parametrize(
    "body",
    [  # code to kramdown, prose to the blog lint, which accepts these
        '> ```python\n> re.compile(r"\\[\\d+\\]")\n> ```',
        '앞 문단.\n\n    pid = "$$"\n    # cleanup "$$"',
    ],
)
def test_code_the_blog_lint_accepts_passes(body: str) -> None:
    assert math_errors(body) == []


@pytest.mark.parametrize(
    "body,rejected",
    [
        ("### 소제목\n확률 $$\\text{a|b}$$ 입니다.", False),
        ("```\ncode\n```\n확률 $$\\text{a|b}$$ 입니다.", False),
        ("확률 $$\\text{a|b}$$ 입니다.", True),
    ],
)
def test_bar_rule_only_where_kramdown_starts_a_table(body: str, rejected: bool) -> None:
    assert any("bare |" in m for m in math_errors(body)) is rejected


def test_korean_jamo_inside_inline_math_is_reported() -> None:
    assert any("Korean" in m for m in math_errors("조건 $$a ㉠ b$$ 입니다."))


def test_list_inside_a_quote_does_not_warn() -> None:
    from tests.test_post_validator import warnings

    post = make_post("> 1. 첫\n>\n>    $$x$$\n>\n> 2. 둘")
    assert not [
        v for v in warnings(validate_post(post, SPEC, "review")) if v.rule == "list"
    ]


@pytest.mark.parametrize("body", ["$$\nL = a + 손실\n$$", "$$L = a + 손실$$"])
def test_korean_in_display_math_is_an_error_like_the_blog_lint(body: str) -> None:
    assert any("Korean" in e for e in math_errors(body))
