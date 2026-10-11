"""Tests for math normalization to the blog's ``$$ ... $$`` convention."""

from __future__ import annotations

import pytest

from scholar_lens.src.markdown_math import has_math
from scholar_lens.src.markdown_math import normalize_math as norm


@pytest.mark.parametrize(
    "src,expected",
    [
        (r"값은 \(W_0 + B_1 x\) 이다", r"값은 $$W_0 + B_1 x$$ 이다"),
        (r"노드 \\(\mathcal{N}_e\\)은", r"노드 $$\mathcal{N}_e$$은"),
        (r"\(\alpha_t = \beta_1\)", r"$$\alpha_t = \beta_1$$"),
    ],
)
def test_paren_delimiters_become_inline_double_dollar(src: str, expected: str) -> None:
    assert norm(src) == expected


def test_bracket_delimiters_become_display_block() -> None:
    assert norm("식\n\\[ x_1 \\]\n끝") == "식\n\n$$x_1$$\n\n끝"


def test_underscore_escapes_removed_inside_math() -> None:
    assert norm(r"$$h = W\_0 x$$") == r"$$h = W_0 x$$"


def test_underscore_escapes_kept_inside_text_commands() -> None:
    src = r"$$\text{batch\_size} + \texttt{max\_len} = b\_1$$"
    assert norm(src) == r"$$\text{batch\_size} + \texttt{max\_len} = b_1$$"


def test_display_block_gets_blank_lines() -> None:
    assert (
        norm("앞 문장\n$$h = W_0 x$$\n뒤 문장") == "앞 문장\n\n$$h = W_0 x$$\n\n뒤 문장"
    )


def test_multiline_display_block_gets_blank_lines() -> None:
    src = "text\n$$\n\\begin{aligned} a &= b \\\\ c &= d \\end{aligned}\n$$\nmore"
    out = norm(src)
    assert "text\n\n$$\n" in out and "\n$$\n\nmore" in out


def test_inline_math_in_sentence_not_isolated() -> None:
    src = "the weight $$W_0$$ stays frozen"
    assert norm(src) == src


@pytest.mark.parametrize(
    "src",
    [
        "| a | $$x_1$$ |\n|---|---|",  # table row
        "- item\n  $$x$$\n",  # indented list continuation
    ],
)
def test_tables_and_lists_left_alone(src: str) -> None:
    assert norm(src) == src


def test_single_dollar_is_currency_text() -> None:
    src = "비용은 $5_000 ... $10 입니다"
    assert norm(src) == src


@pytest.mark.parametrize(
    "src",
    [
        "변수 `W_0` 사용",
        r"코드 `\(raw\)` 유지",
        "```python\nW_0 = 1\n\n\n\nh = r'\\(x\\)'\n```",
    ],
)
def test_code_untouched(src: str) -> None:
    assert norm(src) == src


def test_row_breaks_in_display_math_preserved() -> None:
    src = r"$$\begin{aligned} a &= b \\[2pt] c &= d \end{aligned}$$"
    assert norm(src) == src


@pytest.mark.parametrize(
    "src",
    [
        r"수식 \(a_b\) 와 코드 `c_d` 와 텍스트 e_f",
        "앞\n\\[ y \\]\n뒤",
        r"$$W\_0$$ and \\(x\\)",
    ],
)
def test_idempotent(src: str) -> None:
    once = norm(src)
    assert norm(once) == once


def test_has_math_ignores_code() -> None:
    assert has_math("a $$x$$ b")
    assert not has_math("plain $5 text")
    assert not has_math("`$$x$$` in code")
