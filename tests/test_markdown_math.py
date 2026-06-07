"""Tests for math-underscore normalization (kramdown+GFM blog safety)."""

from __future__ import annotations

from scholar_lens.src.markdown_math import normalize_math_underscores as norm


def test_inline_math_underscores_escaped() -> None:
    # Inline math is \( ... \) (the blog strips single-$ delimiters).
    assert norm(r"값은 \(W_0 + B_1 x\) 이다") == r"값은 \(W\_0 + B\_1 x\) 이다"


def test_display_math_underscores_escaped() -> None:
    src = "$$h = W_0 x + B A x$$"
    assert norm(src) == "$$h = W\\_0 x + B A x$$"


def test_inline_math_with_latex_commands_preserved() -> None:
    # Backslash commands (\alpha) must survive; only underscores are escaped.
    assert norm(r"\(\alpha_t = \beta_1\)") == r"\(\alpha\_t = \beta\_1\)"


def test_single_dollar_is_prose_not_math() -> None:
    # The blog strips $ delimiters, so $...$ is NOT math: underscores inside a
    # single-$ span must be left untouched (treated as prose).
    assert norm(r"식은 $W_0 + B_1$ 인데") == r"식은 $W_0 + B_1$ 인데"


def test_inline_code_untouched() -> None:
    assert norm("변수 `W_0` 사용") == "변수 `W_0` 사용"


def test_fenced_code_untouched() -> None:
    src = "```python\nW_0 = 1\nh_t = 2\n```"
    assert norm(src) == src


def test_prose_underscores_untouched() -> None:
    # Not math, not code — a literal filename. Must stay as-is.
    assert norm("파일 my_file_name 은 그대로") == "파일 my_file_name 은 그대로"


def test_already_escaped_is_idempotent() -> None:
    assert norm(r"\(a\_b\)") == r"\(a\_b\)"
    assert norm(norm(r"\(W_0\)")) == norm(r"\(W_0\)")


def test_currency_dollar_not_treated_as_math() -> None:
    # "$5 ... $10" must not be parsed as a math span swallowing the text.
    out = norm("비용은 $5_000 ... $10_000 원, 코드 `x_y`")
    assert "$5_000" in out  # currency/prose underscore preserved
    assert "`x_y`" in out  # code preserved


def test_mixed_math_and_code_in_one_line() -> None:
    out = norm(r"수식 \(a_b\) 와 코드 `c_d` 와 텍스트 e_f")
    assert r"\(a\_b\)" in out
    assert "`c_d`" in out
    assert "e_f" in out  # plain prose underscore untouched
