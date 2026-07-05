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


def test_double_backslash_inline_delims_collapsed() -> None:
    # \\( ... \\) renders as a literal backslash on the blog (math never
    # activates). Collapse to \( ... \) AND escape underscores inside.
    assert norm(r"노드 \\(\mathcal{N}_e\\)은") == r"노드 \(\mathcal{N}\_e\)은"


def test_double_backslash_display_delims_collapsed() -> None:
    # \\[ ... \\] -> \[ ... \] (delimiter activated).
    assert norm(r"식 \\[ x \\] 끝") == r"식 \[ x \] 끝"


def test_double_backslash_in_code_untouched() -> None:
    assert norm(r"코드 `\\(raw)` 유지") == r"코드 `\\(raw)` 유지"
    src = "```\n\\\\(x\\\\)\n```"
    assert norm(src) == src


def test_double_backslash_delim_idempotent() -> None:
    once = norm(r"\\(a_b\\)")
    assert once == r"\(a\_b\)"
    assert norm(once) == once


def test_row_break_with_spacing_in_display_math_preserved() -> None:
    # Regression: `\\[2pt]` inside $$...$$ is a LaTeX row break with spacing, NOT
    # an over-escaped display delimiter — the collapse must not touch it (it would
    # break aligned/array/cases environments).
    src = r"$$\begin{aligned} a &= b \\[2pt] c &= d \end{aligned}$$"
    assert norm(src) == src


def test_plain_row_break_in_display_math_preserved() -> None:
    src = r"$$a \\ b$$"
    assert norm(src) == src


def test_row_break_with_spacing_in_inline_math_preserved() -> None:
    # `\\` inside an active \(...\) span is a row break; leave it.
    src = r"\(a \\[1em] b\)"
    assert norm(src) == src
