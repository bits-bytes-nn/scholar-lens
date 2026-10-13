"""Tests for math normalization to the blog's ``$$ ... $$`` convention."""

from __future__ import annotations

import pytest

from scholar_lens.src.markdown_math import _fix_bars, has_math
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


class TestFenceSpans:
    def test_four_backtick_fence_holds_a_three_backtick_line(self) -> None:
        from scholar_lens.src.markdown_math import fence_spans

        text = "Intro\n\n````md\n```\na\n```\nx = \\(y\\_z\\)\n````\nafter \\(q\\)\n"
        ((start, end),) = fence_spans(text)
        assert text[start:end].endswith("````")
        out = norm(text)
        assert "x = \\(y\\_z\\)" in out  # code untouched
        assert "after $$q$$" in out

    def test_tilde_inside_backtick_fence_is_code(self) -> None:
        from scholar_lens.src.markdown_math import fence_roles

        lines = ["```bash", "cat <<X", "~~~", "X", "```", "# Heading"]
        assert fence_roles(lines) == ["open", "code", "code", "code", "close", "prose"]

    def test_unclosed_fence_runs_to_the_end(self) -> None:
        from scholar_lens.src.markdown_math import fence_spans, open_fence

        text = "a\n```py\nx = 1\n"
        assert fence_spans(text) == [(2, len(text))]
        assert open_fence(text) == "```"
        assert open_fence(text + "```\n") == ""

    def test_backtick_info_string_with_backtick_is_not_a_fence(self) -> None:
        from scholar_lens.src.markdown_math import fence_roles

        assert fence_roles(["``` `x` ```", "text"]) == ["prose", "prose"]


class TestBars:
    @pytest.mark.parametrize(
        "math,expected",
        [
            (r"\Pr(\|X - \mu\| \geq k\sigma)", r"\Pr(\|X - \mu\| \geq k\sigma)"),
            (r"P(\|w\|_2 > c)", r"P(\|w\|_2 > c)"),
            (r"\pi_\theta(y \| x)", r"\pi_\theta(y \mid x)"),
            (r"p_\theta(x_t \| x_{t-1})", r"p_\theta(x_t \mid x_{t-1})"),
            (r"e^{\|x\|}", r"e^{\|x\|}"),
            (r"D_{KL}(p \| q)", r"D_{KL}(p \| q)"),
            (
                r"\sum_{t=1}^{\lVert o \rVert} \frac{1}{\lVert o \rVert} \|o\|_2",
                r"\sum_{t=1}^{\lvert o \rvert} \frac{1}{\lvert o \rvert} \|o\|_2",
            ),
        ],
    )
    def test_conditionals_and_lengths_only(self, math: str, expected: str) -> None:
        assert _fix_bars(math) == expected

    def test_operatorname_keeps_its_escaped_underscore(self) -> None:
        assert norm(r"$$\operatorname{top\_k}(x)$$") == (
            r"$$\operatorname{top\_k}(x)$$"
        )


@pytest.mark.parametrize(
    "math,expected",
    [
        (r"\pi_\theta(y\|x)", r"\pi_\theta(y\mid x)"),  # never \midx
        (r"p(x\|y)p(y\|z)", r"p(x\mid y)p(y\mid z)"),
        (r"\pi^*(y \| x)", r"\pi^*(y \mid x)"),
        (r"\mathbb{P}(A \| B)", r"\mathbb{P}(A \mid B)"),
        (r"\Pr[A \| B]", r"\Pr[A \mid B]"),
        (r"D_p(x \| y)", r"D_p(x \| y)"),  # a divergence, not a call
        (r"\ell_p(x \| y)", r"\ell_p(x \| y)"),
        (r"Q(s \| a)", r"Q(s \| a)"),  # the action value
    ],
)
def test_conditional_bars(math: str, expected: str) -> None:
    assert _fix_bars(math) == expected
    assert _fix_bars(expected) == expected


@pytest.mark.parametrize(
    "src,expected",
    [
        (
            "1. 손실:\n   \\[ y = f(x) \\]\n2. 다음\n",
            "1. 손실:\n\n   $$y = f(x)$$\n\n2. 다음\n",
        ),
        (
            "1. 손실:\n   \\[\n   L = x\n   \\]\n2. 다음\n",
            "1. 손실:\n\n   $$L = x$$\n\n2. 다음\n",
        ),
        ("- 항목\n  \\\\[ y \\\\]\n- 다음\n", "- 항목\n\n  $$y$$\n\n- 다음\n"),
    ],
)
def test_indented_bracket_display_stays_in_its_list_item(
    src: str, expected: str
) -> None:
    assert norm(src) == expected
    assert norm(expected) == expected


def test_indent_outside_a_list_is_dropped() -> None:
    # Four spaces outside a list would make the display a code block.
    assert "    $$" not in norm("본문입니다.\n\n    \\[ y \\]\n")


def test_norm_on_a_table_row_is_written_vert() -> None:
    assert norm("| x | $$\\|x\\|_2$$ |\n") == "| x | $$\\Vert x\\Vert_2$$ |\n"
    assert norm("본문 $$\\|x\\|$$ 입니다.\n") == "본문 $$\\|x\\|$$ 입니다.\n"


class TestInlineBars:
    """A bare | in inline math is spelled \\vert (kramdown would read the line
    as a table row); display blocks, \\text{} and array specs are kept."""

    @pytest.mark.parametrize(
        "src,out",
        [
            (
                "집합의 크기 $$|S|$$ 입니다.\n",
                "집합의 크기 $$\\vert S\\vert$$ 입니다.\n",
            ),
            (
                "집합 $$\\{x | x > 0\\}$$ 입니다.\n",
                "집합 $$\\{x \\vert x > 0\\}$$ 입니다.\n",
            ),
            ("$$\\left| x \\right|$$ 값.\n", "$$\\left\\vert x \\right\\vert$$ 값.\n"),
            ("조건부 $$p(y|x)$$ 와\n", "조건부 $$p(y\\vert x)$$ 와\n"),
            (
                "크기는 $$|S| +\n|T|$$ 입니다.\n",
                "크기는 $$\\vert S\\vert + \\vert T\\vert$$ 입니다.\n",
            ),
            (
                "| a | b |\n| - | - |\n| 크기 | $$|S|$$ |\n",
                "| a | b |\n| - | - |\n| 크기 | $$\\vert S\\vert$$ |\n",
            ),
            ("- 크기 $$|S|$$ 입니다.\n", "- 크기 $$\\vert S\\vert$$ 입니다.\n"),
            ("> 크기 $$|S|$$ 입니다.\n", "> 크기 $$\\vert S\\vert$$ 입니다.\n"),
        ],
    )
    def test_rewritten(self, src: str, out: str) -> None:
        assert norm(src) == out
        assert norm(out) == out

    @pytest.mark.parametrize(
        "src",
        [
            "앞\n\n$$S = \\{x | x > 0\\}$$\n\n뒤\n",  # a display block
            "앞\n\n$$\nS = |x|\n$$\n\n뒤\n",
            "$$\\text{a | b}$$ 끝.\n",
            "$$\\begin{array}{c|c} a & b \\end{array}$$ 인라인\n",
            "코드 `$$|x|$$` 입니다.\n",
        ],
    )
    def test_kept(self, src: str) -> None:
        assert norm(src) == src


class TestDisplayPlacement:
    def test_bracket_in_a_blockquote_stays_in_the_quote(self) -> None:
        out = norm("> 인용.\n> \\[ y = f(x) \\]\n> 끝.\n")
        assert out == "> 인용.\n>\n> $$y = f(x)$$\n>\n> 끝.\n"

    def test_table_norm_in_a_blockquote_uses_vert(self) -> None:
        src = "> | n | 식 |\n> | --- | --- |\n> | 1 | $$\\|x\\|_2$$ |\n"
        assert "$$\\Vert x\\Vert_2$$" in norm(src)

    @pytest.mark.parametrize(
        "src,out",
        [
            ("본문.\n\n    $$x = 1$$\n\n끝.\n", "본문.\n\n$$x = 1$$\n\n끝.\n"),
            (
                "본문.\n\n    $$\n    x = 1\n    $$\n\n끝.\n",
                "본문.\n\n$$\nx = 1\n$$\n\n끝.\n",
            ),
            (
                "- 항목\n\n        \\[ L = x \\]\n\n- 다음\n",
                "- 항목\n\n  $$L = x$$\n\n- 다음\n",
            ),
            (
                "1. 항목\n\n        $$L = x$$\n\n2. 다음\n",
                "1. 항목\n\n   $$L = x$$\n\n2. 다음\n",
            ),
            (
                "1. 항목\n\n   $$L = x$$\n\n2. 다음\n",
                "1. 항목\n\n   $$L = x$$\n\n2. 다음\n",
            ),
        ],
    )
    def test_indented_display_is_not_code(self, src: str, out: str) -> None:
        # Four or more spaces outside a list (or past the item's content
        # column) is an indented code block in kramdown.
        assert norm(src) == out
        assert norm(out) == out


@pytest.mark.parametrize(
    "src,out",
    [
        (r"$$\hat{p}(y\|x)$$", r"$$\hat{p}(y\mid x)$$"),
        (r"$$p'(y \| x)$$", r"$$p'(y \mid x)$$"),
        (r"$$\tilde{\pi}(a\|s)$$", r"$$\tilde{\pi}(a\mid s)$$"),
        (r"$$p\left(x\|y\right)$$", r"$$p\left(x\mid y\right)$$"),
        (
            r"$$\pi_{\theta_{\text{old}}}(o\|q)$$",
            r"$$\pi_{\theta_{\text{old}}}(o\mid q)$$",
        ),
        (r"$$D_p(x \| y)$$", r"$$D_p(x \| y)$$"),
        (r"$$\exp(a\|b)$$", r"$$\exp(a\|b)$$"),
    ],
)
def test_decorated_conditional_heads(src: str, out: str) -> None:
    assert norm(f"앞\n\n{src}\n\n뒤\n") == f"앞\n\n{out}\n\n뒤\n"


def test_stray_dollars_before_a_table_leave_the_table_alone() -> None:
    from scholar_lens.src.markdown_lint import lint_markdown
    from scholar_lens.src.post_validator import body_errors

    src = (
        "## 비용\n\n비용은 토큰당 $$0.5 수준입니다.\n\n| 모델 | 정확도 |\n"
        "| --- | --- |\n| A | 81.2 |\n\n손실은 다음과 같습니다.\n\n"
        "$$L = \\|x\\|^2$$\n\n끝.\n"
    )
    out = lint_markdown(norm(src), warn=False)
    assert "| 모델 | 정확도 |" in out and "\\vert" not in out
    assert any("unbalanced" in e for e in body_errors(out))
    assert body_errors(out.replace("$$0.5", "$0.5")) == []
    assert norm(out) == out


@pytest.mark.parametrize(
    "math,kept",
    [
        (r"\begin{array}{|c|p{2cm}|} a & b \end{array}", "{|c|p{2cm}|}"),
        (r"\text{a {b {c}} |} + x", r"\text{a {b {c}} |}"),
        (r"\href{https://x.io/a|b}{y} = z", r"\href{https://x.io/a|b}"),
    ],
)
def test_bar_rewrite_keeps_nested_text_array_specs_and_urls(
    math: str, kept: str
) -> None:
    out = norm(f"값은 $${math}$$ 입니다.")
    assert kept in out


def test_attribute_list_stays_glued_to_its_display() -> None:
    src = "앞 문단.\n\n$$x = 1$$\n{: .eq}\n\n뒤 문단.\n"
    assert "$$x = 1$$\n{: .eq}" in norm(src)
    assert norm(norm(src)) == norm(src)


def test_quoted_display_in_a_list_item_keeps_its_indent() -> None:
    src = "1. 항목입니다.\n\n   > 인용입니다.\n   > \\[ a = b \\]\n\n끝.\n"
    out = norm(src)
    assert "   > $$a = b$$" in out and norm(out) == out


def test_display_glued_in_a_list_item_gets_blank_lines() -> None:
    assert norm("- item\n  $$x$$\n") == "- item\n\n  $$x$$\n"


class TestRound20Placement:
    """Placements kramdown reads differently (each checked by rendering)."""

    def test_blank_line_inside_a_block_display_is_still_one_formula(self) -> None:
        src = "문단.\n\n    $$\n    a = b\n\n    c = d\n    $$\n\n뒤."
        assert norm(src) == "문단.\n\n$$\na = b\n\nc = d\n$$\n\n뒤."
        block = "앞\n\n$$\nx\\_t = p(y \\| x)\n\nc = d\n$$\n\n뒤"
        assert "x_t = p(y \\mid x)" in norm(block)

    @pytest.mark.parametrize(
        "src,out",
        [
            ("값 $$a +\nb$$ 와 $$|c|$$ 끝.", "값 $$a + b$$ 와 $$\\vert c\\vert$$ 끝."),
            ("$$a +\nb$$ 는 식입니다.", "$$a + b$$ 는 식입니다."),
            ("> 값 $$a +\n> b$$ 끝.", "> 값 $$a + b$$ 끝."),
        ],
    )
    def test_wrapped_inline_span_is_joined(self, src: str, out: str) -> None:
        assert norm(src) == out

    def test_quoted_bracket_over_indented_in_a_list_item(self) -> None:
        out = norm("- 항목\n\n      > \\[ a = b \\]")
        assert "  > $$a = b$$" in out and "      >" not in out

    @pytest.mark.parametrize(
        "src,attr",
        [
            ("앞\n\\[ a = b \\]\n{: .center}\n\n뒤", "$$a = b$$\n{: .center}"),
            ("> \\[ a = b \\]\n> {: .c}\n\n뒤", "> $$a = b$$\n> {: .c}"),
        ],
    )
    def test_attribute_line_stays_on_a_converted_bracket(
        self, src: str, attr: str
    ) -> None:
        assert attr in norm(src)

    def test_norm_on_a_pipe_less_table_row(self) -> None:
        src = "a | b\n--- | ---\nn | $$\\|x\\|$$"
        assert norm(src).endswith("n | $$\\Vert x\\Vert$$")

    @pytest.mark.parametrize(
        "src",
        ["앞 문단.\n\n    $$|x|$$ code\n\n뒤", "- 항목\n\n      $$|x|$$ in item\n"],
    )
    def test_indented_code_is_not_rewritten(self, src: str) -> None:
        assert norm(src) == src

    @pytest.mark.parametrize(
        "src,out",
        [
            (
                "> 앞 문장\n> $$a = b$$\n> 뒤 문장",
                "> 앞 문장\n>\n> $$a = b$$\n>\n> 뒤 문장",
            ),
            (
                "- 항목 설명\n  $$a = b$$\n  뒤 설명",
                "- 항목 설명\n\n  $$a = b$$\n\n  뒤 설명",
            ),
        ],
    )
    def test_glued_display_in_a_quote_or_list_item_is_isolated(
        self, src: str, out: str
    ) -> None:
        assert norm(src) == out

    def test_bars_in_spaced_text_and_mbox_are_kept(self) -> None:
        src = "$$\\text {a|b} + \\mbox{c|d}$$ 끝"
        assert norm(src) == src  # the validator reports them


def test_bracket_block_with_a_blank_line_becomes_one_display() -> None:
    out = norm("앞\n\n\\[\nx\\_1 = 1\n\ny = 2\n\\]\n\n뒤\n")
    assert out == "앞\n\n$$x_1 = 1\ny = 2$$\n\n뒤\n"


@pytest.mark.parametrize(
    "src,display",
    [
        ("1. 첫 단계\n$$\\log p$$\n2. 둘째\n", "   $$\\log p$$"),
        ("1. 첫 단계\n\\[ \\log p \\]\n2. 둘째\n", "   $$\\log p$$"),
        ("1. 첫 단계\n$$\na = b\n$$\n2. 둘째\n", "   $$\n   a = b\n   $$"),
        ("- 첫째\n$$a$$\n- 둘째\n", "  $$a$$"),
        ("> 1. 첫째\n> $$a$$\n> 2. 둘째\n", ">    $$a$$"),
    ],
)
def test_display_glued_under_a_list_item_stays_in_the_item(
    src: str, display: str
) -> None:
    # At the margin it would split the list (an ordered one restarting at 1).
    out = norm(src)
    assert display in out and norm(out) == out


def test_display_after_a_lists_last_item_stays_at_the_margin() -> None:
    out = norm("- 마지막 항목\n$$a$$\n\n다음 문단\n")
    assert "\n\n$$a$$\n\n" in out


@pytest.mark.parametrize(
    "src",
    [
        "- A100: $$2/h\n- H100: $$4/h\n",  # stray $$ in sibling items
        "| GPU | 가격 |\n| --- | --- |\n| A100 | $$2/h |\n| H100 | $$4/h |\n",
        "본문 $$a\n## 제목 b$$\n",
    ],
)
def test_span_kramdown_splits_into_blocks_is_not_joined(src: str) -> None:
    assert "\n" in norm(src).strip()
    assert norm(src).count("\n") >= src.count("\n")


def test_formula_wrapped_in_a_paragraph_is_still_joined() -> None:
    assert norm("값 $$a\n+ b\n+ c$$ 입니다.\n") == "값 $$a + b + c$$ 입니다.\n"


def test_line_with_two_inline_spans_is_not_split_from_its_quote() -> None:
    src = "> 보상은\n> $$\\beta$$에서 기준값 $$\\gamma$$\n> 를 뺀 값입니다.\n"
    assert norm(src) == src


def test_quoted_pipeless_table_is_left_alone() -> None:
    src = "> $$x$$ | 의미\n> --- | ---\n> $$a$$ | $$b$$\n"
    assert norm(src) == src


def test_escaped_parentheses_around_korean_prose_stay_text() -> None:
    out = norm("가격은 \\(약 3달러\\)입니다. 변수 \\(x\\)와 \\(\\alpha 값\\).")
    assert "\\(약 3달러\\)" in out and "$$x$$" in out and "$$\\alpha 값$$" in out


def test_escaped_plural_suffix_stays_text() -> None:
    out = norm("the model\\(s\\) run; let \\(s\\) and value\\(x+1\\)")
    assert "model\\(s\\)" in out and "$$s$$" in out and "value$$x+1$$" in out


@pytest.mark.parametrize(
    "src,expected",
    [
        (  # the explanation glued after the display goes into the item too
            "1. 첫째\n$$a = b$$\n여기서 a는 입력입니다.\n2. 둘째",
            "1. 첫째\n\n   $$a = b$$\n\n   여기서 a는 입력입니다.\n\n2. 둘째",
        ),
        (
            "1. 첫째\n$$\na = b\n$$\n여기서 a는 입력입니다.\n2. 둘째",
            "1. 첫째\n\n   $$\n   a = b\n   $$\n\n   여기서 a는 입력입니다.\n\n2. 둘째",
        ),
        (
            "1. 첫째\n\\[ a = b \\]\n여기서 a는 입력입니다.\n2. 둘째",
            "1. 첫째\n\n   $$a = b$$\n\n   여기서 a는 입력입니다.\n\n2. 둘째",
        ),
        (  # a blank line before the next item: the same list goes on
            "- 첫째\n$$a$$\n설명입니다.\n\n- 둘째",
            "- 첫째\n\n  $$a$$\n\n  설명입니다.\n\n- 둘째",
        ),
    ],
)
def test_display_and_its_explanation_stay_in_the_list_item(
    src: str, expected: str
) -> None:
    out = norm(src)
    assert out == expected and norm(out) == out


def test_display_and_paragraph_before_another_list_stay_at_the_margin() -> None:
    # The next list is not a sibling (other marker), so nothing is moved in.
    src = "1. 첫째\n$$a = b$$\n여기서 a는 입력입니다.\n\n- 다른 목록\n"
    assert "\n\n$$a = b$$\n\n여기서" in norm(src)


@pytest.mark.parametrize(
    "src,expected",
    [  # glued and indented less than the item's content: into the item
        ("1. 첫째\n  \\[ a = b \\]\n2. 둘째", "1. 첫째\n\n   $$a = b$$\n\n2. 둘째"),
        (
            "  - 하위\n  \\[ a = b \\]\n  - 마지막\n- 다음",
            "  - 하위\n\n    $$a = b$$\n\n  - 마지막\n- 다음",
        ),
        (
            "10. 열째\n   \\[ a = b \\]\n11. 다음",
            "10. 열째\n\n    $$a = b$$\n\n11. 다음",
        ),
    ],
)
def test_under_indented_glued_display_goes_to_the_items_column(
    src: str, expected: str
) -> None:
    out = norm(src)
    assert out == expected and norm(out) == out


@pytest.mark.parametrize(
    "src,expected",
    [
        (
            "- 항목\n  $$\n  a = b\n  $$\n- 다음",
            "- 항목\n\n  $$\n  a = b\n  $$\n\n- 다음",
        ),
        (
            "> **핵심:** 식\n> $$\n> a\n> $$\n> 설명",
            "> **핵심:** 식\n>\n> $$\n> a\n> $$\n>\n> 설명",
        ),
    ],
)
def test_glued_multiline_block_in_a_list_item_or_quote_is_isolated(
    src: str, expected: str
) -> None:
    out = norm(src)
    assert out == expected and norm(out) == out


@pytest.mark.parametrize(
    "src,expected",
    [
        (  # under-indented under "10.": display and explanation move together
            "10. 열째\n   $$x$$\n   여기서 x는 입력입니다.\n11. 다음",
            "10. 열째\n\n    $$x$$\n\n    여기서 x는 입력입니다.\n\n11. 다음",
        ),
        (  # every item numbered "1." still continues the list
            "1. 첫\n$$x$$\n여기서 x입니다.\n1. 둘째",
            "1. 첫\n\n   $$x$$\n\n   여기서 x입니다.\n\n1. 둘째",
        ),
        (  # two displays in one step
            "1. 첫\n$$x$$\n$$y$$\n2. 둘째",
            "1. 첫\n\n   $$x$$\n\n   $$y$$\n\n2. 둘째",
        ),
        (  # explanation then an indented fence
            "1. 첫\n$$x$$\n설명입니다.\n   ```python\n   y = f(x)\n   ```\n2. 둘째",
            "1. 첫\n\n   $$x$$\n\n   설명입니다.\n\n   ```python\n   y = f(x)\n   ```\n\n2. 둘째",
        ),
        (  # after a nested sub-list: back into the enclosing item
            "1. A\n   - 하위\n$$x$$\n설명입니다.\n2. B",
            "1. A\n   - 하위\n\n   $$x$$\n\n   설명입니다.\n\n2. B",
        ),
        (  # a display opening an item keeps its paragraph apart
            "1. $$x$$\n   여기서 x입니다.\n2. 둘째",
            "1. $$x$$\n\n   여기서 x입니다.\n2. 둘째",
        ),
    ],
)
def test_glued_run_stays_in_its_list_item(src: str, expected: str) -> None:
    assert norm(src) == expected
    assert norm(expected) == expected


def test_glued_display_after_the_last_item_is_left_at_the_margin() -> None:
    assert norm("1. 첫\n$$x$$\n\n끝 문단입니다.") == "1. 첫\n\n$$x$$\n\n끝 문단입니다."


def test_display_under_an_indented_heading_in_an_item_is_isolated() -> None:
    out = norm("1. 첫\n\n   #### 세부 식\n   $$x$$\n2. 둘째")
    assert "   #### 세부 식\n\n   $$x$$\n\n2. 둘째" in out


@pytest.mark.parametrize(
    "text",
    [
        # A "$$." closer (text after it) and a wrapped inline span: no clean
        # whole-line display, so nothing after it is pulled into the item.
        "1. 손실입니다.\n  $$\n  L = a + b\n  $$.\n2. 다음입니다.\n\n## B\n\n본문입니다.\n\n$$z = 1$$\n",
        "1. 목적함수입니다.\n  $$\\beta \\log a\n  + c$$로 씁니다.\n2. 다음\n\n## B\n\n본문\n\n$$z = 1$$\n",
        "- 손실입니다.\n$$\nL = a + b\n$$.\n- 다음입니다.\n\n## B\n\n$$z = 1$$\n\n- 마지막입니다.\n",
    ],
)
def test_fold_never_reaches_past_its_own_display(text: str) -> None:
    out = norm(text)
    lines = out.split("\n")
    assert "## B" in lines and "$$z = 1$$" in lines
    assert any(line.startswith(("2. ", "- 다음")) for line in lines)
    assert norm(out) == out


@pytest.mark.parametrize(
    "text",
    [
        "1. $$x = 1$$\n설명입니다.\n2. 다음\n",
        "1. $$\n   x = 1\n   $$\n설명입니다.\n2. 다음\n",
    ],
)
def test_paragraph_after_an_item_first_line_display_joins_the_item(text: str) -> None:
    out = norm(text)
    assert "\n   설명입니다." in out and "\n2. 다음" in out
    assert norm(out) == out


def test_fold_leaves_an_indented_code_sample_alone() -> None:
    text = "예시:\n\n    1. a\n    $$x$$\n    2. b\n\n끝.\n"
    assert norm(text) == text


def test_fold_keeps_rules_and_setext_lines_out_of_the_item() -> None:
    assert "\n***\n" in "\n" + norm("1. 첫\n$$x$$\n***\n2. 둘\n")
    assert "\n---\n" in norm("1. 첫\n$$x$$\n제목\n---\n2. 둘\n")


def test_fold_keeps_an_attribute_line_on_its_display() -> None:
    out = norm("1. 첫\n$$x$$\n{: .center}\n2. 둘\n")
    assert "   $$x$$\n   {: .center}" in out


def test_fold_stays_fast_on_long_items() -> None:
    import time

    # Linear, not quadratic: doubling the input must not quadruple the time.
    # The fastest of three runs filters out scheduler noise.
    def took(n: int, quoted: bool) -> float:
        lead, gap = (
            ("> 1. 항목\n", ">    $$x_{k}$$\n>\n")
            if quoted
            else ("1. 항목\n", "   $$x_{k}$$\n\n")
        )
        text = lead + "".join(gap.format(k=k) for k in range(n))
        runs = []
        for _ in range(3):
            started = time.monotonic()
            norm(text)
            runs.append(time.monotonic() - started)
        return min(runs)

    for quoted in (False, True):
        small, large = took(500, quoted), took(2000, quoted)
        assert large < 8 * max(small, 0.02)


def test_hangul_matches_the_blog_lint() -> None:
    from scholar_lens.src.markdown_math import HANGUL

    for char in ("한", "ㄱ", "㉠", "ᄒ", "ﾡ"):
        assert HANGUL.search(char)


def test_glued_display_with_a_line_starting_with_minus_stays_in_the_step() -> None:
    # An inner line of a display may start with "- " (a loss with two terms);
    # it is part of the display, not a new list item.
    text = (
        "1. 데이터를 준비합니다.\n2. 다음 손실을 최소화합니다.\n"
        "$$\n\\mathcal{L} = -\\log\\sigma(a\n- b)\n$$\n"
        "여기서 a, b는 로그 비율입니다.\n3. 평가합니다.\n"
    )
    from scholar_lens.src.markdown_math import normalize_math

    out = normalize_math(text)
    assert "   $$\n   \\mathcal{L} = -\\log\\sigma(a\n   - b)\n   $$" in out
    assert "   여기서 a, b는" in out and "\n3. 평가합니다." in out
    assert normalize_math(out) == out
