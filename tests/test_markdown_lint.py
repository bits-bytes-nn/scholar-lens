"""Tests for the blog Markdown lint pass (auto-fixes + warnings)."""

from __future__ import annotations

import logging

import pytest

from scholar_lens.src.markdown_lint import (
    drop_unknown_images,
    lint_markdown,
    undefined_macros,
    upgrade_http_links,
)
from scholar_lens.src.post_conventions import load_local_conventions


@pytest.fixture
def _capture_app_logs():
    """The app logger has propagate=False (so it won't reach caplog's root
    handler); enable propagation for the duration of a test so caplog sees the
    lint warnings."""
    app_logger = logging.getLogger("app")
    previous = app_logger.propagate
    app_logger.propagate = True
    try:
        yield
    finally:
        app_logger.propagate = previous


class TestHeadingBlankLine:
    def test_inserts_blank_line_before_glued_heading(self) -> None:
        out = lint_markdown("some text\n## Section\nbody")
        assert "some text\n\n## Section" in out

    def test_leaves_already_spaced_heading(self) -> None:
        src = "some text\n\n## Section\nbody"
        assert lint_markdown(src) == src

    def test_does_not_touch_hash_inside_code_fence(self) -> None:
        src = "```python\nx = 1\n# a comment, not a heading\n```"
        assert lint_markdown(src) == src

    def test_first_line_heading_untouched(self) -> None:
        # No preceding line to glue to.
        src = "## Title\n\nbody"
        assert lint_markdown(src) == src


class TestPlaceholderTagEscaping:
    def test_bare_placeholder_escaped(self) -> None:
        out = lint_markdown("namespace 값이 <none>으로 표시됩니다.")
        assert "&lt;none&gt;" in out
        assert "<none>" not in out

    def test_multiple_placeholders_escaped(self) -> None:
        out = lint_markdown("서버는 <server-host>:<port> 형태.")
        assert "&lt;server-host&gt;" in out and "&lt;port&gt;" in out

    def test_placeholder_with_space_escaped(self) -> None:
        out = lint_markdown("kubectl --namespace <your namespace> 실행")
        assert "&lt;your namespace&gt;" in out

    def test_real_html_tags_untouched(self) -> None:
        src = "줄바꿈 <br> 과 위첨자 <sup>2</sup> 와 <kbd>Ctrl</kbd>."
        assert lint_markdown(src) == src

    def test_placeholder_in_inline_code_untouched(self) -> None:
        src = "값이 `<none>` 으로 표시됩니다."
        assert lint_markdown(src) == src

    def test_placeholder_in_code_fence_untouched(self) -> None:
        src = "```\nserver: <server-host>\n```"
        assert lint_markdown(src) == src

    def test_less_than_in_inline_math_untouched(self) -> None:
        src = r"부등식 \( a < b \) 는 그대로."
        assert lint_markdown(src) == src

    def test_markdown_escaped_placeholder_untouched(self) -> None:
        src = r"출력은 \<ANSWER\> 형식입니다."
        assert lint_markdown(src) == src

    def test_adjacent_inline_code_spans_are_protected(self) -> None:
        src = "태그 `<think>``</think>` 사이"
        assert lint_markdown(src) == src

    def test_url_autolink_not_escaped(self) -> None:
        # A GFM autolink is valid Markdown that renders as a link; escaping it
        # produces dead literal text. Must survive the placeholder escaper.
        src = "See <https://example.com/docs> for details."
        assert lint_markdown(src) == src

    def test_email_autolink_not_escaped(self) -> None:
        src = "Contact <user@example.com> please."
        assert lint_markdown(src) == src

    def test_placeholder_still_escaped_alongside_autolink(self) -> None:
        # The autolink survives; a real placeholder on the same line is escaped.
        out = lint_markdown("Use <your-namespace> at <https://ex.com>.")
        assert "&lt;your-namespace&gt;" in out
        assert "<https://ex.com>" in out


@pytest.mark.usefixtures("_capture_app_logs")
class TestPipeInMath:
    def test_bare_pipe_in_math_warns_not_rewritten(self, caplog) -> None:
        # NOT auto-rewritten (\vert would break \left|, array{c|c}); warn instead.
        with caplog.at_level(logging.WARNING, logger="app"):
            out = lint_markdown(r"prob $$P(a|b)$$ here")
        assert out == r"prob $$P(a|b)$$ here"  # text unchanged
        assert any("bare '|'" in r.message for r in caplog.records)

    def test_left_right_delimiters_not_broken(self, caplog) -> None:
        # The whole point: a \vert rewrite would corrupt these — confirm we don't.
        src = r"$$\left| x \right| + \begin{array}{c|c} a & b \end{array}$$"
        with caplog.at_level(logging.WARNING, logger="app"):
            assert lint_markdown(src) == src

    def test_no_warning_when_no_bare_pipe(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="app"):
            lint_markdown(r"clean \(P(a \mid b)\) here")
        assert not any("bare '|'" in r.message for r in caplog.records)


@pytest.mark.usefixtures("_capture_app_logs")
class TestWarnings:
    def test_single_dollar_math_warns(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="app"):
            lint_markdown(r"inline \(x_1\) ok but $y_2$ bad")
        assert any("single-$" in r.message for r in caplog.records)

    def test_currency_in_code_does_not_warn(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="app"):
            lint_markdown(r"the price `$5` is fine")
        assert not any("single-$" in r.message for r in caplog.records)

    def test_fragile_math_env_and_bm_warn(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="app"):
            lint_markdown(r"$$\begin{align} a &= b \end{align}$$ and \(\bm{x}\)")
        msgs = " ".join(r.message for r in caplog.records)
        assert "render poorly" in msgs

    def test_aligned_env_does_not_warn(self, caplog) -> None:
        # The recommended \begin{aligned} form must NOT trigger the fragile warning.
        with caplog.at_level(logging.WARNING, logger="app"):
            lint_markdown(r"$$\begin{aligned} a &= b \end{aligned}$$")
        assert not any("render poorly" in r.message for r in caplog.records)

    def test_standard_macros_do_not_warn(self, caplog) -> None:
        # Base-MathJax macros (and unknown-but-not-flagged ones) must not warn —
        # we only flag the explicit known-undefined set, never allowlist base.
        with caplog.at_level(logging.WARNING, logger="app"):
            lint_markdown(r"math \(\alpha + \frac{a}{b} + \mathbb{R}\)")
        assert not any("render as red raw text" in r.message for r in caplog.records)

    def test_known_undefined_macro_warns(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="app"):
            lint_markdown(r"math $$a \coloneqq b$$")
        assert any("coloneqq" in r.message for r in caplog.records)

    def test_blog_defined_macros_do_not_warn(self, caplog) -> None:
        # \llbracket, \rrbracket and \textsc are defined in the blog's head.html.
        with caplog.at_level(logging.WARNING, logger="app"):
            lint_markdown(r"math $$\llbracket x \rrbracket \textsc{a}$$")
        assert not any("does not define" in r.message for r in caplog.records)

    def test_defined_macros_come_from_the_conventions(self) -> None:
        # post_conventions.yml math.defined_macros (the blog's head.html list).
        defined = load_local_conventions().math.defined_macros
        assert {"llbracket", "rrbracket", "textsc"} <= set(defined)
        assert not undefined_macros() & set(defined)
        assert "coloneqq" in undefined_macros()

    def test_non_url_link_target_warns(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="app"):
            lint_markdown("see [appendix](부록 비교 참조) for details")
        assert any("not an http(s) URL" in r.message for r in caplog.records)

    def test_real_and_relative_links_do_not_warn(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="app"):
            lint_markdown("[p](https://arxiv.org/abs/1) and [q](/posts/x) and [a](#s)")
        assert not any("not an http(s) URL" in r.message for r in caplog.records)


def test_lint_is_idempotent() -> None:
    src = "text\n## H\nbody with \\(P(a|b)\\)"
    once = lint_markdown(src)
    assert lint_markdown(once) == once


@pytest.mark.usefixtures("_capture_app_logs")
class TestBodyAutoFixes:
    def test_envelope_tags_removed_outside_code(self) -> None:
        out = lint_markdown("body\n</explanation>\n`<summary>` stays")
        assert "</explanation>" not in out
        assert "`<summary>`" in out

    def test_h1_becomes_h2_and_other_levels_stay(self) -> None:
        # Shifting every level would turn "## TL;DR" into "###" and fail the
        # blog's "starts with ## TL;DR" rule.
        out = lint_markdown("## TL;DR\n\n# Title\n\n## Part\n\ntext")
        assert out.startswith("## TL;DR") and "\n## Title" in out
        assert "\n## Part" in out and "###" not in out

    def test_no_h1_leaves_headings(self) -> None:
        src = "## A\n\n### B"
        assert lint_markdown(src) == src

    def test_hash_comment_in_code_is_not_a_heading(self) -> None:
        src = "## A\n\n```python\n# comment\n```"
        assert lint_markdown(src) == src


class TestDropUnknownImages:
    def test_keeps_known_and_drops_invented(self) -> None:
        md = (
            "![a](https://arxiv.org/html/2106.09685) ![b](fig_1.png) ![c](/x/fig_2.png)"
        )
        out = drop_unknown_images(md, {"/tmp/p/figures/fig_1.png", "fig_2.png"})
        assert "arxiv.org/html" not in out
        assert "![b](fig_1.png)" in out and "![c](/x/fig_2.png)" in out

    def test_images_in_code_untouched(self) -> None:
        md = "`![x](nowhere.png)`"
        assert drop_unknown_images(md, set()) == md

    def test_arxiv_figures_need_the_same_paper_in_any_version(self) -> None:
        known = {"https://arxiv.org/html/2106.09685v2/x1.png"}
        md = (
            "![a](https://arxiv.org/html/2106.09685v1/x1.png) "
            "![b](https://arxiv.org/html/2999.99999v1/x1.png)"
        )
        out = drop_unknown_images(md, known)
        # The same figure in another version points at the fetched source.
        assert out.strip() == "![a](https://arxiv.org/html/2106.09685v2/x1.png)"


class TestFillImageAlts:
    def test_caption_without_label_becomes_alt(self) -> None:
        from scholar_lens.src.markdown_lint import fill_image_alts

        out = fill_image_alts(
            "![](https://arxiv.org/html/x/x1.png)\n![kept](/a.png)\n![](/u.png)",
            {"x1.png": "Figure 6: Subspace similarity between A_r=8 and A_r=64."},
            "LoRA",
        )
        assert (
            "![Subspace similarity between A_r=8 and A_r=64.](https://arxiv.org/html/x/x1.png)"
            in out
        )
        assert "![kept](/a.png)" in out
        assert "![LoRA](/u.png)" in out


class TestFigureMarkersToImages:
    def test_marker_becomes_image_with_caption_alt(self) -> None:
        from scholar_lens.src.markdown_lint import figure_markers_to_images

        text = (
            '앞 문장.\n[Image: alt="Refer to caption", src="https://arxiv.org/html/x/a.png", '
            'caption="Subspace similarity between two ranks. More detail."]\n뒤 문장.'
        )
        out = figure_markers_to_images(text)
        assert "[Image:" not in out
        assert (
            "![Subspace similarity between two ranks.](https://arxiv.org/html/x/a.png)"
            in out
        )

    def test_markers_inside_code_are_kept(self) -> None:
        from scholar_lens.src.markdown_lint import figure_markers_to_images

        text = '`[Image: alt="a", src="b.png"]`'
        assert figure_markers_to_images(text) == text


def test_marker_in_table_row_stays_on_the_row() -> None:
    from scholar_lens.src.markdown_lint import figure_markers_to_images

    row = '| [Image: alt="cat", src="https://x/a.png", caption="c"] | A cat |'
    assert figure_markers_to_images(row) == "| ![cat](https://x/a.png) | A cat |"


def test_heading_level_jumps_are_closed() -> None:
    src = "## TL;DR\n\n#### Deep\n\n##### Deeper\n\n## Next\n\n```\n#### code\n```"
    out = lint_markdown(src)
    assert "\n### Deep\n" in out and "\n#### Deeper\n" in out
    assert "#### code" in out and out.startswith("## TL;DR")


def test_heading_moves_are_silent_with_warn_off(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        out = lint_markdown("## TL;DR\n\n#### Deep", warn=False)
    assert "\n### Deep" in out
    assert "moved to level" not in caplog.text


def test_a_same_named_image_on_another_host_is_dropped() -> None:
    known = {
        "https://docs.x.io/img/arch.png",
        "https://cdn.x.io/a.png?v=2",
        "/tmp/p/figures/fig_1.png",
        "https://arxiv.org/html/2106.09685v2/x1.png",
    }
    md = (
        "![a](https://evil.example/arch.png) ![b](https://docs.x.io/img/arch.png) "
        "![c](https://cdn.x.io/a.png) ![d](figures/fig_1.png) "
        "![e](https://arxiv.org/html/2106.09685/x1.png)"
    )
    out = drop_unknown_images(md, known)
    assert "evil.example" not in out
    for kept in ("docs.x.io/img/arch.png", "cdn.x.io/a.png", "fig_1.png", "x1.png"):
        assert kept in out


@pytest.mark.parametrize(
    "src",
    [
        "다음을 실행합니다.\n\n    vllm serve <model-name> --port 8000\n\n끝.",
        "> ```bash\n> vllm serve <model>\n> ```",
    ],
)
def test_placeholders_in_indented_or_quoted_code_are_not_escaped(src: str) -> None:
    assert lint_markdown(src) == src


@pytest.mark.parametrize(
    "src,kept",
    [  # an open "<" in prose never pairs with a ">" on a later line or in code
        ("조건 k<n 일 때 성립합니다.\n\n> **참고:** 인용입니다.", "> **참고:**"),
        ("조건 k<n 일 때 성립합니다.\n\n코드 `x > y` 를 보세요.", "`x > y`"),
        ("타입은 List<T 처럼 씁니다.\n\n- 항목 `a -> b`", "`a -> b`"),
    ],
)
def test_open_angle_bracket_does_not_escape_later_lines(src: str, kept: str) -> None:
    assert kept in lint_markdown(src, warn=False)


def test_publish_body_is_idempotent_with_an_unboxed_callout() -> None:
    from scholar_lens.src.markdown_lint import MAX_CALLOUTS, publish_body

    callouts = "\n\n".join(f"> **핵심:** 요점 {i}." for i in range(MAX_CALLOUTS))
    src = callouts + "\n\n> **핵심:** 아래 식\n> $$\n> a = b\n> $$\n> 설명\n"
    once = publish_body(src, warn=False)
    assert publish_body(once, warn=False) == once


def test_bos_eos_tokens_are_escaped_not_strikethrough() -> None:
    from scholar_lens.src.markdown_lint import lint_markdown

    out = lint_markdown("BOS 토큰 <s>를 문장 앞에, EOS 토큰 </s>를 끝에 붙입니다.")
    assert "&lt;s&gt;" in out and "&lt;/s&gt;" in out


def test_chat_template_tokens_become_inline_code() -> None:
    out = lint_markdown("모델은 <|im_start|> 토큰을 씁니다.", warn=False)
    assert out == "모델은 `<|im_start|>` 토큰을 씁니다."
    assert lint_markdown(out, warn=False) == out
    kept = "$$\\text{<|x|>}$$ 와 `<|eot_id|>`"
    assert lint_markdown(kept, warn=False) == kept


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "| a | b |\n|---|---|\n| 1 | 2 |\n표 아래입니다.",
            "| 1 | 2 |\n\n표 아래입니다.",
        ),
        ("표 위입니다.\n| a | b |\n|---|---|\n| 1 | 2 |", "표 위입니다.\n\n| a | b |"),
    ],
)
def test_tables_glued_to_text_get_blank_lines(text: str, expected: str) -> None:
    out = lint_markdown(text, warn=False)
    assert expected in out and lint_markdown(out, warn=False) == out


def test_summary_tags_are_removed_even_inside_details() -> None:
    # kramdown renders a bare <summary> inside <details> as a broken page, and
    # no prompt asks for <details>: the tag is stripped like any leaked one.
    out = lint_markdown(
        "<details>\n<summary>펼치기</summary>\n\n내용\n</details>", warn=False
    )
    assert "<summary>" not in out and "</summary>" not in out
    assert "</summary>" not in lint_markdown("본문 </summary>", warn=False)


def test_table_glued_under_a_list_item_stays_in_the_item() -> None:
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    out = lint_markdown(
        f"1. 첫 단계입니다.\n{table}\n2. 둘째 단계입니다.\n", warn=False
    )
    assert "\n   | a | b |\n   | --- | --- |\n   | 1 | 2 |\n" in out
    assert "\n2. 둘째 단계입니다." in out and lint_markdown(out, warn=False) == out


def test_table_attribute_line_stays_attached() -> None:
    out = lint_markdown(
        "앞 문장.\n| a | b |\n| --- | --- |\n| 1 | 2 |\n{: .wide}\n", warn=False
    )
    assert "| 1 | 2 |\n{: .wide}" in out


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "[a](http://a.b/x) and ![b](HTTP://a.b/y.png)",
            "[a](https://a.b/x) and ![b](https://a.b/y.png)",
        ),
        ("<http://a.b> and [c](//a.b/c)", "<https://a.b> and [c](https://a.b/c)"),
        (
            '<a href="http://a.b">x</a> <img src="//a.b/i.png" srcset="http://a.b/1.png 1x, http://a.b/2.png 2x">',
            '<a href="https://a.b">x</a> <img src="https://a.b/i.png" srcset="https://a.b/1.png 1x, https://a.b/2.png 2x">',
        ),
        ("[ref]: http://a.b/r", "[ref]: https://a.b/r"),
        # Code, a footnote and a bare URL (no link to kramdown) are left alone.
        (
            "`curl http://localhost:8000` and [x](http://a.b)",
            "`curl http://localhost:8000` and [x](https://a.b)",
        ),
        (
            "```bash\ncurl http://localhost:8000\n```",
            "```bash\ncurl http://localhost:8000\n```",
        ),
        (
            "text.\n\n    curl http://localhost:8000",
            "text.\n\n    curl http://localhost:8000",
        ),
        ("[^1]: http://a.b and http://c.d", "[^1]: http://a.b and http://c.d"),
        # A link showing its own URL shows the upgraded one.
        ("[http://a.b/x](http://a.b/x)", "[https://a.b/x](https://a.b/x)"),
    ],
)
def test_http_links_are_upgraded(src: str, expected: str) -> None:
    assert upgrade_http_links(src) == expected
    assert upgrade_http_links(expected) == expected
