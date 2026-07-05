"""Tests for the blog Markdown lint pass (auto-fixes + warnings)."""

from __future__ import annotations

import logging

import pytest

from scholar_lens.src.markdown_lint import lint_markdown


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
        src = "# Title\n\nbody"
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
            out = lint_markdown(r"prob \(P(a|b)\) here")
        assert out == r"prob \(P(a|b)\) here"  # text unchanged
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
            lint_markdown(r"math \(\llbracket x \rrbracket\)")
        assert any("llbracket" in r.message for r in caplog.records)

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
