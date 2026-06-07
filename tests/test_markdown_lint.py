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


class TestPipeInMath:
    def test_bare_pipe_in_inline_math_becomes_vert(self) -> None:
        out = lint_markdown(r"prob \(P(a|b)\) here")
        assert r"\(P(a\vert b)\)" in out

    def test_bare_pipe_in_display_math_becomes_vert(self) -> None:
        out = lint_markdown(r"$$ P(a|b) = 1 $$")
        assert r"\vert" in out and "|" not in out.replace(r"\vert", "")

    def test_table_pipes_untouched(self) -> None:
        src = "before\n\n| a | b |\n|---|---|\n| 1 | 2 |"
        assert lint_markdown(src) == src

    def test_pipe_in_code_untouched(self) -> None:
        src = "run `cat x | grep y` now"
        assert lint_markdown(src) == src


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

    def test_known_blog_macros_do_not_warn(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="app"):
            lint_markdown(r"math \(\llbracket x \rrbracket + \alpha\)")
        assert not any("macro" in r.message for r in caplog.records)

    def test_undefined_macro_warns(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="app"):
            lint_markdown(r"math \(\foobarbaz x\)")
        assert any("foobarbaz" in r.message for r in caplog.records)

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
