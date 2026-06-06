"""Tests for utils.helpers and utils.parsers (pure logic, no AWS)."""

from __future__ import annotations

import asyncio

import pytest

from scholar_lens.src.utils.helpers import (
    arg_as_bool,
    extract_text_from_html,
    measure_execution_time,
)
from scholar_lens.src.utils.parsers import HTMLTagOutputParser


class TestArgAsBool:
    @pytest.mark.parametrize("value", ["yes", "true", "1", "y", "t", "TRUE", "Yes"])
    def test_truthy(self, value: str) -> None:
        assert arg_as_bool(value) is True

    @pytest.mark.parametrize("value", ["no", "false", "0", "n", "f", "FALSE"])
    def test_falsy(self, value: str) -> None:
        assert arg_as_bool(value) is False

    def test_passthrough_bool(self) -> None:
        assert arg_as_bool(True) is True
        assert arg_as_bool(False) is False

    def test_invalid_raises(self) -> None:
        with pytest.raises(Exception):
            arg_as_bool("maybe")


class TestExtractTextFromHtml:
    def test_strips_tags_and_scripts(self) -> None:
        html = "<p>Hi <b>there</b></p><script>evil()</script>"
        assert extract_text_from_html(html) == "Hi there"

    def test_empty_input(self) -> None:
        assert extract_text_from_html("") == ""

    def test_collapses_whitespace(self) -> None:
        assert extract_text_from_html("<p>a\n\n   b</p>") == "a b"


class TestHTMLTagOutputParser:
    def test_multiple_tags_returns_dict(self) -> None:
        parser = HTMLTagOutputParser(tag_names=["title", "body"])
        assert parser.parse("<title>Hello</title><body>World</body>") == {
            "title": "Hello",
            "body": "World",
        }

    def test_single_tag_returns_string(self) -> None:
        parser = HTMLTagOutputParser(tag_names="answer")
        assert parser.parse("noise <answer>42</answer> noise") == "42"

    def test_missing_tag_is_omitted(self) -> None:
        # The parser only emits keys for tags actually present in the output.
        parser = HTMLTagOutputParser(tag_names=["present", "absent"])
        result = parser.parse("<present>x</present>")
        assert result["present"] == "x"
        assert "absent" not in result


class TestMeasureExecutionTime:
    def test_decorates_sync_function(self) -> None:
        @measure_execution_time
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_decorates_async_function(self) -> None:
        @measure_execution_time
        async def aadd(a: int, b: int) -> int:
            return a + b

        assert asyncio.run(aadd(2, 3)) == 5
