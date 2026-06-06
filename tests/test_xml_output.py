"""Tests for utils.parsers.RobustXMLOutputParser (pure logic, no AWS).

These exercise the cascading fallback strategies of ``RobustXMLOutputParser``
(standard XML -> lxml recovery -> sanitization -> truncation fix -> aggressive
cleaning -> regex/list fallbacks). Assertions reflect the parser's ACTUAL
behavior, observed by running it offline, not idealized behavior.
"""

from __future__ import annotations

import logging

import pytest

from scholar_lens.src.utils.parsers import RobustXMLOutputParser


@pytest.fixture(autouse=True)
def _silence_logging() -> None:
    """Keep debug/error fallback logging out of the test output."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


class TestRobustXMLOutputParserHappyPath:
    def test_clean_wellformed_xml(self) -> None:
        parser = RobustXMLOutputParser()
        result = parser.parse("<root><name>Alice</name><age>30</age></root>")
        # The langchain XMLOutputParser groups repeated/child elements under
        # the root tag as a list of single-key dicts.
        assert result == {"root": [{"name": "Alice"}, {"age": "30"}]}

    def test_nested_repeated_children(self) -> None:
        parser = RobustXMLOutputParser()
        result = parser.parse(
            "<root><items><item>a</item><item>b</item></items></root>"
        )
        assert result == {"root": [{"items": [{"item": "a"}, {"item": "b"}]}]}


class TestRobustXMLOutputParserSanitization:
    def test_unescaped_ampersand_recovers_via_lxml(self) -> None:
        # Standard XML parsing chokes on the bare ``&``; lxml recover mode
        # salvages the structure but DROPS the offending ``&`` character.
        parser = RobustXMLOutputParser()
        result = parser.parse("<note>Tom & Jerry & Co</note>")
        # POTENTIAL SOURCE BUG: lxml recover silently strips bare ampersands
        # rather than escaping them, so the text loses the ``&`` characters.
        # See scholar_lens/src/utils/parsers.py:255 (_try_lxml_recover_parse).
        assert result == {"note": "Tom  Jerry  Co"}

    def test_stray_angle_bracket_recovers(self) -> None:
        # A stray ``<`` inside text content is not valid XML; lxml recovery
        # salvages the element but loses the malformed fragment.
        parser = RobustXMLOutputParser()
        result = parser.parse("<root><expr>a < b and c > d</expr></root>")
        assert result == {"root": {"expr": "a  b and c > d"}}

    def test_multiple_sections_with_ampersand(self) -> None:
        # Two top-level sections; section preservation forces a fallback that
        # wraps leaf text under the ``#text`` key.
        parser = RobustXMLOutputParser()
        result = parser.parse("<summary>R&D and Q&A</summary><title>Hello</title>")
        assert result == {
            "summary": {"#text": "R&D and Q&A"},
            "title": {"#text": "Hello"},
        }


class TestRobustXMLOutputParserTruncation:
    def test_truncated_missing_closing_tag(self) -> None:
        # Missing closing tags on the trailing element. The fallbacks extract
        # only the fully-closed leading section.
        parser = RobustXMLOutputParser()
        result = parser.parse("<root><name>Alice</name><city>NYC")
        assert result == {"name": {"#text": "Alice"}}

    def test_truncated_repeated_children(self) -> None:
        parser = RobustXMLOutputParser()
        result = parser.parse("<analysis><point>p1</point><point>p2")
        # Only the first closed <point> survives extraction.
        assert result == {"point": {"#text": "p1"}}


class TestRobustXMLOutputParserListFallback:
    def test_bullet_list_fallback(self) -> None:
        parser = RobustXMLOutputParser()
        result = parser.parse("* one\n* two\n* three")
        assert result == {"items": ["one", "two", "three"]}

    def test_numbered_list_fallback(self) -> None:
        parser = RobustXMLOutputParser()
        result = parser.parse("1. first\n2. second")
        assert result == {"items": ["first", "second"]}


class TestRobustXMLOutputParserExhausted:
    def test_empty_string_raises(self) -> None:
        parser = RobustXMLOutputParser()
        with pytest.raises(ValueError, match="Failed to parse XML"):
            parser.parse("")

    def test_plain_text_no_tags_no_list_raises(self) -> None:
        # No tags and no list markers -> every strategy fails.
        parser = RobustXMLOutputParser()
        with pytest.raises(ValueError, match="Failed to parse XML"):
            parser.parse("just some plain text no tags at all")


class TestRobustXMLOutputParserStaticHelpers:
    def test_detect_xml_sections(self) -> None:
        sections = RobustXMLOutputParser._detect_xml_sections("<a>1</a><b>2</b>")
        assert sections == {"a", "b"}

    def test_sections_preserved_when_empty_original(self) -> None:
        assert RobustXMLOutputParser._sections_preserved(set(), {}) is True

    def test_sections_preserved_detects_missing(self) -> None:
        assert RobustXMLOutputParser._sections_preserved({"a", "b"}, {"a": 1}) is False

    def test_unescape_xml_tags_restores_brackets(self) -> None:
        out = RobustXMLOutputParser._unescape_xml_tags("&lt;tag&gt;value&lt;/tag&gt;")
        assert out == "<tag>value</tag>"

    def test_extract_list_fallback_returns_none_without_markers(self) -> None:
        assert RobustXMLOutputParser._extract_list_fallback("no markers") is None
