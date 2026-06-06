"""Tests for pure / LLM-free logic in scholar_lens.src.explainer.

The real ``ExplainerGraph.__init__`` downloads NLTK data and constructs Bedrock
chains, so every test bypasses it via ``ExplainerGraph.__new__`` and sets only
the attributes the method under test reads. No AWS or network access.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from scholar_lens.src.explainer import ExplainerGraph
from scholar_lens.src.metrics import TokenBudgetExceeded, TokenUsageTracker


def _bare_graph() -> ExplainerGraph:
    """An ExplainerGraph with no chains/NLTK initialised."""
    g = ExplainerGraph.__new__(ExplainerGraph)
    # Nodes call _enforce_token_budget(); production __init__ sets these, so a
    # bare instance must too (else AttributeError, which the node retry wrapper
    # would re-raise after a long backoff).
    g._token_tracker = None
    g.max_total_tokens = None
    return g


def _make_structure(*start_numbers: int) -> dict[str, Any]:
    """Build a minimal ``paper_structure`` with the given starting indices."""
    return {
        "paper_structure": [
            {
                "section": [
                    {
                        "section_title": f"Section {i}",
                        "starting_sentence_number": str(n),
                        "key_points": [],
                    }
                ]
            }
            for i, n in enumerate(start_numbers)
        ]
    }


class TestExtractParagraphsByIndices:
    def test_valid_indices_split_with_zero_first(self) -> None:
        sentences = [f"s{i}" for i in range(6)]
        structure = _make_structure(0, 3)
        paragraphs, prepended = ExplainerGraph._extract_paragraphs_by_indices(
            sentences, structure
        )
        assert prepended is False
        assert paragraphs == ["s0 s1 s2", "s3 s4 s5"]

    def test_non_zero_first_index_prepends_zero(self) -> None:
        sentences = [f"s{i}" for i in range(5)]
        structure = _make_structure(2)
        paragraphs, prepended = ExplainerGraph._extract_paragraphs_by_indices(
            sentences, structure
        )
        assert prepended is True
        assert paragraphs == ["s0 s1", "s2 s3 s4"]

    def test_out_of_bounds_indices_skipped(self) -> None:
        sentences = [f"s{i}" for i in range(3)]
        # 99 is out of range and dropped; 1 remains -> 0 prepended.
        structure = _make_structure(1, 99)
        paragraphs, prepended = ExplainerGraph._extract_paragraphs_by_indices(
            sentences, structure
        )
        assert prepended is True
        assert paragraphs == ["s0", "s1 s2"]

    def test_no_paper_structure_single_paragraph(self) -> None:
        sentences = ["a", "b", "c"]
        paragraphs, prepended = ExplainerGraph._extract_paragraphs_by_indices(
            sentences, {}
        )
        assert prepended is False
        assert paragraphs == ["a b c"]

    def test_all_indices_invalid_single_paragraph(self) -> None:
        sentences = ["a", "b"]
        structure = _make_structure(50)
        paragraphs, prepended = ExplainerGraph._extract_paragraphs_by_indices(
            sentences, structure
        )
        assert prepended is False
        assert paragraphs == ["a b"]


class TestGetSectionAnalysis:
    def _state(self, offset: int = 0) -> dict[str, Any]:
        return {
            "structure": _make_structure(0, 5, 10),
            "structure_index_offset": offset,
        }

    def test_in_range_returns_section(self) -> None:
        state = self._state()
        section = ExplainerGraph._get_section_analysis(state, 1)
        assert section["section"][0]["section_title"] == "Section 1"

    def test_offset_applied(self) -> None:
        state = self._state(offset=1)
        # current_index 2 -> structure_index 1
        section = ExplainerGraph._get_section_analysis(state, 2)
        assert section["section"][0]["section_title"] == "Section 1"

    def test_out_of_range_returns_default(self) -> None:
        state = self._state()
        section = ExplainerGraph._get_section_analysis(state, 99)
        assert section == {
            "section": [{"section_title": "Introduction", "key_points": []}]
        }

    def test_negative_structure_index_returns_default(self) -> None:
        state = self._state(offset=5)
        section = ExplainerGraph._get_section_analysis(state, 0)
        assert section["section"][0]["section_title"] == "Introduction"


class TestParseReferenceIdentifiers:
    def test_splits_strips_and_drops_blanks(self) -> None:
        text = "  ref1 \n\n ref2\n   \nref3  "
        assert ExplainerGraph._parse_reference_identifiers(text) == [
            "ref1",
            "ref2",
            "ref3",
        ]

    def test_empty_returns_empty_list(self) -> None:
        assert ExplainerGraph._parse_reference_identifiers("") == []


class TestFinalizePaper:
    def test_returns_key_takeaways_and_invokes_with_language(self) -> None:
        g = _bare_graph()
        g.language = "Korean"
        g.finalizer = MagicMock()
        g.finalizer.invoke.return_value = {"key_takeaways": "TK"}
        state = {"explanations": ["e1", "e2"], "key_takeaways": ""}

        result = g.finalize_paper(state)

        assert result == {"key_takeaways": "TK"}
        g.finalizer.invoke.assert_called_once()
        payload = g.finalizer.invoke.call_args.args[0]
        assert "explanation" in payload
        assert payload["language"] == "Korean"


class TestReflectPaper:
    def _state(self, explanation: str = "an explanation") -> dict[str, Any]:
        paper = MagicMock()
        paper.table_of_contents = {}
        return {
            "current_index": 0,
            "explanations": [explanation],
            "paragraphs": ["a paragraph"],
            "structure": _make_structure(0),
            "structure_index_offset": 0,
            "accumulated_feedback": [],
            "synthesis_attempts": 1,
            "citation_summaries": None,
            "code": None,
            "paper": paper,
        }

    def test_quality_score_parsed_and_attempts_incremented(self) -> None:
        g = _bare_graph()
        g.language = "Korean"
        g.translation_guideline = []
        g.reflector = MagicMock()
        g.reflector.invoke.return_value = {
            "quality_score": "85",
            "improvement_feedback": "fix",
        }
        result = g.reflect_paper(self._state())

        assert result["quality_score"] == 85
        assert result["synthesis_attempts"] == 2
        assert "fix" in result["accumulated_feedback"]

    def test_clean_integer_quality_score(self) -> None:
        g = _bare_graph()
        g.language = "Korean"
        g.translation_guideline = []
        g.reflector = MagicMock()
        g.reflector.invoke.return_value = {"quality_score": "100"}
        result = g.reflect_paper(self._state())
        assert result["quality_score"] == 100

    def test_non_numeric_quality_score_does_not_crash(self) -> None:
        # The LLM may return "85/100", "N/A", or "" — must not raise, fall back.
        g = _bare_graph()
        g.language = "Korean"
        g.translation_guideline = []
        g.reflector = MagicMock()
        for raw, expected in [("85/100", 85), ("N/A", 0), ("", 0), ("score: 7", 7)]:
            g.reflector.invoke.return_value = {"quality_score": raw}
            result = g.reflect_paper(self._state())
            assert result["quality_score"] == expected, raw

    def test_empty_explanation_returns_zero_and_length_feedback(self) -> None:
        g = _bare_graph()
        g.language = "Korean"
        g.translation_guideline = []
        g.reflector = MagicMock()  # must not be called
        result = g.reflect_paper(self._state(explanation="   "))

        assert result["quality_score"] == 0
        assert result["synthesis_attempts"] == 2
        assert any(
            "reduce the output length" in f for f in result["accumulated_feedback"]
        )
        g.reflector.invoke.assert_not_called()


class TestSynthesizePaper:
    def _state(self) -> dict[str, Any]:
        paper = MagicMock()
        paper.table_of_contents = {}
        return {
            "current_index": 0,
            "paragraphs": ["a paragraph"],
            "explanations": [],
            "structure": _make_structure(0),
            "structure_index_offset": 0,
            "accumulated_feedback": [],
            "synthesis_attempts": 0,
            "citation_summaries": None,
            "code": None,
            "paper": paper,
        }

    def test_has_more_loop_capped_at_max_continuations(self) -> None:
        g = _bare_graph()
        g.max_continuations = 3
        g._token_tracker = None
        g.max_total_tokens = None
        g._synthesize_paper = MagicMock(
            return_value={"explanation": "chunk", "has_more": "y"}
        )
        g.synthesize_paper(self._state())
        assert g._synthesize_paper.call_count == 3

    def test_has_more_no_stops_after_one(self) -> None:
        g = _bare_graph()
        g.max_continuations = 8
        g._token_tracker = None
        g.max_total_tokens = None
        g._synthesize_paper = MagicMock(
            return_value={"explanation": "chunk", "has_more": "n"}
        )
        out = g.synthesize_paper(self._state())
        assert g._synthesize_paper.call_count == 1
        assert out["explanations"][0] == "chunk"


class TestEnforceTokenBudget:
    def test_over_budget_raises(self) -> None:
        g = _bare_graph()
        g._token_tracker = TokenUsageTracker(input_tokens=900, output_tokens=200)
        g.max_total_tokens = 1000
        with pytest.raises(TokenBudgetExceeded):
            g._enforce_token_budget()

    def test_under_budget_no_raise(self) -> None:
        g = _bare_graph()
        g._token_tracker = TokenUsageTracker(input_tokens=100, output_tokens=100)
        g.max_total_tokens = 1000
        g._enforce_token_budget()  # no raise

    def test_no_tracker_no_raise(self) -> None:
        g = _bare_graph()
        g._token_tracker = None
        g.max_total_tokens = 1000
        g._enforce_token_budget()  # no raise
