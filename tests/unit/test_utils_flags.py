"""Tests for the is_affirmative / is_placeholder LLM-output helpers."""

from __future__ import annotations

import pytest

from scholar_lens.domain.text import is_affirmative, is_placeholder


@pytest.mark.parametrize(
    "value,expected",
    [
        ("y", True),
        ("Y", True),
        (" yes ", True),
        ("Yes", True),
        ("YES", True),
        ("Yes.", True),
        ("y.", True),
        ("n", False),
        ("no", False),
        ("nope", False),
        ("", False),
        (None, False),
        ("maybe", False),
        ("true", False),
    ],
)
def test_is_affirmative(value: str | None, expected: bool) -> None:
    assert is_affirmative(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("N/A", True),
        ("n/a", True),
        (" N/A ", True),
        ("Unknown", True),
        ("unknown", True),
        ("N/A.", True),
        ("None", True),
        ("none.", True),
        ("Not specified", True),
        ("", True),
        (None, True),
        ("Edward Hu", False),
        ("LoRA", False),
    ],
)
def test_is_placeholder(value: str | None, expected: bool) -> None:
    assert is_placeholder(value) is expected


def test_clip_cuts_notes_at_a_sentence_or_word_boundary() -> None:
    from scholar_lens.domain.text import clip

    assert clip("short  note") == "short note"
    long = "첫 문장은 충분히 깁니다. " * 20 + "끝"
    out = clip(long, 100)
    assert len(out) <= 102 and out.endswith("다. …")
    words = "word " * 80
    assert clip(words, 50).endswith("word…") and len(clip(words, 50)) <= 50
