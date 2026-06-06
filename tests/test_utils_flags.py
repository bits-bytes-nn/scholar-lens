"""Tests for the is_affirmative / is_placeholder LLM-output helpers."""

from __future__ import annotations

import pytest

from scholar_lens.src.utils import is_affirmative, is_placeholder


@pytest.mark.parametrize(
    "value,expected",
    [
        ("y", True),
        ("Y", True),
        (" yes ", True),
        ("Yes", True),
        ("true", True),
        ("YEAH", True),
        ("n", False),
        ("no", False),
        ("nope", False),
        ("", False),
        (None, False),
        ("maybe", False),
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
        ("none", True),
        ("anonymous", True),
        ("", True),
        (None, True),
        ("Edward Hu", False),
        ("LoRA", False),
    ],
)
def test_is_placeholder(value: str | None, expected: bool) -> None:
    assert is_placeholder(value) is expected
