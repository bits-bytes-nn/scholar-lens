"""Plain-text helpers."""

from __future__ import annotations

import re

_ABBREVIATIONS = re.compile(r"\b(?:e\.g|i\.e|et al|vs|cf|etc)\.$", re.IGNORECASE)


def first_sentence(text: str) -> str:
    """``text`` up to its first sentence end (``.!?。`` before whitespace),
    not counting abbreviations such as "e.g." or "et al."; all of ``text``
    when it has none."""
    for match in re.finditer(r"[.!?。](?=\s)", text):
        if not _ABBREVIATIONS.search(text[: match.end()]):
            return text[: match.end()]
    return text


# The yes/no and missing-value answers the prompts ask for.
_AFFIRMATIVE = frozenset({"y", "yes"})


# "Unknown" is what ``paper_source`` sets for a source without authors.
_PLACEHOLDER = frozenset({"n/a", "unknown", "none", "not specified"})


def _bare(value: str) -> str:
    return value.strip().rstrip(".").strip().lower()


def is_affirmative(value: str | None) -> bool:
    """Whether an LLM yes/no field ("y or n", "yes or no") means "yes"; case,
    surrounding space and a trailing period are ignored. Anything else (incl.
    None) is no."""
    return value is not None and _bare(value) in _AFFIRMATIVE


def is_placeholder(value: str | None) -> bool:
    """Whether a value is empty or a missing-value marker (N/A, Unknown, None,
    Not specified), with or without a trailing period."""
    return not value or _bare(value) in _PLACEHOLDER
