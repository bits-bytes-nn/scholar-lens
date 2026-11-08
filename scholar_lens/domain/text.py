"""Plain-text helpers."""

from __future__ import annotations

import re
from typing import Any

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


def parse_quality_score(raw: Any) -> int:
    """The evaluator's 0-100 score: the first integer of its
    ``<quality_score>`` tag body, so "85", "85/100" or "Score: 85" read 85.

    Shared by the review and tech-guide evaluate-and-revise loops. The result
    is clamped to 0-100; a non-numeric value reads 0, which forces another
    attempt rather than crashing.
    """
    match = re.search(r"\d+", str(raw if raw is not None else ""))
    return max(0, min(100, int(match.group()))) if match else 0


_NONE_ANSWERS = {
    "none",
    "n/a",
    "nothing missing",
    "없음",
    "해당 없음",
    "없습니다",
    "",
}


def _bare_answer(item: str) -> str:
    """``item`` without list numbering, emphasis, punctuation, a trailing
    dash or parenthetical ("1. None —", "none (all covered)" → "none")."""
    bare = re.sub(r"^\d+[.)]\s*", "", item.lower())
    bare = re.sub(r"\s*(?:\([^)]*\)|[-—–]+)\s*$", "", bare)
    # "None — all items are covered", "없음: 모두 반영됨"
    bare = re.sub(r"^(none|n/a|없음|해당 없음)\s*[-—–:].*$", r"\1", bare)
    return bare.strip("\"'`()[]{}.!;。*_ ")


def parse_lines(raw: str | None) -> list[str]:
    """Non-empty bullet/plain lines of a tag body ("none" means empty, with or
    without trailing punctuation)."""
    lines = []
    for line in (raw or "").splitlines():
        item = line.strip().lstrip("-*•").strip()
        if item and _bare_answer(item) not in _NONE_ANSWERS:
            lines.append(item)
    return lines


def tag_body(raw: str, tag: str) -> str | None:
    """Raw body of the last ``<tag>…</tag>`` in ``raw`` (no unescaping)."""
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", raw or "", re.DOTALL)
    return matches[-1] if matches else None


def escape_yaml_double_quoted(value: str) -> str:
    """Escape a string for use inside a double-quoted YAML scalar (``"..."``).

    arXiv/LLM-extracted titles routinely carry raw LaTeX (``\\mathcal``,
    ``\\nabla``, accented ``\\"o``). A double-quoted YAML scalar treats ``\\`` as
    an escape introducer, so an unescaped backslash either aborts the Jekyll/
    kramdown build (``\\mathcal`` -> unknown-escape ScannerError) or silently
    corrupts the value (``\\nabla`` -> literal newline). Escape backslashes FIRST,
    then double quotes, then the control chars YAML would otherwise mangle.
    """
    value = value.replace("\\", "\\\\").replace('"', '\\"')
    value = value.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    return value
