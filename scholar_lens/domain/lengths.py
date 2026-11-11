"""Length policy shared by guides and surveys (prose characters)."""

from __future__ import annotations

from collections.abc import Sequence

# Prose-length target per section depth, in Korean characters of prose
# (spaces included; code blocks, tables and images excluded). Single source
# for the evidence engines' section targets.
DEPTH_CHARS: dict[str, int] = {"deep": 5_000, "standard": 2_500, "brief": 1_000}


# Languages written in a dense script (a syllable or word per character) need
# about as many characters as Korean for the same content; alphabetic
# languages about this many times as many.
_DENSE_SCRIPTS = frozenset({"korean", "japanese", "chinese"})
NON_KOREAN_CHAR_FACTOR: float = 2.5


def char_factor(language: str) -> float:
    """Characters needed for the same content as in Korean."""
    name = language.strip().lower()
    return (
        1.0
        if any(name.startswith(s) for s in _DENSE_SCRIPTS)
        else (NON_KOREAN_CHAR_FACTOR)
    )


def scaled_targets(
    raw: Sequence[int], *, min_total: int, max_total: int, language: str
) -> list[int]:
    """Per-section prose targets: ``raw`` (Korean characters) scaled so the
    total falls in ``[min_total, max_total]``, in ``language``'s characters."""
    total = sum(raw)
    if not total:
        return [0 for _ in raw]
    scale = min(max(total, min_total), max_total) / total
    factor = char_factor(language)
    return [round(r * scale * factor) for r in raw]
