"""Length policy shared by guides and surveys (prose characters)."""

from __future__ import annotations

# Prose-length target per section depth, in Korean characters of prose as
# ``section_loop.prose_chars`` measures it (PROSE_UNIT). Single source for the
# synopsis budget, the section directives and the evaluator's length report.
DEPTH_CHARS: dict[str, int] = {"deep": 5_000, "standard": 2_500, "brief": 1_000}


# Other languages need about this many times the characters for the same content.
NON_KOREAN_CHAR_FACTOR: float = 2.5
