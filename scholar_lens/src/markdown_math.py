r"""Make LaTeX math survive the blog's kramdown(GFM) + MathJax pipeline.

The target Jekyll blog parses Markdown with ``kramdown`` in GFM mode. GFM treats
``_`` as an emphasis delimiter, so a math span like ``\(W_0 + h_t\)`` can have
its subscript underscores eaten (rendered as italics) before MathJax ever sees
it. The fix the author applies by hand is to escape underscores inside math as
``\_``. This module does that automatically, but ONLY inside math spans, and
without touching code (fenced blocks or inline code), where ``_`` is literal and
must be left alone.

Inline math uses ``\( ... \)`` (the blog strips single-``$`` delimiters to
protect prose like prices, so ``$...$`` does not render and is not treated as
math here); display math uses ``$$ ... $$``.

:func:`normalize_math_underscores` is the entry point.
"""

from __future__ import annotations

import re

# Fenced code blocks (``` or ~~~), inline code (`...`), then display ($$...$$)
# and inline (\( ... \)) math. Order matters: code is matched first so math
# inside code is never rewritten. Single-$ spans are intentionally NOT matched:
# the blog strips $ delimiters (to protect prose like prices), so $...$ is not
# math here and must be left as prose.
_SEGMENT = re.compile(
    r"""
    (?P<fence>^[ \t]*(?P<ff>```|~~~).*?\n(?:.*?\n)*?[ \t]*(?P=ff)[ \t]*$)
    | (?P<icode>`+[^`]*`+)
    | (?P<display>\$\$.+?\$\$)
    | (?P<inline>\\\((?:\\.|[^\n\\])+?\\\))
    """,
    re.VERBOSE | re.MULTILINE | re.DOTALL,
)

# Public alias: the markdown lint pass reuses this validated code/math-span
# splitter so it classifies regions identically (code vs math vs prose).
SEGMENT_PATTERN = _SEGMENT


# Double-backslashed math delimiters: ``\\(``/``\\)``/``\\[``/``\\]``. Some model
# outputs emit these (a JSON/escaping artifact), but on the blog ``\\(`` renders
# as a literal backslash followed by ``(`` — the math never activates. Collapse
# to the single-backslash form the pipeline expects BEFORE segment-splitting, so
# the spans are then recognised as math (and underscore-escaped). Only the four
# math delimiters are collapsed; other ``\\`` (e.g. line breaks in math) stay.
_DOUBLE_BS_DELIM = re.compile(r"\\\\([()\[\]])")


def _fix_double_backslash_delims(markdown: str) -> str:
    """Collapse ``\\\\(``/``\\\\)``/``\\\\[``/``\\\\]`` to a single backslash,
    outside code spans, so the delimiters actually activate math."""

    def repl(match: re.Match[str]) -> str:
        seg = match.group(0)
        if match.lastgroup in ("fence", "icode"):
            return seg  # never touch code
        return _DOUBLE_BS_DELIM.sub(r"\\\1", seg)

    # Split on code spans only, then fix delimiters in the non-code gaps. We run
    # _SEGMENT to protect code; math/prose gaps are handled by the else-branch of
    # a manual walk rather than _SEGMENT.sub (which would miss prose gaps).
    out: list[str] = []
    pos = 0
    for m in _SEGMENT.finditer(markdown):
        if m.lastgroup in ("fence", "icode"):
            out.append(_DOUBLE_BS_DELIM.sub(r"\\\1", markdown[pos : m.start()]))
            out.append(m.group(0))  # code: untouched
            pos = m.end()
    out.append(_DOUBLE_BS_DELIM.sub(r"\\\1", markdown[pos:]))
    return "".join(out)


def _escape_underscores(math: str) -> str:
    """Escape unescaped underscores in a math span (``_`` -> ``\\_``)."""
    return re.sub(r"(?<!\\)_", r"\\_", math)


def normalize_math_underscores(markdown: str) -> str:
    """Escape underscores inside LaTeX math spans for kramdown+GFM safety.

    Leaves code (fenced or inline) and ordinary prose untouched; only the
    contents of ``\\(...\\)`` and ``$$...$$`` are rewritten. Idempotent — already
    escaped ``\\_`` is preserved.

    First collapses double-backslashed math delimiters (``\\\\(`` -> ``\\(`` etc.)
    outside code, so spans that arrived over-escaped are recognised as math and
    get underscore-escaped too.
    """
    markdown = _fix_double_backslash_delims(markdown)

    def repl(match: re.Match[str]) -> str:
        if match.lastgroup in ("fence", "icode"):
            return match.group(0)  # never touch code
        return _escape_underscores(match.group(0))

    return _SEGMENT.sub(repl, markdown)
