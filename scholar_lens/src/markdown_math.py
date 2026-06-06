r"""Make LaTeX math survive the blog's kramdown(GFM) + MathJax pipeline.

The target Jekyll blog parses Markdown with ``kramdown`` in GFM mode. GFM treats
``_`` as an emphasis delimiter, so a math span like ``$W_0 + h_t$`` can have its
subscript underscores eaten (rendered as italics) before MathJax ever sees it.
The fix the author applies by hand is to escape underscores inside math as
``\_``. This module does that automatically, but ONLY inside math spans, and
without touching code (fenced blocks or inline code), where ``_`` is literal and
must be left alone.

:func:`normalize_math_underscores` is the entry point.
"""

from __future__ import annotations

import re

# Fenced code blocks (``` or ~~~), inline code (`...`), then display ($$...$$)
# and inline ($...$) math. Order matters: code is matched first so math inside
# code is never rewritten, and $$ before $ so display math isn't split.
_SEGMENT = re.compile(
    r"""
    (?P<fence>^[ \t]*(?P<ff>```|~~~).*?\n(?:.*?\n)*?[ \t]*(?P=ff)[ \t]*$)
    | (?P<icode>`+[^`]*`+)
    | (?P<display>\$\$.+?\$\$)
    | (?P<inline>(?<!\$)\$(?!\s)(?:\\\$|[^$\n])+?\$(?!\$))
    """,
    re.VERBOSE | re.MULTILINE | re.DOTALL,
)


def _escape_underscores(math: str) -> str:
    """Escape unescaped underscores in a math span (``_`` -> ``\\_``)."""
    return re.sub(r"(?<!\\)_", r"\\_", math)


def normalize_math_underscores(markdown: str) -> str:
    """Escape underscores inside LaTeX math spans for kramdown+GFM safety.

    Leaves code (fenced or inline) and ordinary prose untouched; only the
    contents of ``$...$`` and ``$$...$$`` are rewritten. Idempotent — already
    escaped ``\\_`` is preserved.
    """

    def repl(match: re.Match[str]) -> str:
        if match.lastgroup in ("fence", "icode"):
            return match.group(0)  # never touch code
        return _escape_underscores(match.group(0))

    return _SEGMENT.sub(repl, markdown)
