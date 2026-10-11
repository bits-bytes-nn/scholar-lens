r"""Normalize LaTeX math to the blog's one delimiter: ``$$ ... $$``.

The target Jekyll blog (kramdown GFM + MathJax) renders ``$$...$$`` both inline
and as display math, and nothing else: ``\( ... \)`` and ``\[ ... \]`` come out
as literal "(x_i)", and a ``$$`` block glued to its neighbouring lines renders
inline. kramdown passes math spans through untouched, so underscores inside
``$$`` must NOT be escaped (``\_`` would show as a literal underscore) except
inside ``\text{}``/``\texttt{}``, where they are text.

:func:`normalize_math` rewrites, outside code only:

* ``\( ... \)`` / ``\\( ... \\)`` → inline ``$$ ... $$``;
* ``\[ ... \]`` / ``\\[ ... \\]`` → a display ``$$ ... $$`` block;
* ``\_`` inside ``$$`` → ``_`` (kept within ``\text{}``/``\texttt{}``);
* a ``$$`` span that fills whole lines → surrounded by blank lines.

Single ``$`` is prose (currency) and is left alone. The function is idempotent.
"""

from __future__ import annotations

import re

# Code first (never rewritten), then the math spans. ``display`` is any
# ``$$...$$`` span, inline or block. Single ``$`` is deliberately not matched.
_SEGMENT = re.compile(
    r"""
    (?P<fence>^[ \t]*(?P<ff>```|~~~)[^\n]*\n(?:[^\n]*\n)*?[ \t]*(?P=ff)[ \t]*$)
    | (?P<icode>(?<!`)(?P<ib>`+)[^\n]*?(?<!`)(?P=ib)(?!`))
    | (?P<display>\$\$.+?\$\$)
    | (?P<paren>\\{1,2}\((?:\\.|[^\n\\])+?\\{1,2}\))
    | (?P<bracket>^[ \t]*\\{1,2}\[.+?\\{1,2}\][ \t]*$)
    """,
    re.VERBOSE | re.MULTILINE | re.DOTALL,
)

# Public alias: the lint pass classifies regions with the same splitter.
SEGMENT_PATTERN = _SEGMENT

_PAREN_DELIMS = re.compile(r"^\\{1,2}\(|\\{1,2}\)$")
_BRACKET_DELIMS = re.compile(r"^\\{1,2}\[|\\{1,2}\]$")
# Text-mode commands whose bodies keep a literal ``\_`` (one level of nested
# braces). Shared with the post validator.
TEXT_MODE_COMMANDS: tuple[str, ...] = (
    "text",
    "texttt",
    "textrm",
    "textit",
    "textbf",
    "mathrm",
)
TEXT_COMMAND = _TEXT_COMMAND = re.compile(
    rf"\\(?:{'|'.join(TEXT_MODE_COMMANDS)})\{{(?:[^{{}}]|\{{[^{{}}]*\}})*\}}"
)
_ESCAPED_UNDERSCORE = re.compile(r"\\_")


def _unescape_underscores(math: str) -> str:
    """``\\_`` → ``_`` inside a math span, except within \\text/\\texttt."""
    out: list[str] = []
    pos = 0
    for m in _TEXT_COMMAND.finditer(math):
        out.append(_ESCAPED_UNDERSCORE.sub("_", math[pos : m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_ESCAPED_UNDERSCORE.sub("_", math[pos:]))
    return "".join(out)


def _as_display(body: str) -> str:
    return f"\n\n$${body.strip()}$$\n\n"


def _rewrite_spans(markdown: str) -> str:
    def repl(m: re.Match[str]) -> str:
        kind = m.lastgroup
        text = m.group(0)
        if kind in ("fence", "icode"):
            return text
        if kind == "paren":
            return (
                "$$" + _unescape_underscores(_PAREN_DELIMS.sub("", text)).strip() + "$$"
            )
        if kind == "bracket":
            return _as_display(_unescape_underscores(_BRACKET_DELIMS.sub("", text)))
        return _unescape_underscores(text)

    return _SEGMENT.sub(repl, markdown)


def _is_table_or_list_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("|") or bool(re.match(r"([-*+]|\d+\.)\s", stripped))


def _isolate_display_blocks(markdown: str) -> str:
    """Put blank lines around ``$$`` spans that occupy whole lines, so kramdown
    renders them as display math; table rows and list items are left alone."""
    out: list[str] = []
    pos = 0
    for m in _SEGMENT.finditer(markdown):
        if m.lastgroup != "display":
            continue
        start, end = m.start(), m.end()
        line_start = markdown.rfind("\n", 0, start) + 1
        line_end = markdown.find("\n", end)
        line_end = len(markdown) if line_end == -1 else line_end
        before, after = markdown[line_start:start], markdown[end:line_end]
        if before.strip() or after.strip():
            continue  # inline math inside a sentence
        if before or _is_table_or_list_line(markdown[line_start:line_end]):
            continue  # indented (list continuation) or table row
        prefix = markdown[pos:line_start]
        out.append(prefix)
        if prefix and not prefix.endswith("\n\n"):
            out.append("\n")
        out.append(markdown[line_start:end])
        rest = markdown[end:]
        if rest and not rest.startswith("\n\n"):
            out.append("\n" if rest.startswith("\n") else "\n\n")
        pos = end
    out.append(markdown[pos:])
    return "".join(out)


def _collapse_blank_runs(markdown: str) -> str:
    """Squeeze 3+ newlines (left by display conversion) outside code fences."""
    out: list[str] = []
    pos = 0
    for m in _SEGMENT.finditer(markdown):
        if m.lastgroup == "fence":
            out.append(re.sub(r"\n{3,}", "\n\n", markdown[pos : m.start()]))
            out.append(m.group(0))
            pos = m.end()
    out.append(re.sub(r"\n{3,}", "\n\n", markdown[pos:]))
    return "".join(out)


def normalize_math(markdown: str) -> str:
    """Rewrite math delimiters and escaping to the blog's ``$$`` convention."""
    text = _rewrite_spans(markdown)
    text = _isolate_display_blocks(text)
    return _collapse_blank_runs(text)


def has_math(markdown: str) -> bool:
    """Whether the (normalized) Markdown contains a ``$$`` math span outside code."""
    return any(m.lastgroup == "display" for m in _SEGMENT.finditer(markdown))
