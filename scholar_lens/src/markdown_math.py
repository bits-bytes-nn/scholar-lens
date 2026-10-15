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
* a ``$$`` span that fills whole lines → surrounded by blank lines (an
  indented one is kept in its list item or dedented outside a list);
* ``\|`` as a conditional bar in a probability/policy call → ``\mid``;
  ``\|`` on a table row → ``\Vert``; a bare ``|`` in inline math →
  ``\vert`` (kramdown would split the line into table cells).

Single ``$`` is prose (currency) and is left alone. The function is idempotent.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass

# Markdown syntax shared by every module (one definition each).
# Hangul as the blog lint's \p{Hangul}: syllables, compatibility and
# conjoining jamo (NFD text), circled and halfwidth forms.
HANGUL = re.compile(
    "[\u1100-\u11ff\u302e\u302f\u3131-\u318e\u3200-\u321e\u3260-\u327e"
    "\ua960-\ua97c\uac00-\ud7a3\ud7b0-\ud7fb\uffa0-\uffdc]"
)
# An image: group 1 alt, group 2 source (balanced parentheses allowed), group 3
# an optional title with its leading space.
IMAGE = re.compile(
    r'!\[([^\]]*)\]\(([^()\s]*(?:\([^()]*\)[^()\s]*)*)((?:\s+"[^"]*")?)\)'
)
# A display equation: "$$" alone on a line … "$$", or one whole-line "$$…$$"
# (a prose line that merely starts with inline math is not one).
DISPLAY_MATH = re.compile(
    r"^[ \t]*\$\$[ \t]*\n.*?\n[ \t]*\$\$[ \t]*$"
    r"|^[ \t]*\$\$(?:(?!\$\$)[^\n])+\$\$[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def image_sources(markdown: str) -> list[str]:
    return [m.group(2) for m in IMAGE.finditer(markdown)]


# A fence line: 3+ backticks or tildes. Any indentation counts (a fence nested
# in a list item is indented past CommonMark's three spaces), and so does a
# blockquote prefix (a fence inside a quote).
_FENCE_LINE = re.compile(r"^[ \t]*(?:>[ \t]?)*[ \t]*(`{3,}|~{3,})(.*)$")


_LIST_MARKER = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
# One whole-line display: a single ``$$…$$`` span (not a line that starts and
# ends with two inline spans), a lone ``$$`` delimiter, or a bracket display.
_WHOLE_LINE_DISPLAY = re.compile(
    r"^[ \t]*(?:\$\$(?:(?!\$\$).)+\$\$|\$\$|\\{1,2}\[.*|.*\\{1,2}\])[ \t]*$"
)


# A whole display on one line: one ``$$…$$`` span or ``\[…\]``.
_ONE_LINE_DISPLAY = re.compile(
    r"^[ \t]*(?:\$\$(?:(?!\$\$).)+\$\$|\\{1,2}\[.*\\{1,2}\])[ \t]*$"
)


def indented_code_lines(lines: list[str]) -> set[int]:
    """Indices of lines kramdown reads as an indented code block: four or more
    columns past the enclosing list item's content (or the margin), after a
    blank line. A whole-line ``$$`` display that opens such a block is left
    out: an over-indented equation is meant as math, and the normalizer
    re-indents it."""
    return set(indented_code_bases(lines))


def indented_code_bases(lines: list[str]) -> dict[int, int]:
    """:func:`indented_code_lines` with each line's base column: the content
    column of its list item (0 at the margin); the code starts 4 past it."""
    code: dict[int, int] = {}
    base = 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        width = len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())
        item = _LIST_MARKER.match(line)
        if item and width < base + 4:
            base = len(item.group(0).expandtabs(4))
            continue
        if width == 0:
            base = 0
            continue
        previous_blank = i > 0 and not lines[i - 1].strip()
        if (
            width >= base + 4
            and (previous_blank or i - 1 in code)
            # an over-indented equation opening a block is meant as math; one
            # inside a code block is part of the sample
            and not (previous_blank and _WHOLE_LINE_DISPLAY.match(line))
        ):
            code[i] = base
    return code


def indented_code_test(markdown: str) -> Callable[[int], bool]:
    """A test for "offset sits on a line of an indented code block"."""
    code = indented_code_lines(markdown.split("\n"))
    if not code:
        return lambda _: False
    starts = [0] + [i + 1 for i, ch in enumerate(markdown) if ch == "\n"]
    return lambda offset: bisect_right(starts, offset) - 1 in code


def fence_roles(lines: list[str]) -> list[str]:
    """Each line's role: ``"open"``, ``"code"``, ``"close"`` or ``"prose"``.

    CommonMark rules: a fence closes only on a line of the same character, at
    least as long as the opener, with nothing after it; a backtick opener's
    info string may not contain a backtick; an unclosed fence runs to the end.
    """
    roles: list[str] = []
    opener = ""
    for line in lines:
        match = _FENCE_LINE.match(line.rstrip("\n"))
        if opener:
            run = match.group(1) if match else ""
            if (
                match
                and run[0] == opener[0]
                and len(run) >= len(opener)
                and not match.group(2).strip()
            ):
                roles.append("close")
                opener = ""
            else:
                roles.append("code")
        elif match and not (match.group(1)[0] == "`" and "`" in match.group(2)):
            roles.append("open")
            opener = match.group(1)
        else:
            roles.append("prose")
    return roles


def open_fence_line(lines: list[str]) -> int | None:
    """Index of the opener of a fence left unclosed at the end, if any."""
    opened: int | None = None
    for i, role in enumerate(fence_roles(lines)):
        if role == "open":
            opened = i
        elif role == "close":
            opened = None
    return opened


def open_fence(markdown: str) -> str:
    """The opening run of a fence left unclosed at the end ("" when none)."""
    lines = markdown.splitlines()
    opened = open_fence_line(lines)
    if opened is None:
        return ""
    return _FENCE_LINE.match(lines[opened]).group(1)  # type: ignore[union-attr]


def fence_spans(markdown: str) -> list[tuple[int, int]]:
    """``(start, end)`` offsets of every fenced code block, fence lines
    included (an unclosed fence runs to the end of the text)."""
    spans: list[tuple[int, int]] = []
    lines = markdown.splitlines(keepends=True)
    pos, start = 0, -1
    for line, role in zip(lines, fence_roles(lines), strict=True):
        if role == "open":
            start = pos
        pos += len(line)
        if role == "close":
            spans.append((start, pos - (1 if line.endswith("\n") else 0)))
            start = -1
    if start >= 0:
        spans.append((start, len(markdown)))
    return spans


def without_fences(markdown: str) -> str:
    """``markdown`` with every fenced code block replaced by a space."""
    out, pos = [], 0
    for start, end in fence_spans(markdown):
        out.append(markdown[pos:start])
        pos = end
    out.append(markdown[pos:])
    return " ".join(out)


# Inline code first (never rewritten), then the math spans. ``display`` is any
# ``$$...$$`` span, inline or block. Single ``$`` is deliberately not matched.
# Fenced code is found by :func:`fence_spans` (a regex cannot pair fences).
_SPANS = re.compile(
    r"""
    (?P<icode>(?<!`)(?P<ib>`+)[^\n]*?(?<!`)(?P=ib)(?!`))
    | (?P<display>\$\$.+?\$\$)
    | (?P<paren>\\{1,2}\((?:\\.|[^\n\\])+?\\{1,2}\))
    | (?P<bracket>^[ \t]*(?:>[ \t]?)*[ \t]*\\{1,2}\[.+?\\{1,2}\][ \t]*$)
    """,
    re.VERBOSE | re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class Segment:
    """A protected or math span: kind is ``fence``, ``icode``, ``display``,
    ``paren`` or ``bracket``."""

    kind: str
    start: int
    end: int


def segments(markdown: str) -> list[Segment]:
    """Code and math spans of ``markdown`` in order (math never inside code)."""
    out: list[Segment] = []
    pos = 0
    for start, end in [*fence_spans(markdown), (len(markdown), len(markdown))]:
        gap = markdown[:start]
        for m in _SPANS.finditer(gap, pos):
            out.append(Segment(m.lastgroup or "", m.start(), m.end()))
        if end > start:
            out.append(Segment("fence", start, end))
        pos = end
    return out


_TEX_COMMAND = re.compile(r"\\[A-Za-z]")
_PAREN_DELIMS = re.compile(r"^\\{1,2}\(|\\{1,2}\)$")
# Letters only, so "\(s\)" after a word is a suffix, not math.
_WORD_SUFFIX = re.compile(r"[A-Za-z]{1,3}")
_BRACKET_DELIMS = re.compile(r"^[ \t]*\\{1,2}\[|\\{1,2}\][ \t]*$")
QUOTE_PREFIX = re.compile(r"^[ \t]*(?:>[ \t]?)+")
# Text-mode commands whose bodies keep a literal ``\_`` (braces nested up to
# three levels): every \text… command, \mathrm and \operatorname, as the blog
# lint treats them. Shared with the post validator.
_NESTED_BRACES = r"\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}"
TEXT_COMMAND = re.compile(r"\\(?:text\w*|mathrm|mathtt|operatorname)" + _NESTED_BRACES)
ESCAPED_UNDERSCORE = re.compile(r"\\_")


def _unescape_underscores(math: str) -> str:
    """``\\_`` → ``_`` inside a math span, except within \\text/\\texttt."""
    out: list[str] = []
    pos = 0
    for m in TEXT_COMMAND.finditer(math):
        out.append(ESCAPED_UNDERSCORE.sub("_", math[pos : m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(ESCAPED_UNDERSCORE.sub("_", math[pos:]))
    return "".join(out)


# \| and \Vert are for norms and divergences only. Inside a probability or
# policy call (p, P, q, Pr, \pi) a single \| is a conditional bar, and a
# \lVert x \rVert used as the bound of a sum is a length (blog post fixes,
# PR #86). A sub/superscript is braced, one character or one command.
_BRACED = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"  # two nesting levels
_SUB = rf"(?:'|\s*[_^](?:{_BRACED}|\\[A-Za-z]+|\w|\*))*"
# p, P, q (not Q, the action value; not a subscript as in D_p or \ell_p),
# \mathbb{P}, \Pr (also with brackets) and \pi, bare or decorated (\hat{p},
# \tilde{\pi}) and primed (p'), opened plainly or with \left.
_BASE = r"(?:\\pi|\\Pr|\\mathbb\{P\}|(?<![\\A-Za-z_^])[pPq])"
_CALL = re.compile(
    rf"(?P<head>(?:\\(?:hat|tilde|bar|widehat|widetilde)\{{{_BASE}\}}|{_BASE})"
    rf"{_SUB}\s*(?:\\left\s*)?(?P<open>[(\[]))(?P<arg>[^()\[\]]*)(?P<close>[)\]])"
)
_NORM_BAR = re.compile(r"(?<!\\)\\(?:\||Vert(?![A-Za-z]))")
_LENGTH_BOUND = re.compile(
    r"\\(?:sum|prod)\s*_(?:\{(?:[^{}]|\{[^{}]*\})*\}|\\?\w)\s*"
    r"\^\{\s*(?:\\lVert|\\\|)\s*(?P<x>[^{}\\|]+?|[^{}\\|]*\{[^{}]*\})\s*(?:\\rVert|\\\|)\s*\}"
)


def _fix_bars(math: str) -> str:
    """Conditional bars written as \\| become \\mid; a \\lVert x \\rVert that
    bounds a sum (a length) becomes \\lvert x \\rvert everywhere in the span."""

    def conditional(m: re.Match[str]) -> str:
        arg = m.group("arg")
        if len(_NORM_BAR.findall(arg)) != 1 or {
            m.group("open"),
            m.group("close"),
        } not in (
            {"(", ")"},
            {"[", "]"},
        ):
            return m.group(0)
        # A letter right after the bar would glue onto the command (\midx).
        bar = _NORM_BAR.search(arg)
        if bar is None:
            return m.group(0)
        rest = arg[bar.end() :]
        sep = " " if rest[:1].isalpha() else ""
        return (
            f"{m.group('head')}{arg[: bar.start()]}\\mid{sep}{rest}{m.group('close')}"
        )

    math = _CALL.sub(conditional, math)
    for x in {m.group("x").strip() for m in _LENGTH_BOUND.finditer(math)}:
        # The same quantity elsewhere in the span, unless subscripted like a
        # norm (\|x\|_2).
        norm = re.compile(
            rf"(?:\\lVert|\\\|)\s*{re.escape(x)}\s*(?:\\rVert|\\\|)(?!\s*[_^])"
        )
        math = norm.sub(lambda _, x=x: f"\\lvert {x} \\rvert", math)
    return math


def _as_display(
    body: str, indent: str = "", quote: str = "", attr_follows: bool = False
) -> str:
    """A display block; ``indent`` keeps it inside a list item, ``quote``
    (blockquote markers) inside a quote. A kramdown attribute line right
    after it (``attr_follows``) stays glued to it."""
    lines = [QUOTE_PREFIX.sub("", line).strip() for line in body.strip().splitlines()]
    lines = [line for line in lines if line] or [""]
    if quote:
        bar = quote.rstrip()
        tail = "" if attr_follows else f"\n{bar}"
        return f"{bar}\n{quote}$$" + f"\n{quote}".join(lines) + "$$" + tail
    tail = "" if attr_follows else "\n\n"
    return f"\n\n{indent}$$" + f"\n{indent}".join(lines) + "$$" + tail


_ATTR_LINE = re.compile(r"\n[ \t]*(?:>[ \t]?)*[ \t]*\{:")


def _attr_follows(markdown: str, end: int) -> bool:
    """Whether the next line is a kramdown attribute list (``{: .class}``)."""
    return _ATTR_LINE.match(markdown, end) is not None


# kramdown splits table cells at \| even inside math (a norm turns into an
# absolute value), so on a table row it is written \Vert.
_TABLE_NORM_BAR = re.compile(r"(?<!\\)\\\|(?P<next>[A-Za-z]?)")


def _as_vert(m: re.Match[str]) -> str:
    return "\\Vert" + (" " + m.group("next") if m.group("next") else "")


_DELIMITER_ROW = re.compile(
    r"^[ \t]*(?:>[ \t]?)*[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t]*$"
)


def _on_table_row(markdown: str, start: int) -> bool:
    """Whether ``start`` is on a table row: one that starts with ``|``, or a
    pipe-less row of a table whose delimiter row (``--- | ---``) is in the
    same block of lines."""
    line_start = markdown.rfind("\n", 0, start) + 1
    before = QUOTE_PREFIX.sub("", markdown[line_start:start])
    if before.lstrip().startswith("|"):
        return True
    line_end = markdown.find("\n", start)
    if "|" not in markdown[line_start : len(markdown) if line_end == -1 else line_end]:
        return False  # a pipe-less table row still has its cell separators
    block_start = markdown.rfind("\n\n", 0, line_start)
    block_start = 0 if block_start == -1 else block_start + 2
    block_end = markdown.find("\n\n", start)
    block_end = len(markdown) if block_end == -1 else block_end
    return any(
        _DELIMITER_ROW.match(line)
        for line in markdown[block_start:block_end].split("\n")
    )


def _columns(indent: str) -> int:
    return len(indent.expandtabs(4))


def indent_width(line: str) -> int:
    return _columns(line[: len(line) - len(line.lstrip(" \t"))])


def _list_indent(markdown: str, start: int, text: str) -> str:
    """The indent for an indented display: the content column of the list
    item it continues (so it stays in the item without turning into an
    indented code block), or none outside a list (an indent of four or more
    there would make it code). A display glued to its item (no blank line
    between) goes to the item's column even when it is indented less: at its
    own indent it would end the item and restart the list."""
    indent = text[: len(text) - len(text.lstrip(" \t"))]
    if not indent:
        return ""
    width = _columns(indent)
    line_start = markdown.rfind("\n", 0, start) + 1
    glued = True
    for line in _lines_before(markdown, line_start):
        if not line.strip():
            glued = False
            continue
        item = _LIST_MARKER.match(line)
        if item:
            column = _columns(item.group(0))
            return " " * column if width >= column or glued else indent
        if indent_width(line) == 0:
            return ""  # a paragraph at the margin: not in a list
    return ""


def _lines_before(markdown: str, end: int) -> list[str]:
    """Up to ``_MAX_SCAN`` lines ending at offset ``end``, nearest first (a
    bounded walk back, so long documents stay linear)."""
    out: list[str] = []
    stop = end
    while stop > 0 and len(out) < _MAX_SCAN:
        begin = markdown.rfind("\n", 0, stop - 1) + 1
        out.append(
            markdown[begin : stop - 1]
            if markdown[stop - 1] == "\n"
            else markdown[begin:stop]
        )
        stop = begin
    return out


def _reindent(text: str, indent: str) -> str:
    """Every line of a display at ``indent`` (its own indent replaced)."""
    return "\n".join(indent + line.lstrip(" \t") for line in text.split("\n"))


def _fills_lines(markdown: str, start: int, end: int) -> bool:
    """Whether a span occupies its lines alone (blockquote markers aside)."""
    line_start = markdown.rfind("\n", 0, start) + 1
    line_end = markdown.find("\n", end)
    line_end = len(markdown) if line_end == -1 else line_end
    before = QUOTE_PREFIX.sub("", markdown[line_start:start])
    return not (before + markdown[end:line_end]).strip()


# A "$$ … $$" pair kramdown cannot read as one formula: it crosses a blank
# line or a table's delimiter row (a stray $$ paired with the next real one).
# Its text is prose, not math, so no math rewrite may touch it; the validator
# reports the unpaired $$. (A wrapped formula whose next line starts with | is
# still one formula.)
_UNREADABLE_SPAN = re.compile(r"\n[ \t>]*\n|\n[ \t>]*\|?[ \t]*:?-{3,}[ \t]*[|:]")
# A wrapped span is two blocks to kramdown when a later line starts a heading,
# or a sibling list item of the item it opened in (a stray $$ in "- A: $$2/h"
# paired with the next item's), or when it starts on a table row (rows never
# wrap). A formula wrapped in a paragraph ("$$a\n+ b$$") is still joined.
_NEW_HEADING_LINE = re.compile(r"\n[ \t]*(?:>[ \t]?)*[ \t]*#{1,6}[ \t]")
_NEW_ITEM_LINE = re.compile(r"\n[ \t]*(?:>[ \t]?)*[ \t]*(?:[-*+]|\d+[.)])[ \t]")


def _unreadable(markdown: str, start: int, end: int) -> bool:
    """Whether ``markdown[start:end]`` ("$$ … $$") is not one formula to
    kramdown (see :data:`_UNREADABLE_SPAN`)."""
    text = markdown[start:end]
    if _UNREADABLE_SPAN.search(text):
        return True
    if "\n" not in text:
        return False
    line_start = markdown.rfind("\n", 0, start) + 1
    lead = QUOTE_PREFIX.sub("", markdown[line_start:start]).lstrip()
    if lead.startswith("|") or _NEW_HEADING_LINE.search(text):
        return True
    in_item = _LIST_MARKER.match(lead) is not None
    return in_item and _NEW_ITEM_LINE.search(text) is not None


_BLOCK_OPEN = re.compile(r"[ \t]*(?:>[ \t]?)*[ \t]*(?:\$\$|\\{1,2}\[)[ \t]*\n")
_BLOCK_CLOSE = re.compile(r"\n[ \t]*(?:>[ \t]?)*[ \t]*(?:\$\$|\\{1,2}\])[ \t]*$")


def _is_block_display(markdown: str, start: int, end: int) -> bool:
    """A display on its own lines ("$$\n … \n$$", or the same with \\[ \\]):
    kramdown reads a $$ block as one formula even across blank lines, and a
    bracket block becomes one."""
    text = markdown[start:end]
    return (
        _fills_lines(markdown, start, end)
        and _BLOCK_OPEN.match(text) is not None
        and _BLOCK_CLOSE.search(text) is not None
    )


def _opens_item(markdown: str, start: int, end: int) -> bool:
    """A ``$$`` span that fills the first line of a list item ("1. $$x$$", or
    a block "1. $$" … "$$"): kramdown renders it as the item's display, so
    it is neither joined into one line nor run into the text after it."""
    line_start = markdown.rfind("\n", 0, start) + 1
    line_end = markdown.find("\n", end)
    line_end = len(markdown) if line_end == -1 else line_end
    if _LIST_MARKER.fullmatch(markdown[line_start:start]) is None:
        return False
    return not markdown[end:line_end].strip()


def _rewrite_spans(markdown: str) -> str:
    out: list[str] = []
    pos = 0
    in_code = indented_code_test(markdown)
    for seg in segments(markdown):
        text = markdown[seg.start : seg.end]
        if (
            seg.kind in ("display", "bracket")
            and _unreadable(markdown, seg.start, seg.end)
            and not _is_block_display(markdown, seg.start, seg.end)
        ):
            continue
        if seg.kind != "fence" and in_code(seg.start):
            continue  # code, not math
        if seg.kind == "paren":
            body = _PAREN_DELIMS.sub("", text)
            if HANGUL.search(body) and not _TEX_COMMAND.search(body):
                continue  # Markdown-escaped parentheses around prose: "\(약 3달러\)"
            if (
                _WORD_SUFFIX.fullmatch(body)
                and markdown[seg.start - 1 : seg.start].isalpha()
            ):
                continue  # a plural glued to its word: "model\(s\)"
            text = "$$" + _fix_bars(_unescape_underscores(body)).strip() + "$$"
        elif seg.kind == "bracket":
            quote_match = QUOTE_PREFIX.match(text)
            quote = quote_match.group(0) if quote_match else ""
            if quote:
                # The indent before ">" keeps a quote inside its list item.
                lead = quote[: len(quote) - len(quote.lstrip())]
                lead = _list_indent(markdown, seg.start, lead + ">")
                quote = lead + quote.strip() + " "
                text = text[quote_match.end() :]  # type: ignore[union-attr]
            indent = "" if quote else _list_indent(markdown, seg.start, text)
            body = _BRACKET_DELIMS.sub("", text.strip())
            text = _as_display(
                _fix_bars(_unescape_underscores(body)),
                indent,
                quote,
                attr_follows=_attr_follows(markdown, seg.end),
            )
        elif seg.kind == "display":
            text = _fix_bars(_unescape_underscores(text))
            item_block = _opens_item(markdown, seg.start, seg.end)
            following = markdown[seg.end + 1 :].split("\n", 1)[0]
            if item_block and following.strip() and not _LIST_MARKER.match(following):
                text += "\n"  # a display opening an item, then a paragraph in it
            if (
                "\n" in text
                and not item_block
                and not _fills_lines(markdown, seg.start, seg.end)
                and not _BARE_BAR.search(_vert_bars(text[2:-2]))
            ):
                # An inline span wrapped over lines: one line (same render, and
                # the blog lint checks math line by line). Not with a | that
                # stays (in \text{}): on one line it would split into a table.
                text = re.sub(r"[ \t]*\n[ \t]*(?:>[ \t]?)*[ \t]*", " ", text)
            line_start = markdown.rfind("\n", 0, seg.start) + 1
            lead = markdown[line_start : seg.start]
            if lead and not lead.strip() and _fills_lines(markdown, seg.start, seg.end):
                # An indented whole-line display: in its list item, or dedented.
                indent = _list_indent(markdown, seg.start, lead + text)
                if indent != lead:
                    out.append(markdown[pos:line_start])
                    out.append(_reindent(text, indent))
                    pos = seg.end
                    continue
        if seg.kind in ("paren", "display") and _on_table_row(markdown, seg.start):
            text = _TABLE_NORM_BAR.sub(_as_vert, text)
        out.append(markdown[pos : seg.start])
        out.append(text)
        pos = seg.end
    out.append(markdown[pos:])
    return "".join(out)


def _is_table_or_list_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("|") or bool(re.match(r"([-*+]|\d+\.)\s", stripped))


def _isolate_display_blocks(markdown: str) -> str:
    """Put blank lines around ``$$`` spans that occupy whole lines, so kramdown
    renders them as display math; table rows and list items are left alone."""
    out: list[str] = []
    pos = 0
    for seg in segments(markdown):
        if seg.kind != "display":
            continue
        start, end = seg.start, seg.end
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
        # A kramdown attribute list ("{: .class}") stays glued to its block.
        if rest and not rest.startswith("\n\n") and not _attr_follows(markdown, end):
            out.append("\n" if rest.startswith("\n") else "\n\n")
        pos = end
    out.append(markdown[pos:])
    return "".join(out)


# kramdown reads a bare | in inline math as a table-cell separator: a
# one-line paragraph holding $$|S|$$ or $$p(y|x)$$ becomes a table. \vert is
# the same symbol in TeX (also after \left/\right/\big and in \{x | …\}), so
# inline spans spell it that way. Not inside text-mode commands or an array
# column spec, where | is not the delimiter; the validator still reports
# those.
_BARE_BAR = re.compile(r"(?<!\\)\|(?P<next>[A-Za-z]?)")
_BAR_KEEP = re.compile(
    r"\\(?:text\w*|mbox|mathrm|operatorname)\s*"  # also "\text {…}", \mbox
    + _NESTED_BRACES
    + r"|\\begin\{array\}"
    + _NESTED_BRACES
    + r"|\\href\{[^{}]*\}"  # a URL, not math
)


def _as_vert_bar(m: re.Match[str]) -> str:
    return "\\vert" + (" " + m.group("next") if m.group("next") else "")


def _vert_bars(math: str) -> str:
    out: list[str] = []
    pos = 0
    for keep in _BAR_KEEP.finditer(math):
        out.append(_BARE_BAR.sub(_as_vert_bar, math[pos : keep.start()]))
        out.append(keep.group(0))
        pos = keep.end()
    out.append(_BARE_BAR.sub(_as_vert_bar, math[pos:]))
    return "".join(out)


def _is_display_block(markdown: str, start: int, end: int) -> bool:
    """A ``$$`` span kramdown renders as a block: alone on its lines with a
    blank line (or the text's edge) before and after."""
    if not _fills_lines(markdown, start, end):
        return False
    line_start = markdown.rfind("\n", 0, start) + 1
    line_end = markdown.find("\n", end)
    before = (
        markdown[: max(line_start - 1, 0)].rsplit("\n", 1)[-1] if line_start else ""
    )
    after = "" if line_end == -1 else markdown[line_end + 1 :].split("\n", 1)[0]
    return not QUOTE_PREFIX.sub("", before).strip() and not (
        QUOTE_PREFIX.sub("", after).strip()
    )


def _vert_inline_bars(markdown: str) -> str:
    """Bare ``|`` → ``\\vert`` in every ``$$`` span that is not a display block
    (inline, wrapped across lines, in a list item, quote or table row)."""
    out: list[str] = []
    pos = 0
    in_code = indented_code_test(markdown)
    for seg in segments(markdown):
        if seg.kind != "display" or _is_display_block(markdown, seg.start, seg.end):
            continue
        text = markdown[seg.start : seg.end]
        if _unreadable(markdown, seg.start, seg.end) or in_code(seg.start):
            continue
        out.append(markdown[pos : seg.start])
        out.append("$$" + _vert_bars(text[2:-2]) + "$$")
        pos = seg.end
    out.append(markdown[pos:])
    return "".join(out)


def _collapse_blank_runs(markdown: str) -> str:
    """Squeeze 3+ newlines (left by display conversion) outside code fences."""
    out: list[str] = []
    pos = 0
    for start, end in fence_spans(markdown):
        out.append(re.sub(r"\n{3,}", "\n\n", markdown[pos:start]))
        out.append(markdown[start:end])
        pos = end
    out.append(re.sub(r"\n{3,}", "\n\n", markdown[pos:]))
    return "".join(out)


def _isolate_nested_displays(markdown: str) -> str:
    """Blank lines around a whole-line ``$$`` display (one line, or a block
    over several) glued to the text of a blockquote (``>`` blank lines) or a
    list item (blank lines; the display keeps the item's content indent), so
    kramdown renders it as display math rather than folding it into the
    paragraph."""
    lines = markdown.split("\n")
    roles = fence_roles(lines)
    in_code = indented_code_test(markdown)
    blocks: dict[int, int] = {}  # first line -> last line of a whole-line display
    for seg in segments(markdown):
        if (
            seg.kind != "display"
            or not _fills_lines(markdown, seg.start, seg.end)
            or in_code(seg.start)
        ):
            continue
        if _unreadable(markdown, seg.start, seg.end) and not _is_block_display(
            markdown, seg.start, seg.end
        ):
            continue
        blocks[markdown.count("\n", 0, seg.start)] = markdown.count("\n", 0, seg.end)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        last = blocks.get(i)
        if last is None or roles[i] != "prose":
            out.append(line)
            i += 1
            continue
        block = lines[i : last + 1]
        quote = QUOTE_PREFIX.match(line)
        prefix = quote.group(0) if quote else line[: len(line) - len(line.lstrip())]
        rest = line[len(prefix) :]
        if quote and not all(QUOTE_PREFIX.match(other) for other in block):
            quote = None  # a lazy continuation, not a quoted block
        if not (quote or prefix):
            out.extend(block)
            i = last + 1
            continue
        if not quote:
            # A list continuation: only at the item's own content column.
            column = list_column(lines, i)
            if column is None or len(prefix.expandtabs(4)) != column:
                out.extend(block)
                i = last + 1
                continue
        else:
            # Glued under a list item that goes on inside the quote: at the
            # item's column, so the list does not split around it.
            lo, hi = max(0, i - _MAX_SCAN), min(len(lines), last + 1 + _MAX_SCAN)
            inner = [QUOTE_PREFIX.sub("", other) for other in lines[lo:hi]]
            column = list_column(inner, i - lo)
            goes_on = _LIST_MARKER.match(
                next((t for t in inner[last + 1 - lo :] if t.strip()), "")
            )
            if (
                column
                and goes_on
                and len(rest.expandtabs(4)) - len(rest.lstrip()) < column
            ):
                block = [
                    prefix + " " * column + QUOTE_PREFIX.sub("", b).lstrip()
                    for b in block
                ]
        separator = prefix.rstrip() if quote else ""
        if _has_text(lines, i - 1, bool(quote)):
            out.append(separator)
        out.extend(block)
        if _has_text(lines, last + 1, bool(quote)) and not _ATTR_LINE.match(
            "\n" + lines[last + 1]
        ):
            out.append(separator)
        i = last + 1
    return "\n".join(out)


def _has_text(lines: list[str], j: int, quoted: bool) -> bool:
    """Whether line ``j`` exists and holds text (after quote markers)."""
    if not 0 <= j < len(lines):
        return False
    other = QUOTE_PREFIX.sub("", lines[j]) if quoted else lines[j]
    return bool(other.strip())


def list_column(lines: list[str], i: int) -> int | None:
    """Content column of the list item line ``i`` continues, if any: back
    over its glued lines, and over blank lines when everything after them is
    indented to the item's content (an indented heading or paragraph)."""
    crossed_blank, least = False, None
    for line in reversed(lines[max(0, i - _MAX_SCAN) : i]):
        if not line.strip():
            crossed_blank = True
            continue
        item = _LIST_MARKER.match(line)
        if item:
            column = len(item.group(0).expandtabs(4))
            if crossed_blank and (least is None or least < column):
                return None
            return column
        if not line[:1].isspace():
            return None
        width = indent_width(line)
        least = width if least is None else min(least, width)
    return None


_DISPLAY_START = re.compile(r"^[ \t]*(?:\$\$|\\{1,2}\[)")
_NOT_PROSE_START = ("#", "|", ">")
# A thematic break (***, ---, ___) or a setext underline (===): never part of
# a run moved into a list item.
_RULE_LINE = re.compile(r"^[ \t]*(?:([-*_])(?:[ \t]*\1){2,}|=+)[ \t]*$")
# Lines a glued run or a multi-line display is followed over at most (a
# bound, so a pathological post stays linear).
_MAX_SCAN = 200


def _not_prose(line: str) -> bool:
    return line.lstrip().startswith(_NOT_PROSE_START) or bool(_RULE_LINE.match(line))


def _owning_items(lines: list[str], i: int) -> list[int]:
    """The list item lines that line ``i`` is glued under (no blank line
    between): the nearest first, then the items enclosing it, back over
    indented continuation lines."""
    found: list[int] = []
    least = None
    for j in range(i - 1, max(-1, i - 1 - _MAX_SCAN), -1):
        line = lines[j]
        if not line.strip() or line.lstrip().startswith(">"):
            break
        indent = indent_width(line)
        if _LIST_MARKER.match(line) and (least is None or indent < least):
            found.append(j)
            least = indent
        elif not line[:1].isspace():
            break
        if indent == 0:
            break
    return found


def _opens_display(line: str) -> bool:
    """Whether a line starts a whole-line display: one ``$$…$$`` or ``\\[…\\]``,
    or the first line of one over several (``$$`` alone or with the first
    part of the formula)."""
    stripped = line.strip()
    if stripped.startswith("$$"):
        return bool(_ONE_LINE_DISPLAY.match(line)) or stripped.count("$$") == 1
    return bool(_DISPLAY_START.match(line))


def _list_goes_on(item: str, other: str) -> bool:
    """Whether item line ``other`` continues ``item``'s list (same indent and
    marker kind; kramdown ignores the numbers) or opens a list nested in it."""
    a, b = _LIST_MARKER.match(item), _LIST_MARKER.match(other)
    if not (a and b):
        return False
    if indent_width(other) != indent_width(item):
        return indent_width(other) > indent_width(item)
    first, second = (m.group(0).strip()[0] for m in (a, b))
    return first == second or (first.isdigit() and second.isdigit())


def _display_close(lines: list[str], roles: list[str], i: int) -> int | None:
    """The line that closes the multi-line display opened at line ``i``: the
    first line after it with an odd ``$$`` count (``\\]`` for a bracket
    display), as kramdown pairs them. None when that line has text after the
    closer (an inline span wrapped over lines, ``$$.``), or a blank line or
    code comes first (inner lines may start with ``-`` or ``|``)."""
    bracket = "$$" not in lines[i]
    for k in range(i + 1, min(len(lines), i + 1 + _MAX_SCAN)):
        line = lines[k]
        stripped = line.strip()
        if roles[k] != "prose" or not stripped:
            return None
        closes = "\\]" in stripped if bracket else stripped.count("$$") % 2 == 1
        if closes:
            closer = "\\]" if bracket else "$$"
            return k if stripped.endswith(closer) else None
    return None


def _glued_run(
    lines: list[str], roles: list[str], i: int
) -> list[tuple[int, int]] | None:
    """The run glued under a list item from the display at line ``i`` down to
    the first blank line (blank lines inside a fence or a ``$$`` block do not
    end it) or the next list item: its chunks as ``(first, last)`` line
    ranges, or None when it holds a heading, table or quote."""
    chunks: list[tuple[int, int]] = []
    j = i
    while j < len(lines) and lines[j].strip():
        if j >= i + _MAX_SCAN:
            return None  # an unbounded run: left alone
        line = lines[j]
        if j > i and _LIST_MARKER.match(line):
            break
        if roles[j] == "open":
            k = j
            while (
                k + 1 < len(lines)
                and roles[k] != "close"
                and (k == j or roles[k] == "code")
            ):
                k += 1
            chunks.append((j, k))
            j = k + 1
            continue
        if _not_prose(line):
            return None
        if _opens_display(line) and not _ONE_LINE_DISPLAY.match(line):
            close = _display_close(lines, roles, j)
            if close is None:
                return None  # not a clean whole-line display: left alone
            chunks.append((j, close))
            j = close + 1
            continue
        chunks.append((j, j))
        j += 1
    return chunks


def _item_display_tail(
    lines: list[str], roles: list[str], i: int
) -> tuple[int, int] | None:
    """For a list item whose first line is a whole display (``1. $$x$$``, or
    ``1. $$`` … ``$$``), the glued paragraph lines after it that sit left of
    the item's content column, as ``(first, last)``; None otherwise."""
    marker = _LIST_MARKER.match(lines[i])
    if not marker:
        return None
    rest = lines[i][marker.end() :]
    if _ONE_LINE_DISPLAY.match(rest) and rest.strip().startswith("$$"):
        end = i
    elif rest.strip() == "$$":
        close = _display_close(lines, roles, i)
        if close is None:
            return None
        end = close
    else:
        return None
    column = _columns(marker.group(0))
    first = last = end + 1
    while (
        last < len(lines)
        and lines[last].strip()
        and roles[last] == "prose"
        and indent_width(lines[last]) < column
        and not _LIST_MARKER.match(lines[last])
        and not _not_prose(lines[last])
        and not _opens_display(lines[last])
    ):
        last += 1
    return (first, last - 1) if last > first else None


def _fold_glued_runs(markdown: str) -> str:
    """Keep a display glued under a list item (no blank line between) in the
    item. A whole-line display at the margin, or indented less than the
    item's content, would end the item and split the list (an ordered one
    restarting at 1). So the whole glued run from the display to the first
    blank line (explanations, more displays, fences) moves to the item's
    content column as one unit, with blank lines around its displays and
    fences. It applies when the list goes on after the run, or when the run
    is indented (already inside the item); otherwise the text is left for the
    validator to report."""
    lines = markdown.split("\n")
    roles = fence_roles(lines)
    # Lines of an indented code block (a list sample inside code) stay as written.
    code = indented_code_lines(lines)
    roles = ["code" if k in code else role for k, role in enumerate(roles)]
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if roles[i] == "prose" and (tail := _item_display_tail(lines, roles, i)):
            # A display that fills an item's first line, then a paragraph at
            # the margin: indent the paragraph into the item, or the list
            # ends there and the next step restarts at 1.
            column = _columns(_LIST_MARKER.match(line).group(0))  # type: ignore[union-attr]
            first, last = tail
            out.extend(lines[i:first])
            out.extend(" " * column + b.lstrip(" \t") for b in lines[first : last + 1])
            i = last + 1
            continue
        items = (
            [j for j in _owning_items(lines, i) if roles[j] == "prose"]
            if roles[i] == "prose" and _opens_display(line)
            else []
        )
        chunks = _glued_run(lines, roles, i) if items else None
        if not chunks:
            out.append(line)
            i += 1
            continue
        end = chunks[-1][1] + 1
        after = next((k for k in range(end, len(lines)) if lines[k].strip()), None)
        # The item whose list goes on after the run (an enclosing one when the
        # display follows a nested list), else the nearest one.
        going = [
            j
            for j in items
            if after is not None and _list_goes_on(lines[j], lines[after])
        ]
        goes_on = bool(going)
        item = going[0] if going else items[0]
        marker = _LIST_MARKER.match(lines[item])
        column = _columns(marker.group(0))  # type: ignore[union-attr]
        width = indent_width(line)
        if width >= column or not (goes_on or width > 0):
            out.append(line)
            i += 1
            continue
        pad = " " * column
        out.append("")
        previous = "block"  # the item's text before the run
        for first, last in chunks:
            block = lines[first : last + 1]
            kind = "text" if first == last and not _opens_display(block[0]) else "block"
            if roles[first] == "open":
                base = len(block[0]) - len(block[0].lstrip(" \t"))
                block = [
                    pad + b[min(base, len(b) - len(b.lstrip(" \t"))) :] for b in block
                ]
            else:
                block = [pad + b.lstrip(" \t") for b in block]
            attribute = block[0].lstrip().startswith("{:")
            if out[-1].strip() and (kind == "block" or previous == "block"):
                if not (attribute and previous == "block"):
                    out.append("")
            out.extend(block)
            previous = kind
        if after is not None and after == end:
            out.append("")
        i = end
    return "\n".join(out)


def normalize_math(markdown: str) -> str:
    """Rewrite math delimiters and escaping to the blog's ``$$`` convention."""
    text = _rewrite_spans(_fold_glued_runs(markdown))
    text = _isolate_display_blocks(text)
    text = _isolate_nested_displays(text)
    text = _vert_inline_bars(text)
    return _collapse_blank_runs(text)


# The blog lint's own reading of a body (script/lint-posts.rb), ported so the
# checks that mirror it see the same lines. Its fence (line 189) opens on a
# line starting with 3+ backticks or tildes after any whitespace: a fence
# inside a blockquote ("> ```") is no fence there, and an indented code block
# is prose, so math in either is checked like any other line.
BLOG_FENCE = re.compile(r"\A\s*(`{3,}|~{3,})")
# A code span, opened and closed by the same run of backticks (line 191).
BLOG_CODE_SPAN = re.compile(r"(`+).*?\1")
_BLOG_QUOTE = re.compile(r"\A\s*(?:>\s?)+")  # unquote (line 348)


def blog_unquote(line: str) -> str:
    """The text of a line after any blockquote markers (lint-posts.rb ``unquote``)."""
    return _BLOG_QUOTE.sub("", line, count=1)


def blog_bare(line: str) -> str:
    """A line as the blog lint reads its math: unquoted, code spans removed
    (lint-posts.rb line 261)."""
    return BLOG_CODE_SPAN.sub("", blog_unquote(line))


def blog_fence_roles(lines: list[str]) -> list[str]:
    """Each line's role as the blog lint's body loop assigns it
    (lint-posts.rb lines 241-280): ``"open"``, ``"code"`` or ``"close"`` (a
    fence's opener, a line inside it, its closer), ``"display"`` (a line of a
    ``$$`` display block opened on a line starting with ``$$``, delimiters
    included) or ``"prose"``.

    A fence marker starts its line (indentation only: one after ``>`` is
    text there) and closes only on a line of its own character, at least as
    long, with nothing else on it (line 250); an unclosed one runs to the end.
    """
    roles: list[str] = []
    fence: str | None = None
    block = False  # inside a multi-line display block
    para = False  # a paragraph with an odd number of $$ is open
    for line in lines:
        if m := BLOG_FENCE.match(line):
            if fence is None:
                fence = m.group(1)
                roles.append("open")
            elif re.fullmatch(f"{re.escape(fence[0])}{{{len(fence)},}}", line.strip()):
                fence = None
                roles.append("close")
            else:
                roles.append("code")
            continue
        if fence is not None:
            roles.append("code")
            continue
        bare = blog_bare(line)
        odd = bare.count("$$") % 2 == 1
        if para and not bare.strip():
            para = False
        if block or (odd and not para and bare.lstrip().startswith("$$")):
            if odd:
                block = not block
            roles.append("display")
            continue
        if odd:
            para = not para
        roles.append("prose")
    return roles


def blog_line_roles(lines: list[str]) -> list[str]:
    """:func:`blog_fence_roles` with a fence's lines all ``"fence"``."""
    return [
        "fence" if role in ("open", "code", "close") else role
        for role in blog_fence_roles(lines)
    ]


def blog_open_fence_line(lines: list[str]) -> int | None:
    """Index of the opener of a fence the blog lint finds never closed
    (:func:`blog_fence_roles`), if any."""
    opened: int | None = None
    for i, role in enumerate(blog_fence_roles(lines)):
        if role == "open":
            opened = i
        elif role == "close":
            opened = None
    return opened


def has_math(markdown: str) -> bool:
    """Whether the Markdown has ``$$`` outside code, as the blog lint decides
    a post needs ``use_math: true`` (lint-posts.rb lines 262 and 300): on any
    line outside its fences, code spans removed. Math in an indented code
    block or a fence inside a blockquote counts, as it does there."""
    lines = markdown.split("\n")
    return any(
        role != "fence" and "$$" in blog_bare(line)
        for line, role in zip(lines, blog_line_roles(lines), strict=True)
    )
