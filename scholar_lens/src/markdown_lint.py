r"""Lint + auto-fix a published Markdown post for the blog's kramdown/MathJax quirks.

A companion to :mod:`markdown_math`. Where that module escapes underscores, this
one catches the *other* recurring blog-rendering failures. It reuses the same
validated code/math-span splitter (:data:`markdown_math.SEGMENT_PATTERN`) so it
never mistakes code for prose or math.

Two auto-fixes are applied — both genuinely lossless:

* **Auto-fix** (deterministic, lossless): insert the blank line kramdown needs
  before a heading glued to the previous line (outside fenced code).
* **Auto-fix** (deterministic, lossless): escape bare ``<placeholder>`` angle-
  bracket tokens in prose (e.g. ``<none>``, ``<your-namespace>``,
  ``<server-host>``) so kramdown does not parse them as HTML open tags and emit
  a stray ``</...>`` closing tag at the next block boundary. Real inline HTML
  (``<br>``, ``<sup>``, ``<kbd>``, …) and code/math spans are left untouched.

Everything else is a **warning** (logged, never rewritten — a wrong rewrite
would ship a new bug):

* A bare ``|`` inside a math span (kramdown may read it as a table delimiter).
  NOT auto-rewritten to ``\vert``: that breaks ``\left|``/``\right|``/``\middle|``
  and ``array{c|c}`` column specs, where the ``|`` must stay a literal delimiter.
* A single-``$ ... $`` inline math span (the blog strips ``$`` delimiters, so it
  won't render — but it could also be a currency amount).
* A LaTeX macro from a known problematic non-standard package (e.g. stmaryrd's
  ``\llbracket``) that the blog does NOT define, which renders as red raw text.
* A Markdown link whose target is not an http(s)/site-relative URL (likely a
  broken cross-reference).

:func:`lint_markdown` returns the auto-fixed text and logs any warnings.
"""

from __future__ import annotations

import re

from .logger import logger
from .markdown_math import SEGMENT_PATTERN

# Non-standard-package macros that are commonly emitted but are NOT loaded on the
# blog's MathJax, so they render as red raw text. We warn ONLY on this small,
# explicit "known-bad-unless-defined" set rather than trying to allowlist the
# hundreds of base-MathJax macros (which would mis-warn on valid ones). Drop a
# macro from here once the blog adds a \newcommand/package for it.
_UNDEFINED_MACROS = frozenset(
    {
        "llbracket",  # stmaryrd
        "rrbracket",  # stmaryrd
        "textsc",  # not in MathJax core
        "coloneqq",  # mathtools
        "eqqcolon",  # mathtools
        "vcentcolon",  # mathtools
    }
)

# A bare "|" not already escaped or part of \vert/\mid/\| etc.
_BARE_PIPE = re.compile(r"(?<!\\)\|")
# Single-$ inline math: a $...$ that is not part of a $$ display block. Used only
# to warn, never to rewrite (could be currency).
_SINGLE_DOLLAR = re.compile(r"(?<!\$)\$(?!\$)(?!\s)[^$\n]+?\$(?!\$)")
# Markdown links: capture the target. Skip image links (handled elsewhere).
_MD_LINK = re.compile(r"(?<!\!)\[(?:[^\]]*)\]\(([^)]*)\)")
# LaTeX control words (\foo); \\ and single-char controls (\,, \{) are ignored.
_MACRO = re.compile(r"\\([a-zA-Z]+)")
# A heading line that needs a blank line before it.
_HEADING = re.compile(r"^#{1,6}[ \t]")

# Real inline HTML tags a post may legitimately use; these must NOT be escaped.
# Everything else that looks like <word ...> in prose is treated as a
# placeholder token (e.g. <none>, <server-host>, <your-namespace>) that kramdown
# would otherwise parse as an HTML open tag, emitting a stray closing tag later.
_REAL_HTML_TAGS = frozenset(
    {
        "a",
        "abbr",
        "b",
        "br",
        "code",
        "dd",
        "del",
        "details",
        "div",
        "dl",
        "dt",
        "em",
        "figure",
        "figcaption",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "i",
        "img",
        "ins",
        "kbd",
        "li",
        "mark",
        "ol",
        "p",
        "pre",
        "q",
        "s",
        "samp",
        "small",
        "span",
        "strong",
        "sub",
        "summary",
        "sup",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "u",
        "ul",
        "var",
    }
)
# An angle-bracket token: an opening/closing/self-closing tag-like run with no
# embedded '<'/'>' and no whitespace-only body. group(1) = optional '/', group(2)
# = the leading name token (letters/digits/_-). We only escape when that name is
# not a real HTML tag, so <br>, <sup>, </table> stay intact but <none>,
# <your-namespace>, <server-host> get escaped.
_ANGLE_TOKEN = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9._-]*)[^<>]*?/?>")
# Constructs that render poorly on the blog's MathJax: the standalone amsmath
# display environments (use \begin{aligned} inside $$...$$ instead) and \bm (use
# \boldsymbol). Warn-only — fixing requires understanding the equation.
_FRAGILE_MATH = re.compile(r"\\begin\{(align|equation|gather)\*?\}|\\bm\b")


def _classify_regions(markdown: str) -> list[tuple[int, int, str]]:
    """Return (start, end, kind) for every code/math span, kind in
    {fence, icode, display, inline}. Used to know which offsets are 'protected'
    (code) or 'math' so prose-only checks skip them."""
    return [
        (m.start(), m.end(), m.lastgroup or "")
        for m in SEGMENT_PATTERN.finditer(markdown)
    ]


def _fix_headings(markdown: str) -> str:
    """Insert a blank line before any heading glued to the previous line.

    kramdown only recognises a heading when it is preceded by a blank line; a
    heading stuck to the previous paragraph gets absorbed into it. Fenced code
    blocks are skipped so a shell comment like ``# do this`` is never touched.
    """
    fences = [(s, e) for s, e, k in _classify_regions(markdown) if k == "fence"]

    def in_fence(pos: int) -> bool:
        return any(s <= pos < e for s, e in fences)

    lines = markdown.split("\n")
    out: list[str] = []
    offset = 0
    for line in lines:
        line_start = offset
        offset += len(line) + 1  # +1 for the '\n'
        if (
            _HEADING.match(line)
            and not in_fence(line_start)
            and out
            and out[-1].strip() != ""
        ):
            out.append("")
        out.append(line)
    return "\n".join(out)


def _escape_placeholder_tags(markdown: str, regions: list[tuple[int, int, str]]) -> str:
    """Escape bare ``<placeholder>`` tokens in prose so kramdown doesn't treat
    them as HTML.

    Tokens like ``<none>``, ``<server-host>``, ``<your-namespace>`` in body text
    are parsed by kramdown as HTML open tags; since they're never closed, it
    emits a stray ``</...>`` at the next block boundary, corrupting the render.
    We escape the angle brackets (``<`` -> ``&lt;``, ``>`` -> ``&gt;``) for any
    ``<name ...>`` whose name isn't a real HTML tag (:data:`_REAL_HTML_TAGS`).
    Code spans and math spans are skipped, so shell redirs, generics, and
    ``\\langle`` stay untouched.
    """
    protected = [
        (s, e) for s, e, k in regions if k in ("fence", "icode", "display", "inline")
    ]

    def repl(m: re.Match[str]) -> str:
        if m.group(2).lower() in _REAL_HTML_TAGS:
            return m.group(0)
        return m.group(0).replace("<", "&lt;").replace(">", "&gt;")

    out: list[str] = []
    pos = 0
    for m in _ANGLE_TOKEN.finditer(markdown):
        if any(s <= m.start() < e for s, e in protected):
            continue
        out.append(markdown[pos : m.start()])
        out.append(repl(m))
        pos = m.end()
    out.append(markdown[pos:])
    return "".join(out)


def _warn_pipes_in_math(markdown: str, regions: list[tuple[int, int, str]]) -> None:
    """Warn about a bare ``|`` inside a math span (kramdown may read it as a
    table-cell delimiter). NOT auto-rewritten: ``\\vert`` would be wrong after
    ``\\left``/``\\right``/``\\middle`` and inside ``array{c|c}`` column specs, so
    the safe fix is context-dependent and left to a human."""
    for ms, me, kind in regions:
        if kind not in ("display", "inline"):
            continue
        if _BARE_PIPE.search(markdown[ms:me]):
            logger.warning(
                "Markdown lint: bare '|' inside math %r — kramdown may read it as "
                "a table delimiter. Use \\vert / \\mid (or \\left|...\\right| for "
                "delimiters) so it renders correctly.",
                markdown[ms:me][:60],
            )


def _warn_fragile_math(markdown: str, regions: list[tuple[int, int, str]]) -> None:
    """Warn about amsmath display environments / ``\\bm`` that the blog's MathJax
    renders poorly. NOT auto-rewritten (the right replacement depends on the
    equation); the generator prompts are told to avoid these."""
    seen: set[str] = set()
    for ms, me, kind in regions:
        if kind not in ("display", "inline"):
            continue
        for m in _FRAGILE_MATH.finditer(markdown[ms:me]):
            seen.add(m.group(0))
    if seen:
        logger.warning(
            "Markdown lint: math uses %s — these render poorly on the blog's "
            "MathJax. Use \\begin{aligned}...\\end{aligned} inside $$...$$ and "
            "\\boldsymbol instead of \\bm.",
            sorted(seen),
        )


def _warn_single_dollar(markdown: str, regions: list[tuple[int, int, str]]) -> None:
    code = [(s, e) for s, e, k in regions if k in ("fence", "icode")]
    for m in _SINGLE_DOLLAR.finditer(markdown):
        if any(s <= m.start() < e for s, e in code):
            continue
        logger.warning(
            "Markdown lint: single-$ inline math %r will NOT render on the blog "
            "(use \\( ... \\)); left as-is in case it is a currency amount.",
            m.group(0)[:60],
        )


def _warn_undefined_macros(markdown: str, regions: list[tuple[int, int, str]]) -> None:
    """Warn when math uses a macro from the known-problematic set (non-standard
    packages the blog doesn't load), which renders as red raw text. We check this
    small explicit set rather than allowlisting all of base MathJax."""
    seen: set[str] = set()
    for ms, me, kind in regions:
        if kind not in ("display", "inline"):
            continue
        for mm in _MACRO.finditer(markdown[ms:me]):
            if mm.group(1) in _UNDEFINED_MACROS:
                seen.add(mm.group(1))
    if seen:
        logger.warning(
            "Markdown lint: math uses macro(s) %s the blog's MathJax does not "
            "define — they will render as red raw text. Replace with a base "
            "MathJax equivalent.",
            sorted(seen),
        )


def _warn_non_url_links(markdown: str, regions: list[tuple[int, int, str]]) -> None:
    code = [(s, e) for s, e, k in regions if k in ("fence", "icode")]
    for m in _MD_LINK.finditer(markdown):
        if any(s <= m.start() < e for s, e in code):
            continue
        target = m.group(1).strip()
        if not target:
            continue
        if not target.startswith(("http://", "https://", "/", "#")):
            logger.warning(
                "Markdown lint: link target %r is not an http(s) URL (nor a site-"
                "relative path) — likely a broken link; use plain prose for "
                "internal cross-references.",
                target[:80],
            )


def lint_markdown(markdown: str) -> str:
    """Auto-fix the safe blog-rendering issues and warn about risky ones.

    Auto-fixed (lossless): blank line before a heading glued to the previous
    line; bare ``<placeholder>`` angle-bracket tokens in prose escaped so
    kramdown doesn't emit stray HTML closing tags. Warned (left untouched, since
    a rewrite could ship a new bug): bare ``|`` in math, fragile amsmath envs /
    ``\\bm``, single-$ inline math, known-undefined macros, and non-URL link
    targets. Returns the fixed Markdown.
    """
    fixed = _fix_headings(markdown)

    # Escape bare <placeholder> tokens before warnings run so region offsets are
    # computed against the final text. Re-classify since escaping shifts offsets.
    fixed = _escape_placeholder_tags(fixed, _classify_regions(fixed))

    regions = _classify_regions(fixed)
    _warn_pipes_in_math(fixed, regions)
    _warn_fragile_math(fixed, regions)
    _warn_single_dollar(fixed, regions)
    _warn_undefined_macros(fixed, regions)
    _warn_non_url_links(fixed, regions)
    return fixed
