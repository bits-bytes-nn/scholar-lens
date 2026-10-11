r"""Lint + auto-fix a published Markdown post for the blog's kramdown/MathJax quirks.

A companion to :mod:`markdown_math`. Where that module normalizes math
delimiters, this one catches the *other* recurring blog-rendering failures. It reuses the same
validated code/math-span splitter (:data:`markdown_math.SEGMENT_PATTERN`) so it
never mistakes code for prose or math.

Auto-fixes (deterministic, outside code):

* Remove leftover generation envelope tags (``<explanation>``, ``</summary>``…).
* Demote every heading one level when the body contains an H1: the post title
  comes from the front matter, so a body H1 is always a generation slip.
* Insert the blank line kramdown needs before a heading glued to the previous
  line.
* **Auto-fix** (deterministic, lossless): escape bare ``<placeholder>`` angle-
  bracket tokens in prose (e.g. ``<none>``, ``<your-namespace>``,
  ``<server-host>``) so kramdown does not parse them as HTML open tags and emit
  a stray ``</...>`` closing tag at the next block boundary. Real inline HTML
  (``<br>``, ``<sup>``, ``<kbd>``, …) and code/math spans are left untouched.

Everything else is a **warning** (logged, never rewritten — a wrong rewrite
would ship a new bug):

* A bare ``|`` inside a math span (kramdown may read it as a table delimiter).
  NOT auto-rewritten: ``\left|``/``array{c|c}`` need the literal ``|``.
* A single-``$ ... $`` span that looks like LaTeX (only ``$$`` renders; it may
  also be a currency amount, so it is not rewritten).
* A LaTeX macro from a non-standard package the blog does NOT define.
* A Markdown link whose target is not an http(s)/site-relative URL.
* A reference to the paper's own numbering ("Figure 3", "Table 2", "3.2절"),
  which does not exist in the post.

:func:`drop_unknown_images` is a separate pure helper: it removes image links
whose source is not a known figure of the artifact.

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
# (\llbracket, \rrbracket and \textsc are defined by the blog's head.html.)
_UNDEFINED_MACROS = frozenset(
    {
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
_H1 = re.compile(r"^#[ \t]")
# Wrapper tags of the generation prompts; never valid in a post.
ENVELOPE_TAGS: tuple[str, ...] = (
    "explanation",
    "has_more",
    "key_takeaways",
    "closing",
    "subtitle",
    "summary",
    "tags",
    "urls",
    "controlled_tags",
    "section_markdown",
    "grounded_markdown",
    "edit",
    "find",
    "replace",
)
_ENVELOPE_TAG = re.compile(rf"</?(?:{'|'.join(ENVELOPE_TAGS)})>")
# The paper's numbering of figures/tables/equations/sections, in prose.
NUMBERED_REF = _NUMBERED_REF = re.compile(
    r"\b(?:Figure|Fig\.|Table|Section|Sec\.|Eq\.|Equation|Algorithm)\s*\(?\d+"
    r"|(?:그림|표|수식|알고리즘)\s*\d+|\d+(?:\.\d+)*\s*(?:절|장)\b"
)
_LATEX_LIKE = re.compile(r"[\\^_{]")
_IMAGE_LINK = re.compile(r"!\[([^\]]*)\]\(([^()\s]*(?:\([^()]*\)[^()\s]*)*)\)")

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
# GFM autolinks — ``<https://…>`` / ``<mailto:…>`` / ``<user@host>`` — are valid
# Markdown that kramdown renders as links; escaping them produces dead literal
# text. Recognise and leave them alone (distinct from ``<placeholder>`` tokens).
_AUTOLINK = re.compile(r"<(?:[A-Za-z][A-Za-z0-9+.-]*://[^<>\s]+|[^<>\s@]+@[^<>\s@]+)>")
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
        # A valid GFM autolink (URL/email) must render as a link, not be escaped.
        if _AUTOLINK.fullmatch(m.group(0)):
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
                "a table delimiter. Use \\lvert ... \\rvert or \\mid instead.",
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
    code = [(s, e) for s, e, k in regions if k in ("fence", "icode", "display")]
    for m in _SINGLE_DOLLAR.finditer(markdown):
        if any(s <= m.start() < e for s, e in code):
            continue
        if not _LATEX_LIKE.search(m.group(0)):
            continue  # most likely a currency amount
        logger.warning(
            "Markdown lint: single-$ math %r will NOT render on the blog (use "
            "$$ ... $$); left as-is in case it is a currency amount.",
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


def _outside(regions: list[tuple[int, int, str]], kinds: tuple[str, ...]):  # type: ignore[no-untyped-def]
    spans = [(s, e) for s, e, k in regions if k in kinds]
    return lambda pos: not any(s <= pos < e for s, e in spans)


def _remove_envelope_tags(markdown: str) -> str:
    keep = _outside(_classify_regions(markdown), ("fence", "icode"))
    return _ENVELOPE_TAG.sub(lambda m: "" if keep(m.start()) else m.group(0), markdown)


def _demote_headings_if_h1(markdown: str) -> str:
    """Rewrite body H1 headings (outside code) as H2; other levels are kept so
    the post still opens with ``## TL;DR``."""
    fences = [(s, e) for s, e, k in _classify_regions(markdown) if k == "fence"]
    lines = markdown.split("\n")
    offsets, offset = [], 0
    for line in lines:
        offsets.append(offset)
        offset += len(line) + 1

    def in_fence(i: int) -> bool:
        return any(s <= offsets[i] < e for s, e in fences)

    if not any(_H1.match(line) and not in_fence(i) for i, line in enumerate(lines)):
        return markdown
    logger.warning("Markdown lint: body contains an H1; rewriting it as '##'.")
    return "\n".join(
        "#" + line if _H1.match(line) and not in_fence(i) else line
        for i, line in enumerate(lines)
    )


def _warn_numbered_refs(markdown: str, regions: list[tuple[int, int, str]]) -> None:
    keep = _outside(regions, ("fence", "icode", "display"))
    found = sorted(
        {m.group(0) for m in _NUMBERED_REF.finditer(markdown) if keep(m.start())}
    )
    if found:
        logger.warning(
            "Markdown lint: prose cites the paper's numbering %s, which does not "
            "exist in the post; refer to it descriptively.",
            found[:10],
        )


def drop_unknown_images(markdown: str, known_sources: set[str]) -> str:
    """Remove image links whose source is not a known figure.

    ``known_sources`` holds the artifact's real figure paths/URLs; a link is
    kept when its source, or its file name, is in that set. Anything else is a
    hallucinated image (e.g. a paper's HTML page used as an image) and would
    render as a broken image.
    """
    known_names = {s.rsplit("/", 1)[-1] for s in known_sources}
    keep = _outside(_classify_regions(markdown), ("fence", "icode"))

    def repl(m: re.Match[str]) -> str:
        src = m.group(2)
        if not keep(m.start()) or src in known_sources:
            return m.group(0)
        if src.rsplit("/", 1)[-1] in known_names:
            return m.group(0)
        logger.warning("Dropping image with unknown source: %r", src[:100])
        return ""

    return _IMAGE_LINK.sub(repl, markdown)


_CAPTION_LABEL = re.compile(r"^\s*(?:Figure|Fig\.|Table|그림|표)\s*\d+\s*[:.]\s*", re.I)


_FIGURE_MARKER = re.compile(
    r'\[Image:\s*alt="(?P<alt>[^"]*)",\s*src="(?P<src>[^"]+)"'
    r'(?:,\s*caption="(?P<caption>[^"]*)")?\]'
)
_PLACEHOLDER_ALTS = {"", "refer to caption", "image", "figure"}


def figure_markers_to_images(markdown: str) -> str:
    """Turn any internal ``[Image: alt=…, src=…, caption=…]`` marker a writer
    copied verbatim into a Markdown image (alt from the alt text, or the
    caption's first sentence when the alt is a placeholder)."""
    keep = _outside(_classify_regions(markdown), ("fence", "icode"))

    def repl(m: re.Match[str]) -> str:
        if not keep(m.start()):
            return m.group(0)
        alt = m.group("alt").strip()
        if alt.lower() in _PLACEHOLDER_ALTS:
            caption = (m.group("caption") or "").strip()
            alt = re.split(r"(?<=[.!?])\s", caption, maxsplit=1)[0]
        alt = " ".join(alt.split())[:160].replace("[", "(").replace("]", ")")
        alt = alt.replace("|", "/")  # a bare | would split a table cell
        image = f"![{alt}]({m.group('src')})"
        line_start = markdown.rfind("\n", 0, m.start()) + 1
        if markdown[line_start:].lstrip().startswith("|"):
            return image  # inside a table row: stay on the row
        return f"\n\n{image}\n\n"

    return _FIGURE_MARKER.sub(repl, markdown)


def fill_image_alts(markdown: str, captions: dict[str, str], fallback: str) -> str:
    """Give every image alt text: the figure's caption (without its "Figure 3:"
    label) when the source is known, else ``fallback``."""
    keep = _outside(_classify_regions(markdown), ("fence", "icode"))

    def repl(m: re.Match[str]) -> str:
        alt, src = m.group(1).strip(), m.group(2)
        if alt or not keep(m.start()):
            return m.group(0)
        caption = captions.get(src) or captions.get(src.rsplit("/", 1)[-1]) or ""
        text = _CAPTION_LABEL.sub("", caption).strip() or fallback
        text = " ".join(text.split())[:160].replace("[", "(").replace("]", ")")
        return f"![{text}]({src})"

    return _IMAGE_LINK.sub(repl, markdown)


def lint_markdown(markdown: str) -> str:
    """Auto-fix the safe blog-rendering issues and warn about risky ones.

    Auto-fixed (lossless): blank line before a heading glued to the previous
    line; bare ``<placeholder>`` angle-bracket tokens in prose escaped so
    kramdown doesn't emit stray HTML closing tags. Warned (left untouched, since
    a rewrite could ship a new bug): bare ``|`` in math, fragile amsmath envs /
    ``\\bm``, single-$ inline math, known-undefined macros, and non-URL link
    targets. Returns the fixed Markdown.
    """
    fixed = _remove_envelope_tags(markdown)
    fixed = _demote_headings_if_h1(fixed)
    fixed = _fix_headings(fixed)

    # Escape bare <placeholder> tokens before warnings run so region offsets are
    # computed against the final text. Re-classify since escaping shifts offsets.
    fixed = _escape_placeholder_tags(fixed, _classify_regions(fixed))

    regions = _classify_regions(fixed)
    _warn_pipes_in_math(fixed, regions)
    _warn_fragile_math(fixed, regions)
    _warn_single_dollar(fixed, regions)
    _warn_undefined_macros(fixed, regions)
    _warn_non_url_links(fixed, regions)
    _warn_numbered_refs(fixed, regions)
    return fixed
