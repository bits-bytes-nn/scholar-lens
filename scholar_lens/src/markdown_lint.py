r"""Lint + auto-fix a published Markdown post for the blog's kramdown/MathJax quirks.

A companion to :mod:`markdown_math`. Where that module normalizes math
delimiters, this one catches the *other* recurring blog-rendering failures. It reuses the same
validated code/math-span splitter (:func:`markdown_math.segments`) so it
never mistakes code for prose or math.

Auto-fixes (deterministic, outside code):

* Remove leftover generation envelope tags (``<explanation>``, ``</summary>``…).
* Turn a body H1 into H2 (the post title comes from the front matter) and
  raise any heading that skips a level, so the outline never jumps.
* Callouts beyond :data:`MAX_CALLOUTS` become plain paragraphs.
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

Writing-rule slips (the paper's numbering, self-references, …) are listed by
:func:`writing_issues` for the editor and the PR checklist.

:func:`drop_unknown_images` is a separate pure helper: it removes image links
whose source is not a known figure of the artifact.

:func:`lint_markdown` returns the auto-fixed text and logs any warnings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import Literal

from .logger import logger
from .markdown_math import (
    BLOG_CODE_SPAN,
    BLOG_FENCE,
    HANGUL,
    IMAGE,
    QUOTE_PREFIX,
    blog_bare,
    blog_fence_roles,
    blog_line_roles,
    blog_unquote,
    fence_roles,
    indent_width,
    indented_code_bases,
    indented_code_lines,
    indented_code_test,
    list_column,
    normalize_math,
    open_fence,
    segments,
)
from .post_conventions import load_local_conventions

# Macros of non-standard packages that writers commonly emit; base MathJax
# does not load them, so they render as red raw text unless the blog defines
# them. We warn only on this small explicit set rather than allowlisting the
# hundreds of base-MathJax macros (which would mis-warn on valid ones); the
# ones the blog defines (post_conventions.yml math.defined_macros, the list
# _includes/head.html sets) are left out at run time.
_EXTENSION_MACROS = frozenset(
    {
        "coloneqq",  # mathtools
        "eqqcolon",  # mathtools
        "vcentcolon",  # mathtools
        "llbracket",  # stmaryrd
        "rrbracket",  # stmaryrd
        "textsc",  # text mode
    }
)


@cache
def undefined_macros() -> frozenset[str]:
    """The macros of :data:`_EXTENSION_MACROS` the blog's MathJax does not
    define, by the conventions file's ``math.defined_macros``."""
    defined = load_local_conventions().math.defined_macros
    return _EXTENSION_MACROS - set(defined)


# A bare "|" not already escaped or part of \vert/\mid/\| etc.
_BARE_PIPE = re.compile(r"(?<!\\)\|")
# Single-$ inline math: a $...$ that is not part of a $$ display block. Used only
# to warn, never to rewrite (could be currency).
_SINGLE_DOLLAR = re.compile(r"(?<![\\$])\$(?!\$)(?!\s)[^$\n]+?(?<!\\)\$(?!\$)")
# Markdown links: capture the target. Skip image links (handled elsewhere).
_MD_LINK = re.compile(r"(?<!\!)\[(?:[^\]]*)\]\(([^)]*)\)")
# LaTeX control words (\foo); \\ and single-char controls (\,, \{) are ignored.
_MACRO = re.compile(r"\\([a-zA-Z]+)")
# A heading line that needs a blank line before it.
_HEADING = re.compile(r"^#{1,6}[ \t]")
_HEADING_LINE = re.compile(r"^(#{1,6})[ \t]")
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
ENVELOPE_TAG = re.compile(rf"</?(?:{'|'.join(ENVELOPE_TAGS)})>")


# What can follow a Korean numbered reference: the end, a space or punctuation,
# or a particle. Anything else continues a word ("그림 3장", "3 절감", "1 절전").
# 가/도/만 are left out on purpose: after a number they read as counters
# ("2가지", "3도", "2만").
_KO_REF_END = r"(?!\w)|(?=에서|에|의|은|는|을|를|이|과|와|로|으로|까지|부터|보다|처럼)"

# The paper's numbering of figures/tables/equations/sections, in prose. A bare
# "<n>장" is a counter ("GPU 2장", "그림 3장"), so only a dotted number counts
# as a chapter; "지표 2" is not "표 2".
NUMBERED_REF = re.compile(
    r"\b(?:Figure|Fig\.|Table|Section|Sec\.|Eq\.|Equation|Algorithm|Theorem|Lemma"
    r"|Proposition|Corollary)\s*\(?\d+"
    rf"|(?<![가-힣])(?:그림|표|수식|식|알고리즘)\s*(?:\(\d+\)|\d+[a-z]?(?:{_KO_REF_END}))"
    rf"|\d+(?:\.\d+)*\s*절(?:{_KO_REF_END})|\d+(?:\.\d+)+장(?:{_KO_REF_END})"
    r"|§\s*\d+"
)
_LATEX_LIKE = re.compile(r"[\\^_{]")

# Real inline HTML tags a post may legitimately use; these must NOT be escaped.
# Everything else that looks like <word ...> in prose is treated as a
# placeholder token (e.g. <none>, <server-host>, <your-namespace>) that kramdown
# would otherwise parse as an HTML open tag, emitting a stray closing tag later.
# The HTML a post writes, as the blog lint knows it (script/lint-posts.rb
# ``HTML_ELEMENTS``, lines 231-237): any other ``<name>`` is passed through by kramdown and
# dropped by the browser, so a prompt token (<think>, <input>, <s>) would
# vanish. The last row is an inline <svg>'s children.
BLOG_HTML_ELEMENTS = frozenset(
    """a abbr audio b blockquote br caption center cite code col colgroup dd del
    details dfn div dl dt em figcaption figure h1 h2 h3 h4 h5 h6 hr i iframe img
    ins kbd li mark ol p picture pre q samp script section small source span
    strong sub summary sup svg table tbody td tfoot th thead time tr track u ul
    var video wbr
    circle defs g line path polygon polyline rect text tspan use""".split()
)
# Elements that mean nothing bare: <img> or <a> alone is a token, not markup.
BLOG_NEEDS_ATTRIBUTES = frozenset("a audio iframe img script source video".split())
# An angle-bracket token: an opening/closing/self-closing tag-like run with no
# embedded '<'/'>' or line break and no whitespace-only body. group(1) = optional '/', group(2)
# = the leading name token (letters/digits/_-). We only escape when that name is
# not a real HTML tag, so <br>, <sup>, </table> stay intact but <none>,
# <your-namespace>, <server-host> get escaped.
_ANGLE_TOKEN = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9._-]*)[^<>\n]*?/?>")
# GFM autolinks — ``<https://…>`` / ``<mailto:…>`` / ``<user@host>`` — are valid
# Markdown that kramdown renders as links; escaping them produces dead literal
# text. Recognise and leave them alone (distinct from ``<placeholder>`` tokens).
_AUTOLINK = re.compile(r"<(?:[A-Za-z][A-Za-z0-9+.-]*://[^<>\s]+|[^<>\s@]+@[^<>\s@]+)>")
# Constructs that render poorly on the blog's MathJax: the standalone amsmath
# display environments (use \begin{aligned} inside $$...$$ instead) and \bm (use
# \boldsymbol). Warn-only — fixing requires understanding the equation.
_FRAGILE_MATH = re.compile(r"\\begin\{(align|equation|gather)\*?\}|\\bm\b")


# Math span kinds of the splitter ($$…$$, \(…\), \[…\]).
_MATH_KINDS = ("display", "paren", "bracket")


def _classify_regions(markdown: str) -> list[tuple[int, int, str]]:
    """Return (start, end, kind) for every code/math span, kind in
    {fence, icode, display, paren, bracket}. Used to know which offsets are 'protected'
    (code) or 'math' so prose-only checks skip them."""
    return [(seg.start, seg.end, seg.kind) for seg in segments(markdown)]


TABLE_ROW = re.compile(r"^[ \t]*\|")
TABLE_DELIMITER = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$"
)


def _separate_tables(markdown: str) -> str:
    """Blank lines around a pipe table glued to text above or below it:
    kramdown reads a table run into a paragraph (or a paragraph run into
    it) as plain text with literal pipes. A table glued under a list item is
    indented to the item's content so the list does not end there. Code and
    quotes are left alone."""
    lines = markdown.split("\n")
    roles = fence_roles(lines)
    code = indented_code_lines(lines)

    def row(k: int) -> bool:
        return (
            0 <= k < len(lines)
            and roles[k] == "prose"
            and k not in code
            and bool(TABLE_ROW.match(lines[k]))
        )

    out: list[str] = []
    i = 0
    while i < len(lines):
        if not (row(i) and row(i + 1) and TABLE_DELIMITER.match(lines[i + 1])):
            out.append(lines[i])
            i += 1
            continue
        end = i + 2
        while row(end):
            end += 1
        before = out[-1] if out else ""
        if before.strip() and not before.lstrip().startswith(">"):
            out.append("")
        rows = lines[i:end]
        column = list_column(lines, i) if before.strip() else None
        if column is not None and indent_width(lines[i]) < column:
            rows = [" " * column + r.lstrip(" \t") for r in rows]  # stay in the item
        out.extend(rows)
        after = lines[end] if end < len(lines) else ""
        # "{: .class}" belongs to the table it follows.
        if after.strip() and not after.lstrip().startswith((">", "{:")):
            out.append("")
        i = end
    return "\n".join(out)


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


# A chat-template token ("<|im_start|>"): in prose its "|"s split the line
# into a table, so it is set as inline code.
_CHAT_TOKEN = re.compile(r"<\|[\w.-]+\|>")


def _code_chat_tokens(markdown: str) -> str:
    regions = _classify_regions(markdown)
    protected = [(s, e) for s, e, k in regions if k in ("fence", "icode", *_MATH_KINDS)]
    in_code = indented_code_test(markdown)

    def repl(m: re.Match[str]) -> str:
        if any(s <= m.start() < e for s, e in protected) or in_code(m.start()):
            return m.group(0)
        return f"`{m.group(0)}`"

    return _CHAT_TOKEN.sub(repl, markdown)


def _escape_placeholder_tags(markdown: str, regions: list[tuple[int, int, str]]) -> str:
    """Escape bare ``<placeholder>`` tokens in prose so kramdown doesn't treat
    them as HTML.

    Tokens like ``<none>``, ``<server-host>``, ``<your-namespace>`` in body text
    are parsed by kramdown as HTML open tags; since they're never closed, it
    emits a stray ``</...>`` at the next block boundary, corrupting the render.
    We escape the angle brackets (``<`` -> ``&lt;``, ``>`` -> ``&gt;``) for any
    ``<name ...>`` whose name isn't a real HTML tag (:data:`BLOG_HTML_ELEMENTS`).
    Code spans and math spans are skipped, so shell redirs, generics, and
    ``\\langle`` stay untouched.
    """
    protected = [(s, e) for s, e, k in regions if k in ("fence", "icode", *_MATH_KINDS)]
    in_code = indented_code_test(markdown)  # an indented code block

    def repl(m: re.Match[str]) -> str:
        name = m.group(2).lower()
        bare = not m.group(1) and not m.group(0)[len(m.group(2)) + 1 : -1].strip(" /")
        if name in BLOG_HTML_ELEMENTS and not (bare and name in BLOG_NEEDS_ATTRIBUTES):
            return m.group(0)
        return m.group(0).replace("<", "&lt;").replace(">", "&gt;")

    out: list[str] = []
    pos = 0
    for m in _ANGLE_TOKEN.finditer(markdown):
        # Skipped when any part of it is code or math: a "k<n" in prose must
        # not pair with the ">" of a later callout or inline code.
        if any(s < m.end() and m.start() < e for s, e in protected) or any(
            in_code(i) for i in (m.start(), m.end() - 1)
        ):
            continue
        # A valid GFM autolink (URL/email) must render as a link, not be escaped;
        # a Markdown-escaped "\\<name\\>" already renders as text.
        if (
            _AUTOLINK.fullmatch(m.group(0))
            or markdown[m.start() - 1 : m.start()] == "\\"
        ):
            continue
        out.append(markdown[pos : m.start()])
        out.append(repl(m))
        pos = m.end()
    out.append(markdown[pos:])
    return "".join(out)


def _fills_its_lines(markdown: str, start: int, end: int) -> bool:
    line_start = markdown.rfind("\n", 0, start) + 1
    line_end = markdown.find("\n", end)
    line_end = len(markdown) if line_end == -1 else line_end
    return not (markdown[line_start:start] + markdown[end:line_end]).strip()


def _warn_pipes_in_math(markdown: str, regions: list[tuple[int, int, str]]) -> None:
    """Warn about a bare ``|`` left inside an inline math span (kramdown may
    read it as a table-cell delimiter). ``normalize_math`` already rewrote the
    rest to ``\\vert`` (the same symbol, also after ``\\left``/``\\right``);
    what remains sits in ``\\text{}`` or an ``array{c|c}`` column spec, where
    the fix depends on intent. A display block on its own lines is safe."""
    for ms, me, kind in regions:
        if kind not in _MATH_KINDS or _fills_its_lines(markdown, ms, me):
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
        if kind not in _MATH_KINDS:
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
        if kind not in _MATH_KINDS:
            continue
        for mm in _MACRO.finditer(markdown[ms:me]):
            if mm.group(1) in undefined_macros():
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


# The URLs the blog's htmlproofer fails (.github/workflows/jekyll.yml runs it
# with enforce_https, its default): an <a>/<link> href or an <img> src or
# srcset with the http scheme, or protocol-relative ("//host"). In Markdown
# those are inline link and image targets, reference definitions (not
# footnotes), autolinks and the attributes of raw HTML; text in code is no
# link, and a bare URL in prose is not one either (kramdown does not autolink).
_INSECURE_URL = re.compile(
    r"(?P<lead>\]\([ \t]*<?|^[ \t]{0,3}\[(?!\^)[^\]\n]+\]:[ \t]*<?"
    r"|(?<![\w-])(?:href|src)[ \t]*=[ \t]*[\"']?[ \t]*|<(?=http:))"
    r"(?P<url>(?:http:)?//[^\s\"'<>)]*)",
    re.IGNORECASE,
)
_SRCSET = re.compile(
    r"(?<![\w-])srcset[ \t]*=[ \t]*([\"'])(?P<value>.*?)\1", re.IGNORECASE
)
_SRCSET_URL = re.compile(r"(?P<lead>(?:^|,)[ \t]*)(?P<url>(?:http:)?//\S*)", re.I)


def _https(url: str) -> str:
    return "https://" + re.sub(r"^(?:http:)?//", "", url, flags=re.IGNORECASE)


def _upgrade_in_text(text: str) -> str:
    def srcset(m: re.Match[str]) -> str:
        value = _SRCSET_URL.sub(
            lambda u: u.group("lead") + _https(u.group("url")), m.group("value")
        )
        return m.group(0).replace(m.group("value"), value, 1)

    text = _INSECURE_URL.sub(lambda m: m.group("lead") + _https(m.group("url")), text)
    # A link that shows its own URL ("[http://x](https://x)") shows the new one.
    text = _SELF_LABEL.sub(
        lambda m: (
            f"[{m.group('target')}]"
            if m.group("label")[len("http://") :]
            == m.group("target")[len("https://") :]
            else m.group(0)
        ),
        text,
    )
    return _SRCSET.sub(srcset, text)


_SELF_LABEL = re.compile(
    r"\[(?P<label>http://[^\]\s]+)\](?=\((?P<target>https://[^)\s]+)\))"
)


def _code_split(line: str) -> list[tuple[str, bool]]:
    """``line`` as (text, is_code) pieces around its code spans."""
    pieces, pos = [], 0
    for m in BLOG_CODE_SPAN.finditer(line):
        pieces += [(line[pos : m.start()], False), (m.group(0), True)]
        pos = m.end()
    return [*pieces, (line[pos:], False)]


def _link_lines(markdown: str) -> list[int]:
    """Indices of the lines kramdown renders (not fenced or indented code)."""
    lines = markdown.split("\n")
    code = indented_code_lines(lines)
    roles = fence_roles(lines)
    return [i for i in range(len(lines)) if roles[i] == "prose" and i not in code]


def insecure_links(line: str) -> list[str]:
    """The http:// and protocol-relative link targets of one rendered line
    (code spans excluded) that the blog's htmlproofer fails."""
    found: list[str] = []
    for text, is_code in _code_split(line):
        if is_code:
            continue
        found += [m.group("url") for m in _INSECURE_URL.finditer(text)]
        for m in _SRCSET.finditer(text):
            found += [u.group("url") for u in _SRCSET_URL.finditer(m.group("value"))]
    return found


def upgrade_http_links(markdown: str) -> str:
    """Rewrite every link target htmlproofer fails (:func:`insecure_links`)
    to ``https://`` on the same host and path, outside code. Text in code
    (``curl http://localhost:8000``) is left as written."""
    lines = markdown.split("\n")
    for i in _link_lines(markdown):
        if "//" in lines[i]:
            lines[i] = "".join(
                text if is_code else _upgrade_in_text(text)
                for text, is_code in _code_split(lines[i])
            )
    return "\n".join(lines)


def _outside(regions: list[tuple[int, int, str]], kinds: tuple[str, ...]):  # type: ignore[no-untyped-def]
    spans = [(s, e) for s, e, k in regions if k in kinds]
    return lambda pos: not any(s <= pos < e for s, e in spans)


def _remove_envelope_tags(markdown: str) -> str:
    keep = _outside(_classify_regions(markdown), ("fence", "icode"))
    return ENVELOPE_TAG.sub(
        lambda m: m.group(0) if not keep(m.start()) else "",
        markdown,
    )


def _level_headings(markdown: str, *, warn: bool = True) -> str:
    """Body headings (outside code) on valid levels: an H1 becomes H2 (other
    levels are kept, so the post still opens with ``## TL;DR``), and a heading
    more than one level below the previous one is raised to the next level."""
    fences = [(s, e) for s, e, k in _classify_regions(markdown) if k == "fence"]
    lines = markdown.split("\n")
    out, offset, previous = [], 0, 1
    for line in lines:
        match = _HEADING_LINE.match(line)
        if match and not any(s <= offset < e for s, e in fences):
            hashes = len(match.group(1))
            level = min(max(hashes, 2), previous + 1)
            if warn and level != hashes:
                logger.warning(
                    "Markdown lint: heading '%s' moved to level %d.", line, level
                )
            out.append("#" * level + line[hashes:])
            previous = level
        else:
            out.append(line)
        offset += len(line) + 1
    return "\n".join(out)


def close_open_fence(text: str) -> str:
    """Close a code fence a writer left open at the end of its section, so the
    rest of the post is not read as code."""
    opener = open_fence(text)
    return f"{text.rstrip()}\n{opener}" if opener else text


def _file_name(source: str) -> str:
    return source.split("?", 1)[0].split("#", 1)[0].rsplit("/", 1)[-1]


def _host(source: str) -> str:
    """The host of a URL; "" for a local or relative path."""
    match = re.match(r"^(?:[a-z][a-z0-9+.-]*:)?//([^/?#]+)", source, re.I)
    return match.group(1).lower() if match else ""


# arXiv figure names (x1.png) are generic: on arXiv the paper id, not just the
# host, says whose figure it is (any version).
_ARXIV_PAPER_ID = re.compile(r"/(\d{4}\.\d{4,5})(?:v\d+)?/")


def _paper_id(source: str) -> str:
    match = _ARXIV_PAPER_ID.search(source)
    return match.group(1) if match else ""


def drop_unknown_images(markdown: str, known_sources: set[str]) -> str:
    """Remove image links whose source is not a known figure.

    ``known_sources`` holds the artifact's real figure paths/URLs; a link is
    kept when its source is in that set, or when its file name is and it is a
    local/relative path; on the same host as that figure (for arXiv, the same
    paper in any version) it is rewritten to the known source. A same-named
    image on another host is not the figure.
    Anything else is a hallucinated image (e.g. a paper's HTML page used as an
    image) and would render as a broken image.
    """
    known_names: dict[str, dict[tuple[str, str], str]] = {}
    for source in sorted(known_sources):
        name = _file_name(source)
        known_names.setdefault(name, {}).setdefault(
            (_host(source), _paper_id(source)), source
        )
    keep = _outside(_classify_regions(markdown), ("fence", "icode"))

    def repl(m: re.Match[str]) -> str:
        src = m.group(2)
        if not keep(m.start()) or src in known_sources:
            return m.group(0)
        origins = known_names.get(_file_name(src))
        if origins is not None and not _host(src):
            return m.group(0)
        known = (origins or {}).get((_host(src), _paper_id(src)))
        if known is not None:
            # The same figure under another version or path: point at the
            # source that was actually fetched (another one may 404).
            head, sep, tail = m.group(0).partition("](")
            return head + sep + tail.replace(src, known, 1)
        logger.warning("Dropping image with unknown source: %r", src[:100])
        return ""

    return IMAGE.sub(repl, markdown)


_CAPTION_LABEL = re.compile(r"^\s*(?:Figure|Fig\.|Table|그림|표)\s*\d+\s*[:.]\s*", re.I)


# The figure marker writers see in their sources (quoted fields; the parser's
# attach_figure_context writes it). One parser, shared with review_plan.
FIGURE_MARKER = re.compile(
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

    return FIGURE_MARKER.sub(repl, markdown)


def _alt_text(text: str, limit: int = 160) -> str:
    """``text`` flattened into safe alt text, cut at a word boundary."""
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+|[*`{}]", "", text)
    text = " ".join(text.split()).replace("[", "(").replace("]", ")")
    text = text.replace("|", "/")  # a bare | would split a table cell
    if len(text) > limit:
        text = text[: limit + 1].rsplit(" ", 1)[0]
    return text.strip(" ,;:")


def _section_heading_before(markdown: str, start: int) -> str:
    """The nearest heading above ``start`` as plain words (emoji, symbols and
    a question mark removed)."""
    prose = _masked(markdown[:start], ("fence",))
    headings = re.findall(r"^#{2,4}[ \t]+(.+)$", prose, re.MULTILINE)
    if not headings:
        return ""
    words = re.sub(r"[^\w\s·:,()'-]", " ", headings[-1])
    return " ".join(words.split()).strip(" :,")


def fill_image_alts(
    markdown: str, captions: dict[str, str], fallback: str, *, korean: bool
) -> str:
    """Give every image usable alt text.

    In a Korean post (``korean``: the post's language) an alt without Hangul
    (the parser's captions are English) becomes the nearest section heading.
    Otherwise an empty alt gets the figure's caption (without its "Figure 3:"
    label) when the source is known, else ``fallback``.
    """
    keep = _outside(_classify_regions(markdown), ("fence", "icode"))

    def repl(m: re.Match[str]) -> str:
        alt, src = m.group(1).strip(), m.group(2)
        if not keep(m.start()):
            return m.group(0)
        if korean and not HANGUL.search(alt):
            heading = _section_heading_before(markdown, m.start())
            if HANGUL.search(heading):
                return f"![{_alt_text(heading, 80)}]({src}{m.group(3)})"
        if alt:
            return m.group(0)
        caption = captions.get(src) or captions.get(src.rsplit("/", 1)[-1]) or ""
        text = _CAPTION_LABEL.sub("", caption).strip() or fallback
        return f"![{_alt_text(text)}]({src}{m.group(3)})"

    return IMAGE.sub(repl, markdown)


# An English word used as a Korean verb or adjective: its stem followed by a
# 하다/되다 form ("prefill하면서", "swap-in되어", "attention하도록"); the
# particle 한테 ("Claude Code한테") is not one.
_ENGLISH_STEM_ENDING = re.compile(
    r"(?<![A-Za-z0-9_./-])([A-Za-z][A-Za-z-]*)"
    r"(?!한테)([하한할함합해했되된될됨됩됐돼][가-힣]*)"
)
# "장" and "섹션" also name chapters and config-file sections ("다음 섹션을
# 추가", "다음 섹션에 키를 추가"), so "섹션" counts only with 에서/은/는 and
# the object/subject endings only count for "절".
_SELF_REFERENCE = re.compile(
    r"(?<![가-힣])(?:이|앞|첫|다음|이번)\s*(?:절(?=을|이|가)|섹션(?=에서|은|는)"
    r"|(?:절|장)(?=에서|에|은|는|의|과|까지|부터|[\s.,)]))"
    r"|\b(?:[Tt]his|[Tt]he (?:previous|next)) (?:section|chapter)\b"
)
# A labelled callout: "> **핵심:** …" or "> **핵심**: …" (the colon is what
# separates a label from a bold phrase that starts a quote).
_CALLOUT = re.compile(
    r"^>[ \t]*(?:\*\*[^*\n]{1,30}:\*\*|\*\*[^*\n]{1,30}\*\*:)", re.MULTILINE
)
_CALLOUT_LABEL = re.compile(r"\A(?:\*\*[^*\n]{1,30}:\*\*|\*\*[^*\n]{1,30}\*\*:)[ \t]*")
MAX_CALLOUTS = 3
# Rule hits listed on a PR checklist (the rest are the same kinds of slip).
MAX_WRITING_NOTES = 10


_LINK_TARGET = re.compile(
    r"!\[[^\]\n]*\]\([^)\n]*\)|\]\([^)\n]*\)|<[A-Za-z][A-Za-z0-9+.-]*://[^<>\s]*>|https?://\S+"
    r"|<[A-Za-z][^<>\n]*>"  # raw HTML tags (attribute text is not prose)
)


def _blank(m: re.Match[str]) -> str:
    return " " * len(m.group(0))


def _masked(markdown: str, kinds: tuple[str, ...]) -> str:
    """``markdown`` with the given span kinds blanked out (offsets and line
    breaks kept)."""
    out = markdown
    for s, e, kind in _classify_regions(markdown):
        if kind in kinds:
            out = out[:s] + re.sub(r"[^\n]", " ", out[s:e]) + out[e:]
    return out


def _snippet(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start : line_end if line_end != -1 else len(text)]
    a, b = max(0, start - line_start - 30), end - line_start + 30
    return " ".join(line[a:b].split())


# A reader-facing note about the writer's own sources ("(미검증)", "제공된 자료에는
# 나와 있지 않습니다"): the pipeline's doubt leaking into the post instead of the
# claim being fixed or dropped.
_VERIFICATION_HEDGE = re.compile(
    r"미검증|\(\s*unverified\s*\)|\bnot (?:yet )?verified\b"
    r"|(?:공식 문서|자료|출처)(?:에서는|에서|에는|에)\s*(?:는\s*)?확인되지 않"
    r"|(?:자료|문서|출처)(?:에는|에서는|에서|에)\s*(?:나와 있지|나오지) 않"
)


# The blog lint's prose rules (script/lint-posts.rb lines 241-259), with its
# patterns and exemptions as they are, so a post passes here exactly when it
# passes there. Translationese with no natural reading (prompts/base.py
# KOREAN_STYLE): a double passive ("되어진", "보여지는") and the calque "에 있어"
# where a particle or a nominalized verb shows it ("에 있어서는", "학습함에
# 있어"). A bare "에 있어서" (also "in" in Korean) and patterns that are normal
# Korean in moderation ("에 대한", "~라는 점") are left to the LLM checkers.
TRANSLATIONESE = re.compile(  # lint-posts.rb line 245
    r"되어[지진져집]|보여[지진져]|에 있어(?:서[는도의](?! 안)|[는의])|[음함]에 있어(?![야요])"
)
# A sentence ending glued to a colon ("다음과 같습니다:"): end the sentence.
SENTENCE_COLON = re.compile(r"다:")  # line 247
_QUOTED_SPAN = re.compile(r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|「[^」\n]*」')  # line 248
_REFERENCES_HEADING = re.compile(  # line 249
    r"\A#{1,6}\s+(?:references|bibliography|참고 ?자료|참고 ?문헌)\b",
    re.IGNORECASE,
)
_ALT_IMAGE = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")  # line 256
_DOLLAR_SPAN = re.compile(r"\$\$.*?\$\$")  # line 317


@dataclass(frozen=True)
class StyleHit:
    """One prose-rule hit: the body line (0-based), the rule and the match."""

    line: int
    rule: Literal["translationese", "colon"]
    match: str


def style_hits(body: str) -> list[StyleHit]:
    """Translationese and colon-ended sentences where the blog lint looks for
    them (lint-posts.rb ``prose``, lines 251-259, called from line 322): prose
    lines outside its fences and display blocks, not a heading, a
    blockquote, a table row (indented under a list too) or a line of the
    reference list; quoted spans, image syntax (alt text), code spans and
    ``$$`` spans removed."""
    hits: list[StyleHit] = []
    lines = body.split("\n")
    in_references = False
    for i, (line, role) in enumerate(zip(lines, blog_line_roles(lines), strict=True)):
        if role != "prose":
            continue
        if line.startswith("#"):  # line 312
            in_references = bool(_REFERENCES_HEADING.match(line))
        if (
            in_references
            or line.startswith("#")
            or line.lstrip().startswith((">", "|"))
        ):
            continue
        text = _DOLLAR_SPAN.sub("", blog_bare(line))
        text = _ALT_IMAGE.sub("", _QUOTED_SPAN.sub("", text))
        for rule, pattern in (
            ("translationese", TRANSLATIONESE),
            ("colon", SENTENCE_COLON),
        ):
            hits += [
                StyleHit(i, rule, m.group(0))  # type: ignore[arg-type]
                for m in pattern.finditer(text)
            ]
    return hits


def writing_issues(markdown: str) -> list[str]:
    """Deterministic writing-rule violations in a generated post, one line
    each with the offending text, for an editor or reviser to fix.

    Covers references to the paper's numbering, an English stem with a Korean
    verb ending ("fine-tune합니다"), the post referring to its own parts
    ("이 절에서"), reader-facing verification hedges ("(미검증)") and
    single-``$`` math. Code and math are ignored; image alt text and callouts
    are fixed in post-processing instead. The blog's prose rules
    (:func:`style_hits`) are publish-blocking body errors
    (:func:`post_validator.body_errors`), so they are not repeated here.
    """
    # Link and image targets are copied verbatim (a file named "Figure1.png"
    # is not a reference the writer can reword).
    prose = _masked(markdown, ("fence", "icode", *_MATH_KINDS))
    issues: list[str] = []
    # A copied internal figure marker: its caption's numbering is not prose.
    if marker := FIGURE_MARKER.search(prose):
        issues.append(
            "copied figure marker (write a Markdown image ![alt](src) instead): "
            f"{marker.group(0)[:80]}"
        )
        prose = FIGURE_MARKER.sub(_blank, prose)
    prose = _LINK_TARGET.sub(_blank, prose)
    for m in NUMBERED_REF.finditer(prose):
        issues.append(
            f"numbered reference '{m.group(0)}' (refer descriptively): "
            f"{_snippet(markdown, m.start(), m.end())}"
        )
    for m in _ENGLISH_STEM_ENDING.finditer(prose):
        # An acronym used as a verb ("SVD하면") is standard Korean usage.
        if not m.group(1).isupper():
            issues.append(
                f"English stem with a Korean ending '{m.group(0)}': "
                f"{_snippet(markdown, m.start(), m.end())}"
            )
    for m in _SELF_REFERENCE.finditer(prose):
        issues.append(
            f"self-reference '{m.group(0).strip()}': "
            f"{_snippet(markdown, m.start(), m.end())}"
        )
    for m in _VERIFICATION_HEDGE.finditer(prose):
        issues.append(
            "verification hedge (fix the claim from an official source or drop it): "
            f"{_snippet(markdown, m.start(), m.end())}"
        )
    code = _masked(markdown, ("fence", "icode", "display"))
    for m in _SINGLE_DOLLAR.finditer(code):
        if _LATEX_LIKE.search(m.group(0)):
            issues.append(f"single-$ math '{m.group(0)[:60]}' (use $$ … $$)")
    return list(dict.fromkeys(issues))


def _issue_kind(issue: str) -> str:
    return re.split(r" '| \(|:", issue, maxsplit=1)[0]


def writing_notes(markdown: str) -> list[str]:
    """:func:`writing_issues` for a PR checklist: the first
    :data:`MAX_WRITING_NOTES`, then one line counting the rest by kind."""
    issues = writing_issues(markdown)
    if len(issues) <= MAX_WRITING_NOTES:
        return issues
    rest = issues[MAX_WRITING_NOTES:]
    kinds = ", ".join(dict.fromkeys(_issue_kind(i) for i in rest))
    return [
        *issues[:MAX_WRITING_NOTES],
        f"…and {len(rest)} more writing-rule hit(s) (kinds: {kinds})",
    ]


def callout_lines(markdown: str) -> list[str]:
    """The labelled callout lines of ``markdown`` (without the ``>``)."""
    keep = _outside(_classify_regions(markdown), ("fence", "icode"))
    return [
        m.group(0).lstrip("> \t")
        for m in re.finditer(r"^.*$", markdown, re.MULTILINE)
        if _CALLOUT.match(m.group(0)) and keep(m.start())
    ]


def limit_callouts(markdown: str) -> str:
    """Keep the first :data:`MAX_CALLOUTS` callouts (``> **핵심:** …``); later
    ones become plain paragraphs (the ``>`` markers and the leading bold label
    go, every other word stays). Code is untouched."""
    keep = _outside(_classify_regions(markdown), ("fence", "icode"))
    starts = [m for m in _CALLOUT.finditer(markdown) if keep(m.start())]
    for m in reversed(starts[MAX_CALLOUTS:]):
        end = markdown.find("\n\n", m.end())
        end = len(markdown) if end == -1 else end
        block = markdown[m.start() : end]
        plain = re.sub(r"^>[ \t]?", "", block, flags=re.MULTILINE)
        plain = _CALLOUT_LABEL.sub("", plain, count=1)
        markdown = markdown[: m.start()] + plain + markdown[end:]
    return markdown


def lint_markdown(markdown: str, *, warn: bool = True) -> str:
    """Auto-fix the safe blog-rendering issues and warn about risky ones.

    Auto-fixed: leftover envelope tags removed; callouts beyond
    :data:`MAX_CALLOUTS` become paragraphs; a body H1 becomes H2 and level
    jumps are raised; a blank line before a heading glued to the previous line, and around a
    pipe table glued to text;
    bare ``<placeholder>`` angle-bracket tokens in prose escaped so kramdown
    doesn't emit stray HTML closing tags; chat-template tokens (``<|im_start|>``)
    set as inline code; http:// and protocol-relative link targets upgraded
    to https:// (:func:`upgrade_http_links`). Warned (left untouched, since
    a rewrite could ship a new bug): bare ``|`` in math, fragile amsmath envs /
    ``\\bm``, single-$ inline math, known-undefined macros, and non-URL link
    targets (``warn=False`` skips them). Returns the fixed Markdown.
    """
    fixed = _remove_envelope_tags(markdown)
    fixed = limit_callouts(fixed)
    fixed = _level_headings(fixed, warn=warn)
    fixed = _fix_headings(fixed)
    fixed = _separate_tables(fixed)
    fixed = _code_chat_tokens(fixed)
    fixed = upgrade_http_links(fixed)

    # Escape bare <placeholder> tokens before warnings run so region offsets are
    # computed against the final text. Re-classify since escaping shifts offsets.
    fixed = _escape_placeholder_tags(fixed, _classify_regions(fixed))

    if not warn:
        return fixed
    regions = _classify_regions(fixed)
    _warn_pipes_in_math(fixed, regions)
    _warn_fragile_math(fixed, regions)
    _warn_single_dollar(fixed, regions)
    _warn_undefined_macros(fixed, regions)
    _warn_non_url_links(fixed, regions)
    return fixed


_FENCE_INFO = re.compile(r"^(\s*(?:>\s?)*\s*(?:`{3,}|~{3,}))\s*(\S+)\s+\S.*$")


def _bare_fence_info(markdown: str) -> str:
    """Keep only the language word of a fence's info string: kramdown's GFM
    fence takes one word, and with more (```python title="a.py") it renders
    the block as inline code."""
    lines = markdown.split("\n")
    for i, role in enumerate(fence_roles(lines)):
        if role == "open" and (m := _FENCE_INFO.match(lines[i])):
            lines[i] = m.group(1) + m.group(2)
    return "\n".join(lines)


_RAW_TAG = re.compile(r"(\{%-?\s*(?:end)?raw\s*-?%\})")
_LIQUID = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{\{|\{%")


def _wrap_liquid(markdown: str) -> str:
    """Wrap Liquid-looking text in ``{% raw %}``: Jekyll renders a post as
    Liquid first, so a prompt template's ``{{ question }}`` (even in code)
    would render as nothing. A code block gets raw lines around it; anything
    else is wrapped where it stands. Already-raw text is left alone."""
    lines = markdown.split("\n")
    roles = fence_roles(lines)
    out: list[str] = []
    i = 0
    while i < len(lines):
        if roles[i] == "open":
            end = i + 1
            while end < len(lines) and roles[end] == "code":
                end += 1
            block = lines[i : end + 1]
            before = out[-1].strip() if out else ""
            needs = any(_LIQUID.search(line) for line in block[1:-1] or block)
            if needs and not re.fullmatch(r"\{%-?\s*raw\s*-?%\}", before):
                indent = re.match(r"[ \t>]*", lines[i]).group(0)  # type: ignore[union-attr]
                out += [f"{indent}{{% raw %}}", *block, f"{indent}{{% endraw %}}"]
            else:
                out += block
            i = end + 1
            continue
        out.append(lines[i])
        i += 1
    raw = False
    wrapped: list[str] = []
    for line in out:
        parts = []
        for part in _RAW_TAG.split(line):
            tag = re.fullmatch(r"\{%-?\s*((?:end)?raw)\s*-?%\}", part)
            if tag:
                raw = tag.group(1) == "raw"
            elif not raw:
                part = _LIQUID.sub(
                    lambda m: "{% raw %}" + m.group(0) + "{% endraw %}", part
                )
            parts.append(part)
        wrapped.append("".join(parts))
    return "\n".join(wrapped)


def _fence_indented_code(markdown: str) -> str:
    """Indented code blocks as fenced ones, which kramdown renders the same:
    the blog lint reads an indented block as prose, so a ``<tag>`` or ``$$``
    in a sample there would fail it. The fence sits at the block's base
    column (its list item's content) and the code keeps its own indent."""
    lines = markdown.split("\n")
    bases = indented_code_bases(lines)
    roles = fence_roles(lines)
    code = {i: base for i, base in bases.items() if roles[i] == "prose"}
    if not code:
        return markdown
    out: list[str] = []
    i = 0
    while i < len(lines):
        if i not in code:
            out.append(lines[i])
            i += 1
            continue
        base, end = code[i], i
        j = i + 1
        while j < len(lines) and (j in code or not lines[j].strip()):
            if j in code:
                end = j
            j += 1
        block = [
            line.expandtabs(4)[base + 4 :] if line.strip() else ""
            for line in lines[i : end + 1]
        ]
        runs = [len(m.group(0)) for m in re.finditer(r"`{3,}", "\n".join(block))]
        fence = "`" * max([3, *(n + 1 for n in runs)])
        indent = " " * base
        out += [
            f"{indent}{fence}",
            *(f"{indent}{line}" if line else "" for line in block),
            f"{indent}{fence}",
        ]
        i = end + 1
    return "\n".join(out)


def close_quoted_fences(markdown: str) -> str:
    """Close each fence a blockquote opened and left open, at the end of its
    blockquote (before the blank line that ends it, or at the end of the
    text), with the opener's own quote markers and fence: both lints read a
    bare closer there as prose, so ``close_open_fence`` cannot fix it."""
    lines = markdown.split("\n")
    roles = blog_fence_roles(lines)
    closers: dict[int, str] = {}
    opener: str | None = None
    for i, role in enumerate(roles):
        if role == "open":
            prefix = QUOTE_PREFIX.match(lines[i])
            fence = BLOG_FENCE.match(blog_unquote(lines[i]))
            opener = (
                f"{prefix.group(0).rstrip()} {fence.group(1)}"
                if prefix and fence
                else None
            )
        elif role == "close":
            opener = None
        elif role == "unclosed" and opener:
            closers[i] = opener
            opener = None
    if opener and roles and roles[-1] == "code":
        closers[len(lines)] = opener
    for i in sorted(closers, reverse=True):
        lines.insert(i, closers[i])
    return "\n".join(lines)


def publish_body(markdown: str, *, warn: bool = True) -> str:
    """The publisher's body pipeline, in one place: an open fence closed
    (a quoted one where its blockquote ends),
    callouts over the cap unboxed (before the math pass, so a display left in
    an unboxed callout is isolated too), the blog math rules applied, then the
    other auto-fixes of :func:`lint_markdown`, indented code blocks fenced,
    fence info strings cut to the language and Liquid-looking text wrapped
    in ``{% raw %}``. Idempotent."""
    fenced = _fence_indented_code(close_open_fence(close_quoted_fences(markdown)))
    linted = lint_markdown(
        normalize_math(limit_callouts(_bare_fence_info(fenced))),
        warn=warn,
    )
    return _wrap_liquid(linted)
