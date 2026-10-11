r"""Lint + auto-fix a published Markdown post for the blog's kramdown/MathJax quirks.

A companion to :mod:`markdown_math`. Where that module normalizes math
delimiters, this one catches the *other* recurring blog-rendering failures. It reuses the same
validated code/math-span splitter (:data:`markdown_math.SEGMENT_PATTERN`) so it
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
_ENVELOPE_TAG = re.compile(rf"</?(?:{'|'.join(ENVELOPE_TAGS)})>")
# The paper's numbering of figures/tables/equations/sections, in prose. A bare
# "<n>장" is a counter ("GPU 2장", "그림 3장"), so only a dotted number counts
# as a chapter; "지표 2" is not "표 2"; "3 절감" is not a section.
NUMBERED_REF = re.compile(
    r"\b(?:Figure|Fig\.|Table|Section|Sec\.|Eq\.|Equation|Algorithm|Theorem|Lemma"
    r"|Proposition|Corollary)\s*\(?\d+"
    r"|(?<![가-힣])(?:그림|표|수식|식|알고리즘)\s*(?:\(\d+\)|\d+(?![\d\s]*[개장가]))"
    r"|\d+(?:\.\d+)*\s*절(?![감대단편충댓반차약])|\d+(?:\.\d+)+장"
    r"|§\s*\d+"
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


# Math span kinds of the splitter ($$…$$, \(…\), \[…\]).
_MATH_KINDS = ("display", "paren", "bracket")


def _classify_regions(markdown: str) -> list[tuple[int, int, str]]:
    """Return (start, end, kind) for every code/math span, kind in
    {fence, icode, display, paren, bracket}. Used to know which offsets are 'protected'
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
    protected = [(s, e) for s, e, k in regions if k in ("fence", "icode", *_MATH_KINDS)]

    def repl(m: re.Match[str]) -> str:
        if m.group(2).lower() in _REAL_HTML_TAGS:
            return m.group(0)
        return m.group(0).replace("<", "&lt;").replace(">", "&gt;")

    out: list[str] = []
    pos = 0
    for m in _ANGLE_TOKEN.finditer(markdown):
        if any(s <= m.start() < e for s, e in protected):
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


def _warn_pipes_in_math(markdown: str, regions: list[tuple[int, int, str]]) -> None:
    """Warn about a bare ``|`` inside a math span (kramdown may read it as a
    table-cell delimiter). NOT auto-rewritten: ``\\vert`` would be wrong after
    ``\\left``/``\\right``/``\\middle`` and inside ``array{c|c}`` column specs, so
    the safe fix is context-dependent and left to a human."""
    for ms, me, kind in regions:
        if kind not in _MATH_KINDS:
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


def _level_headings(markdown: str) -> str:
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
            if level != hashes:
                logger.warning(
                    "Markdown lint: heading '%s' moved to level %d.", line, level
                )
            out.append("#" * level + line[hashes:])
            previous = level
        else:
            out.append(line)
        offset += len(line) + 1
    return "\n".join(out)


_FENCE_LINE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")


def close_open_fence(text: str) -> str:
    """Close a code fence a writer left open at the end of its section, so the
    rest of the post is not read as code (CommonMark: a closing fence is the
    same character, at least as long, with nothing after it)."""
    opener = ""
    for line in text.splitlines():
        match = _FENCE_LINE.match(line)
        if not match:
            continue
        run, rest = match.group(1), match.group(2).strip()
        if not opener:
            if not (run[0] == "`" and "`" in rest):
                opener = run
        elif run[0] == opener[0] and len(run) >= len(opener) and not rest:
            opener = ""
    return f"{text.rstrip()}\n{opener}" if opener else text


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


def fill_image_alts(markdown: str, captions: dict[str, str], fallback: str) -> str:
    """Give every image usable alt text.

    In a mostly-Hangul post an alt without Hangul (the parser's captions are
    English) becomes the nearest section heading. Otherwise an empty alt gets the figure's
    caption (without its "Figure 3:" label) when the source is known, else
    ``fallback``.
    """
    keep = _outside(_classify_regions(markdown), ("fence", "icode"))
    korean = mostly_hangul(markdown)

    def repl(m: re.Match[str]) -> str:
        alt, src = m.group(1).strip(), m.group(2)
        if not keep(m.start()):
            return m.group(0)
        if korean and not _HANGUL.search(alt):
            heading = _section_heading_before(markdown, m.start())
            if _HANGUL.search(heading):
                return f"![{_alt_text(heading, 80)}]({src})"
        if alt:
            return m.group(0)
        caption = captions.get(src) or captions.get(src.rsplit("/", 1)[-1]) or ""
        text = _CAPTION_LABEL.sub("", caption).strip() or fallback
        return f"![{_alt_text(text)}]({src})"

    return _IMAGE_LINK.sub(repl, markdown)


_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_ENGLISH_STEM_ENDING = re.compile(
    r"(?<![A-Za-z0-9_./-])([A-Za-z][A-Za-z-]*)"
    r"(합니다|하는|하여|하고|하면|한|할|했습니다|했다|된다|됩니다|된|되는)(?![가-힣])"
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


def mostly_hangul(markdown: str) -> bool:
    """True when the prose is Korean: Hangul syllables outweigh Latin letters
    (one syllable ≈ two letters); code, math and link targets excluded."""
    prose = _masked(markdown, ("fence", "icode", *_MATH_KINDS))
    prose = re.sub(r"\]\([^)]*\)", "]", prose)
    hangul = len(_HANGUL.findall(prose))
    return hangul > 0 and hangul * 2 > len(_LATIN.findall(prose))


def _snippet(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start : line_end if line_end != -1 else len(text)]
    a, b = max(0, start - line_start - 30), end - line_start + 30
    return " ".join(line[a:b].split())


def writing_issues(markdown: str) -> list[str]:
    """Deterministic writing-rule violations in a generated post, one line
    each with the offending text, for an editor or reviser to fix.

    Covers references to the paper's numbering, an English stem with a Korean
    verb ending ("fine-tune합니다"), the post referring to its own parts
    ("이 절에서") and single-``$`` math. Code and math are ignored; image alt text
    and callouts are fixed in post-processing instead.
    """
    # Link and image targets are copied verbatim (a file named "Figure1.png"
    # is not a reference the writer can reword).
    prose = _masked(markdown, ("fence", "icode", *_MATH_KINDS))
    issues: list[str] = []
    # A copied internal figure marker: its caption's numbering is not prose.
    if marker := _FIGURE_MARKER.search(prose):
        issues.append(
            "copied figure marker (write a Markdown image ![alt](src) instead): "
            f"{marker.group(0)[:80]}"
        )
        prose = _FIGURE_MARKER.sub(_blank, prose)
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


def lint_markdown(markdown: str) -> str:
    """Auto-fix the safe blog-rendering issues and warn about risky ones.

    Auto-fixed: leftover envelope tags removed; callouts beyond
    :data:`MAX_CALLOUTS` become paragraphs; a body H1 becomes H2 and level
    jumps are raised; a blank line before a heading glued to the previous line;
    bare ``<placeholder>`` angle-bracket tokens in prose escaped so kramdown
    doesn't emit stray HTML closing tags. Warned (left untouched, since
    a rewrite could ship a new bug): bare ``|`` in math, fragile amsmath envs /
    ``\\bm``, single-$ inline math, known-undefined macros, and non-URL link
    targets. Returns the fixed Markdown.
    """
    fixed = _remove_envelope_tags(markdown)
    fixed = limit_callouts(fixed)
    fixed = _level_headings(fixed)
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
    return fixed
