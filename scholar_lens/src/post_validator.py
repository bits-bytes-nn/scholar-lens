"""Final gate for a generated post: does it follow the blog's conventions?

:func:`validate_post` checks the finished Markdown (front matter + body) that is
about to be published. An ``error`` means the blog would render it wrongly or a
person would have to fix it, so publishing stops; a ``warning`` is attached to
the pull request as a review checklist item. All checks are deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

import yaml

from .front_matter import split_front_matter_text
from .markdown_lint import ENVELOPE_TAG, mostly_hangul
from .markdown_math import (
    ESCAPED_UNDERSCORE,
    HANGUL,
    TEXT_COMMAND,
    fence_roles,
    has_math,
    image_sources,
    indented_code_lines,
)
from .post_conventions import PostConventions
from .tags import format_tag

ArtifactKind = Literal["review", "summarize", "guide"]
Level = Literal["error", "warning"]

ARXIV_IMAGE_HOSTS = ("arxiv.org", "ar5iv.org", "ar5iv.labs.arxiv.org")

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_INLINE_MATH = re.compile(r"\$\$(.+?)\$\$")
_INLINE_CODE = re.compile(r"`[^`]*`")
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
# A cell holding a value ("63.6", "1,820", "37.4%", "5.3±0.2"), not a label
# that merely contains a digit ("pass@1").
_NUMERIC_CELL = re.compile(
    r"[\s*~≈<>+\-−$]*\d[\d.,]*\s*(?:%|[KMBkmb×x])?(?:\s*(?:±|\+/-)\s*[\d.,]+%?)?[\s*]*"
)
_PERSON = re.compile(r"^[A-Z][a-zA-Z'.\-]+(?:\s+[A-Z][a-zA-Z'.\-]+){1,3}$")
# Words that mark a name as an organisation rather than a person.
_ORG_WORDS = re.compile(
    r"\b(AI|Lab|Labs|Laboratory|Research|University|Institute|College|School|"
    r"Inc|Corp|Corporation|Ltd|LLC|Company|Group|Team|Foundation|Center|Centre|"
    r"Academy|Technologies|Technology|DeepMind|Google|Meta|Microsoft|OpenAI|"
    r"Anthropic|NVIDIA|Amazon|Apple|IBM|Alibaba|Tencent|Baidu|ByteDance|Mistral)\b"
)


@dataclass(frozen=True)
class Violation:
    level: Level
    rule: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] line {self.line}: {self.message}"


class PostValidationError(Exception):
    """A post broke at least one convention; nothing was published."""

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        # Where the rejected post was kept for inspection (set by the publisher).
        self.rejected_path: str | None = None
        self.rejected_s3_uri: str | None = None
        errors = [v for v in violations if v.level == "error"]
        summary = "; ".join(str(v) for v in errors[:10])
        more = f" (+{len(errors) - 10} more)" if len(errors) > 10 else ""
        super().__init__(f"{len(errors)} convention error(s): {summary}{more}")


def errors(violations: list[Violation]) -> list[Violation]:
    return [v for v in violations if v.level == "error"]


def warnings(violations: list[Violation]) -> list[Violation]:
    return [v for v in violations if v.level == "warning"]


def split_front_matter(markdown: str) -> tuple[dict[str, Any] | None, str, int]:
    """Return ``(front_matter, body, body_start_line)``; ``None`` if absent or
    unparsable."""
    front, body = split_front_matter_text(markdown)
    if not front:
        return None, markdown, 1
    try:
        data = yaml.safe_load(front[4:-5])
    except yaml.YAMLError:
        return None, markdown, 1
    return (data if isinstance(data, dict) else None), body, front.count("\n") + 1


def _prose_lines(body: str) -> list[tuple[int, str]]:
    """Body lines outside fenced and indented code, as (offset, text)."""
    lines = body.split("\n")
    code = indented_code_lines(lines)
    return [
        (i, line)
        for i, (line, role) in enumerate(zip(lines, fence_roles(lines), strict=True))
        if role == "prose" and i not in code
    ]


_LINT_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def _math_lines(body: str) -> list[tuple[int, str]]:
    """Body lines the blog lint checks for math (script/lint-posts.rb): those
    outside its code fences, which it finds with ``FENCE = /\\A\\s*(`{3,}|~{3,})/``
    (a fence inside a blockquote is not one, and indented code is checked).
    The math rules run on these lines so a post the blog's CI would reject is
    never passed here."""
    out: list[tuple[int, str]] = []
    fence: str | None = None
    for i, line in enumerate(body.split("\n")):
        m = _LINT_FENCE.match(line)
        if m:
            if fence is None:
                fence = m.group(1)
            elif set(line.strip()) == {fence[0]} and len(line.strip()) >= len(fence):
                fence = None
            continue
        if fence is None:
            out.append((i, line))
    return out


_LINT_TEXT_GROUP = re.compile(r"\\(?:text\w*|mathrm|operatorname)\{")
_LINT_TABLE_PIPE = re.compile(r"\A\$\$(?:(?!\$\$).)+\$\$\Z")


def _lint_without_text_groups(math: str) -> str:
    out, rest = [], math
    while m := _LINT_TEXT_GROUP.search(rest):
        out.append(rest[: m.start()])
        depth, k = 1, m.end()
        while k < len(rest) and depth:
            depth += {"{": 1, "}": -1}.get(rest[k], 0)
            k += 1
        rest = rest[k:]
    out.append(rest)
    return "".join(out)


def _lint_span_findings(spans: list[str]) -> list[str]:
    spans = [_lint_without_text_groups(m) for m in spans]
    found = []
    if any("\\_" in m for m in spans):
        found.append(r"no \_ inside $$ (only within \text{})")
    if any(HANGUL.search(m) for m in spans):
        found.append("Korean text inside $$; a currency sign written $$ opens math")
    return found


def _lint_after_display(line: str | None) -> bool:
    rest = _unquote(line or "").strip()
    return not rest or rest.startswith("{:")


def _blog_lint_math(body: str) -> list[tuple[int, str]]:
    """The blog lint's math findings (``script/lint-posts.rb``, ``body`` and
    ``math``) as ``(line index, message)``: ported as written, so the lines
    it reads as prose but kramdown shows as code are judged its way."""
    lines = body.split("\n")
    found: list[tuple[int, str]] = []
    fence: str | None = None
    block: int | None = None
    span: int | None = None
    for i, line in enumerate(lines):
        if m := _LINT_FENCE.match(line):
            if fence is None:
                fence = m.group(1)
            elif re.fullmatch(f"{re.escape(fence[0])}{{{len(fence)},}}", line.strip()):
                fence = None
            continue
        if fence:
            continue
        bare = _unquote(line)
        odd = re.sub(r"`[^`]*`", "", bare).count("$$") % 2 == 1
        if span is not None and not bare.strip():
            found.append((span, "unbalanced $$ in the paragraph"))
            span = None
        if block is not None or (
            odd and span is None and bare.lstrip().startswith("$$")
        ):
            found += [(i, f) for f in _lint_span_findings([line.replace("$", "")])]
            if odd and block is not None:
                if not _lint_after_display(
                    lines[i + 1] if i + 1 < len(lines) else None
                ):
                    found.append((i, "display $$ needs a blank line after it"))
            elif odd and i > 0 and _unquote(lines[i - 1]).strip():
                found.append((i, "display $$ needs a blank line before it"))
            if odd:
                block = None if block is not None else i
            continue
        if odd:
            span = None if span is not None else i
        prose = re.sub(r"`[^`]*`", "", bare)
        found += [
            (i, f) for f in _lint_span_findings(re.findall(r"\$\$(.+?)\$\$", prose))
        ]
        outside = re.sub(r"\$\$.+?\$\$", "", prose)
        if re.search(r"\\\\[(\[]", outside):
            found.append((i, "math written \\\\( or \\\\[; write $$ … $$"))
        if re.search(r"\}\$(?!\$)|(?<!\$)\$\\[a-zA-Z]", outside):
            found.append((i, "TeX next to a single `$`; write the formula as $$ … $$"))
        if _LINT_TABLE_PIPE.match(prose.strip()) and not prose.lstrip().startswith("|"):
            before = i == 0 or not _unquote(lines[i - 1]).strip()
            after = _lint_after_display(lines[i + 1] if i + 1 < len(lines) else None)
            if not (before and after):
                found.append((i, "display $$ needs a blank line before and after"))
    if block is not None:
        found.append((block, "$$ block never closed"))
    if span is not None:
        found.append((span, "unbalanced $$ in the paragraph"))
    return found


def _check_all_math(front, body, lines, offset, add) -> None:  # type: ignore[no-untyped-def]
    """The math rules on the prose ``lines``, plus the blog lint's own math
    findings on the lines it reads as prose although kramdown shows them as
    code (a fence inside a blockquote, an indented code block): its CI would
    reject those, so they must not pass here either."""
    _check_math(front, body, lines, offset, add)
    code_only = {i for i, _ in _math_lines(body)} - {i for i, _ in lines}
    if not code_only:
        return
    for i, message in _blog_lint_math(body):
        if i in code_only:
            add(
                "error",
                "math",
                offset + i,
                f"{message} (the blog lint reads this code as prose)",
            )


def _unclosed_fence_line(body: str) -> int | None:
    """0-based line of the opener of a fence that never closes, if any."""
    lines = body.split("\n")
    opened: int | None = None
    for i, role in enumerate(fence_roles(lines)):
        if role == "open":
            opened = i
        elif role == "close":
            opened = None
    return opened


def validate_post(
    markdown: str, spec: PostConventions, kind: ArtifactKind
) -> list[Violation]:
    """Check ``markdown`` against ``spec`` for an artifact of ``kind``."""
    found: list[Violation] = []

    def add(level: Level, rule: str, line: int, message: str) -> None:
        found.append(Violation(level, rule, line, message))

    front, body, offset = split_front_matter(markdown)
    if front is None:
        add("error", "front-matter", 1, "missing or invalid YAML front matter")
        return found

    lines = _prose_lines(body)
    unclosed = _unclosed_fence_line(body)
    if unclosed is not None:
        add("error", "code", offset + unclosed, "a code fence is never closed")
    _check_front_matter(front, body, spec, kind, add)
    _check_headings(lines, offset, kind, add)
    _check_all_math(front, body, lines, offset, add)
    _check_tables(lines, offset, add)
    _check_images(lines, offset, add)
    _check_text(lines, offset, add)
    _check_description(lines, offset, add)
    return found


def body_errors(markdown: str) -> list[str]:
    """The publish-blocking body checks (tables, math, code fences) as short
    issue strings, so a writing loop can fix them before the post is final."""
    found: list[str] = []

    def add(level: Level, rule: str, line: int, message: str) -> None:
        if level == "error" and rule != "use-math":
            found.append(f"{rule}: {message} (line {line})")

    lines = _prose_lines(markdown)
    unclosed = _unclosed_fence_line(markdown)
    if unclosed is not None:
        add("error", "code", 1 + unclosed, "a code fence is never closed")
    _check_all_math({}, markdown, lines, 1, add)
    _check_tables(lines, 1, add)
    return found


def _check_front_matter(front, body, spec, kind, add) -> None:  # type: ignore[no-untyped-def]
    for key in spec.forbidden_keys:
        if key in front:
            add("error", "forbidden-key", 1, f"front matter must not contain '{key}'")

    title = str(front.get("title") or "").strip()
    if not title:
        add("error", "title", 1, "title is empty")
    date = front.get("date")
    if isinstance(date, datetime):
        stamp = date if date.tzinfo else date.replace(tzinfo=UTC)
        if stamp > datetime.now(UTC):
            # Jekyll (timezone UTC, no `future`) skips future-dated posts.
            add(
                "error",
                "date",
                1,
                f"date {date} is in the future (UTC); the post would not appear",
            )

    categories = front.get("categories")
    if not (isinstance(categories, list) and len(categories) == 2):
        add("error", "categories", 1, "categories must be [<type>, <topic>]")
        category_type, topic = "", ""
    else:
        category_type, topic = str(categories[0]), str(categories[1])
        if category_type not in spec.types:
            add("error", "categories", 1, f"unknown type '{category_type}'")
        elif spec.is_paper_type(category_type) != (kind != "guide"):
            add(
                "error",
                "categories",
                1,
                f"type '{category_type}' does not fit a {kind}",
            )
        if topic not in spec.topics:
            add("error", "categories", 1, f"unknown topic '{topic}'")
        elif category_type in spec.types:
            expected = spec.cover_for(category_type, topic)
            if front.get("cover") != expected:
                add("error", "cover", 1, f"cover must be '{expected}'")

    if kind != "guide":
        _check_paper_author(front.get("paper_author"), add)
        subtitle = str(front.get("subtitle") or "").strip()
        if not subtitle:
            add("error", "subtitle", 1, "paper posts need a subtitle")
        elif subtitle == title:
            add("error", "subtitle", 1, "subtitle must differ from the title")
    elif "paper_author" in front:
        add("error", "paper-author", 1, "guides do not take paper_author")

    seo_title = front.get("seo_title")
    limit = spec.seo_title_max_length
    if len(title) > limit and not seo_title:
        add("error", "seo-title", 1, f"title is over {limit} characters; add seo_title")
    if seo_title and len(str(seo_title)) > limit:
        add("error", "seo-title", 1, f"seo_title is over {limit} characters")

    _check_tags(front.get("tags"), body, spec, add)

    lang = front.get("lang")
    if lang is not None:
        if lang not in ("ko", "en"):
            add("error", "translation", 1, "lang must be 'ko' or 'en'")
        if not front.get("translation_id"):
            add("error", "translation", 1, "a translated post needs translation_id")


def _check_paper_author(value: Any, add) -> None:  # type: ignore[no-untyped-def]
    author = str(value or "").strip()
    if not author:
        add(
            "error",
            "paper-author",
            1,
            "paper_author (publishing organisation) is empty",
        )
        return
    if re.search(r"\bet\s+al\b", author, re.IGNORECASE):
        add("error", "paper-author", 1, f"paper_author is a person list: '{author}'")
        return
    parts = [p.strip() for p in re.split(r",|\band\b|&", author) if p.strip()]
    people = [p for p in parts if _PERSON.match(p) and not _ORG_WORDS.search(p)]
    if len(parts) >= 2 and len(people) == len(parts):
        add("error", "paper-author", 1, f"paper_author lists people: '{author}'")


def _check_tags(tags: Any, body: str, spec: PostConventions, add) -> None:  # type: ignore[no-untyped-def]
    if not isinstance(tags, list) or not tags:
        add("error", "tags", 1, "tags must be a non-empty list")
        return
    words = spec.tags.lowercase_words
    controlled = set(spec.tags.controlled)
    aliases = {k.lower() for k in spec.tags.aliases}
    for tag in map(str, tags):
        if format_tag(tag, words) != tag:
            add(
                "error",
                "tags",
                1,
                f"tag '{tag}' is not normalized ('{format_tag(tag, words)}')",
            )
        if tag.lower() in aliases:
            add(
                "error",
                "tags",
                1,
                f"tag '{tag}' is retired; use '{spec.tags.aliases.get(tag, '')}'",
            )
    used = [t for t in map(str, tags) if t in controlled]
    if not used:
        add("error", "tags", 1, "at least one controlled tag is required")
    if len(used) > spec.tags.max_controlled:
        add(
            "warning",
            "tags",
            1,
            f"more than {spec.tags.max_controlled} controlled tags",
        )
    specific = [t for t in map(str, tags) if t not in controlled]
    if len(specific) > spec.tags.max_specific:
        add("error", "tags", 1, f"at most {spec.tags.max_specific} specific tags")


def _check_headings(lines, offset, kind, add) -> None:  # type: ignore[no-untyped-def]
    previous = 1
    first = True
    for i, line in lines:
        match = _HEADING.match(line)
        if not match:
            continue
        level, text = len(match.group(1)), match.group(2)
        if level == 1:
            add("error", "headings", offset + i, f"H1 in the body: '{text}'")
        elif level > previous + 1:
            add(
                "error",
                "headings",
                offset + i,
                f"heading jumps from H{previous} to H{level}",
            )
        if first and kind != "guide" and not (level == 2 and text == "TL;DR"):
            add("error", "headings", offset + i, "a paper post starts with '## TL;DR'")
        first = False
        previous = level
    if first and kind != "guide":
        add("error", "headings", offset, "a paper post starts with '## TL;DR'")
    if kind != "guide":
        _check_tldr(lines, offset, kind, add)


def _check_tldr(lines, offset, kind, add) -> None:  # type: ignore[no-untyped-def]
    """The ``## TL;DR`` section has prose (and, in a review, its ``###``
    questions): an empty one means the finalizer failed."""
    start = next((i for i, line in lines if line.strip() == "## TL;DR"), None)
    if start is None:
        return  # reported above
    prose = questions = 0
    for i, line in lines:
        if i <= start:
            continue
        stripped = line.strip()
        if stripped.startswith("## ") or stripped == "- - -":
            break
        if stripped.startswith("### "):
            questions += 1
        elif stripped:
            prose += 1
    if not prose:
        add("error", "headings", offset + start, "the '## TL;DR' section is empty")
    elif kind == "review" and not questions:
        add(
            "error",
            "headings",
            offset + start,
            "the review's '## TL;DR' has no '###' questions",
        )


_QUOTE = re.compile(r"^\s*(?:>\s?)+")
# Math the Markdown reads first: `\\(`/`\\[` reach MathJax only after `*` and
# `_` inside them turned into <em>. (A single `\(` is a Markdown escape for a
# literal parenthesis; normalize_math has already turned `\(x\)` math into $$.)
_ESCAPED_DELIMS = re.compile(r"\\\\[(\[]")
# TeX next to a single `$`: math that never opened (a price is neither
# preceded by `}` nor followed by a command).
_LONE_DOLLAR_TEX = re.compile(r"\}\$(?!\$)|(?<!\$)\$\\[a-zA-Z]")
_INLINE_SPAN = re.compile(r"\$\$(?:(?!\$\$).)+\$\$")
_WRAPPED_SPAN = re.compile(r"\$\$((?:(?!\$\$).)+)\$\$", re.DOTALL)
# Lines that start a new block: a heading, or a list item (which ends a span
# opened in a sibling item). An inline $$ span left open before it never
# closes there.
_STARTS_HEADING = re.compile(r"#{1,6}[ \t]")
_STARTS_ITEM = re.compile(r"(?:[-*+]|\d+[.)])[ \t]")
# A single-backslash \[ … \] around TeX in prose: display math the
# normalizer could not convert (Markdown shows it as "[ … ]").
# A TeX command or a spaced "=" marks math; "\[^1\]" (a footnote) and
# "\[--max_len\]" (an option) do not.
_BRACKET_TEX = re.compile(
    r"(?<!\\)\\\[(?=[^\]]*(?:\\[A-Za-z]|\s=\s))"
    r"|^[ \t]*\\{1,2}[\[\]][ \t]*$"  # a bracket display's own delimiter line
)
# The same over lines: a "\[" ending a line and its "\]" on a later line of
# the paragraph, with math between them ("배열은 a\[\ni\]" is not math).
_BRACKET_BLOCK = re.compile(
    r"(?<!\\)\\\[[ \t]*\n(?P<body>(?:(?!\\\]).)*?(?:\\[A-Za-z]|\s=\s).*?)\\\]",
    re.DOTALL,
)


def _unquote(line: str) -> str:
    """The text of a line after any blockquote markers."""
    return _QUOTE.sub("", line)


def _check_math(front, body, lines, offset, add) -> None:  # type: ignore[no-untyped-def]
    """The blog's math rules (script/lint-posts.rb), blockquotes included:
    only ``$$`` delimiters; a display block opens on a line starting with
    ``$$`` and closes on the next line with an odd number of them, with blank
    lines around it; an inline span may wrap within its paragraph."""
    if bool(front.get("use_math")) != has_math(body):
        add(
            "error",
            "use-math",
            1,
            "use_math must be true exactly when the post contains $$ math",
        )
    bare_by_index = {i: _unquote(line) for i, line in lines}

    raw_by_index = dict(lines)
    body_lines = body.split("\n")

    def blank(i: int) -> bool:
        return not bare_by_index.get(i, "").strip()

    def starts_block(i: int) -> bool:
        """Whether a bare | on line ``i`` splits it into table cells: a table
        row, or a paragraph's, list item's or quote's first line (not a
        heading, a continuation line, or a line glued under a heading or a
        closing fence, which kramdown keeps as a paragraph)."""
        text = bare_by_index.get(i, "").lstrip()
        if text.startswith("#"):
            return False
        if text.startswith("|"):  # a table row: kramdown splits it at the bar
            return True
        before_raw = body_lines[i - 1] if 0 < i <= len(body_lines) else ""
        if i - 1 not in bare_by_index and before_raw.strip():
            return False  # glued under a code line (a closing fence)
        before = raw_by_index.get(i - 1, "").lstrip()
        if before.startswith("#"):
            return False
        if i not in bare_by_index or blank(i - 1) or _STARTS_ITEM.match(text):
            return True
        quoted = raw_by_index.get(i, "").lstrip().startswith(">")
        return quoted and not before.startswith(">")

    def closes_mid_line(i: int) -> bool:
        """Whether the ``$$`` opened on line ``i`` closes on a later line of
        the paragraph with text after it: an inline span that starts a
        paragraph line, not a display block."""
        j = i + 1
        while j in bare_by_index and not blank(j):
            later = _INLINE_CODE.sub("", bare_by_index[j]).strip()
            if later.count("$$") % 2:
                return bool(later.split("$$", 1)[1].strip())
            j += 1
        return False

    block: int | None = None  # line a multi-line display block opened on
    span: int | None = None  # line an inline span wrapping onto the next opened on
    for i, _ in lines:
        bare = _INLINE_CODE.sub("", bare_by_index[i])
        stripped = bare.strip()
        odd = stripped.count("$$") % 2 == 1
        if span is not None and (
            not stripped
            or _STARTS_HEADING.match(stripped)
            or (
                _STARTS_ITEM.match(stripped)
                and _STARTS_ITEM.match(bare_by_index[span].strip())
            )
            or bare_by_index[span].lstrip().startswith("|")  # rows never wrap
        ):
            # A blank line, a sibling list item or a heading ends the block:
            # the open $$ never closes there.
            add("error", "math", offset + span, "unbalanced $$ in the paragraph")
            span = None
        opens_block = (
            odd
            and span is None
            and stripped.startswith("$$")
            and (blank(i - 1) or not closes_mid_line(i))
        )
        if block is not None or opens_block:
            _check_math_span(stripped.replace("$$", ""), offset + i, add, inline=False)
            if odd and block is not None and not blank(i + 1):
                if not bare_by_index.get(i + 1, "").strip().startswith("{:"):
                    add(
                        "error",
                        "math",
                        offset + i,
                        "display $$ needs a blank line after it",
                    )
            elif odd and block is None and i > 0 and not blank(i - 1):
                add(
                    "error",
                    "math",
                    offset + i,
                    "display $$ needs a blank line before it",
                )
            if odd:
                block = None if block is not None else i
            continue
        if odd:
            if span is not None:
                # A span wrapped over lines: the per-line checks never see it.
                text = "\n".join(
                    _INLINE_CODE.sub("", bare_by_index.get(j, ""))
                    for j in range(span, i + 1)
                )
                for math in _WRAPPED_SPAN.findall(text):
                    if "\n" in math:
                        _check_math_span(
                            math,
                            offset + span,
                            add,
                            inline=True,
                            row=starts_block(span),
                        )
                # The rest of this line, after the closing $$, is checked as
                # its own line below.
                bare = bare.split("$$", 1)[1]
                span = None
            else:
                span = i
                bare = bare.rsplit("$$", 1)[0]  # the wrapped span is checked whole
        outside = _INLINE_SPAN.sub("", bare)
        if _ESCAPED_DELIMS.search(outside) or _BRACKET_TEX.search(outside):
            add("error", "math", offset + i, r"math written \( or \[; use $$ … $$")
        if _LONE_DOLLAR_TEX.search(outside):
            add(
                "error",
                "math",
                offset + i,
                "TeX next to a single `$`; write the formula as $$ … $$",
            )
        if _INLINE_SPAN.fullmatch(stripped) and not stripped.startswith("|"):
            # kramdown attribute lists ("{: …}") may follow a display directly.
            after_ok = blank(i + 1) or bare_by_index.get(i + 1, "").strip().startswith(
                "{:"
            )
            if not (blank(i - 1) and after_ok):
                add(
                    "error",
                    "math",
                    offset + i,
                    "display $$ needs blank lines around it",
                )
            _check_math_span(stripped[2:-2], offset + i, add, inline=False)
            continue
        for math in _INLINE_MATH.findall(bare):
            _check_math_span(math, offset + i, add, inline=True, row=starts_block(i))
    if block is not None:
        add("error", "math", offset + block, "$$ block never closed")
    if span is not None:
        add("error", "math", offset + span, "unbalanced $$ in the paragraph")
    _check_bracket_blocks(lines, bare_by_index, offset, add)
    _check_list_restarts(lines, bare_by_index, offset, add)


_ORDERED_ITEM = re.compile(r"(\d+)[.)][ \t]")


def _check_list_restarts(lines, bare_by_index, offset, add) -> None:  # type: ignore[no-untyped-def]
    """An ordered item numbered above 1 after a display indented less than the
    item's content (at the margin, or glued and under-indented), with or
    without a paragraph between them: the display ended the list, so kramdown
    starts a new one and shows "1."."""

    def width(text: str) -> int:
        return len(text.expandtabs(4)) - len(text.expandtabs(4).lstrip())

    display_indent: int | None = None  # least indent of a display since an item
    for i, raw in lines:
        line = bare_by_index[i]
        stripped = line.strip()
        if not stripped:
            continue
        # A quoted line: the indent before its ">" (a callout in a step) plus
        # the indent inside the quote (a list in a quote).
        indent = width(line) + (width(raw) if raw.lstrip().startswith(">") else 0)
        item = _ORDERED_ITEM.match(stripped)
        if item and _STARTS_ITEM.match(stripped):
            column = indent + len(item.group(0))
            if (
                int(item.group(1)) > 1
                and display_indent is not None
                and display_indent < column
            ):
                add(
                    "warning",
                    "list",
                    offset + i,
                    f"ordered list restarts at {item.group(1)} after a display, so "
                    "it renders as 1.; indent the display (and any paragraph after "
                    "it) into the item above",
                )
            display_indent = None
            continue
        if _STARTS_ITEM.match(stripped):
            display_indent = None
        elif stripped.startswith("$$") and (
            stripped.endswith("$$") or stripped.count("$$") % 2
        ):  # a whole-line display or a block's opening line
            display_indent = (
                indent if display_indent is None else min(display_indent, indent)
            )


def _check_bracket_blocks(lines, bare_by_index, offset, add) -> None:  # type: ignore[no-untyped-def]
    """A "\\[ … \\]" display written over lines (kramdown shows it as text)."""
    paragraph: list[int] = []

    def flush() -> None:
        text = "\n".join(_INLINE_CODE.sub("", bare_by_index[j]) for j in paragraph)
        prose = _INLINE_SPAN.sub("", text)
        for m in _BRACKET_BLOCK.finditer(prose):
            if not prose[prose.rfind("\n", 0, m.start()) + 1 : m.start()].strip():
                continue  # a "\\[" on its own line is reported line by line
            line = paragraph[0] + prose.count("\n", 0, m.start())
            add("error", "math", offset + line, r"math written \( or \[; use $$ … $$")
        paragraph.clear()

    for i, _ in lines:
        if not bare_by_index[i].strip() or (paragraph and i != paragraph[-1] + 1):
            flush()
        if bare_by_index[i].strip():
            paragraph.append(i)
    flush()


def _check_math_span(math: str, line: int, add, *, inline: bool, row: bool = False) -> None:  # type: ignore[no-untyped-def]
    outside_text = TEXT_COMMAND.sub("", math)
    if ESCAPED_UNDERSCORE.search(outside_text):
        add("error", "math", line, r"no \_ inside $$ (only within \text{})")
    if HANGUL.search(outside_text):  # the blog lint checks display math too
        add("error", "math", line, "Korean text inside $$ (currency or prose?)")
    # kramdown reads a bare | in inline math on the first line of a block (a
    # paragraph, list item or quote; ``row``) as a table cell separator, so
    # the block becomes a table. normalize_math rewrites it to \vert except
    # inside \text{} or an array column spec, so what reaches here needs a
    # hand fix.
    if inline and row and re.search(r"(?<!\\)\|", math):
        add(
            "error",
            "math",
            line,
            r"bare | inside inline $$ (in \text{} or an array spec); use \vert,"
            r" \lvert … \rvert or \mid",
        )


def _split_cells(row: str) -> list[str]:
    masked = _INLINE_CODE.sub(lambda m: "x" * len(m.group(0)), row)
    masked = _INLINE_MATH.sub(lambda m: "x" * len(m.group(0)), masked)
    masked = masked.replace("\\|", "xx")
    cells, start = [], 0
    for pos, char in enumerate(masked):
        if char == "|":
            cells.append(row[start:pos])
            start = pos + 1
    cells.append(row[start:])
    stripped = row.strip()
    if stripped.startswith("|"):
        cells = cells[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        cells = cells[:-1]
    return [c.strip() for c in cells]


def _check_tables(lines, offset, add) -> None:  # type: ignore[no-untyped-def]
    index = 0
    while index < len(lines):
        i, line = lines[index]
        nxt = lines[index + 1][1] if index + 1 < len(lines) else ""
        if line.lstrip().startswith("|") and _TABLE_SEPARATOR.match(nxt.strip()):
            header = _split_cells(line)
            width = len(header)
            block = [(i, line)]
            index += 2
            while index < len(lines) and lines[index][1].lstrip().startswith("|"):
                block.append(lines[index])
                index += 1
            if len(_split_cells(nxt)) != width:
                add(
                    "error",
                    "table",
                    offset + i + 1,
                    "separator cell count differs from header",
                )
            for col, label in enumerate(header):
                if col and not label:
                    add(
                        "error",
                        "table",
                        offset + i,
                        f"empty header cell in column {col + 1}",
                    )
            for col in range(1, width):
                if header[col] and header[col] == header[col - 1]:
                    add(
                        "warning",
                        "table",
                        offset + i,
                        f"repeated header '{header[col]}'",
                    )
            previous: list[str] | None = None
            for row_number, (j, row) in enumerate(block[1:]):
                cells = _split_cells(row)
                if cells == previous and any(cells):
                    add("error", "table", offset + j, "row repeats the row above")
                previous = cells
                # Fewer cells are a group-label row kramdown pads (the blog's
                # lint allows them); more cells shift the columns.
                if len(cells) > width:
                    add(
                        "error",
                        "table",
                        offset + j,
                        f"{len(cells)} cells, header has {width}",
                    )
                if (
                    row_number == 0
                    and cells
                    and not cells[0]
                    and not any(_NUMERIC_CELL.fullmatch(c) for c in cells)
                ):
                    add(
                        "error",
                        "table",
                        offset + j,
                        "sub-header row; merge it into the header",
                    )
            continue
        index += 1


_NUMBER = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")
# Numbers too small or too common to tell a copied value from an invented one.
_TRIVIAL_NUMBER = re.compile(r"\d{1,2}|\d{4}")
MAX_UNSOURCED_NUMBERS = 8


def _canonical_number(token: str) -> str:
    """``1,234.50`` → ``1234.5``; ``-0.30`` → ``0.3`` (sign ignored)."""
    digits = token.lstrip("-+−").replace(",", "").lower()
    if "e" in digits:
        try:
            return repr(float(digits))
        except ValueError:
            return digits
    if "." in digits:
        digits = digits.rstrip("0").rstrip(".")
    return digits.lstrip("0") or "0"


def unsourced_table_numbers(markdown: str, source_text: str) -> list[str]:
    """Table rows of the post whose numbers never appear together in one
    line or one table of the source paper, plus lone values
    the paper never states. Thousands separators, trailing zeros and signs
    are normalised; small integers and years count only within a row.

    A row that regroups or derives values can be legitimate, so these are
    review notes, not errors.
    """
    _, body, _ = split_front_matter(markdown)
    # A source unit is a line (a sentence or table row) or a whole table, so a
    # transposed or regrouped table still finds its values together.
    units: list[str] = []
    table: list[str] = []
    for line in [*source_text.splitlines(), ""]:
        units.append(line)
        if line.lstrip().startswith("|"):
            table.append(line)
        elif table:
            units.append("\n".join(table))
            table = []
    source_rows = [{_canonical_number(t) for t in _NUMBER.findall(u)} for u in units]
    known = set().union(*source_rows) if source_rows else set()
    missing: list[str] = []
    for _, line in _prose_lines(body):
        if not line.lstrip().startswith("|") or _TABLE_SEPARATOR.match(line.strip()):
            continue
        cells = _split_cells(_INLINE_MATH.sub(" ", _INLINE_CODE.sub(" ", line)))
        tokens = [
            t for cell in cells[1:] for t in _NUMBER.findall(cell)
        ]  # not the label
        values = {_canonical_number(t) for t in tokens}
        if len(values) >= 2:
            if not any(values <= row for row in source_rows):
                label = cells[0] or "row"
                missing.append(f"{label}: {' / '.join(tokens)}")
            continue
        for token in tokens:
            value = _canonical_number(token)
            if not _TRIVIAL_NUMBER.fullmatch(value) and value not in known:
                missing.append(token)
    return missing[:MAX_UNSOURCED_NUMBERS]


_IMAGE_ALT = re.compile(r"!\[([^\]]*)\]\(")


def _check_images(lines, offset, add) -> None:  # type: ignore[no-untyped-def]
    korean = mostly_hangul("\n".join(line for _, line in lines))
    for i, line in lines:
        if korean:
            for alt in _IMAGE_ALT.findall(_INLINE_CODE.sub("", line)):
                if not HANGUL.search(alt):
                    add(
                        "warning",
                        "image-alt",
                        offset + i,
                        f"image alt text is not Korean: '{alt[:60]}'",
                    )
        for src in image_sources(line):
            host = urlparse(src).hostname or ""
            if src.startswith(("http://", "https://")):
                if host in ARXIV_IMAGE_HOSTS or host.endswith(
                    tuple("." + h for h in ARXIV_IMAGE_HOSTS)
                ):
                    add(
                        "error",
                        "image",
                        offset + i,
                        f"arXiv image not localized: {src}",
                    )
                else:
                    add("warning", "image", offset + i, f"external image: {src}")
            elif not re.fullmatch(r"/assets/images/[^/]+", src):
                add(
                    "error",
                    "image",
                    offset + i,
                    f"image must live in /assets/images/: {src}",
                )


def _check_description(lines, offset, add) -> None:  # type: ignore[no-untyped-def]
    """The blog derives the post description from the first prose paragraph
    and strips math from it, so math there leaves a broken sentence."""
    for i, line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ">", "|", "-", "*", "!", "$$")):
            continue
        if re.match(r"\d+\.\s", stripped):
            continue
        if "$$" in stripped:
            add(
                "warning",
                "description",
                offset + i,
                "first paragraph (the search description) contains math",
            )
        return


def _check_text(lines, offset, add) -> None:  # type: ignore[no-untyped-def]
    for i, line in lines:
        prose = _INLINE_CODE.sub("", line)
        if ENVELOPE_TAG.search(prose):
            add("error", "envelope", offset + i, "generator tag leaked into the post")
        if "[Image:" in prose:
            add(
                "error",
                "envelope",
                offset + i,
                "internal figure marker leaked into the post",
            )
