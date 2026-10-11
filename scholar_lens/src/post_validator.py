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

from .markdown_lint import ENVELOPE_TAGS, NUMBERED_REF
from .markdown_math import TEXT_COMMAND, has_math
from .post_conventions import PostConventions
from .tags import format_tag

ArtifactKind = Literal["review", "summarize", "guide"]
Level = Literal["error", "warning"]

ARXIV_IMAGE_HOSTS = ("arxiv.org", "ar5iv.org", "ar5iv.labs.arxiv.org")

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_HANGUL = re.compile(r"[\uac00-\ud7a3\u3131-\u318e]")
_INLINE_MATH = re.compile(r"\$\$(.+?)\$\$")
_INLINE_CODE = re.compile(r"`[^`]*`")
_IMAGE = re.compile(
    r"!\[[^\]]*\]\(([^()\s]*(?:\([^()]*\)[^()\s]*)*)(?:\s+\"[^\"]*\")?\)"
)
_ENVELOPE = re.compile(rf"</?(?:{'|'.join(ENVELOPE_TAGS)})>")
_NUMBERED_REF = NUMBERED_REF
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
# A cell holding a value ("63.6", "1,820", "37.4%", "5.3±0.2"), not a label
# that merely contains a digit ("pass@1").
_NUMERIC_CELL = re.compile(
    r"[\s*~≈<>+\-−$]*\d[\d.,]*\s*(?:%|[KMBkmb×x])?(?:\s*(?:±|\+/-)\s*[\d.,]+%?)?[\s*]*"
)
_ESCAPED_UNDERSCORE = re.compile(r"\\_")
_TEXT_COMMAND = TEXT_COMMAND
_LEGACY_DELIMS = re.compile(r"\\\\?\(|\\\\?\)")
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
    if not markdown.startswith("---\n"):
        return None, markdown, 1
    end = markdown.find("\n---\n", 4)
    if end == -1:
        return None, markdown, 1
    try:
        data = yaml.safe_load(markdown[4:end])
    except yaml.YAMLError:
        return None, markdown, 1
    body = markdown[end + 5 :]
    return (
        (data if isinstance(data, dict) else None),
        body,
        markdown[: end + 5].count("\n") + 1,
    )


def _prose_lines(body: str) -> list[tuple[int, str]]:
    """Body lines outside fenced code, as (offset, text)."""
    out: list[tuple[int, str]] = []
    in_fence = False
    for i, line in enumerate(body.split("\n")):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append((i, line))
    return out


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
    _check_front_matter(front, body, spec, kind, add)
    _check_headings(lines, offset, kind, add)
    _check_math(front, body, lines, offset, add)
    _check_tables(lines, offset, add)
    _check_images(lines, offset, add)
    _check_text(lines, offset, add)
    _check_description(lines, offset, add)
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


def _is_display_line(line: str) -> bool:
    stripped = line.strip()
    return stripped == "$$" or (
        stripped.startswith("$$")
        and stripped.endswith("$$")
        and len(stripped) > 4
        and stripped.count("$$") == 2
    )


def _check_math(front, body, lines, offset, add) -> None:  # type: ignore[no-untyped-def]
    if bool(front.get("use_math")) != has_math(body):
        add(
            "error",
            "use-math",
            1,
            "use_math must be true exactly when the post contains $$ math",
        )
    text_by_index = dict(lines)
    in_block = False
    for i, raw in lines:
        line = _INLINE_CODE.sub("", raw)
        stripped = line.strip()
        if _LEGACY_DELIMS.search(line) and not in_block:
            add("error", "math", offset + i, r"use $$ … $$, not \( … \)")
        # A display block may span lines: "$$" alone, or "$$\begin{cases}" …
        # "\end{cases}$$" — a line with an odd number of $$ opens or closes it.
        odd = stripped.count("$$") % 2 == 1
        opens = not in_block and odd and stripped.startswith("$$")
        closes = in_block and odd and stripped.endswith("$$")
        if opens or closes:
            if opens and text_by_index.get(i - 1, "").strip():
                add(
                    "error",
                    "math",
                    offset + i,
                    "display $$ needs a blank line before it",
                )
            if closes and text_by_index.get(i + 1, "").strip():
                add(
                    "error",
                    "math",
                    offset + i,
                    "display $$ needs a blank line after it",
                )
            _check_math_span(stripped.strip("$"), offset + i, add, inline=False)
            in_block = not in_block
            continue
        if in_block:
            _check_math_span(stripped, offset + i, add, inline=False)
            continue
        if _is_display_line(line) and not stripped.startswith("|"):
            before, after = text_by_index.get(i - 1, ""), text_by_index.get(i + 1, "")
            if before.strip() or after.strip():
                add(
                    "error",
                    "math",
                    offset + i,
                    "display $$ needs blank lines around it",
                )
            _check_math_span(stripped[2:-2], offset + i, add, inline=False)
            continue
        if line.count("$$") % 2:
            add("error", "math", offset + i, "unpaired $$ (currency written as math?)")
            continue
        spans = _INLINE_MATH.findall(line)
        for span in spans:
            _check_math_span(
                span,
                offset + i,
                add,
                inline=True,
                crowded=len(spans) > 1 or stripped.startswith("|"),
            )


def _check_math_span(math: str, line: int, add, *, inline: bool, crowded: bool = False) -> None:  # type: ignore[no-untyped-def]
    outside_text = _TEXT_COMMAND.sub("", math)
    if _ESCAPED_UNDERSCORE.search(outside_text):
        add("error", "math", line, r"no \_ inside $$ (only within \text{})")
    if inline and _HANGUL.search(outside_text):
        add("error", "math", line, "Korean text inside inline $$ (currency or prose?)")
    if inline and crowded and re.search(r"(?<!\\)\|", math):
        add("error", "math", line, r"bare | inside inline $$; use \lvert … \rvert")


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
            for row_number, (j, row) in enumerate(block[1:]):
                cells = _split_cells(row)
                if len(cells) != width:
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


def _check_images(lines, offset, add) -> None:  # type: ignore[no-untyped-def]
    for i, line in lines:
        for src in _IMAGE.findall(line):
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
        if _ENVELOPE.search(prose):
            add("error", "envelope", offset + i, "generator tag leaked into the post")
        if "[Image:" in prose:
            add(
                "error",
                "envelope",
                offset + i,
                "internal figure marker leaked into the post",
            )
        if (m := _NUMBERED_REF.search(_INLINE_MATH.sub("", prose))) is not None:
            add(
                "warning",
                "numbered-ref",
                offset + i,
                f"numbered reference '{m.group(0)}'",
            )
