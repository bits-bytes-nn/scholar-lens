"""Draft checks shared by the writers: the measured prose length and the
structure a minimal edit must keep (the summary's targeted fix).
"""

from __future__ import annotations

import re
from collections import Counter

from scholar_lens.domain.markdown.lint import TABLE_DELIMITER, TABLE_ROW
from scholar_lens.domain.markdown.math import (
    DISPLAY_MATH,
    IMAGE,
    image_sources,
    segments,
    without_fences,
)

_QUOTE_PREFIX = re.compile(r"^[ \t]*(?:>[ \t]?)+")
_LINKED_IMAGE = re.compile(rf"\[{IMAGE.pattern}\]\([^)\s]*\)")


def prose_chars(markdown: str) -> int:
    """Characters of prose: whitespace collapsed to
    single spaces; fenced code, table rows (in a quote too, and a table
    written without outer pipes) and images (a linked one too) excluded, the
    text beside an image kept."""
    lines = [
        _QUOTE_PREFIX.sub("", line) for line in without_fences(markdown).splitlines()
    ]
    table = [bool(TABLE_ROW.match(line)) for line in lines]
    for i, line in enumerate(lines):
        # A pipe-less table: its header, its "--- | ---" row and the "a | b"
        # rows under it.
        if "|" in line and TABLE_DELIMITER.match(line) and i and "|" in lines[i - 1]:
            table[i - 1] = True
            end = i
            while end < len(lines) and "|" in lines[end]:
                table[end] = True
                end += 1
    kept = [
        IMAGE.sub("", _LINKED_IMAGE.sub("", line))
        for line, is_table in zip(lines, table, strict=True)
        if not is_table
    ]
    return len(" ".join(" ".join(kept).split()))


# How the summary's targeted fix frames the blockers for its writer.
TARGETED_FIX = (
    "TARGETED FIX: this draft is final except for these blocking issues. "
    "Make the smallest edit that resolves each one: rewrite or remove exactly "
    "the sentence(s) it names, and copy every other sentence, heading, "
    "citation, equation, figure, table and code block unchanged (edit a code "
    "block in place, never drop it). Do not rewrite, reorder, "
    "shorten or extend anything else.\n"
)
# A targeted fix may shrink the prose by at most this share.
_FIX_MAX_PROSE_LOSS = 0.25
_HEADING_LINE = re.compile(r"^#{1,6}[ \t]+\S.*$", re.MULTILINE)
_CITATION_GROUP = re.compile(r"\[(@[^\[\]\n]+)\]")


def _citation_keys(text: str) -> set[str]:
    return {
        key.strip().removeprefix("@").strip()
        for group in _CITATION_GROUP.findall(text)
        for key in group.split(";")
    }


def _fence_count(text: str) -> int:
    return sum(seg.kind == "fence" for seg in segments(text))


def lost_structure(draft: str, edited: str) -> str:
    """What ``edited`` lost of ``draft``'s structure ("" when nothing): a
    heading, an image or a ``[@…]`` citation key it no longer has, fewer
    code blocks, display equations or tables (one may be corrected in place, not
    removed), or more than a quarter of the prose (:func:`prose_chars`).
    These are invariants of a minimal edit, not judgements of its quality."""
    if _fence_count(edited) < _fence_count(draft):
        return "loses a code block"
    before, after = without_fences(draft), without_fences(edited)
    for kind, items in (
        ("heading", lambda text: [h.strip() for h in _HEADING_LINE.findall(text)]),
        ("image", image_sources),
    ):
        if lost := list((Counter(items(before)) - Counter(items(after))).elements()):
            return f"loses {kind} {lost[0]!r}"
    if lost_keys := sorted(_citation_keys(before) - _citation_keys(after)):
        return f"loses citation [@{lost_keys[0]}]"
    for kind, count in (
        ("display equation", lambda text: len(DISPLAY_MATH.findall(text))),
        (
            "table",
            lambda text: sum(
                bool(TABLE_DELIMITER.match(line)) for line in text.splitlines()
            ),
        ),
    ):
        if count(after) < count(before):
            return f"loses a {kind}"
    old, new = prose_chars(draft), prose_chars(edited)
    if old and new < old * (1 - _FIX_MAX_PROSE_LOSS):
        return f"shrinks the prose from {old} to {new} characters"
    return ""
