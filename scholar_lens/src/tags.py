"""Deterministic blog-tag normalization against the post conventions.

The LLM proposes topic tags; this module decides what is published: tags are
hyphen-joined, function words lowercased, retired names replaced by their
aliases, duplicates and title restatements dropped, and the controlled tags
(what related-post links are built on) restricted to the closed list.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .post_conventions import GENERATED_SPECIFIC_TAGS, PostConventions

_SEPARATORS = re.compile(r"[\s_\u2010-\u2015]+")  # incl. en/em dashes
_DROP_CHARS = re.compile(r"[/\"'`’“”]")  # deleted, not separators
_INVALID_CHARS = re.compile(r"[^0-9A-Za-z.+\-]")
_HYPHENS = re.compile(r"-{2,}")
# The blog lint's tag form (tech-blog script/lint-posts.rb line 165): a letter
# or digit first (no ".NET" or "+x"), then letters, digits and . + -.
TAG_FORMAT = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*")


def _inner_words(tag: str) -> list[str]:
    """The words between the first and the last, split as Ruby's
    ``split("-")`` does (trailing empty fields dropped; line 168): a last
    "In" or "On" is a particle ("Opt-In", "Add-On"), not a function word."""
    words = tag.split("-")
    while words and not words[-1]:
        words.pop()
    return words[1:-1]


def tag_problem(tag: str, lowercase_words: list[str]) -> str | None:
    """What the blog lint rejects in a tag's form (lint-posts.rb lines
    165-170), or None: anything but :data:`TAG_FORMAT`, or an inner function
    word not in lowercase."""
    if not TAG_FORMAT.fullmatch(tag):
        return f"tag '{tag}' must be words joined by hyphens"
    lower = {w.lower() for w in lowercase_words}
    word = next(
        (w for w in _inner_words(tag) if w.lower() in lower and w != w.lower()), None
    )
    return f"tag '{tag}' should have {word.lower()}, lowercase" if word else None


def format_tag(raw: str, lowercase_words: list[str]) -> str:
    """Normalize spelling only: ``"Pan & Scan / Image processing"`` →
    ``"Pan-and-Scan-Image-processing"``. Word case is kept (acronyms, ``k-bit``)
    except function words between the first and the last word, which are
    lowercased (the blog lint's rule, :func:`tag_problem`); a leading "." or
    "+" is dropped."""
    text = html.unescape(raw).strip().replace("&", " and ")
    text = _DROP_CHARS.sub("", text)
    text = _SEPARATORS.sub("-", text.strip())
    text = _INVALID_CHARS.sub("", text)
    text = _HYPHENS.sub("-", text).lstrip(".+-").strip("-")
    if not text:
        return ""
    lower = {w.lower() for w in lowercase_words}
    words = text.split("-")
    first = words[0]
    # Capitalize a lowercase first word, but keep stylized names ("vLLM").
    if first.islower():
        first = first[:1].upper() + first[1:]
    inner = [w.lower() if w.lower() in lower else w for w in words[1:-1]]
    return "-".join([first, *inner, *words[1:][-1:]])


def _key(tag: str) -> str:
    return tag.lower()


def _title_variants(title: str, lowercase_words: list[str]) -> set[str]:
    parts = [title, *title.split(":", 1)] if ":" in title else [title]
    return {_key(format_tag(p, lowercase_words)) for p in parts if p.strip()}


@dataclass
class TagSelection:
    """Published tags (specific first, controlled last) and what was changed."""

    tags: list[str]
    controlled: list[str]
    specific: list[str]
    replaced: dict[str, str] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)


def normalize_tags(
    specific: list[str],
    controlled: list[str],
    *,
    title: str,
    spec: PostConventions,
    known_tags: list[str] | None = None,
) -> TagSelection:
    """Build the published tag list from the model's proposals.

    ``controlled`` proposals outside the closed list are dropped; a specific
    tag that is itself a controlled tag moves to the controlled group. The
    caller (the post validator) rejects a result without any controlled tag.
    """
    words = spec.tags.lowercase_words
    canonical_controlled = {_key(t): t for t in spec.tags.controlled}
    aliases = {_key(k): v for k, v in spec.tags.aliases.items()}
    # Jekyll tags are case-sensitive: reuse the blog's existing spelling.
    spelling = {_key(t): t for t in known_tags or [] if format_tag(t, words) == t}
    title_keys = _title_variants(title, words)

    selection = TagSelection(tags=[], controlled=[], specific=[])
    seen: set[str] = set()

    def place(raw: str, *, prefer_controlled: bool) -> None:
        tag = format_tag(raw, words)
        if not tag:
            return
        if (alias := aliases.get(_key(tag))) is not None:
            selection.replaced[tag] = alias
            tag = alias
        tag = spelling.get(_key(tag), tag)
        key = _key(tag)
        if key in seen:
            return
        # A tag restating the title is dropped unless the blog already uses
        # it: "vLLM" for a "vLLM: …" guide still links to the other vLLM posts.
        if (
            key in title_keys
            and key not in spelling
            and key not in canonical_controlled
        ):
            selection.dropped.append(tag)
            return
        if key in canonical_controlled:
            seen.add(key)
            selection.controlled.append(canonical_controlled[key])
        elif prefer_controlled:
            selection.dropped.append(tag)  # not on the closed list
        else:
            seen.add(key)
            selection.specific.append(tag)

    for raw in controlled:
        place(raw, prefer_controlled=True)
    for raw in specific:
        place(raw, prefer_controlled=False)

    overflow = selection.controlled[spec.tags.max_controlled :]
    selection.controlled = selection.controlled[: spec.tags.max_controlled]
    selection.dropped.extend(overflow)
    limit = min(GENERATED_SPECIFIC_TAGS, spec.tags.max_specific)
    selection.dropped.extend(selection.specific[limit:])
    selection.specific = selection.specific[:limit]
    selection.tags = [*selection.specific, *selection.controlled]
    return selection


def acronym(tag: str) -> str:
    """Initials of the capitalized words: ``Direct-Preference-Optimization`` → ``DPO``."""
    return "".join(w[0] for w in tag.split("-") if w and w[0].isupper())


def unsupported_tags(tags: list[str], text: str) -> list[str]:
    """Tags whose phrase, or its acronym, never appears in ``text`` as whole
    words (the phrase may end in a plural ``s``/``es``) — a likely mis-tag
    (e.g. a DPO tag on a paper that never uses DPO; a "RAG" tag is not
    supported by "storage")."""
    haystack = re.sub(r"[\s\-_]+", " ", text.lower())
    missing = []
    for tag in tags:
        phrase = tag.replace("-", " ").lower()
        initials = acronym(tag)
        if re.search(
            rf"(?<![a-z0-9]){re.escape(phrase)}(?:e?s)?(?![a-z0-9])", haystack
        ):
            continue
        if len(initials) >= 2 and re.search(
            rf"(?<![A-Za-z0-9]){re.escape(initials)}(?![A-Za-z0-9])", text
        ):
            continue
        missing.append(tag)
    return missing


NO_CONTROLLED_TAG = (
    "controlled_tags has no value from controlled_tag_options; the blog "
    "requires at least one, so copy the closest fit"
)


def controlled_tag_problems(answer: str | None, options: Sequence[str]) -> list[str]:
    """The taxonomy answer's controlled-tag problem: none of its
    comma-separated values is one of ``options`` (the blog lint requires at
    least one controlled tag)."""
    picked = {value.strip().lower() for value in (answer or "").split(",")}
    return (
        [] if picked & {option.lower() for option in options} else [NO_CONTROLLED_TAG]
    )
