"""Post file names and slugs: one post across slugify versions, and a
summary and a review of one paper apart."""

from __future__ import annotations

import hashlib
import re

SUMMARY_SLUG_SUFFIX = "-summary"
# Post files are named "<YYYY-MM-DD>-<slug>.md" (see Publisher.publish).
_DATED_POST = re.compile(r"^(\d{4}-\d{2}-\d{2})-(?P<slug>.+)$")


def title_slug(title: str, kind: str) -> str:
    """A new post's slug: a summary and a review of one paper must not share
    a file name, image names or URL."""
    base = slugify(title)
    return f"{base}{SUMMARY_SLUG_SUFFIX}" if kind == "summarize" else base


def slugify(title: str, max_length: int = 80) -> str:
    """Filesystem- and S3-safe ASCII slug for a document title, cut at a word
    boundary (the blog URL comes from it, so keep it readable and stable).

    A title whose letters are all non-ASCII (e.g. Korean) gets a stable hash
    suffix, so two such titles never share a file, S3 key or branch.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if len(slug) > max_length:
        cut = slug[: max_length + 1]
        slug = (cut.rsplit("-", 1)[0] if "-" in cut else cut[:max_length]).strip("-")
    if slug:
        return slug
    if not any(ch.isalnum() for ch in title):
        return "untitled"
    digest = hashlib.sha1(title.strip().encode("utf-8")).hexdigest()[:8]
    return f"untitled-{digest}"


def post_slug(stem: str) -> str:
    """A post file's stem without its date prefix."""
    match = _DATED_POST.match(stem)
    return match.group("slug") if match else stem


def post_key(slug: str) -> tuple[bool, str]:
    """``(is_summary, normalized base slug)`` naming one post across slugify
    versions: older posts kept dots and the double hyphen left by ": "
    ("gemini-2.5--pushing-…") and were cut at 120 characters, not at a word
    boundary near 80."""
    summary = slug.endswith(SUMMARY_SLUG_SUFFIX)
    base = slug[: -len(SUMMARY_SLUG_SUFFIX)] if summary else slug
    return summary, slugify(base)


def same_post(slug: str, other: str) -> bool:
    return post_key(slug) == post_key(other)
