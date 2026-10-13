"""Types shared by the publisher and the blog branch sync (no I/O)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # a type only: keeps this module a leaf
    from .post_validator import ArtifactKind

# Blog image convention: one flat folder, referenced by absolute path.
BLOG_IMAGES_DIR = "assets/images"
# Scratch folder (inside the artifact's work_dir) holding the localized images
# that are uploaded to S3 and committed to the blog, one sub-folder per post
# slug: a paper's summary and review share the work_dir.
_STAGED_IMAGES_DIRNAME = "blog-images"
SUMMARY_SLUG_SUFFIX = "-summary"


class PublishError(Exception):
    """Publishing to the blog repository failed."""


@dataclass
class PublishRequest:
    """Everything needed to publish one artifact."""

    title: str
    markdown: str
    work_dir: Path
    branch_id: str
    pr_title: str
    pr_body: str
    commit_message: str
    # Artifact kind the post is validated as ("review" | "summarize" | "guide").
    kind: ArtifactKind = "review"
    # Pipeline-side findings for the reviewer (e.g. a tag the paper never
    # mentions); shown in the PR checklist with the validator's warnings.
    review_notes: list[str] = field(default_factory=list)
    # Filled by Publisher.publish: non-blocking post-validation findings, shown
    # as a checklist in the PR body.
    validation_warnings: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        """Post slug: a summary and a review of one paper must not share a
        file name, image names or URL."""
        base = slugify(self.title)
        return f"{base}{SUMMARY_SLUG_SUFFIX}" if self.kind == "summarize" else base

    @property
    def staged_images_dir(self) -> Path:
        """This post's localized images; a rerun replaces only these."""
        return self.work_dir / _STAGED_IMAGES_DIRNAME / self.slug

    def pr_body_with_checklist(self, extra_notes: Sequence[str] = ()) -> str:
        return with_review_checklist(
            self.pr_body, [*self.validation_warnings, *extra_notes]
        )


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


def with_review_checklist(body: str, warnings: Sequence[str]) -> str:
    """Append post-validation warnings to a PR body as unchecked items."""
    if not warnings:
        return body
    items = "\n".join(f"- [ ] {w}" for w in warnings)
    return f"{body}\n\n### Review checklist\n{items}"
