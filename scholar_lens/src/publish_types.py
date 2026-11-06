"""Types shared by the publisher and the blog branch sync (no I/O)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..domain.post_names import (  # noqa: F401 - re-exported
    post_slug,
    same_post,
    slugify,
    title_slug,
)
from .constants import LocalPaths

if TYPE_CHECKING:  # a type only: keeps this module a leaf
    from ..domain.validation import ArtifactKind

# Blog image convention: one flat folder, referenced by absolute path.
BLOG_IMAGES_DIR = "assets/images"
# Scratch folder (inside the artifact's work_dir) holding the localized images
# that are uploaded to S3 and committed to the blog, one sub-folder per post
# slug: a paper's summary and review share the work_dir.
_STAGED_IMAGES_DIRNAME = "blog-images"


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
    # Artifact kind the post is validated as ("review", "summarize", "guide",
    # "survey").
    kind: ArtifactKind = "review"
    # Pipeline-side findings for the reviewer (e.g. a tag the paper never
    # mentions); shown in the PR checklist with the validator's warnings.
    review_notes: list[str] = field(default_factory=list)
    # Filled by Publisher.publish: non-blocking post-validation findings, shown
    # as a checklist in the PR body.
    validation_warnings: list[str] = field(default_factory=list)
    # The blog's post this one replaces (``_posts/<date>-<slug>.md``, see
    # Publisher.find_published): the post keeps that file name, URL and image
    # names.
    published_path: str | None = None

    @property
    def slug(self) -> str:
        """Post slug: the replaced post's, else the title's."""
        if self.published_path:
            return post_slug(Path(self.published_path).stem)
        return title_slug(self.title, self.kind)

    @property
    def staged_images_dir(self) -> Path:
        """This post's localized images; a rerun replaces only these."""
        return self.work_dir / _STAGED_IMAGES_DIRNAME / self.slug

    def pr_body_with_checklist(self, extra_notes: Sequence[str] = ()) -> str:
        return with_review_checklist(
            self.pr_body, [*self.validation_warnings, *extra_notes]
        )


def with_review_checklist(body: str, warnings: Sequence[str]) -> str:
    """Append post-validation warnings to a PR body as unchecked items."""
    if not warnings:
        return body
    items = "\n".join(f"- [ ] {w}" for w in warnings)
    return f"{body}\n\n### Review checklist\n{items}"


def is_post_file(path: str) -> bool:
    """A post directly in the blog's ``_posts/`` folder (the only place this
    tool writes posts)."""
    posts = LocalPaths.POSTS_DIR.value
    return (
        path.startswith(f"{posts}/") and path.count("/") == 1 and path.endswith(".md")
    )
