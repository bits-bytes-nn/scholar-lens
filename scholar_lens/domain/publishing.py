"""Types shared by the publisher and the blog branch sync (no I/O)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scholar_lens.domain.constants import LocalPaths
from scholar_lens.domain.markdown.front_matter import split_front_matter_text
from scholar_lens.domain.markdown.links import LinkSources, verify_links
from scholar_lens.domain.post_names import (
    post_slug,
    title_slug,
)
from scholar_lens.domain.validation import ArtifactKind

# Blog image convention: one flat folder, referenced by absolute path.
BLOG_IMAGES_DIR = "assets/images"
# An image a post shows: any reference into the blog's image folder.
BLOG_IMAGE_REF = re.compile(rf"/{re.escape(BLOG_IMAGES_DIR)}/([^\s)\"'<>?#]+)")
# A post that was not published is kept as ``<slug>.rejected.md``; why
# follows its text in a comment that opens with this line.
REJECTED_SUFFIX = ".rejected.md"
VIOLATIONS_OPENING = "<!-- violations\n"
# The saved publish request of ``<name>.md`` is ``<name>.request.json``.
REQUEST_SUFFIX = ".request.json"
# Inside the rejected area (S3): the rejected posts' localized images.
REJECTED_IMAGES_DIRNAME = "blog-images"
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
    # Problems the post may not be published with as is (a check that did
    # not run, a claim check that objects to most of a section); listed
    # first in the PR body, under "Blocking".
    blocking: list[str] = field(default_factory=list)
    # Filled by Publisher.publish: non-blocking post-validation findings, shown
    # as a checklist in the PR body.
    validation_warnings: list[str] = field(default_factory=list)
    # The blog's post this one replaces (``_posts/<date>-<slug>.md``, see
    # Publisher.find_published): the post keeps that file name, URL and image
    # names.
    published_path: str | None = None
    # The post file's name without ``.md`` once a publish named it (set by
    # Publisher.publish): a later ``publish`` of the saved post keeps it.
    post_name: str | None = None
    # What the post may link (its sources, see :func:`verify_post_links`),
    # set by every generating run. None for a saved post published again:
    # its links were checked when it was written, and edits since are the
    # user's.
    link_sources: LinkSources | None = None

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
            self.pr_body,
            [*self.validation_warnings, *extra_notes],
            blocking=self.blocking,
        )


def verify_post_links(markdown: str, sources: LinkSources) -> tuple[str, list[str]]:
    """``markdown`` (front matter and body) with every body link to a
    target ``sources`` do not allow reduced to its text
    (:func:`verify_links`), and a run note naming the removed targets
    (none when nothing was removed). Every kind of post publishes through
    this one check."""
    front, body = split_front_matter_text(markdown)
    body, removed = verify_links(body, sources)
    if not removed:
        return markdown, []
    return f"{front}{body}", [
        f"Unlinked {len(removed)} link(s) to targets that are not among the "
        "post's sources (kept as text): " + ", ".join(removed)
    ]


def with_review_checklist(
    body: str, warnings: Sequence[str], *, blocking: Sequence[str] = ()
) -> str:
    """Append the blocking problems, then the post-validation warnings, to a
    PR body as unchecked items."""
    parts = [body]
    if blocking:
        items = "\n".join(f"- [ ] {b}" for b in blocking)
        parts.append(
            "### Blocking\nDo not merge before each of these is resolved:\n" + items
        )
    if warnings:
        items = "\n".join(f"- [ ] {w}" for w in warnings)
        parts.append(f"### Review checklist\n{items}")
    return "\n\n".join(parts)


def is_post_file(path: str) -> bool:
    """A post directly in the blog's ``_posts/`` folder (the only place this
    tool writes posts)."""
    posts = LocalPaths.POSTS_DIR.value
    return (
        path.startswith(f"{posts}/") and path.count("/") == 1 and path.endswith(".md")
    )


def request_name(markdown_name: str) -> str:
    """The saved publish request's name (a file or S3 key) next to the post
    ``markdown_name`` (``x.md`` or ``x.rejected.md``)."""
    return markdown_name.removesuffix(".md") + REQUEST_SUFFIX


def without_violations(markdown: str) -> str:
    """A rejected post's text without the violations comment the publisher
    appended (any other text unchanged)."""
    at = markdown.rfind("\n\n" + VIOLATIONS_OPENING)
    if at == -1 or not markdown.rstrip().endswith("-->"):
        return markdown
    return markdown[:at].rstrip("\n") + "\n"


class SavedPublishRequest(BaseModel):
    """A :class:`PublishRequest` without the post's text, saved as JSON next
    to the saved or rejected post (and in S3): ``scholar-lens publish``
    reads it back to check the (possibly edited) post again and open its PR
    on the same branch. A field this version does not know fails loudly; a
    missing optional field takes its default."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    title: str
    # Where the run kept the post's files (a Batch run's is gone: publish
    # then uses a local directory of the same name).
    work_dir: str
    branch_id: str
    pr_title: str
    pr_body: str
    commit_message: str
    kind: ArtifactKind = "review"
    review_notes: list[str] = Field(default_factory=list)
    blocking: list[str] = Field(default_factory=list)
    published_path: str | None = None
    post_name: str | None = None

    @classmethod
    def of(cls, request: PublishRequest) -> SavedPublishRequest:
        """``request`` without its text (and its validation warnings, which
        a publish computes again)."""
        return cls(
            title=request.title,
            work_dir=str(request.work_dir),
            branch_id=request.branch_id,
            pr_title=request.pr_title,
            pr_body=request.pr_body,
            commit_message=request.commit_message,
            kind=request.kind,
            review_notes=list(request.review_notes),
            blocking=list(request.blocking),
            published_path=request.published_path,
            post_name=request.post_name,
        )

    def request(self, markdown: str, work_dir: Path) -> PublishRequest:
        """The publish request of ``markdown``, its files in ``work_dir``."""
        return PublishRequest(
            title=self.title,
            markdown=markdown,
            work_dir=work_dir,
            branch_id=self.branch_id,
            pr_title=self.pr_title,
            pr_body=self.pr_body,
            commit_message=self.commit_message,
            kind=self.kind,
            review_notes=list(self.review_notes),
            blocking=list(self.blocking),
            published_path=self.published_path,
            post_name=self.post_name,
        )


# Footer shared by every auto-generated blog PR body.
_PR_FOOTER = "This pull request was automatically generated by the Scholar-Lens system."


def build_pr_body(
    *,
    kind: str,
    title: str,
    fields: dict[str, str],
) -> str:
    """Build a consistent blog-PR description across all artifact types.

    ``kind`` is the human label ("paper review" / "paper summary" / "technical
    guide"); ``fields`` are the artifact-specific metadata rows (rendered as
    ``- **Key**: value``), in insertion order. Empty-valued fields are skipped so
    each artifact only shows what's relevant, while the header/footer stay
    identical; the review checklist is added at publish time."""
    header = f"This PR adds an AI-generated {kind}: **{title}**"
    rows = "\n".join(f"- **{k}**: {v}" for k, v in fields.items() if v)
    return f"{header}\n\n{rows}\n\n{_PR_FOOTER}"
