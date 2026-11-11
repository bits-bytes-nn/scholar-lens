"""What every use case shares: the run's context (its settings and the
blog, publisher and bucket ports), the outcome a use case returns, and the
publishing steps every post goes through."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scholar_lens.application.ports import (
    BlogSource,
    Files,
    ObjectStore,
    PostPublisher,
    RunSettings,
)
from scholar_lens.domain.constants import LocalPaths, S3Paths
from scholar_lens.domain.conventions import BlogPost, PostConventions, tag_counts
from scholar_lens.domain.post_format import with_spec_cover
from scholar_lens.domain.publishing import (
    PublishError,
    PublishRequest,
    verify_post_links,
)
from scholar_lens.domain.tags import format_tag
from scholar_lens.domain.validation import PAPER_KINDS

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RunContext:
    """Per-run context shared by every use case: the settings, the blog,
    the publisher and the bucket (built by the composition root)."""

    config: RunSettings
    # The blog's conventions and posts.
    blog: BlogSource
    # Builds the run's publisher (once the conventions are known).
    publisher_factory: Callable[[RunContext], PostPublisher]
    # The run's bucket, when one is configured.
    s3_handler: ObjectStore | None = None
    # Local files and the run's bucket by location (a path or ``s3://``).
    files: Files
    # The run's working directory (papers, figures, generated posts).
    root_dir: Path
    # The run ignores its saved stage checkpoints and reading cache (--fresh).
    fresh: bool = False
    # The blog's post conventions, loaded once per run.
    conventions: PostConventions | None = None
    # The blog's posts (tags, descriptions, translation ids), loaded on first
    # use, and the tags they use that a new post may reuse, with their post
    # counts, most used first.
    blog_posts: list[BlogPost] | None = None
    blog_tags: dict[str, int] | None = None
    # Why the blog's posts could not be listed (they are then none).
    blog_listing_error: str | None = None

    @property
    def papers_dir(self) -> Path:
        """Where the run keeps each paper's and guide's working files."""
        return self.root_dir / LocalPaths.PAPERS_DIR.value

    def require_conventions(self) -> PostConventions:
        if self.conventions is None:
            self.conventions = self.blog.conventions()
        return self.conventions

    def require_blog_posts(self) -> list[BlogPost]:
        if self.blog_posts is None:
            try:
                self.blog_posts = self.blog.posts()
            except Exception as e:  # noqa: BLE001 - hints and a CI-backed check only
                logger.warning("Could not list the blog's posts: %s", e)
                self.blog_posts, self.blog_listing_error = [], str(e)
        return self.blog_posts

    def require_blog_tags(self) -> list[str]:
        if self.blog_tags is None:
            spec = self.require_conventions()
            retired = {k.lower() for k in spec.tags.aliases}
            controlled = set(spec.tags.controlled)
            words = spec.tags.lowercase_words
            self.blog_tags = {
                t: n
                for t, n in tag_counts(self.require_blog_posts()).items()
                if t.lower() not in retired
                and t not in controlled
                and format_tag(t, words) == t  # only spellings a post may use
            }
        return list(self.blog_tags)

    def blog_tag_hints(self, limit: int) -> list[str]:
        """The most used blog tags worth suggesting: a tag on a single post is
        often a one-off, so only tags shared by two or more posts are hints."""
        tags = self.require_blog_tags()
        counts = self.blog_tags or {}
        return [t for t in tags if counts[t] >= 2][:limit]


@dataclass
class PipelineOutcome:
    """What a use case produced; folded into the run's job result."""

    title: str
    # A post; a survey plan run produces the plan file instead.
    markdown_path: Path | None = None
    plan_path: Path | None = None
    s3_uri: str | None = None
    pr_url: str | None = None
    # The blog PR was requested and enabled but could not be opened.
    pr_error: str | None = None
    warnings: list[str] = field(default_factory=list)
    # Problems the post may not be merged with as is (also first in
    # ``warnings``, marked, for readers of the warnings alone).
    blocking: list[str] = field(default_factory=list)


def blocking_notes(blocking: Sequence[str]) -> list[str]:
    """Blocking problems as the job result lists them (first, marked)."""
    return [f"Blocking: {b}" for b in blocking]


def check_post_type(
    spec: PostConventions, category: str, setting: str, *, artifact: str
) -> None:
    """Fail before any model call when ``resources.github.<setting>`` cannot
    label the post (publishing would reject it after the paid work)."""
    paper = artifact in PAPER_KINDS
    if problem := spec.type_problem(category, paper=paper, artifact=artifact):
        raise ValueError(
            f"resources.github.{setting}: {problem}. Set it to one of the blog's "
            "types or add the type to the blog's _data/post_conventions.yml."
        )


def refresh_conventions(context: RunContext, request: PublishRequest) -> None:
    """Re-read the blog's conventions before publishing: a long run may
    outlive a blog change (e.g. a topic's cover). When they differ from the
    ones the run started with, the post is validated against the new ones,
    its ``cover`` is recomputed from its categories (:func:`with_spec_cover`)
    and a review note says so. Fields that need a model call (the topic,
    tags, description) are left for the validator to check. An unreadable
    blog keeps the conventions the run started with."""
    started = context.conventions
    if not isinstance(started, PostConventions):
        # Never loaded: the publisher reads the current ones now.
        return
    try:
        fresh = context.blog.conventions(fallback=False)
    except Exception as e:  # noqa: BLE001 - the run's conventions still apply
        logger.warning(
            "Could not re-read the blog's post conventions before publishing "
            "(%s); validating with the ones the run started with.",
            e,
        )
        return
    if fresh == started:
        return
    context.conventions = fresh
    request.markdown, cover = with_spec_cover(request.markdown, fresh)
    note = (
        "Blog conventions changed during the run: the post was validated "
        "against the blog's current conventions"
        + (f" and its cover recomputed ('{cover}')" if cover else "")
        + "; check its topic, tags and description still fit them."
    )
    logger.warning(note)
    request.review_notes.append(note)


async def find_published_post(
    context: RunContext, *, slug: str, branch_id: str, category: str
) -> tuple[BlogPost | None, list[str]]:
    """The blog's post of ``category`` a new post replaces
    (:meth:`PostPublisher.find_published`) and review notes: when the blog's
    posts or pull requests cannot be read, the post is published as new, and
    the note asks to check it duplicates none."""
    publisher = context.publisher_factory(context)
    notes: list[str] = []
    if context.blog_listing_error:
        notes.append(
            f"Could not list the blog's posts ({context.blog_listing_error}); "
            "this post was written as a new one: check the blog has no older "
            "copy of it, and that its tags and description fit the blog."
        )
    try:
        found = await asyncio.to_thread(
            publisher.find_published, slug, branch_id, category
        )
        return found, notes
    except PublishError as e:
        logger.warning(
            "Could not look up the posts branch '%s' published: %s", branch_id, e
        )
        return None, [
            *notes,
            f"Could not read which post an earlier run of this artifact published "
            f"({e}); check the blog has no older copy of this post.",
        ]


async def publish_artifact(
    context: RunContext, request: PublishRequest, *, create_pr: bool
) -> PipelineOutcome:
    """Save + upload the document, then open the blog PR when enabled. The
    body's links are checked against the post's sources first
    (:func:`verify_post_links`), then the blog's conventions are re-read
    (:func:`refresh_conventions`).

    The outcome's warnings are the post's checklist (review notes, rule slips,
    validator warnings), so ``--no-pr`` runs report them too.

    A PR failure is kept on the outcome (``pr_error``) next to the saved
    artifact's paths; the run is reported as failed.
    """
    if request.link_sources is not None:
        request.markdown, notes = verify_post_links(
            request.markdown, request.link_sources
        )
        request.review_notes += notes
    await asyncio.to_thread(refresh_conventions, context, request)
    publisher = context.publisher_factory(context)
    try:
        s3_uri, markdown_path = await publisher.publish(request)
    except Exception as e:
        # The generators' notes (skipped sections, degraded parts) still
        # belong on the failed run's result.
        e.run_notes = [  # type: ignore[attr-defined]
            *blocking_notes(request.blocking),
            *request.review_notes,
        ]
        e.run_blocking = list(request.blocking)  # type: ignore[attr-defined]
        raise
    outcome = PipelineOutcome(
        title=request.title,
        markdown_path=markdown_path,
        s3_uri=s3_uri,
        # Review notes, rule slips and validator warnings: the PR checklist,
        # and the job result for runs without a PR.
        warnings=[
            *blocking_notes(request.blocking),
            *request.validation_warnings,
        ],
        blocking=list(request.blocking),
    )
    github = context.config.resources.github
    if create_pr and not github.enabled:
        outcome.warnings.append(
            "No blog PR: resources.github.enabled is false (pass --no-pr to "
            "skip this note)."
        )
    if create_pr and github.enabled:
        try:
            outcome.pr_url = await publisher.create_pull_request(request, markdown_path)
        except PublishError as e:
            logger.error("Blog PR failed: %s", e)
            outcome.pr_error = str(e)
    return outcome


async def load_translation_guideline(
    context: RunContext,
) -> list[dict[str, Any]] | None:
    """The glossary-style translation guideline (curated term renderings that
    reviews, summaries and guides follow): the local file when present, else
    the bucket's copy. It is optional, so a missing object yields ``None``;
    any other bucket error still raises (before any paid stage runs)."""
    guideline_path = (
        context.root_dir
        / LocalPaths.ASSETS_DIR.value
        / LocalPaths.TRANSLATION_GUIDELINE_FILE.value
    )
    text = await asyncio.to_thread(context.files.read_text, str(guideline_path))
    if text is not None:
        return _guideline(text, str(guideline_path))
    if not context.s3_handler:
        return None
    s3_key = (
        f"{context.config.resources.s3_prefix}/"
        f"{S3Paths.TRANSLATION_GUIDELINE.value}/{guideline_path.name}"
    ).lstrip("/")
    location = f"s3://{context.s3_handler.bucket_name}/{s3_key}"
    text = await asyncio.to_thread(context.files.read_text, location)
    if text is None:
        logger.info("No translation guideline at '%s'; continuing without.", s3_key)
        return None
    return _guideline(text, location)


def _guideline(text: str, where: str) -> list[dict[str, Any]]:
    """The guideline's entries.

    Raises:
        ValueError: The text is not a JSON list of objects.
    """
    data = json.loads(text)
    if not isinstance(data, list) or not all(isinstance(e, dict) for e in data):
        raise ValueError(
            f"The translation guideline at '{where}' is not a JSON list of objects."
        )
    return data
