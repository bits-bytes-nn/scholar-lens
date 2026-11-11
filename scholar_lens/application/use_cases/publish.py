"""``scholar-lens publish``: check a saved or rejected post again and open
its blog PR.

Every run saves its :class:`~scholar_lens.domain.publishing.PublishRequest`
without the post text (:class:`~scholar_lens.domain.publishing.SavedPublishRequest`)
next to the saved or rejected post, locally and in S3. ``publish`` reads the
(possibly edited) post and that request back, runs the publisher's
deterministic post-processing and validation again and opens or updates the
PR on the request's branch, exactly as the run's last step would. It runs
locally: a Batch run's post is read from S3.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from scholar_lens.application.ports import FileLocationError, Files
from scholar_lens.application.run import PipelineOutcome, RunContext, publish_artifact
from scholar_lens.domain.jobs import ArtifactKind
from scholar_lens.domain.publishing import (
    BLOG_IMAGE_REF,
    BLOG_IMAGES_DIR,
    REJECTED_IMAGES_DIRNAME,
    REJECTED_SUFFIX,
    PublishRequest,
    SavedPublishRequest,
    request_name,
    without_violations,
)

logger = logging.getLogger(__name__)


# The job kind a post's validation kind reports as.
RESULT_KINDS: dict[str, ArtifactKind] = {
    "review": "review",
    "summarize": "summarize",
    "guide": "guide",
    "survey": "survey_write",
}


class SavedPostError(ValueError):
    """The post or its saved publish request cannot be read: a rejected
    input (nothing was published)."""


def load_saved_post(context: RunContext, target: str) -> PublishRequest:
    """The publish request of the saved or rejected post at ``target``.

    Args:
        context: The run context (its bucket, prefix and work directory).
        target: A local ``.md`` path, or ``s3://<bucket>/<key>`` in the
            configured bucket and prefix.

    Returns:
        The request, with the post's text (its violations comment removed)
        and a work directory holding its staged images.

    Raises:
        SavedPostError: No such post, no request next to it, or a request
            this version does not read.
    """
    if target.startswith("s3://"):
        return _from_s3(context, target)
    return _from_file(context.files, Path(target).expanduser())


def _parse(text: str, where: str) -> SavedPublishRequest:
    try:
        return SavedPublishRequest.model_validate_json(text)
    except ValidationError as e:
        raise SavedPostError(
            f"The publish request {where} is not one this version reads: {e}"
        ) from e


def _read(files: Files, location: str) -> str | None:
    try:
        return files.read_text(location)
    except FileLocationError as e:
        raise SavedPostError(str(e)) from e
    except UnicodeDecodeError as e:
        raise SavedPostError(f"{location} is not UTF-8: {e}") from e


def _from_file(files: Files, path: Path) -> PublishRequest:
    markdown = _read(files, str(path)) if path.suffix == ".md" else None
    if markdown is None:
        raise SavedPostError(f"No Markdown post at {path}.")
    request_path = path.with_name(request_name(path.name))
    raw = _read(files, str(request_path))
    if raw is None:
        raise SavedPostError(
            f"No publish request next to {path} ({request_path.name}): only a "
            "post a run saved or rejected can be published."
        )
    saved = _parse(raw, str(request_path))
    return saved.request(without_violations(markdown), path.parent.resolve())


def _from_s3(context: RunContext, uri: str) -> PublishRequest:
    handler = context.s3_handler
    if handler is None:
        raise SavedPostError(f"No S3 bucket is configured to read {uri} from.")
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    prefix = context.config.resources.s3_prefix.strip("/")
    scope = f"{prefix}/" if prefix else ""
    if bucket != handler.bucket_name or not key.startswith(scope):
        raise SavedPostError(
            f"The post {uri!r} is not under s3://{handler.bucket_name}/{scope}."
        )
    if not key.endswith(".md"):
        raise SavedPostError(f"Not a Markdown post: {uri}.")
    files = context.files
    markdown = _read(files, uri)
    if markdown is None:
        raise SavedPostError(f"No post at {uri}.")
    request_uri = f"s3://{bucket}/{request_name(key)}"
    raw = _read(files, request_uri)
    if raw is None:
        raise SavedPostError(
            f"No publish request next to {uri}: only a post a run saved or "
            "rejected can be published."
        )
    saved = _parse(raw, request_uri)
    # The run's work directory is a Batch container's: a local one of the
    # same name (where a local run would have kept the post) holds it now.
    work_dir = context.papers_dir / Path(saved.work_dir).name
    files.make_dir(str(work_dir))
    request = saved.request(without_violations(markdown), work_dir)
    request.review_notes += _fetch_images(files, bucket, key, scope, request)
    return request


def _fetch_images(
    files: Files, bucket: str, key: str, scope: str, request: PublishRequest
) -> list[str]:
    """The post's localized images, copied from S3 into its staging folder
    (where the publisher keeps the images the post shows): a rejected post's
    from ``rejected/blog-images/<slug>/``, a saved post's from the uploaded
    blog images. Returns a review note naming any that are not there."""
    folder = key.rpartition("/")[0]
    source = (
        f"{folder}/{REJECTED_IMAGES_DIRNAME}/{request.slug}"
        if key.endswith(REJECTED_SUFFIX)
        else f"{scope}{BLOG_IMAGES_DIR}"
    ).lstrip("/")
    names = sorted(
        {
            name
            for name in BLOG_IMAGE_REF.findall(request.markdown)
            # Only this post's own images, by plain file name.
            if name.startswith(f"{request.slug}-") and "/" not in name
        }
    )
    staged = request.staged_images_dir
    files.make_dir(str(staged))
    missing = []
    for name in names:
        data = files.read_bytes(f"s3://{bucket}/{source}/{name}")
        if data is None:
            missing.append(name)
        else:
            files.write_bytes(str(staged / name), data)
    if not missing:
        return []
    return [
        "Images the post shows that were not found in S3 (check the blog has "
        f"them): {', '.join(missing)}"
    ]


RESOLVED_NOTE = "Marked resolved by the editor at publish (check before merging): "


def mark_resolved(request: PublishRequest) -> None:
    """The editor states the run's blocking problems are fixed in the
    (edited) post: they stop blocking and stay in the PR checklist as items
    to check, so the claim is reviewed rather than the problem dropped."""
    request.review_notes += [f"{RESOLVED_NOTE}{b}" for b in request.blocking]
    request.blocking = []


async def publish_saved_post(
    context: RunContext,
    request: PublishRequest,
    *,
    create_pr: bool,
    resolved: bool = False,
) -> PipelineOutcome:
    """Run the publisher's post-processing and validation on ``request``
    again and open (or update) its blog PR, exactly as the run's last step
    would. ``resolved``: the editor fixed the run's blocking problems
    (:func:`mark_resolved`)."""
    if resolved:
        mark_resolved(request)
    return await publish_artifact(context, request, create_pr=create_pr)
