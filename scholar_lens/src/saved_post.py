"""``scholar-lens publish``: check a saved or rejected post again and open
its blog PR.

Every run saves its :class:`~.publish_types.PublishRequest` without the post
text (:class:`~.publish_types.SavedPublishRequest`) next to the saved or
rejected post, locally and in S3. ``publish`` reads the (possibly edited)
post and that request back, runs the publisher's deterministic
post-processing and validation (:meth:`~.publisher.Publisher.publish`) again
and opens or updates the PR on the request's branch, exactly as the run's
last step would. It runs locally: a Batch run's post is read from S3.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import ValidationError

from ..adapters.aws.storage import S3Handler
from ..configs import Config
from .constants import RUN_SECRETS
from .jobs import ArtifactKind, JobResult
from .logger import logger
from .publish_types import (
    BLOG_IMAGE_REF,
    BLOG_IMAGES_DIR,
    REJECTED_IMAGES_DIRNAME,
    REJECTED_SUFFIX,
    PublishRequest,
    SavedPublishRequest,
    request_name,
    without_violations,
)
from .runtime import (
    RunContext,
    build_context,
    check_pr_token,
    load_secrets_from_ssm,
    publish_artifact,
    record_error,
    record_outcome,
    run_coroutine,
)

# The job kind a post's validation kind reports as.
_RESULT_KINDS: dict[str, ArtifactKind] = {
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
    return _from_file(Path(target).expanduser())


def _parse(text: str, where: str) -> SavedPublishRequest:
    try:
        return SavedPublishRequest.model_validate_json(text)
    except ValidationError as e:
        raise SavedPostError(
            f"The publish request {where} is not one this version reads: {e}"
        ) from e


def _from_file(path: Path) -> PublishRequest:
    if not path.is_file() or path.suffix != ".md":
        raise SavedPostError(f"No Markdown post at {path}.")
    request_path = path.with_name(request_name(path.name))
    if not request_path.is_file():
        raise SavedPostError(
            f"No publish request next to {path} ({request_path.name}): only a "
            "post a run saved or rejected can be published."
        )
    saved = _parse(request_path.read_text(encoding="utf-8"), str(request_path))
    markdown = without_violations(path.read_text(encoding="utf-8"))
    return saved.request(markdown, path.parent.resolve())


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
    markdown = handler.get_text(key)
    if markdown is None:
        raise SavedPostError(f"No post at {uri}.")
    raw = handler.get_text(request_name(key))
    if raw is None:
        raise SavedPostError(
            f"No publish request next to {uri}: only a post a run saved or "
            "rejected can be published."
        )
    saved = _parse(raw, f"s3://{bucket}/{request_name(key)}")
    # The run's work directory is a Batch container's: a local one of the
    # same name (where a local run would have kept the post) holds it now.
    work_dir = context.papers_dir / Path(saved.work_dir).name
    work_dir.mkdir(parents=True, exist_ok=True)
    request = saved.request(without_violations(markdown), work_dir)
    request.review_notes += _fetch_images(handler, key, scope, request)
    return request


def _fetch_images(
    handler: S3Handler, key: str, scope: str, request: PublishRequest
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
    staged.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in names:
        data = handler.get_bytes(f"{source}/{name}")
        if data is None:
            missing.append(name)
        else:
            (staged / name).write_bytes(data)
    if not missing:
        return []
    return [
        "Images the post shows that were not found in S3 (check the blog has "
        f"them): {', '.join(missing)}"
    ]


def run_publish(target: str, *, create_pr: bool) -> JobResult:
    """Check the saved or rejected post at ``target`` again and open its PR.

    The result is a run's: ``succeeded``; ``failed`` (the post broke a blog
    rule or its images could not be stored, with the rejected copy's paths,
    or the PR failed, with ``pr_error``); ``rejected`` when the post or its
    publish request cannot be read. Never raises for these.

    Args:
        target: A local ``.md`` path or an ``s3://`` URI (see
            :func:`load_saved_post`).
        create_pr: Open (or update) the blog PR when GitHub is enabled.

    Returns:
        The result, as a run reports it.
    """
    result = JobResult(kind=None, input=target)
    started = time.monotonic()
    try:
        context = build_context(Config.load())
    except Exception as e:  # noqa: BLE001 - reported, publishing cannot start
        result.error = f"{type(e).__name__}: {e}"
        logger.error("Publish could not start: %s", e)
        return result
    load_secrets_from_ssm(context, RUN_SECRETS)
    try:
        request = load_saved_post(context, target)
        result.kind = _RESULT_KINDS.get(request.kind)
        check_pr_token(context, create_pr)
        outcome = run_coroutine(publish_artifact(context, request, create_pr=create_pr))
        record_outcome(result, outcome)
    except Exception as e:  # noqa: BLE001 - folded into the result
        record_error(result, e, (SavedPostError,))
    finally:
        result.duration_seconds = round(time.monotonic() - started, 1)
    return result
