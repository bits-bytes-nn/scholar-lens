"""Run one :class:`~scholar_lens.src.jobs.JobRequest` end to end.

:func:`run_job` is the single entry for every artifact kind, whether invoked by
the CLI locally or by the Batch worker. It builds the AWS/config context, loads
secrets, dispatches to the paper or guide pipeline, and — whatever happens —
records metrics, sends the SNS notification, and (for Batch runs) stores the
:class:`~scholar_lens.src.jobs.JobResult` in S3.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict

from ..configs import Config
from .aws_helpers import S3Handler, get_ssm_param_value
from .constants import RUN_SECRETS, EnvVars, SSMParams
from .jobs import JobRequest, JobResult, ssm_base, write_result_manifest
from .logger import is_running_in_aws, logger
from .metrics import MetricsEmitter, TokenUsageTracker
from .post_conventions import PostConventions, fetch_blog_tags, load_conventions
from .post_validator import PostValidationError
from .publisher import Publisher, PublishError, PublishRequest
from .tags import format_tag

# Working directory for papers, figures and generated posts: the container's
# scratch space in AWS, the repository root locally.
ROOT_DIR: Path = (
    Path("/tmp") if is_running_in_aws() else Path(__file__).resolve().parents[2]
)


class RunContext(BaseModel):
    """Per-run AWS/config context shared by every entrypoint."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: Config
    default_boto_session: boto3.Session
    bedrock_boto_session: boto3.Session
    s3_handler: S3Handler | None = None
    # The blog's post conventions, loaded once per run (see load_run_conventions).
    conventions: PostConventions | None = None

    # Tags the blog already uses (reuse candidates) with their post counts,
    # most used first; loaded on first use.
    blog_tags: dict[str, int] | None = None

    def require_conventions(self) -> PostConventions:
        if self.conventions is None:
            self.conventions = load_run_conventions(self.config)
        return self.conventions

    def require_blog_tags(self) -> list[str]:
        if self.blog_tags is None:
            spec = self.require_conventions()
            retired = {k.lower() for k in spec.tags.aliases}
            controlled = set(spec.tags.controlled)
            words = spec.tags.lowercase_words
            self.blog_tags = {
                t: n
                for t, n in load_run_blog_tags(self.config).items()
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


def build_context(config: Config) -> RunContext:
    """Build a :class:`RunContext` (boto sessions + optional S3 handler).

    The AWS profile comes from the env in Batch (instance role / ``AWS_PROFILE``)
    and from config locally.
    """
    profile_name = (
        os.getenv(EnvVars.AWS_PROFILE_NAME.value)
        if is_running_in_aws()
        else config.resources.profile_name
    )
    default_session = boto3.Session(
        profile_name=profile_name, region_name=config.resources.default_region_name
    )
    context = RunContext(
        config=config,
        default_boto_session=default_session,
        bedrock_boto_session=boto3.Session(
            profile_name=profile_name,
            region_name=config.resources.bedrock_region_name,
        ),
    )
    if config.resources.s3_bucket_name:
        context.s3_handler = S3Handler(default_session, config.resources.s3_bucket_name)
    return context


def load_secrets_from_ssm(
    context: RunContext, ssm_map: dict[SSMParams, EnvVars]
) -> None:
    """Load the given SSM SecureString params into ``os.environ`` (AWS only).

    Every secret is optional: one that was never written is skipped quietly; any
    other failure (e.g. AccessDenied) is logged as a warning and the run goes on
    without it.
    """
    if not is_running_in_aws():
        return
    resources = context.config.resources
    base = ssm_base(resources.project_name, resources.stage)
    for ssm_param, env_var in ssm_map.items():
        try:
            os.environ[env_var.value] = get_ssm_param_value(
                context.default_boto_session, f"{base}/{ssm_param.value}"
            )
            logger.info("Set env var '%s' from SSM.", env_var.value)
        except Exception as e:  # noqa: BLE001 - optional secret; never abort the run
            if _is_parameter_not_found(e):
                logger.info("SSM secret for '%s' is not set.", env_var.value)
            else:
                logger.warning("Could not read '%s' from SSM: %s", env_var.value, e)


def _is_parameter_not_found(error: BaseException) -> bool:
    current: BaseException | None = error
    while current is not None:
        if (
            isinstance(current, ClientError)
            and current.response.get("Error", {}).get("Code") == "ParameterNotFound"
        ):
            return True
        current = current.__cause__
    return False


PROJECT = "scholar-lens"


def format_alarm(
    *,
    event: str,
    status: str,
    fields: dict[str, str],
    project: str = PROJECT,
    timestamp: datetime | None = None,
) -> tuple[str, str]:
    """Build a ``(subject, message)`` pair in the project family's unified alarm
    format, shared verbatim across omnisummary/tech-digest/paper-bridge:

        Subject: [<project>] <event> — <STATUS>

        <event> <STATUS>

        Key:   Value

        — 2026-06-10 04:12:00 UTC

    ``status`` is a short uppercase state (``FAILED``/``ALERT``). ``fields`` is an
    ordered mapping; single-line values render as an aligned ``Key: Value`` block,
    multi-line values render under their own ``Key:`` header. Omit a row by leaving
    it out of the dict.
    """
    ts = (timestamp or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[{project}] {event} — {status}"

    inline = {k: v for k, v in fields.items() if "\n" not in v}
    block = {k: v for k, v in fields.items() if "\n" in v}

    lines = [f"{event} {status}", ""]
    if inline:
        width = max(len(k) for k in inline)
        lines += [f"{k + ':':<{width + 1}} {v}" for k, v in inline.items()]
    for k, v in block.items():
        lines += ["", f"{k}:", v.strip("\n")]
    lines.append("")
    lines.append(f"— {ts} UTC")

    return subject, "\n".join(lines)


def publish_sns(
    session: boto3.Session, topic_arn: str, *, subject: str, message: str
) -> bool:
    """Publish a plain-text SNS notification; a failure must not fail the job.

    Returns whether the notification was sent.
    """
    try:
        session.client("sns").publish(
            TopicArn=topic_arn, Subject=subject, Message=message
        )
        return True
    except Exception as e:  # noqa: BLE001 - notification failure must not fail the job
        logger.warning("Failed to send SNS notification: %s", e)
        return False


def load_run_conventions(config: Config) -> PostConventions:
    """The blog's base-branch conventions when GitHub is configured, else the
    bundled copy."""
    github = config.resources.github
    return load_conventions(
        repo_name=github.repo_name,
        branch=github.base_branch,
        token=os.getenv(EnvVars.GITHUB_TOKEN.value),
        use_blog=github.enabled,
    )


def load_run_blog_tags(config: Config) -> dict[str, int]:
    """The blog's existing tags with their post counts, or none when GitHub is
    not configured or the listing fails (the model then proposes fresh short
    tags)."""
    github = config.resources.github
    if not (github.enabled and github.repo_name):
        return {}
    try:
        return fetch_blog_tags(
            github.repo_name, github.base_branch, os.getenv(EnvVars.GITHUB_TOKEN.value)
        )
    except Exception as e:  # noqa: BLE001 - reuse hints only; never fail the run
        logger.warning("Could not list the blog's tags: %s", e)
        return {}


def build_publisher(context: RunContext) -> Publisher:
    """Construct the artifact Publisher from the run context."""
    return Publisher(
        context.config.resources.github,
        root_dir=ROOT_DIR,
        s3_handler=context.s3_handler,
        s3_bucket_name=context.config.resources.s3_bucket_name,
        s3_prefix=context.config.resources.s3_prefix,
        conventions=context.require_conventions(),
    )


@dataclass
class PipelineOutcome:
    """What a pipeline produced; folded into the run's :class:`JobResult`."""

    title: str
    markdown_path: Path
    s3_uri: str | None = None
    pr_url: str | None = None
    warnings: list[str] = field(default_factory=list)


async def publish_artifact(
    context: RunContext, request: PublishRequest, *, create_pr: bool
) -> PipelineOutcome:
    """Save + upload the document, then open the blog PR when enabled.

    The outcome's warnings are the post's checklist (review notes, rule slips,
    validator warnings), so ``--no-pr`` runs report them too.

    A PR failure does not fail the run (the artifact already exists locally and
    in S3); it is reported as a warning on the result instead.
    """
    publisher = build_publisher(context)
    try:
        s3_uri, markdown_path = await publisher.publish(request)
    except Exception as e:
        # The generators' notes (skipped sections, degraded parts) still
        # belong on the failed run's result.
        e.run_notes = list(request.review_notes)  # type: ignore[attr-defined]
        raise
    outcome = PipelineOutcome(
        title=request.title,
        markdown_path=markdown_path,
        s3_uri=s3_uri,
        # Review notes, rule slips and validator warnings: the PR checklist,
        # and the job result for runs without a PR.
        warnings=list(request.validation_warnings),
    )
    if create_pr and context.config.resources.github.enabled:
        try:
            outcome.pr_url = await publisher.create_pull_request(request, markdown_path)
        except PublishError as e:
            logger.warning("Blog PR failed: %s", e)
            outcome.warnings.append(f"Blog PR not created: {e}")
    return outcome


def _is_fetch_outage(error: BaseException) -> bool:
    """A transport failure or 429/5xx while fetching the source: the request
    is fine, so the run failed rather than being rejected."""
    from .paper_source import PaperFetchError

    return isinstance(error, PaperFetchError)


def _rejected_input_errors() -> tuple[type[BaseException], ...]:
    """Errors caused by the request itself (bad URL, not a paper, not docs).

    Imported lazily so ``status``/``submit`` never load the paper stack.
    """
    from .arxiv_handler import ArxivNotFoundError
    from .paper_source import PaperSourceError
    from .tech_guide import NotTechnicalContentError
    from .url_guard import UnsafeUrlError

    return (
        ArxivNotFoundError,
        PaperSourceError,
        NotTechnicalContentError,
        UnsafeUrlError,
    )


async def _run_pipeline(
    context: RunContext, request: JobRequest, tracker: TokenUsageTracker
) -> PipelineOutcome:
    if request.kind == "guide":
        from .guide_pipeline import run_guide_pipeline

        return await run_guide_pipeline(context, request, tracker)
    from .paper_pipeline import run_paper_pipeline

    return await run_paper_pipeline(context, request, tracker)


def run_job(request: JobRequest, *, job_id: str | None = None) -> JobResult:
    """Run ``request`` and return its result; never raises for pipeline errors
    (a ``SystemExit``/``KeyboardInterrupt`` still propagates after cleanup)."""
    result = JobResult(kind=request.kind, input=request.describe(), job_id=job_id)
    try:
        context = build_context(Config.load())
    except Exception as e:  # noqa: BLE001 - reported, the job cannot start
        result.error = f"{type(e).__name__}: {e}"
        logger.error("Run could not start: %s", e)
        result.notification_failed = not record_unstarted_run(result)
        return result
    load_secrets_from_ssm(context, RUN_SECRETS)
    tracker = TokenUsageTracker()
    started = time.monotonic()
    try:
        outcome = run_coroutine(_run_pipeline(context, request, tracker))
        result.status = "succeeded"
        result.title = outcome.title
        result.markdown_path = str(outcome.markdown_path)
        result.s3_uri = outcome.s3_uri
        result.pr_url = outcome.pr_url
        result.warnings = outcome.warnings
    except _rejected_input_errors() as e:
        if _is_fetch_outage(e):  # the source exists but could not be reached
            result.error = f"{type(e).__name__}: {e}"
            logger.error("Run failed: %s", e)
        else:
            result.status = "rejected"
            result.error = str(e)
            logger.warning("Request rejected: %s", e)
    except PostValidationError as e:
        # The post broke a blog convention: nothing was published.
        result.error = str(e)
        result.violations = [str(v) for v in e.violations]
        result.markdown_path = e.rejected_path
        result.s3_uri = e.rejected_s3_uri
        result.warnings = list(getattr(e, "run_notes", []))
        logger.error("Post failed validation: %s", e)
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.warnings = list(getattr(e, "run_notes", []))
        logger.error("Run failed: %s", e, exc_info=True)
    except KeyboardInterrupt as e:
        # The caller still prints this run's result (filled in below).
        e.job_result = result  # type: ignore[attr-defined]
        raise
    finally:
        # Neither returned nor raised an Exception: SIGTERM (Batch timeout,
        # cancel, host termination) or Ctrl-C.
        interrupted = result.status == "failed" and result.error is None
        if interrupted:
            result.error = INTERRUPTED_ERROR
        result.duration_seconds = round(time.monotonic() - started, 1)
        result.input_tokens = tracker.input_tokens
        result.output_tokens = tracker.output_tokens
        result.estimated_cost_usd = round(tracker.estimated_cost_usd(), 4)
        _finish(context, result, tracker, notify=not interrupted)
    return result


# SIGTERM (Batch timeout, cancel, host termination) or Ctrl-C. Batch may still
# retry an interrupted job, so the run does not announce a failure; Batch's
# own FAILED event reports a terminal one.
INTERRUPTED_ERROR = "Interrupted before completion."


def run_coroutine[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """``asyncio.run`` that does not wait for executor threads on the way out.

    ``asyncio.run`` joins the default executor before returning, so after a
    SIGTERM the cleanup (manifest) would wait for an in-flight Bedrock call in
    a worker thread, minutes past Batch's ~30 s SIGKILL. Here an interrupted
    run cancels its tasks and returns at once. The interpreter would still
    join abandoned threads at exit, so the Batch worker leaves through
    ``os._exit`` after a SIGTERM (``cli._work``).
    """
    loop = asyncio.new_event_loop()
    executor = ThreadPoolExecutor(thread_name_prefix="scholar-lens")
    loop.set_default_executor(executor)
    task = loop.create_task(coroutine)
    try:
        return loop.run_until_complete(task)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                loop.run_until_complete(task)
        executor.shutdown(wait=False, cancel_futures=True)
        loop.close()


def _finish(
    context: RunContext,
    result: JobResult,
    tracker: TokenUsageTracker,
    *,
    notify: bool = True,
) -> None:
    """Metrics, notification and result manifest; each is best-effort."""
    in_aws = is_running_in_aws()
    MetricsEmitter(context.default_boto_session, enabled=in_aws).emit_run(
        mode=result.kind or "unknown",
        success=result.status == "succeeded",
        duration_seconds=result.duration_seconds,
        tracker=tracker,
    )
    topic_arn = os.getenv(EnvVars.TOPIC_ARN.value)
    if notify and in_aws and topic_arn:
        subject, message = format_result_notification(result)
        result.notification_failed = not publish_sns(
            context.default_boto_session, topic_arn, subject=subject, message=message
        )
    if result.job_id and context.s3_handler is not None:
        try:
            write_result_manifest(
                context.s3_handler, context.config.resources.s3_prefix, result
            )
        except Exception as e:  # noqa: BLE001 - the run itself already finished
            logger.warning("Failed to store the result manifest: %s", e)


def record_unstarted_run(result: JobResult) -> bool:
    """Store the manifest and notify for a Batch run that failed before a
    context existed (bad config or request, startup crash).

    Uses the job's default credential chain; the topic's region comes from its
    ARN. Returns False only when a notification was due but could not be sent
    (outside AWS, or without a topic, there is nothing to send).
    """
    topic_arn = os.getenv(EnvVars.TOPIC_ARN.value)
    region = (
        topic_arn.split(":")[3] if topic_arn and topic_arn.count(":") >= 5 else None
    )
    due = is_running_in_aws() and bool(topic_arn)
    try:
        session = boto3.Session(region_name=region or None)
    except Exception as e:  # noqa: BLE001 - e.g. the AWS profile that broke startup
        logger.warning("No AWS session to report the failed start: %s", e)
        return not due
    if result.job_id:  # set only inside a Batch job
        try:
            resources = Config.load().resources
            if resources.s3_bucket_name:
                write_result_manifest(
                    S3Handler(session, resources.s3_bucket_name),
                    resources.s3_prefix,
                    result,
                )
        except Exception as e:  # noqa: BLE001 - the notification still goes out
            logger.warning("Failed to store the result manifest: %s", e)
    if not (due and topic_arn):
        return True
    subject, message = format_result_notification(result)
    return publish_sns(session, topic_arn, subject=subject, message=message)


_KIND_LABELS = {
    "review": "Paper Review",
    "summarize": "Paper Summary",
    "guide": "Tech Guide",
}


def format_result_notification(result: JobResult) -> tuple[str, str]:
    """SNS subject/body for a finished run (success or failure)."""
    fields = {"Input": result.input}
    if result.title:
        fields["Title"] = result.title
    if result.pr_url:
        fields["Pull request"] = result.pr_url
    if result.s3_uri:
        fields["Output"] = result.s3_uri
    if result.job_id:
        fields["Batch job"] = result.job_id
    fields["Cost (est.)"] = f"${result.estimated_cost_usd:.2f}"
    if result.warnings:
        fields["Warnings"] = "\n".join(f"- {w}" for w in result.warnings)
    if result.error:
        fields["Error"] = result.error
    label = _KIND_LABELS.get(result.kind or "", "Scholar Lens Job")
    return format_alarm(event=label, status=result.status.upper(), fields=fields)
