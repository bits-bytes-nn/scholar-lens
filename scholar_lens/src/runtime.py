"""Run one :class:`~scholar_lens.src.jobs.JobRequest` end to end.

:func:`run_job` is the single entry for every artifact kind, whether invoked by
the CLI locally or by the Batch worker. It builds the AWS/config context, loads
secrets, dispatches to the paper or guide pipeline, and — whatever happens —
records metrics, sends the SNS notification, and (for Batch runs) stores the
:class:`~scholar_lens.src.jobs.JobResult` in S3.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import boto3
from pydantic import BaseModel, ConfigDict

from ..configs import Config
from .aws_helpers import S3Handler, get_ssm_param_value
from .constants import RUN_SECRETS, EnvVars, SSMParams
from .jobs import JobRequest, JobResult, write_result_manifest
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

    # Tags the blog already uses (reuse candidates); loaded on first use.
    blog_tags: list[str] | None = None

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
            self.blog_tags = [
                t
                for t in load_run_blog_tags(self.config)
                if t.lower() not in retired
                and t not in controlled
                and format_tag(t, words) == t  # only spellings a post may use
            ]
        return self.blog_tags


def build_context(config: Config) -> RunContext:
    """Build a :class:`RunContext` (boto sessions + optional S3 handler).

    The AWS profile comes from the env in Batch (instance role / ``AWS_PROFILE``)
    and from config locally — matching the previous per-entrypoint behaviour.
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

    Best-effort per secret: a missing/optional secret is logged and skipped so a
    run that doesn't need it still proceeds.
    """
    if not is_running_in_aws():
        return
    base = f"/{context.config.resources.project_name}/{context.config.resources.stage}"
    for ssm_param, env_var in ssm_map.items():
        try:
            os.environ[env_var.value] = get_ssm_param_value(
                context.default_boto_session, f"{base}/{ssm_param.value}"
            )
            logger.info("Set env var '%s' from SSM.", env_var.value)
        except Exception as e:  # noqa: BLE001 - optional secret; never abort the run
            logger.info("Could not set '%s' from SSM: %s", env_var.value, e)


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
) -> None:
    """Publish a plain-text SNS notification; a failure must not fail the job."""
    try:
        session.client("sns").publish(
            TopicArn=topic_arn, Subject=subject, Message=message
        )
    except Exception as e:  # noqa: BLE001 - notification failure must not fail the job
        logger.error("Failed to send SNS notification: %s", e)


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


def load_run_blog_tags(config: Config) -> list[str]:
    """The blog's existing tags, or none when GitHub is not configured or the
    listing fails (the model then proposes fresh short tags)."""
    github = config.resources.github
    if not (github.enabled and github.repo_name):
        return []
    try:
        return fetch_blog_tags(
            github.repo_name, github.base_branch, os.getenv(EnvVars.GITHUB_TOKEN.value)
        )
    except Exception as e:  # noqa: BLE001 - reuse hints only; never fail the run
        logger.warning("Could not list the blog's tags: %s", e)
        return []


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

    A PR failure does not fail the run (the artifact already exists locally and
    in S3); it is reported as a warning on the result instead.
    """
    publisher = build_publisher(context)
    s3_uri, markdown_path = await publisher.publish(request)
    outcome = PipelineOutcome(
        title=request.title, markdown_path=markdown_path, s3_uri=s3_uri
    )
    if create_pr and context.config.resources.github.enabled:
        try:
            outcome.pr_url = await publisher.create_pull_request(request, markdown_path)
        except PublishError as e:
            logger.error("Blog PR failed: %s", e)
            outcome.warnings.append(f"Blog PR not created: {e}")
    return outcome


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
        notify_without_context(result)
        return result
    load_secrets_from_ssm(context, RUN_SECRETS)
    tracker = TokenUsageTracker()
    started = time.monotonic()
    try:
        outcome = asyncio.run(_run_pipeline(context, request, tracker))
        result.status = "succeeded"
        result.title = outcome.title
        result.markdown_path = str(outcome.markdown_path)
        result.s3_uri = outcome.s3_uri
        result.pr_url = outcome.pr_url
        result.warnings = outcome.warnings
    except _rejected_input_errors() as e:
        result.status = "rejected"
        result.error = str(e)
        logger.warning("Request rejected: %s", e)
    except PostValidationError as e:
        # The post broke a blog convention: nothing was published.
        result.error = str(e)
        result.violations = [str(v) for v in e.violations]
        result.markdown_path = e.rejected_path
        result.s3_uri = e.rejected_s3_uri
        logger.error("Post failed validation: %s", e)
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        logger.error("Run failed: %s", e, exc_info=True)
    finally:
        if result.status == "failed" and result.error is None:
            result.error = "Interrupted before completion."
        result.duration_seconds = round(time.monotonic() - started, 1)
        result.input_tokens = tracker.input_tokens
        result.output_tokens = tracker.output_tokens
        result.estimated_cost_usd = round(tracker.estimated_cost_usd(), 4)
        _finish(context, result, tracker)
    return result


def _finish(context: RunContext, result: JobResult, tracker: TokenUsageTracker) -> None:
    """Metrics, notification and result manifest; each is best-effort."""
    in_aws = is_running_in_aws()
    MetricsEmitter(context.default_boto_session, enabled=in_aws).emit_run(
        mode=result.kind,
        success=result.status == "succeeded",
        duration_seconds=result.duration_seconds,
        tracker=tracker,
    )
    topic_arn = os.getenv(EnvVars.TOPIC_ARN.value)
    if in_aws and topic_arn:
        subject, message = format_result_notification(result)
        publish_sns(
            context.default_boto_session, topic_arn, subject=subject, message=message
        )
    if result.job_id and context.s3_handler is not None:
        try:
            write_result_manifest(
                context.s3_handler, context.config.resources.s3_prefix, result
            )
        except Exception as e:  # noqa: BLE001 - the run itself already finished
            logger.error("Failed to store the result manifest: %s", e)


def notify_without_context(result: JobResult) -> None:
    """SNS for a run that failed before a context existed (bad config or
    request): the default credential chain and region of the Batch job."""
    topic_arn = os.getenv(EnvVars.TOPIC_ARN.value)
    if not (is_running_in_aws() and topic_arn):
        return
    subject, message = format_result_notification(result)
    publish_sns(boto3.Session(), topic_arn, subject=subject, message=message)


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
    for warning in result.warnings:
        fields.setdefault("Warning", warning)
    if result.error:
        fields["Error"] = result.error
    return format_alarm(
        event=_KIND_LABELS[result.kind], status=result.status.upper(), fields=fields
    )
