"""Run one :class:`~scholar_lens.domain.jobs.JobRequest` end to end.

:func:`run_job` is the single entry for every artifact kind, whether invoked by
the CLI locally or by the Batch worker. It builds the AWS/config context, loads
secrets, dispatches to the paper or guide pipeline, and — whatever happens —
records metrics, sends the SNS notification, and (for Batch runs) stores the
:class:`~scholar_lens.domain.jobs.JobResult` in S3.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.exceptions import ClientError

from scholar_lens.adapters.aws.batch import write_result_manifest
from scholar_lens.adapters.aws.environment import is_running_in_aws
from scholar_lens.adapters.aws.metrics import MetricsEmitter
from scholar_lens.adapters.aws.storage import S3Handler, get_ssm_param_value
from scholar_lens.adapters.bedrock.usage import TokenUsageTracker
from scholar_lens.adapters.github.publisher import PostAssetError
from scholar_lens.application.run import PipelineOutcome
from scholar_lens.application.use_cases.publish import (
    RESULT_KINDS,
    SavedPostError,
    load_saved_post,
    publish_saved_post,
)
from scholar_lens.bootstrap import AwsRunContext, build_context, run_use_case
from scholar_lens.configs import Config
from scholar_lens.domain.constants import (
    RUN_SECRETS,
    EnvVars,
    SSMParams,
)
from scholar_lens.domain.jobs import JobRequest, JobResult, ssm_base
from scholar_lens.domain.usage_context import cost_allocation
from scholar_lens.domain.validation import PostValidationError

logger = logging.getLogger(__name__)


def load_secrets_from_ssm(
    context: AwsRunContext, ssm_map: dict[SSMParams, EnvVars]
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


def _is_fetch_outage(error: BaseException) -> bool:
    """A transport failure or 429/5xx while fetching the source: the request
    is fine, so the run failed rather than being rejected."""
    from scholar_lens.adapters.scholarly.paper_source import PaperFetchError

    return isinstance(error, PaperFetchError)


def _rejected_input_errors() -> tuple[type[BaseException], ...]:
    """Errors caused by the request itself (bad URL, not a paper, not docs).

    Imported lazily so ``status``/``submit`` never load the paper stack.
    """
    from scholar_lens.adapters.http.url_guard import UnsafeUrlError
    from scholar_lens.adapters.scholarly.arxiv import ArxivNotFoundError
    from scholar_lens.adapters.scholarly.paper_source import PaperSourceError
    from scholar_lens.domain.guide.models import NotTechnicalContentError
    from scholar_lens.domain.survey.plan import SurveyInputError

    return (
        ArxivNotFoundError,
        PaperSourceError,
        NotTechnicalContentError,
        SurveyInputError,
        UnsafeUrlError,
    )


def check_pr_ready(context: AwsRunContext, request: JobRequest) -> None:
    """Fail before any model call when the run is to open a blog PR it
    cannot open (no token: the post would be paid for twice to get one)."""
    check_pr_token(context, request.create_pr and request.kind != "survey_plan")


def check_pr_token(context: AwsRunContext, create_pr: bool) -> None:
    """Fail when a blog PR is wanted, GitHub is enabled and no token is set."""
    github = context.config.resources.github
    if create_pr and github.enabled and not os.getenv(EnvVars.GITHUB_TOKEN.value):
        raise ValueError(
            f"{EnvVars.GITHUB_TOKEN.value} is not set, so the blog PR cannot be "
            "opened: set it (in Batch, the SSM parameter github-token) or pass "
            "--no-pr."
        )


def run_job(request: JobRequest, *, job_id: str | None = None) -> JobResult:
    """Run ``request`` and return its result; never raises for pipeline errors
    (a ``SystemExit``/``KeyboardInterrupt`` still propagates after cleanup)."""
    result = JobResult(kind=request.kind, input=request.describe(), job_id=job_id)
    try:
        # Resolved before the run: an except clause that imports could raise.
        rejected_errors = _rejected_input_errors()
        context = build_context(Config.load())
    except Exception as e:  # noqa: BLE001 - reported, the job cannot start
        result.error = f"{type(e).__name__}: {e}"
        logger.error("Run could not start: %s", e)
        result.notification_failed = not record_unstarted_run(result)
        return result
    # Resumes from the stages a run with the same inputs saved, unless fresh.
    context.fresh = request.fresh
    load_secrets_from_ssm(context, RUN_SECRETS)
    tracker = TokenUsageTracker()
    started = time.monotonic()
    try:
        resources = context.config.resources
        check_pr_ready(context, request)
        with cost_allocation(resources.project_name, resources.stage, request.kind):
            outcome = run_coroutine(run_use_case(context, request, tracker))
        record_outcome(result, outcome, keep_plan_path=job_id is None)
    except Exception as e:
        record_error(result, e, rejected_errors)
    except (KeyboardInterrupt, SystemExit) as e:
        # Ctrl-C or SIGTERM: the caller still prints this run's result
        # (filled in below).
        e.job_result = result  # type: ignore[union-attr]
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
        result.usage = tracker.run_usage()
        _finish(context, result, tracker, notify=not interrupted)
    return result


def record_outcome(
    result: JobResult, outcome: PipelineOutcome, *, keep_plan_path: bool = True
) -> None:
    """Fold a pipeline's (or ``publish``'s) outcome into ``result``: a PR
    that was requested but failed fails the run, its artifact's paths kept.
    ``keep_plan_path``: False for a Batch run, whose plan file is gone with
    its container (``s3_uri`` only)."""
    result.status = "succeeded"
    result.title = outcome.title
    result.markdown_path = str(outcome.markdown_path) if outcome.markdown_path else None
    result.plan_path = (
        str(outcome.plan_path) if outcome.plan_path and keep_plan_path else None
    )
    result.s3_uri = outcome.s3_uri
    result.pr_url = outcome.pr_url
    result.warnings = outcome.warnings
    if outcome.pr_error is not None:
        # The artifact is saved (paths above) but not published.
        result.status = "failed"
        result.pr_error = outcome.pr_error
        result.error = f"Blog PR not created: {outcome.pr_error}"


def record_error(
    result: JobResult,
    error: Exception,
    rejected_errors: tuple[type[BaseException], ...],
) -> None:
    """Fold a run's (or ``publish``'s) failure into ``result``: a rejected
    input, a post the blog rules or its images blocked (its rejected copy's
    paths kept), or a failure."""
    if isinstance(error, rejected_errors):
        if _is_fetch_outage(error):  # the source exists but could not be reached
            result.error = f"{type(error).__name__}: {error}"
            logger.error("Run failed: %s", error)
        else:
            result.status = "rejected"
            result.error = str(error)
            logger.warning("Request rejected: %s", error)
    elif isinstance(error, PostValidationError):
        # The post broke a blog convention: nothing was published.
        result.error = str(error)
        result.violations = [str(v) for v in error.violations]
        result.markdown_path = error.rejected_path
        result.s3_uri = error.rejected_s3_uri
        result.warnings = list(getattr(error, "run_notes", []))
        logger.error("Post failed validation: %s", error)
    elif isinstance(error, PostAssetError):
        # A paper figure could not be stored: nothing was published.
        result.error = f"{type(error).__name__}: {error}"
        result.markdown_path = error.rejected_path
        result.s3_uri = error.rejected_s3_uri
        result.warnings = list(getattr(error, "run_notes", []))
        logger.error("Post blocked by its images: %s", error)
    else:
        result.error = f"{type(error).__name__}: {error}"
        result.warnings = list(getattr(error, "run_notes", []))
        logger.error("Run failed: %s", error, exc_info=error)


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
    context: AwsRunContext,
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
    "survey_plan": "Survey Plan",
    "survey_write": "Paper Survey",
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
    if result.usage and result.usage.by_stage:
        fields["Costliest stages"] = ", ".join(
            f"{stage} ${cost:.2f}" for stage, cost in result.usage.top_stages()
        )
    if result.warnings:
        fields["Warnings"] = "\n".join(f"- {w}" for w in result.warnings)
    if result.error:
        fields["Error"] = result.error
    label = _KIND_LABELS.get(result.kind or "", "Scholar Lens Job")
    return format_alarm(event=label, status=result.status.upper(), fields=fields)


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
        result.kind = RESULT_KINDS.get(request.kind)
        check_pr_token(context, create_pr)
        outcome = run_coroutine(
            publish_saved_post(context, request, create_pr=create_pr)
        )
        record_outcome(result, outcome)
    except Exception as e:  # noqa: BLE001 - folded into the result
        record_error(result, e, (SavedPostError,))
    finally:
        result.duration_seconds = round(time.monotonic() - started, 1)
    return result
