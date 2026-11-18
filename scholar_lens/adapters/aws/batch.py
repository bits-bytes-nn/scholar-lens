"""The AWS Batch job queue: submit a :class:`~scholar_lens.domain.jobs.JobRequest`
as the job definition's single ``request`` parameter, report a job's state
with its stored result or last progress, and cancel a job."""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    HTTPClientError,
)
from botocore.exceptions import ConnectionError as BotoConnectionError
from pydantic import ValidationError

from scholar_lens.adapters.aws.storage import (
    BatchJobError,
    S3Handler,
    S3OperationError,
    get_ssm_param_value,
)
from scholar_lens.adapters.retry import is_transient_error
from scholar_lens.domain.constants import SSMParams
from scholar_lens.domain.jobs import (
    TERMINAL_BATCH_STATES,
    CancelledJob,
    JobAlreadyFinished,
    JobProgress,
    JobRequest,
    JobResult,
    JobStatusReport,
    SubmittedJob,
    progress_key,
    result_manifest_key,
    ssm_base,
)

logger = logging.getLogger(__name__)

# Where Batch sends container logs when the job definition names no group.
DEFAULT_BATCH_LOG_GROUP = "/aws/batch/job"


def write_result_manifest(
    s3_handler: S3Handler, s3_prefix: str, result: JobResult
) -> None:
    if not result.job_id:
        raise ValueError("Only Batch runs (with a job id) have a result manifest.")
    s3_handler.put_text(
        result_manifest_key(s3_prefix, result.job_id), result.model_dump_json(indent=2)
    )


def read_result_manifest(
    s3_handler: S3Handler, s3_prefix: str, job_id: str
) -> JobResult | None:
    key = result_manifest_key(s3_prefix, job_id)
    text = s3_handler.get_text(key)
    if text is None:
        return None
    try:
        return JobResult.model_validate_json(text)
    except ValidationError as e:  # a broken manifest is not a rejected input
        raise S3OperationError(f"Malformed result manifest '{key}': {e}") from e


_SUBMIT_ATTEMPTS = 2
# Lookups after a submit that may have been accepted, each after a pause.
_LOOKUPS = 2
_LOOKUP_WAIT_SECONDS = 3.0
_RESUBMIT_WAIT_SECONDS = 2.0
_THROTTLED = frozenset(
    {"TooManyRequestsException", "ThrottlingException", "Throttling"}
)
_sleep = time.sleep  # replaced in tests


def _not_accepted(error: Exception) -> bool:
    """Whether a failed SubmitJob certainly never reached Batch: throttled,
    or no connection was made. A timeout, a dropped connection or a 5xx may
    come after Batch accepted the job."""
    if isinstance(error, (EndpointConnectionError, ConnectTimeoutError)):
        return True
    if isinstance(error, ClientError):
        return error.response.get("Error", {}).get("Code") in _THROTTLED
    return False


def _submitted(session: boto3.Session, job_queue: str, job_name: str) -> str | None:
    """The id of a job named ``job_name`` in ``job_queue`` (one Batch accepted
    before its answer was lost), or None.

    Raises:
        BatchJobError: the lookup failed, so a second submit could duplicate
            the job.
    """
    try:
        # A read: the SDK's own retries are safe here.
        jobs = (
            session.client("batch")
            .list_jobs(
                jobQueue=job_queue,
                filters=[{"name": "JOB_NAME", "values": [job_name]}],
            )
            .get("jobSummaryList", [])
        )
    except (ClientError, HTTPClientError, BotoConnectionError) as e:
        raise BatchJobError(
            f"Submitting Batch job '{job_name}' failed and it could not be looked "
            f"up ({e}); it may have been submitted: check the queue before "
            "submitting again."
        ) from e
    return str(jobs[0]["jobId"]) if jobs else None


def submit_job(
    session: boto3.Session,
    request: JobRequest,
    *,
    project_name: str,
    stage: str,
) -> SubmittedJob:
    """Submit ``request`` to the project's Batch queue and return its id."""
    base = ssm_base(project_name, stage)
    job_queue = get_ssm_param_value(
        session, f"{base}/{SSMParams.BATCH_JOB_QUEUE.value}"
    )
    job_definition = get_ssm_param_value(
        session, f"{base}/{SSMParams.BATCH_JOB_DEFINITION.value}"
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    # A random tail: two submits in one second never share a name, so the
    # lookup below cannot return another request's job.
    job_name = f"{project_name}-{stage}-{request.kind}-{timestamp}-{uuid4().hex[:6]}"
    # SubmitJob has no idempotency token: the SDK must not resend it (a
    # timeout after Batch accepted the job would start a second, billed run).
    # Only a failure that means the request was not accepted (throttled, no
    # connection made) is submitted again; after one that may have been
    # accepted, the job is looked up by its name and never submitted twice.
    client = session.client(
        "batch",
        config=Config(retries={"total_max_attempts": 1, "mode": "standard"}),
    )
    job_id = ""
    for attempt in range(_SUBMIT_ATTEMPTS):
        try:
            job_id = client.submit_job(
                jobName=job_name,
                jobQueue=job_queue,
                jobDefinition=job_definition,
                # Defaults are left out so a worker image from before a newly
                # added field (extra="forbid") still accepts requests that do
                # not use it.
                parameters={"request": request.model_dump_json(exclude_defaults=True)},
            )["jobId"]
            break
        except (ClientError, HTTPClientError, BotoConnectionError) as e:
            if _not_accepted(e) and attempt < _SUBMIT_ATTEMPTS - 1:
                logger.warning(
                    "Submitting '%s' failed (%s); trying again.", job_name, e
                )
                _sleep(_RESUBMIT_WAIT_SECONDS * (1 + random.random()))
                continue
            if _not_accepted(e) or not is_transient_error(e):
                raise BatchJobError(
                    f"Failed to submit Batch job '{job_name}': {e}"
                ) from e
            # Batch may have accepted it: find it, or stop.
            found = None
            for _ in range(_LOOKUPS):
                _sleep(_LOOKUP_WAIT_SECONDS)
                if found := _submitted(session, job_queue, job_name):
                    break
            if not found:
                raise BatchJobError(
                    f"Failed to submit Batch job '{job_name}' ({e}); it may have "
                    "been submitted: check the queue before submitting again."
                ) from e
            job_id = found
            break
    logger.info("Submitted Batch job '%s' (%s)", job_name, job_id)
    return SubmittedJob(
        job_id=job_id,
        job_name=job_name,
        kind=request.kind,
        input=request.describe(),
    )


def _iso(epoch_ms: int | None) -> str | None:
    if not epoch_ms:
        return None
    return datetime.fromtimestamp(epoch_ms / 1000, UTC).isoformat()


def describe_job(
    session: boto3.Session,
    job_id: str,
    *,
    s3_handler: S3Handler | None,
    s3_prefix: str,
) -> JobStatusReport:
    """Report a Batch job's state; attach the stored result once it finished."""
    try:
        jobs = session.client("batch").describe_jobs(jobs=[job_id]).get("jobs", [])
    except (ClientError, HTTPClientError, BotoConnectionError) as e:
        raise BatchJobError(f"Failed to describe Batch job '{job_id}': {e}") from e
    if not jobs:
        raise BatchJobError(f"Batch job '{job_id}' not found.")
    job = jobs[0]
    container = job.get("container") or {}
    options = (container.get("logConfiguration") or {}).get("options") or {}
    report = JobStatusReport(
        job_id=job_id,
        job_name=job.get("jobName"),
        batch_status=job["status"],
        status_reason=job.get("statusReason"),
        created_at=_iso(job.get("createdAt")),
        stopped_at=_iso(job.get("stoppedAt")),
        log_group=(
            options.get("awslogs-group", DEFAULT_BATCH_LOG_GROUP)
            if container.get("logStreamName")
            else None
        ),
        log_stream=container.get("logStreamName"),
    )
    if report.batch_status in TERMINAL_BATCH_STATES and s3_handler is not None:
        report.result = read_result_manifest(s3_handler, s3_prefix, job_id)
    if s3_handler is not None and report.result is None:
        # A running job's progress, or what a job that died without a result
        # (out of memory, a lost host) last reported.
        report.progress = read_progress(s3_handler, s3_prefix, job_id)
    return report


def read_progress(
    s3_handler: S3Handler, s3_prefix: str, job_id: str
) -> JobProgress | None:
    """A running job's last progress report; None when there is none or it
    cannot be read (progress is advisory)."""
    try:
        text = s3_handler.get_text(progress_key(s3_prefix, job_id))
        return JobProgress.model_validate_json(text) if text else None
    except (S3OperationError, ValidationError) as e:
        logger.warning("Progress of job '%s' unreadable: %s", job_id, e)
        return None


class ProgressReporter:
    """Writes a Batch job's progress (stage, calls, cost) to S3 after a
    model call, at most every ``interval`` seconds; a write that fails is
    logged and never stops the run."""

    def __init__(
        self,
        s3_handler: S3Handler,
        s3_prefix: str,
        job_id: str,
        *,
        interval: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.s3_handler = s3_handler
        self.key = progress_key(s3_prefix, job_id)
        self.interval = interval
        self.clock = clock
        # Batch retries a job under the same id: the counts restart.
        attempt = os.environ.get("AWS_BATCH_JOB_ATTEMPT", "1")
        self.attempt = int(attempt) if attempt.isdigit() else 1
        self._last: float | None = None

    def __call__(self, tracker: Any, event: Any) -> None:  # noqa: ANN401
        now = self.clock()
        if self._last is not None and now - self._last < self.interval:
            return
        self._last = now
        progress = JobProgress(
            stage=event.stage,
            calls=tracker.call_count,
            cost_usd=round(tracker.estimated_cost_usd(), 4),
            updated_at=datetime.now(UTC).isoformat(),
            attempt=self.attempt,
        )
        try:
            self.s3_handler.put_text(self.key, progress.model_dump_json())
        except S3OperationError as e:
            logger.warning("Could not write job progress: %s", e)


def cancel_job(session: boto3.Session, job_id: str, *, reason: str) -> CancelledJob:
    """Ask Batch to stop a job: ``TerminateJob`` cancels a job that has not
    started and stops a running one (its worker reports the run as
    interrupted). Batch stops it asynchronously.

    Raises:
        JobAlreadyFinished: The job already succeeded or failed.
        BatchJobError: Batch refused or could not be reached, or no such job.
    """
    try:
        client = session.client("batch")
        jobs = client.describe_jobs(jobs=[job_id]).get("jobs", [])
        if not jobs:
            raise BatchJobError(f"Batch job '{job_id}' not found.")
        status = jobs[0]["status"]
        if status in TERMINAL_BATCH_STATES:
            raise JobAlreadyFinished(f"Batch job '{job_id}' already {status}.")
        client.terminate_job(jobId=job_id, reason=reason)
    except (ClientError, HTTPClientError, BotoConnectionError) as e:
        raise BatchJobError(f"Failed to cancel Batch job '{job_id}': {e}") from e
    try:
        after = client.describe_jobs(jobs=[job_id]).get("jobs", [])
    except (ClientError, HTTPClientError, BotoConnectionError) as e:
        # The stop was requested; only the follow-up read failed.
        logger.warning("Could not read job '%s' after cancelling: %s", job_id, e)
        after = []
    return CancelledJob(
        job_id=job_id,
        previous_status=status,
        reason=reason,
        status=after[0]["status"] if after else status,
    )


def wait_for_job(
    describe: Callable[[], JobStatusReport],
    *,
    timeout_seconds: float,
    poll_seconds: float = 30.0,
    sleep: Callable[[float], Any] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> JobStatusReport:
    """Poll ``describe`` until the job is terminal or ``timeout_seconds`` pass.

    Returns the last report either way; the caller checks ``batch_status``. A
    throttled, 5xx or network-failed ``describe`` or result-manifest read (the
    shared retry policy's transient errors) is skipped — the job keeps running — unless no report
    was obtained before the deadline; any other error propagates.
    """
    deadline = clock() + timeout_seconds
    report: JobStatusReport | None = None
    while True:
        try:
            report = describe()
        except (BatchJobError, S3OperationError) as e:
            if not is_transient_error(e) or (report is None and clock() >= deadline):
                raise
            logger.warning("Transient error polling the job; retrying: %s", e)
            if report is not None and clock() >= deadline:
                return report
        else:
            if report.batch_status in TERMINAL_BATCH_STATES or clock() >= deadline:
                return report
        sleep(min(poll_seconds, max(0.0, deadline - clock())))


class BatchJobs:
    """The :class:`~scholar_lens.application.ports.JobQueue` on AWS Batch,
    the finished jobs' results read from their S3 manifests."""

    def __init__(
        self, session: boto3.Session, *, s3_handler: S3Handler | None, s3_prefix: str
    ) -> None:
        self.session = session
        self.s3_handler = s3_handler
        self.s3_prefix = s3_prefix

    def describe(self, job_id: str) -> JobStatusReport:
        return describe_job(
            self.session, job_id, s3_handler=self.s3_handler, s3_prefix=self.s3_prefix
        )

    def wait(self, job_id: str, *, timeout_seconds: float) -> JobStatusReport:
        return wait_for_job(
            lambda: self.describe(job_id), timeout_seconds=timeout_seconds
        )

    def cancel(self, job_id: str, *, reason: str) -> CancelledJob:
        return cancel_job(self.session, job_id, reason=reason)
