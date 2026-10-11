"""Job contract shared by the CLI, the AWS Batch worker, and agent callers.

A :class:`JobRequest` fully describes one artifact run. The CLI runs it locally
or serialises it into the single ``request`` parameter of the Batch job
definition (``worker --request Ref::request``), so the local and remote paths
parse exactly the same object and cannot drift. Every run ends in a
:class:`JobResult`, printed as JSON by the CLI and, for Batch runs, stored in S3
so ``status`` can report it after the container is gone.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

import boto3
from botocore.exceptions import ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotoConnectionError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .aws_helpers import (
    BatchJobError,
    S3Handler,
    S3OperationError,
    get_ssm_param_value,
)
from .constants import S3Paths, SSMParams
from .logger import logger
from .utils.retry import is_transient_error

ArtifactKind = Literal["review", "summarize", "guide"]
JobStatus = Literal["succeeded", "failed", "rejected"]

# Batch job states that will not change any more.
TERMINAL_BATCH_STATES = frozenset({"SUCCEEDED", "FAILED"})


class JobRequest(BaseModel):
    """One artifact run: what to read and whether to open a blog PR."""

    model_config = ConfigDict(extra="forbid")

    kind: ArtifactKind
    # review / summarize: an arXiv id or a paper PDF URL.
    source: str | None = None
    repo_urls: list[str] = Field(default_factory=list)
    parse_pdf: bool = False
    # guide: documentation URLs plus optional extra web-search queries.
    urls: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    discover_subpages: bool = True
    # Opens a blog PR when the GitHub integration is enabled in config.
    create_pr: bool = True

    @model_validator(mode="after")
    def _check_inputs(self) -> JobRequest:
        if self.kind == "guide":
            if not self.urls:
                raise ValueError("A guide request needs at least one URL.")
            if self.source or self.repo_urls:
                raise ValueError("A guide request takes 'urls', not 'source'.")
        else:
            if not (self.source and self.source.strip()):
                raise ValueError(f"A {self.kind} request needs a 'source'.")
            if self.urls or self.search_queries:
                raise ValueError(f"A {self.kind} request takes 'source', not 'urls'.")
        return self

    def describe(self) -> str:
        """Short human label of the input, for logs and notifications."""
        return self.source or ", ".join(self.urls)


class JobResult(BaseModel):
    """Outcome of one run. ``warnings`` lists non-fatal problems (e.g. a PR
    that could not be opened even though the artifact was generated)."""

    # None only for a Batch request too malformed to name its kind.
    kind: ArtifactKind | None
    status: JobStatus = "failed"
    input: str
    title: str | None = None
    markdown_path: str | None = None
    s3_uri: str | None = None
    pr_url: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    # Blog-convention violations that blocked publishing (see post_validator).
    violations: list[str] = Field(default_factory=list)
    job_id: str | None = None
    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    # A failure notification was due but SNS rejected it; the worker then
    # exits with a code the stack's EventBridge rule reports instead.
    notification_failed: bool = Field(default=False, exclude=True)


class SubmittedJob(BaseModel):
    job_id: str
    job_name: str
    kind: ArtifactKind
    input: str


class JobStatusReport(BaseModel):
    """Batch state of a remote job plus its result once it has finished."""

    job_id: str
    job_name: str | None = None
    batch_status: str
    status_reason: str | None = None
    created_at: str | None = None
    stopped_at: str | None = None
    result: JobResult | None = None


def ssm_base(project_name: str, stage: str) -> str:
    """Prefix of every SSM parameter of one deployment."""
    return f"/{project_name}/{stage}"


def result_manifest_key(s3_prefix: str, job_id: str) -> str:
    return f"{s3_prefix}/{S3Paths.RESULTS.value}/{job_id}.json".lstrip("/")


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
    job_name = f"{project_name}-{stage}-{request.kind}-{timestamp}"
    try:
        response = session.client("batch").submit_job(
            jobName=job_name,
            jobQueue=job_queue,
            jobDefinition=job_definition,
            parameters={"request": request.model_dump_json()},
        )
    except ClientError as e:
        raise BatchJobError(f"Failed to submit Batch job '{job_name}': {e}") from e
    logger.info("Submitted Batch job '%s' (%s)", job_name, response["jobId"])
    return SubmittedJob(
        job_id=response["jobId"],
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
    report = JobStatusReport(
        job_id=job_id,
        job_name=job.get("jobName"),
        batch_status=job["status"],
        status_reason=job.get("statusReason"),
        created_at=_iso(job.get("createdAt")),
        stopped_at=_iso(job.get("stoppedAt")),
    )
    if report.batch_status in TERMINAL_BATCH_STATES and s3_handler is not None:
        report.result = read_result_manifest(s3_handler, s3_prefix, job_id)
    return report


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
