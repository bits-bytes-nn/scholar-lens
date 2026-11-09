"""``scholar-lens status`` and ``cancel``: a remote job's state and, once it
finished, its result; a request to stop it."""

from __future__ import annotations

from scholar_lens.application.ports import JobQueue
from scholar_lens.domain.jobs import CancelledJob, JobStatusReport


def job_status(
    jobs: JobQueue, job_id: str, *, wait: bool, timeout_seconds: float
) -> JobStatusReport:
    """The job's report; with ``wait``, polled until it is terminal or
    ``timeout_seconds`` pass."""
    if wait:
        return jobs.wait(job_id, timeout_seconds=timeout_seconds)
    return jobs.describe(job_id)


def cancel_job(jobs: JobQueue, job_id: str) -> CancelledJob:
    """Ask the job to stop, with a reason Batch keeps on the job."""
    return jobs.cancel(job_id, reason="Cancelled by the user (scholar-lens cancel).")
