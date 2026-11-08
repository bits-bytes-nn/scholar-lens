"""``scholar-lens status``: a remote job's state and, once it finished, its
result."""

from __future__ import annotations

from scholar_lens.application.ports import JobQueue
from scholar_lens.domain.jobs import JobStatusReport


def job_status(
    jobs: JobQueue, job_id: str, *, wait: bool, timeout_seconds: float
) -> JobStatusReport:
    """The job's report; with ``wait``, polled until it is terminal or
    ``timeout_seconds`` pass."""
    if wait:
        return jobs.wait(job_id, timeout_seconds=timeout_seconds)
    return jobs.describe(job_id)
