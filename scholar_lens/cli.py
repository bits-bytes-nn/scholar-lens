"""``scholar-lens`` command line: generate, submit, and track artifacts.

Every command prints one JSON object on stdout (logs go to stderr), so agents
and scripts can parse the outcome directly::

    scholar-lens summarize 2106.09685
    scholar-lens review https://openreview.net/pdf?id=nZeVKeeFYf9 --remote
    scholar-lens guide https://docs.vllm.ai/en/latest/ --search-query "vLLM paged attention"
    scholar-lens status <job-id> --wait

Exit codes: 0 succeeded (or submitted), 1 failed, 2 rejected input,
3 still running (``status``).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from scholar_lens.src.jobs import JobRequest, JobResult

EXIT_OK, EXIT_FAILED, EXIT_REJECTED, EXIT_PENDING = 0, 1, 2, 3
_EXIT_BY_STATUS = {
    "succeeded": EXIT_OK,
    "failed": EXIT_FAILED,
    "rejected": EXIT_REJECTED,
}

# How long `status --wait` polls before returning the current state. Batch jobs
# are capped at 3 hours by the job definition.
DEFAULT_WAIT_SECONDS = 4 * 60 * 60


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scholar-lens",
        description="Generate AI/ML paper reviews, summaries and technical guides.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_run_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--remote",
            action="store_true",
            help="Submit to AWS Batch and return the job id instead of running here.",
        )
        sub.add_argument(
            "--no-pr",
            dest="create_pr",
            action="store_false",
            help="Do not open a blog pull request (the post is still saved).",
        )

    for kind, help_text in (
        ("review", "In-depth, section-by-section paper review."),
        ("summarize", "Concise five-part paper summary."),
    ):
        sub = commands.add_parser(kind, help=help_text, description=help_text)
        sub.add_argument(
            "source", help="arXiv id (e.g. 2106.09685) or a paper PDF URL."
        )
        sub.add_argument(
            "--repo-url",
            dest="repo_urls",
            action="append",
            default=[],
            help="Code repository of the paper (repeatable).",
        )
        sub.add_argument(
            "--parse-pdf",
            action="store_true",
            help="Parse the PDF even when an arXiv HTML rendering exists.",
        )
        add_run_options(sub)

    guide = commands.add_parser(
        "guide",
        help="Hands-on technical guide from documentation URLs.",
        description="Hands-on technical guide from documentation URLs.",
    )
    guide.add_argument("urls", nargs="+", help="Documentation URLs.")
    guide.add_argument(
        "--search-query",
        dest="search_queries",
        action="append",
        default=[],
        help="Extra web-search query (repeatable; quote multi-word queries).",
    )
    guide.add_argument(
        "--no-subpages",
        dest="discover_subpages",
        action="store_false",
        help="Do not follow in-scope sub-pages of the URLs.",
    )
    add_run_options(guide)

    status = commands.add_parser("status", help="Show a remote job's state and result.")
    status.add_argument("job_id")
    status.add_argument(
        "--wait", action="store_true", help="Poll until the job finishes."
    )
    status.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_WAIT_SECONDS,
        help="Seconds to wait with --wait (default: %(default)s).",
    )

    worker = commands.add_parser(
        "worker", help="Run a JSON job request (AWS Batch entrypoint)."
    )
    worker.add_argument("--request", required=True, help="JobRequest as JSON.")
    return parser


def _request_from_args(args: argparse.Namespace) -> JobRequest:
    if args.command == "guide":
        return JobRequest(
            kind="guide",
            urls=args.urls,
            search_queries=args.search_queries,
            discover_subpages=args.discover_subpages,
            create_pr=args.create_pr,
        )
    return JobRequest(
        kind=args.command,
        source=args.source,
        repo_urls=args.repo_urls,
        parse_pdf=args.parse_pdf,
        create_pr=args.create_pr,
    )


def _emit(payload: BaseModel | dict[str, Any]) -> None:
    data = (
        payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    )
    sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()


def _aws_context() -> Any:
    from scholar_lens.configs import Config
    from scholar_lens.src.runtime import build_context

    return build_context(Config.load())


def _submit(request: JobRequest) -> int:
    from scholar_lens.src.jobs import submit_job

    context = _aws_context()
    resources = context.config.resources
    submitted = submit_job(
        context.default_boto_session,
        request,
        project_name=resources.project_name,
        stage=resources.stage,
    )
    _emit(submitted)
    return EXIT_OK


def _status(job_id: str, *, wait: bool, timeout: float) -> int:
    from scholar_lens.src.jobs import describe_job, wait_for_job

    context = _aws_context()

    def describe():  # type: ignore[no-untyped-def]
        return describe_job(
            context.default_boto_session,
            job_id,
            s3_handler=context.s3_handler,
            s3_prefix=context.config.resources.s3_prefix,
        )

    report = wait_for_job(describe, timeout_seconds=timeout) if wait else describe()
    _emit(report)
    if report.result is not None:
        return _EXIT_BY_STATUS[report.result.status]
    if report.batch_status == "FAILED":
        return EXIT_FAILED
    if report.batch_status == "SUCCEEDED":
        return EXIT_OK  # finished; no manifest (e.g. no S3 bucket configured)
    return EXIT_PENDING  # not finished yet: ask again later


def _run(request: JobRequest, *, job_id: str | None = None) -> int:
    from scholar_lens.src.runtime import run_job

    result: JobResult = run_job(request, job_id=job_id)
    _emit(result)
    return _EXIT_BY_STATUS[result.status]


def _notify_bad_request(raw: str, error: ValidationError) -> None:
    """A Batch job whose request does not parse still notifies (SNS)."""
    from scholar_lens.src.runtime import notify_without_context

    notify_without_context(
        JobResult(
            kind="review",
            input=raw[:200],
            status="rejected",
            error=f"Invalid job request: {error}",
            job_id=os.getenv("AWS_BATCH_JOB_ID"),
        )
    )


def _raise_on_sigterm() -> None:
    """Batch stops a job with SIGTERM (timeout, cancel, spot reclaim). Turn it
    into SystemExit so cleanup, metrics and the result manifest still run."""

    def handler(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handler)


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # FAISS + torch OpenMP clash
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            return _status(args.job_id, wait=args.wait, timeout=args.timeout)
        if args.command == "worker":
            _raise_on_sigterm()
            try:
                request = JobRequest.model_validate_json(args.request)
            except ValidationError as e:
                _notify_bad_request(args.request, e)
                raise
            return _run(request, job_id=os.getenv("AWS_BATCH_JOB_ID"))
        request = _request_from_args(args)
        if args.remote:
            return _submit(request)
    except ValidationError as e:
        _emit({"status": "rejected", "error": str(e)})
        return EXIT_REJECTED
    except Exception as e:  # noqa: BLE001 - report AWS/config errors as JSON
        _emit({"status": "failed", "error": f"{type(e).__name__}: {e}"})
        return EXIT_FAILED
    return _run(request)


if __name__ == "__main__":
    sys.exit(main())
