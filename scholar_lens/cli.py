"""``scholar-lens`` command line: generate, submit, and track artifacts.

Every command prints one JSON object on stdout (logs go to stderr), so agents
and scripts can parse the outcome directly::

    scholar-lens summarize 2106.09685
    scholar-lens review "https://openreview.net/pdf?id=nZeVKeeFYf9" --remote
    scholar-lens guide "https://docs.vllm.ai/en/latest/" --search-query "vLLM paged attention"
    scholar-lens guide "https://docs.vllm.ai/en/latest/" --reference "https://arxiv.org/abs/2309.06180"
    scholar-lens status <job-id> --wait

Exit codes: 0 succeeded (or submitted), 1 failed, 2 rejected input,
3 still running (``status``), 4 the command itself failed (``"status":
"error"``: AWS credentials, config, Batch/S3 API errors), so the job's state
is unknown. The Batch ``worker`` exits 10 (failed) or 11 (rejected) once it
has sent its own notification, and 4 when that notification could not be sent.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from scholar_lens.src.jobs import JobRequest, JobResult

EXIT_OK, EXIT_FAILED, EXIT_REJECTED, EXIT_PENDING, EXIT_ERROR = 0, 1, 2, 3, 4
_EXIT_BY_STATUS = {
    "succeeded": EXIT_OK,
    "failed": EXIT_FAILED,
    "rejected": EXIT_REJECTED,
}
# The Batch worker exits with these after it has sent its own failure
# notification. They differ from 1/2 so that a Python crash (1) or an argparse
# error (2) before the worker code runs is still reported by the stack's
# EventBridge rule, which notifies on every other failed job.
WORKER_REPORTED_FAILED, WORKER_REPORTED_REJECTED = 10, 11
APP_REPORTED_EXIT_CODES = (WORKER_REPORTED_FAILED, WORKER_REPORTED_REJECTED)
_WORKER_EXIT_BY_STATUS = {
    "succeeded": EXIT_OK,
    "failed": WORKER_REPORTED_FAILED,
    "rejected": WORKER_REPORTED_REJECTED,
}

# How long `status --wait` polls before returning the current state: one
# 3-hour Batch attempt plus queueing time.
DEFAULT_WAIT_SECONDS = 4 * 60 * 60


def _seconds(value: str) -> float:
    """A finite, non-negative number of seconds (``nan`` would never expire)."""
    try:
        seconds = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from e
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError(f"must be finite and >= 0: {value!r}")
    return seconds


class _JsonArgumentParser(argparse.ArgumentParser):
    """Usage errors keep the one-JSON-object contract: usage goes to stderr,
    ``{"status": "rejected"}`` to stdout, exit 2 (sub-commands inherit this
    class)."""

    def error(self, message: str) -> Any:  # argparse's NoReturn hook
        self.print_usage(sys.stderr)
        _emit({"status": "rejected", "error": f"{self.prog}: {message}"})
        sys.exit(EXIT_REJECTED)


def build_parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(
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
        description=(
            "Hands-on technical guide from documentation URLs and optional "
            "reference material (at least one URL of either kind)."
        ),
    )
    guide.add_argument(
        "urls",
        nargs="*",
        help="Documentation URLs: the authority for commands, flags and defaults.",
    )
    guide.add_argument(
        "--reference",
        dest="reference_urls",
        action="append",
        default=[],
        metavar="URL",
        help=(
            "Reference material to build on: a paper, design doc, deep-dive post "
            "or talk page (repeatable; always used and listed, never treated as "
            "official documentation)."
        ),
    )
    guide.add_argument(
        "--search-query",
        dest="search_queries",
        action="append",
        default=[],
        help=(
            "Web-search query (repeatable; quote multi-word queries); replaces "
            "the queries the guide would plan itself."
        ),
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
        type=_seconds,
        default=DEFAULT_WAIT_SECONDS,
        help="Seconds to wait with --wait (default: %(default)s).",
    )

    worker = commands.add_parser(
        "worker", help="Run a JSON job request (AWS Batch entrypoint)."
    )
    worker.add_argument("--request", required=True, help="JobRequest as JSON.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line. ``guide`` takes its documentation URLs before,
    between or after its options (``guide DOC1 --reference R DOC2``): the URLs
    argparse leaves over after an option are the guide's too."""
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra and args.command == "guide" and not any(e.startswith("-") for e in extra):
        args.urls += extra
    elif extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    return args


def _request_from_args(args: argparse.Namespace) -> JobRequest:
    if args.command == "guide":
        return JobRequest(
            kind="guide",
            urls=args.urls,
            reference_urls=args.reference_urls,
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


def _run(request: JobRequest, *, job_id: str | None = None) -> JobResult:
    from scholar_lens.src.runtime import run_job

    result: JobResult = run_job(request, job_id=job_id)
    _emit(result)
    return result


def _worker_exit(result: JobResult, *, reported: bool) -> int:
    """A failed run whose notification did not go out exits 4, which the
    stack's EventBridge rule reports instead."""
    if result.status != "succeeded" and not reported:
        return EXIT_ERROR
    return _WORKER_EXIT_BY_STATUS[result.status]


def _raw_kind(raw: str) -> Any:
    """The request's ``kind`` when it is a known one, for labelling failures."""
    try:
        kind = json.loads(raw).get("kind")
    except (ValueError, AttributeError):
        return None
    return kind if kind in {"review", "summarize", "guide"} else None


def _report_worker_failure(result: JobResult) -> int:
    """Notify (and store the manifest) for a Batch run that failed outside
    ``run_job``. The exit code tells the stack's alarm whether the app's own
    notification went out (see ``APP_REPORTED_EXIT_CODES``)."""
    _emit(result)
    try:
        from scholar_lens.src.runtime import record_unstarted_run

        reported = record_unstarted_run(result)
    except Exception as e:  # noqa: BLE001 - e.g. the runtime stack fails to import
        sys.stderr.write(f"Could not report the failure: {type(e).__name__}: {e}\n")
        reported = False
    return _worker_exit(result, reported=reported)


def _work(raw: str) -> int:
    """Batch entrypoint: run the JSON request; every failure is reported.

    After a SIGTERM (cleanup and manifest already done inside ``run_job``) the
    process ends at once with 143: a normal interpreter exit would join the
    worker thread of an in-flight model call, past Batch's ~30 s SIGKILL.
    """
    _raise_on_sigterm()
    try:
        return _work_request(raw)
    except SystemExit as e:
        if e.code != _SIGTERM_EXIT:
            raise
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_SIGTERM_EXIT)


def _work_request(raw: str) -> int:
    job_id = os.getenv("AWS_BATCH_JOB_ID")
    try:
        request = JobRequest.model_validate_json(raw)
    except ValidationError as e:
        return _report_worker_failure(
            JobResult(
                kind=_raw_kind(raw),
                input=raw[:200],
                status="rejected",
                error=f"Invalid job request: {e}",
                job_id=job_id,
            )
        )
    try:
        result = _run(request, job_id=job_id)
    except Exception as e:  # noqa: BLE001 - run_job reports its own errors
        return _report_worker_failure(
            JobResult(
                kind=request.kind,
                input=request.describe(),
                error=f"{type(e).__name__}: {e}",
                job_id=job_id,
            )
        )
    return _worker_exit(result, reported=not result.notification_failed)


def _raise_on_sigterm() -> None:
    """Batch stops a job with SIGTERM (timeout, cancel, host termination). Turn it
    into SystemExit so cleanup, metrics and the result manifest still run
    (``runtime.run_coroutine`` does not wait for in-flight model calls)."""

    def handler(_signum: int, _frame: Any) -> None:
        raise SystemExit(_SIGTERM_EXIT)

    signal.signal(signal.SIGTERM, handler)


_SIGTERM_EXIT = 128 + signal.SIGTERM


def main(argv: Sequence[str] | None = None) -> int:
    # faiss-cpu bundles its own OpenMP runtime; tolerate a second copy loaded
    # by another native wheel instead of aborting.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = parse_args(argv)
    if args.command == "worker":
        return _work(args.request)
    if args.command != "status":
        try:
            request = _request_from_args(args)
        except ValidationError as e:
            _emit({"status": "rejected", "error": str(e)})
            return EXIT_REJECTED
    try:
        if args.command == "status":
            return _status(args.job_id, wait=args.wait, timeout=args.timeout)
        if args.remote:
            return _submit(request)
    except Exception as e:  # noqa: BLE001 - report AWS/config errors as JSON
        _emit({"status": "error", "error": f"{type(e).__name__}: {e}"})
        return EXIT_ERROR
    try:
        return _EXIT_BY_STATUS[_run(request).status]
    except KeyboardInterrupt as e:
        # Ctrl-C: run_job has cleaned up and re-raised with its recorded result
        # (cost, tokens, duration); still print one result.
        from scholar_lens.src.runtime import INTERRUPTED_ERROR

        _emit(
            getattr(e, "job_result", None)
            or JobResult(
                kind=request.kind, input=request.describe(), error=INTERRUPTED_ERROR
            )
        )
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
