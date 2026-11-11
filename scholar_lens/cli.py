"""``scholar-lens`` command line: generate, submit, and track artifacts.

Every command prints one JSON object on stdout (logs go to stderr), so agents
and scripts can parse the outcome directly::

    scholar-lens summarize 2106.09685
    scholar-lens review "https://openreview.net/pdf?id=nZeVKeeFYf9" --remote
    scholar-lens guide "https://docs.vllm.ai/en/latest/" --search-query "vLLM paged attention"
    scholar-lens guide "https://docs.vllm.ai/en/latest/" --reference "https://arxiv.org/abs/2309.06180"
    scholar-lens survey plan "LLM preference optimization" --paper 2305.18290 --reference "https://huggingface.co/blog/rlhf"
    scholar-lens survey write --plan papers/survey-llm-preference-optimization/plan.json
    scholar-lens publish papers/guide-x/x.rejected.md
    scholar-lens status <job-id> --wait
    scholar-lens cancel <job-id>
    scholar-lens costs --since 2026-11-01 --by stage --billed

Exit codes: 0 succeeded (or submitted), 1 failed, 2 rejected input,
3 still running (``status``), 4 a remote command itself failed (``--remote``,
``status`` or ``cancel``: ``"status": "error"`` for AWS credentials, config, Batch/S3 API
errors), so the job's state is unknown, 5 the artifact was saved but the
requested blog PR failed (a ``failed`` result with ``pr_error``, its
``markdown_path`` and ``s3_uri`` kept; also from ``status`` and ``publish``).
``publish`` exits like a run: 0, 1 (the post still breaks a blog rule, its
rejected copy kept), 2 (no such post or no publish request next to it), 5. A
local run that cannot start (a
broken config) reports ``failed`` and exits 1. The Batch ``worker`` exits 10 (failed) or 11 (rejected) once it
has sent its own notification, 4 when that notification could not be sent,
and 143 (128 + SIGTERM) when Batch stopped it (timeout, cancel, host
termination) after printing the interrupted run's result.
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

from scholar_lens.domain.jobs import (
    JobAlreadyFinished,
    JobRequest,
    JobResult,
    validation_summary,
)
from scholar_lens.domain.survey.plan import SurveyInputError
from scholar_lens.entrypoints.logs import configure_logging

EXIT_OK, EXIT_FAILED, EXIT_REJECTED, EXIT_PENDING, EXIT_ERROR = 0, 1, 2, 3, 4
EXIT_PR_FAILED = 5
PROG = "scholar-lens"
_EXIT_CODES_HELP = (
    "Exit codes: 0 succeeded or submitted, 1 failed, 2 rejected input, "
    "3 still running, 4 the command itself failed, 5 artifact saved but the "
    "blog PR failed (see pr_error)."
)
_EXIT_BY_STATUS = {
    "succeeded": EXIT_OK,
    "failed": EXIT_FAILED,
    "rejected": EXIT_REJECTED,
}


def result_exit_code(result: JobResult) -> int:
    """The CLI exit code of a finished run: its status, except a run whose
    artifact was saved but whose blog PR failed."""
    if result.pr_error is not None:
        return EXIT_PR_FAILED
    return _EXIT_BY_STATUS[result.status]


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

# The cost allocation tag on every stack resource (scripts/deploy_infra.py).
# The stack tags its resources ProjectName=<resources.project_name>.

# How long `status --wait` polls before returning the current state: one
# 3-hour Batch attempt plus queueing time.
DEFAULT_WAIT_SECONDS = 4 * 60 * 60


def _day(value: str) -> Any:
    """A UTC day (YYYY-MM-DD) as the datetime of its midnight."""
    from datetime import UTC, datetime

    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM-DD day: {value!r}") from e


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
        prog=PROG,
        description=(
            "Generate AI/ML paper reviews, summaries, technical guides and "
            "paper surveys."
        ),
        epilog=_EXIT_CODES_HELP,
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
        sub.add_argument(
            "--max-cost-usd",
            type=float,
            default=None,
            metavar="USD",
            help=(
                "This run's cost ceiling instead of the configured one (a Batch "
                "job otherwise uses the config.yaml of the last deploy)."
            ),
        )
        sub.add_argument(
            "--fresh",
            action="store_true",
            help=(
                "Compute every stage again instead of resuming from the stages "
                "a run with the same inputs saved (also reads the sources again "
                "and, for a guide, gathers them again). The evidence engines "
                "only: a review or summary on the sections engine ignores it."
            ),
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
            "--engine",
            choices=("sections", "evidence"),
            default=None,
            help="This run's writing engine instead of the configured one.",
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
    guide.add_argument(
        "--fresh-sources",
        action="store_true",
        help=(
            "Gather the sources again instead of reusing the corpus a run with "
            "the same inputs saved (tech_guide.source_snapshot_ttl_days)."
        ),
    )
    add_run_options(guide)

    survey = commands.add_parser(
        "survey",
        help="Survey of a research topic's major papers (plan, then write).",
        description=(
            "Survey of a research topic's major papers. 'plan' finds and "
            "selects the papers and saves an editable plan file; 'write' "
            "turns a (possibly edited) plan into the post."
        ),
    )
    survey_stages = survey.add_subparsers(dest="survey_stage", required=True)
    survey_plan = survey_stages.add_parser(
        "plan",
        help="Find the topic's major papers and save an editable plan (JSON).",
        description=(
            "Find the topic's major papers (OpenAlex, arXiv), select and group "
            "them into lineages, and save an editable plan file."
        ),
    )
    survey_plan.add_argument(
        "topic", help="Research topic, e.g. 'LLM preference optimization'."
    )
    survey_plan.add_argument(
        "--paper",
        dest="paper_ids",
        action="append",
        default=[],
        metavar="ID_OR_URL",
        help="A paper the survey must include: arXiv id or URL, or DOI (repeatable).",
    )
    survey_plan.add_argument(
        "--reference",
        dest="reference_urls",
        action="append",
        default=[],
        metavar="URL",
        help=(
            "Reference material to consult: a blog post, survey article or paper "
            "page (repeatable). It shapes the scope, the papers it links become "
            "candidates (not required), and the write stage reads its articles "
            "again and lists them as further reading."
        ),
    )
    survey_plan.add_argument(
        "--since",
        dest="since_year",
        type=int,
        metavar="YEAR",
        help="First year covered.",
    )
    survey_plan.add_argument(
        "--until",
        dest="until_year",
        type=int,
        metavar="YEAR",
        help="Last year covered.",
    )
    survey_plan.add_argument(
        "--remote",
        action="store_true",
        help="Submit to AWS Batch and return the job id instead of running here.",
    )

    survey_write = survey_stages.add_parser(
        "write",
        help="Write the survey post from a plan file.",
        description=(
            "Write the survey post from a plan file (as 'survey plan' saved it, "
            "or edited): paper reading, outline, sections, blog checks, PR."
        ),
    )
    survey_write.add_argument(
        "--plan",
        required=True,
        metavar="PATH_OR_S3_URI",
        help=(
            "The plan file: a local path or s3://<bucket>/<key> in the configured "
            "bucket (with --remote a local plan is uploaded there first)."
        ),
    )
    add_run_options(survey_write)

    publish = commands.add_parser(
        "publish",
        help="Check a saved or rejected post again and open its blog PR.",
        description=(
            "Check a post a run saved (--no-pr) or rejected (*.rejected.md), "
            "possibly edited, against the blog rules again and open or update "
            "its pull request on the run's branch. Runs locally."
        ),
    )
    publish.add_argument(
        "post",
        metavar="MARKDOWN_PATH_OR_S3_URI",
        help=(
            "The post: a local path, or s3://<bucket>/<key> in the configured "
            "bucket (a Batch run's posts/ or rejected/ copy)."
        ),
    )
    publish.add_argument(
        "--no-pr",
        dest="create_pr",
        action="store_false",
        help="Check and save the post without opening a pull request.",
    )
    publish.add_argument(
        "--resolved",
        action="store_true",
        help=(
            "The run's blocking problems are fixed in the edited post: they no "
            "longer block the PR and stay in its checklist as items to check."
        ),
    )

    cancel = commands.add_parser(
        "cancel",
        help="Stop a remote job (cancelled before it starts, terminated after).",
    )
    cancel.add_argument("job_id")

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

    costs = commands.add_parser(
        "costs",
        help="Report what the Batch runs of a period cost (from result manifests).",
    )
    costs.add_argument(
        "--since", required=True, type=_day, help="First day, YYYY-MM-DD (UTC)."
    )
    costs.add_argument(
        "--until",
        type=_day,
        default=None,
        help="Day after the last, YYYY-MM-DD (UTC; default: tomorrow).",
    )
    costs.add_argument(
        "--by",
        choices=["kind", "stage", "model", "day"],
        default="kind",
        help="Group the runs by (default: %(default)s).",
    )
    costs.add_argument(
        "--billed",
        action="store_true",
        help="Add the billed cost of the project's tagged resources (Cost Explorer).",
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
    if args.command == "survey" and args.survey_stage == "write":
        return JobRequest(
            kind="survey_write",
            plan=args.plan,
            create_pr=args.create_pr,
            fresh=args.fresh,
            max_cost_usd=args.max_cost_usd,
        )
    if args.command == "survey":
        return JobRequest(
            kind="survey_plan",
            topic=args.topic,
            paper_ids=args.paper_ids,
            reference_urls=args.reference_urls,
            since_year=args.since_year,
            until_year=args.until_year,
        )
    if args.command == "guide":
        return JobRequest(
            kind="guide",
            urls=args.urls,
            reference_urls=args.reference_urls,
            search_queries=args.search_queries,
            discover_subpages=args.discover_subpages,
            fresh_sources=args.fresh_sources,
            create_pr=args.create_pr,
            fresh=args.fresh,
            max_cost_usd=args.max_cost_usd,
        )
    return JobRequest(
        kind=args.command,
        source=args.source,
        repo_urls=args.repo_urls,
        parse_pdf=args.parse_pdf,
        create_pr=args.create_pr,
        fresh=args.fresh,
        engine=args.engine,
        max_cost_usd=args.max_cost_usd,
    )


def _emit(payload: BaseModel | dict[str, Any]) -> None:
    data = (
        payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    )
    sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    sys.stdout.flush()


def _aws_context() -> Any:
    from scholar_lens.bootstrap import build_context
    from scholar_lens.configs import Config

    return build_context(Config.load())


def _submit(request: JobRequest) -> int:
    from scholar_lens.adapters.aws.batch import submit_job

    context = _aws_context()
    resources = context.config.resources
    if request.kind == "survey_write":
        from scholar_lens.application.use_cases.survey_write import stage_plan_for_batch

        _check_plan_bucket(request.plan or "", context.s3_handler, resources.s3_prefix)
        request = stage_plan_for_batch(
            request, context.s3_handler, resources.s3_prefix, files=context.files
        )
    submitted = submit_job(
        context.default_boto_session,
        request,
        project_name=resources.project_name,
        stage=resources.stage,
    )
    _emit(submitted)
    return EXIT_OK


def _check_plan_bucket(plan: str, s3_handler: Any, prefix: str) -> None:
    """An ``s3://`` plan must be under ``<bucket>/<prefix>/`` of the configured
    bucket, the only place the job role can read: ``SurveyInputError``
    (rejected) before anything is submitted, instead of a job that is
    denied later."""
    if not plan.startswith("s3://"):
        return
    bucket, _, key = plan.removeprefix("s3://").partition("/")
    configured = getattr(s3_handler, "bucket_name", None)
    scope = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
    if bucket != configured or not key.startswith(scope):
        raise SurveyInputError(
            f"The plan {plan!r} is not under s3://{configured or '<no bucket>'}/"
            f"{scope}, where the job can read it."
        )


def _status(job_id: str, *, wait: bool, timeout: float) -> int:
    from scholar_lens.application.use_cases.status import job_status
    from scholar_lens.bootstrap import job_queue

    report = job_status(
        job_queue(_aws_context()), job_id, wait=wait, timeout_seconds=timeout
    )
    _emit(report)
    if report.result is not None:
        return result_exit_code(report.result)
    if report.batch_status == "FAILED":
        return EXIT_FAILED
    if report.batch_status == "SUCCEEDED":
        return EXIT_OK  # finished; no manifest (e.g. no S3 bucket configured)
    return EXIT_PENDING  # not finished yet: ask again later


def _cancel(job_id: str) -> int:
    from scholar_lens.application.use_cases.status import cancel_job
    from scholar_lens.bootstrap import job_queue

    _emit(cancel_job(job_queue(_aws_context()), job_id))
    return EXIT_OK


def _costs(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime, timedelta

    from scholar_lens.application.use_cases.costs import report_costs
    from scholar_lens.bootstrap import cost_sources

    until = args.until or datetime.now(UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(days=1)
    if until <= args.since:  # a usage error needs no credentials
        _emit({"status": "rejected", "error": "--until must be after --since."})
        return EXIT_REJECTED
    context = _aws_context()
    if not context.config.resources.s3_bucket_name:
        _emit({"status": "error", "error": "No S3 bucket is configured."})
        return EXIT_ERROR
    ledger, billing = cost_sources(context, billed=args.billed)
    report = report_costs(
        ledger, since=args.since, until=until, by=args.by, billing=billing
    )
    _emit(report)
    return EXIT_OK


def _run(request: JobRequest, *, job_id: str | None = None) -> JobResult:
    from scholar_lens.entrypoints.runtime import run_job

    result: JobResult = run_job(request, job_id=job_id)
    _emit(result)
    return result


def _publish(post: str, *, create_pr: bool, resolved: bool) -> int:
    """``publish``: one result, the exit code a run would have."""
    from scholar_lens.entrypoints.runtime import run_publish

    result = run_publish(post, create_pr=create_pr, resolved=resolved)
    _emit(result)
    return result_exit_code(result)


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
    kinds = {"review", "summarize", "guide", "survey_plan", "survey_write"}
    return kind if kind in kinds else None


def _report_worker_failure(result: JobResult) -> int:
    """Notify (and store the manifest) for a Batch run that failed outside
    ``run_job``. The exit code tells the stack's alarm whether the app's own
    notification went out (see ``APP_REPORTED_EXIT_CODES``)."""
    _emit(result)
    try:
        from scholar_lens.entrypoints.runtime import record_unstarted_run

        reported = record_unstarted_run(result)
    except Exception as e:  # noqa: BLE001 - e.g. the runtime stack fails to import
        sys.stderr.write(f"Could not report the failure: {type(e).__name__}: {e}\n")
        reported = False
    return _worker_exit(result, reported=reported)


def _work(raw: str) -> int:
    """Batch entrypoint: run the JSON request; every failure is reported.

    After a SIGTERM (cleanup and manifest already done inside ``run_job``) the
    run's result is printed and the process ends at once with 143: a normal
    interpreter exit would join the worker thread of an in-flight model call,
    past Batch's ~30 s SIGKILL. A SIGTERM outside ``run_job`` (before the run
    starts or after its result was printed) prints nothing more.
    """
    _raise_on_sigterm()
    try:
        return _work_request(raw)
    except SystemExit as e:
        if e.code != _SIGTERM_EXIT:
            raise
        if (recorded := getattr(e, "job_result", None)) is not None:
            _emit(recorded)
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
    configure_logging()
    args = parse_args(argv)
    if args.command == "worker":
        return _work(args.request)
    if args.command == "publish":
        return _publish(args.post, create_pr=args.create_pr, resolved=args.resolved)
    if args.command == "costs":
        try:
            return _costs(args)
        except Exception as e:  # noqa: BLE001 - report AWS/config errors as JSON
            _emit({"status": "error", "error": f"{type(e).__name__}: {e}"})
            return EXIT_ERROR
    if args.command == "cancel":
        try:
            return _cancel(args.job_id)
        except JobAlreadyFinished as e:
            _emit({"status": "rejected", "error": str(e)})
            return EXIT_REJECTED
        except Exception as e:  # noqa: BLE001 - report AWS/config errors as JSON
            _emit({"status": "error", "error": f"{type(e).__name__}: {e}"})
            return EXIT_ERROR
    if args.command != "status":
        try:
            request = _request_from_args(args)
        except ValidationError as e:
            # A usage error, as argparse reports its own (prog: message).
            _emit(
                {
                    "status": "rejected",
                    "error": f"{PROG}: invalid arguments: {validation_summary(e)}",
                }
            )
            return EXIT_REJECTED
    try:
        if args.command == "status":
            return _status(args.job_id, wait=args.wait, timeout=args.timeout)
        if args.remote:
            return _submit(request)
    except SurveyInputError as e:  # e.g. a survey plan that cannot be staged
        _emit({"status": "rejected", "error": str(e)})
        return EXIT_REJECTED
    except Exception as e:  # noqa: BLE001 - report AWS/config errors as JSON
        _emit({"status": "error", "error": f"{type(e).__name__}: {e}"})
        return EXIT_ERROR
    try:
        return result_exit_code(_run(request))
    except KeyboardInterrupt as e:
        # Ctrl-C: run_job has cleaned up and re-raised with its recorded result
        # (cost, tokens, duration); still print one result.
        from scholar_lens.entrypoints.runtime import INTERRUPTED_ERROR

        _emit(
            getattr(e, "job_result", None)
            or JobResult(
                kind=request.kind, input=request.describe(), error=INTERRUPTED_ERROR
            )
        )
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
