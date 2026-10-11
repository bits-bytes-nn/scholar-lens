"""CLI exit codes and worker failure reporting (no real AWS)."""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import pytest
from botocore.exceptions import ClientError
from pydantic import BaseModel, ValidationError

from scholar_lens import cli
from scholar_lens.src.aws_helpers import BatchJobError
from scholar_lens.src.jobs import (
    JobRequest,
    JobResult,
    JobStatusReport,
    SubmittedJob,
    wait_for_job,
)


def _output(capsys: pytest.CaptureFixture) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


class TestStatus:
    @pytest.mark.parametrize(
        "batch_status,result_status,code",
        [
            ("RUNNING", None, cli.EXIT_PENDING),
            ("FAILED", None, cli.EXIT_FAILED),
            ("SUCCEEDED", None, cli.EXIT_OK),  # no manifest (no bucket)
            ("SUCCEEDED", "succeeded", cli.EXIT_OK),
            ("FAILED", "rejected", cli.EXIT_REJECTED),
        ],
    )
    def test_exit_code(
        self,
        batch_status: str,
        result_status: str | None,
        code: int,
        capsys: pytest.CaptureFixture,
    ) -> None:
        report = JobStatusReport(job_id="j", batch_status=batch_status)
        if result_status:
            report.result = JobResult(kind="review", input="x", status=result_status)  # type: ignore[arg-type]
        with (
            mock.patch.object(cli, "_aws_context"),
            mock.patch("scholar_lens.src.jobs.describe_job", return_value=report),
        ):
            assert cli.main(["status", "j"]) == code
        assert _output(capsys)["batch_status"] == batch_status

    def test_api_error_is_not_a_job_failure(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with (
            mock.patch.object(cli, "_aws_context"),
            mock.patch(
                "scholar_lens.src.jobs.describe_job",
                side_effect=BatchJobError("AccessDenied"),
            ),
        ):
            assert cli.main(["status", "j"]) == cli.EXIT_ERROR
        assert _output(capsys)["status"] == "error"

    def test_malformed_manifest_is_an_error_not_a_rejection(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        context = mock.MagicMock()
        context.default_boto_session.client.return_value.describe_jobs.return_value = {
            "jobs": [{"status": "SUCCEEDED", "jobName": "n"}]
        }
        context.s3_handler.get_text.return_value = (
            '{"kind": "review", "status": "done", "input": "x"}'
        )
        context.config.resources.s3_prefix = "p"
        with mock.patch.object(cli, "_aws_context", return_value=context):
            assert cli.main(["status", "j"]) == cli.EXIT_ERROR
        assert _output(capsys)["status"] == "error"


def test_remote_submits_and_prints_the_job(capsys: pytest.CaptureFixture) -> None:
    submitted = SubmittedJob(job_id="j-1", job_name="n", kind="summarize", input="x")
    with (
        mock.patch.object(cli, "_aws_context"),
        mock.patch("scholar_lens.src.jobs.submit_job", return_value=submitted) as sub,
    ):
        assert cli.main(["summarize", "2106.09685", "--remote", "--no-pr"]) == 0
    request = sub.call_args.args[1]
    assert request == JobRequest(kind="summarize", source="2106.09685", create_pr=False)
    assert _output(capsys)["job_id"] == "j-1"


@pytest.mark.parametrize(
    "argv", [["status", "j"], ["guide", "https://docs.example.com", "--remote"]]
)
def test_config_validation_error_is_an_error_not_a_rejection(
    argv: list[str], capsys: pytest.CaptureFixture
) -> None:
    class Strict(BaseModel):  # stands in for a config.yaml with a bad key
        n: int

    with pytest.raises(ValidationError) as config_error:
        Strict(n="x")  # type: ignore[arg-type]
    with mock.patch.object(cli, "_aws_context", side_effect=config_error.value):
        assert cli.main(argv) == cli.EXIT_ERROR
    assert _output(capsys)["status"] == "error"


def test_invalid_request_is_rejected(capsys: pytest.CaptureFixture) -> None:
    with mock.patch.object(cli, "_aws_context") as context:
        assert cli.main(["summarize", " ", "--remote"]) == cli.EXIT_REJECTED
    context.assert_not_called()
    assert _output(capsys)["status"] == "rejected"


class TestWorker:
    @pytest.fixture(autouse=True)
    def _no_signal_handler(self) -> Any:
        with mock.patch.object(cli, "_raise_on_sigterm"):
            yield

    def test_bad_request_is_reported_with_its_kind(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setenv("AWS_BATCH_JOB_ID", "job-7")
        with mock.patch(
            "scholar_lens.src.runtime.record_unstarted_run", return_value=True
        ) as record:
            code = cli.main(["worker", "--request", '{"kind": "guide"}'])
        assert code == cli.WORKER_REPORTED_REJECTED
        result = record.call_args.args[0]
        assert result.kind == "guide" and result.job_id == "job-7"
        assert _output(capsys)["status"] == "rejected"

    def test_sigterm_ends_the_process_without_joining_threads(self) -> None:
        def interrupted(*_args: Any, **_kwargs: Any) -> Any:
            raise SystemExit(143)  # what the SIGTERM handler raises

        request = '{"kind": "summarize", "source": "2106.09685"}'
        with (
            mock.patch("scholar_lens.src.runtime.run_job", side_effect=interrupted),
            mock.patch.object(cli.os, "_exit", side_effect=RuntimeError) as exit_,
            pytest.raises(RuntimeError),
        ):
            cli.main(["worker", "--request", request])
        exit_.assert_called_once_with(143)

    def test_other_system_exits_propagate(self) -> None:
        def stopped(*_args: Any, **_kwargs: Any) -> Any:
            raise SystemExit(3)

        request = '{"kind": "summarize", "source": "2106.09685"}'
        with (
            mock.patch("scholar_lens.src.runtime.run_job", side_effect=stopped),
            mock.patch.object(cli.os, "_exit") as exit_,
            pytest.raises(SystemExit),
        ):
            cli.main(["worker", "--request", request])
        exit_.assert_not_called()

    def test_unparseable_request_has_no_kind(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with mock.patch(
            "scholar_lens.src.runtime.record_unstarted_run", return_value=True
        ) as record:
            code = cli.main(["worker", "--request", "not json"])
        assert code == cli.WORKER_REPORTED_REJECTED
        assert record.call_args.args[0].kind is None
        capsys.readouterr()

    def test_crash_outside_run_job_is_reported(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        request = JobRequest(kind="summarize", source="x")
        with (
            mock.patch(
                "scholar_lens.src.runtime.run_job", side_effect=ImportError("faiss")
            ),
            mock.patch(
                "scholar_lens.src.runtime.record_unstarted_run", return_value=True
            ) as record,
        ):
            code = cli.main(["worker", "--request", request.model_dump_json()])
        assert code == cli.WORKER_REPORTED_FAILED
        assert "ImportError: faiss" in (record.call_args.args[0].error or "")
        assert _output(capsys)["status"] == "failed"

    def test_failed_run_whose_notification_failed_exits_4(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        request = JobRequest(kind="summarize", source="x")
        result = JobResult(kind="summarize", input="x", error="boom")
        result.notification_failed = True
        with mock.patch("scholar_lens.src.runtime.run_job", return_value=result):
            code = cli.main(["worker", "--request", request.model_dump_json()])
        assert code == cli.EXIT_ERROR and code not in cli.APP_REPORTED_EXIT_CODES
        assert "notification_failed" not in _output(capsys)

    def test_reported_failure_and_success_exit_codes(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        request = JobRequest(kind="summarize", source="x")
        for status, expected in (
            ("failed", cli.WORKER_REPORTED_FAILED),
            ("succeeded", cli.EXIT_OK),
        ):
            result = JobResult(kind="summarize", input="x", status=status)
            with mock.patch("scholar_lens.src.runtime.run_job", return_value=result):
                assert (
                    cli.main(["worker", "--request", request.model_dump_json()])
                    == expected
                )
        capsys.readouterr()

    def test_unsent_report_exits_outside_the_app_reported_codes(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        # The stack's EventBridge rule then notifies instead.
        with mock.patch(
            "scholar_lens.src.runtime.record_unstarted_run",
            side_effect=ImportError("broken image"),
        ):
            code = cli.main(["worker", "--request", "{}"])
        assert code == cli.EXIT_ERROR
        assert code not in cli.APP_REPORTED_EXIT_CODES
        capsys.readouterr()


class TestWaitForJobTransientErrors:
    @staticmethod
    def _error(code: str, status: int) -> BatchJobError:
        try:
            raise ClientError(
                {
                    "Error": {"Code": code},
                    "ResponseMetadata": {"HTTPStatusCode": status},
                },
                "DescribeJobs",
            )
        except ClientError as e:
            try:
                raise BatchJobError("describe failed") from e
            except BatchJobError as wrapped:
                return wrapped

    def test_throttled_poll_is_skipped(self) -> None:
        replies: list[Any] = [
            JobStatusReport(job_id="j", batch_status="RUNNING"),
            self._error("ThrottlingException", 400),
            JobStatusReport(job_id="j", batch_status="SUCCEEDED"),
        ]

        def describe() -> JobStatusReport:
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        report = wait_for_job(
            describe, timeout_seconds=100, sleep=lambda _s: None, clock=lambda: 0.0
        )
        assert report.batch_status == "SUCCEEDED"

    def test_network_error_while_polling_is_skipped(self) -> None:
        from botocore.exceptions import EndpointConnectionError

        def wrapped() -> BatchJobError:
            try:
                raise EndpointConnectionError(endpoint_url="https://batch")
            except EndpointConnectionError as e:
                try:
                    raise BatchJobError("describe failed") from e
                except BatchJobError as error:
                    return error

        replies: list[Any] = [
            JobStatusReport(job_id="j", batch_status="RUNNING"),
            wrapped(),
            JobStatusReport(job_id="j", batch_status="FAILED"),
        ]

        def describe() -> JobStatusReport:
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        report = wait_for_job(
            describe, timeout_seconds=100, sleep=lambda _s: None, clock=lambda: 0.0
        )
        assert report.batch_status == "FAILED"

    def test_first_poll_is_protected_too(self) -> None:
        replies: list[Any] = [
            self._error("ThrottlingException", 400),
            JobStatusReport(job_id="j", batch_status="SUCCEEDED"),
        ]

        def describe() -> JobStatusReport:
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        report = wait_for_job(
            describe, timeout_seconds=100, sleep=lambda _s: None, clock=lambda: 0.0
        )
        assert report.batch_status == "SUCCEEDED"

    def test_access_denied_propagates(self) -> None:
        error = self._error("AccessDeniedException", 403)
        calls = iter([JobStatusReport(job_id="j", batch_status="RUNNING")])

        def describe() -> JobStatusReport:
            report = next(calls, None)
            if report is None:
                raise error
            return report

        with pytest.raises(BatchJobError):
            wait_for_job(
                describe, timeout_seconds=100, sleep=lambda _s: None, clock=lambda: 0.0
            )


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "soon"])
def test_status_timeout_must_be_finite_and_non_negative(value: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(["status", "job-1", "--wait", "--timeout", value])
    assert exit_info.value.code == 2


def test_status_timeout_accepts_seconds() -> None:
    args = cli.build_parser().parse_args(["status", "job-1", "--timeout", "90"])
    assert args.timeout == 90.0
