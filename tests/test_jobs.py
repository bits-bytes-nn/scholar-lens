"""Job contract, Batch submit/status, and CLI argument mapping (no real AWS)."""

from __future__ import annotations

import json
from unittest import mock

import boto3
import pytest
from moto import mock_aws
from pydantic import ValidationError

from scholar_lens import cli
from scholar_lens.src.aws_helpers import BatchJobError, S3Handler
from scholar_lens.src.jobs import (
    JobRequest,
    JobResult,
    JobStatusReport,
    describe_job,
    read_result_manifest,
    result_manifest_key,
    submit_job,
    wait_for_job,
    write_result_manifest,
)

BUCKET = "scholar-lens-test-bucket"


class TestJobRequest:
    def test_paper_request_needs_source(self) -> None:
        with pytest.raises(ValidationError, match="needs a 'source'"):
            JobRequest(kind="review")

    def test_guide_request_needs_urls(self) -> None:
        with pytest.raises(ValidationError, match="at least one URL"):
            JobRequest(kind="guide")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "summarize", "source": "2106.09685", "urls": ["https://x.dev"]},
            {"kind": "guide", "urls": ["https://x.dev"], "source": "2106.09685"},
        ],
    )
    def test_mixed_inputs_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            JobRequest(**kwargs)

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JobRequest.model_validate({"kind": "review", "source": "x", "mode": "y"})

    def test_json_round_trip_preserves_multiword_queries(self) -> None:
        request = JobRequest(
            kind="guide",
            urls=["https://docs.x.dev/a", "https://docs.x.dev/b"],
            search_queries=["argo cd rollout strategy"],
        )
        restored = JobRequest.model_validate_json(request.model_dump_json())
        assert restored == request
        assert restored.search_queries == ["argo cd rollout strategy"]


class TestCliMapping:
    def _request(self, argv: list[str]) -> JobRequest:
        return cli._request_from_args(cli.build_parser().parse_args(argv))

    def test_review_options(self) -> None:
        request = self._request(
            [
                "review",
                "2106.09685",
                "--repo-url",
                "https://github.com/microsoft/LoRA",
                "--repo-url",
                "https://github.com/o/r",
                "--parse-pdf",
                "--no-pr",
            ]
        )
        assert request.kind == "review"
        assert request.repo_urls == [
            "https://github.com/microsoft/LoRA",
            "https://github.com/o/r",
        ]
        assert request.parse_pdf is True
        assert request.create_pr is False

    def test_guide_options(self) -> None:
        request = self._request(
            [
                "guide",
                "https://docs.x.dev",
                "--search-query",
                "paged attention internals",
                "--no-subpages",
            ]
        )
        assert request.urls == ["https://docs.x.dev"]
        assert request.search_queries == ["paged attention internals"]
        assert request.discover_subpages is False

    def test_invalid_request_exits_rejected(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        code = cli.main(["worker", "--request", '{"kind": "review"}'])
        assert code == cli.EXIT_REJECTED
        assert json.loads(capsys.readouterr().out)["status"] == "rejected"

    @pytest.mark.parametrize(
        "status,code",
        [("succeeded", 0), ("failed", 1), ("rejected", 2)],
    )
    def test_run_exit_code_follows_status(
        self, status: str, code: int, capsys: pytest.CaptureFixture
    ) -> None:
        result = JobResult(kind="summarize", input="2106.09685", status=status)
        with mock.patch("scholar_lens.src.runtime.run_job", return_value=result):
            assert cli.main(["summarize", "2106.09685"]) == code
        assert json.loads(capsys.readouterr().out)["status"] == status

    def test_worker_passes_batch_job_id(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setenv("AWS_BATCH_JOB_ID", "job-123")
        result = JobResult(kind="summarize", input="x", status="succeeded")
        request = JobRequest(kind="summarize", source="x")
        with (
            mock.patch("scholar_lens.cli._raise_on_sigterm"),
            mock.patch("scholar_lens.src.runtime.run_job", return_value=result) as run,
        ):
            cli.main(["worker", "--request", request.model_dump_json()])
        assert run.call_args.kwargs["job_id"] == "job-123"
        assert run.call_args.args[0] == request
        capsys.readouterr()


@pytest.fixture
def aws(boto_session: boto3.Session):
    with mock_aws():
        boto_session.client("s3").create_bucket(Bucket=BUCKET)
        ssm = boto_session.client("ssm")
        ssm.put_parameter(Name="/p/dev/batch-job-queue", Value="q", Type="String")
        ssm.put_parameter(Name="/p/dev/batch-job-definition", Value="d", Type="String")
        yield boto_session


class TestSubmitAndDescribe:
    def test_submit_sends_request_as_single_parameter(self, aws: boto3.Session) -> None:
        batch = mock.MagicMock()
        batch.submit_job.return_value = {"jobId": "job-1"}
        real_client = aws.client
        with mock.patch.object(
            aws,
            "client",
            side_effect=lambda name, **kw: (
                batch if name == "batch" else real_client(name, **kw)
            ),
        ):
            request = JobRequest(kind="summarize", source="2106.09685")
            submitted = submit_job(aws, request, project_name="p", stage="dev")
        kwargs = batch.submit_job.call_args.kwargs
        assert submitted.job_id == "job-1"
        assert kwargs["jobQueue"] == "q" and kwargs["jobDefinition"] == "d"
        assert (
            JobRequest.model_validate_json(kwargs["parameters"]["request"]) == request
        )

    def test_describe_attaches_manifest_when_finished(self, aws: boto3.Session) -> None:
        handler = S3Handler(aws, BUCKET)
        result = JobResult(
            kind="review", input="x", status="succeeded", job_id="job-1", pr_url="u"
        )
        write_result_manifest(handler, "scholar-lens", result)
        batch = mock.MagicMock()
        batch.describe_jobs.return_value = {
            "jobs": [
                {"jobName": "n", "status": "SUCCEEDED", "createdAt": 1_700_000_000_000}
            ]
        }
        with mock.patch.object(aws, "client", return_value=batch):
            report = describe_job(
                aws, "job-1", s3_handler=handler, s3_prefix="scholar-lens"
            )
        assert report.result == result
        assert report.created_at is not None

    def test_describe_running_job_has_no_result(self, aws: boto3.Session) -> None:
        batch = mock.MagicMock()
        batch.describe_jobs.return_value = {"jobs": [{"status": "RUNNING"}]}
        with mock.patch.object(aws, "client", return_value=batch):
            report = describe_job(
                aws, "job-1", s3_handler=S3Handler(aws, BUCKET), s3_prefix=""
            )
        assert report.batch_status == "RUNNING" and report.result is None

    def test_describe_unknown_job_raises(self, aws: boto3.Session) -> None:
        batch = mock.MagicMock()
        batch.describe_jobs.return_value = {"jobs": []}
        with mock.patch.object(aws, "client", return_value=batch):
            with pytest.raises(BatchJobError, match="not found"):
                describe_job(aws, "nope", s3_handler=None, s3_prefix="")

    def test_manifest_missing_returns_none(self, aws: boto3.Session) -> None:
        assert read_result_manifest(S3Handler(aws, BUCKET), "p", "job-x") is None
        assert result_manifest_key("", "job-x") == "results/job-x.json"


class TestWaitForJob:
    def test_polls_until_terminal(self) -> None:
        states = iter(["RUNNABLE", "RUNNING", "SUCCEEDED"])
        sleeps: list[float] = []

        report = wait_for_job(
            lambda: JobStatusReport(job_id="j", batch_status=next(states)),
            timeout_seconds=1000,
            poll_seconds=5,
            sleep=sleeps.append,
            clock=lambda: 0.0,
        )
        assert report.batch_status == "SUCCEEDED"
        assert sleeps == [5, 5]

    def test_returns_last_state_on_timeout(self) -> None:
        now = iter([0.0, 0.0, 20.0, 20.0])
        report = wait_for_job(
            lambda: JobStatusReport(job_id="j", batch_status="RUNNING"),
            timeout_seconds=10,
            poll_seconds=5,
            sleep=lambda _s: None,
            clock=lambda: next(now),
        )
        assert report.batch_status == "RUNNING"
