"""Job contract, Batch submit/status, and CLI argument mapping (no real AWS)."""

from __future__ import annotations

import json
from unittest import mock

import boto3
import pytest
from moto import mock_aws
from pydantic import BaseModel, ConfigDict, ValidationError

from scholar_lens import cli
from scholar_lens.src.aws_helpers import BatchJobError, S3Handler, S3OperationError
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
        return cli._request_from_args(cli.parse_args(argv))

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
        assert code == cli.WORKER_REPORTED_REJECTED
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

    def test_malformed_manifest_is_a_storage_error(self, aws: boto3.Session) -> None:
        handler = S3Handler(aws, BUCKET)
        handler.put_text(result_manifest_key("p", "job-x"), '{"status": "done"}')
        with pytest.raises(S3OperationError, match="Malformed"):
            read_result_manifest(handler, "p", "job-x")

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

    def test_transient_manifest_read_is_polled_again(self) -> None:
        from botocore.exceptions import ClientError

        from scholar_lens.src.aws_helpers import S3OperationError

        slow_down = ClientError(
            {
                "Error": {"Code": "SlowDown"},
                "ResponseMetadata": {"HTTPStatusCode": 503},
            },
            "GetObject",
        )
        calls = 0

        def describe() -> JobStatusReport:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise S3OperationError("read failed") from slow_down
            return JobStatusReport(job_id="j", batch_status="SUCCEEDED")

        report = wait_for_job(
            describe, timeout_seconds=100, sleep=lambda _s: None, clock=lambda: 0.0
        )
        assert calls == 2 and report.batch_status == "SUCCEEDED"


class TestReferenceMaterial:
    def test_guide_from_reference_material_alone(self) -> None:
        request = JobRequest(kind="guide", reference_urls=["https://blog.x.io/a"])
        assert request.urls == [] and request.describe() == "https://blog.x.io/a"

    def test_describe_lists_docs_then_references(self) -> None:
        request = JobRequest(
            kind="guide",
            urls=["https://docs.x.dev"],
            reference_urls=["https://arxiv.org/abs/2309.06180"],
        )
        assert request.describe() == (
            "https://docs.x.dev, https://arxiv.org/abs/2309.06180"
        )

    def test_paper_request_rejects_reference_urls(self) -> None:
        with pytest.raises(ValidationError):
            JobRequest(
                kind="review", source="2106.09685", reference_urls=["https://x.io"]
            )

    def test_json_round_trip_keeps_reference_urls(self) -> None:
        request = JobRequest(
            kind="guide",
            urls=["https://docs.x.dev"],
            reference_urls=["https://blog.x.io/a", "https://arxiv.org/abs/1"],
        )
        restored = JobRequest.model_validate_json(request.model_dump_json())
        assert restored == request

    def test_cli_maps_repeated_reference_options(self) -> None:
        args = cli.parse_args(
            [
                "guide",
                "https://docs.x.dev",
                "https://docs.x.dev/serve",
                "--reference",
                "https://blog.x.io/a",
                "--reference",
                "https://arxiv.org/abs/2309.06180",
            ]
        )
        request = cli._request_from_args(args)
        assert request.urls == ["https://docs.x.dev", "https://docs.x.dev/serve"]
        assert request.reference_urls == [
            "https://blog.x.io/a",
            "https://arxiv.org/abs/2309.06180",
        ]

    def test_cli_accepts_docs_urls_after_options(self) -> None:
        args = cli.parse_args(
            [
                "guide",
                "https://docs.x.dev",
                "--reference",
                "https://blog.x.io/a",
                "https://docs.x.dev/serve",
                "--search-query",
                "x internals",
                "https://docs.x.dev/api",
            ]
        )
        request = cli._request_from_args(args)
        assert request.urls == [
            "https://docs.x.dev",
            "https://docs.x.dev/serve",
            "https://docs.x.dev/api",
        ]
        assert request.reference_urls == ["https://blog.x.io/a"]
        assert request.search_queries == ["x internals"]

    def test_cli_accepts_references_without_docs(self) -> None:
        args = cli.parse_args(["guide", "--reference", "https://blog.x.io/a"])
        request = cli._request_from_args(args)
        assert request.urls == [] and request.reference_urls == ["https://blog.x.io/a"]

    def test_cli_rejects_an_unknown_guide_option(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with pytest.raises(SystemExit) as exit_info:
            cli.parse_args(["guide", "https://docs.x.dev", "--bogus", "x"])
        assert exit_info.value.code == cli.EXIT_REJECTED
        assert json.loads(capsys.readouterr().out)["status"] == "rejected"

    def test_cli_guide_without_any_url_is_rejected(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert cli.main(["guide"]) == cli.EXIT_REJECTED
        assert json.loads(capsys.readouterr().out)["status"] == "rejected"


def test_remote_guide_carries_every_url_to_the_worker(aws: boto3.Session) -> None:
    batch = mock.MagicMock()
    batch.submit_job.return_value = {"jobId": "job-1"}
    real_client = aws.client
    request = JobRequest(
        kind="guide",
        urls=["https://docs.x.dev/a", "https://docs.x.dev/b"],
        reference_urls=["https://blog.x.io/a", "https://arxiv.org/abs/2309.06180"],
    )
    with mock.patch.object(
        aws,
        "client",
        side_effect=lambda name, **kw: (
            batch if name == "batch" else real_client(name, **kw)
        ),
    ):
        submitted = submit_job(aws, request, project_name="p", stage="dev")
    sent = batch.submit_job.call_args.kwargs["parameters"]["request"]
    assert JobRequest.model_validate_json(sent) == request
    assert "https://arxiv.org/abs/2309.06180" in submitted.input


class _WorkerRequestDdd37c8(BaseModel):
    """Static copy of the JobRequest fields a worker image built at ddd37c8
    accepts; it forbids unknown fields, so a newer CLI must not send any."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    source: str | None = None
    repo_urls: list[str] = []
    parse_pdf: bool = False
    urls: list[str] = []
    search_queries: list[str] = []
    discover_subpages: bool = True
    create_pr: bool = True


def _sent_request(aws: boto3.Session, request: JobRequest) -> str:
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
        submit_job(aws, request, project_name="p", stage="dev")
    return str(batch.submit_job.call_args.kwargs["parameters"]["request"])


@pytest.mark.parametrize(
    "request_",
    [
        JobRequest(kind="review", source="1706.03762", repo_urls=["https://g/x"]),
        JobRequest(kind="summarize", source="https://x.org/p.pdf", parse_pdf=True),
        JobRequest(
            kind="guide",
            urls=["https://docs.x.dev/a"],
            search_queries=["x internals"],
            discover_subpages=False,
            create_pr=False,
        ),
    ],
)
def test_submitted_request_is_accepted_by_an_older_worker(
    aws: boto3.Session, request_: JobRequest
) -> None:
    sent = _sent_request(aws, request_)
    _WorkerRequestDdd37c8.model_validate_json(sent)
    assert JobRequest.model_validate_json(sent) == request_
