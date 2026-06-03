"""Tests for the AWS helper layer (S3, STS, SSM, Batch).

All AWS access is faked with ``moto``; the Batch helpers are exercised with
``unittest.mock`` because a full moto Batch setup (VPC + compute environment)
is disproportionately heavy for these unit tests. No real AWS or network
calls are made.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import boto3
import pytest
from moto import mock_aws

from scholar_lens.src.aws_helpers import (
    BatchJobError,
    S3Handler,
    get_account_id,
    get_ssm_param_value,
    submit_batch_job,
    wait_for_batch_job_completion,
)

BUCKET_NAME = "scholar-lens-test-bucket"


@pytest.fixture
def s3_bucket(boto_session: boto3.Session):
    """Start moto and create the working bucket; yield its name.

    ``mock_aws`` is driven from the fixture (rather than a class decorator) so
    that both the bucket creation and the test body share one mock context.
    """
    with mock_aws():
        boto_session.client("s3").create_bucket(Bucket=BUCKET_NAME)
        yield BUCKET_NAME


class TestS3HandlerConstruction:
    def test_empty_bucket_name_rejected(self, boto_session: boto3.Session) -> None:
        with pytest.raises(ValueError):
            S3Handler(boto_session, "")


class TestS3HandlerSync:
    def test_upload_then_exists_returns_true(
        self, boto_session: boto3.Session, s3_bucket: str, tmp_path: Path
    ) -> None:
        local = tmp_path / "doc.txt"
        local.write_text("hello", encoding="utf-8")
        handler = S3Handler(boto_session, s3_bucket)

        handler.upload_file(local, s3_prefix="docs")

        assert handler.exists("docs/doc.txt") is True

    def test_exists_false_for_missing_key(
        self, boto_session: boto3.Session, s3_bucket: str
    ) -> None:
        handler = S3Handler(boto_session, s3_bucket)
        assert handler.exists("nope/missing.txt") is False

    def test_download_file_round_trips_bytes(
        self, boto_session: boto3.Session, s3_bucket: str, tmp_path: Path
    ) -> None:
        payload = b"\x00\x01round-trip\x02\x03"
        boto_session.client("s3").put_object(
            Bucket=s3_bucket, Key="blob.bin", Body=payload
        )
        handler = S3Handler(boto_session, s3_bucket)

        dest = tmp_path / "nested" / "blob.bin"
        handler.download_file("blob.bin", dest)

        assert dest.read_bytes() == payload

    def test_upload_directory_filters_by_extension(
        self, boto_session: boto3.Session, s3_bucket: str, tmp_path: Path
    ) -> None:
        src = tmp_path / "assets"
        src.mkdir()
        (src / "image.png").write_bytes(b"\x89PNG\r\n")
        (src / "notes.txt").write_text("ignore me", encoding="utf-8")
        handler = S3Handler(boto_session, s3_bucket)

        count = handler.upload_directory(src, s3_prefix="out", file_extensions=[".png"])

        assert count == 1
        assert handler.exists("out/image.png") is True
        assert handler.exists("out/notes.txt") is False

    def test_upload_directory_public_readable_does_not_error(
        self, boto_session: boto3.Session, s3_bucket: str, tmp_path: Path
    ) -> None:
        src = tmp_path / "pub"
        src.mkdir()
        (src / "page.png").write_bytes(b"\x89PNG\r\n")
        handler = S3Handler(boto_session, s3_bucket)

        count = handler.upload_directory(src, s3_prefix="public", public_readable=True)

        assert count == 1
        assert handler.exists("public/page.png") is True

    def test_upload_directory_rejects_non_directory(
        self, boto_session: boto3.Session, s3_bucket: str, tmp_path: Path
    ) -> None:
        not_a_dir = tmp_path / "file.txt"
        not_a_dir.write_text("x", encoding="utf-8")
        handler = S3Handler(boto_session, s3_bucket)

        with pytest.raises(NotADirectoryError):
            handler.upload_directory(not_a_dir, s3_prefix="out")

    def test_upload_file_missing_local_raises(
        self, boto_session: boto3.Session, s3_bucket: str, tmp_path: Path
    ) -> None:
        handler = S3Handler(boto_session, s3_bucket)
        with pytest.raises(FileNotFoundError):
            handler.upload_file(tmp_path / "ghost.txt")


class TestS3HandlerAsync:
    # ``mock_aws`` is used as a context manager here (rather than a class
    # decorator) so it does not interfere with pytest-asyncio's detection of
    # the async test functions.
    async def test_upload_file_async_round_trips(
        self, boto_session: boto3.Session, tmp_path: Path
    ) -> None:
        with mock_aws():
            boto_session.client("s3").create_bucket(Bucket=BUCKET_NAME)
            local = tmp_path / "async.txt"
            local.write_text("async upload", encoding="utf-8")
            handler = S3Handler(boto_session, BUCKET_NAME)

            await handler.upload_file_async(local, s3_prefix="async")

            assert handler.exists("async/async.txt") is True

    async def test_download_file_async_round_trips(
        self, boto_session: boto3.Session, tmp_path: Path
    ) -> None:
        with mock_aws():
            client = boto_session.client("s3")
            client.create_bucket(Bucket=BUCKET_NAME)
            client.put_object(Bucket=BUCKET_NAME, Key="a.bin", Body=b"async-bytes")
            handler = S3Handler(boto_session, BUCKET_NAME)

            dest = tmp_path / "a.bin"
            await handler.download_file_async("a.bin", dest)

            assert dest.read_bytes() == b"async-bytes"


@mock_aws
class TestAccountAndSsm:
    def test_get_account_id_returns_moto_account(
        self, boto_session: boto3.Session
    ) -> None:
        assert get_account_id(boto_session) == "123456789012"

    def test_get_ssm_param_value_round_trips_securestring(
        self, boto_session: boto3.Session
    ) -> None:
        ssm = boto_session.client("ssm")
        ssm.put_parameter(
            Name="/scholar-lens/secret",
            Value="s3cr3t-value",
            Type="SecureString",
        )

        assert (
            get_ssm_param_value(boto_session, "/scholar-lens/secret") == "s3cr3t-value"
        )

    def test_get_ssm_param_value_round_trips_string(
        self, boto_session: boto3.Session
    ) -> None:
        ssm = boto_session.client("ssm")
        ssm.put_parameter(
            Name="/scholar-lens/plain", Value="plain-value", Type="String"
        )

        assert get_ssm_param_value(boto_session, "/scholar-lens/plain") == "plain-value"


class TestSubmitBatchJob:
    def test_returns_job_id_from_response(self, boto_session: boto3.Session) -> None:
        fake_client = mock.Mock()
        fake_client.submit_job.return_value = {"jobId": "job-123"}

        with mock.patch.object(boto_session, "client", return_value=fake_client):
            job_id = submit_batch_job(
                boto_session,
                job_name="my-job",
                job_queue="my-queue",
                job_definition="my-def",
                parameters={"k": "v"},
            )

        assert job_id == "job-123"
        fake_client.submit_job.assert_called_once_with(
            jobName="my-job",
            jobQueue="my-queue",
            jobDefinition="my-def",
            parameters={"k": "v"},
        )

    def test_defaults_parameters_to_empty_dict(
        self, boto_session: boto3.Session
    ) -> None:
        fake_client = mock.Mock()
        fake_client.submit_job.return_value = {"jobId": "job-456"}

        with mock.patch.object(boto_session, "client", return_value=fake_client):
            submit_batch_job(
                boto_session,
                job_name="j",
                job_queue="q",
                job_definition="d",
            )

        _, kwargs = fake_client.submit_job.call_args
        assert kwargs["parameters"] == {}

    def test_client_error_wrapped_in_batch_job_error(
        self, boto_session: boto3.Session
    ) -> None:
        from botocore.exceptions import ClientError

        fake_client = mock.Mock()
        fake_client.submit_job.side_effect = ClientError(
            {"Error": {"Code": "X", "Message": "boom"}}, "SubmitJob"
        )

        with mock.patch.object(boto_session, "client", return_value=fake_client):
            with pytest.raises(BatchJobError):
                submit_batch_job(
                    boto_session,
                    job_name="j",
                    job_queue="q",
                    job_definition="d",
                )


class TestWaitForBatchJobCompletion:
    def test_returns_on_succeeded(self, boto_session: boto3.Session) -> None:
        fake_client = mock.Mock()
        fake_client.describe_jobs.return_value = {"jobs": [{"status": "SUCCEEDED"}]}

        with mock.patch.object(boto_session, "client", return_value=fake_client):
            # Should return None without raising.
            assert wait_for_batch_job_completion(boto_session, "job-1") is None

    def test_polls_until_succeeded(
        self, boto_session: boto3.Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = mock.Mock()
        fake_client.describe_jobs.side_effect = [
            {"jobs": [{"status": "RUNNING"}]},
            {"jobs": [{"status": "RUNNABLE"}]},
            {"jobs": [{"status": "SUCCEEDED"}]},
        ]
        # Avoid real sleeping during polling.
        monkeypatch.setattr("scholar_lens.src.aws_helpers.time.sleep", lambda _s: None)

        with mock.patch.object(boto_session, "client", return_value=fake_client):
            wait_for_batch_job_completion(
                boto_session, "job-1", poll_interval_seconds=0
            )

        assert fake_client.describe_jobs.call_count == 3

    @pytest.mark.parametrize("status", ["FAILED", "CANCELLED"])
    def test_raises_on_terminal_failure(
        self, boto_session: boto3.Session, status: str
    ) -> None:
        fake_client = mock.Mock()
        fake_client.describe_jobs.return_value = {
            "jobs": [{"status": status, "statusReason": "nope"}]
        }

        with mock.patch.object(boto_session, "client", return_value=fake_client):
            with pytest.raises(BatchJobError):
                wait_for_batch_job_completion(boto_session, "job-1")

    def test_raises_when_job_not_found(self, boto_session: boto3.Session) -> None:
        fake_client = mock.Mock()
        fake_client.describe_jobs.return_value = {"jobs": []}

        with mock.patch.object(boto_session, "client", return_value=fake_client):
            with pytest.raises(BatchJobError):
                wait_for_batch_job_completion(boto_session, "job-1")

    def test_times_out(
        self, boto_session: boto3.Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = mock.Mock()
        fake_client.describe_jobs.return_value = {"jobs": [{"status": "RUNNING"}]}
        # Drive the monotonic clock forward so the timeout window elapses.
        clock = iter([0.0, 1.0, 100.0, 200.0])
        monkeypatch.setattr(
            "scholar_lens.src.aws_helpers.time.monotonic",
            lambda: next(clock),
        )
        monkeypatch.setattr("scholar_lens.src.aws_helpers.time.sleep", lambda _s: None)

        with mock.patch.object(boto_session, "client", return_value=fake_client):
            with pytest.raises(TimeoutError):
                wait_for_batch_job_completion(boto_session, "job-1", timeout_seconds=10)
