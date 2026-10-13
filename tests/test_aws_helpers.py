"""Tests for the AWS helper layer (S3, STS, SSM), faked with ``moto``."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import boto3
import pytest
from boto3.exceptions import S3UploadFailedError
from moto import mock_aws

from scholar_lens.src.aws_helpers import (
    S3Handler,
    S3OperationError,
    get_account_id,
    get_ssm_param_value,
)

BUCKET_NAME = "scholar-lens-test-bucket"


def _keys(session: boto3.Session, bucket: str = BUCKET_NAME) -> set[str]:
    listed = session.client("s3").list_objects_v2(Bucket=bucket)
    return {o["Key"] for o in listed.get("Contents", [])}


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
    def test_upload_file_lands_under_the_prefix(
        self, boto_session: boto3.Session, s3_bucket: str, tmp_path: Path
    ) -> None:
        local = tmp_path / "doc.txt"
        local.write_text("hello", encoding="utf-8")
        S3Handler(boto_session, s3_bucket).upload_file(local, s3_prefix="docs")
        assert _keys(boto_session) == {"docs/doc.txt"}

    def test_upload_directory_uploads_every_file(
        self, boto_session: boto3.Session, s3_bucket: str, tmp_path: Path
    ) -> None:
        src = tmp_path / "assets"
        (src / "sub").mkdir(parents=True)
        (src / "image.png").write_bytes(b"\x89PNG\r\n")
        (src / "sub" / "b.jpg").write_bytes(b"\xff\xd8")
        count = S3Handler(boto_session, s3_bucket).upload_directory(src, "out")
        assert count == 2
        assert _keys(boto_session) == {"out/image.png", "out/sub/b.jpg"}

    def test_upload_directory_public_readable_does_not_error(
        self, boto_session: boto3.Session, s3_bucket: str, tmp_path: Path
    ) -> None:
        src = tmp_path / "pub"
        src.mkdir()
        (src / "page.png").write_bytes(b"\x89PNG\r\n")
        handler = S3Handler(boto_session, s3_bucket)

        count = handler.upload_directory(src, s3_prefix="public", public_readable=True)

        assert count == 1
        assert _keys(boto_session) == {"public/page.png"}

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

            assert _keys(boto_session) == {"async/async.txt"}


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


class TestS3Text:
    def test_put_then_get_round_trips_unicode(
        self, boto_session: boto3.Session, s3_bucket: str
    ) -> None:
        handler = S3Handler(boto_session, s3_bucket)
        handler.put_text("results/job.json", '{"title": "리뷰"}')
        assert handler.get_text("results/job.json") == '{"title": "리뷰"}'

    def test_get_missing_returns_none(
        self, boto_session: boto3.Session, s3_bucket: str
    ) -> None:
        assert S3Handler(boto_session, s3_bucket).get_text("nope.json") is None


class TestUploadDirectoryPartialFailure:
    def test_raises_when_a_file_fails(
        self, boto_session: boto3.Session, s3_bucket: str, tmp_path: Path
    ) -> None:
        (tmp_path / "a.png").write_bytes(b"x")
        (tmp_path / "b.png").write_bytes(b"y")
        handler = S3Handler(boto_session, s3_bucket)
        real_upload = handler._s3_client.upload_file

        def flaky(path, *args, **kwargs):  # type: ignore[no-untyped-def]
            if path.endswith("b.png"):
                # What boto3's managed transfer raises for a denied PutObject.
                raise S3UploadFailedError("An error occurred (AccessDenied)")
            return real_upload(path, *args, **kwargs)

        with mock.patch.object(handler._s3_client, "upload_file", side_effect=flaky):
            with pytest.raises(S3OperationError, match="1 of 2"):
                handler.upload_directory(tmp_path, "assets")

    def test_upload_file_wraps_the_managed_transfer_error(
        self, boto_session: boto3.Session, s3_bucket: str, tmp_path: Path
    ) -> None:
        local = tmp_path / "a.md"
        local.write_text("x", encoding="utf-8")
        handler = S3Handler(boto_session, s3_bucket)
        failing = mock.MagicMock(side_effect=S3UploadFailedError("denied"))
        with mock.patch.object(handler._s3_client, "upload_file", failing):
            with pytest.raises(S3OperationError, match="Failed to upload"):
                handler.upload_file(local, "posts")


def test_s3_network_errors_become_s3_operation_errors() -> None:
    from botocore.exceptions import EndpointConnectionError

    handler = S3Handler.__new__(S3Handler)
    handler.bucket_name = "b"
    handler._s3_client = mock.MagicMock()
    handler._s3_client.get_object.side_effect = EndpointConnectionError(
        endpoint_url="https://s3"
    )
    handler._s3_client.put_object.side_effect = EndpointConnectionError(
        endpoint_url="https://s3"
    )
    with pytest.raises(S3OperationError):
        handler.get_text("k")
    with pytest.raises(S3OperationError):
        handler.put_text("k", "v")
