import asyncio
from pathlib import Path

import boto3
from boto3.exceptions import S3UploadFailedError
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import BotoCoreError, ClientError

from .logger import logger


class AWSHandlerError(Exception):
    pass


class S3OperationError(AWSHandlerError):
    pass


class BatchJobError(AWSHandlerError):
    pass


# boto3's managed upload re-raises service errors as S3UploadFailedError, which
# is not a ClientError; connection problems surface as BotoCoreError.
_UPLOAD_ERRORS = (ClientError, S3UploadFailedError, BotoCoreError)


class S3Handler:
    def __init__(self, boto_session: boto3.Session, bucket_name: str):
        if not bucket_name:
            raise ValueError("S3 bucket name is required.")
        self.boto_session = boto_session
        self.bucket_name = bucket_name
        self._s3_client = self.boto_session.client("s3")

    def put_text(self, s3_key: str, text: str) -> None:
        try:
            self._s3_client.put_object(
                Bucket=self.bucket_name, Key=s3_key, Body=text.encode("utf-8")
            )
        except (ClientError, BotoCoreError) as e:
            raise S3OperationError(f"Failed to write '{s3_key}'") from e

    def get_text(self, s3_key: str) -> str | None:
        """Return the object's text, or None when it does not exist."""
        try:
            body = self._s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            data: bytes = body["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise S3OperationError(f"Failed to read '{s3_key}'") from e
        except BotoCoreError as e:  # connection / read timeout (transient)
            raise S3OperationError(f"Failed to read '{s3_key}'") from e
        return data.decode("utf-8")

    def upload_directory(
        self,
        local_dir: Path,
        s3_prefix: str,
        public_readable: bool = False,
    ) -> int:
        if not local_dir.is_dir():
            raise NotADirectoryError(f"Local path is not a directory: {local_dir}")

        config = TransferConfig(use_threads=True, max_concurrency=10)
        extra_args = {"ACL": "public-read"} if public_readable else {}
        upload_count = 0

        files_to_upload = [p for p in local_dir.rglob("*") if p.is_file()]

        for local_path in files_to_upload:
            relative_path = local_path.relative_to(local_dir)
            s3_key = (Path(s3_prefix) / relative_path).as_posix()
            try:
                self._s3_client.upload_file(
                    str(local_path),
                    self.bucket_name,
                    s3_key,
                    Config=config,
                    ExtraArgs=extra_args,
                )
                upload_count += 1
            except _UPLOAD_ERRORS as e:
                logger.error(
                    "Failed to upload '%s' to 's3://%s/%s': %s",
                    local_path,
                    self.bucket_name,
                    s3_key,
                    e,
                )
        logger.info(
            "Completed directory upload. Uploaded %d of %d files.",
            upload_count,
            len(files_to_upload),
        )
        if upload_count < len(files_to_upload):
            # A partial upload publishes a post with broken images; surface it.
            raise S3OperationError(
                f"Uploaded only {upload_count} of {len(files_to_upload)} files "
                f"from '{local_dir}'."
            )
        return upload_count

    def upload_file(self, local_path: Path, s3_prefix: str | None = None):
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        s3_key = self._construct_s3_key(local_path.name, s3_prefix)
        try:
            self._s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
            logger.info(
                "Successfully uploaded '%s' to 's3://%s/%s'",
                local_path.name,
                self.bucket_name,
                s3_key,
            )
        except _UPLOAD_ERRORS as e:
            raise S3OperationError(f"Failed to upload '{local_path}'") from e

    @staticmethod
    def _construct_s3_key(file_name: str, s3_prefix: str | None = None) -> str:
        if not s3_prefix:
            return file_name
        return f"{s3_prefix.strip('/')}/{file_name}"

    async def upload_file_async(self, local_path: Path, s3_prefix: str | None = None):
        try:
            await asyncio.to_thread(self.upload_file, local_path, s3_prefix)
        except (S3OperationError, FileNotFoundError) as e:
            logger.error("Async upload failed: %s", e)
            raise


def get_account_id(boto_session: boto3.Session) -> str:
    try:
        sts_client = boto_session.client("sts")
        return sts_client.get_caller_identity()["Account"]
    except ClientError as e:
        raise AWSHandlerError("Failed to get AWS Account ID") from e


def get_ssm_param_value(boto_session: boto3.Session, param_name: str) -> str:
    try:
        ssm_client = boto_session.client("ssm")
        response = ssm_client.get_parameter(Name=param_name, WithDecryption=True)
        return response["Parameter"]["Value"]
    except ClientError as e:
        raise AWSHandlerError(f"Failed to get SSM parameter '{param_name}'") from e
