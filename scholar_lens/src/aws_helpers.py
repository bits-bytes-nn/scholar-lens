import asyncio
import time
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

from .logger import logger


class AWSHandlerError(Exception):
    pass


class S3OperationError(AWSHandlerError):
    pass


class BatchJobError(AWSHandlerError):
    pass


class S3Handler:
    def __init__(self, boto_session: boto3.Session, bucket_name: str):
        if not bucket_name:
            raise ValueError("S3 bucket name is required.")
        self.boto_session = boto_session
        self.bucket_name = bucket_name
        self._s3_client = self.boto_session.client("s3")

    def download_file(self, s3_key: str, local_path: Path):
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            logger.info(
                "Successfully downloaded 's3://%s/%s' to '%s'",
                self.bucket_name,
                s3_key,
                local_path,
            )
        except ClientError as e:
            raise S3OperationError(f"Failed to download '{s3_key}'") from e

    async def download_file_async(self, s3_key: str, local_path: Path):
        try:
            await asyncio.to_thread(self.download_file, s3_key, local_path)
        except S3OperationError as e:
            logger.error("Async download failed: %s", e)
            raise

    def exists(self, s3_key: str) -> bool:
        try:
            self._s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise S3OperationError(f"Failed to check existence of '{s3_key}'") from e

    def upload_directory(
        self,
        local_dir: Path,
        s3_prefix: str,
        file_extensions: list[str] | None = None,
        public_readable: bool = False,
    ) -> int:
        if not local_dir.is_dir():
            raise NotADirectoryError(f"Local path is not a directory: {local_dir}")

        config = TransferConfig(use_threads=True, max_concurrency=10)
        extra_args = {"ACL": "public-read"} if public_readable else {}
        upload_count = 0

        files_to_upload = [
            p
            for p in local_dir.rglob("*")
            if p.is_file() and (not file_extensions or p.suffix in file_extensions)
        ]

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
            except ClientError as e:
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
        except ClientError as e:
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


def submit_batch_job(
    boto_session: boto3.Session,
    job_name: str,
    job_queue: str,
    job_definition: str,
    parameters: dict[str, str] | None = None,
) -> str:
    try:
        batch_client = boto_session.client("batch")
        response = batch_client.submit_job(
            jobName=job_name,
            jobQueue=job_queue,
            jobDefinition=job_definition,
            parameters=parameters or {},
        )
        job_id = response["jobId"]
        logger.info(
            "Successfully submitted batch job '%s' (Job ID: %s)", job_name, job_id
        )
        return job_id
    except ClientError as e:
        raise BatchJobError(f"Failed to submit batch job '{job_name}'") from e


def wait_for_batch_job_completion(
    boto_session: boto3.Session,
    job_id: str,
    timeout_seconds: int = 3600,
    poll_interval_seconds: int = 30,
):
    batch_client = boto_session.client("batch")
    start_time = time.monotonic()

    logger.info(f"Waiting for Batch Job '{job_id}' to complete...")
    while time.monotonic() - start_time < timeout_seconds:
        try:
            response = batch_client.describe_jobs(jobs=[job_id])
            job = response["jobs"][0] if response.get("jobs") else None

            if not job:
                raise BatchJobError(f"Job '{job_id}' not found.")

            status = job["status"]
            if status == "SUCCEEDED":
                logger.info(f"Job '{job_id}' completed successfully.")
                return
            elif status in ["FAILED", "CANCELLED"]:
                reason = job.get("statusReason", "No reason provided.")
                raise BatchJobError(
                    f"Job '{job_id}' ended with status '{status}': {reason}"
                )

            time.sleep(poll_interval_seconds)

        except ClientError as e:
            raise BatchJobError(f"Error describing job '{job_id}'") from e

    raise TimeoutError(f"Timed out waiting for job '{job_id}' to complete.")
