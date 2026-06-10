"""Shared runtime wiring for the CLI/Batch entrypoints.

``main.py`` (review/summary) and ``tech_guide_main.py`` (tech guide) both need
the same boilerplate: build boto sessions + an optional S3 handler from config,
load secrets from SSM into the environment, publish an SNS notification, and
construct the Publisher. These lived as drifting copies in each entrypoint (the
SSM-load loop had even diverged to different log levels); they are consolidated
here so both paths share one implementation.
"""

from __future__ import annotations

import os
from pathlib import Path

import boto3
from pydantic import BaseModel, ConfigDict

from ..configs import Config
from .aws_helpers import S3Handler, get_ssm_param_value
from .constants import EnvVars, SSMParams
from .logger import is_running_in_aws, logger
from .publisher import Publisher


class RunContext(BaseModel):
    """Per-run AWS/config context shared by every entrypoint."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: Config
    default_boto_session: boto3.Session
    bedrock_boto_session: boto3.Session
    s3_handler: S3Handler | None = None


def build_context(config: Config) -> RunContext:
    """Build a :class:`RunContext` (boto sessions + optional S3 handler).

    The AWS profile comes from the env in Batch (instance role / ``AWS_PROFILE``)
    and from config locally — matching the previous per-entrypoint behaviour.
    """
    profile_name = (
        os.getenv(EnvVars.AWS_PROFILE_NAME.value)
        if is_running_in_aws()
        else config.resources.profile_name
    )
    default_session = boto3.Session(
        profile_name=profile_name, region_name=config.resources.default_region_name
    )
    context = RunContext(
        config=config,
        default_boto_session=default_session,
        bedrock_boto_session=boto3.Session(
            profile_name=profile_name,
            region_name=config.resources.bedrock_region_name,
        ),
    )
    if config.resources.s3_bucket_name:
        context.s3_handler = S3Handler(default_session, config.resources.s3_bucket_name)
    return context


def load_secrets_from_ssm(
    context: RunContext, ssm_map: dict[SSMParams, EnvVars]
) -> None:
    """Load the given SSM SecureString params into ``os.environ`` (AWS only).

    Best-effort per secret: a missing/optional secret is logged and skipped so a
    run that doesn't need it still proceeds.
    """
    if not is_running_in_aws():
        return
    base = f"/{context.config.resources.project_name}/{context.config.resources.stage}"
    for ssm_param, env_var in ssm_map.items():
        try:
            os.environ[env_var.value] = get_ssm_param_value(
                context.default_boto_session, f"{base}/{ssm_param.value}"
            )
            logger.info("Set env var '%s' from SSM.", env_var.value)
        except Exception as e:  # noqa: BLE001 - optional secret; never abort the run
            logger.info("Could not set '%s' from SSM: %s", env_var.value, e)


def publish_sns(
    session: boto3.Session, topic_arn: str, *, subject: str, lines: list[str]
) -> None:
    """Publish a plain-text SNS notification; a failure must not fail the job."""
    try:
        session.client("sns").publish(
            TopicArn=topic_arn, Subject=subject, Message="\n".join(lines)
        )
    except Exception as e:  # noqa: BLE001 - notification failure must not fail the job
        logger.error("Failed to send SNS notification: %s", e)


def build_publisher(context: RunContext, root_dir: Path) -> Publisher:
    """Construct the artifact Publisher from the run context."""
    return Publisher(
        context.config.resources.github,
        root_dir=root_dir,
        s3_handler=context.s3_handler,
        s3_bucket_name=context.config.resources.s3_bucket_name,
        s3_prefix=context.config.resources.s3_prefix,
    )
