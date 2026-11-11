"""Where the process runs."""

from __future__ import annotations

import os

_AWS_ENV_VARS = (
    "AWS_EXECUTION_ENV",
    "AWS_LAMBDA_FUNCTION_NAME",
    "AWS_BATCH_JOB_ID",
    "ECS_CONTAINER_METADATA_URI",
)


def is_running_in_aws() -> bool:
    return any(env_var in os.environ for env_var in _AWS_ENV_VARS)
