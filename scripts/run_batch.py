import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import boto3
from pytz import timezone

sys.path.append(str(Path(__file__).parent.parent))
from scholar_lens.configs import Config
from scholar_lens.src import (
    AppConstants,
    EnvVars,
    SSMParams,
    arg_as_bool,
    get_ssm_param_value,
    is_running_in_aws,
    logger,
    submit_batch_job,
    wait_for_batch_job_completion,
)


def main(job_prefix: str, params: Dict[str, Any]) -> None:
    config = Config.load()
    profile_name = (
        os.environ.get(EnvVars.AWS_PROFILE_NAME.value)
        if is_running_in_aws()
        else config.resources.profile_name
    )
    boto_session = boto3.Session(
        region_name=config.resources.default_region_name, profile_name=profile_name
    )

    job_queue, job_definition = _get_batch_job_details(boto_session, config)
    timestamp = datetime.now(timezone("UTC")).strftime("%Y%m%d%H%M%S")
    job_name = f"{config.resources.project_name}-{config.resources.stage}-{job_prefix}-{timestamp}"

    sanitized_params = _sanitize_parameters_for_batch(params)
    logger.info(
        "Submitting batch job '%s' with parameters: %s", job_name, sanitized_params
    )

    job_id = submit_batch_job(
        boto_session,
        job_name,
        job_queue,
        job_definition,
        parameters=sanitized_params,
    )

    logger.info("Batch job submitted with ID '%s'", job_id)
    wait_for_batch_job_completion(boto_session, job_id)


def _get_batch_job_details(
    boto3_session: boto3.Session, config: Config
) -> tuple[str, str]:
    base_path = f"/{config.resources.project_name}/{config.resources.stage}"
    try:
        job_queue = get_ssm_param_value(
            boto3_session, f"{base_path}/{SSMParams.BATCH_JOB_QUEUE.value}"
        )
        job_definition = get_ssm_param_value(
            boto3_session, f"{base_path}/{SSMParams.BATCH_JOB_DEFINITION.value}"
        )
        if not job_queue or not job_definition:
            raise ValueError("Job queue or definition is missing from SSM parameters.")
        return job_queue, job_definition
    except Exception as e:
        logger.error("Failed to retrieve Batch job details from SSM: %s", e)
        raise


def _sanitize_parameters_for_batch(params: Dict[str, Any]) -> Dict[str, str]:
    return {
        key: str(value) if value is not None else AppConstants.NULL_STRING
        for key, value in params.items()
    }


def _parse_cli_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Scholar Lens Batch Job Submitter")
    parser.add_argument(
        "--arxiv-id", type=str, required=True, help="arXiv ID of the paper."
    )
    parser.add_argument(
        "--repo-urls", type=str, nargs="*", help="Associated GitHub repository URLs."
    )
    parser.add_argument(
        "--parse-pdf", type=arg_as_bool, default=False, help="Force PDF parsing."
    )
    args = parser.parse_args()

    if not args.arxiv_id or args.arxiv_id.lower() == AppConstants.NULL_STRING:
        logger.error("A valid arXiv ID is required.")
        sys.exit(1)

    repo_urls = None
    if args.repo_urls and args.repo_urls != [AppConstants.NULL_STRING]:
        repo_urls = " ".join(args.repo_urls)

    return {
        "arxiv_id": args.arxiv_id,
        "repo_urls": repo_urls,
        "parse_pdf": str(args.parse_pdf).lower(),
    }


if __name__ == "__main__":
    try:
        job_params = _parse_cli_args()
        main("paper-review", params=job_params)
    except Exception as e:
        logger.error("Job submission failed: %s", e, exc_info=True)
        sys.exit(1)
