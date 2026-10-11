"""CI-only CDK synthesis check (no AWS credentials needed).

Synthesizes ``PaperReviewStack`` against a fake account and asserts the parts
of the template the application depends on: the single job definition's
command (the CLI ``worker`` contract) and its exit-code-aware retry strategy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import aws_cdk as core
from aws_cdk import Environment

sys.path.append(str(Path(__file__).parent.parent))

from scripts.deploy_infra import JOB_COMMAND, PaperReviewStack  # noqa: E402

FAKE_ACCOUNT = "123456789012"
FAKE_REGION = "ap-northeast-2"
STACK_NAME = "PaperReviewCiSynthStack"


def synth_template() -> dict[str, Any]:
    app = core.App()
    PaperReviewStack(
        app,
        STACK_NAME,
        project_name="scholar-lens",
        stage="dev",
        email_address="ci@example.com",
        s3_bucket_name="scholar-lens-ci-bucket",
        s3_prefix="scholar-lens",
        bedrock_region_name="us-west-2",
        environment_vars={},
        env=Environment(account=FAKE_ACCOUNT, region=FAKE_REGION),
    )
    return app.synth().get_stack_by_name(STACK_NAME).template


def resources_of_type(template: dict[str, Any], type_name: str) -> list[dict[str, Any]]:
    return [
        r for r in template.get("Resources", {}).values() if r.get("Type") == type_name
    ]


def check_template(template: dict[str, Any]) -> None:
    """Raise SystemExit when the template breaks an application contract."""
    job_definitions = resources_of_type(template, "AWS::Batch::JobDefinition")
    if len(job_definitions) != 1:
        raise SystemExit(f"Expected 1 job definition, found {len(job_definitions)}.")
    props = job_definitions[0]["Properties"]
    command = props["ContainerProperties"]["Command"]
    if command != JOB_COMMAND:
        raise SystemExit(f"Job command {command} != contract {JOB_COMMAND}.")
    if not props.get("RetryStrategy", {}).get("EvaluateOnExit"):
        raise SystemExit("Job definition must retry only on infrastructure failures.")
    if resources_of_type(template, "AWS::Lambda::Function"):
        raise SystemExit("Unexpected Lambda functions in the stack.")


def main() -> None:
    template = synth_template()
    check_template(template)
    print(f"CDK synth OK: {len(template.get('Resources', {}))} resources synthesized.")


if __name__ == "__main__":
    main()
