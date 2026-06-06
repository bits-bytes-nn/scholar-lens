"""CI-only CDK synthesis check.

Synthesizes the ``PaperReviewStack`` against a fake account/region so the
construct graph and IAM/security wiring are validated on every push **without**
needing real AWS credentials (the production ``deploy_infra.py`` entrypoint
resolves the account via STS, which CI cannot do). This is a smoke test: if the
stack fails to synthesize, CI fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import aws_cdk as core
from aws_cdk import Environment

sys.path.append(str(Path(__file__).parent.parent))

from scripts.deploy_infra import PaperReviewStack  # noqa: E402

FAKE_ACCOUNT = "123456789012"
FAKE_REGION = "ap-northeast-2"


def main() -> None:
    app = core.App()
    PaperReviewStack(
        app,
        "PaperReviewCiSynthStack",
        project_name="scholar-lens",
        stage="dev",
        vpc_id=None,
        subnet_ids=None,
        email_address="ci@example.com",
        github_token="dummy",
        langchain_api_key="dummy",
        upstage_api_key="dummy",
        s3_bucket_name="scholar-lens-ci-bucket",
        s3_prefix="scholar-lens",
        bedrock_region_name="us-west-2",
        environment_vars={},
        env=Environment(account=FAKE_ACCOUNT, region=FAKE_REGION),
    )
    assembly = app.synth()
    stack = assembly.get_stack_by_name("PaperReviewCiSynthStack")
    resource_count = len(stack.template.get("Resources", {}))
    print(f"CDK synth OK: {resource_count} resources synthesized.")
    if resource_count == 0:
        raise SystemExit("CDK synth produced no resources.")


if __name__ == "__main__":
    main()
