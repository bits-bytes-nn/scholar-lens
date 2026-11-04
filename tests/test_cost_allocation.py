"""Per-kind application inference profiles: names, resolution, the stack."""

from __future__ import annotations

from unittest.mock import MagicMock

import aws_cdk as core
import pytest

from scholar_lens.application.usage import cost_allocation
from scholar_lens.domain.cost_allocation import ARTIFACT_KINDS, application_profile_name
from scholar_lens.src.utils import factories
from scholar_lens.src.utils.models import model_info_for_invoked_id
from scripts.deploy_infra import BedrockProfilesStack

OPUS = "global.anthropic.claude-opus-5-5"
ARN = "arn:aws:bedrock:us-west-2:111122223333:application-inference-profile/p1"


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("review", "scholar-lens-dev-review-claude-opus-5-5"),
        ("survey_write", "scholar-lens-dev-survey_write-claude-opus-5-5"),
    ],
)
def test_profile_names_are_valid_bedrock_names(kind: str, expected: str) -> None:
    name = application_profile_name("scholar-lens", "dev", kind, OPUS)
    assert name == expected and len(name) <= 64


def _session(profiles: list[dict[str, str]]) -> MagicMock:
    session = MagicMock()
    session.client.return_value.list_inference_profiles.return_value = {
        "inferenceProfileSummaries": profiles
    }
    return session


def test_allocated_runs_call_their_kind_profile_and_price_as_the_model() -> None:
    factories._application_profiles.cache_clear()
    name = application_profile_name("scholar-lens", "dev", "guide", OPUS)
    session = _session([{"inferenceProfileName": name, "inferenceProfileArn": ARN}])
    with cost_allocation("scholar-lens", "dev", "guide"):
        assert factories._allocated_profile(session, "us-west-2", OPUS) == ARN
    assert model_info_for_invoked_id(ARN) == model_info_for_invoked_id(OPUS)
    # Outside a run (no allocation) the system profile is called.
    assert factories._allocated_profile(session, "us-west-2", OPUS) == OPUS


def test_a_missing_profile_falls_back_to_the_system_profile() -> None:
    factories._application_profiles.cache_clear()
    with cost_allocation("scholar-lens", "dev", "review"):
        assert factories._allocated_profile(_session([]), "us-west-2", OPUS) == OPUS


def test_profiles_stack_tags_one_profile_per_kind_and_model() -> None:
    app = core.App()
    stack = BedrockProfilesStack(
        app,
        "ProfilesCi",
        project_name="scholar-lens",
        stage="dev",
        system_profile_ids=[OPUS, "global.anthropic.claude-haiku-5-5"],
        env=core.Environment(account="111122223333", region="us-west-2"),
    )
    template = app.synth().get_stack_by_name(stack.stack_name).template
    profiles = [
        r["Properties"]
        for r in template["Resources"].values()
        if r["Type"] == "AWS::Bedrock::ApplicationInferenceProfile"
    ]
    assert len(profiles) == 2 * len(ARTIFACT_KINDS)
    for p in profiles:
        tags = {t["Key"]: t["Value"] for t in p["Tags"]}
        assert tags["ProjectName"] == "scholar-lens" and tags["Kind"] in ARTIFACT_KINDS
        source = p["ModelSource"]["CopyFrom"]
        source = (
            "".join(map(str, source["Fn::Join"][1]))
            if isinstance(source, dict)
            else source
        )
        assert p["InferenceProfileName"] == application_profile_name(
            "scholar-lens", "dev", tags["Kind"], source.rsplit("/", 1)[1]
        )
