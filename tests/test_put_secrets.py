"""scripts/put_secrets.py against a moto SSM (no real AWS)."""

from __future__ import annotations

from unittest import mock

import boto3
import pytest
from moto import mock_aws

from scholar_lens.configs import Config
from scholar_lens.src.constants import SSMParams
from scripts import put_secrets


@mock_aws
def test_writes_only_the_set_secrets_as_securestring() -> None:
    session = boto3.Session(region_name="us-east-1")
    written = put_secrets.put_secure_secrets(
        session,
        "scholar-lens",
        "dev",
        {SSMParams.GITHUB_TOKEN: "ghp_new", SSMParams.BRAVE_API_KEY: None},
    )
    assert written == ["/scholar-lens/dev/github-token"]
    param = session.client("ssm").get_parameter(
        Name="/scholar-lens/dev/github-token", WithDecryption=True
    )["Parameter"]
    assert param["Type"] == "SecureString" and param["Value"] == "ghp_new"


@mock_aws
def test_named_parameters_limit_what_is_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_new")
    monkeypatch.setenv("BRAVE_API_KEY", "brave")
    session = boto3.Session(region_name="us-east-1")
    with (
        mock.patch.object(Config, "load", return_value=Config()),
        mock.patch.object(put_secrets, "deploy_session", return_value=session),
    ):
        put_secrets.main(["github-token"])
    names = [
        p["Name"] for p in session.client("ssm").describe_parameters()["Parameters"]
    ]
    assert names == ["/scholar-lens/dev/github-token"]


def test_unknown_parameter_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        put_secrets.main(["not-a-secret"])
    assert exit_info.value.code == 2


def test_named_parameter_that_is_unset_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with (
        mock.patch.object(Config, "load", return_value=Config()),
        pytest.raises(SystemExit) as exit_info,
    ):
        put_secrets.main(["github-token"])
    assert exit_info.value.code == 2
