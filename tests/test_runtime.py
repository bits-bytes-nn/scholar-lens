"""Tests for the shared runtime helpers (no real AWS)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scholar_lens.src import runtime
from scholar_lens.src.constants import EnvVars, SSMParams


class TestLoadSecretsFromSsm:
    def test_noop_outside_aws(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runtime, "is_running_in_aws", lambda: False)
        called = MagicMock()
        monkeypatch.setattr(runtime, "get_ssm_param_value", called)
        runtime.load_secrets_from_ssm(
            MagicMock(), {SSMParams.GITHUB_TOKEN: EnvVars.GITHUB_TOKEN}
        )
        called.assert_not_called()

    def test_loads_into_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runtime, "is_running_in_aws", lambda: True)
        monkeypatch.setattr(
            runtime, "get_ssm_param_value", lambda session, path: "secret"
        )
        monkeypatch.delenv(EnvVars.GITHUB_TOKEN.value, raising=False)
        ctx = MagicMock()
        ctx.config.resources.project_name = "p"
        ctx.config.resources.stage = "dev"
        runtime.load_secrets_from_ssm(
            ctx, {SSMParams.GITHUB_TOKEN: EnvVars.GITHUB_TOKEN}
        )
        import os

        assert os.environ[EnvVars.GITHUB_TOKEN.value] == "secret"

    def test_missing_secret_is_skipped_not_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runtime, "is_running_in_aws", lambda: True)

        def boom(session, path):  # type: ignore[no-untyped-def]
            raise RuntimeError("not found")

        monkeypatch.setattr(runtime, "get_ssm_param_value", boom)
        ctx = MagicMock()
        ctx.config.resources.project_name = "p"
        ctx.config.resources.stage = "dev"
        # Must not raise — an optional secret simply isn't set.
        runtime.load_secrets_from_ssm(
            ctx, {SSMParams.BRAVE_API_KEY: EnvVars.BRAVE_API_KEY}
        )


class TestPublishSns:
    def test_publishes_subject_and_body(self) -> None:
        session = MagicMock()
        sns = session.client.return_value
        runtime.publish_sns(session, "arn:topic", subject="Hi", lines=["a", "b"])
        sns.publish.assert_called_once_with(
            TopicArn="arn:topic", Subject="Hi", Message="a\nb"
        )

    def test_failure_is_swallowed(self) -> None:
        session = MagicMock()
        session.client.return_value.publish.side_effect = RuntimeError("sns down")
        # Must not raise — a notification failure can't fail the job.
        runtime.publish_sns(session, "arn", subject="s", lines=["x"])
