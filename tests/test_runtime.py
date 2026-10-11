"""Tests for the shared runtime helpers (no real AWS)."""

from __future__ import annotations

from datetime import UTC, datetime
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
        runtime.publish_sns(session, "arn:topic", subject="Hi", message="a\nb")
        sns.publish.assert_called_once_with(
            TopicArn="arn:topic", Subject="Hi", Message="a\nb"
        )

    def test_failure_is_swallowed(self) -> None:
        session = MagicMock()
        session.client.return_value.publish.side_effect = RuntimeError("sns down")
        # Must not raise — a notification failure can't fail the job.
        runtime.publish_sns(session, "arn", subject="s", message="x")


class TestFormatAlarm:
    def test_subject_and_inline_fields_are_aligned(self) -> None:
        ts = datetime(2026, 6, 10, 4, 12, 0, tzinfo=UTC)
        subject, message = runtime.format_alarm(
            event="Tech Guide",
            status="FAILED",
            fields={"Sources": "https://x", "Error": "boom"},
            timestamp=ts,
        )
        assert subject == "[scholar-lens] Tech Guide — FAILED"
        assert message == (
            "Tech Guide FAILED\n"
            "\n"
            "Sources: https://x\n"
            "Error:   boom\n"
            "\n"
            "— 2026-06-10 04:12:00 UTC"
        )

    def test_multiline_field_renders_as_block(self) -> None:
        ts = datetime(2026, 6, 10, 4, 12, 0, tzinfo=UTC)
        _, message = runtime.format_alarm(
            event="Paper Review",
            status="FAILED",
            fields={"Source": "abc", "Trace": "line1\nline2"},
            timestamp=ts,
        )
        assert "Source: abc" in message
        assert "Trace:\nline1\nline2" in message


class TestRunJob:
    """run_job maps pipeline outcomes to a JobResult and always finishes up."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch, pipeline):  # type: ignore[no-untyped-def]
        context = MagicMock()
        context.config.resources.s3_prefix = "scholar-lens"
        monkeypatch.setattr(
            runtime.Config, "load", classmethod(lambda cls: MagicMock())
        )
        monkeypatch.setattr(runtime, "build_context", lambda config: context)
        monkeypatch.setattr(runtime, "load_secrets_from_ssm", lambda *a: None)
        monkeypatch.setattr(runtime, "_run_pipeline", pipeline)
        finished: list = []
        monkeypatch.setattr(
            runtime, "_finish", lambda ctx, result, tracker: finished.append(result)
        )
        return finished

    def test_success_carries_outcome(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from scholar_lens.src.jobs import JobRequest

        async def pipeline(context, request, tracker):  # type: ignore[no-untyped-def]
            return runtime.PipelineOutcome(
                title="LoRA",
                markdown_path=tmp_path / "post.md",
                s3_uri="s3://b/post.md",
                pr_url="https://github.com/o/r/pull/7",
                warnings=["w"],
            )

        finished = self._patch(monkeypatch, pipeline)
        result = runtime.run_job(JobRequest(kind="review", source="2106.09685"))
        assert result.status == "succeeded"
        assert result.pr_url == "https://github.com/o/r/pull/7"
        assert result.warnings == ["w"]
        assert finished == [result]

    def test_rejected_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scholar_lens.src.jobs import JobRequest
        from scholar_lens.src.paper_source import NotAPdfError

        async def pipeline(context, request, tracker):  # type: ignore[no-untyped-def]
            raise NotAPdfError("returned HTML")

        self._patch(monkeypatch, pipeline)
        result = runtime.run_job(JobRequest(kind="summarize", source="https://x.dev/a"))
        assert result.status == "rejected"
        assert "returned HTML" in (result.error or "")

    def test_failure_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.src.jobs import JobRequest

        async def pipeline(context, request, tracker):  # type: ignore[no-untyped-def]
            raise RuntimeError("bedrock down")

        finished = self._patch(monkeypatch, pipeline)
        result = runtime.run_job(JobRequest(kind="guide", urls=["https://docs.x.dev"]))
        assert result.status == "failed"
        assert result.error == "RuntimeError: bedrock down"
        assert finished == [result]

    def test_interrupt_still_finishes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scholar_lens.src.jobs import JobRequest

        async def pipeline(context, request, tracker):  # type: ignore[no-untyped-def]
            raise SystemExit(143)

        finished = self._patch(monkeypatch, pipeline)
        with pytest.raises(SystemExit):
            runtime.run_job(JobRequest(kind="summarize", source="2106.09685"))
        assert finished[0].status == "failed"
        assert finished[0].error == "Interrupted before completion."


class TestResultNotification:
    def test_success_lists_pr_and_cost(self) -> None:
        from scholar_lens.src.jobs import JobResult

        subject, message = runtime.format_result_notification(
            JobResult(
                kind="summarize",
                input="2106.09685",
                status="succeeded",
                title="LoRA",
                pr_url="https://github.com/o/r/pull/7",
                estimated_cost_usd=0.42,
            )
        )
        assert subject == "[scholar-lens] Paper Summary — SUCCEEDED"
        assert "https://github.com/o/r/pull/7" in message
        assert "$0.42" in message

    def test_failure_includes_error(self) -> None:
        from scholar_lens.src.jobs import JobResult

        subject, message = runtime.format_result_notification(
            JobResult(kind="guide", input="u", error="RuntimeError: boom")
        )
        assert subject.endswith("FAILED")
        assert "RuntimeError: boom" in message


class TestFinishAndEarlyFailure:
    def test_finish_publishes_sns_and_writes_manifest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.src.jobs import JobResult
        from scholar_lens.src.metrics import TokenUsageTracker

        monkeypatch.setattr(runtime, "is_running_in_aws", lambda: True)
        monkeypatch.setenv("TOPIC_ARN", "arn:aws:sns:us-west-2:1:t")
        published: list[tuple] = []
        monkeypatch.setattr(
            runtime, "publish_sns", lambda *a, **kw: published.append((a, kw))
        )
        monkeypatch.setattr(runtime, "MetricsEmitter", MagicMock())
        context = MagicMock()
        context.config.resources.s3_prefix = "scholar-lens"
        result = JobResult(kind="review", input="x", status="succeeded", job_id="job-9")
        runtime._finish(context, result, TokenUsageTracker())
        assert published and "SUCCEEDED" in published[0][1]["subject"]
        key, body = context.s3_handler.put_text.call_args.args
        assert key == "scholar-lens/results/job-9.json" and '"job-9"' in body

    def test_missing_config_in_aws_still_notifies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.src.jobs import JobRequest

        def no_config(cls):  # type: ignore[no-untyped-def]
            raise FileNotFoundError("config.yaml")

        monkeypatch.setattr(runtime.Config, "load", classmethod(no_config))
        monkeypatch.setattr(runtime, "is_running_in_aws", lambda: True)
        monkeypatch.setenv("TOPIC_ARN", "arn:aws:sns:us-west-2:1:t")
        sent: list = []
        monkeypatch.setattr(runtime, "publish_sns", lambda *a, **kw: sent.append(kw))
        result = runtime.run_job(JobRequest(kind="summarize", source="2106.09685"))
        assert result.status == "failed" and "config.yaml" in (result.error or "")
        assert sent and "FAILED" in sent[0]["subject"]
