"""Tests for the shared runtime helpers (no real AWS)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scholar_lens import bootstrap
from scholar_lens.adapters.files import LocalOrS3Files
from scholar_lens.application import run as app_run
from scholar_lens.domain.constants import EnvVars, SSMParams
from scholar_lens.entrypoints import runtime


def _run_context(config: object, **fields: object) -> app_run.RunContext:
    """A run context on the configured blog (no AWS)."""
    fields.setdefault("publisher_factory", lambda ctx: MagicMock())
    fields.setdefault("files", LocalOrS3Files(fields.get("s3_handler")))  # type: ignore[arg-type]
    return app_run.RunContext(
        config=config,  # type: ignore[arg-type]
        blog=bootstrap.ConfiguredBlog(config),  # type: ignore[arg-type]
        root_dir=Path("."),
        **fields,  # type: ignore[arg-type]
    )


@pytest.fixture(autouse=True)
def _github_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # A run that would open a PR needs a token before it starts.
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")


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
        monkeypatch.delenv(EnvVars.BRAVE_API_KEY.value, raising=False)
        # Must not raise — an optional secret simply isn't set.
        runtime.load_secrets_from_ssm(
            ctx, {SSMParams.BRAVE_API_KEY: EnvVars.BRAVE_API_KEY}
        )
        import os

        assert EnvVars.BRAVE_API_KEY.value not in os.environ

    @pytest.mark.parametrize(
        "code,level",
        [("ParameterNotFound", "INFO"), ("AccessDeniedException", "WARNING")],
    )
    def test_only_an_unset_secret_is_quiet(
        self, monkeypatch: pytest.MonkeyPatch, code: str, level: str
    ) -> None:
        from botocore.exceptions import ClientError

        monkeypatch.setattr(runtime, "is_running_in_aws", lambda: True)

        def fail(session, path):  # type: ignore[no-untyped-def]
            try:
                raise ClientError({"Error": {"Code": code}}, "GetParameter")
            except ClientError as e:
                raise RuntimeError("SSM read failed") from e

        monkeypatch.setattr(runtime, "get_ssm_param_value", fail)
        logged = MagicMock()
        monkeypatch.setattr(runtime, "logger", logged)
        ctx = MagicMock()
        ctx.config.resources.project_name = "p"
        ctx.config.resources.stage = "dev"
        runtime.load_secrets_from_ssm(
            ctx, {SSMParams.BRAVE_API_KEY: EnvVars.BRAVE_API_KEY}
        )
        assert getattr(logged, level.lower()).called
        if level == "INFO":
            logged.warning.assert_not_called()


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
        assert runtime.publish_sns(session, "arn", subject="s", message="x") is False


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
        monkeypatch.setattr(runtime, "run_use_case", pipeline)
        finished: list = []
        self.notified: list[bool] = []

        def finish(ctx, result, tracker, *, notify=True):  # type: ignore[no-untyped-def]
            finished.append(result)
            self.notified.append(notify)

        monkeypatch.setattr(runtime, "_finish", finish)
        return finished

    def test_success_carries_outcome(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from scholar_lens.domain.jobs import JobRequest

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

    @pytest.mark.parametrize("fresh", [False, True])
    def test_the_run_context_carries_fresh(
        self, monkeypatch: pytest.MonkeyPatch, fresh: bool
    ) -> None:
        from scholar_lens.domain.jobs import JobRequest

        seen: list[bool] = []

        async def pipeline(context, request, tracker):  # type: ignore[no-untyped-def]
            seen.append(context.fresh)
            return runtime.PipelineOutcome(title="LoRA")

        self._patch(monkeypatch, pipeline)
        runtime.run_job(JobRequest(kind="review", source="2106.09685", fresh=fresh))
        assert seen == [fresh]

    def test_a_failed_pr_fails_the_run_and_keeps_the_artifact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scholar_lens.domain.jobs import JobRequest

        async def pipeline(context, request, tracker):  # type: ignore[no-untyped-def]
            return runtime.PipelineOutcome(
                title="LoRA",
                markdown_path=tmp_path / "post.md",
                s3_uri="s3://b/post.md",
                pr_error="403 from GitHub",
                warnings=["w"],
            )

        finished = self._patch(monkeypatch, pipeline)
        result = runtime.run_job(JobRequest(kind="review", source="2106.09685"))
        assert result.status == "failed"
        assert result.pr_error == "403 from GitHub"
        assert result.error and "403 from GitHub" in result.error
        assert result.markdown_path == str(tmp_path / "post.md")
        assert result.s3_uri == "s3://b/post.md" and result.pr_url is None
        assert result.warnings == ["w"]
        # A finished failure, not an interrupt: the failure is announced.
        assert finished == [result] and self.notified == [True]

    def test_a_batch_result_points_to_s3_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A Batch run's local files are gone with its container."""
        from scholar_lens.domain.jobs import JobRequest

        async def pipeline(context, request, tracker):  # type: ignore[no-untyped-def]
            return runtime.PipelineOutcome(
                title="LoRA",
                markdown_path=tmp_path / "post.md",
                plan_path=tmp_path / "plan.md",
                s3_uri="s3://b/post.md",
                blocking=["b"],
            )

        self._patch(monkeypatch, pipeline)
        request = JobRequest(kind="review", source="2106.09685")
        result = runtime.run_job(request, job_id="job-1")
        assert result.markdown_path is None and result.plan_path is None
        assert result.s3_uri == "s3://b/post.md" and result.blocking == ["b"]
        local = runtime.run_job(request)
        assert local.markdown_path == str(tmp_path / "post.md")
        assert local.plan_path == str(tmp_path / "plan.md")

    def test_rejected_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scholar_lens.adapters.scholarly.paper_source import NotAPdfError
        from scholar_lens.domain.jobs import JobRequest

        async def pipeline(context, request, tracker):  # type: ignore[no-untyped-def]
            raise NotAPdfError("returned HTML")

        self._patch(monkeypatch, pipeline)
        result = runtime.run_job(JobRequest(kind="summarize", source="https://x.dev/a"))
        assert result.status == "rejected"
        assert "returned HTML" in (result.error or "")

    def test_fetch_outage_is_a_failure_not_a_rejection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.adapters.scholarly.paper_source import PaperFetchError
        from scholar_lens.domain.jobs import JobRequest

        async def pipeline(context, request, tracker):  # type: ignore[no-untyped-def]
            raise PaperFetchError("503 from the host")

        self._patch(monkeypatch, pipeline)
        result = runtime.run_job(JobRequest(kind="summarize", source="https://x.dev/a"))
        assert result.status == "failed"
        assert "503" in (result.error or "")

    def test_failure_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.domain.jobs import JobRequest

        async def pipeline(context, request, tracker):  # type: ignore[no-untyped-def]
            raise RuntimeError("bedrock down")

        finished = self._patch(monkeypatch, pipeline)
        result = runtime.run_job(JobRequest(kind="guide", urls=["https://docs.x.dev"]))
        assert result.status == "failed"
        assert result.error == "RuntimeError: bedrock down"
        assert finished == [result]
        assert self.notified == [True]

    def test_interrupt_still_finishes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scholar_lens.domain.jobs import JobRequest

        async def pipeline(context, request, tracker):  # type: ignore[no-untyped-def]
            raise SystemExit(143)

        finished = self._patch(monkeypatch, pipeline)
        with pytest.raises(SystemExit) as raised:
            runtime.run_job(JobRequest(kind="summarize", source="2106.09685"))
        # The worker prints this result before it exits.
        assert raised.value.job_result is finished[0]  # type: ignore[attr-defined]
        assert finished[0].status == "failed"
        assert finished[0].error == runtime.INTERRUPTED_ERROR
        # Batch may retry the job: no FAILED notification from the app.
        assert self.notified == [False]

    def test_ctrl_c_carries_the_recorded_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.domain.jobs import JobRequest

        async def pipeline(context, request, tracker):  # type: ignore[no-untyped-def]
            tracker.input_tokens = 1_200
            raise KeyboardInterrupt

        finished = self._patch(monkeypatch, pipeline)
        with pytest.raises(KeyboardInterrupt) as raised:
            runtime.run_job(JobRequest(kind="summarize", source="2106.09685"))
        result = raised.value.job_result  # type: ignore[attr-defined]
        assert result is finished[0]
        assert result.error == runtime.INTERRUPTED_ERROR
        assert result.input_tokens == 1_200


@pytest.mark.parametrize(
    ("kind", "function", "fields"),
    [
        ("review", "run_paper", {"source": "2106.09685"}),
        ("summarize", "run_paper", {"source": "2106.09685"}),
        ("guide", "run_guide", {"urls": ["https://d.io"]}),
        ("survey_plan", "run_survey_plan", {"topic": "LLM preference optimization"}),
        ("survey_write", "run_survey_write", {"plan": "plan.json"}),
    ],
)
def test_run_pipeline_dispatches_each_kind(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    function: str,
    fields: dict[str, object],
) -> None:
    import asyncio

    from scholar_lens.domain.jobs import JobRequest

    request = JobRequest(kind=kind, **fields)  # type: ignore[arg-type]
    outcome = runtime.PipelineOutcome(title=kind)
    calls: list[tuple[str, object]] = []
    for attr in ("run_paper", "run_guide", "run_survey_plan", "run_survey_write"):

        async def fake(context, req, tracker, tools, _attr=attr):  # type: ignore[no-untyped-def]
            calls.append((_attr, req))
            return outcome

        monkeypatch.setattr(bootstrap, attr, fake)

    got = asyncio.run(bootstrap.run_use_case(MagicMock(), request, MagicMock()))
    assert got is outcome
    assert calls == [(function, request)]


class TestResultNotification:
    def test_success_lists_pr_and_cost(self) -> None:
        from scholar_lens.domain.jobs import JobResult

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
        from scholar_lens.domain.jobs import JobResult

        subject, message = runtime.format_result_notification(
            JobResult(kind="guide", input="u", error="RuntimeError: boom")
        )
        assert subject.endswith("FAILED")
        assert "RuntimeError: boom" in message

    def test_every_warning_is_listed(self) -> None:
        from scholar_lens.domain.jobs import JobResult

        _, message = runtime.format_result_notification(
            JobResult(kind="review", input="x", warnings=["PR failed", "tag new"])
        )
        assert "- PR failed" in message and "- tag new" in message

    def test_unknown_kind_has_a_generic_label(self) -> None:
        from scholar_lens.domain.jobs import JobResult

        subject, _ = runtime.format_result_notification(
            JobResult(kind=None, input="{}", status="rejected")
        )
        assert subject == "[scholar-lens] Scholar Lens Job — REJECTED"


class TestFinishAndEarlyFailure:
    def test_finish_publishes_sns_and_writes_manifest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.adapters.bedrock.usage import TokenUsageTracker
        from scholar_lens.domain.jobs import JobResult

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

    def test_unsent_failure_notification_is_flagged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.adapters.bedrock.usage import TokenUsageTracker
        from scholar_lens.domain.jobs import JobResult

        monkeypatch.setattr(runtime, "is_running_in_aws", lambda: True)
        monkeypatch.setenv("TOPIC_ARN", "arn:aws:sns:us-west-2:1:t")
        monkeypatch.setattr(runtime, "publish_sns", lambda *a, **kw: False)
        monkeypatch.setattr(runtime, "MetricsEmitter", MagicMock())
        result = JobResult(kind="review", input="x", error="boom")
        runtime._finish(MagicMock(), result, TokenUsageTracker())
        assert result.notification_failed
        assert "notification_failed" not in result.model_dump()  # not in the manifest

    def test_missing_config_in_aws_still_notifies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.domain.jobs import JobRequest

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


class TestRecordUnstartedRun:
    def test_notifies_in_the_topic_region_and_writes_manifest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.domain.jobs import JobResult

        monkeypatch.setattr(runtime, "is_running_in_aws", lambda: True)
        monkeypatch.setenv("TOPIC_ARN", "arn:aws:sns:us-east-1:1:t")
        sessions: list = []

        def session(**kwargs):  # type: ignore[no-untyped-def]
            sessions.append(kwargs)
            return MagicMock()

        monkeypatch.setattr(runtime, "aws_session", session)
        config = MagicMock()
        config.resources.s3_bucket_name = "b"
        config.resources.s3_prefix = "p"
        monkeypatch.setattr(runtime.Config, "load", classmethod(lambda cls: config))
        handler = MagicMock()
        monkeypatch.setattr(runtime, "S3Handler", lambda *a: handler)
        monkeypatch.setattr(runtime, "publish_sns", lambda *a, **kw: True)
        result = JobResult(kind="guide", input="u", job_id="job-1")
        assert runtime.record_unstarted_run(result) is True
        assert sessions == [{"region_name": "us-east-1"}]
        assert handler.put_text.call_args.args[0] == "p/results/job-1.json"

    def test_unsent_notification_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.domain.jobs import JobResult

        monkeypatch.setattr(runtime, "is_running_in_aws", lambda: True)
        monkeypatch.setenv("TOPIC_ARN", "arn:aws:sns:us-east-1:1:t")
        monkeypatch.setattr(runtime, "aws_session", lambda **kw: MagicMock())
        monkeypatch.setattr(runtime, "publish_sns", lambda *a, **kw: False)
        assert runtime.record_unstarted_run(JobResult(kind=None, input="x")) is False

    def test_manifest_is_written_without_a_topic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.domain.jobs import JobResult

        monkeypatch.setattr(runtime, "is_running_in_aws", lambda: True)
        monkeypatch.delenv("TOPIC_ARN", raising=False)
        config = MagicMock()
        config.resources.s3_bucket_name = "b"
        config.resources.s3_prefix = "p"
        monkeypatch.setattr(runtime.Config, "load", classmethod(lambda cls: config))
        handler = MagicMock()
        monkeypatch.setattr(runtime, "S3Handler", lambda *a: handler)
        result = JobResult(kind="guide", input="u", job_id="job-2")
        assert runtime.record_unstarted_run(result) is True
        assert handler.put_text.call_args.args[0] == "p/results/job-2.json"

    @pytest.mark.parametrize("in_aws,expected", [(False, True), (True, False)])
    def test_broken_aws_profile_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch, in_aws: bool, expected: bool
    ) -> None:
        # The same missing profile that stopped the run must not crash the
        # report (the CLI still prints one JSON object).
        from botocore.exceptions import ProfileNotFound

        from scholar_lens.domain.jobs import JobResult

        def broken(**_: object) -> None:
            raise ProfileNotFound(profile="missing")

        monkeypatch.setattr(runtime, "is_running_in_aws", lambda: in_aws)
        monkeypatch.setenv("TOPIC_ARN", "arn:aws:sns:us-east-1:1:t")
        monkeypatch.setattr(runtime, "aws_session", broken)
        result = JobResult(kind=None, input="x", job_id="job-3")
        assert runtime.record_unstarted_run(result) is expected

    def test_nothing_to_send_outside_aws(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scholar_lens.domain.jobs import JobResult

        monkeypatch.setattr(runtime, "is_running_in_aws", lambda: False)
        assert runtime.record_unstarted_run(JobResult(kind=None, input="x")) is True


class TestPublishArtifact:
    def test_pr_failure_is_kept_next_to_the_saved_artifact(
        self, tmp_path: Path
    ) -> None:
        import asyncio

        from scholar_lens.domain.publishing import PublishError

        publisher = MagicMock()

        async def publish(request):  # type: ignore[no-untyped-def]
            return "s3://b/post.md", tmp_path / "post.md"

        async def create_pull_request(request, path):  # type: ignore[no-untyped-def]
            raise PublishError("403 from GitHub")

        publisher.publish = publish
        publisher.create_pull_request = create_pull_request
        context = MagicMock(publisher_factory=lambda ctx: publisher)
        context.config.resources.github.enabled = True
        request = MagicMock(link_sources=None)
        request.title = "LoRA"
        outcome = asyncio.run(
            app_run.publish_artifact(context, request, create_pr=True)
        )
        assert outcome.s3_uri == "s3://b/post.md" and outcome.pr_url is None
        assert outcome.markdown_path == tmp_path / "post.md"
        assert outcome.pr_error == "403 from GitHub"
        assert outcome.warnings == []


def test_an_unreadable_branch_history_becomes_a_review_note() -> None:
    import asyncio

    from scholar_lens.domain.publishing import PublishError

    publisher = MagicMock()
    publisher.find_published.side_effect = PublishError("502 from GitHub")
    published, notes = asyncio.run(
        app_run.find_published_post(
            MagicMock(blog_listing_error=None, publisher_factory=lambda c: publisher),
            slug="x",
            branch_id="guide-abc",
            category="Tech Guides",
        )
    )
    assert published is None
    assert len(notes) == 1 and "502 from GitHub" in notes[0]
    publisher.find_published.assert_called_once_with("x", "guide-abc", "Tech Guides")


def test_a_failed_blog_listing_becomes_a_review_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    def fail(*args: object) -> list[object]:
        raise RuntimeError("403 rate limit")

    monkeypatch.setattr(bootstrap, "fetch_blog_posts", fail)
    config = MagicMock()
    config.resources.github.enabled = True
    context = _run_context(
        config=config,
        blog_posts=None,
        blog_listing_error=None,
        publisher_factory=lambda ctx: _publisher(ctx.require_blog_posts()),
    )
    published, notes = asyncio.run(
        app_run.find_published_post(
            context, slug="x", branch_id="b", category="Tech Guides"
        )
    )
    assert published is None and context.blog_posts == []
    assert len(notes) == 1 and "403 rate limit" in notes[0]


def _publisher(site_posts: list[object]) -> MagicMock:
    publisher = MagicMock()
    publisher.find_published.return_value = None
    return publisher


def test_a_listed_blog_adds_no_lookup_note() -> None:
    import asyncio

    context = MagicMock(
        blog_listing_error=None, publisher_factory=lambda c: _publisher([])
    )
    _, notes = asyncio.run(
        app_run.find_published_post(
            context,
            slug="x",
            branch_id="b",
            category="Tech Guides",
        )
    )
    assert notes == []


def test_no_pr_run_reports_the_post_checklist(tmp_path: Path) -> None:
    import asyncio

    publisher = MagicMock()

    async def publish(request):  # type: ignore[no-untyped-def]
        request.validation_warnings = ["Summary is still long", "numbered reference"]
        return "s3://b/post.md", tmp_path / "post.md"

    publisher.publish = publish
    request = MagicMock(link_sources=None)
    context = MagicMock(publisher_factory=lambda ctx: publisher)
    outcome = asyncio.run(app_run.publish_artifact(context, request, create_pr=False))
    assert outcome.warnings == ["Summary is still long", "numbered reference"]


@pytest.mark.parametrize("kind", ["validation", "asset"])
def test_generator_notes_survive_a_failed_publish(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from scholar_lens.adapters.github.publisher import PostAssetError
    from scholar_lens.domain.jobs import JobRequest
    from scholar_lens.domain.validation import PostValidationError, Violation

    publisher = MagicMock()

    async def publish(request):  # type: ignore[no-untyped-def]
        error: PostValidationError | PostAssetError
        if kind == "validation":
            error = PostValidationError([Violation("error", "subtitle", 1, "missing")])
        else:
            error = PostAssetError("figure missing")
        error.rejected_path = "/work/post.rejected.md"
        error.rejected_s3_uri = "s3://b/rejected/post.rejected.md"
        raise error

    publisher.publish = publish

    async def pipeline(context, request, tracker):  # type: ignore[no-untyped-def]
        context.publisher_factory = lambda ctx: publisher
        publish_request = MagicMock(link_sources=None)
        publish_request.review_notes = ["Review section 'Method' produced no text"]
        return await app_run.publish_artifact(context, publish_request, create_pr=False)

    TestRunJob()._patch(monkeypatch, pipeline)
    result = runtime.run_job(JobRequest(kind="review", source="2106.09685"))
    assert result.status == "failed"
    assert result.warnings == ["Review section 'Method' produced no text"]
    assert result.blocking == []
    # The rejected draft is where the result points.
    assert result.markdown_path == "/work/post.rejected.md"
    assert result.s3_uri == "s3://b/rejected/post.rejected.md"


class TestBlogTagFilter:
    def test_only_reusable_spellings_are_offered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.adapters.conventions_file import load_local_conventions
        from scholar_lens.domain.conventions import BlogPost

        spec = load_local_conventions()
        controlled = spec.tags.controlled[0]
        retired = next(iter(spec.tags.aliases))
        posts = [
            BlogPost(path="_posts/a.md", tags=[controlled, retired, "LoRA", "a/b"]),
            BlogPost(path="_posts/b.md", tags=["LoRA", "Paged Attention", "a/b"]),
            BlogPost(path="_posts/c.md", tags=["vLLM", "Paged Attention"]),
        ]
        monkeypatch.setattr(bootstrap, "load_run_blog_posts", lambda config: posts)
        context = _run_context(
            config=MagicMock(), conventions=spec, blog_posts=None, blog_tags=None
        )
        # Controlled and retired tags, and spellings a post may not use, are out.
        assert context.require_blog_tags() == ["LoRA", "vLLM"]
        # Only tags shared by two or more posts are offered as hints.
        assert context.blog_tag_hints(10) == ["LoRA"]


class TestRunCoroutine:
    def test_returns_the_result(self) -> None:
        async def answer() -> int:
            return 42

        assert runtime.run_coroutine(answer()) == 42

    def test_interrupt_does_not_wait_for_worker_threads(self) -> None:
        import asyncio
        import threading
        import time

        release = threading.Event()

        async def pipeline() -> None:
            async def interrupt() -> None:
                await asyncio.sleep(0.05)
                raise SystemExit(143)  # what the SIGTERM handler raises

            await asyncio.gather(asyncio.to_thread(release.wait, 5), interrupt())

        started = time.monotonic()
        try:
            with pytest.raises(SystemExit):
                runtime.run_coroutine(pipeline())
            # asyncio.run would block here until the thread finished.
            assert time.monotonic() - started < 2
        finally:
            release.set()


class TestConventionsChangedDuringTheRun:
    """The blog's conventions are re-read before publishing."""

    TOPIC = "Language-Models"
    NEW_COVER = "/assets/images/language-models-v2.jpg"

    @staticmethod
    def _markdown(cover: str) -> str:
        from scholar_lens.domain.post_format import render_front_matter

        return (
            render_front_matter(
                title="LoRA",
                date=datetime(2026, 10, 1, tzinfo=UTC),
                categories=["Tech Guides", "Language-Models"],
                tags=["LoRA"],
                cover=cover,
                use_math=False,
            )
            + "Body.\n"
        )

    def _run(
        self, monkeypatch: pytest.MonkeyPatch, fresh: object
    ) -> tuple[object, object, object]:
        import asyncio

        from scholar_lens.adapters.conventions_file import load_local_conventions
        from scholar_lens.domain.publishing import PublishRequest

        started = load_local_conventions()

        def load(config: object, *, fallback: bool = True) -> object:
            assert fallback is False
            if isinstance(fresh, Exception):
                raise fresh
            return fresh if fresh is not None else started

        monkeypatch.setattr(bootstrap, "load_run_conventions", load)
        validated_with: list[object] = []
        publisher = MagicMock()

        async def publish(request):  # type: ignore[no-untyped-def]
            request.validation_warnings = list(request.review_notes)
            return None, MagicMock()

        publisher.publish = publish

        def build(context):  # type: ignore[no-untyped-def]
            validated_with.append(context.require_conventions())
            return publisher

        context = _run_context(
            config=MagicMock(), conventions=started, publisher_factory=build
        )
        request = PublishRequest(
            title="LoRA",
            markdown=self._markdown(started.topics[self.TOPIC]),
            work_dir=MagicMock(),
            branch_id="guide-x",
            pr_title="t",
            pr_body="b",
            commit_message="c",
            kind="guide",
        )
        outcome = asyncio.run(
            app_run.publish_artifact(context, request, create_pr=False)
        )
        return request, validated_with[0], outcome

    def test_a_changed_cover_is_recomputed_and_noted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.adapters.conventions_file import load_local_conventions

        fresh = load_local_conventions()
        fresh.topics[self.TOPIC] = self.NEW_COVER
        request, spec, outcome = self._run(monkeypatch, fresh)
        assert spec == fresh
        assert request.markdown == self._markdown(self.NEW_COVER)  # type: ignore[attr-defined]
        assert any(
            "conventions changed during the run" in w and self.NEW_COVER in w
            for w in outcome.warnings  # type: ignore[attr-defined]
        )

    def test_a_change_elsewhere_is_noted_without_touching_the_post(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.adapters.conventions_file import load_local_conventions

        fresh = load_local_conventions()
        fresh.tags.controlled = fresh.tags.controlled[:-1]
        request, spec, outcome = self._run(monkeypatch, fresh)
        assert spec == fresh
        cover = fresh.topics[self.TOPIC]
        assert request.markdown == self._markdown(cover)  # type: ignore[attr-defined]
        notes = [w for w in outcome.warnings if "conventions changed" in w]  # type: ignore[attr-defined]
        assert len(notes) == 1 and "cover" not in notes[0]

    @pytest.mark.parametrize(
        "fresh", [None, RuntimeError("502")], ids=["unchanged", "unreachable"]
    )
    def test_unchanged_or_unreadable_conventions_add_no_note(
        self, monkeypatch: pytest.MonkeyPatch, fresh: object
    ) -> None:
        from scholar_lens.adapters.conventions_file import load_local_conventions

        request, spec, outcome = self._run(monkeypatch, fresh)
        assert spec == load_local_conventions()
        assert outcome.warnings == []  # type: ignore[attr-defined]


def test_with_spec_cover_leaves_an_unknown_type_to_the_validator() -> None:
    from scholar_lens.adapters.conventions_file import load_local_conventions
    from scholar_lens.domain.post_format import with_spec_cover

    spec = load_local_conventions()
    markdown = TestConventionsChangedDuringTheRun._markdown("/x.jpg").replace(
        "Tech Guides", "Unknown Type"
    )
    assert with_spec_cover(markdown, spec) == (markdown, None)
    assert with_spec_cover("No front matter.", spec) == ("No front matter.", None)


def test_a_pr_run_without_a_token_fails_before_any_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from scholar_lens.domain.jobs import JobRequest
    from scholar_lens.entrypoints.runtime import check_pr_ready

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    context = MagicMock()
    context.config.resources.github.enabled = True
    with pytest.raises(ValueError, match="GITHUB_TOKEN"):
        check_pr_ready(context, JobRequest(kind="summarize", source="2106.09685"))
    check_pr_ready(
        context, JobRequest(kind="summarize", source="2106.09685", create_pr=False)
    )
    context.config.resources.github.enabled = False
    check_pr_ready(context, JobRequest(kind="summarize", source="2106.09685"))


def test_a_request_overrides_the_engine_and_cost_ceiling_for_its_run() -> None:
    from scholar_lens.configs import Config
    from scholar_lens.domain.jobs import JobRequest

    config = Config()
    review = runtime.with_overrides(
        config,
        JobRequest(
            kind="review", source="2106.09685", engine="evidence", max_cost_usd=5
        ),
    )
    assert review.explanation.engine == "evidence"
    assert review.explanation.max_cost_usd == 5
    assert config.explanation.engine == "sections"  # the loaded config is untouched
    survey = runtime.with_overrides(
        config, JobRequest(kind="survey_write", plan="p.json", max_cost_usd=20)
    )
    assert survey.survey.write_max_cost_usd == 20
    assert (
        runtime.with_overrides(
            config, JobRequest(kind="guide", urls=["https://d.dev/"])
        )
        is config
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "guide", "urls": ["https://d.dev/"], "engine": "evidence"},
        {"kind": "survey_plan", "topic": "t", "max_cost_usd": 3},
    ],
)
def test_overrides_a_kind_cannot_take_are_rejected(fields: dict) -> None:
    from pydantic import ValidationError

    from scholar_lens.domain.jobs import JobRequest

    with pytest.raises(ValidationError):
        JobRequest(**fields)
