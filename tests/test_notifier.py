"""Tests for the best-effort Slack result notifier.

``post_slack_result`` must never raise: it degrades to a no-op when there is no
channel context or token, and swallows any Slack delivery error. No network.
"""

from __future__ import annotations

import builtins

import pytest

from scholar_lens.slack import notifier
from scholar_lens.slack.blocks import mrkdwn_safe as _mrkdwn_safe
from scholar_lens.slack.blocks import one_line as _short
from scholar_lens.slack.notifier import (
    _build_result_message,
    _clean,
    post_slack_result,
)
from scholar_lens.src.constants import EnvVars


def test_mrkdwn_safe_neutralizes_formatting() -> None:
    out = _mrkdwn_safe("fail *boom* _x_ `code`\nline2")
    assert "*boom*" not in out  # asterisks swapped for a homoglyph
    assert "\\" not in out  # no backslash escaping (Slack renders it literally)
    assert "`" not in out  # backticks replaced
    assert "\n" not in out  # newlines collapsed


def test_mrkdwn_safe_keeps_arxiv_id_clean() -> None:
    # A mid-token underscore must NOT be escaped — it shows literally on Slack.
    assert _mrkdwn_safe("2504_03182") == "2504_03182"


class TestClean:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("null", None),
            ("NULL", None),
            ("", None),
            (None, None),
            # A whitespace-only string is truthy and not the NULL sentinel, so it
            # falls through to ``value.strip()`` -> "" (not None). Documented quirk.
            ("   ", ""),
            ("  C123 ", "C123"),
            ("C123", "C123"),
        ],
    )
    def test_clean(self, value: str | None, expected: str | None) -> None:
        assert _clean(value) == expected


class TestPostSlackResult:
    def test_no_channel_is_noop(self) -> None:
        # Must not raise even with no channel.
        post_slack_result(
            channel=None,
            thread_ts=None,
            success=True,
            artifact_label="review",
            title="Some Paper",
            s3_url=None,
        )

    def test_null_channel_is_noop(self) -> None:
        post_slack_result(
            channel="null",
            thread_ts=None,
            success=True,
            artifact_label="review",
            title="Some Paper",
            s3_url="s3://x",
        )

    def test_no_token_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(EnvVars.SLACK_BOT_TOKEN.value, raising=False)
        post_slack_result(
            channel="C123",
            thread_ts="123.45",
            success=False,
            artifact_label="summary",
            title="Some Paper",
            s3_url=None,
            error="bad",
            bot_token=None,
        )

    def test_slack_import_failure_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the lazy ``from slack_sdk import WebClient`` to fail.
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "slack_sdk":
                raise ImportError("no slack_sdk")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        post_slack_result(
            channel="C123",
            thread_ts="123.45",
            success=True,
            artifact_label="review",
            title="Some Paper",
            s3_url="s3://bucket/key",
            bot_token="xoxb-fake",
        )  # no raise

    def test_webclient_raises_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Inject a fake ``slack_sdk`` whose WebClient.chat_postMessage raises.
        import sys
        import types

        fake_module = types.ModuleType("slack_sdk")

        class _FakeWebClient:
            def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                pass

            def chat_postMessage(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("slack api error")

        fake_module.WebClient = _FakeWebClient
        monkeypatch.setitem(sys.modules, "slack_sdk", fake_module)
        post_slack_result(
            channel="C123",
            thread_ts="123.45",
            success=True,
            artifact_label="review",
            title="Some Paper",
            s3_url="s3://bucket/key",
            bot_token="xoxb-fake",
        )  # no raise

    def test_success_post_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types

        calls: list[dict] = []
        fake_module = types.ModuleType("slack_sdk")

        class _FakeWebClient:
            def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                pass

            def chat_postMessage(self, **kwargs):  # type: ignore[no-untyped-def]
                calls.append(kwargs)

        fake_module.WebClient = _FakeWebClient
        monkeypatch.setitem(sys.modules, "slack_sdk", fake_module)
        monkeypatch.setenv(EnvVars.SLACK_BOT_TOKEN.value, "xoxb-env")
        post_slack_result(
            channel="C123",
            thread_ts="null",
            success=True,
            artifact_label="review",
            title="Some Paper",
            s3_url="s3://bucket/key",
        )
        assert len(calls) == 1
        assert calls[0]["channel"] == "C123"
        # "null" thread_ts is cleaned to None.
        assert calls[0]["thread_ts"] is None
        assert "ready" in calls[0]["text"].lower()
        # Rich Block Kit message: a header block + the title section.
        blocks = calls[0]["blocks"]
        assert blocks[0]["type"] == "header"
        assert any(
            b["type"] == "section" and "Some Paper" in b["text"]["text"] for b in blocks
        )
        # An s3:// URI (not http) is shown as a code section, not a button.
        assert not any(b["type"] == "actions" for b in blocks)


class TestBuildResultMessage:
    def test_failure_renders_error_in_code_block(self) -> None:
        text, blocks = _build_result_message(
            success=False,
            artifact_label="review",
            title="Some Paper",
            s3_url=None,
            error="AccessDeniedException: bedrock:InvokeModel not allowed",
        )
        assert "failed" in text.lower()
        assert blocks[0]["type"] == "header"
        # Error is shown in its own section as a fenced code block, plus a hint.
        err_section = next(
            b for b in blocks if b["type"] == "section" and "```" in b["text"]["text"]
        )
        assert "AccessDeniedException" in err_section["text"]["text"]
        assert any(b["type"] == "context" for b in blocks)

    def test_https_url_renders_button(self) -> None:
        _, blocks = _build_result_message(
            success=True,
            artifact_label="summary",
            title="Some Paper",
            s3_url="https://blog.example.com/post",
            error=None,
        )
        actions = next(b for b in blocks if b["type"] == "actions")
        assert actions["elements"][0]["url"] == "https://blog.example.com/post"

    def test_short_collapses_and_caps(self) -> None:
        assert _short("a\n\n  b\tc") == "a b c"
        assert _short("x" * 1000).endswith("…")
        assert "`" not in _short("danger ``` fence")


def test_notifier_module_importable() -> None:
    assert hasattr(notifier, "post_slack_result")
