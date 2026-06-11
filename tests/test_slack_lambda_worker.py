"""Tests for the Slack worker Lambda (intent parse + dispatch, mocked).

The worker reuses PaperBot; here build_bot/build_context/secret-loading and the
Slack post are patched so nothing touches AWS or Slack.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from scholar_lens.slack import lambda_worker
from scholar_lens.slack.bot import PaperBot, SlackReply


@pytest.fixture(autouse=True)
def _patch_aws(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise config load + SSM secret load (no AWS)."""
    monkeypatch.setattr(lambda_worker, "build_context", lambda config: MagicMock())
    monkeypatch.setattr(
        lambda_worker.Config, "load", classmethod(lambda cls: MagicMock())
    )
    monkeypatch.setattr(lambda_worker, "load_secrets_from_ssm", lambda ctx, m: None)
    monkeypatch.delenv("SLACK_ALLOWED_USER_IDS", raising=False)


def _patch_bot(monkeypatch: pytest.MonkeyPatch, reply: SlackReply) -> MagicMock:
    bot = MagicMock()
    bot.handle_message = AsyncMock(return_value=reply)
    monkeypatch.setattr(lambda_worker, "build_bot", lambda config: bot)
    return bot


def _capture_posts(monkeypatch: pytest.MonkeyPatch) -> list:
    posted: list = []
    monkeypatch.setattr(
        lambda_worker,
        "_post_message",
        lambda channel, thread_ts, reply: posted.append((channel, thread_ts, reply)),
    )
    return posted


def _event(**overrides) -> dict:  # type: ignore[no-untyped-def]
    inner = {
        "type": "app_mention",
        "channel": "C1",
        "ts": "111.222",
        "user": "U1",
        "text": "review 2401.06066",
    }
    inner.update(overrides)
    return {"slack_event": inner}


class TestWorkerHandler:
    def test_actionable_posts_ack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reply = SlackReply(text="Review started", blocks=[{"x": 1}])
        bot = _patch_bot(monkeypatch, reply)
        posted = _capture_posts(monkeypatch)
        result = lambda_worker.handler(_event(), None)
        assert result == {"ok": True}
        bot.handle_message.assert_awaited_once()
        assert posted == [("C1", "111.222", reply)]

    def test_uses_thread_ts_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_bot(monkeypatch, SlackReply(text="ok"))
        posted = _capture_posts(monkeypatch)
        lambda_worker.handler(_event(thread_ts="999.888"), None)
        assert posted[0][1] == "999.888"

    def test_unauthorized_user_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLACK_ALLOWED_USER_IDS", "U_ALLOWED")
        bot = _patch_bot(monkeypatch, SlackReply(text="ignored"))
        posted = _capture_posts(monkeypatch)
        result = lambda_worker.handler(_event(user="U_OTHER"), None)
        assert result == {"ok": True}
        bot.handle_message.assert_not_called()
        # A refusal is posted to the thread; the dispatch never runs.
        assert len(posted) == 1
        assert posted[0][0] == "C1"

    def test_handle_failure_reports_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bot = MagicMock()
        bot.handle_message = AsyncMock(side_effect=RuntimeError("boom"))
        monkeypatch.setattr(lambda_worker, "build_bot", lambda config: bot)
        posted = _capture_posts(monkeypatch)
        result = lambda_worker.handler(_event(), None)
        assert result == {"ok": False}
        assert len(posted) == 1  # an error message was posted


class TestPostMessage:
    def test_no_token_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        # Should not raise even though no WebClient is constructed.
        lambda_worker._post_message("C1", "1.2", SlackReply(text="x"))

    def test_post_failure_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-x")
        import slack_sdk

        boom = MagicMock(side_effect=RuntimeError("slack down"))
        monkeypatch.setattr(slack_sdk, "WebClient", boom)
        # Must not raise — a Slack failure can't crash the worker.
        lambda_worker._post_message("C1", "1.2", SlackReply(text="x", blocks=[]))

    def test_unauthorized_reply_is_postable(self) -> None:
        # The refusal reply object the worker posts must be a real SlackReply.
        reply = PaperBot._unauthorized_reply()
        assert reply.text
        assert reply.blocks
