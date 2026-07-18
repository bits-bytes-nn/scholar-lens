"""Tests for PaperBot.handle_message (intent parser + dispatcher mocked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from scholar_lens.slack.bot import (
    PaperBot,
    SlackAppMismatchError,
    _app_id_from_app_token,
    _SeenEvents,
    is_user_authorized,
    verify_slack_app_identity,
)
from scholar_lens.slack.dispatcher import DispatchResult
from scholar_lens.slack.intent import ParsedIntent, SlackIntent


def _bot(parsed: ParsedIntent, *, dispatch=None) -> PaperBot:
    parser = MagicMock()
    parser.parse = AsyncMock(return_value=parsed)
    dispatcher = MagicMock()
    if dispatch is not None:
        dispatcher.dispatch = dispatch
    return PaperBot(MagicMock(), parser, dispatcher)


def _blocks_text(reply) -> str:  # type: ignore[no-untyped-def]
    """Flatten all text out of a SlackReply's Block Kit blocks for assertions."""
    parts: list[str] = [reply.text]
    for block in reply.blocks:
        if "text" in block and isinstance(block["text"], dict):
            parts.append(block["text"]["text"])
        for el in block.get("elements", []):
            if isinstance(el, dict) and isinstance(el.get("text"), str):
                parts.append(el["text"])
    return "\n".join(parts)


class TestHandleMessage:
    async def test_actionable_dispatches_and_confirms(self) -> None:
        parsed = ParsedIntent(intent=SlackIntent.REVIEW, sources=["2401.06066"])
        dispatch = MagicMock(
            return_value=DispatchResult(
                job_id="j1",
                job_name="scholar-lens-dev-review-x",
                intent=SlackIntent.REVIEW,
            )
        )
        bot = _bot(parsed, dispatch=dispatch)
        reply = await bot.handle_message("review 2401.06066")
        dispatch.assert_called_once()
        assert reply.blocks  # rich Block Kit, not just text
        text = _blocks_text(reply)
        assert "Review" in text
        assert "started" in text.lower()
        assert "2401.06066" in text

    async def test_ack_shows_repo_and_parse_pdf(self) -> None:
        parsed = ParsedIntent(
            intent=SlackIntent.SUMMARIZE,
            sources=["2401.06066"],
            repo_urls=["https://github.com/x/y"],
            parse_pdf=True,
        )
        dispatch = MagicMock(
            return_value=DispatchResult(
                job_id="j1", job_name="n", intent=SlackIntent.SUMMARIZE
            )
        )
        reply = await _bot(parsed, dispatch=dispatch).handle_message("...")
        text = _blocks_text(reply)
        assert "github.com/x/y" in text
        assert "PDF parsing on" in text

    async def test_ack_echoes_only_dispatched_sources(self) -> None:
        # Regression (ui-1): a multi-source review parses 2 papers but dispatches
        # only the first. The ack must echo exactly what was dispatched, not
        # promise the dropped second paper.
        parsed = ParsedIntent(
            intent=SlackIntent.REVIEW, sources=["2401.06066", "2405.04434"]
        )
        dispatch = MagicMock(
            return_value=DispatchResult(
                job_id="j1",
                job_name="n",
                intent=SlackIntent.REVIEW,
                dispatched_sources=["2401.06066"],
            )
        )
        reply = await _bot(parsed, dispatch=dispatch).handle_message("...")
        text = _blocks_text(reply)
        assert "2401.06066" in text
        assert "2405.04434" not in text

    async def test_unknown_returns_help(self) -> None:
        parsed = ParsedIntent(
            intent=SlackIntent.UNKNOWN, sources=[], reason="just chatting"
        )
        bot = _bot(parsed)
        reply = await bot.handle_message("hello there")
        text = _blocks_text(reply)
        assert "couldn't tell what you'd like" in text
        assert "just chatting" in text
        bot.dispatcher.dispatch.assert_not_called()

    async def test_dispatch_failure_is_reported(self) -> None:
        parsed = ParsedIntent(intent=SlackIntent.GUIDE, sources=["https://docs.x.io"])
        dispatch = MagicMock(side_effect=RuntimeError("batch down"))
        bot = _bot(parsed, dispatch=dispatch)
        reply = await bot.handle_message("guide https://docs.x.io")
        text = _blocks_text(reply)
        assert "Couldn't start the job" in text
        assert "batch down" in text


class TestSlackAppIdentityGuard:
    def test_app_id_parsed_from_token(self) -> None:
        assert _app_id_from_app_token("xapp-1-A08KWUTLR9S-123-abc") == "A08KWUTLR9S"
        assert _app_id_from_app_token("not-a-token") is None
        assert _app_id_from_app_token("") is None

    def test_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Token belongs to OmniSummary's app, but we expect Paper Bot's app.
        monkeypatch.setenv("SLACK_EXPECTED_APP_ID", "A_PAPER_BOT")
        with pytest.raises(SlackAppMismatchError):
            verify_slack_app_identity("xoxb-x", "xapp-1-A08KWUTLR9S-1-2")

    def test_match_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLACK_EXPECTED_APP_ID", "A08KWUTLR9S")
        verify_slack_app_identity("xoxb-x", "xapp-1-A08KWUTLR9S-1-2")  # no raise

    def test_no_expected_id_skips_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLACK_EXPECTED_APP_ID", raising=False)
        # Even a clearly-foreign token is allowed when the guard is unset.
        verify_slack_app_identity("xoxb-x", "xapp-1-AFOREIGN-1-2")  # no raise

    def test_unparseable_token_skips_when_expected_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLACK_EXPECTED_APP_ID", "A_PAPER_BOT")
        verify_slack_app_identity("xoxb-x", "garbage")  # degrades gracefully


class TestSlackAuthorization:
    def test_open_mode_allows_anyone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SLACK_ALLOWED_USER_IDS", raising=False)
        assert is_user_authorized("U_ANYONE") is True
        assert is_user_authorized(None) is True

    def test_allowlist_permits_listed_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLACK_ALLOWED_USER_IDS", "U_A, U_B")
        assert is_user_authorized("U_A") is True
        assert is_user_authorized("U_B") is True

    def test_allowlist_refuses_others(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SLACK_ALLOWED_USER_IDS", "U_A")
        assert is_user_authorized("U_OTHER") is False
        assert is_user_authorized(None) is False


class TestSeenEvents:
    def test_dedups_repeated_keys(self) -> None:
        seen = _SeenEvents()
        assert seen.seen("evt1") is False  # first time
        assert seen.seen("evt1") is True  # duplicate
        assert seen.seen("evt2") is False

    def test_empty_key_never_dedups(self) -> None:
        seen = _SeenEvents()
        assert seen.seen(None) is False
        assert seen.seen("") is False

    def test_evicts_oldest_beyond_capacity(self) -> None:
        seen = _SeenEvents(capacity=2)
        seen.seen("a")
        seen.seen("b")
        seen.seen("c")  # evicts "a"
        assert seen.seen("a") is False  # "a" was evicted, treated as new
