"""Tests for PaperBot.handle_message (intent parser + dispatcher mocked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from scholar_lens.slack.bot import PaperBot
from scholar_lens.slack.dispatcher import DispatchResult
from scholar_lens.slack.intent import ParsedIntent, SlackIntent


def _bot(parsed: ParsedIntent, *, dispatch=None) -> PaperBot:
    parser = MagicMock()
    parser.parse = AsyncMock(return_value=parsed)
    dispatcher = MagicMock()
    if dispatch is not None:
        dispatcher.dispatch = dispatch
    return PaperBot(MagicMock(), parser, dispatcher)


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
        assert ":rocket:" in reply
        assert "review" in reply
        assert "2401.06066" in reply

    async def test_unknown_returns_help(self) -> None:
        parsed = ParsedIntent(
            intent=SlackIntent.UNKNOWN, sources=[], reason="just chatting"
        )
        bot = _bot(parsed)
        reply = await bot.handle_message("hello there")
        assert "couldn't turn that into an action" in reply
        assert "just chatting" in reply
        bot.dispatcher.dispatch.assert_not_called()

    async def test_dispatch_failure_is_reported(self) -> None:
        parsed = ParsedIntent(intent=SlackIntent.GUIDE, sources=["https://docs.x.io"])
        dispatch = MagicMock(side_effect=RuntimeError("batch down"))
        bot = _bot(parsed, dispatch=dispatch)
        reply = await bot.handle_message("guide https://docs.x.io")
        assert ":warning:" in reply
        assert "batch down" in reply
