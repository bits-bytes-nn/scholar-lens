"""Slack Events API worker (AWS Lambda, container image).

The receiver (``lambda_receiver.py``) async-invokes this function with the inner
Slack event once the request is verified. Freed from Slack's 3-second deadline,
the worker does the expensive part: parse the intent (a Bedrock LLM call),
dispatch the AWS Batch job, and post the acknowledgement back into the
originating thread. It reuses the existing :class:`PaperBot` machinery verbatim —
this module is just the Lambda entrypoint that wires it up.

The result of the Batch job itself is posted later by the pipeline via
``notifier.post_slack_result``; this worker only handles the inbound ack.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from ..configs import Config
from ..src.constants import EnvVars, SSMParams
from ..src.logger import logger
from ..src.runtime import build_context, load_secrets_from_ssm
from .bot import PaperBot, build_bot, is_user_authorized
from .dispatcher import SlackContext


def _post_message(channel: str, thread_ts: str | None, reply: Any) -> None:
    """Best-effort post of a Block Kit reply (mirrors notifier's lazy import)."""
    token = os.getenv(EnvVars.SLACK_BOT_TOKEN.value)
    if not token:
        logger.warning("No Slack bot token; cannot post ack to channel '%s'.", channel)
        return
    try:
        from slack_sdk import WebClient

        WebClient(token=token).chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=reply.text,
            blocks=reply.blocks,
        )
    except Exception as e:  # noqa: BLE001 - a Slack failure must not crash the worker
        logger.warning("Failed to post ack to Slack: %s", e)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda handler. ``event`` is ``{"slack_event": <inner Slack event>}``."""
    inner = event.get("slack_event") or {}
    channel = inner.get("channel", "")
    thread_ts = inner.get("thread_ts") or inner.get("ts")
    user = inner.get("user")
    text = inner.get("text", "")

    # The whole flow — secret load, bot build, intent parse, dispatch — is wrapped
    # so ANY failure still surfaces in the thread. The receiver suppresses Slack
    # retries and this Lambda has retry_attempts=0, so a silent crash here would
    # leave the user with no response at all (which is exactly what an early
    # import/SSM/Bedrock error would otherwise cause).
    try:
        # Load the Slack bot token from SSM into the environment (AWS-only no-op
        # guard lives inside load_secrets_from_ssm; Lambda counts as "in AWS").
        run_context = build_context(Config.load())
        load_secrets_from_ssm(
            run_context, {SSMParams.SLACK_BOT_TOKEN: EnvVars.SLACK_BOT_TOKEN}
        )

        if not is_user_authorized(user):
            logger.warning("Unauthorized Slack user '%s' attempted a job.", user)
            if channel:
                _post_message(channel, thread_ts, PaperBot._unauthorized_reply())
            return {"ok": True}

        bot = build_bot(run_context.config)
        slack_context = SlackContext(channel=channel, thread_ts=thread_ts, user=user)
        reply = asyncio.run(bot.handle_message(text, slack_context=slack_context))
    except Exception as e:  # noqa: BLE001 - report failure to the thread, don't crash
        logger.error("Worker failed to handle message: %s", e, exc_info=True)
        if channel:
            _post_message(channel, thread_ts, PaperBot._dispatch_error_reply(e))
        return {"ok": False}

    if channel:
        _post_message(channel, thread_ts, reply)
    return {"ok": True}
