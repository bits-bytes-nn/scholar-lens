"""Post job results back to the originating Slack channel/thread.

The pipeline (running in AWS Batch) calls :func:`post_slack_result` at the end
of a run so the user who triggered it from Slack gets the outcome in the same
thread — closing the loop the bot promises ("I'll post the result here"). Slack
context (channel / thread / user) is threaded in as Batch parameters; when it is
absent (e.g. a CLI run) this is a no-op.

``slack_sdk`` ships with ``slack_bolt``; we import it lazily so non-Slack runs
and the test suite never require it.
"""

from __future__ import annotations

import os

from ..src.constants import EnvVars
from ..src.logger import logger

_NULL = "null"


def _clean(value: str | None) -> str | None:
    """Treat the Batch NULL sentinel and blanks as absent."""
    if not value or value.strip().lower() == _NULL:
        return None
    return value.strip()


def _mrkdwn_safe(value: str) -> str:
    """Neutralise Slack mrkdwn control chars in user-influenced text.

    The title and error strings originate from paper metadata / exception text,
    so escape the formatting characters (`*_~` and backtick) and collapse
    newlines to keep an injected string from reshaping the status message.
    """
    out = value.replace("`", "ʼ")
    for ch in ("*", "_", "~"):
        out = out.replace(ch, "\\" + ch)
    return " ".join(out.split())


def post_slack_result(
    *,
    channel: str | None,
    thread_ts: str | None,
    success: bool,
    artifact_label: str,
    title: str,
    s3_url: str | None,
    error: str | None = None,
    bot_token: str | None = None,
) -> None:
    """Best-effort post of a completion message to Slack.

    No-ops (with a debug log) when there is no Slack channel context or no bot
    token, and never raises — a Slack failure must not fail the pipeline.
    """
    channel = _clean(channel)
    if not channel:
        logger.debug("No Slack channel context; skipping Slack result post.")
        return
    token = bot_token or os.getenv(EnvVars.SLACK_BOT_TOKEN.value)
    if not token:
        logger.info("No Slack bot token; cannot post result to channel '%s'.", channel)
        return

    safe_title = _mrkdwn_safe(title)
    if success:
        text = (
            f":white_check_mark: *{artifact_label.capitalize()}* ready for "
            f"*{safe_title}*."
        )
        if s3_url:
            text += f"\n• Output: `{_mrkdwn_safe(s3_url)}`"
    else:
        text = f":x: *{artifact_label.capitalize()}* failed for *{safe_title}*."
        if error:
            text += f"\n• Error: {_mrkdwn_safe(error)}"

    try:
        from slack_sdk import WebClient

        WebClient(token=token).chat_postMessage(
            channel=channel,
            thread_ts=_clean(thread_ts),
            text=text,
        )
        logger.info("Posted %s result to Slack channel '%s'.", artifact_label, channel)
    except Exception as e:  # noqa: BLE001 - Slack delivery must never fail the job
        logger.warning("Failed to post result to Slack: %s", e)
