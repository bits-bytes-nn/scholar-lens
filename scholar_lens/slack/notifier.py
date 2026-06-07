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

    The title and error strings come from paper metadata / exception text. Slack
    mrkdwn does NOT honour backslash escaping (``\\_`` renders the backslash
    literally), so instead of escaping we swap the characters that genuinely
    break a layout — backtick and ``*`` — for visually-identical homoglyphs.
    ``_`` and ``~`` only italicise/strike when they wrap text at word
    boundaries, so a mid-token ``_`` (e.g. an arXiv id like ``2504_03182``) is
    left untouched and displays cleanly. Newlines are collapsed so an injected
    string can't reshape the message.
    """
    out = value.replace("`", "ʼ").replace("*", "∗")
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

    text, blocks = _build_result_message(
        success=success,
        artifact_label=artifact_label,
        title=title,
        s3_url=s3_url,
        error=error,
    )
    try:
        from slack_sdk import WebClient

        WebClient(token=token).chat_postMessage(
            channel=channel,
            thread_ts=_clean(thread_ts),
            text=text,  # notification/fallback text
            blocks=blocks,
        )
        logger.info("Posted %s result to Slack channel '%s'.", artifact_label, channel)
    except Exception as e:  # noqa: BLE001 - Slack delivery must never fail the job
        logger.warning("Failed to post result to Slack: %s", e)


# Friendly labels + emoji per artifact type, shown in the result header.
_ARTIFACT_META = {
    "review": (":memo:", "Paper Review"),
    "summary": (":page_facing_up:", "Paper Summary"),
    "guide": (":books:", "Tech Guide"),
}


def _build_result_message(
    *,
    success: bool,
    artifact_label: str,
    title: str,
    s3_url: str | None,
    error: str | None,
) -> tuple[str, list[dict]]:
    """Build (fallback_text, Block Kit blocks) for a completion message."""
    emoji, nice = _ARTIFACT_META.get(
        artifact_label.lower(), (":robot_face:", artifact_label.capitalize())
    )
    safe_title = _mrkdwn_safe(title)

    if success:
        header = f"{emoji} {nice} ready"
        fallback = f"{nice} ready for {title}"
        section = f"*{safe_title}*"
        blocks: list[dict] = [
            {"type": "header", "text": {"type": "plain_text", "text": header}},
            {"type": "section", "text": {"type": "mrkdwn", "text": section}},
        ]
        if s3_url:
            # Render https links as a clickable button; show s3:// URIs as code.
            if s3_url.startswith(("http://", "https://")):
                blocks.append(
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": ":open_file_folder: View output",
                                },
                                "url": s3_url,
                            }
                        ],
                    }
                )
            else:
                blocks.append(
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f":open_file_folder: Output: `{_mrkdwn_safe(s3_url)}`",
                        },
                    }
                )
    else:
        header = f"{nice} couldn't be completed"
        fallback = f"{nice} failed for {title}"
        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": f":x: {header}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*{safe_title}*"}},
        ]
        if error:
            blocks.append({"type": "divider"})
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":warning: *What went wrong*\n```{_short(error)}```",
                    },
                }
            )
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "Check the AWS Batch job logs for the full trace.",
                        }
                    ],
                }
            )
    return fallback, blocks


def _short(text: str, *, limit: int = 600) -> str:
    """Collapse whitespace and cap an error string for display in a code block.

    Backticks are stripped so the surrounding ``` fence can't be broken out of.
    """
    flat = " ".join(text.replace("`", "ʼ").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
