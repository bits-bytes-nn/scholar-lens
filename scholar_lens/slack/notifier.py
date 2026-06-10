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

from ..src.constants import AppConstants, EnvVars
from ..src.logger import logger
from .blocks import context, header, mrkdwn_safe, one_line, section


def _clean(value: str | None) -> str | None:
    """Treat the Batch NULL sentinel and blanks as absent."""
    if not value or value.strip().lower() == AppConstants.NULL_STRING:
        return None
    return value.strip()


def post_slack_result(
    *,
    channel: str | None,
    thread_ts: str | None,
    success: bool,
    artifact_label: str,
    title: str,
    s3_url: str | None,
    pr_url: str | None = None,
    error: str | None = None,
    sources: str | None = None,
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
        pr_url=pr_url,
        error=error,
        sources=sources,
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


def _link_button(text: str, url: str) -> dict:
    return {"type": "button", "text": {"type": "plain_text", "text": text}, "url": url}


def _build_result_message(
    *,
    success: bool,
    artifact_label: str,
    title: str,
    s3_url: str | None,
    pr_url: str | None = None,
    error: str | None = None,
    sources: str | None = None,
) -> tuple[str, list[dict]]:
    """Build (fallback_text, Block Kit blocks) for a completion message."""
    emoji, nice = _ARTIFACT_META.get(
        artifact_label.lower(), (":robot_face:", artifact_label.capitalize())
    )
    # Coerce None/blank defensively: this runs OUTSIDE post_slack_result's try,
    # so an AttributeError here would fail the whole job, not just the post.
    safe_title = mrkdwn_safe(title or "(untitled)")
    # Show the source(s) as a muted sub-line only when they differ from the
    # title (e.g. a guide's title is the generated topic, not its source URL).
    source_line = None
    safe_sources = mrkdwn_safe(sources) if sources else ""
    if safe_sources and safe_sources != safe_title:
        source_line = safe_sources

    if success:
        fallback = f"{nice} ready: {title}"
        blocks: list[dict] = [
            header(f"{emoji} {nice} ready"),
            section(f"*{safe_title}*"),
        ]
        if source_line:
            blocks.append(context(f":link: {source_line}"))
        # Clickable buttons for any https links (blog PR, http output); s3:// URIs
        # aren't clickable, so they go in a muted context line instead.
        buttons = []
        if pr_url and pr_url.startswith(("http://", "https://")):
            buttons.append(_link_button(":github: View pull request", pr_url))
        if s3_url and s3_url.startswith(("http://", "https://")):
            buttons.append(_link_button(":open_file_folder: View output", s3_url))
        if buttons:
            blocks.append({"type": "actions", "elements": buttons})
        if s3_url and not s3_url.startswith(("http://", "https://")):
            blocks.append(context(f":floppy_disk: `{mrkdwn_safe(s3_url)}`"))
    else:
        fallback = f"{nice} failed: {title}"
        blocks = [
            header(f":x: {nice} couldn't be completed"),
            section(f"*{safe_title}*"),
        ]
        if source_line:
            blocks.append(context(f":link: {source_line}"))
        if error:
            blocks.append({"type": "divider"})
            blocks.append(
                section(f":warning: *What went wrong*\n```{one_line(error)}```")
            )
            blocks.append(context("Check the AWS Batch job logs for the full trace."))
    return fallback, blocks
