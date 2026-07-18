"""Shared Slack Block Kit builders and text sanitisers.

Both the bot (acknowledgement / help / error replies) and the notifier (job-result
messages) build Block Kit blocks and must sanitise user-influenced text (paper
titles, exception messages) before placing it in mrkdwn. These helpers live here
so the two modules share one implementation instead of drifting copies.
"""

from __future__ import annotations

# Slack mrkdwn does NOT honour backslash escaping ("\_" renders the backslash
# literally), so to neutralise control characters we swap the two that actually
# break a layout — backtick and "*" — for visually-identical homoglyphs. "_" and
# "~" only italicise/strike when they wrap text at word boundaries, so leaving a
# mid-token "_" (e.g. an arXiv id like "2504_03182") untouched displays cleanly.
_MRKDWN_SUBSTITUTIONS = str.maketrans({"`": "ʼ", "*": "∗"})

# Cap for error/detail strings shown in a code block, so a stack trace can't
# flood the channel. Shared by the bot and notifier.
DEFAULT_DETAIL_LIMIT = 600

# Slack's hard per-field character limits. Exceeding them makes chat.postMessage
# reject the whole message (invalid_blocks), which the notifier swallows — so a
# job with an over-long title/source list (e.g. a guide whose title falls back to
# 40+ joined URLs) would post NOTHING. Enforce the caps in the builders so no
# caller can construct an over-limit block.
_SECTION_TEXT_LIMIT = 3000
_HEADER_TEXT_LIMIT = 150


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def mrkdwn_safe(value: str) -> str:
    """Neutralise Slack mrkdwn control chars in user-influenced text and collapse
    whitespace so an injected string can't reshape the message."""
    return " ".join(value.translate(_MRKDWN_SUBSTITUTIONS).split())


def one_line(text: str, *, limit: int = DEFAULT_DETAIL_LIMIT) -> str:
    """Collapse whitespace and cap an error string for display in a code block.

    Backticks are stripped so the surrounding ``` fence can't be broken out of."""
    flat = " ".join(text.replace("`", "ʼ").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def header(text: str) -> dict:
    """A header block (plain_text only; standard emoji are still rendered)."""
    return {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": _truncate(text, _HEADER_TEXT_LIMIT),
            "emoji": True,
        },
    }


def section(text: str) -> dict:
    """A mrkdwn section block (text capped to Slack's 3000-char limit)."""
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": _truncate(text, _SECTION_TEXT_LIMIT)},
    }


def context(text: str) -> dict:
    """A mrkdwn context block (smaller, muted text; capped to Slack's limit)."""
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": _truncate(text, _SECTION_TEXT_LIMIT)}],
    }
