"""Paper Bot — a thin Slack dispatcher.

Listens for app mentions / direct messages, parses the requested action with an
LLM (:class:`IntentParser`), and submits the corresponding AWS Batch job
(:class:`JobDispatcher`). It acknowledges quickly and reports the dispatched job
back to the channel; the heavy pipeline runs asynchronously in Batch and posts
its result via the existing SNS path.

``slack_bolt`` is imported lazily so the rest of the package (and the test
suite) does not depend on it.
"""

from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field

import boto3

from ..configs import Config
from ..src.aws_helpers import get_ssm_param_value
from ..src.constants import EnvVars, LanguageModelId, SSMParams
from ..src.logger import logger
from .dispatcher import JobDispatcher, SlackContext, utc_timestamp
from .intent import IntentParser, ParsedIntent

_INTENT_MODEL = LanguageModelId.CLAUDE_V4_5_HAIKU

# Friendly emoji + label per intent, used in the acknowledgement reply.
_INTENT_LABELS = {
    "review": (":memo:", "Review"),
    "summarize": (":page_facing_up:", "Summary"),
    "guide": (":books:", "Tech guide"),
}


@dataclass
class SlackReply:
    """A Slack reply carrying both fallback text and rich Block Kit blocks.

    ``text`` is the notification/fallback string (shown in notifications and by
    older clients); ``blocks`` is the rich rendering. Both ``handle_message`` and
    the Socket Mode handler pass these straight to ``say(text=..., blocks=...)``.
    """

    text: str
    blocks: list[dict] = field(default_factory=list)


def _section(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _context(text: str) -> dict:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}


def _header(text: str) -> dict:
    # Header blocks are plain_text only (no mrkdwn/emoji-colon expansion beyond
    # standard emoji), so keep them short and literal.
    return {
        "type": "header",
        "text": {"type": "plain_text", "text": text, "emoji": True},
    }


def _one_line(text: str, *, limit: int = 600) -> str:
    """Collapse whitespace and cap an error string for a code block.

    Backticks are stripped so the surrounding ``` fence can't be broken out of.
    """
    flat = " ".join(text.replace("`", "ʼ").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def is_user_authorized(user_id: str | None) -> bool:
    """Whether a Slack user may trigger jobs.

    Controlled by ``SLACK_ALLOWED_USER_IDS`` (comma-separated). Unset ⇒ everyone
    is allowed (open mode). Set ⇒ only listed users; everyone else is refused.
    """
    allowlist = os.getenv(EnvVars.SLACK_ALLOWED_USER_IDS.value, "").strip()
    if not allowlist:
        return True
    allowed = {u.strip() for u in allowlist.split(",") if u.strip()}
    return bool(user_id and user_id in allowed)


class _SeenEvents:
    """Bounded set for de-duplicating at-least-once Slack event deliveries.

    Slack may redeliver the same event (same ``event_id``/``client_msg_id``);
    without this a single mention could launch duplicate Batch jobs. Bolt's
    SocketModeHandler dispatches sync handlers on a thread pool, so the
    check-insert-evict must be atomic — guarded by a lock.
    """

    def __init__(self, capacity: int = 512) -> None:
        self._capacity = capacity
        self._seen: dict[str, None] = {}
        self._lock = threading.Lock()

    def seen(self, key: str | None) -> bool:
        if not key:
            return False
        with self._lock:
            if key in self._seen:
                return True
            self._seen[key] = None
            if len(self._seen) > self._capacity:
                # Drop oldest insertion (dicts preserve insertion order).
                self._seen.pop(next(iter(self._seen)))
            return False


class PaperBot:
    """Glues intent parsing to job dispatch for a Slack workspace."""

    def __init__(
        self,
        config: Config,
        intent_parser: IntentParser,
        dispatcher: JobDispatcher,
    ) -> None:
        self.config = config
        self.intent_parser = intent_parser
        self.dispatcher = dispatcher

    async def handle_message(
        self, text: str, *, slack_context: SlackContext | None = None
    ) -> SlackReply:
        """Parse a message, dispatch a job, and return a user-facing reply.

        When ``slack_context`` is provided, it is threaded into the Batch job so
        the pipeline can post its result back to the originating channel/thread.
        """
        parsed = await self.intent_parser.parse(text)
        if not parsed.is_actionable:
            return self._help_reply(parsed)
        try:
            result = self.dispatcher.dispatch(
                parsed, timestamp=utc_timestamp(), slack_context=slack_context
            )
        except Exception as e:  # noqa: BLE001 - surface any dispatch error to the user
            logger.error("Dispatch failed: %s", e, exc_info=True)
            return self._dispatch_error_reply(e)
        return self._ack_reply(parsed, result.intent.value)

    @staticmethod
    def _ack_reply(parsed: ParsedIntent, intent_value: str) -> SlackReply:
        emoji, nice = _INTENT_LABELS.get(
            intent_value, (":rocket:", intent_value.capitalize())
        )
        sources = ", ".join(parsed.sources)
        extras = []
        if parsed.repo_urls:
            extras.append(f":link: code: {', '.join(parsed.repo_urls)}")
        if parsed.parse_pdf:
            extras.append(":page_with_curl: PDF parsing on")
        blocks = [
            _section(f"{emoji}  *{nice}* started for `{sources}`"),
            _context(
                ":hourglass_flowing_sand: I'll post the result here when it's ready."
            ),
        ]
        if extras:
            blocks.insert(1, _context("   ·   ".join(extras)))
        return SlackReply(text=f"{nice} started for {sources}", blocks=blocks)

    @staticmethod
    def _dispatch_error_reply(error: Exception) -> SlackReply:
        return SlackReply(
            text=f"Couldn't start the job: {error}",
            blocks=[
                _header(":x: Couldn't start the job"),
                _section(f"```{_one_line(str(error))}```"),
                _context(
                    ":arrows_counterclockwise: Please try again in a moment, "
                    "or check the bot logs if it keeps happening."
                ),
            ],
        )

    @staticmethod
    def _help_reply(parsed: ParsedIntent) -> SlackReply:
        reason = f"  _({parsed.reason})_" if parsed.reason else ""
        return SlackReply(
            text="I couldn't tell what you'd like me to do.",
            blocks=[
                _section(
                    f":wave: I couldn't tell what you'd like me to do.{reason}\n"
                    "Here's what I can do:"
                ),
                _section(
                    ":memo:  `review 2401.06066`\n_In-depth, section-by-section paper review._\n\n"
                    ":page_facing_up:  `summarize https://arxiv.org/pdf/2401.06066`\n"
                    "_Concise five-point summary._\n\n"
                    ":books:  `guide https://docs.framework.io/start`\n"
                    "_Technical how-to guide built from documentation._"
                ),
                _context(
                    ":bulb: Tip: add a GitHub repo to ground the implementation, "
                    'or say "parse the PDF" to force PDF parsing.'
                ),
            ],
        )

    @staticmethod
    def _unauthorized_reply() -> SlackReply:
        return SlackReply(
            text="You're not authorized to run jobs.",
            blocks=[
                _section(
                    ":lock:  *Sorry, you're not authorized to run jobs.*\n"
                    "_Ask an admin to add you to the allowlist._"
                )
            ],
        )


def build_bot(config: Config | None = None) -> PaperBot:
    """Construct a :class:`PaperBot` from configuration + SSM/Batch details."""
    config = config or Config.load()
    profile_name = os.getenv(EnvVars.AWS_PROFILE_NAME.value) or (
        config.resources.profile_name
    )
    default_session = boto3.Session(
        profile_name=profile_name, region_name=config.resources.default_region_name
    )
    bedrock_session = boto3.Session(
        profile_name=profile_name, region_name=config.resources.bedrock_region_name
    )

    base = f"/{config.resources.project_name}/{config.resources.stage}"
    job_queue = get_ssm_param_value(
        default_session, f"{base}/{SSMParams.BATCH_JOB_QUEUE.value}"
    )
    job_definition = get_ssm_param_value(
        default_session, f"{base}/{SSMParams.BATCH_JOB_DEFINITION.value}"
    )
    guide_job_queue = _optional_ssm(
        default_session, f"{base}/{SSMParams.GUIDE_JOB_QUEUE.value}"
    )
    guide_job_definition = _optional_ssm(
        default_session, f"{base}/{SSMParams.GUIDE_JOB_DEFINITION.value}"
    )

    dispatcher = JobDispatcher(
        default_session,
        project_name=config.resources.project_name,
        stage=config.resources.stage,
        review_job_queue=job_queue,
        review_job_definition=job_definition,
        guide_job_queue=guide_job_queue,
        guide_job_definition=guide_job_definition,
    )
    return PaperBot(config, IntentParser(_INTENT_MODEL, bedrock_session), dispatcher)


def _optional_ssm(session: boto3.Session, name: str) -> str | None:
    """Read an SSM parameter, returning None if it is not present."""
    try:
        return get_ssm_param_value(session, name)
    except Exception as e:  # noqa: BLE001 - missing optional param is non-fatal
        logger.info("Optional SSM parameter '%s' not found: %s", name, e)
        return None


class SlackAppMismatchError(RuntimeError):
    """Raised when the Slack app token does not belong to the expected app."""


def _app_id_from_app_token(app_token: str) -> str | None:
    """Extract the Slack app id from an app-level token.

    App tokens are formatted ``xapp-1-<APP_ID>-<...>``; the app id is the second
    dash-delimited segment after the ``xapp-1`` prefix. Returns ``None`` if the
    token is malformed so the caller can degrade gracefully.
    """
    parts = app_token.split("-")
    if len(parts) >= 3 and parts[0] == "xapp":
        return parts[2]
    return None


def verify_slack_app_identity(bot_token: str, app_token: str) -> None:
    """Fail fast if the tokens belong to the wrong Slack app.

    Paper Bot must run on its OWN Slack app. Sharing one app's tokens with
    another Socket Mode bot (e.g. OmniSummary) makes Slack deliver each event to
    only one of the connected processes at random, so mentions silently route to
    the wrong bot. When ``SLACK_EXPECTED_APP_ID`` is set, a mismatch raises
    :class:`SlackAppMismatchError` at startup instead of causing that confusing
    runtime behaviour. The check is offline: the app id lives inside the app
    token itself (``xapp-1-<APP_ID>-...``).
    """
    actual_app_id = _app_id_from_app_token(app_token)
    expected = os.getenv(EnvVars.SLACK_EXPECTED_APP_ID.value)
    if not expected:
        # The guard is opt-in. When the env var is unset we simply skip it — no
        # warning, since running a single dedicated app is the normal setup.
        logger.debug("SLACK_EXPECTED_APP_ID is not set; skipping app-identity check.")
        return
    if actual_app_id is None:
        logger.warning(
            "Could not parse an app id from SLACK_APP_TOKEN; skipping identity "
            "check (expected app id %s).",
            expected,
        )
        return
    if actual_app_id != expected:
        raise SlackAppMismatchError(
            f"SLACK_APP_TOKEN belongs to app {actual_app_id}, but "
            f"SLACK_EXPECTED_APP_ID={expected}. Paper Bot must use its OWN Slack "
            "app — refusing to start to avoid colliding with another bot."
        )
    logger.info("Slack app identity verified: app id %s.", actual_app_id)


def run_socket_mode() -> None:  # pragma: no cover - requires live Slack tokens
    """Run the bot in Slack Socket Mode (long-running process).

    Requires ``slack_bolt`` and the ``SLACK_BOT_TOKEN`` / ``SLACK_APP_TOKEN``
    environment variables. Set ``SLACK_EXPECTED_APP_ID`` to bind this process to
    Paper Bot's own Slack app and avoid colliding with another Socket Mode bot.
    """
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    bot_token = os.environ["SLACK_BOT_TOKEN"]
    app_token = os.environ["SLACK_APP_TOKEN"]
    verify_slack_app_identity(bot_token, app_token)

    bot = build_bot()
    app = App(token=bot_token)
    seen_events = _SeenEvents()

    def _handle(event: dict, say) -> None:  # type: ignore[no-untyped-def]
        # Drop redelivered events so one mention can't launch duplicate jobs.
        dedup_key = event.get("client_msg_id") or event.get("event_ts")
        if seen_events.seen(dedup_key):
            logger.info("Ignoring duplicate Slack event '%s'.", dedup_key)
            return
        # Authorization: refuse users not on the allowlist (when configured).
        user = event.get("user")
        thread_ts = event.get("thread_ts") or event.get("ts")
        if not is_user_authorized(user):
            logger.warning("Unauthorized Slack user '%s' attempted a job.", user)
            denied = PaperBot._unauthorized_reply()
            say(text=denied.text, blocks=denied.blocks, thread_ts=thread_ts)
            return
        # Reply (and later post the result) in the originating thread.
        ctx = SlackContext(
            channel=event.get("channel", ""),
            thread_ts=thread_ts,
            user=user,
        )
        reply = asyncio.run(
            bot.handle_message(event.get("text", ""), slack_context=ctx)
        )
        say(text=reply.text, blocks=reply.blocks, thread_ts=thread_ts)

    @app.event("app_mention")
    def _on_mention(event: dict, say) -> None:  # type: ignore[no-untyped-def]
        _handle(event, say)

    @app.event("message")
    def _on_message(event: dict, say) -> None:  # type: ignore[no-untyped-def]
        if event.get("channel_type") == "im" and not event.get("bot_id"):
            _handle(event, say)

    logger.info("Starting Paper Bot in Socket Mode...")
    SocketModeHandler(app, app_token).start()


if __name__ == "__main__":  # pragma: no cover
    run_socket_mode()
