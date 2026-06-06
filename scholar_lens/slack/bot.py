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

import boto3

from ..configs import Config
from ..src.aws_helpers import get_ssm_param_value
from ..src.constants import EnvVars, LanguageModelId, SSMParams
from ..src.logger import logger
from .dispatcher import JobDispatcher, SlackContext, utc_timestamp
from .intent import IntentParser, ParsedIntent

_INTENT_MODEL = LanguageModelId.CLAUDE_V4_5_HAIKU


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
    without this a single mention could launch duplicate Batch jobs. Kept small
    and in-process — sufficient for a single Socket Mode worker.
    """

    def __init__(self, capacity: int = 512) -> None:
        self._capacity = capacity
        self._seen: dict[str, None] = {}

    def seen(self, key: str | None) -> bool:
        if not key:
            return False
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
    ) -> str:
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
            return f":warning: Failed to start the job: {e}"
        return (
            f":rocket: Started a *{result.intent.value}* job "
            f"(`{result.job_name}`) for: {', '.join(parsed.sources)}.\n"
            f"I'll post the result here when it's ready."
        )

    @staticmethod
    def _help_reply(parsed: ParsedIntent) -> str:
        return (
            ":wave: I couldn't turn that into an action"
            + (f" ({parsed.reason})" if parsed.reason else "")
            + ".\nTry:\n"
            "• `review 2401.06066`\n"
            "• `summarize https://arxiv.org/pdf/2401.06066`\n"
            "• `guide https://docs.framework.io/start https://docs.framework.io/api`"
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
        logger.warning(
            "SLACK_EXPECTED_APP_ID is not set; skipping Slack app-identity check. "
            "Set it to Paper Bot's own app id (starts with 'A') to guard against "
            "sharing a token with another bot — see README.",
        )
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
            say(
                text=":lock: Sorry, you're not authorized to run Paper Bot jobs.",
                thread_ts=thread_ts,
            )
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
        say(text=reply, thread_ts=thread_ts)

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
