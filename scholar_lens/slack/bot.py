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
from .dispatcher import JobDispatcher, utc_timestamp
from .intent import IntentParser, ParsedIntent

_INTENT_MODEL = LanguageModelId.CLAUDE_V4_5_HAIKU


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

    async def handle_message(self, text: str) -> str:
        """Parse a message, dispatch a job, and return a user-facing reply."""
        parsed = await self.intent_parser.parse(text)
        if not parsed.is_actionable:
            return self._help_reply(parsed)
        try:
            result = self.dispatcher.dispatch(parsed, timestamp=utc_timestamp())
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


def run_socket_mode() -> None:  # pragma: no cover - requires live Slack tokens
    """Run the bot in Slack Socket Mode (long-running process).

    Requires ``slack_bolt`` and the ``SLACK_BOT_TOKEN`` / ``SLACK_APP_TOKEN``
    environment variables.
    """
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    bot = build_bot()
    app = App(token=os.environ["SLACK_BOT_TOKEN"])

    @app.event("app_mention")
    def _on_mention(event: dict, say) -> None:  # type: ignore[no-untyped-def]
        text = event.get("text", "")
        reply = asyncio.run(bot.handle_message(text))
        say(reply)

    @app.event("message")
    def _on_message(event: dict, say) -> None:  # type: ignore[no-untyped-def]
        if event.get("channel_type") == "im" and not event.get("bot_id"):
            reply = asyncio.run(bot.handle_message(event.get("text", "")))
            say(reply)

    logger.info("Starting Paper Bot in Socket Mode...")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":  # pragma: no cover
    run_socket_mode()
