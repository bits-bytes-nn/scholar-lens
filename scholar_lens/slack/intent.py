"""Intent parsing for the Paper Bot Slack dispatcher.

The only "intelligence" the bot needs is turning a free-form chat message into a
structured action (``review`` / ``summarize`` / ``guide`` + inputs). That single
LLM call lives here, isolated and unit-testable; the Bolt wiring and job
dispatch are kept separate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import boto3
from langchain_core.runnables import Runnable

from ..src.constants import LanguageModelId
from ..src.logger import logger
from ..src.prompts import SlackIntentPrompt
from ..src.utils import BedrockLanguageModelFactory, HTMLTagOutputParser

# Slack wraps URLs as <url> or <url|label>; strip that decoration.
_SLACK_LINK = re.compile(r"<(https?://[^>|]+)(?:\|[^>]*)?>")


class SlackIntent(str, Enum):
    REVIEW = "review"
    SUMMARIZE = "summarize"
    GUIDE = "guide"
    UNKNOWN = "unknown"


@dataclass
class ParsedIntent:
    intent: SlackIntent
    sources: list[str] = field(default_factory=list)
    repo_urls: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.intent is not SlackIntent.UNKNOWN and bool(self.sources)


class IntentParser:
    """Classifies a Slack message into a :class:`ParsedIntent` via an LLM."""

    def __init__(self, model_id: LanguageModelId, boto_session: boto3.Session) -> None:
        factory = BedrockLanguageModelFactory(boto_session=boto_session)
        self.chain: Runnable = (
            SlackIntentPrompt.get_prompt()
            | factory.get_model(model_id, temperature=0.0)
            | HTMLTagOutputParser(tag_names=SlackIntentPrompt.output_variables)
        )

    async def parse(self, message: str) -> ParsedIntent:
        cleaned = _unwrap_slack_links(message)
        result = await self.chain.ainvoke({"message": cleaned})
        return self.from_raw(result)

    @staticmethod
    def from_raw(result: dict[str, str]) -> ParsedIntent:
        """Build a ParsedIntent from raw tag output (pure; easy to test)."""
        intent = _coerce_intent(result.get("intent", "unknown"))
        return ParsedIntent(
            intent=intent,
            sources=_split_csv(result.get("sources", "")),
            repo_urls=_split_csv(result.get("repo_urls", "")),
            reason=result.get("reason", "").strip(),
        )


def _coerce_intent(value: str) -> SlackIntent:
    try:
        return SlackIntent(value.strip().lower())
    except ValueError:
        return SlackIntent.UNKNOWN


def _split_csv(value: str) -> list[str]:
    items = [item.strip() for item in re.split(r"[,\n]", value or "")]
    return [item for item in items if item and item.lower() != "empty"]


def _unwrap_slack_links(message: str) -> str:
    cleaned = _SLACK_LINK.sub(r"\1", message)
    if cleaned != message:
        logger.debug("Unwrapped Slack link decoration in message.")
    return cleaned
