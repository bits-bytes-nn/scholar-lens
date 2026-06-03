"""Paper summary generation.

A lighter-weight counterpart to :class:`ExplainerGraph`: instead of a
section-by-section multi-stage review, this produces a single, structured
*summary* in the proven five-question format (motivation / novel solution /
implementation / results / significance), with inline images, tables, maths and
code. It reuses the project's Bedrock factory and prompt conventions.
"""

from __future__ import annotations

from typing import Any

import boto3
from langchain_core.runnables import Runnable

from .constants import LanguageModelId
from .explainer import Paper
from .logger import logger
from .prompts import PaperSummaryPrompt
from .utils import (
    BedrockLanguageModelFactory,
    HTMLTagOutputParser,
    RetryableBase,
    measure_execution_time,
)

DEFAULT_LANGUAGE: str = "Korean"


class PaperSummarizer(RetryableBase):
    """Generates a structured five-section summary for a :class:`Paper`."""

    def __init__(
        self,
        summary_model_id: LanguageModelId,
        boto_session: boto3.Session,
        *,
        language: str = DEFAULT_LANGUAGE,
        translation_guideline: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        thinking_effort: str = "medium",
    ) -> None:
        self.language = language
        self.translation_guideline = translation_guideline or []
        self.llm_factory = BedrockLanguageModelFactory(boto_session=boto_session)
        self.summary_chain: Runnable = self._build_chain(
            summary_model_id,
            enable_thinking=enable_thinking,
            thinking_effort=thinking_effort,
        )

    def _build_chain(
        self, model_id: LanguageModelId, *, enable_thinking: bool, thinking_effort: str
    ) -> Runnable:
        llm = self.llm_factory.get_model(
            model_id,
            temperature=0.0,
            enable_thinking=enable_thinking,
            thinking_effort=thinking_effort,
            supports_1m_context_window=True,
        )
        return (
            PaperSummaryPrompt.get_prompt()
            | llm
            | HTMLTagOutputParser(tag_names=PaperSummaryPrompt.output_variables)
        )

    @measure_execution_time
    async def summarize(self, paper: Paper) -> dict[str, str]:
        """Return ``{"summary", "tags", "urls"}`` for the paper.

        ``summary`` is an HTML fragment (the five emoji sections); ``tags`` is a
        comma-separated keyword list; ``urls`` is a comma-separated markdown link
        list. All three are LLM-extracted from the paper content.
        """
        result = await self._summarize(paper.content.text)
        summary = result.get("summary", "").strip()
        if not summary:
            raise ValueError("Summarizer returned an empty summary.")
        logger.info(
            "Generated summary (%d chars, %d tags).",
            len(summary),
            len(self._split_tags(result.get("tags", ""))),
        )
        return {
            "summary": summary,
            "tags": result.get("tags", "").strip(),
            "urls": result.get("urls", "").strip(),
        }

    @RetryableBase._retry("paper_summarization")
    async def _summarize(self, content: str) -> dict[str, str]:
        return await self.summary_chain.ainvoke(
            {
                "content": content,
                "language": self.language,
                "translation_guideline": str(self.translation_guideline),
            }
        )

    @staticmethod
    def _split_tags(tags: str) -> list[str]:
        return [t.strip() for t in tags.split(",") if t.strip()]
