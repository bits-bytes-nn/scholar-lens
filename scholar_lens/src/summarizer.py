"""Paper summary generation.

A lighter-weight counterpart to :class:`ExplainerGraph`: one call produces a
structured summary — a lead blockquote and five question sections (motivation,
solution, implementation, results, significance and limits) — with figures,
tables and math, using the same writing rules as the review.
"""

from __future__ import annotations

from typing import Any

import boto3
from langchain_core.runnables import Runnable

from .constants import LanguageModelId
from .explainer import Paper
from .logger import logger
from .metrics import TokenBudgetGuard
from .prompts import PaperSummaryPrompt
from .utils import (
    BedrockLanguageModelFactory,
    HTMLTagOutputParser,
    RetryableBase,
    measure_execution_time,
)
from .web_research import neutralize_prompt_tags

DEFAULT_LANGUAGE: str = "Korean"
# Language-neutral sentinel for "no code repository was provided"; the summary
# prompt keys off this exact string to decide whether to use a codebase.
_NO_CODEBASE: str = "(no code repository provided)"


class PaperSummarizer(RetryableBase, TokenBudgetGuard):
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
        max_total_tokens: int | None = None,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.language = language
        self.translation_guideline = translation_guideline or []
        self.summary_model_id = summary_model_id
        # Hard total-token ceiling (None = no limit). Checked before the summary
        # call so an already-exhausted budget (from data prep) aborts rather than
        # spending more — parity with the review/guide token guards.
        self._init_token_budget(callbacks, max_total_tokens)
        self.llm_factory = BedrockLanguageModelFactory(boto_session=boto_session)
        self.summary_chain: Runnable = self._build_chain(
            summary_model_id,
            enable_thinking=enable_thinking,
            thinking_effort=thinking_effort,
            callbacks=callbacks,
        )

    def _build_chain(
        self,
        model_id: LanguageModelId,
        *,
        enable_thinking: bool,
        thinking_effort: str,
        callbacks: list[Any] | None = None,
    ) -> Runnable:
        llm = self.llm_factory.get_model(
            model_id,
            temperature=0.0,
            enable_thinking=enable_thinking,
            thinking_effort=thinking_effort,
            supports_1m_context_window=True,
            callbacks=callbacks or None,
        )
        return (
            PaperSummaryPrompt.get_prompt()
            | llm
            | HTMLTagOutputParser(tag_names=PaperSummaryPrompt.output_variables)
        )

    @measure_execution_time
    async def summarize(self, paper: Paper) -> dict[str, str]:
        """Return ``{"subtitle", "summary", "urls"}`` for the paper.

        ``summary`` is a Markdown fragment (a lead blockquote plus the five emoji
        ``##`` sections); ``subtitle`` is the title in the output language;
        ``urls`` is a comma-separated Markdown link list taken from the paper.
        """
        self._enforce_token_budget()
        result = await self._summarize(paper.content.text, paper.codebase_summary)
        summary = result.get("summary", "").strip()
        if not summary:
            raise ValueError("Summarizer returned an empty summary.")
        logger.info("Generated summary (%d chars).", len(summary))
        return {
            "subtitle": result.get("subtitle", "").strip(),
            "summary": summary,
            "urls": result.get("urls", "").strip(),
        }

    @RetryableBase._retry("paper_summarization")
    async def _summarize(
        self, content: str, codebase_summary: str | None = None
    ) -> dict[str, str]:
        # Fit the paper to the model's context window (exact, via CountTokens) so
        # a very long paper is trimmed rather than overflowing the call.
        content = self.llm_factory.fit_text(
            self.summary_model_id, content, label="summary content"
        )
        # Defang prompt-fence tags in the untrusted paper/codebase text so a
        # literal "</paper>" in the source can't break out of the data fence.
        return await self.summary_chain.ainvoke(
            {
                "content": neutralize_prompt_tags(content),
                "codebase_summary": neutralize_prompt_tags(
                    codebase_summary or _NO_CODEBASE
                ),
                "language": self.language,
                "translation_guideline": str(self.translation_guideline),
            }
        )
