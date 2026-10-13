"""Paper summary generation.

A lighter-weight counterpart to :class:`PaperReviewer`: one call produces a
structured summary — a lead blockquote and five question sections (motivation,
solution, implementation, results, significance and limits) — with figures,
tables and math, using the same writing rules as the review.
"""

from __future__ import annotations

import re
from typing import Any

import boto3
from langchain_core.runnables import Runnable

from .constants import LanguageModelId
from .explainer import NO_CODEBASE, Paper
from .formatting import derived_subtitle, split_leading_h1, summary_lead
from .logger import logger
from .markdown_math import DISPLAY_MATH, HANGUL, image_sources
from .metrics import TokenBudgetExceeded, TokenBudgetGuard
from .post_validator import (
    description_markup,
    description_problems,
    paper_page_title,
    unsourced_numbers,
)
from .prompt_safety import neutralize_prompt_tags
from .prompts import BasePrompt, PaperSummaryPrompt, PaperSummaryShortenPrompt
from .review_plan import publish_blockers
from .utils import (
    BedrockLanguageModelFactory,
    HTMLTagOutputParser,
    RetryableBase,
    measure_execution_time,
)

DEFAULT_LANGUAGE: str = "Korean"


def _blockers(text: str) -> set[str]:
    """Publish blockers without their line numbers (the shortened text moves
    lines): a shorter summary may not add a kind the original did not have."""
    return {re.sub(r" \(line \d+\)$", "", b) for b in publish_blockers(text)}


# A summary over this many Hangul syllables gets one shortening pass down to
# the top of the range SUMMARY_DEPTH_RULES asks for (2,000-3,000).
MAX_SUMMARY_HANGUL: int = 3_500
SHORTEN_TARGET_HANGUL: int = 3_000


def hangul_length(text: str) -> int:
    return len(HANGUL.findall(text))


def _skeleton(summary: str) -> tuple[bool, list[str], list[str], int]:
    """What shortening must keep: the lead blockquote, the ``##`` headings,
    the images and the display equations (inline math may go with a cut
    list item)."""
    return (
        summary.lstrip().startswith(">"),
        re.findall(r"^##\s.*$", summary, re.MULTILINE),
        image_sources(summary),
        len(DISPLAY_MATH.findall(summary)),
    )


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
        self.notes: list[str] = []
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
        self.shorten_chain: Runnable = self._build_chain(
            summary_model_id,
            enable_thinking=False,
            thinking_effort=thinking_effort,
            callbacks=callbacks,
            prompt_cls=PaperSummaryShortenPrompt,
        )

    def _build_chain(
        self,
        model_id: LanguageModelId,
        *,
        enable_thinking: bool,
        thinking_effort: str,
        callbacks: list[Any] | None = None,
        prompt_cls: type[BasePrompt] = PaperSummaryPrompt,
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
            prompt_cls.get_prompt()
            | llm
            | HTMLTagOutputParser(tag_names=prompt_cls.output_variables or [])
        )

    @measure_execution_time
    async def summarize(self, paper: Paper) -> dict[str, str]:
        """Return ``{"subtitle", "description", "summary", "urls"}`` for the
        paper.

        ``summary`` is a Markdown fragment (a lead blockquote plus the five emoji
        ``##`` sections); ``subtitle`` is the title in the output language;
        ``description`` the search snippet (empty lets the blog derive one);
        ``urls`` is a comma-separated Markdown link list taken from the paper.
        There is no repair round: numbers in the TL;DR the paper never states,
        and a description that breaks its rules, become review notes.
        """
        self._enforce_token_budget()
        result = await self._summarize(paper.content.text, paper.codebase_summary)
        summary = result.get("summary", "").strip()
        if not summary:
            raise ValueError("Summarizer returned an empty summary.")
        logger.info("Generated summary (%d chars).", len(summary))
        summary = await self._shorten_if_long(summary)
        subtitle = result.get("subtitle", "").strip()
        if derived_subtitle(subtitle, split_leading_h1(summary)[0], paper.title):
            self.notes.append(
                "Summary subtitle was missing or repeated the title; it was "
                "derived from the first sentence: check it."
            )
        tldr = summary_lead(summary)
        if missing := unsourced_numbers(tldr, paper.content.text):
            self.notes.append(
                "Summary TL;DR numbers the paper never states (check each against "
                "the paper; 1B is 10억, 1M is 100만): " + ", ".join(missing)
            )
        description = result.get("description", "").strip()
        if problems := description_problems(
            description, f"{tldr}\n{paper.abstract or ''}"
        ):
            self.notes.append("Summary " + "; ".join(problems) + ": fix it.")
            if description_markup(description):
                description = ""  # the blog fails markup; it derives one instead
        page_title = paper_page_title(paper.title, korean=self.language == "Korean")
        if description and len(" ".join(description.split())) <= len(page_title):
            # The blog fails a description no longer than the page title.
            self.notes.append(
                "Summary description was not longer than the page title; dropped "
                "so the blog derives one."
            )
            description = ""
        return {
            "subtitle": subtitle,
            "description": description,
            "summary": summary,
            "urls": result.get("urls", "").strip(),
        }

    async def _shorten_if_long(self, summary: str) -> str:
        """One bounded shortening call for an over-long summary (never a
        loop). The shorter text is kept only when it is shorter and keeps the
        summary's structure without adding a publish-blocking error; any failure
        keeps the original. The cap is
        measured in Hangul syllables, so it applies to Korean summaries only."""
        length = hangul_length(summary)
        if self.language != "Korean" or length <= MAX_SUMMARY_HANGUL:
            return summary
        logger.info("Summary has %d Hangul syllables; shortening once.", length)
        try:
            self._enforce_token_budget()
            result = await self._shorten(summary)
            shorter = (result.get("summary") or "").strip()
        except TokenBudgetExceeded:
            raise  # an exhausted budget fails fast
        except Exception as e:  # noqa: BLE001 - the long summary is still usable
            logger.warning("Summary shortening failed: %s", e)
            shorter = ""
        if (
            shorter
            and hangul_length(shorter) < length
            and _skeleton(shorter) == _skeleton(summary)
            and _blockers(shorter) <= _blockers(summary)
        ):
            summary, length = shorter, hangul_length(shorter)
        elif shorter:
            logger.warning(
                "Shortened summary changed its structure or added a publish "
                "blocker; kept the original."
            )
        if length > MAX_SUMMARY_HANGUL:
            self.notes.append(
                f"Summary is still long ({length:,} Hangul syllables; target "
                f"{SHORTEN_TARGET_HANGUL:,}): trim it."
            )
        return summary

    @RetryableBase._retry("paper_summary_shortening")
    async def _shorten(self, summary: str) -> dict[str, str]:
        return await self.shorten_chain.ainvoke(
            {
                "summary": summary,
                "max_hangul": f"{SHORTEN_TARGET_HANGUL:,}",
                "language": self.language,
            }
        )

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
                    codebase_summary or NO_CODEBASE
                ),
                "language": self.language,
                "translation_guideline": str(self.translation_guideline),
            }
        )
