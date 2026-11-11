"""The survey's LangChain model calls: the plan stage's scope and selection
(JSON in a tag, parsed and checked in code with one repair round) and the
write stage's finalization (description and taxonomy)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any, TypeVar

import boto3
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from scholar_lens.adapters.bedrock.chat_models import BedrockLanguageModelFactory
from scholar_lens.adapters.bedrock.output_parsers import HTMLTagOutputParser
from scholar_lens.adapters.bedrock.prompts.base import BasePrompt
from scholar_lens.adapters.bedrock.prompts.survey import (
    SurveyFinalizationPrompt,
    SurveyScopePrompt,
    SurveySelectionPrompt,
)
from scholar_lens.adapters.bedrock.usage import TokenBudgetGuard
from scholar_lens.adapters.retry import RetryableBase
from scholar_lens.domain.constants import LanguageModelId
from scholar_lens.domain.prompt_safety import neutralize_prompt_tags
from scholar_lens.domain.review_plan import digest_sections
from scholar_lens.domain.survey.plan import (
    Candidate,
    SurveyPlanError,
    SurveyScope,
    SurveySelection,
    TimeRange,
    check_selection,
    parse_tagged_json,
    shortlist_digest,
)
from scholar_lens.domain.survey.references import (
    CandidatePool,
    ReadReference,
    reference_excerpts,
)
from scholar_lens.domain.tags import controlled_tag_problems
from scholar_lens.domain.usage_context import staged
from scholar_lens.domain.validation import (
    description_markup,
    description_problems,
    fit_description,
)

logger = logging.getLogger(__name__)


M = TypeVar("M")


# Reference material shown to the scope call, over all references.
SCOPE_REFERENCE_CHARS = 8_000


class SurveyCalls(RetryableBase, TokenBudgetGuard):
    """Model calls answering with JSON in a tag, parsed and checked in code
    with one repair round, within the run's two-tier token budget (shared
    by the plan and write stages)."""

    async def _ask(
        self,
        chain: Runnable,
        payload: dict[str, Any],
        parse: Callable[[str], tuple[M, list[str]]],
        *,
        tag: str,
        accept_last: bool = False,
    ) -> M:
        """Call, parse and check; once more with the problems listed (within
        the soft token limit); then fail with ``SurveyPlanError``. With
        ``accept_last`` the last answer that parses is returned despite its
        problems (the caller removes what they concern)."""
        payload = {**payload, "repair_note": ""}
        problems: list[str] = []
        for attempt in range(2):
            raw = await self._invoke(chain, payload)
            try:
                result, problems = parse(raw)
            except SurveyPlanError as e:
                problems = [str(e)[:1000]]
            else:
                last = attempt == 1 or self._over_soft_token_budget()
                if not problems or (accept_last and last):
                    return result
            logger.warning(
                "Survey %s unusable (attempt %d): %s", tag, attempt + 1, problems
            )
            if self._over_soft_token_budget():
                break
            payload["repair_note"] = (
                "\nYour previous answer could not be used:\n"
                + "\n".join(f"- {p}" for p in problems)
                + f"\nReturn the complete answer again as valid JSON inside <{tag}> "
                "tags, fixing every problem."
            )
        raise SurveyPlanError(
            f"The survey {tag} call gave no usable answer: {problems}"
        )

    @RetryableBase._retry("survey_call")
    async def _invoke(self, chain: Runnable, payload: dict[str, Any]) -> str:
        self._enforce_hard_token_budget()
        raw: str = await chain.ainvoke(payload)
        return raw


class SurveyPlanner(SurveyCalls):
    """The scope and selection calls, each parsed and checked in code with
    one repair round."""

    def __init__(
        self,
        *,
        scope_model_id: LanguageModelId,
        selection_model_id: LanguageModelId,
        boto_session: boto3.Session,
        thinking_effort: str,
        max_total_tokens: int | None,
        callbacks: list[Any] | None = None,
        llm_factory: BedrockLanguageModelFactory | None = None,
    ) -> None:
        self.callbacks = callbacks or []
        self._init_token_budget(self.callbacks, max_total_tokens)
        factory = llm_factory or BedrockLanguageModelFactory(boto_session=boto_session)

        def chain(
            prompt: type[BasePrompt], model_id: LanguageModelId, *, thinking: bool
        ) -> Runnable:
            model = factory.get_model(
                model_id,
                temperature=0.0,
                callbacks=self.callbacks or None,
                enable_thinking=thinking,
                thinking_effort=thinking_effort,
            )
            return prompt.get_prompt() | model | StrOutputParser()

        self.scope_chain = chain(SurveyScopePrompt, scope_model_id, thinking=False)
        # Choosing and grouping the papers is the stage's main judgment.
        self.selection_chain = chain(
            SurveySelectionPrompt, selection_model_id, thinking=True
        )

    @staged("scope")
    async def scope(
        self,
        topic: str,
        time_range: TimeRange,
        required: list[Candidate],
        references: Sequence[ReadReference] = (),
    ) -> SurveyScope:
        payload = {
            "topic": neutralize_prompt_tags(topic),
            "time_range": _range_text(time_range),
            "required_papers": neutralize_prompt_tags(
                "\n".join(f"- {c.title} ({c.year or 'year unknown'})" for c in required)
                or "- none"
            ),
            "references": reference_excerpts(
                [r.page for r in references if r.page],
                topic,
                SCOPE_REFERENCE_CHARS,
            ),
        }

        def parse(raw: str) -> tuple[SurveyScope, list[str]]:
            return parse_tagged_json(raw, "scope", SurveyScope), []

        return await self._ask(self.scope_chain, payload, parse, tag="scope")

    @staged("select")
    async def select(
        self,
        topic: str,
        scope: SurveyScope,
        pool: CandidatePool,
        shortlist_: list[Candidate],
        *,
        min_papers: int,
        max_papers: int,
        today: date,
    ) -> SurveySelection:
        payload = {
            "topic": neutralize_prompt_tags(topic),
            "scope": scope.topic,
            "lineages": "\n".join(
                f"- {lineage.id}: {lineage.name} — {lineage.description}"
                for lineage in scope.lineages
            ),
            "candidates": neutralize_prompt_tags(
                shortlist_digest(shortlist_, pool.edges, today)
            ),
            "min_papers": min(min_papers, len(shortlist_)),
            "max_papers": max_papers,
        }

        def parse(raw: str) -> tuple[SurveySelection, list[str]]:
            selection = parse_tagged_json(raw, "selection", SurveySelection)
            problems = check_selection(
                selection, shortlist_, min_papers=min_papers, max_papers=max_papers
            )
            return selection, problems

        return await self._ask(self.selection_chain, payload, parse, tag="selection")


def _range_text(time_range: TimeRange) -> str:
    if time_range.since is None and time_range.until is None:
        return "none"
    return f"{time_range.since or 'open'} to {time_range.until or 'open'}"


class SurveyWriter(RetryableBase):
    """The write stage's finalization call (description and taxonomy of
    the finished post)."""

    def __init__(
        self,
        *,
        finalization_model_id: LanguageModelId,
        boto_session: boto3.Session,
        language: str,
        thinking_effort: str,
        callbacks: list[Any] | None = None,
        llm_factory: BedrockLanguageModelFactory | None = None,
    ) -> None:
        self.callbacks = callbacks or []
        self.language = language
        self.notes: list[str] = []
        factory = llm_factory or BedrockLanguageModelFactory(boto_session=boto_session)
        model = factory.get_model(
            finalization_model_id,
            temperature=0.0,
            callbacks=self.callbacks or None,
            enable_thinking=False,
            thinking_effort=thinking_effort,
        )
        self.finalization_chain: Runnable = (
            SurveyFinalizationPrompt.get_prompt()
            | model
            | HTMLTagOutputParser(
                tag_names=SurveyFinalizationPrompt.output_variables or []
            )
        )

    async def finalize(
        self,
        title: str,
        body: str,
        *,
        category_options: Sequence[str],
        controlled_tags: Sequence[str],
    ) -> dict[str, str]:
        """Topic, tags and the search description, checked by the post
        validator's description rules with one repair round. A description
        the validator would reject (markup, not longer than the title) is
        dropped: the blog then derives one from the first paragraph."""
        payload = {
            "title": title,
            "outline": neutralize_prompt_tags(digest_sections([body])),
            "category_options": ", ".join(category_options),
            "controlled_tags": ", ".join(controlled_tags),
            "language": self.language,
            "repair_note": "",
        }
        result: dict[str, str] = {}
        problems: list[str] = []
        description = ""
        for attempt in range(2):
            result = await self._finalize(payload)
            description = fit_description(
                " ".join((result.get("description") or "").split())
            )
            result["description"] = description
            if not (
                problems := description_problems(description, body)
                + controlled_tag_problems(
                    result.get("controlled_tags"), controlled_tags
                )
            ):
                return result
            logger.warning(
                "Survey finalization answer unusable (attempt %d): %s",
                attempt + 1,
                problems,
            )
            payload["repair_note"] = (
                "Your previous answer could not be used: "
                + "; ".join(problems)
                + ". Answer all four parts again."
            )
        self.notes.append(
            "Survey description still breaks the rules after a repair: "
            + "; ".join(problems)
        )
        if description_markup(description) or len(description) <= len(title):
            result["description"] = ""
        return result

    @RetryableBase._retry("survey_finalization")
    @staged("finalize")
    async def _finalize(self, payload: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = await self.finalization_chain.ainvoke(payload)
        return result
