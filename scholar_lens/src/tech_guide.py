"""Technical guide / tutorial generator.

Given a list of URLs (library/framework/platform docs), this agent:
1. Researches the URLs (+ in-scope sub-pages + optional web search) into a corpus.
2. Gates on relevance — refuses to write if the sources are not technical
   developer documentation (raises :class:`NotTechnicalContentError`).
3. Drafts a synopsis (ordered outline) grounded in the corpus.
4. Writes the guide section-by-section in Markdown with code, tables, math and
   images sourced only from the gathered material.

It reuses the project's Bedrock factory, prompt conventions, and the
:mod:`web_research` primitives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import boto3
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from .constants import LanguageModelId
from .logger import logger
from .prompts import (
    TechGuideRelevancePrompt,
    TechGuideSectionPrompt,
    TechGuideSynopsisPrompt,
)
from .utils import (
    BedrockLanguageModelFactory,
    HTMLTagOutputParser,
    RetryableBase,
    measure_execution_time,
)
from .web_research import ResearchCorpus, WebResearcher

DEFAULT_LANGUAGE: str = "Korean"
_SOURCE_CHAR_BUDGET: int = 120_000
_MAX_SECTIONS: int = 12


class NotTechnicalContentError(Exception):
    """Raised when the supplied URLs are not technical documentation."""


@dataclass
class TechGuide:
    topic: str
    synopsis: str
    body: str
    image_urls: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)


class TechGuideGenerator(RetryableBase):
    """Generates a self-study technical guide from a list of source URLs."""

    def __init__(
        self,
        relevance_model_id: LanguageModelId,
        synopsis_model_id: LanguageModelId,
        writing_model_id: LanguageModelId,
        boto_session: boto3.Session,
        *,
        researcher: WebResearcher | None = None,
        language: str = DEFAULT_LANGUAGE,
        enable_thinking: bool = False,
        max_sections: int = _MAX_SECTIONS,
    ) -> None:
        self.language = language
        self.max_sections = max_sections
        self.researcher = researcher or WebResearcher()
        self.llm_factory = BedrockLanguageModelFactory(boto_session=boto_session)

        self.relevance_chain: Runnable = (
            TechGuideRelevancePrompt.get_prompt()
            | self.llm_factory.get_model(relevance_model_id, temperature=0.0)
            | HTMLTagOutputParser(tag_names=TechGuideRelevancePrompt.output_variables)
        )
        self.synopsis_chain: Runnable = (
            TechGuideSynopsisPrompt.get_prompt()
            | self.llm_factory.get_model(
                synopsis_model_id, temperature=0.0, enable_thinking=enable_thinking
            )
            | HTMLTagOutputParser(tag_names=TechGuideSynopsisPrompt.output_variables)
        )
        self.section_chain: Runnable = (
            TechGuideSectionPrompt.get_prompt()
            | self.llm_factory.get_model(
                writing_model_id,
                temperature=0.0,
                enable_thinking=enable_thinking,
                supports_1m_context_window=True,
            )
            | HTMLTagOutputParser(tag_names=TechGuideSectionPrompt.output_variables)
        )
        # Lightweight chain reused if structured parsing of a section fails.
        self._section_str_chain: Runnable = (
            TechGuideSectionPrompt.get_prompt()
            | self.llm_factory.get_model(writing_model_id, temperature=0.0)
            | StrOutputParser()
        )

    @measure_execution_time
    async def generate(
        self,
        urls: list[str],
        *,
        discover_subpages: bool = True,
        search_queries: list[str] | None = None,
    ) -> TechGuide:
        if not urls:
            raise ValueError("At least one source URL is required.")

        corpus = self.researcher.research(
            urls,
            discover_subpages=discover_subpages,
            search_queries=search_queries,
        )
        if not corpus.pages:
            raise NotTechnicalContentError(
                "None of the supplied URLs returned readable content."
            )

        topic = await self._assert_relevant(corpus)
        synopsis = await self._draft_synopsis(topic, corpus)
        body = await self._write_sections(topic, synopsis, corpus)

        return TechGuide(
            topic=topic,
            synopsis=synopsis,
            body=body,
            image_urls=corpus.image_urls,
            source_urls=[p.url for p in corpus.pages],
        )

    async def _assert_relevant(self, corpus: ResearchCorpus) -> str:
        sources = self._sources_digest(corpus)
        result = await self.relevance_chain.ainvoke({"sources": sources})
        is_relevant = result.get("is_relevant", "no").strip().lower().startswith("y")
        reason = result.get("reason", "").strip()
        if not is_relevant:
            raise NotTechnicalContentError(
                f"Supplied URLs are not technical documentation: {reason}"
            )
        topic = result.get("topic", "").strip()
        if not topic or topic.upper() == "N/A":
            raise NotTechnicalContentError(
                "Could not determine a technical topic from the sources."
            )
        logger.info("Relevance gate passed. Topic: '%s'", topic)
        return topic

    async def _draft_synopsis(self, topic: str, corpus: ResearchCorpus) -> str:
        result = await self.synopsis_chain.ainvoke(
            {
                "topic": topic,
                "sources": corpus.combined_text(_SOURCE_CHAR_BUDGET),
                "search_results": self._search_digest(corpus),
                "language": self.language,
            }
        )
        synopsis = result.get("synopsis", "").strip()
        if not synopsis:
            raise ValueError("Synopsis generation returned empty output.")
        return synopsis

    async def _write_sections(
        self, topic: str, synopsis: str, corpus: ResearchCorpus
    ) -> str:
        sections = self._parse_synopsis_sections(synopsis)
        sources = corpus.combined_text(_SOURCE_CHAR_BUDGET)
        available_images = "\n".join(corpus.image_urls) or "(none)"

        written: list[str] = []
        for index, section in enumerate(sections, start=1):
            logger.info("Writing section %d/%d: %s", index, len(sections), section)
            markdown = await self._write_one_section(
                topic=topic,
                synopsis=synopsis,
                section=section,
                previous_sections="\n\n".join(written),
                sources=sources,
                available_images=available_images,
            )
            if markdown:
                written.append(markdown)
        return "\n\n".join(written)

    @RetryableBase._retry("tech_guide_section")
    async def _write_one_section(
        self,
        *,
        topic: str,
        synopsis: str,
        section: str,
        previous_sections: str,
        sources: str,
        available_images: str,
    ) -> str:
        payload = {
            "topic": topic,
            "synopsis": synopsis,
            "section": section,
            "previous_sections": previous_sections,
            "sources": sources,
            "available_images": available_images,
            "language": self.language,
        }
        result = await self.section_chain.ainvoke(payload)
        markdown = (result or {}).get("section_markdown", "").strip()
        if markdown:
            return markdown
        # Fallback: model omitted the wrapper tag — use raw text.
        raw = (await self._section_str_chain.ainvoke(payload)).strip()
        match = re.search(r"<section_markdown>(.*?)</section_markdown>", raw, re.DOTALL)
        return (match.group(1) if match else raw).strip()

    def _parse_synopsis_sections(self, synopsis: str) -> list[str]:
        sections: list[str] = []
        for line in synopsis.splitlines():
            stripped = line.strip()
            if re.match(r"^(\d+[.)]|[-*•])\s+", stripped):
                sections.append(re.sub(r"^(\d+[.)]|[-*•])\s+", "", stripped))
        if not sections:
            # Fall back to non-empty lines if the model didn't number them.
            sections = [ln.strip() for ln in synopsis.splitlines() if ln.strip()]
        return sections[: self.max_sections]

    @staticmethod
    def _sources_digest(corpus: ResearchCorpus, max_chars: int = 6000) -> str:
        lines = []
        for page in corpus.pages:
            snippet = page.text[:500].replace("\n", " ")
            lines.append(f"- {page.url}\n  {page.title}\n  {snippet}")
        return "\n".join(lines)[:max_chars]

    @staticmethod
    def _search_digest(corpus: ResearchCorpus) -> str:
        if not corpus.search_results:
            return "(no web search results)"
        return "\n".join(
            f"- [{r.title}]({r.url}): {r.description}" for r in corpus.search_results
        )
