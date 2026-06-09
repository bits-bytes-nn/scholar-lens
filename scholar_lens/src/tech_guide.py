"""Technical guide / tutorial generator.

Given a list of URLs (library/framework/platform docs), this agent:
1. Plans deep research — derives the topic and a set of web-search queries from
   the seed docs, then researches the seed URLs (+ in-scope sub-pages + the
   planned/explicit search queries) into a corpus so the guide draws on broad,
   complementary material rather than translating a single page.
2. Gates on relevance — refuses to write if the sources are not technical
   developer documentation (raises :class:`NotTechnicalContentError`).
3. Plans a structured synopsis: each section is tagged with a concern area
   (concept / detail / usage / application) and a depth (deep / standard /
   brief), so important parts go deep and peripheral parts stay short.
4. Writes the guide section-by-section in Markdown with code, tables, math and
   source images, honouring each section's depth directive and avoiding
   duplication of earlier sections.
5. Evaluates each drafted section (a 0-100 score + feedback) and revises it
   below a quality threshold — the review pipeline's reflect-and-revise loop
   applied to guides — then optionally fact-checks it against the sources.

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
    TechGuideEvaluationPrompt,
    TechGuideGroundingPrompt,
    TechGuideRelevancePrompt,
    TechGuideResearchPlanPrompt,
    TechGuideSectionPrompt,
    TechGuideSynopsisPrompt,
)
from .utils import (
    BedrockLanguageModelFactory,
    HTMLTagOutputParser,
    RetryableBase,
    is_affirmative,
    is_placeholder,
    measure_execution_time,
)
from .web_research import ResearchCorpus, WebResearcher

DEFAULT_LANGUAGE: str = "Korean"
# Headroom reserved from the writer model's window when fitting the source
# corpus, leaving room for the prompt template, the model's output, and the
# previous-sections context that grows as sections are written.
_SECTION_CONTEXT_RESERVE_TOKENS: int = 80_000
# Upper bound on guide sections. The synopsis is told this budget so it plans a
# self-contained guide rather than over-proposing (which left the old vLLM guide
# promising chapters 13-26 that were never written).
_MAX_SECTIONS: int = 16
# Default number of web-search queries the research planner may propose.
_MAX_RESEARCH_QUERIES: int = 6
# Section quality gate (mirrors the review pipeline's reflect loop).
_MIN_QUALITY_SCORE: int = 75
_MAX_REVISION_ATTEMPTS: int = 2

# Per-depth writing directive injected into the section prompt. Keeps the
# "depth is set per section, not by a fixed length" principle in one place.
_DEPTH_DIRECTIVES: dict[str, str] = {
    "deep": (
        "DEEP section: this is a conceptually central or hard topic. Explain it "
        "thoroughly across multiple substantial paragraphs, build intuition, and "
        "include worked examples / code / a table where they aid understanding. "
        "Do not gloss over the hard parts — this is where the reader should learn "
        "the most."
    ),
    "standard": (
        "STANDARD section: normal importance. Cover it solidly and clearly, but "
        "do not pad — be complete without exhaustive detail."
    ),
    "brief": (
        "BRIEF section: an easy, routine, or peripheral topic. Keep it short — a "
        "few sentences or a small example. Do not over-explain well-known basics."
    ),
}
_DEFAULT_DEPTH: str = "standard"
_VALID_AREAS: frozenset[str] = frozenset({"CONCEPT", "DETAIL", "USAGE", "APPLICATION"})


class NotTechnicalContentError(Exception):
    """Raised when the supplied URLs are not technical documentation."""


@dataclass
class PlannedSection:
    """One outline entry: the title plus its concern area and writing depth."""

    title: str
    area: str = ""
    depth: str = _DEFAULT_DEPTH

    @property
    def depth_directive(self) -> str:
        return _DEPTH_DIRECTIVES.get(self.depth, _DEPTH_DIRECTIVES[_DEFAULT_DEPTH])


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
        thinking_effort: str = "medium",
        max_sections: int = _MAX_SECTIONS,
        verify_grounding: bool = True,
        auto_research: bool = True,
        max_research_queries: int = _MAX_RESEARCH_QUERIES,
        min_quality_score: int = _MIN_QUALITY_SCORE,
        max_revision_attempts: int = _MAX_REVISION_ATTEMPTS,
    ) -> None:
        self.language = language
        self.max_sections = max_sections
        # When True, each drafted section is fact-checked against the sources to
        # strip ungrounded claims (the review pipeline's reflect-and-revise idea
        # applied to guides). Cheap insurance against hallucinated APIs/flags.
        self.verify_grounding = verify_grounding
        # When True (and no explicit queries are supplied), the planner derives
        # web-search queries from the seed docs so the corpus is broadened beyond
        # the seed page before writing.
        self.auto_research = auto_research
        self.max_research_queries = max_research_queries
        self.min_quality_score = min_quality_score
        self.max_revision_attempts = max_revision_attempts
        self.researcher = researcher or WebResearcher()
        self.llm_factory = BedrockLanguageModelFactory(boto_session=boto_session)

        self.relevance_chain: Runnable = (
            TechGuideRelevancePrompt.get_prompt()
            | self.llm_factory.get_model(relevance_model_id, temperature=0.0)
            | HTMLTagOutputParser(tag_names=TechGuideRelevancePrompt.output_variables)
        )
        # The research planner is cheap and reuses the relevance model tier.
        self.research_plan_chain: Runnable = (
            TechGuideResearchPlanPrompt.get_prompt()
            | self.llm_factory.get_model(relevance_model_id, temperature=0.0)
            | HTMLTagOutputParser(
                tag_names=TechGuideResearchPlanPrompt.output_variables
            )
        )
        self.synopsis_model_id = synopsis_model_id
        self.writing_model_id = writing_model_id
        self.synopsis_chain: Runnable = (
            TechGuideSynopsisPrompt.get_prompt()
            | self.llm_factory.get_model(
                synopsis_model_id,
                temperature=0.0,
                enable_thinking=enable_thinking,
                thinking_effort=thinking_effort,
                supports_1m_context_window=True,
            )
            | HTMLTagOutputParser(tag_names=TechGuideSynopsisPrompt.output_variables)
        )
        self.section_chain: Runnable = (
            TechGuideSectionPrompt.get_prompt()
            | self.llm_factory.get_model(
                writing_model_id,
                temperature=0.0,
                enable_thinking=enable_thinking,
                thinking_effort=thinking_effort,
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
        # Evaluation pass: scores a drafted section 0-100 and returns feedback,
        # driving a score-and-revise loop (the review pipeline's reflect node).
        self.evaluation_chain: Runnable = (
            TechGuideEvaluationPrompt.get_prompt()
            | self.llm_factory.get_model(
                synopsis_model_id,
                temperature=0.0,
                enable_thinking=enable_thinking,
                thinking_effort=thinking_effort,
                supports_1m_context_window=True,
            )
            | HTMLTagOutputParser(tag_names=TechGuideEvaluationPrompt.output_variables)
        )
        # Fact-checking pass: rewrites each section to drop ungrounded claims.
        self.grounding_chain: Runnable = (
            TechGuideGroundingPrompt.get_prompt()
            | self.llm_factory.get_model(
                writing_model_id,
                temperature=0.0,
                enable_thinking=enable_thinking,
                thinking_effort=thinking_effort,
                supports_1m_context_window=True,
            )
            | HTMLTagOutputParser(tag_names=TechGuideGroundingPrompt.output_variables)
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

        # Fetch the seed URLs (+ in-scope sub-pages) once.
        corpus = self.researcher.research(
            urls,
            discover_subpages=discover_subpages,
            search_queries=None,
        )
        if not corpus.pages:
            raise NotTechnicalContentError(
                "None of the supplied URLs returned readable content."
            )

        # Deep-research planning: when no explicit queries are supplied, derive
        # search queries (and an early topic guess) from the seed pages so the
        # corpus is broadened with complementary material — the guide should not
        # be a translation of a single doc page.
        planned_topic: str | None = None
        queries = list(search_queries or [])
        if not queries and self.auto_research:
            planned_topic, queries = await self._plan_research(corpus)
        if queries:
            self.researcher.run_searches(corpus, queries)

        topic = await self._assert_relevant(corpus, fallback_topic=planned_topic)
        synopsis = await self._draft_synopsis(topic, corpus)
        body = await self._write_sections(topic, synopsis, corpus)

        return TechGuide(
            topic=topic,
            synopsis=synopsis,
            body=body,
            image_urls=corpus.image_urls,
            source_urls=[p.url for p in corpus.pages],
        )

    async def _plan_research(
        self, corpus: ResearchCorpus
    ) -> tuple[str | None, list[str]]:
        """Plan web-search queries from the seed corpus to broaden coverage.

        Best-effort: a planning failure degrades to researching only the
        supplied URLs, never aborting the run.
        """
        try:
            result = await self.research_plan_chain.ainvoke(
                {
                    "sources": self._sources_digest(corpus),
                    "max_queries": self.max_research_queries,
                }
            )
            topic = (result.get("topic") or "").strip()
            queries = self._parse_queries(result.get("queries", ""))
            logger.info(
                "Research plan: topic='%s', %d queries.",
                topic,
                len(queries),
            )
            return (topic or None), queries
        except Exception as e:  # noqa: BLE001 - planning is best-effort
            logger.warning("Research planning failed; using seed URLs only: %s", e)
            return None, []

    async def _assert_relevant(
        self, corpus: ResearchCorpus, *, fallback_topic: str | None = None
    ) -> str:
        sources = self._sources_digest(corpus)
        result = await self.relevance_chain.ainvoke({"sources": sources})
        is_relevant = is_affirmative(result.get("is_relevant"))
        reason = result.get("reason", "").strip()
        if not is_relevant:
            raise NotTechnicalContentError(
                f"Supplied URLs are not technical documentation: {reason}"
            )
        topic = result.get("topic", "").strip()
        if is_placeholder(topic):
            # Fall back to the research planner's topic guess before giving up.
            topic = (fallback_topic or "").strip()
        if is_placeholder(topic):
            raise NotTechnicalContentError(
                "Could not determine a technical topic from the sources."
            )
        logger.info("Relevance gate passed. Topic: '%s'", topic)
        return topic

    async def _draft_synopsis(self, topic: str, corpus: ResearchCorpus) -> str:
        result = await self.synopsis_chain.ainvoke(
            {
                "topic": topic,
                "sources": self.llm_factory.fit_text(
                    self.synopsis_model_id,
                    corpus.combined_text(),
                    label="synopsis sources",
                ),
                "search_results": self._search_digest(corpus),
                "max_sections": self.max_sections,
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
        all_sections = self._parse_synopsis_sections(synopsis, apply_cap=False)
        sections = all_sections[: self.max_sections]
        if len(all_sections) > len(sections):
            # Don't silently drop: the writer must NOT cross-reference sections
            # that won't exist (the cause of "see chapter 17" hallucinations).
            logger.warning(
                "Synopsis proposed %d sections; capping to max_sections=%d. "
                "Dropped: %s",
                len(all_sections),
                self.max_sections,
                [s.title for s in all_sections[self.max_sections :]],
            )
        # Fit the corpus to the writer model's window, reserving headroom for the
        # prompt plus the previous-sections context that accumulates per section.
        sources = self.llm_factory.fit_text(
            self.writing_model_id,
            corpus.combined_text(),
            reserve_tokens=_SECTION_CONTEXT_RESERVE_TOKENS,
            label="section sources",
        )
        available_images = "\n".join(corpus.image_urls) or "(none)"

        # Canonical numbered outline of ONLY the sections that will be written,
        # so the model never sees (and never references) dropped section numbers.
        final_outline = "\n".join(
            f"{i}. {s.title}" for i, s in enumerate(sections, start=1)
        )
        total = len(sections)

        written: list[str] = []
        for index, planned in enumerate(sections, start=1):
            logger.info("Writing section %d/%d: %s", index, total, planned.title)
            previous = "\n\n".join(written)
            section_label = f"{index}. {planned.title}"
            markdown = await self._write_one_section(
                topic=topic,
                synopsis=final_outline,
                section=section_label,
                section_number=index,
                total_sections=total,
                previous_sections=previous,
                sources=sources,
                available_images=available_images,
                depth_directive=planned.depth_directive,
            )
            if not markdown:
                continue
            markdown = await self._evaluate_and_revise(
                topic=topic,
                planned=planned,
                section_label=section_label,
                markdown=markdown,
                section_number=index,
                total_sections=total,
                previous_sections=previous,
                sources=sources,
                available_images=available_images,
                final_outline=final_outline,
            )
            if self.verify_grounding:
                markdown = await self._ground_section(markdown, sources, total)
            written.append(markdown)
        return "\n\n".join(written)

    async def _evaluate_and_revise(
        self,
        *,
        topic: str,
        planned: PlannedSection,
        section_label: str,
        markdown: str,
        section_number: int,
        total_sections: int,
        previous_sections: str,
        sources: str,
        available_images: str,
        final_outline: str,
    ) -> str:
        """Score a section and revise it while it stays below the threshold.

        Mirrors the review pipeline's reflect loop: evaluate -> if below the
        quality bar and attempts remain, re-write with the feedback appended to
        the depth directive; otherwise keep the best draft so far.
        """
        for attempt in range(1, self.max_revision_attempts + 1):
            score, feedback = await self._evaluate_section(
                topic=topic,
                planned=planned,
                section_label=section_label,
                markdown=markdown,
                total_sections=total_sections,
                previous_sections=previous_sections,
            )
            if score >= self.min_quality_score or not feedback:
                logger.info(
                    "Section %d scored %d (>= %d or no feedback) - accepting.",
                    section_number,
                    score,
                    self.min_quality_score,
                )
                return markdown
            logger.info(
                "Section %d scored %d (< %d) - revising (attempt %d/%d).",
                section_number,
                score,
                self.min_quality_score,
                attempt,
                self.max_revision_attempts,
            )
            revised = await self._write_one_section(
                topic=topic,
                synopsis=final_outline,
                section=section_label,
                section_number=section_number,
                total_sections=total_sections,
                previous_sections=previous_sections,
                sources=sources,
                available_images=available_images,
                depth_directive=(
                    f"{planned.depth_directive}\n\nREVISION FEEDBACK to address: "
                    f"{feedback}"
                ),
            )
            if revised:
                markdown = revised
        return markdown

    @RetryableBase._retry("tech_guide_evaluation")
    async def _evaluate_section(
        self,
        *,
        topic: str,
        planned: PlannedSection,
        section_label: str,
        markdown: str,
        total_sections: int,
        previous_sections: str,
    ) -> tuple[int, str]:
        """Return (quality_score 0-100, improvement_feedback) for a section."""
        result = await self.evaluation_chain.ainvoke(
            {
                "topic": topic,
                "section": markdown,
                "section_to_write": section_label,
                "depth_directive": planned.depth_directive,
                "previous_sections": previous_sections,
                "total_sections": total_sections,
                "language": self.language,
            }
        )
        score = self._parse_quality_score((result or {}).get("quality_score"))
        feedback = ((result or {}).get("improvement_feedback") or "").strip()
        return score, feedback

    @RetryableBase._retry("tech_guide_grounding")
    async def _ground_section(
        self, section_markdown: str, sources: str, total_sections: int
    ) -> str:
        """Fact-check a drafted section, returning the cleaned version.

        Falls back to the original draft if the grounding pass returns nothing,
        so verification can only improve—never drop—a section.
        """
        result = await self.grounding_chain.ainvoke(
            {
                "section": section_markdown,
                "sources": sources,
                "total_sections": total_sections,
                "language": self.language,
            }
        )
        grounded = (result or {}).get("grounded_markdown", "").strip()
        return grounded or section_markdown

    @RetryableBase._retry("tech_guide_section")
    async def _write_one_section(
        self,
        *,
        topic: str,
        synopsis: str,
        section: str,
        section_number: int,
        total_sections: int,
        previous_sections: str,
        sources: str,
        available_images: str,
        depth_directive: str,
    ) -> str:
        payload = {
            "topic": topic,
            "synopsis": synopsis,
            "section": section,
            "section_number": section_number,
            "total_sections": total_sections,
            "previous_sections": previous_sections,
            "sources": sources,
            "available_images": available_images,
            "depth_directive": depth_directive,
            "language": self.language,
        }
        result = await self.section_chain.ainvoke(payload)
        markdown = (result or {}).get("section_markdown", "").strip()
        if markdown:
            return markdown
        # Fallback: model omitted the wrapper tag — use raw text.
        raw = (await self._section_str_chain.ainvoke(payload)).strip()
        match = re.search(r"<section_markdown>(.*?)</section_markdown>", raw, re.DOTALL)
        text = match.group(1) if match else raw
        # Strip any stray wrapper tag fragments (e.g. an unclosed opening tag) so
        # raw XML never leaks into the published Markdown.
        text = re.sub(r"</?section_markdown>", "", text)
        return text.strip()

    def _parse_synopsis_sections(
        self, synopsis: str, *, apply_cap: bool = True
    ) -> list[PlannedSection]:
        """Parse the structured outline into PlannedSection entries.

        Each outline line looks like:
            1. [CONCEPT] [deep] Section Title — description (visuals: ...)
        The area/depth tags are optional; missing or unrecognised values fall
        back to a sensible default so a loosely-formatted synopsis still works.
        """
        sections: list[PlannedSection] = []
        for line in synopsis.splitlines():
            stripped = line.strip()
            if not re.match(r"^(\d+[.)]|[-*•])\s+", stripped):
                continue
            body = re.sub(r"^(\d+[.)]|[-*•])\s+", "", stripped)
            area, depth, title = self._extract_area_depth(body)
            if title:
                sections.append(PlannedSection(title=title, area=area, depth=depth))
        if not sections:
            # Fall back to non-empty lines if the model didn't number them.
            sections = [
                PlannedSection(title=ln.strip())
                for ln in synopsis.splitlines()
                if ln.strip()
            ]
        return sections[: self.max_sections] if apply_cap else sections

    @staticmethod
    def _extract_area_depth(body: str) -> tuple[str, str, str]:
        """Pull leading [AREA] [depth] tags off an outline line.

        Returns (area, depth, title). Tags MUST be bracketed (the synopsis
        prompt always brackets them); this avoids mis-reading a title word that
        happens to match a keyword (e.g. "Usage — ..."). They may appear in
        either order; the remaining text (minus a trailing "(visuals: ...)"
        hint) is the title."""
        area = ""
        depth = _DEFAULT_DEPTH
        # Consume up to two leading [..] bracketed tokens.
        for _ in range(2):
            match = re.match(r"^\[([A-Za-z]+)\]\s*", body)
            if not match:
                break
            token = match.group(1)
            if token.upper() in _VALID_AREAS:
                area = token.upper()
            elif token.lower() in _DEPTH_DIRECTIVES:
                depth = token.lower()
            else:
                break
            body = body[match.end() :]
        # Title is everything up to the description separator / visuals hint.
        title = re.split(r"\s+[—–-]\s+|\(visuals?:", body, maxsplit=1)[0].strip()
        return area, depth, title or body.strip()

    @staticmethod
    def _parse_queries(raw: str) -> list[str]:
        queries: list[str] = []
        for line in raw.splitlines():
            stripped = re.sub(r"^(\d+[.)]|[-*•])\s+", "", line.strip()).strip()
            if stripped and not stripped.startswith("<"):
                queries.append(stripped)
        return queries

    @staticmethod
    def _parse_quality_score(raw: object) -> int:
        """Parse the evaluator's 0-100 score robustly.

        Accepts a bare integer, a "85/100" form (takes 85, not 100), or junk;
        a non-numeric value falls back to 0, which forces a revision attempt
        rather than crashing the run.
        """
        text = str(raw if raw is not None else "").strip()
        try:
            return int(text)
        except ValueError:
            pass
        match = re.match(r"-?\d+", text)
        return int(match.group()) if match else 0

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
