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
from typing import Any

import boto3
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from .constants import LanguageModelId
from .logger import logger
from .metrics import TokenUsageTracker
from .prompts import (
    BasePrompt,
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
    parse_quality_score,
)
from .web_research import ResearchCorpus, WebResearcher, neutralize_prompt_tags

DEFAULT_LANGUAGE: str = "Korean"
# Headroom reserved from the writer model's window when fitting the source
# corpus, leaving room for the prompt template, the model's output, and the
# previous-sections context that grows as sections are written.
_SECTION_CONTEXT_RESERVE_TOKENS: int = 80_000
# Reserve used when bounding the accumulated previous-sections context. The
# writer is a 1M-window model, so this caps that context at ~150k tokens
# (1M - 850k) — enough for dedup/flow without crowding out sources + output.
_PREVIOUS_SECTIONS_RESERVE_TOKENS: int = 850_000
# Upper bound on guide sections. The synopsis is told this budget so it plans a
# self-contained guide rather than over-proposing (which left the old vLLM guide
# promising chapters 13-26 that were never written).
_MAX_SECTIONS: int = 16
# Default number of web-search queries the research planner may propose.
_MAX_RESEARCH_QUERIES: int = 6
# Default number of top search-result pages fetched into the corpus so the
# gathered material reaches the writer (not just the outline planner).
_FETCH_TOP_RESULTS: int = 4
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
    """One outline entry: title, concern area, writing depth, and visual hint."""

    title: str
    area: str = ""
    depth: str = _DEFAULT_DEPTH
    visuals: str = ""

    @property
    def depth_directive(self) -> str:
        directive = _DEPTH_DIRECTIVES.get(self.depth, _DEPTH_DIRECTIVES[_DEFAULT_DEPTH])
        if self.visuals and self.visuals != "none":
            directive += (
                f"\nPlanned visual aid for this section: {self.visuals}. Include it "
                f"if it genuinely helps the reader."
            )
        return directive


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
        evaluation_model_id: LanguageModelId,
        boto_session: boto3.Session,
        *,
        researcher: WebResearcher | None = None,
        language: str = DEFAULT_LANGUAGE,
        enable_thinking: bool = False,
        evaluator_enable_thinking: bool | None = None,
        thinking_effort: str = "medium",
        max_sections: int = _MAX_SECTIONS,
        verify_grounding: bool = True,
        auto_research: bool = True,
        max_research_queries: int = _MAX_RESEARCH_QUERIES,
        fetch_top_results: int = _FETCH_TOP_RESULTS,
        min_quality_score: int = _MIN_QUALITY_SCORE,
        max_revision_attempts: int = _MAX_REVISION_ATTEMPTS,
        max_total_tokens: int | None = None,
        callbacks: list[Any] | None = None,
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
        self.fetch_top_results = fetch_top_results
        self.min_quality_score = min_quality_score
        self.max_revision_attempts = max_revision_attempts
        # Hard total-token ceiling for one guide run (None = no limit). The
        # evaluate-and-revise + grounding loop multiplies per-section calls, so
        # this guards against runaway cost the way the review pipeline does.
        self.max_total_tokens = max_total_tokens
        self.callbacks = callbacks or []
        self._token_tracker = next(
            (cb for cb in self.callbacks if isinstance(cb, TokenUsageTracker)), None
        )
        self.researcher = researcher or WebResearcher()
        self.llm_factory = BedrockLanguageModelFactory(boto_session=boto_session)
        self.synopsis_model_id = synopsis_model_id
        self.writing_model_id = writing_model_id
        self.evaluation_model_id = evaluation_model_id

        # The relevance gate and research planner are cheap and reuse the
        # relevance model tier (no thinking / 1M context).
        self.relevance_chain = self._build_chain(
            TechGuideRelevancePrompt, relevance_model_id
        )
        self.research_plan_chain = self._build_chain(
            TechGuideResearchPlanPrompt, relevance_model_id
        )
        self.synopsis_chain = self._build_chain(
            TechGuideSynopsisPrompt,
            synopsis_model_id,
            enable_thinking=enable_thinking,
            thinking_effort=thinking_effort,
            supports_1m_context_window=True,
        )
        self.section_chain = self._build_chain(
            TechGuideSectionPrompt,
            writing_model_id,
            enable_thinking=enable_thinking,
            thinking_effort=thinking_effort,
            supports_1m_context_window=True,
        )
        # Lightweight chain reused if structured parsing of a section fails.
        self._section_str_chain: Runnable = (
            TechGuideSectionPrompt.get_prompt()
            | self.llm_factory.get_model(
                writing_model_id,
                temperature=0.0,
                callbacks=self.callbacks or None,
            )
            | StrOutputParser()
        )
        # Evaluation pass: scores a drafted section 0-100 and returns feedback,
        # driving a score-and-revise loop (the review pipeline's reflect node).
        # Uses its own model (Opus 4.8 by default) rather than the synopsis model
        # so scoring stays well-calibrated independent of the drafting tier, and
        # its own thinking flag (defaults to the writer's when unset) so the
        # evaluator can think independently of drafting — parity with the review
        # pipeline's separate reflector/synthesizer thinking flags.
        evaluator_thinking = (
            enable_thinking
            if evaluator_enable_thinking is None
            else evaluator_enable_thinking
        )
        self.evaluation_chain = self._build_chain(
            TechGuideEvaluationPrompt,
            evaluation_model_id,
            enable_thinking=evaluator_thinking,
            thinking_effort=thinking_effort,
            supports_1m_context_window=True,
        )
        # Fact-checking pass: rewrites each section to drop ungrounded claims.
        self.grounding_chain = self._build_chain(
            TechGuideGroundingPrompt,
            writing_model_id,
            enable_thinking=enable_thinking,
            thinking_effort=thinking_effort,
            supports_1m_context_window=True,
        )

    def _build_chain(
        self,
        prompt_cls: type[BasePrompt],
        model_id: LanguageModelId,
        **model_kwargs: Any,
    ) -> Runnable:
        """Prompt | model | HTMLTagOutputParser, with shared callbacks attached.

        Callbacks carry the TokenUsageTracker so every chain's usage counts
        toward the run's token budget.
        """
        model = self.llm_factory.get_model(
            model_id,
            temperature=0.0,
            callbacks=self.callbacks or None,
            **model_kwargs,
        )
        return (
            prompt_cls.get_prompt()
            | model
            | HTMLTagOutputParser(tag_names=prompt_cls.output_variables or [])
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
        # be a translation of a single doc page. Skipped (no wasted LLM call) when
        # the search backend can't actually return results.
        can_search = self.researcher.search_provider.supports_search
        planned_topic: str | None = None
        queries = list(search_queries or [])
        if not queries and self.auto_research and can_search:
            planned_topic, queries = await self._plan_research(corpus)
        if queries and can_search:
            # Fetch the top results into the corpus so the gathered material
            # actually reaches the writer, not just the outline planner.
            self.researcher.run_searches(
                corpus, queries, fetch_top=self.fetch_top_results
            )
        elif queries:
            logger.info(
                "Search backend has no results; skipping %d queries.", len(queries)
            )

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
            # Bound the accumulated previous-sections context: it grows every
            # section and, with the fitted sources, could otherwise overflow the
            # window on a long guide. fit_text keeps the longest prefix — the
            # early sections where foundational concepts are defined, which is
            # what the dedup check most needs.
            previous = self.llm_factory.fit_text(
                self.writing_model_id,
                "\n\n".join(written),
                reserve_tokens=_PREVIOUS_SECTIONS_RESERVE_TOKENS,
                label="previous sections",
            )
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
            # Skip the expensive evaluate/revise + grounding passes once the
            # token budget is spent — still WRITE every section, just without
            # the optional refinement, so the guide is complete rather than cut.
            if not self._budget_exhausted():
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

    def _budget_exhausted(self) -> bool:
        """Whether the run has hit its total-token ceiling (if one is set)."""
        if self._token_tracker is None or not self.max_total_tokens:
            return False
        if self._token_tracker.total_tokens >= self.max_total_tokens:
            logger.warning(
                "Token budget reached (%d >= %d); writing remaining sections "
                "without evaluate/revise/grounding.",
                self._token_tracker.total_tokens,
                self.max_total_tokens,
            )
            return True
        return False

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

        Mirrors the review pipeline's reflect loop, but keeps the BEST-scoring
        draft: every revision is re-scored, and a revision is only kept if it
        beats the best so far — so a feedback-driven rewrite can never make the
        section worse than the draft it replaced.
        """
        if self.max_revision_attempts <= 0:
            # No revision budget: skip evaluation entirely (no wasted LLM call).
            return markdown
        best_markdown = markdown
        best_score, feedback = await self._evaluate_section(
            topic=topic,
            planned=planned,
            section_label=section_label,
            markdown=markdown,
            total_sections=total_sections,
            previous_sections=previous_sections,
        )
        for attempt in range(1, self.max_revision_attempts + 1):
            if best_score >= self.min_quality_score or not feedback:
                logger.info(
                    "Section %d scored %d (>= %d or no feedback) - accepting.",
                    section_number,
                    best_score,
                    self.min_quality_score,
                )
                return best_markdown
            logger.info(
                "Section %d scored %d (< %d) - revising (attempt %d/%d).",
                section_number,
                best_score,
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
            if not revised:
                break
            score, feedback = await self._evaluate_section(
                topic=topic,
                planned=planned,
                section_label=section_label,
                markdown=revised,
                total_sections=total_sections,
                previous_sections=previous_sections,
            )
            # Keep the revision only if it scored higher than the best so far;
            # otherwise discard it (a rewrite can regress).
            if score > best_score:
                best_markdown, best_score = revised, score
            else:
                logger.info(
                    "Section %d revision scored %d (<= best %d) - keeping prior draft.",
                    section_number,
                    score,
                    best_score,
                )
        return best_markdown

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
        score = parse_quality_score((result or {}).get("quality_score"))
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
            area, depth, visuals, title = self._extract_area_depth(body)
            if title:
                sections.append(
                    PlannedSection(title=title, area=area, depth=depth, visuals=visuals)
                )
        if not sections:
            # Fall back to non-empty lines if the model didn't number them.
            sections = [
                PlannedSection(title=ln.strip())
                for ln in synopsis.splitlines()
                if ln.strip()
            ]
        return sections[: self.max_sections] if apply_cap else sections

    @staticmethod
    def _extract_area_depth(body: str) -> tuple[str, str, str, str]:
        """Pull leading [AREA] [depth] tags and a trailing visuals hint off a line.

        Returns (area, depth, visuals, title). Tags MUST be bracketed (the
        synopsis prompt always brackets them); this avoids mis-reading a title
        word that happens to match a keyword (e.g. "Usage — ..."). They may
        appear in either order; the remaining text (minus a trailing
        "(visuals: ...)" hint) is the title."""
        area = ""
        depth = _DEFAULT_DEPTH
        # Consume up to two leading [..] bracketed tokens. An unrecognised token
        # (typo'd / hallucinated tag) is still consumed so it never leaks into
        # the rendered heading.
        for _ in range(2):
            match = re.match(r"^\[([A-Za-z]+)\]\s*", body)
            if not match:
                break
            token = match.group(1)
            if token.upper() in _VALID_AREAS:
                area = token.upper()
            elif token.lower() in _DEPTH_DIRECTIVES:
                depth = token.lower()
            body = body[match.end() :]
        # Capture the planned visual aid, e.g. "(visuals: table)".
        visuals = ""
        if vmatch := re.search(r"\(visuals?:\s*([^)]*)\)", body, re.IGNORECASE):
            visuals = vmatch.group(1).strip().lower()
        # Title is everything up to the description separator / visuals hint.
        # Only an em/en dash separates title from description — a spaced ASCII
        # hyphen is left intact so titles like "Using gRPC - the basics" survive.
        title = re.split(r"\s+[—–]\s+|\(visuals?:", body, maxsplit=1)[0].strip()
        return area, depth, visuals, title or body.strip()

    @staticmethod
    def _parse_queries(raw: str) -> list[str]:
        queries: list[str] = []
        seen: set[str] = set()
        for line in raw.splitlines():
            stripped = re.sub(r"^(\d+[.)]|[-*•])\s+", "", line.strip()).strip()
            # Skip blanks, tag lines, and punctuation-only lines like the prompt's
            # literal "..." placeholder (which models often echo verbatim).
            if not stripped or stripped.startswith("<"):
                continue
            if not re.search(r"[A-Za-z0-9]", stripped):
                continue
            key = stripped.lower()
            if key in seen:  # dedup case-insensitively to avoid paid dup searches
                continue
            seen.add(key)
            queries.append(stripped)
        return queries

    @staticmethod
    def _sources_digest(corpus: ResearchCorpus, max_chars: int = 6000) -> str:
        lines = []
        for page in corpus.pages:
            snippet = page.text[:500].replace("\n", " ")
            lines.append(f"- {page.url}\n  {page.title}\n  {snippet}")
        # Untrusted page text → defang prompt-fence tags before interpolation.
        return neutralize_prompt_tags("\n".join(lines)[:max_chars])

    @staticmethod
    def _search_digest(corpus: ResearchCorpus) -> str:
        if not corpus.search_results:
            return "(no web search results)"
        return neutralize_prompt_tags(
            "\n".join(
                f"- [{r.title}]({r.url}): {r.description}"
                for r in corpus.search_results
            )
        )
