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
   below a quality threshold — the review pipeline's evaluate-and-revise loop
   applied to guides — then optionally fact-checks it against the sources.

It reuses the project's Bedrock factory, prompt conventions, and the
:mod:`web_research` primitives.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import boto3
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from .constants import LanguageModelId
from .logger import logger
from .markdown_lint import close_open_fence
from .markdown_math import segments, without_fences
from .metrics import TokenBudgetGuard
from .prompt_safety import neutralize_prompt_tags
from .prompts import (
    BasePrompt,
    TechGuideEvaluationPrompt,
    TechGuideGroundingPrompt,
    TechGuideRelevancePrompt,
    TechGuideResearchPlanPrompt,
    TechGuideSectionPrompt,
    TechGuideSynopsisPrompt,
)
from .prompts.tech_guide import (
    DEPTH_CHARS,
    NON_KOREAN_CHAR_FACTOR,
    TechGuideTaxonomyPrompt,
)
from .review_plan import (
    added_slips,
    article_ledger,
    digest_sections,
    draft_rank,
    publish_blockers,
    rejected_revision_note,
    rule_issues,
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
from .web_research import (
    OfficialDocs,
    ResearchCorpus,
    WebResearcher,
    normalize_url,
    product_names,
)

DEFAULT_LANGUAGE: str = "Korean"
# Upper bound on guide sections. The synopsis is told this budget so it plans a
# self-contained guide rather than over-proposing (which left the old vLLM guide
# promising chapters 13-26 that were never written).
_MAX_SECTIONS: int = 10
# Source material handed to one section's writer/evaluator/grounding calls:
# the most relevant pages only (about 15k tokens), not the whole corpus.
_SECTION_SOURCE_CHARS: int = 60_000
# Official reference excerpt put ahead of a section's sources for grounding.
_REFERENCE_SOURCE_CHARS: int = 20_000
# Default number of web-search queries the research planner may propose.
_MAX_RESEARCH_QUERIES: int = 6
# Default number of top search-result pages fetched into the corpus so the
# gathered material reaches the writer (not just the outline planner).
_FETCH_TOP_RESULTS: int = 4
# Section quality gate (mirrors the review pipeline's evaluate loop).
_MIN_QUALITY_SCORE: int = 75
_MAX_REVISION_ATTEMPTS: int = 2
# Hard-abort multiple of max_total_tokens. The budget is two-tier: at 1x the
# guide soft-degrades (skips evaluate/revise/grounding but keeps writing so the
# guide is complete); past this multiple it hard-aborts like the review/summary
# pipelines, so a runaway can't keep spending on writer calls indefinitely.
_HARD_BUDGET_MULTIPLE: float = 1.5

# Section directives and the evaluator's length report use DEPTH_CHARS (one
# source with the synopsis budget).
_DEPTH_GUIDANCE: dict[str, str] = {
    "deep": (
        "a conceptually central or hard topic. Explain it thoroughly across "
        "several substantial paragraphs, build intuition, and include worked "
        "examples / code / a table where they aid understanding. Do not gloss "
        "over the hard parts — this is where the reader should learn the most."
    ),
    "standard": (
        "normal importance. Cover it solidly and clearly, but do not pad — be "
        "complete without exhaustive detail."
    ),
    "brief": (
        "an easy, routine, or peripheral topic. A few sentences or a small "
        "example. Do not over-explain well-known basics."
    ),
}


def _depth_target(depth: str, language: str) -> tuple[str, int]:
    """``(bound, characters)`` of prose for a section of ``depth``."""
    factor = 1.0 if language == "Korean" else NON_KOREAN_CHAR_FACTOR
    bound = "at most" if depth == "brief" else "about"
    return bound, round(DEPTH_CHARS[depth] * factor)


_DEFAULT_DEPTH: str = "standard"
GETTING_STARTED = "getting-started"
_GETTING_STARTED_SCOPE = (
    "This is a getting-started guide: explain internals only as far as the "
    "reader needs them to use and tune the tool (no simulators, derivations or "
    "from-scratch reimplementations)."
)


def prose_chars(markdown: str) -> int:
    """Characters of prose (whitespace collapsed, code blocks excluded)."""
    return len(" ".join(without_fences(markdown).split()))


def length_report(markdown: str, depth: str, language: str) -> str:
    """Measured length against the depth's target, for the evaluator."""
    depth = depth if depth in DEPTH_CHARS else _DEFAULT_DEPTH
    bound, target = _depth_target(depth, language)
    return (
        f"Measured: {prose_chars(markdown):,} characters of prose (code blocks "
        f"excluded). Target for a {depth} section: {bound} {target:,}."
    )


class NotTechnicalContentError(Exception):
    """Raised when the supplied URLs are not technical documentation."""


# Heading of the opening section list, per output language (others: English).
_OVERVIEW_HEADINGS = {"Korean": "이 가이드에서 다루는 것", "English": "In This Guide"}


def parse_section_output(raw: str) -> dict[str, str]:
    """Extract ``<section_markdown>`` from the writer output; if the wrapper is
    missing or unbalanced, keep the text and drop stray tag fragments."""
    match = re.search(r"<section_markdown>(.*?)</section_markdown>", raw, re.DOTALL)
    text = match.group(1) if match else raw
    return {"section_markdown": re.sub(r"</?section_markdown>", "", text).strip()}


@dataclass
class PlannedSection:
    """One outline entry: title, writing depth, and visual hint."""

    title: str
    depth: str = _DEFAULT_DEPTH
    visuals: str = ""
    description: str = ""

    def label(self, index: int) -> str:
        """Outline line shown to the writer/evaluator: number, title, scope."""
        line = f"{index}. {self.title}"
        return f"{line} — {self.description}" if self.description else line

    def depth_directive(self, language: str, guide_type: str = "") -> str:
        depth = self.depth if self.depth in DEPTH_CHARS else _DEFAULT_DEPTH
        bound, target = _depth_target(depth, language)
        directive = (
            f"{depth.upper()} section ({bound} {target:,} characters of prose, "
            f"code excluded): {_DEPTH_GUIDANCE[depth]}"
        )
        if guide_type == GETTING_STARTED:
            directive += f"\n{_GETTING_STARTED_SCOPE}"
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
    source_urls: list[str] = field(default_factory=list)
    # Degraded-output notes for the PR checklist and the job result.
    notes: list[str] = field(default_factory=list)
    # Text of every fetched page (taxonomy checks read it).
    source_text: str = field(default="", repr=False)
    # Images the writer was offered; any other image in the body is dropped.
    image_urls: list[str] = field(default_factory=list)


class TechGuideGenerator(RetryableBase, TokenBudgetGuard):
    """Generates a self-study technical guide from a list of source URLs."""

    # The requested docs (hosts and versions): the authority for flags and defaults.
    _guide_type: str = ""
    _official: OfficialDocs = OfficialDocs(frozenset(), frozenset())
    _source_urls: tuple[str, ...] = ()

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
        # strip ungrounded claims (the review pipeline's evaluate-and-revise idea
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
        self.callbacks = callbacks or []
        self._init_token_budget(self.callbacks, max_total_tokens)
        self.notes: list[str] = []
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
        self.taxonomy_chain = self._build_chain(
            TechGuideTaxonomyPrompt, relevance_model_id
        )
        self.synopsis_chain = self._build_chain(
            TechGuideSynopsisPrompt,
            synopsis_model_id,
            enable_thinking=enable_thinking,
            thinking_effort=thinking_effort,
            supports_1m_context_window=True,
        )
        # Parsed leniently (one call): a missing wrapper tag falls back to the
        # raw text instead of paying for a second full writer call.
        self.section_chain = (
            TechGuideSectionPrompt.get_prompt()
            | self.llm_factory.get_model(
                writing_model_id,
                temperature=0.0,
                callbacks=self.callbacks or None,
                enable_thinking=enable_thinking,
                thinking_effort=thinking_effort,
                supports_1m_context_window=True,
            )
            | StrOutputParser()
            | RunnableLambda(parse_section_output)
        )
        # Evaluation pass: scores a drafted section 0-100 and returns feedback,
        # driving a score-and-revise loop (the review pipeline's evaluate node).
        # Uses its own model (Opus 4.8 by default) rather than the synopsis model
        # so scoring stays well-calibrated independent of the drafting tier, and
        # its own thinking flag (defaults to the writer's when unset) so the
        # evaluator can think independently of drafting — parity with the review
        # pipeline's separate evaluator/synthesizer thinking flags.
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
        # The requested sites are the authority for flags, defaults and names.
        self._official = OfficialDocs.from_urls(urls)
        self._source_urls = tuple(urls)

        # Fetch the seed URLs (+ in-scope sub-pages) once. WebResearcher is
        # synchronous (blocking HTTP crawl); run it off the event loop.
        corpus = await asyncio.to_thread(
            self.researcher.research,
            urls,
            discover_subpages=discover_subpages,
            search_queries=None,
        )
        if not corpus.pages:
            raise NotTechnicalContentError(
                "None of the supplied URLs returned readable content."
            )
        # Where the requested pages actually landed (redirects, a pinned
        # version behind "stable") is official too.
        self._official = OfficialDocs.from_research(urls, corpus.pages)

        # Judge the seed pages alone: search results never redefine what the
        # user asked for, and a rejected input spends nothing on searches.
        gate_topic = await self._assert_relevant(corpus)

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
            await asyncio.to_thread(
                self.researcher.run_searches,
                corpus,
                queries,
                fetch_top=self.fetch_top_results,
            )
        elif queries:
            logger.info(
                "Search backend has no results; skipping %d queries.", len(queries)
            )

        topic = (
            planned_topic
            if planned_topic and not is_placeholder(planned_topic)
            else gate_topic
        )
        synopsis = await self._draft_synopsis(topic, corpus)
        body = await self._write_sections(topic, synopsis, corpus)
        body = self._overview(synopsis) + body
        unverified = unverified_identifiers(body, *official_reference(urls, corpus))
        if unverified:
            self.notes.append(
                "Flags/metrics the official pages fetched for this guide never "
                "mention; verify them against the current reference: "
                + ", ".join(unverified)
            )

        return TechGuide(
            topic=topic,
            synopsis=synopsis,
            body=body,
            source_urls=[p.url for p in corpus.pages],
            notes=list(self.notes),
            source_text="\n\n".join(p.text for p in corpus.pages),
            image_urls=list(corpus.image_urls),
        )

    @RetryableBase._retry("tech_guide_taxonomy")
    async def classify(
        self,
        guide: TechGuide,
        *,
        category_options: list[str],
        controlled_tags: list[str],
    ) -> dict[str, str]:
        """Propose the blog topic, tags and SEO title for a finished guide (the
        caller normalizes and validates them against the post conventions).
        Past the soft cap the finished guide is still classified: only the
        hard ceiling stops a guide whose sections are already paid for."""
        self._enforce_hard_token_budget()
        result: dict[str, str] = await self.taxonomy_chain.ainvoke(
            {
                "title": guide.topic,
                "synopsis": guide.synopsis,
                "category_options": ", ".join(category_options),
                "controlled_tags": ", ".join(controlled_tags),
            }
        )
        return result

    def _overview(self, synopsis: str) -> str:
        """Bulleted list of the planned section titles, opening the guide."""
        titles = [
            s.title for s in self._parse_synopsis_sections(synopsis, apply_cap=True)
        ]
        if len(titles) < 3:
            return ""
        bullets = "\n".join(f"- {t}" for t in titles)
        return f"## {_OVERVIEW_HEADINGS.get(self.language, 'In This Guide')}\n\n{bullets}\n\n"

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

    async def _assert_relevant(self, corpus: ResearchCorpus) -> str:
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
                "input_outline": input_outline(corpus, self._source_urls),
                "max_sections": self.max_sections,
                "language": self.language,
            }
        )
        synopsis = result.get("synopsis", "").strip()
        if not synopsis:
            raise ValueError("Synopsis generation returned empty output.")
        self._guide_type = (result.get("guide_type") or "").strip().lower()
        logger.info("Guide type: %s", self._guide_type or "?")
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
        available_images = "\n".join(corpus.image_urls) or "(none)"

        # Canonical numbered outline of ONLY the sections that will be written,
        # so the model never sees (and never references) dropped section numbers.
        final_outline = "\n".join(s.label(i) for i, s in enumerate(sections, start=1))
        total = len(sections)

        written: list[str] = []
        to_ground: list[tuple[int, str]] = []  # (index in written, its sources)
        for index, planned in enumerate(sections, start=1):
            # Hard ceiling: past _HARD_BUDGET_MULTIPLE × the budget, abort like
            # the review/summary pipelines rather than keep spending on the
            # (expensive) writer calls — the soft-degrade below is the earlier,
            # gentler tier.
            self._enforce_hard_token_budget()
            logger.info("Writing section %d/%d: %s", index, total, planned.title)
            # Bound the accumulated previous-sections context: it grows every
            # section and, with the fitted sources, could otherwise overflow the
            # window on a long guide. fit_text keeps the longest prefix — the
            # early sections where foundational concepts are defined, which is
            # what the dedup check most needs.
            # Earlier sections: the last two in full, a digest of the rest —
            # enough to avoid repeats without resending the whole guide.
            previous = "\n\n".join(
                part
                for part in (digest_sections(written[:-2]), "\n\n".join(written[-2:]))
                if part
            )
            section_label = planned.label(index)
            sources = corpus.relevant_text(
                f"{topic} {planned.title} {planned.description}",
                max_chars=_SECTION_SOURCE_CHARS,
                official=self._official,
            )
            markdown = await self._write_one_section(
                topic=topic,
                synopsis=final_outline,
                section=section_label,
                section_number=index,
                total_sections=total,
                previous_sections=previous,
                already_in_guide=article_ledger(written),
                sources=sources,
                available_images=available_images,
                depth_directive=planned.depth_directive(
                    self.language, self._guide_type
                ),
            )
            if not markdown:
                # A dropped section leaves a hole: `final_outline`/`total` (fixed
                # before writing) still list it, so later sections may reference a
                # number that won't render. Rare (writer returned nothing), but
                # log it so the gap is visible rather than silent.
                logger.warning(
                    "Section %d/%d (%s) produced no content; skipping. Later "
                    "sections may reference it — review the guide.",
                    index,
                    total,
                    planned.title,
                )
                self.notes.append(
                    f"Guide section '{planned.title}' produced no content and was "
                    "skipped; the overview still lists it."
                )
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
                    already_in_guide=article_ledger(written),
                    sources=sources,
                    available_images=available_images,
                    final_outline=final_outline,
                )
                if self.verify_grounding:
                    to_ground.append((len(written), sources))
            written.append(close_open_fence(markdown))
        if to_ground:
            await self._ground_sections(written, to_ground, corpus)
        return "\n\n".join(written)

    async def _ground_sections(
        self,
        written: list[str],
        to_ground: list[tuple[int, str]],
        corpus: ResearchCorpus,
    ) -> None:
        """Fact-check the drafted sections in place. Flags and metric names the
        official pages fetched so far never mention first pull the official
        reference pages that can settle them into each section's sources, so
        a stale flag is corrected instead of hedged."""
        official_text, products = official_reference(list(self._source_urls), corpus)
        unchecked = unverified_identifiers(
            "\n\n".join(written), official_text, products
        )
        reference = ResearchCorpus()
        if unchecked:
            reference.pages = await asyncio.to_thread(
                self.researcher.add_official_reference,
                corpus,
                self._official,
                unchecked,
                product=products[0] if products else "",
            )
        for index, sources in to_ground:
            self._enforce_hard_token_budget()
            if reference.pages:
                excerpt = reference.relevant_text(
                    " ".join(unchecked) + " " + written[index][:2_000],
                    max_chars=_REFERENCE_SOURCE_CHARS,
                    official=self._official,
                )
                sources = f"{excerpt}\n\n---\n\n{sources}"
            draft = written[index]
            grounded = close_open_fence(await self._ground_section(draft, sources))
            if added_slips(draft, grounded):
                # Grounding is the last pass; one retry if it broke a rule
                # (e.g. hedged a claim instead of removing it).
                retried = close_open_fence(await self._ground_section(draft, sources))
                if len(added_slips(draft, retried)) < len(added_slips(draft, grounded)):
                    grounded = retried
                if len(publish_blockers(grounded)) > len(publish_blockers(draft)):
                    # Both attempts broke something the validator rejects:
                    # keep the ungrounded draft rather than fail the post.
                    grounded = draft
            written[index] = grounded

    def _enforce_hard_token_budget(self) -> None:
        """Raise ``TokenBudgetExceeded`` once spend passes the hard ceiling.

        The soft tier (:meth:`_budget_exhausted`) only skips refinement; this is
        the true cap so a runaway can't keep issuing writer calls forever —
        parity with the review/summary hard budget guards.
        """
        if self._token_tracker is None or not self.max_total_tokens:
            return
        self._token_tracker.check_budget(
            int(self.max_total_tokens * _HARD_BUDGET_MULTIPLE)
        )

    def _budget_exhausted(self) -> bool:
        """Whether the run has hit its total-token ceiling (if one is set)."""
        if self._token_tracker is None or not self.max_total_tokens:
            return False
        if self._token_tracker.total_tokens >= self.max_total_tokens:
            note = (
                "Token budget reached: later guide sections were written without "
                "evaluation, revision or grounding."
            )
            if note not in self.notes:
                self.notes.append(note)
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
        already_in_guide: str,
        sources: str,
        available_images: str,
        final_outline: str,
    ) -> str:
        """Score a section and revise it while it stays below the threshold or
        breaks a deterministic writing rule (:func:`review_plan.rule_issues`).

        Mirrors the review pipeline's evaluate loop, but keeps the BEST draft:
        every revision is re-scored, and a revision is only kept if it ranks
        above the best so far (:func:`review_plan.draft_rank`) — so a feedback-driven rewrite can never make the section worse
        than the draft it replaced.
        """
        if self.max_revision_attempts <= 0:
            # No revision budget: skip evaluation entirely (no wasted LLM call).
            return markdown

        def rank(
            score: int, text: str, issues: list[str]
        ) -> tuple[bool, bool, int, int]:
            return draft_rank(
                passes=score >= self.min_quality_score and not issues,
                publishable=not publish_blockers(text),
                blocking=len(issues),
                score=score,
            )

        best_markdown, best_issues = markdown, rule_issues(markdown)
        best_score, feedback = await self._evaluate_section(
            topic=topic,
            planned=planned,
            section_label=section_label,
            markdown=markdown,
            previous_sections=previous_sections,
        )
        feedback = best_feedback = self._feedback_or_generic(best_score, feedback)
        for attempt in range(1, self.max_revision_attempts + 1):
            # Accept once the score clears the threshold and no rule is broken.
            # Do NOT accept a below-threshold section just because feedback is
            # empty: the prompt only omits feedback when a section is genuinely
            # excellent (90+), so empty feedback with a low score is a model
            # slip — fall back to a generic directive and still revise.
            if best_score >= self.min_quality_score and not best_issues:
                logger.info(
                    "Section %d scored %d (>= %d) - accepting.",
                    section_number,
                    best_score,
                    self.min_quality_score,
                )
                return best_markdown
            if best_issues:
                feedback = "\n\n".join(
                    part
                    for part in (
                        feedback,
                        "Fix every one of these rule violations:\n"
                        + "\n".join(f"- {i}" for i in best_issues),
                    )
                    if part
                )
            logger.info(
                "Section %d scored %d (min %d), %d rule violation(s) - revising "
                "(attempt %d/%d).",
                section_number,
                best_score,
                self.min_quality_score,
                len(best_issues),
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
                already_in_guide=already_in_guide,
                sources=sources,
                available_images=available_images,
                # Show the writer the concrete draft being revised (not just the
                # feedback), so it edits the actual text the feedback refers to
                # rather than re-drafting blind — at temperature 0 a blind rewrite
                # tends to reproduce the same rejected draft and waste the
                # (expensive) writer+evaluator calls.
                depth_directive=(
                    f"{planned.depth_directive(self.language, self._guide_type)}\n\nYou are REVISING an existing "
                    f"draft of this section that did not pass review. "
                    f"Address this feedback: {feedback}\n\n"
                    f"CURRENT DRAFT to improve (rewrite it, keeping what works):\n"
                    f"{best_markdown}"
                ),
            )
            if not revised:
                break
            score, feedback = await self._evaluate_section(
                topic=topic,
                planned=planned,
                section_label=section_label,
                markdown=revised,
                previous_sections=previous_sections,
            )
            feedback = self._feedback_or_generic(score, feedback)
            issues = rule_issues(revised)
            # Keep the revision only if it ranks above the best so far;
            # otherwise discard it (a rewrite can regress).
            if rank(score, revised, issues) > rank(
                best_score, best_markdown, best_issues
            ):
                best_markdown, best_score, best_issues = revised, score, issues
                best_feedback = feedback
            else:
                logger.info(
                    "Section %d revision scored %d (best %d) - keeping prior draft.",
                    section_number,
                    score,
                    best_score,
                )
                # The next revision starts from the kept draft again: give its
                # own feedback plus what went wrong in the rejected attempt.
                rejected = feedback if feedback != best_feedback else ""
                rejected += "".join(f"\n- {issue}" for issue in issues)
                note = rejected_revision_note(score, len(issues))
                if not rejected.strip():
                    note = note.removesuffix(":") + "."
                feedback = f"{best_feedback}\n\n{note} {rejected}".strip()
        if best_score < self.min_quality_score or best_issues:
            self.notes.append(
                f"Guide section '{planned.title}' kept below the bar (score "
                f"{best_score}, {len(best_issues)} rule violation(s)): check it."
            )
        return best_markdown

    def _feedback_or_generic(self, score: int, feedback: str) -> str:
        """Evaluator feedback, or a generic directive when a below-bar score
        came without any (the prompt only omits feedback for 90+ sections, so
        empty feedback with a low score is a model slip)."""
        if feedback or score >= self.min_quality_score:
            return feedback
        return (
            "The section scored below the quality bar. Increase technical "
            "depth and concrete detail, tighten structure, and remove any "
            "redundancy with earlier sections."
        )

    @RetryableBase._retry("tech_guide_evaluation")
    async def _evaluate_section(
        self,
        *,
        topic: str,
        planned: PlannedSection,
        section_label: str,
        markdown: str,
        previous_sections: str,
    ) -> tuple[int, str]:
        """Return (quality_score 0-100, improvement_feedback) for a section."""
        result = await self.evaluation_chain.ainvoke(
            {
                "topic": topic,
                "section": markdown,
                "section_to_write": section_label,
                "depth_directive": planned.depth_directive(
                    self.language, self._guide_type
                ),
                "previous_sections": previous_sections,
                "length_report": length_report(markdown, planned.depth, self.language),
                "language": self.language,
            }
        )
        score = parse_quality_score((result or {}).get("quality_score"))
        feedback = ((result or {}).get("improvement_feedback") or "").strip()
        return score, feedback

    @RetryableBase._retry("tech_guide_grounding")
    async def _ground_section(self, section_markdown: str, sources: str) -> str:
        """Fact-check a drafted section, returning the cleaned version.

        Falls back to the original draft if the grounding pass returns nothing,
        so verification can only improve—never drop—a section.
        """
        result = await self.grounding_chain.ainvoke(
            {
                "section": section_markdown,
                "sources": sources,
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
        already_in_guide: str,
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
            "already_in_guide": already_in_guide,
            "sources": sources,
            "available_images": available_images,
            "depth_directive": depth_directive,
            "language": self.language,
        }
        result = await self.section_chain.ainvoke(payload)
        return (result or {}).get("section_markdown", "").strip()

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
            depth, visuals, title = self._extract_depth(body)
            if title:
                sections.append(
                    PlannedSection(
                        title=title,
                        depth=depth,
                        visuals=visuals,
                        description=self._extract_description(body),
                    )
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
    def _extract_depth(body: str) -> tuple[str, str, str]:
        """Pull leading [AREA] [depth] tags and a trailing visuals hint off a line.

        Returns (depth, visuals, title); the area tag is consumed, not kept. Tags MUST be bracketed (the
        synopsis prompt always brackets them); this avoids mis-reading a title
        word that happens to match a keyword (e.g. "Usage — ..."). They may
        appear in either order; the remaining text (minus a trailing
        "(visuals: ...)" hint) is the title."""
        depth = _DEFAULT_DEPTH
        # Consume up to two leading [..] bracketed tokens. An unrecognised token
        # (typo'd / hallucinated tag) is still consumed so it never leaks into
        # the rendered heading.
        for _ in range(2):
            match = re.match(r"^\[([A-Za-z]+)\]\s*", body)
            if not match:
                break
            token = match.group(1)
            if token.lower() in DEPTH_CHARS:
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
        return depth, visuals, title or body.strip()

    @staticmethod
    def _extract_description(body: str) -> str:
        """The planned scope after the em/en dash, minus the visuals hint."""
        parts = re.split(r"\s+[—–]\s+", body, maxsplit=1)
        if len(parts) < 2:
            return ""
        return re.split(r"\(visuals?:", parts[1], maxsplit=1, flags=re.I)[0].strip()

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


# Identifiers a reader copies verbatim: CLI flags and Prometheus-style metric
# names ("vllm:time_to_first_token_seconds", "..._requests_total").
_FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*")
_METRIC = re.compile(
    r"\b[a-z][a-z0-9]*(?::[a-z0-9]+(?:_[a-z0-9]+)+"
    r"|(?:_[a-z0-9]+)+_(?:total|seconds|bytes|count|bucket|sum))\b"
)
MAX_UNVERIFIED = 10


_OUTLINE_HEADING = re.compile(r"^##[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
_MAX_OUTLINE_HEADINGS = 40


def input_outline(corpus: ResearchCorpus, urls: tuple[str, ...] | list[str]) -> str:
    """The ``##`` headings of the requested pages (not sub-pages or search
    results), one per line: the topics a getting-started guide must cover."""
    wanted = {normalize_url(u) for u in urls}
    headings: list[str] = []
    for page in corpus.pages:
        if normalize_url(page.url) not in wanted:
            continue
        for title in _OUTLINE_HEADING.findall(page.text):
            if title not in headings:
                headings.append(title)
    lines = [f"- {neutralize_prompt_tags(h)}" for h in headings[:_MAX_OUTLINE_HEADINGS]]
    return "\n".join(lines) or "(none)"


def official_reference(
    urls: list[str], corpus: ResearchCorpus
) -> tuple[str, list[str]]:
    """(text of the fetched pages of the requested docs — where they landed,
    in a compatible docs version — and the names their commands go by)."""
    official = OfficialDocs.from_research(urls, corpus.pages)
    text = "\n".join(
        p.text for p in corpus.pages if official.covers(p.final_url or p.url)
    )
    return text, product_names(urls, corpus.pages)


def _flag_spellings(flag: str, both_styles: bool) -> list[str]:
    """Ways the official text may spell ``flag``: as written, as a Python API
    argument (``--max-model-len`` → ``max_model_len``, multi-word names only,
    since a bare word like ``model`` appears in any prose), and in the other
    dash style when the docs use both."""
    name = flag[2:]
    spellings = [flag]
    if "-" in name:  # an underscore flag already is its own API form
        spellings.append(name.replace("-", "_"))
    if re.search(r"[-_]", name) and both_styles:
        spellings += ["--" + name.replace("_", "-"), "--" + name.replace("-", "_")]
    return spellings


def unverified_identifiers(
    body: str, official_text: str, products: str | list[str]
) -> list[str]:
    """CLI flags of ``product`` commands and metric names the guide shows in
    code that the official pages never mention (a stale or invented flag is
    the costliest guide error: readers paste it). Flags count only after the
    product's name on a command line (or alone in inline code), so
    ``docker``/``curl`` options are not checked. Note-only: an empty ``official_text`` checks nothing.
    """
    if not official_text.strip():
        return []
    names = [products] if isinstance(products, str) else list(products)
    names = [n.lower() for n in names if n]
    tokens = [re.compile(rf"(?<![\w/.-]){re.escape(n)}(?![\w-])") for n in names]
    known = official_text.lower()
    # CLIs that take both "--a-b" and "--a_b" (torchrun, accelerate) show both
    # styles in their docs; one that shows only dashes (ruff) rejects "_".
    both_styles = bool(re.search(r"--[a-z0-9]+-[a-z]", known)) and bool(
        re.search(r"--[a-z0-9]+_[a-z]", known)
    )
    found: list[str] = []

    def check(token: str, variants: list[str]) -> None:
        mentioned = any(
            re.search(rf"(?<![\w-]){re.escape(v)}(?![\w-])", known) for v in variants
        )
        if token not in found and not mentioned:
            found.append(token)

    for seg in segments(body):
        if seg.kind not in ("fence", "icode"):
            continue
        code = re.sub(r"\\\n\s*", " ", body[seg.start : seg.end])  # join lines
        for line in code.splitlines():
            lowered = line.lower()
            # The product as a command word, not inside HF_TOKEN or a path.
            hits = [
                m.start() for m in (pattern.search(lowered) for pattern in tokens) if m
            ]
            start = min(hits) if hits else (0 if not names else -1)
            if seg.kind == "fence" and start < 0:
                continue
            # Inline code without the product counts a flag only when it is a
            # bare flag (`--max-model-len`), not another tool's (`curl -s`).
            bare_flag = line.strip("` ").startswith("--")
            flags = (
                _FLAG.findall(line[max(start, 0) :]) if start >= 0 or bare_flag else []
            )
            for flag in flags:
                check(flag, _flag_spellings(flag, both_styles))
            for metric in _METRIC.findall(line):
                check(metric, [metric, metric.replace(":", "_")])
    return found[:MAX_UNVERIFIED]
