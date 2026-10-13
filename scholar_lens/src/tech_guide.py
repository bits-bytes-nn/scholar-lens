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
import difflib
import re
from dataclasses import dataclass, field
from typing import Any

import boto3
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from .constants import LanguageModelId
from .guide_authority import (
    input_outline,
    official_reference,
    parse_section_plan,
    third_party_identifiers,
    unverified_identifiers,
    used_source_urls,
)
from .logger import logger
from .markdown_lint import close_open_fence
from .markdown_math import without_fences
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
    parse_lines,
    publish_blockers,
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
    MAX_REFERENCE_PAGES,
    MAX_REFERENCE_SEARCHES,
    OfficialDocs,
    PageContent,
    ResearchCorpus,
    WebResearcher,
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
# The lookup before writing may use only part of the per-guide reference
# budget; the rest stays for the identifiers the drafts introduce.
_PRE_WRITE_REFERENCE_PAGES: int = MAX_REFERENCE_PAGES * 2 // 3
_PRE_WRITE_REFERENCE_SEARCHES: int = MAX_REFERENCE_SEARCHES * 2 // 3
# Default number of web-search queries the research planner may propose.
_MAX_RESEARCH_QUERIES: int = 6
# Default number of top search-result pages fetched into the corpus so the
# gathered material reaches the writer (not just the outline planner).
_FETCH_TOP_RESULTS: int = 4
# Section quality gate (mirrors the review pipeline's evaluate loop).
_MIN_QUALITY_SCORE: int = 75
_MAX_REVISION_ATTEMPTS: int = 1
# Evaluator feedback logged per section (the full text goes to the writer).
_LOGGED_FEEDBACK_CHARS: int = 300
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


def parse_section_output(raw: str) -> dict[str, str]:
    """Extract ``<section_markdown>`` from the writer output; if the wrapper is
    missing or unbalanced, keep the text and drop stray tag fragments."""
    match = re.search(r"<section_markdown>(.*?)</section_markdown>", raw, re.DOTALL)
    text = match.group(1) if match else raw
    return {"section_markdown": re.sub(r"</?section_markdown>", "", text).strip()}


_NO_OFFICIAL_SOURCE = (
    "No official documentation page covers this section: state only what "
    "<sources> show, and leave out any flag, default or command they do not."
)


@dataclass
class PlannedSection:
    """One outline entry: title, writing depth, visual hint, and the concepts
    it owns (from the synopsis's section plan)."""

    title: str
    depth: str = _DEFAULT_DEPTH
    visuals: str = ""
    description: str = ""
    owns: list[str] = field(default_factory=list)
    # Corpus pages the plan lists for the section, plus an official page its
    # title lookup found: put first in the section's sources.
    sources: list[str] = field(default_factory=list)
    # False once the official-site lookup found nothing for the section.
    official_source: bool = True

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
        if not self.official_source:
            directive += f"\n{_NO_OFFICIAL_SOURCE}"
        return directive


def owned_elsewhere(sections: list[PlannedSection], index: int) -> str:
    """The concepts the other sections own (``index`` is 0-based), for the
    writer of section ``index``: used there, explained only by their owner."""
    lines = [
        f"- {'earlier' if i < index else 'later'}, '{s.title}': {'; '.join(s.owns)}"
        for i, s in enumerate(sections)
        if i != index and s.owns
    ]
    return "\n".join(lines) or "- none"


@dataclass
class TechGuide:
    topic: str
    body: str
    source_urls: list[str] = field(default_factory=list)
    # Degraded-output notes for the PR checklist and the job result.
    notes: list[str] = field(default_factory=list)
    # Images the writer was offered; any other image in the body is dropped.
    image_urls: list[str] = field(default_factory=list)


class TechGuideGenerator(RetryableBase, TokenBudgetGuard):
    """Generates a self-study technical guide from a list of source URLs."""

    # The requested docs (hosts and versions): the authority for flags and defaults.
    _guide_type: str = ""
    _official: OfficialDocs = OfficialDocs(frozenset(), frozenset())
    _source_urls: tuple[str, ...] = ()
    # The names the docs' commands go by (set by the relevance gate).
    _products: tuple[str, ...] = ()
    # The synopsis's raw <section_plan> (per-section sources and concepts).
    _section_plan: str = ""
    # Official reference pages fetched this run, the identifiers they were
    # looked up for and the searches spent (capped per guide).
    _reference_pages: list[PageContent]
    _reference_ids: list[str]
    _reference_searches: int = 0
    # Pages that made it into some section's sources this run.
    _section_source_urls: set[str]

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
        self._reference_pages, self._reference_ids = [], []
        self._reference_searches = 0
        self._section_source_urls = set()

        # Fetch the seed URLs (+ in-scope sub-pages) once. WebResearcher is
        # synchronous (blocking HTTP crawl); run it off the event loop.
        corpus = await asyncio.to_thread(
            self.researcher.research, urls, discover_subpages=discover_subpages
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

        # Flags and arguments only third-party pages give: settle them from the
        # official reference before writing. The writer leaves out a flag no
        # official source backs, and the drafts' own flags come too late to
        # bring back the ones a getting-started guide needs.
        await self._add_reference(
            corpus,
            third_party_identifiers(corpus, self._official, self._products),
            max_pages=_PRE_WRITE_REFERENCE_PAGES,
            max_searches=_PRE_WRITE_REFERENCE_SEARCHES,
        )

        topic = (
            planned_topic
            if planned_topic and not is_placeholder(planned_topic)
            else gate_topic
        )
        synopsis = await self._draft_synopsis(topic, corpus)
        body = await self._write_sections(topic, synopsis, corpus)
        unverified = unverified_identifiers(
            body, official_reference(self._official, corpus), self._products
        )
        if unverified:
            self.notes.append(
                "Flags/metrics the official pages fetched for this guide never "
                "mention; verify them against the current reference: "
                + ", ".join(unverified)
            )

        return TechGuide(
            topic=topic,
            body=body,
            source_urls=used_source_urls(
                body,
                corpus,
                urls,
                self._official,
                self._reference_pages,
                self._section_source_urls,
            ),
            notes=list(self.notes),
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
        """Propose the blog topic, tags and SEO title for a finished guide from
        its body's headings and first sentences, so no tag names something the
        guide does not teach (the caller normalizes and validates them against
        the post conventions). Past the soft cap the finished guide is still
        classified: only the hard ceiling stops a guide whose sections are
        already paid for."""
        self._enforce_hard_token_budget()
        result: dict[str, str] = await self.taxonomy_chain.ainvoke(
            {
                "title": guide.topic,
                "outline": digest_sections([guide.body]),
                "category_options": ", ".join(category_options),
                "controlled_tags": ", ".join(controlled_tags),
            }
        )
        return result

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
        self._products = tuple(
            product_names(result.get("commands", ""), self._source_urls, corpus.pages)
        )
        logger.info(
            "Relevance gate passed. Topic: '%s'; commands: %s", topic, self._products
        )
        return topic

    async def _draft_synopsis(self, topic: str, corpus: ResearchCorpus) -> str:
        result = await self.synopsis_chain.ainvoke(
            {
                "topic": topic,
                "sources": self.llm_factory.fit_text(
                    self.synopsis_model_id,
                    corpus.combined_text(self._official),
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
        self._section_plan = result.get("section_plan") or ""
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
        plan = parse_section_plan(self._section_plan, corpus.pages)
        for number, section in enumerate(sections, start=1):
            # A section without a plan line has no official page either.
            urls, section.owns = plan.get(number, ([], []))
            section.sources = list(urls)
            if not any(self._official.covers(u) for u in urls):
                await self._find_official_source(corpus, section)
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
                included=self._section_source_urls,
                prefer=planned.sources,
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
                owned_elsewhere=owned_elsewhere(sections, index - 1),
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
                    "skipped."
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
                    owned_elsewhere=owned_elsewhere(sections, index - 1),
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
        unchecked = unverified_identifiers(
            "\n\n".join(written),
            official_reference(self._official, corpus),
            self._products,
        )
        await self._add_reference(corpus, unchecked)
        reference = ResearchCorpus(pages=list(self._reference_pages))
        for index, sources in to_ground:
            self._enforce_hard_token_budget()
            if unchecked and reference.pages:
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
            changed = sum(
                1
                for line in difflib.unified_diff(
                    draft.splitlines(), grounded.splitlines(), n=0, lineterm=""
                )
                if line[:1] in "+-" and not line.startswith(("+++", "---"))
            )
            logger.info(
                "Grounded section %d: %d line(s) changed, %d -> %d chars.",
                index + 1,
                changed,
                len(draft),
                len(grounded),
            )
            written[index] = grounded

    async def _find_official_source(
        self, corpus: ResearchCorpus, planned: PlannedSection
    ) -> None:
        """Search the official site for the section's title when the synopsis
        named no official page for it, within the pre-write reference budget
        (shared with the third-party identifier lookup, so grounding keeps the
        rest). Only a search result counts: the reference links of the pages
        already fetched say nothing about the section. Without a search
        backend, or when the search finds nothing, the section is written from
        what its sources show. Once the budget is spent the section is left as
        planned: no search ran, so nothing says it lacks an official page."""
        searchable = self.researcher.search_provider.supports_search
        if searchable and (
            self._reference_searches >= _PRE_WRITE_REFERENCE_SEARCHES
            or len(self._reference_pages) >= _PRE_WRITE_REFERENCE_PAGES
        ):
            return
        added: list[PageContent] = []
        if searchable:
            added = await self._add_reference(
                corpus,
                [planned.title],
                max_pages=_PRE_WRITE_REFERENCE_PAGES,
                max_searches=_PRE_WRITE_REFERENCE_SEARCHES,
                follow_links=False,
            )
        planned.official_source = bool(added)
        planned.sources += [page.url for page in added]
        if not added:
            logger.info("Section '%s' has no official source.", planned.title)

    async def _add_reference(
        self,
        corpus: ResearchCorpus,
        identifiers: list[str],
        *,
        max_pages: int = MAX_REFERENCE_PAGES,
        max_searches: int = MAX_REFERENCE_SEARCHES,
        follow_links: bool = True,
    ) -> list[PageContent]:
        """Fetch official reference pages for identifiers (or section titles)
        not looked up yet, within the per-guide caps shared by every lookup of
        the run (or the lower running totals ``max_pages``/``max_searches``),
        and return the pages added."""
        identifiers = [i for i in identifiers if i not in self._reference_ids]
        pages_left = min(max_pages, MAX_REFERENCE_PAGES) - len(self._reference_pages)
        if not identifiers or pages_left <= 0:
            return []
        searches_left = (
            min(max_searches, MAX_REFERENCE_SEARCHES) - self._reference_searches
        )
        added = await asyncio.to_thread(
            self.researcher.add_official_reference,
            corpus,
            self._official,
            identifiers,
            product=self._products[0] if self._products else "",
            max_searches=searches_left,
            max_pages=pages_left,
            follow_links=follow_links,
        )
        if self.researcher.search_provider.supports_search:
            self._reference_searches += min(searches_left, len(identifiers))
        self._reference_ids += identifiers
        self._reference_pages += added
        return added

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
        owned_elsewhere: str,
    ) -> str:
        """Score a section and revise it while it stays below the threshold or
        has a blocking issue: one the evaluator names (a factual error, an
        internal contradiction, an unsupported flag or default, a structural
        self-reference) or a writing rule it breaks in code
        (:func:`review_plan.rule_issues`).

        Keeps the BEST draft: a revision is re-scored and kept only if it ranks
        above the draft it revises (:func:`review_plan.draft_rank`, with the
        evaluator's blocking issues ahead of the score since they alone decide
        acceptance), so a feedback-driven rewrite can never make the section
        worse.
        """
        if self.max_revision_attempts <= 0:
            # No revision budget: skip evaluation entirely (no wasted LLM call).
            return markdown

        async def evaluate(text: str) -> tuple[int, str, list[str], int]:
            score, feedback, blocking = await self._evaluate_section(
                topic=topic,
                planned=planned,
                section_label=section_label,
                markdown=text,
                previous_sections=previous_sections,
                sources=sources,
                final_outline=final_outline,
            )
            feedback = self._feedback_or_generic(score, feedback)
            # The issues to fix, and how many the evaluator raised.
            return score, feedback, blocking + rule_issues(text), len(blocking)

        def rank(
            score: int, text: str, issues: list[str], factual: int
        ) -> tuple[bool, bool, int, int, int]:
            passes, publishable, score, slips = draft_rank(
                passes=score >= self.min_quality_score and not issues,
                publishable=not publish_blockers(text),
                blocking=len(issues),
                score=score,
            )
            return passes, publishable, -factual, score, slips

        best_markdown = markdown
        best_score, best_feedback, best_issues, best_factual = await evaluate(markdown)
        for attempt in range(1, self.max_revision_attempts + 1):
            # Accept only a score at the bar with no blocking issue. Empty
            # feedback with a low score is a model slip (the prompt omits it
            # only for 90+), so it still revises with a generic directive.
            if best_score >= self.min_quality_score and not best_issues:
                break
            feedback = "\n\n".join(
                part
                for part in (
                    best_feedback,
                    (
                        "Fix every one of these blocking issues:\n"
                        + "\n".join(f"- {i}" for i in best_issues)
                        if best_issues
                        else ""
                    ),
                )
                if part
            )
            logger.info(
                "Section %d scored %d (min %d), %d blocking issue(s) - revising "
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
                owned_elsewhere=owned_elsewhere,
                # The writer edits the draft the feedback refers to: a blind
                # rewrite at temperature 0 tends to reproduce the rejected one.
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
            score, revised_feedback, issues, factual = await evaluate(revised)
            if rank(score, revised, issues, factual) <= rank(
                best_score, best_markdown, best_issues, best_factual
            ):
                # A rewrite of the same draft would mostly repeat this one.
                logger.info(
                    "Section %d revision scored %d (discarded, best %d) - "
                    "stopping revisions.",
                    section_number,
                    score,
                    best_score,
                )
                break
            best_markdown, best_score = revised, score
            best_feedback, best_issues = revised_feedback, issues
            best_factual = factual
        logger.info(
            "Section %d final: score %d, %d blocking issue(s). Evaluator: %s",
            section_number,
            best_score,
            len(best_issues),
            " ".join(best_feedback.split())[:_LOGGED_FEEDBACK_CHARS] or "(none)",
        )
        if best_score < self.min_quality_score or best_issues:
            self.notes.append(
                f"Guide section '{planned.title}' kept below the bar (score "
                f"{best_score}, {len(best_issues)} blocking issue(s)): check it."
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
        sources: str,
        final_outline: str,
    ) -> tuple[int, str, list[str]]:
        """Return (quality_score 0-100, improvement_feedback, blocking_issues)
        for a section, judged against the sources it was written from and the
        full outline (so it can tell an unsupported claim or another section's
        material)."""
        result = await self.evaluation_chain.ainvoke(
            {
                "topic": topic,
                "section": markdown,
                "section_to_write": section_label,
                "outline": final_outline,
                "sources": sources,
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
        return score, feedback, parse_lines((result or {}).get("blocking_issues"))

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
        owned_elsewhere: str,
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
            "owned_elsewhere": owned_elsewhere,
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
        lines = [line.strip() for line in synopsis.splitlines()]
        # Numbered lines are the sections (the section plan's numbers refer
        # to them); a stray bullet under one is not. Bullets count only in an
        # outline with no numbered line.
        items = [ln for ln in lines if re.match(r"^\d+[.)]\s+", ln)] or [
            ln for ln in lines if re.match(r"^[-*•]\s+", ln)
        ]
        for stripped in items:
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
