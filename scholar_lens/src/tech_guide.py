"""Technical guide / tutorial generator.

Given documentation URLs (library/framework/platform docs: the authority for
commands, flags and defaults) and optional reference material (a paper, design
doc or deep-dive post the user wants the guide to build on), this agent:
1. Plans deep research — derives the topic and a set of web-search queries
   (usage, internals and design, production integration) from the inputs, then
   researches the docs URLs (+ in-scope sub-pages), the reference material and
   the planned/explicit search queries into a corpus so the guide draws on
   broad, complementary material rather than translating a single page.
2. Gates on relevance — refuses to write if the docs (or, without docs, the
   reference material) are not technical developer material (raises
   :class:`NotTechnicalContentError`).
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
import functools
import re
from dataclasses import dataclass, field
from typing import Any

import boto3
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from .constants import LanguageModelId
from .guide_authority import (
    input_outline,
    mentions_any,
    official_reference,
    parse_section_plan,
    third_party_identifiers,
    unverified_identifiers,
    used_source_urls,
)
from .logger import logger
from .markdown_lint import close_open_fence
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
from .prompts.base import PROSE_UNIT, blocking_issues_block
from .prompts.tech_guide import (
    DEPTH_CHARS,
    NON_KOREAN_CHAR_FACTOR,
    REFERENCE_LABEL,
    TechGuideTaxonomyPrompt,
)
from .review_plan import (
    article_ledger,
    digest_sections,
)
from .section_loop import (
    EMPTY_DRAFT,
    FIXING,
    REVISING,
    checked_grounding,
    feedback_or_generic,
    glossary_view,
    measured_length,
    owned_elsewhere,
    parse_glossary,
    parse_section_output,
    revise_section,
)
from .utils import (
    BedrockLanguageModelFactory,
    HTMLTagOutputParser,
    RetryableBase,
    is_affirmative,
    is_placeholder,
    measure_execution_time,
    parse_lines,
    parse_quality_score,
)
from .web_research import (
    MAX_REFERENCE_PAGES,
    MAX_REFERENCE_SEARCHES,
    OfficialDocs,
    PageContent,
    ResearchCorpus,
    SourceFetchError,
    WebResearcher,
    page_key,
    product_names,
    source_label,
)

DEFAULT_LANGUAGE: str = "Korean"
# Source material handed to one section's writer/evaluator/grounding calls:
# the most relevant pages only (about 15k tokens), not the whole corpus.
_SECTION_SOURCE_CHARS: int = 60_000
# Official reference excerpt put ahead of a section's sources for grounding.
_REFERENCE_SOURCE_CHARS: int = 20_000
# The lookups before writing may use only part of the per-guide reference
# budget; the rest stays for the identifiers the drafts introduce. Of that
# part, the third-party identifier lookup takes at most half, so the
# per-section title searches keep a share.
_PRE_WRITE_REFERENCE_PAGES: int = MAX_REFERENCE_PAGES * 2 // 3
_PRE_WRITE_REFERENCE_SEARCHES: int = MAX_REFERENCE_SEARCHES * 2 // 3
_IDENTIFIER_REFERENCE_PAGES: int = _PRE_WRITE_REFERENCE_PAGES // 2
_IDENTIFIER_REFERENCE_SEARCHES: int = _PRE_WRITE_REFERENCE_SEARCHES // 2
# Grounding that cuts a section by more than this share is logged.
_GROUNDING_CUT_LOGGED: float = 0.25
# Section quality gate (mirrors the review pipeline's evaluate loop).
_MIN_QUALITY_SCORE: int = 75
_MAX_REVISION_ATTEMPTS: int = 1

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
    "This is a getting-started guide: usage leads, and internals are explained "
    "by how the mechanism works and why it is built that way, so the reader "
    "understands what they run and tune."
)


def length_report(markdown: str, depth: str, language: str) -> str:
    """Measured length against the depth's target, for the evaluator."""
    depth = depth if depth in DEPTH_CHARS else _DEFAULT_DEPTH
    bound, target = _depth_target(depth, language)
    return measured_length(
        markdown, label=f"a {depth} section", bound=bound, target=target
    )


class NotTechnicalContentError(Exception):
    """Raised when the supplied URLs are not technical documentation."""


_NO_OFFICIAL_SOURCE = (
    "No official documentation page covers this section: it rests on the "
    "explanatory sources in <sources>."
)
_NO_DOCS_NOTE = (
    "No documentation URL was given, so no page counted as official: verify "
    "the guide's commands, flags and defaults against the product's current "
    "docs."
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
            f"{depth.upper()} section ({bound} {target:,} {PROSE_UNIT}): "
            f"{_DEPTH_GUIDANCE[depth]}"
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


@dataclass
class TechGuide:
    topic: str
    body: str
    source_urls: list[str] = field(default_factory=list)
    # Page title of each source URL, for the source list's link text.
    source_titles: dict[str, str] = field(default_factory=dict)
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
    # The synopsis's glossary: English term -> rendering.
    _glossary: dict[str, str] = {}
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
        max_sections: int,
        title_max_length: int,
        verify_grounding: bool = True,
        auto_research: bool = True,
        max_research_queries: int,
        fetch_top_results: int,
        min_quality_score: int = _MIN_QUALITY_SCORE,
        max_revision_attempts: int = _MAX_REVISION_ATTEMPTS,
        max_total_tokens: int | None = None,
        translation_guideline: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.language = language
        # Curated term renderings; the synopsis's glossary applies them.
        self.translation_guideline = translation_guideline or []
        self.max_sections = max_sections
        # The blog's title limit (the conventions' ``seo_title_max_length``).
        self.title_max_length = title_max_length
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
        # The evaluator's blocking issues the last reviewed section kept.
        self._leftover_blocking: list[str] = []
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
        reference_urls: list[str] | None = None,
        discover_subpages: bool = True,
        search_queries: list[str] | None = None,
    ) -> TechGuide:
        """Write a guide from documentation ``urls`` (the authority for
        commands, flags and defaults) and ``reference_urls``: explanatory
        material to build on, always fetched and listed but never official.
        Either may be empty, not both; without docs no page is official."""
        # One spelling per page (and per arXiv paper: abs, pdf, any version).
        held = {page_key(u) for u in urls}
        references: list[str] = []
        for url in reference_urls or []:
            if (key := page_key(url)) not in held:
                held.add(key)
                references.append(url)
        if not urls and not references:
            raise ValueError("At least one source URL is required.")
        # The requested docs are the authority for flags, defaults and names.
        self._official = OfficialDocs.from_urls(urls)
        self._source_urls = tuple(urls)
        self._reference_pages, self._reference_ids = [], []
        self._reference_searches = 0
        self._section_source_urls = set()

        # Fetch the docs URLs (+ in-scope sub-pages) once. WebResearcher is
        # synchronous (blocking HTTP crawl); run it off the event loop.
        corpus = ResearchCorpus()
        if urls:
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
        else:
            self.notes.append(_NO_DOCS_NOTE)
        docs = ResearchCorpus(pages=list(corpus.pages))
        await self._add_material(corpus, references)

        # Judge the docs alone (the reference material when there are none):
        # search results never redefine what the user asked for, and a
        # rejected input spends nothing on searches.
        gate_topic = await self._assert_relevant(docs if docs.pages else corpus)

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
            before = len(corpus.pages)
            await asyncio.to_thread(
                self.researcher.run_searches,
                corpus,
                queries,
                fetch_top=self.fetch_top_results,
            )
            self._log_added("search", corpus.pages[before:])
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
            max_pages=_IDENTIFIER_REFERENCE_PAGES,
            max_searches=_IDENTIFIER_REFERENCE_SEARCHES,
        )

        topic = (
            planned_topic
            if planned_topic and not is_placeholder(planned_topic)
            else gate_topic
        )
        # The planned topic steers the searches and the outline; the title
        # the outline comes with names the scope it actually planned.
        topic, synopsis = await self._draft_synopsis(topic, corpus)
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

        source_urls = used_source_urls(
            body,
            corpus,
            [*urls, *references],
            self._official,
            self._reference_pages,
            self._section_source_urls,
        )
        titles = {page.url: page.title for page in corpus.pages}
        return TechGuide(
            topic=topic,
            body=body,
            source_urls=source_urls,
            source_titles={url: titles[url] for url in source_urls if titles.get(url)},
            notes=list(self.notes),
            image_urls=list(corpus.image_urls),
        )

    async def _add_material(
        self, corpus: ResearchCorpus, references: list[str]
    ) -> None:
        """Fetch the user's reference material (no sub-pages: a post's links
        are other posts) into ``corpus``, marked so prompts label it
        reference material. A page the docs crawl already holds stays as it
        is. Pages that could not be read go to the notes; with no docs and no
        readable reference the request has nothing to write from."""
        if not references:
            return
        try:
            fetched = await asyncio.to_thread(
                self.researcher.research, references, discover_subpages=False
            )
        except SourceFetchError:
            if not corpus.pages:
                raise
            fetched = ResearchCorpus()
        held = {page_key(u) for p in corpus.pages for u in (p.url, p.final_url) if u}
        for page in fetched.pages:
            keys = {page_key(u) for u in (page.url, page.final_url) if u}
            if not keys & held:
                page.user_reference = True
                corpus.pages.append(page)
                held |= keys
                self._log_added("reference-material", [page])
        if not corpus.pages:
            raise NotTechnicalContentError(
                "None of the supplied URLs returned readable content."
            )
        read = {page_key(p.url) for p in fetched.pages}
        missing = [u for u in references if page_key(u) not in read]
        if missing:
            self.notes.append(
                "Reference material that could not be read (left out): "
                + ", ".join(missing)
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
                "title_max_length": self.title_max_length,
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
                "Research plan: topic='%s', %d queries: %s",
                topic,
                len(queries),
                queries,
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

    async def _draft_synopsis(
        self, topic: str, corpus: ResearchCorpus
    ) -> tuple[str, str]:
        """The guide's title (``topic`` when the outline came without one)
        and its outline."""
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
                "translation_guideline": str(self.translation_guideline or ""),
                "max_sections": self.max_sections,
                "title_max_length": self.title_max_length,
                "language": self.language,
            }
        )
        synopsis = result.get("synopsis", "").strip()
        if not synopsis:
            raise ValueError("Synopsis generation returned empty output.")
        self._guide_type = (result.get("guide_type") or "").strip().lower()
        self._section_plan = result.get("section_plan") or ""
        self._glossary = parse_glossary(result.get("glossary") or "")
        title = " ".join((result.get("title") or "").split())
        # An unclosed tag would make the rest of the output the title.
        if not title or is_placeholder(title) or "<" in title or len(title) > 200:
            title = topic
        logger.info("Guide type: %s; title: '%s'", self._guide_type or "?", title)
        return title, synopsis

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
            if not self._official.hosts:
                section.official_source = False  # no docs: nothing to look up
            elif not any(self._official.covers(u) for u in urls):
                await self._find_official_source(corpus, section)
        labels = {page.url: source_label(page, self._official) for page in corpus.pages}
        for number, section in enumerate(sections, start=1):
            logger.info(
                "Section %d planned sources: %s",
                number,
                ", ".join(f"[{labels.get(u, '?')}] {u}" for u in section.sources)
                or "(none)",
            )
        available_images = "\n".join(corpus.image_urls) or "(none)"

        # Canonical numbered outline of ONLY the sections that will be written,
        # so the model never sees (and never references) dropped section numbers.
        final_outline = "\n".join(s.label(i) for i, s in enumerate(sections, start=1))
        total = len(sections)

        written: list[str] = []
        # (index in written, its sources, its glossary view)
        to_ground: list[tuple[int, str, str, list[str]]] = []
        for index, planned in enumerate(sections, start=1):
            # Hard ceiling: past HARD_BUDGET_MULTIPLE × the budget, abort like
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
            glossary = glossary_view(self._glossary, written)
            sources = corpus.relevant_text(
                f"{topic} {planned.title} {planned.description}",
                max_chars=_SECTION_SOURCE_CHARS,
                official=self._official,
                included=self._section_source_urls,
                prefer=planned.sources,
            )
            write = functools.partial(
                self._write_one_section,
                topic=topic,
                synopsis=final_outline,
                section=section_label,
                section_number=index,
                total_sections=total,
                previous_sections=previous,
                already_in_guide=article_ledger(written),
                sources=sources,
                available_images=available_images,
                owned_elsewhere=owned_elsewhere(sections, index - 1),
                glossary=glossary,
            )
            directive = planned.depth_directive(self.language, self._guide_type)
            markdown = await write(depth_directive=directive)
            if not markdown:
                # Once more, like the review loop, before leaving a hole.
                logger.warning("Section %d came back empty - retrying once.", index)
                markdown = await write(depth_directive=f"{directive}\n\n{EMPTY_DRAFT}")
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
                    glossary=glossary,
                )
                if self.verify_grounding:
                    to_ground.append(
                        (len(written), sources, glossary, self._leftover_blocking)
                    )
            written.append(close_open_fence(markdown))
        if to_ground:
            await self._ground_sections(written, to_ground, corpus)
        return "\n\n".join(written)

    async def _ground_sections(
        self,
        written: list[str],
        to_ground: list[tuple[int, str, str, list[str]]],
        corpus: ResearchCorpus,
    ) -> None:
        """Fact-check the drafted sections in place. Flags and metric names the
        official pages fetched so far never mention first pull the official
        reference pages that can settle them into each section's sources, so
        a stale flag is corrected instead of hedged. The evaluator's blocking
        issues a section's revisions left go to its fact-checker to fix."""
        unchecked = unverified_identifiers(
            "\n\n".join(written),
            official_reference(self._official, corpus),
            self._products,
        )
        await self._add_reference(corpus, unchecked)
        reference = ResearchCorpus(pages=list(self._reference_pages))
        for index, sources, glossary, blocking in to_ground:
            self._enforce_hard_token_budget()
            # The section's own unchecked identifiers pick the reference parts.
            own = [i for i in unchecked if i in written[index]]
            if own and reference.pages:
                excerpt = reference.relevant_text(
                    " ".join(own) + " " + written[index][:2_000],
                    max_chars=_REFERENCE_SOURCE_CHARS,
                    official=self._official,
                )
                sources = f"{excerpt}\n\n---\n\n{sources}"
            draft = written[index]
            if blocking:
                logger.info(
                    "Grounding section %d with %d leftover blocking issue(s).",
                    index + 1,
                    len(blocking),
                )
            grounded = await checked_grounding(
                draft,
                functools.partial(
                    self._ground_section, draft, sources, glossary, blocking
                ),
            )
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
            if len(grounded) < len(draft) * (1 - _GROUNDING_CUT_LOGGED):
                logger.info(
                    "Grounding cut section %d by %.0f%%: check what it removed.",
                    index + 1,
                    100 * (1 - len(grounded) / len(draft)),
                )
            written[index] = grounded

    async def _find_official_source(
        self, corpus: ResearchCorpus, planned: PlannedSection
    ) -> None:
        """Search the official site for the section's title when the synopsis
        named no official page for it, within the pre-write reference budget
        (the third-party identifier lookup used at most half of it, and
        grounding keeps the rest). Only a search result counts: the reference links of the pages
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
        # An identifier lookup keeps only pages that settle an identifier, as
        # the unverified check reads them; a title lookup keeps its hits.
        official_text = official_reference(self._official, corpus)
        settles = (
            functools.partial(
                mentions_any, identifiers=identifiers, official_text=official_text
            )
            if follow_links
            else None
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
            settles=settles,
        )
        if self.researcher.search_provider.supports_search:
            self._reference_searches += min(searches_left, len(identifiers))
        self._reference_ids += identifiers
        self._reference_pages += added
        self._log_added("reference-lookup", added)
        return added

    def _log_added(self, kind: str, pages: list[PageContent]) -> None:
        """One INFO line per page added to the corpus: how it came in, the
        label prompts give it, and its URL."""
        for page in pages:
            logger.info(
                "Added %s page [%s] %s",
                kind,
                source_label(page, self._official),
                page.final_url or page.url,
            )

    def _budget_exhausted(self) -> bool:
        """Whether the run has hit its total-token ceiling (if one is set)."""
        if self._token_tracker is None or not self.max_total_tokens:
            return False
        if self._over_soft_token_budget():
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
        glossary: str,
    ) -> str:
        """Score a section and revise it (:func:`revise_section`) while it
        stays below the threshold or has a blocking issue: one the evaluator
        names (a factual error, an internal contradiction, an unsupported
        flag or default, a structural self-reference) or a writing rule it
        breaks in code (:func:`review_plan.rule_issues`)."""
        # The evaluator's blocking issues the revisions leave, for grounding.
        self._leftover_blocking = []
        if self.max_revision_attempts <= 0:
            # No revision budget: skip evaluation entirely (no wasted LLM call).
            return markdown

        async def evaluate(text: str) -> tuple[int, str, list[str], list[str]]:
            score, feedback, blocking, wording = await self._evaluate_section(
                topic=topic,
                planned=planned,
                section_label=section_label,
                markdown=text,
                previous_sections=previous_sections,
                sources=sources,
                final_outline=final_outline,
                owned_elsewhere=owned_elsewhere,
                glossary=glossary,
            )
            feedback = feedback_or_generic(score, feedback, self.min_quality_score)
            return score, feedback, blocking, wording

        async def write(directive: str) -> str:
            return await self._write_one_section(
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
                glossary=glossary,
                depth_directive=(
                    f"{planned.depth_directive(self.language, self._guide_type)}\n\n{directive}"
                ),
            )

        async def revise(feedback: str, draft: str) -> str:
            # The writer edits the draft the feedback refers to: a blind
            # rewrite at temperature 0 tends to reproduce the rejected one.
            return await write(REVISING.format(feedback=feedback, draft=draft))

        async def fix(request: str, draft: str) -> str:
            return await write(FIXING.format(request=request, draft=draft))

        best = await revise_section(
            markdown,
            evaluate=evaluate,
            revise=revise,
            length=lambda text: length_report(text, planned.depth, self.language),
            min_score=self.min_quality_score,
            max_attempts=self.max_revision_attempts,
            label=str(section_number),
            fix=fix,
        )
        if note := best.below_bar_note("Guide", planned.title, self.min_quality_score):
            self.notes.append(note)
        self._leftover_blocking = best.blocking
        return best.text

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
        owned_elsewhere: str,
        glossary: str,
    ) -> tuple[int, str, list[str], list[str]]:
        """Return (quality_score 0-100, improvement_feedback, blocking_issues,
        wording_issues) for a section, judged against the sources it was written from and the
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
                "owned_elsewhere": owned_elsewhere,
                "glossary": glossary,
                "length_report": length_report(markdown, planned.depth, self.language),
                "language": self.language,
            }
        )
        score = parse_quality_score((result or {}).get("quality_score"))
        feedback = ((result or {}).get("improvement_feedback") or "").strip()
        return (
            score,
            feedback,
            parse_lines((result or {}).get("blocking_issues")),
            parse_lines((result or {}).get("wording_issues")),
        )

    @RetryableBase._retry("tech_guide_grounding")
    async def _ground_section(
        self,
        section_markdown: str,
        sources: str,
        glossary: str,
        blocking: list[str] | None = None,
    ) -> str:
        """Fact-check a drafted section, returning the cleaned version; the
        evaluator's leftover ``blocking`` issues are corrected or removed.

        Falls back to the original draft if the grounding pass returns nothing,
        so verification can only improve—never drop—a section.
        """
        result = await self.grounding_chain.ainvoke(
            {
                "section": section_markdown,
                "sources": sources,
                "glossary": glossary,
                "language": self.language,
                "blocking_issues": blocking_issues_block(blocking or [], "<sources>"),
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
        glossary: str,
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
            "glossary": glossary,
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
        """URL, title and opening of each page within ``max_chars``: the
        user's reference material (labelled) and then the other pages, each
        group with at least half of ``max_chars`` when the other needs it, so
        neither a long docs crawl nor many references push the other out."""

        def digest(pages: list[PageContent], label: str) -> str:
            return "\n".join(
                f"- {page.url}{label}\n  {page.title}\n  "
                + page.text[:500].replace("\n", " ")
                for page in pages
            )

        references = digest(
            [p for p in corpus.pages if p.user_reference], f" ({REFERENCE_LABEL})"
        )
        others = digest([p for p in corpus.pages if not p.user_reference], "")
        references = references[: max(max_chars // 2, max_chars - len(others) - 1)]
        others = others[: max_chars - len(references) - 1]
        text = "\n".join(part for part in (references, others) if part)
        # Untrusted page text → defang prompt-fence tags before interpolation.
        return neutralize_prompt_tags(text)

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
