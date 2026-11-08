"""Technical guide acquisition and filing.

Given documentation URLs (library/framework/platform docs: the authority for
commands, flags and defaults) and optional reference material (a paper, design
doc or deep-dive post the user wants the guide to build on),
:class:`TechGuideGenerator`:
1. Plans deep research — derives the topic and a set of web-search queries
   (usage, internals and design, production integration) from the inputs, then
   researches the docs URLs (+ in-scope sub-pages), the reference material and
   the planned/explicit search queries into a corpus so the guide draws on
   broad, complementary material rather than translating a single page.
2. Gates on relevance — refuses to write if the docs (or, without docs, the
   reference material) are not technical developer material (raises
   :class:`NotTechnicalContentError`).
3. Settles flags and arguments only third-party pages give from the official
   reference pages.

The evidence engine (:mod:`guide_writer`) writes the guide from the gathered
corpus; :meth:`TechGuideGenerator.classify` files the finished guide into the
blog's taxonomy.
"""

from __future__ import annotations

import asyncio
import functools
import re
from dataclasses import dataclass, field
from typing import Any

import boto3
from langchain_core.runnables import Runnable

from ..application.usage import staged
from ..domain.review_plan import digest_sections
from ..domain.tags import controlled_tag_problems
from ..domain.validation import description_markup, description_problems
from .constants import LanguageModelId
from .guide_authority import (
    mentions_any,
    official_reference,
    third_party_identifiers,
)
from .logger import logger
from .prompt_safety import neutralize_prompt_tags
from .prompts import (
    BasePrompt,
    TechGuideRelevancePrompt,
    TechGuideResearchPlanPrompt,
)
from .prompts.tech_guide import REFERENCE_LABEL, TechGuideTaxonomyPrompt
from .source_set import SourceSet
from .utils import (
    BedrockLanguageModelFactory,
    HTMLTagOutputParser,
    RetryableBase,
    is_affirmative,
    is_placeholder,
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
# The third-party identifier lookup before writing uses a third of the
# per-guide reference caps.
_IDENTIFIER_REFERENCE_PAGES: int = MAX_REFERENCE_PAGES * 2 // 3 // 2
_IDENTIFIER_REFERENCE_SEARCHES: int = MAX_REFERENCE_SEARCHES * 2 // 3 // 2


class NotTechnicalContentError(Exception):
    """Raised when the supplied URLs are not technical documentation."""


_NO_DOCS_NOTE = (
    "No documentation URL was given, so no page counted as official: verify "
    "the guide's commands, flags and defaults against the product's current "
    "docs."
)


@dataclass
class TechGuide:
    topic: str
    body: str
    source_urls: list[str] = field(default_factory=list)
    # Page title of each source URL, for the source list's link text.
    source_titles: dict[str, str] = field(default_factory=dict)
    # Degraded-output notes for the PR checklist and the job result.
    notes: list[str] = field(default_factory=list)
    # Problems the post may not be published with as is (listed first).
    blocking: list[str] = field(default_factory=list)
    # Images the writer was offered; any other image in the body is dropped.
    image_urls: list[str] = field(default_factory=list)
    # The search description (``classify``); empty lets the blog derive one.
    description: str = ""


@dataclass
class GuideMaterial:
    """What :meth:`TechGuideGenerator.gather` collected."""

    topic: str
    corpus: ResearchCorpus
    # The user's reference URLs not already among the docs URLs.
    references: list[str] = field(default_factory=list)


class TechGuideGenerator(RetryableBase):
    """Gathers a guide's sources and files the finished guide."""

    # Decides each gathered page's source record (authority, version, date,
    # language) once; the requested docs are the authority for flags and
    # defaults.
    _sources: SourceSet = SourceSet(OfficialDocs(frozenset(), frozenset()))
    # The names the docs' commands go by (set by the relevance gate).
    _products: tuple[str, ...] = ()
    # Official reference pages fetched this run, the identifiers they were
    # looked up for and the searches spent (capped per guide).
    _reference_pages: list[PageContent]
    _reference_ids: list[str]
    _reference_searches: int = 0

    def __init__(
        self,
        relevance_model_id: LanguageModelId,
        boto_session: boto3.Session,
        *,
        researcher: WebResearcher | None = None,
        language: str = DEFAULT_LANGUAGE,
        title_max_length: int,
        auto_research: bool = True,
        max_research_queries: int,
        fetch_top_results: int,
        callbacks: list[Any] | None = None,
        llm_factory: BedrockLanguageModelFactory | None = None,
    ) -> None:
        self.language = language
        # The blog's title limit (the conventions' ``seo_title_max_length``).
        self.title_max_length = title_max_length
        # When True (and no explicit queries are supplied), the planner derives
        # web-search queries from the seed docs so the corpus is broadened beyond
        # the seed page before writing.
        self.auto_research = auto_research
        self.max_research_queries = max_research_queries
        self.fetch_top_results = fetch_top_results
        self.callbacks = callbacks or []
        self.notes: list[str] = []
        self._reference_pages, self._reference_ids = [], []
        self.researcher = researcher or WebResearcher()
        self.llm_factory = llm_factory or BedrockLanguageModelFactory(
            boto_session=boto_session
        )

        # The relevance gate, research planner and taxonomy are cheap and use
        # the relevance model tier (no thinking).
        self.relevance_chain = self._build_chain(
            TechGuideRelevancePrompt, relevance_model_id
        )
        self.research_plan_chain = self._build_chain(
            TechGuideResearchPlanPrompt, relevance_model_id
        )
        self.taxonomy_chain = self._build_chain(
            TechGuideTaxonomyPrompt, relevance_model_id
        )

    def _build_chain(
        self, prompt_cls: type[BasePrompt], model_id: LanguageModelId
    ) -> Runnable:
        """Prompt | model | HTMLTagOutputParser, with shared callbacks attached
        (the TokenUsageTracker records every chain's usage)."""
        model = self.llm_factory.get_model(
            model_id, temperature=0.0, callbacks=self.callbacks or None
        )
        return (
            prompt_cls.get_prompt()
            | model
            | HTMLTagOutputParser(tag_names=prompt_cls.output_variables or [])
        )

    @property
    def official(self) -> OfficialDocs:
        """The requested docs (set by :meth:`gather`)."""
        return self._sources.official

    @property
    def sources(self) -> SourceSet:
        """The run's source set (set by :meth:`gather`)."""
        return self._sources

    @property
    def products(self) -> tuple[str, ...]:
        """The names the docs' commands go by (set by :meth:`gather`)."""
        return self._products

    async def gather(
        self,
        urls: list[str],
        *,
        reference_urls: list[str] | None = None,
        discover_subpages: bool = True,
        search_queries: list[str] | None = None,
    ) -> GuideMaterial:
        """Everything a guide is written from: the docs (and sub-pages), the
        user's reference material, the planned searches' top pages and the
        official reference pages for third-party identifiers, after the
        relevance gate (raises :class:`NotTechnicalContentError`)."""
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
        self._sources = SourceSet.from_urls(urls)
        self._reference_pages, self._reference_ids = [], []
        self._reference_searches = 0

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
        else:
            self.notes.append(_NO_DOCS_NOTE)
        docs = ResearchCorpus(pages=list(corpus.pages))
        await self._add_material(corpus, references)

        # Judge the docs alone (the reference material when there are none):
        # search results never redefine what the user asked for, and a
        # rejected input spends nothing on searches.
        gate_topic = await self._assert_relevant(docs if docs.pages else corpus)

        # Every page's record is decided here, once: where the requested
        # pages landed (redirects, a pinned version behind "stable") is
        # official too, and so are the repositories the docs link to.
        self._sources = self._source_set(urls, docs.pages)
        self._record(corpus.pages, "seed")

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
            self._record(corpus.pages[before:], "search")
        elif queries:
            logger.info(
                "Search backend has no results; skipping %d queries.", len(queries)
            )

        # Flags and arguments only third-party pages give: settle them from the
        # official reference before writing: the writer leaves out a flag no
        # official source backs.
        await self._add_reference(
            corpus,
            third_party_identifiers(corpus, self._products),
            max_pages=_IDENTIFIER_REFERENCE_PAGES,
            max_searches=_IDENTIFIER_REFERENCE_SEARCHES,
        )

        topic = (
            planned_topic
            if planned_topic and not is_placeholder(planned_topic)
            else gate_topic
        )
        return GuideMaterial(topic=topic, corpus=corpus, references=references)

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

    async def classify(
        self,
        guide: TechGuide,
        *,
        category_options: list[str],
        controlled_tags: list[str],
    ) -> dict[str, str]:
        """Propose the blog topic, tags, SEO title and search description for
        a finished guide from its body's headings and first sentences, so no
        tag names something the guide does not teach (the caller normalizes
        and validates them against the post conventions). The description is
        checked by the post validator's description rules with one repair
        round and set on ``guide.description``; one the validator would
        reject (markup, not longer than the title) is dropped so the blog
        derives one."""
        payload: dict[str, Any] = {
            "title": guide.topic,
            "outline": digest_sections([guide.body]),
            "category_options": ", ".join(category_options),
            "controlled_tags": ", ".join(controlled_tags),
            "title_max_length": self.title_max_length,
            "language": self.language,
            "repair_note": "",
        }
        result: dict[str, str] = {}
        problems: list[str] = []
        description = ""
        for attempt in range(2):
            result = await self._classify(payload)
            description = " ".join((result.get("description") or "").split())
            if not (
                problems := description_problems(description, guide.body)
                + controlled_tag_problems(
                    result.get("controlled_tags"), controlled_tags
                )
            ):
                break
            logger.warning(
                "Guide taxonomy answer unusable (attempt %d): %s", attempt + 1, problems
            )
            payload["repair_note"] = (
                "Your previous answer could not be used: "
                + "; ".join(problems)
                + ". Answer all five parts again."
            )
        else:
            # The guide's notes reach the PR.
            guide.notes.append(
                "Guide description still breaks the rules after a repair: "
                + "; ".join(problems)
            )
            if description_markup(description) or len(description) <= len(guide.topic):
                description = ""
        guide.description = description
        result["description"] = description
        return result

    @RetryableBase._retry("tech_guide_taxonomy")
    @staged("finalize")
    async def _classify(self, payload: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = await self.taxonomy_chain.ainvoke(payload)
        return result

    @staged("research_plan")
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

    @staged("relevance")
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
        self._products = tuple(product_names(result.get("commands", ""), corpus.pages))
        logger.info(
            "Relevance gate passed. Topic: '%s'; commands: %s", topic, self._products
        )
        return topic

    @staged("official_lookup")
    async def _add_reference(
        self,
        corpus: ResearchCorpus,
        identifiers: list[str],
        *,
        max_pages: int = MAX_REFERENCE_PAGES,
        max_searches: int = MAX_REFERENCE_SEARCHES,
    ) -> list[PageContent]:
        """Fetch official reference pages for identifiers not looked up yet,
        within the per-guide caps (or the lower running totals
        ``max_pages``/``max_searches``), and return the pages added."""
        identifiers = [i for i in identifiers if i not in self._reference_ids]
        pages_left = min(max_pages, MAX_REFERENCE_PAGES) - len(self._reference_pages)
        if not identifiers or pages_left <= 0:
            return []
        searches_left = (
            min(max_searches, MAX_REFERENCE_SEARCHES) - self._reference_searches
        )
        # Only pages that settle an identifier, as the unverified check reads
        # them, are kept.
        official_text = official_reference(corpus)
        settles = functools.partial(
            mentions_any, identifiers=identifiers, official_text=official_text
        )
        added = await asyncio.to_thread(
            self.researcher.add_official_reference,
            corpus,
            self._sources,
            identifiers,
            product=self._products[0] if self._products else "",
            max_searches=searches_left,
            max_pages=pages_left,
            settles=settles,
        )
        if self.researcher.search_provider.supports_search:
            self._reference_searches += min(searches_left, len(identifiers))
        self._reference_ids += identifiers
        self._reference_pages += added
        self._log_added("reference-lookup", added)
        return added

    def _source_set(self, urls: list[str], docs: list[PageContent]) -> SourceSet:
        """The run's source set from the requested URLs and the docs pages
        they returned."""
        return SourceSet.from_urls(urls, docs).with_repositories(docs)

    def _record(self, pages: list[PageContent], kind: str) -> None:
        """Record ``pages`` (:meth:`SourceSet.record`) and log them."""
        self._sources.record_all(pages)
        self._log_added(kind, pages)

    def _log_added(self, kind: str, pages: list[PageContent]) -> None:
        """One INFO line per page added to the corpus: how it came in, its
        record and its URL."""
        for page in pages:
            logger.info(
                "Added %s page [%s%s%s] %s",
                kind,
                page.authority or source_label(page),
                f" {page.version}" if page.version else "",
                f" {page.published}" if page.published else "",
                page.final_url or page.url,
            )

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
