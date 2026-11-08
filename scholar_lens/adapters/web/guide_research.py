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
import logging
import re
from collections.abc import Collection, Mapping
from typing import Any
from urllib.parse import urlparse

import boto3
from langchain_core.runnables import Runnable

from scholar_lens.adapters.bedrock.chat_models import BedrockLanguageModelFactory
from scholar_lens.adapters.bedrock.output_parsers import HTMLTagOutputParser
from scholar_lens.adapters.bedrock.prompts.base import BasePrompt
from scholar_lens.adapters.bedrock.prompts.tech_guide import (
    TechGuideRelevancePrompt,
    TechGuideResearchPlanPrompt,
    TechGuideTaxonomyPrompt,
)
from scholar_lens.adapters.retry import RetryableBase, is_access_error
from scholar_lens.adapters.web.research import (
    MAX_REFERENCE_PAGES,
    MAX_REFERENCE_SEARCHES,
    WebResearcher,
    search_chain,
)
from scholar_lens.adapters.web.snapshot import (
    CorpusSnapshot,
    SnapshotPolicy,
    read_snapshot,
    save_snapshot,
    snapshot_key,
)
from scholar_lens.domain.constants import LanguageModelId
from scholar_lens.domain.guide.authority import (
    mentions_any,
    official_reference,
    third_party_identifiers,
)
from scholar_lens.domain.guide.evidence_guide import install_requirements
from scholar_lens.domain.guide.models import (
    GuideMaterial,
    NotTechnicalContentError,
    TechGuide,
)
from scholar_lens.domain.prompt_safety import neutralize_prompt_tags
from scholar_lens.domain.research import (
    REFERENCE_LABEL,
    PageContent,
    ResearchCorpus,
    SourceFetchError,
    arxiv_paper_urls,
    page_key,
    product_names,
    source_label,
)
from scholar_lens.domain.review_plan import digest_sections
from scholar_lens.domain.sources import (
    OFFICIAL_AUTHORITIES,
    DomainClasses,
    IndexUnreachable,
    OfficialDocs,
    ReleaseLookup,
    SourceSet,
    TagLookup,
    search_exclusions,
)
from scholar_lens.domain.tags import controlled_tag_problems
from scholar_lens.domain.text import is_affirmative, is_placeholder
from scholar_lens.domain.usage import TokenBudgetExceeded
from scholar_lens.domain.usage_context import staged
from scholar_lens.domain.validation import description_markup, description_problems

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE: str = "Korean"
# The third-party identifier lookup before writing uses a third of the
# per-guide reference caps.
_IDENTIFIER_REFERENCE_PAGES: int = MAX_REFERENCE_PAGES * 2 // 3 // 2
_IDENTIFIER_REFERENCE_SEARCHES: int = MAX_REFERENCE_SEARCHES * 2 // 3 // 2


def _normalized(name: str) -> str:
    """A distribution or command name as PEP 503 compares them."""
    return re.sub(r"[-_.]+", "-", name).lower()


_NO_DOCS_NOTE = (
    "No documentation URL was given, so no page counted as official: verify "
    "the guide's commands, flags and defaults against the product's current "
    "docs."
)


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
    # Why the last gather is incomplete (planning failed, a search failed, a
    # requested page was not read): such a corpus is not saved for reuse.
    _gaps: list[str]
    # The package whose latest release set the target version ("" when the
    # docs URL pinned it or none was looked up).
    _package: str = ""

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
        release_lookup: ReleaseLookup | None = None,
        repo_tags: TagLookup | None = None,
        max_third_party_pages: int | None = None,
        min_secondary_pages: int = 0,
        official_scope_link_share: float | None = None,
        demoted_domains: Mapping[str, Collection[str]] | None = None,
        excluded_domains: Collection[str] = (),
    ) -> None:
        """Args beyond the obvious:
        release_lookup: A package's latest release on its index (the
            run's target version when the docs URL pins none); None
            looks nothing up. An index that cannot answer leaves a run
            note and keeps the corpus from being saved for reuse.
        repo_tags: A repository's tags (``host/owner/name``); None
            looks nothing up.
        max_third_party_pages: At most this many third-party and demoted
            search-result pages (None: no cap).
        min_secondary_pages: Fetched search-result slots kept for primary
            non-official pages (papers, design docs and posts) while such
            results remain; within the third-party cap.
        official_scope_link_share: The share of a requested page's
            same-host links its official scope must hold (None: the
            page's first path segment).
        demoted_domains: Domain classes (forum, social, aggregator …) ->
            host patterns whose pages rank last and count as third-party.
        excluded_domains: Domains the search backends leave out.
        """
        self.relevance_model_id = relevance_model_id
        self.release_lookup = release_lookup
        self.repo_tags = repo_tags
        self.max_third_party_pages = max_third_party_pages
        self.min_secondary_pages = min_secondary_pages
        self.official_scope_link_share = official_scope_link_share
        self.demoted = DomainClasses.from_config(demoted_domains or {})
        self.excluded_domains = search_exclusions(excluded_domains)
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
        self._gaps = []
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
        snapshots: SnapshotPolicy | None = None,
    ) -> GuideMaterial:
        """Everything a guide is written from: the docs (and sub-pages), the
        user's reference material, the planned searches' top pages and the
        official reference pages for third-party identifiers, after the
        relevance gate (raises :class:`NotTechnicalContentError`).

        With ``snapshots``, a corpus gathered from the same inputs (URLs,
        references, explicit queries and the settings that change what is
        gathered) within the policy's TTL is reused while the package
        release it targeted is still the latest, and a newly gathered one is
        saved when it is complete (planning ran, no search failed, every
        requested page was read)."""
        if snapshots is None:
            return await self._gather(
                urls,
                reference_urls=reference_urls or [],
                discover_subpages=discover_subpages,
                search_queries=search_queries or [],
            )
        key = snapshot_key(
            self._gather_inputs(
                urls, reference_urls or [], discover_subpages, search_queries or []
            )
        )
        found = await asyncio.to_thread(
            read_snapshot,
            snapshots,
            key,
            demoted=self.demoted,
            repo_tags=self.repo_tags,
        )
        if found is not None and await asyncio.to_thread(self._release_holds, found):
            logger.info("Reusing the corpus snapshot %s.", key)
            return self._restore(found)
        start = len(self.notes)
        material = await self._gather(
            urls,
            reference_urls=reference_urls or [],
            discover_subpages=discover_subpages,
            search_queries=search_queries or [],
        )
        snapshot = CorpusSnapshot(
            topic=material.topic,
            corpus=material.corpus,
            references=material.references,
            queries=material.queries,
            products=self._products,
            sources=self._sources,
            notes=[*self.notes[start:], *self.researcher.search_notes()],
            reference_ids=list(self._reference_ids),
            reference_searches=self._reference_searches,
            reference_urls=[p.url for p in self._reference_pages],
            package=self._package,
        )
        if self._gaps:
            logger.info(
                "Not saving the corpus snapshot %s: the gather is incomplete (%s).",
                key,
                "; ".join(self._gaps),
            )
            return material
        try:
            await asyncio.to_thread(save_snapshot, snapshots, key, snapshot)
        except OSError as e:  # the corpus is in hand; only a rerun gathers again
            logger.warning("Could not save the corpus snapshot %s: %s", key, e)
        return material

    def _gather_inputs(
        self,
        urls: list[str],
        references: list[str],
        discover_subpages: bool,
        queries: list[str],
    ) -> dict[str, object]:
        """What decides a gathered corpus: the request and every setting
        that changes what is fetched, searched or how it is recorded."""
        return {
            "urls": list(urls),
            "references": list(references),
            "discover_subpages": discover_subpages,
            "search_queries": list(queries),
            "relevance_model_id": str(self.relevance_model_id),
            "auto_research": self.auto_research,
            "max_research_queries": self.max_research_queries,
            "fetch_top_results": self.fetch_top_results,
            "max_third_party_pages": self.max_third_party_pages,
            "min_secondary_pages": self.min_secondary_pages,
            "official_scope_link_share": self.official_scope_link_share,
            "demoted_domains": {k: list(v) for k, v in self.demoted.classes.items()},
            "excluded_domains": list(self.excluded_domains),
            "search_backend": search_chain(self.researcher.search_provider),
            "max_subpages": self._max_subpages(),
        }

    def _release_holds(self, snapshot: CorpusSnapshot) -> bool:
        """Whether a snapshot's target release is still the one a gather
        would resolve: the latest release of the package it looked up (a
        pinned docs URL, or no lookup, always holds)."""
        if not snapshot.package or self.release_lookup is None:
            return True
        try:
            latest = self.release_lookup(snapshot.package)
        except IndexUnreachable as e:
            # The snapshot's release was the latest when it was saved
            # (within its TTL); whether a newer one exists is unknown.
            self.notes.append(
                f"The package index could not be reached ({e}): the saved "
                f"corpus targets {snapshot.package} "
                f"{snapshot.sources.target.release}, not checked against the "
                "latest release."
            )
            return True
        if latest == snapshot.sources.target.release:
            return True
        logger.info(
            "The corpus snapshot targets %s %s, the latest is %s: gathering again.",
            snapshot.package,
            snapshot.sources.target.release,
            latest,
        )
        return False

    def _gap(self, note: str) -> None:
        """A run note for a part of the gather that is missing, which also
        keeps the corpus from being saved for reuse."""
        self.notes.append(note)
        self._gaps.append(note)

    def _note_abstract_pages(self, corpus: ResearchCorpus) -> None:
        """A run note for the arXiv papers read from their abstract page
        because their HTML rendering could not be used (with the reason)."""
        fallbacks = getattr(self.researcher, "arxiv_fallbacks", None)
        reasons = fallbacks if isinstance(fallbacks, dict) else {}
        found = []
        for page in corpus.pages:
            url = page.final_url or page.url
            paper = arxiv_paper_urls(url)
            if paper and "/abs/" in urlparse(url).path:
                found.append(f"{url} ({reasons.get(paper[0], 'no HTML rendering')})")
        if found:
            self.notes.append(
                "arXiv papers read from their abstract only (no usable HTML "
                "rendering): " + ", ".join(found)
            )

    def _max_subpages(self) -> int | None:
        found = getattr(self.researcher, "max_subpages", None)
        return found if isinstance(found, int) else None

    def _restore(self, snapshot: CorpusSnapshot) -> GuideMaterial:
        """The generator's state and material as a saved gather left them."""
        self._sources = snapshot.sources
        self._products = snapshot.products
        self._reference_ids = list(snapshot.reference_ids)
        self._reference_searches = snapshot.reference_searches
        looked_up = set(snapshot.reference_urls)
        self._reference_pages = [p for p in snapshot.corpus.pages if p.url in looked_up]
        self.notes += [
            *snapshot.notes,
            f"Sources reused from the corpus gathered at {snapshot.gathered_at} "
            f"({len(snapshot.corpus.pages)} pages); rerun with --fresh-sources "
            "to gather them again.",
        ]
        return GuideMaterial(
            topic=snapshot.topic,
            corpus=snapshot.corpus,
            references=snapshot.references,
            queries=snapshot.queries,
        )

    async def _gather(
        self,
        urls: list[str],
        *,
        reference_urls: list[str],
        discover_subpages: bool,
        search_queries: list[str],
    ) -> GuideMaterial:
        # One spelling per page (and per arXiv paper: abs, pdf, any version).
        held = {page_key(u) for u in urls}
        references: list[str] = []
        for url in reference_urls:
            if (key := page_key(url)) not in held:
                held.add(key)
                references.append(url)
        if not urls and not references:
            raise ValueError("At least one source URL is required.")
        # The requested docs are the authority for flags, defaults and names.
        self._sources = SourceSet.from_urls(urls)
        self._reference_pages, self._reference_ids = [], []
        self._reference_searches = 0
        self._gaps, self._package = [], ""
        failed_before = self.researcher.search_provider.failed_searches
        failures = getattr(self.researcher, "tag_lookup_failures", None)
        tags_before = len(failures) if isinstance(failures, list) else 0

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
            read = {page_key(p.url) for p in corpus.pages}
            if missing := [u for u in urls if page_key(u) not in read]:
                self._gap(
                    "Documentation pages that could not be read (left out): "
                    + ", ".join(missing)
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
        queries = list(search_queries)
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
                sources=self._sources,
                max_third_party=self.max_third_party_pages,
                min_secondary=self.min_secondary_pages,
                exclude=self.excluded_domains,
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

        self._note_abstract_pages(corpus)
        if self.researcher.search_provider.failed_searches > failed_before:
            # The researcher's search notes say how many.
            self._gaps.append("web searches failed")
        failures = getattr(self.researcher, "tag_lookup_failures", None)
        if isinstance(failures, list) and (unknown := failures[tags_before:]):
            self._gap(
                "The tags of "
                + ", ".join(dict.fromkeys(unknown))
                + " could not be listed (rate limit or error): their pages at "
                "refs other than the default branch count as development code."
            )
        topic = (
            planned_topic
            if planned_topic and not is_placeholder(planned_topic)
            else gate_topic
        )
        return GuideMaterial(
            topic=topic, corpus=corpus, references=references, queries=queries
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
        if not corpus.pages:
            raise NotTechnicalContentError(
                "None of the supplied URLs returned readable content."
            )
        read = {page_key(p.url) for p in fetched.pages}
        missing = [u for u in references if page_key(u) not in read]
        if missing:
            self._gap(
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

        Best-effort: once the retry policy gives up, a planning failure
        degrades to the seed pages (a run note, and the corpus is not saved
        for reuse); a refused permission or a spent budget still raises.
        """
        try:
            result = await self._invoke_plan(
                {
                    "sources": self._sources_digest(corpus),
                    "max_queries": self.max_research_queries,
                }
            )
        except Exception as e:  # noqa: BLE001 - planning is best-effort
            if is_access_error(e) or isinstance(e, TokenBudgetExceeded):
                raise
            logger.warning("Research planning failed; using seed URLs only: %s", e)
            self._gap(
                f"Research planning failed ({type(e).__name__}: {str(e)[:160]}); "
                "the corpus is the seed pages only."
            )
            return None, []
        topic = (result.get("topic") or "").strip()
        queries = self._parse_queries(result.get("queries", ""))
        logger.info(
            "Research plan: topic='%s', %d queries: %s",
            topic,
            len(queries),
            queries,
        )
        return (topic or None), queries

    @RetryableBase._retry("tech_guide_research_plan")
    async def _invoke_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = await self.research_plan_chain.ainvoke(payload)
        return result

    @RetryableBase._retry("tech_guide_relevance")
    async def _invoke_relevance(self, payload: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = await self.relevance_chain.ainvoke(payload)
        return result

    @staged("relevance")
    async def _assert_relevant(self, corpus: ResearchCorpus) -> str:
        sources = self._sources_digest(corpus)
        result = await self._invoke_relevance({"sources": sources})
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
        they returned. Its target version is the release the requested docs
        URL pins, else the latest release of the product's package (one the
        docs install under a product name), else the release a rolling docs
        link landed on."""
        draft = SourceSet.from_urls(
            urls, docs, link_share=self.official_scope_link_share
        )
        release, package = None, ""
        if self.release_lookup is not None and not draft.target.pinned:
            official_text = "\n".join(
                p.text
                for p in docs
                if draft.classify(p.final_url or p.url).authority
                in OFFICIAL_AUTHORITIES
            )
            products = {_normalized(p) for p in self._products}
            for name, _ in install_requirements(official_text):
                if _normalized(name) not in products:
                    continue
                try:
                    release = self.release_lookup(name)
                except IndexUnreachable as e:
                    self._gap(
                        f"The package index could not be reached for {name} "
                        f"({e}): the target version is not its latest release, "
                        "and pages are not marked by version against it."
                    )
                    continue
                if release:
                    package = name
                    break
        sources = SourceSet.from_urls(
            urls,
            docs,
            package_release=release,
            link_share=self.official_scope_link_share,
            demoted=self.demoted,
            repo_tags=self.repo_tags,
        ).with_repositories(docs)
        target = sources.target
        self._package = package if release and target.release == release else ""
        if target.release:
            origin = (
                "the requested docs URL"
                if target.pinned
                else f"the latest {package} release" if package else "the docs"
            )
            logger.info("Target version %s (%s).", target.release, origin)
            self.notes.append(
                f"Target version {target.release} ({origin}): docs and "
                "repository pages at other versions are marked as another "
                "version."
            )
        return sources

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
