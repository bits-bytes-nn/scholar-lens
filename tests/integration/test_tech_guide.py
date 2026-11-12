"""Tests for the technical-guide generator's acquisition and filing (LLM
chains stubbed)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import boto3
import pytest

from scholar_lens.adapters.bedrock.prompts.tech_guide import (
    TechGuideRelevancePrompt,
    TechGuideResearchPlanPrompt,
    TechGuideTaxonomyPrompt,
)
from scholar_lens.adapters.web.guide_research import _NO_DOCS_NOTE, TechGuideGenerator
from scholar_lens.domain.constants import LanguageModelId
from scholar_lens.domain.guide.models import GuideMaterial, NotTechnicalContentError
from scholar_lens.domain.research import (
    OFFICIAL_LABEL,
    REFERENCE_LABEL,
    PageContent,
    ResearchCorpus,
    source_label,
)
from scholar_lens.sections_engine.section_loop import lost_structure
from tests.integration.fakes import FakeLLMFactory


class TestPromptsBuild:
    """Every declared input variable must appear in its template (get_prompt
    raises otherwise), so this catches a renamed/missing placeholder."""

    PROMPTS = [
        TechGuideResearchPlanPrompt,
        TechGuideRelevancePrompt,
        TechGuideTaxonomyPrompt,
    ]

    def test_get_prompt_does_not_raise(self) -> None:
        for prompt in self.PROMPTS:
            assert prompt.get_prompt() is not None, prompt.__name__


def _make_generator(*, auto_research: bool = False) -> TechGuideGenerator:
    """Construct a generator without touching Bedrock or the network.

    Auto-research defaults OFF so a test exercises only what it opts into.
    """
    researcher = MagicMock()
    researcher.add_official_reference = MagicMock(return_value=[])
    # Searches that failed (the real backends count them).
    researcher.search_provider.failed_searches = 0
    gen = TechGuideGenerator(
        LanguageModelId.CLAUDE_V4_5_HAIKU,
        boto3.Session(region_name="us-east-1"),
        researcher=researcher,
        language="English",
        title_max_length=70,
        auto_research=auto_research,
        max_research_queries=6,
        fetch_top_results=4,
        llm_factory=FakeLLMFactory(),  # type: ignore[arg-type]
    )
    gen.relevance_chain = MagicMock()
    gen.research_plan_chain = MagicMock()
    gen.taxonomy_chain = MagicMock()
    return gen


def _relevant(topic: str = "Using X") -> AsyncMock:
    return AsyncMock(
        return_value={"is_relevant": "yes", "topic": topic, "reason": "ok"}
    )


def _corpus() -> ResearchCorpus:
    return ResearchCorpus(
        pages=[PageContent("https://docs.x.com", "X Docs", "How to use X.")]
    )


class TestParseQueries:
    def test_drops_ellipsis_and_dedups(self) -> None:
        raw = (
            "1. argo cd architecture\n"
            "- argo cd vs flux\n"
            "...\n"
            "Argo CD Architecture\n"  # case-dup of the first
            "best practices"
        )
        assert TechGuideGenerator._parse_queries(raw) == [
            "argo cd architecture",
            "argo cd vs flux",
            "best practices",
        ]

    def test_skips_tag_and_punctuation_lines(self) -> None:
        assert TechGuideGenerator._parse_queries("<queries>\n…\n***\nreal query") == [
            "real query"
        ]


class TestBuildSearchProvider:
    """build_search_provider prefers Tavily > Brave > Null."""

    def _provider(self, monkeypatch, *, tavily=None, brave=None):  # type: ignore[no-untyped-def]
        from scholar_lens import bootstrap as tg

        for var in ("TAVILY_API_KEY", "BRAVE_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        if tavily:
            monkeypatch.setenv("TAVILY_API_KEY", tavily)
        if brave:
            monkeypatch.setenv("BRAVE_API_KEY", brave)
        return tg.build_search_provider()

    def test_prefers_tavily(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scholar_lens.adapters.web.research import TavilySearchProvider

        p = self._provider(monkeypatch, tavily="tvly-k")
        assert isinstance(p, TavilySearchProvider)

    def test_tavily_falls_back_to_brave_when_both_are_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.adapters.web.research import (
            BraveSearchProvider,
            FallbackSearchProvider,
            TavilySearchProvider,
        )

        p = self._provider(monkeypatch, tavily="tvly-k", brave="brave-k")
        assert isinstance(p, FallbackSearchProvider)
        assert isinstance(p.primary, TavilySearchProvider)
        assert isinstance(p.secondary, BraveSearchProvider)

    def test_falls_back_to_brave(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scholar_lens.adapters.web.research import BraveSearchProvider

        p = self._provider(monkeypatch, brave="brave-k")
        assert isinstance(p, BraveSearchProvider)

    def test_null_when_no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scholar_lens.adapters.web.research import NullSearchProvider

        p = self._provider(monkeypatch)
        assert isinstance(p, NullSearchProvider)


class TestRelevanceGate:
    async def test_relevant_returns_topic(self) -> None:
        gen = _make_generator()
        gen.relevance_chain.ainvoke = AsyncMock(
            return_value={"is_relevant": "yes", "topic": "Using X", "reason": "docs"}
        )
        topic = await gen._assert_relevant(_corpus())
        assert topic == "Using X"

    async def test_irrelevant_raises(self) -> None:
        gen = _make_generator()
        gen.relevance_chain.ainvoke = AsyncMock(
            return_value={
                "is_relevant": "no",
                "topic": "N/A",
                "reason": "marketing page",
            }
        )
        with pytest.raises(NotTechnicalContentError, match="marketing page"):
            await gen._assert_relevant(_corpus())

    async def test_relevant_but_no_topic_raises(self) -> None:
        gen = _make_generator()
        gen.relevance_chain.ainvoke = AsyncMock(
            return_value={"is_relevant": "yes", "topic": "N/A", "reason": "ok"}
        )
        with pytest.raises(NotTechnicalContentError):
            await gen._assert_relevant(_corpus())


class TestGather:
    async def test_empty_urls_rejected(self) -> None:
        gen = _make_generator()
        with pytest.raises(ValueError):
            await gen.gather([])

    async def test_no_readable_pages_rejected(self) -> None:
        gen = _make_generator()
        gen.researcher.research = MagicMock(return_value=ResearchCorpus())
        with pytest.raises(NotTechnicalContentError):
            await gen.gather(["https://x.com"])

    async def test_gather_returns_the_gated_topic_and_corpus(self) -> None:
        gen = _make_generator()
        gen.researcher.research = MagicMock(return_value=_corpus())
        gen.relevance_chain.ainvoke = _relevant()
        material = await gen.gather(["https://docs.x.com"], discover_subpages=False)
        assert isinstance(material, GuideMaterial)
        assert material.topic == "Using X"
        assert [p.url for p in material.corpus.pages] == ["https://docs.x.com"]
        assert material.references == []
        assert gen.official.hosts == frozenset({"docs.x.com"})


class TestResearchPlanning:
    async def test_plan_research_parses_topic_and_queries(self) -> None:
        gen = _make_generator(auto_research=True)
        gen.research_plan_chain.ainvoke = AsyncMock(
            return_value={
                "topic": "Argo CD",
                "queries": "1. argo cd architecture\n- argo cd vs flux\nbest practices",
            }
        )
        topic, queries = await gen._plan_research(_corpus())
        assert topic == "Argo CD"
        assert queries == [
            "argo cd architecture",
            "argo cd vs flux",
            "best practices",
        ]

    async def test_plan_research_failure_degrades_gracefully(self) -> None:
        gen = _make_generator(auto_research=True)
        gen.research_plan_chain.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        # A planning failure must not abort the run — returns no queries.
        topic, queries = await gen._plan_research(_corpus())
        assert topic is None
        assert queries == []

    async def test_auto_research_runs_searches_when_enabled(self) -> None:
        gen = _make_generator(auto_research=True)
        gen.researcher.research = MagicMock(return_value=_corpus())
        gen.researcher.run_searches = MagicMock()
        gen.research_plan_chain.ainvoke = AsyncMock(
            return_value={"topic": "X", "queries": "q1\nq2"}
        )
        gen.relevance_chain.ainvoke = _relevant()
        await gen.gather(["https://docs.x.com"], discover_subpages=False)
        gen.researcher.run_searches.assert_called_once()
        assert gen.researcher.run_searches.call_args.args[1] == ["q1", "q2"]
        # Top results are fetched into the corpus (reaches the writer, not just
        # the outline).
        assert gen.researcher.run_searches.call_args.kwargs["fetch_top"] == 4

    async def test_planning_skipped_when_backend_cannot_search(self) -> None:
        # With a no-op search backend, the planner LLM call is not made.
        gen = _make_generator(auto_research=True)
        gen.researcher.search_provider.supports_search = False
        gen.researcher.research = MagicMock(return_value=_corpus())
        gen.researcher.run_searches = MagicMock()
        gen.research_plan_chain.ainvoke = AsyncMock()
        gen.relevance_chain.ainvoke = _relevant()
        await gen.gather(["https://docs.x.com"], discover_subpages=False)
        gen.research_plan_chain.ainvoke.assert_not_awaited()
        gen.researcher.run_searches.assert_not_called()

    async def test_explicit_queries_skip_planning(self) -> None:
        gen = _make_generator(auto_research=True)
        gen.researcher.research = MagicMock(return_value=_corpus())
        gen.researcher.run_searches = MagicMock()
        gen.research_plan_chain.ainvoke = AsyncMock()
        gen.relevance_chain.ainvoke = _relevant()
        await gen.gather(
            ["https://docs.x.com"],
            discover_subpages=False,
            search_queries=["explicit query"],
        )
        # Explicit queries bypass the planner but still drive the searches.
        gen.research_plan_chain.ainvoke.assert_not_awaited()
        gen.researcher.run_searches.assert_called_once()
        assert gen.researcher.run_searches.call_args.args[1] == ["explicit query"]

    async def test_gate_judges_the_seed_pages_before_any_search(self) -> None:
        # Search results never rewrite the topic: the gate runs on the seed
        # corpus first, and the planner's topic (from the same seed) wins.
        gen = _make_generator(auto_research=True)
        order: list[str] = []
        gen.researcher.research = MagicMock(return_value=_corpus())
        gen.researcher.run_searches = MagicMock(
            side_effect=lambda *a, **k: order.append("search")
        )

        async def relevance(_inputs: dict) -> dict:
            order.append("gate")
            return {"is_relevant": "yes", "topic": "X Internals", "reason": "ok"}

        gen.relevance_chain.ainvoke = AsyncMock(side_effect=relevance)
        gen.research_plan_chain.ainvoke = AsyncMock(
            return_value={"topic": "Getting Started with X", "queries": "q1"}
        )
        material = await gen.gather(["https://docs.x.com"], discover_subpages=False)
        assert order == ["gate", "search"]
        assert material.topic == "Getting Started with X"

    async def test_placeholder_planner_topic_falls_back_to_the_gate(self) -> None:
        gen = _make_generator(auto_research=True)
        gen.researcher.research = MagicMock(return_value=_corpus())
        gen.researcher.run_searches = MagicMock()
        gen.relevance_chain.ainvoke = _relevant()
        gen.research_plan_chain.ainvoke = AsyncMock(
            return_value={"topic": "N/A", "queries": "q1"}
        )
        material = await gen.gather(["https://docs.x.com"], discover_subpages=False)
        assert material.topic == "Using X"

    async def test_third_party_flags_are_looked_up_before_writing(self) -> None:
        gen = _make_generator()
        corpus = ResearchCorpus(
            pages=[
                PageContent("https://docs.x.com", "X Docs", "```\nx serve\n```"),
                PageContent(
                    "https://blog.y.io/x", "Blog", "```\nx serve --fast-mode\n```"
                ),
            ]
        )
        gen.researcher.research = MagicMock(return_value=corpus)
        gen.relevance_chain.ainvoke = AsyncMock(
            return_value={
                "is_relevant": "yes",
                "topic": "X",
                "commands": "x",
                "reason": "ok",
            }
        )
        await gen.gather(["https://docs.x.com"], discover_subpages=False)
        gen.researcher.add_official_reference.assert_called_once()
        assert gen.researcher.add_official_reference.call_args.args[2] == [
            "--fast-mode"
        ]
        assert gen._reference_ids == ["--fast-mode"]

    async def test_the_blogs_title_limit_reaches_the_taxonomy(self) -> None:
        from scholar_lens.domain.guide.models import TechGuide

        gen = _make_generator()
        gen.title_max_length = 55
        gen.taxonomy_chain.ainvoke = AsyncMock(return_value={})
        await gen.classify(
            TechGuide(topic="X", body="## A\n\nb."),
            category_options=["A"],
            controlled_tags=["B"],
        )
        sent = gen.taxonomy_chain.ainvoke.await_args.args[0]
        assert sent["title_max_length"] == 55
        prompt = TechGuideTaxonomyPrompt
        rendered = prompt.get_prompt().format_messages(
            **sent | {name: "x" for name in prompt.input_variables if name not in sent}
        )
        text = " ".join(str(m.content) for m in rendered)
        assert "55 characters" in text and "70" not in text


class TestProseChars:
    def test_tables_and_images_are_not_prose(self) -> None:
        from scholar_lens.sections_engine.section_loop import prose_chars

        md = (
            "## 제목\n\n"
            + "가" * 100
            + "\n\n| a | b |\n|---|---|\n| "
            + "x" * 500
            + " | y |\n\n![그림](https://x.io/a.png)\n"
        )
        assert prose_chars(md) == len("## 제목 " + "가" * 100)

    @pytest.mark.parametrize(
        ("markdown", "prose"),
        [
            ("> | a | b |\n> |---|---|\n> | " + "x" * 300 + " | y |\n", ""),
            # A table without outer pipes.
            ("a | b\n--- | ---\n" + "x" * 300 + " | y\n\n뒤 문단.\n", "뒤 문단."),
            # A pipe in a sentence and a rule are prose.
            ("a | b 비교입니다.\n\n---\n", "a | b 비교입니다. ---"),
            ("[![logo](https://x.io/a.png)](https://x.io)\n", ""),
            ("![그림](https://x.io/a.png) 설명입니다.\n", "설명입니다."),
            ("> 인용문입니다.\n", "인용문입니다."),
        ],
    )
    def test_quoted_tables_linked_images_and_text_beside_images(
        self, markdown: str, prose: str
    ) -> None:
        from scholar_lens.sections_engine.section_loop import prose_chars

        assert prose_chars("본문 " + "\n" + markdown) == len(f"본문 {prose}".strip())


class TestReferenceLookup:
    async def test_reference_lookup_logs_each_page_with_its_label(self) -> None:
        from unittest.mock import patch

        from scholar_lens.domain.sources import SourceSet

        gen = _make_generator()
        gen._sources = SourceSet.from_urls(["https://docs.x.io/en/latest/"])
        # The lookup returns pages it recorded.
        page = PageContent(
            url="https://docs.x.io/en/latest/cli.html",
            title="t",
            text="",
            authority="official",
            published="2026-01-02",
        )
        gen.researcher.add_official_reference = MagicMock(return_value=[page])
        with patch("scholar_lens.adapters.web.guide_research.logger") as log:
            await gen._add_reference(ResearchCorpus(), ["--flag"])
        log.info.assert_any_call(
            "Added %s page [%s%s%s] %s",
            "reference-lookup",
            "official",
            "",
            " 2026-01-02",
            page.url,
        )


class TestClassify:
    async def test_classify_sets_a_checked_description_with_one_repair(
        self,
    ) -> None:
        from scholar_lens.domain.guide.models import TechGuide

        good = (
            "vLLM serves large language models with paged attention and continuous "
            "batching; this guide covers its engine, scheduler and deployment."
        )
        gen = _make_generator()
        gen.taxonomy_chain = MagicMock()
        gen.taxonomy_chain.ainvoke = AsyncMock(
            side_effect=[
                {"description": "`vLLM` is fast.", "controlled_tags": "B"},
                {"description": good, "controlled_tags": "B"},
            ]
        )
        guide = TechGuide(topic="vLLM", body="## Engine\n\nThe engine.")
        result = await gen.classify(
            guide, category_options=["A"], controlled_tags=["B"]
        )
        assert guide.description == result["description"] == good
        second = gen.taxonomy_chain.ainvoke.await_args_list[1].args[0]
        assert "description is plain text" in second["repair_note"]
        assert second["language"] == "English"
        assert guide.notes == []

    async def test_classify_accepts_no_controlled_tag(self) -> None:
        from scholar_lens.domain.guide.models import TechGuide

        good = (
            "Ray Serve deploys models as scalable online APIs; this guide covers "
            "deployments, composition, autoscaling and running it on Kubernetes."
        )
        gen = _make_generator()
        gen.taxonomy_chain = MagicMock()
        gen.taxonomy_chain.ainvoke = AsyncMock(
            return_value={"description": good, "controlled_tags": ""}
        )
        guide = TechGuide(topic="Ray Serve", body="## Serve\n\nServing.")
        result = await gen.classify(
            guide, category_options=["A"], controlled_tags=["B", "C"]
        )
        # None fits: no near-miss tag is forced, and the answer is not asked again.
        assert result["controlled_tags"] == ""
        assert gen.taxonomy_chain.ainvoke.await_count == 1

    async def test_a_description_still_marked_up_after_the_repair_is_dropped(
        self,
    ) -> None:
        from scholar_lens.domain.guide.models import TechGuide

        gen = _make_generator()
        gen.taxonomy_chain = MagicMock()
        gen.taxonomy_chain.ainvoke = AsyncMock(
            return_value={"description": "`vLLM` is fast."}
        )
        guide = TechGuide(topic="vLLM", body="## Engine\n\nThe engine.")
        await gen.classify(guide, category_options=["A"], controlled_tags=["B"])
        assert gen.taxonomy_chain.ainvoke.await_count == 2
        assert guide.description == ""
        assert any("Guide description" in n for n in guide.notes)

    async def test_classify_reads_the_finished_body(self) -> None:
        from scholar_lens.domain.guide.models import TechGuide

        gen = _make_generator()
        gen.taxonomy_chain = MagicMock()
        gen.taxonomy_chain.ainvoke = AsyncMock(return_value={})
        guide = TechGuide(
            topic="X",
            body="## Install\n\nInstall X with pip. More text.\n\n```bash\npip install x\n```",
        )
        await gen.classify(guide, category_options=["A"], controlled_tags=["B"])
        outline = gen.taxonomy_chain.ainvoke.await_args.args[0]["outline"]
        assert "## Install" in outline and "Install X with pip." in outline

    async def test_relevance_gate_keeps_the_commands_the_docs_show(self) -> None:
        gen = _make_generator()
        gen.relevance_chain.ainvoke = AsyncMock(
            return_value={
                "is_relevant": "yes",
                "topic": "uv",
                "commands": "uv, uvx, astral-cli",
                "reason": "docs",
            }
        )
        corpus = ResearchCorpus(
            pages=[
                PageContent(
                    "https://docs.astral.sh/uv/", "uv", "```\nuv sync\nuvx ruff\n```"
                )
            ]
        )
        await gen._assert_relevant(corpus)
        assert gen._products == ("uv", "uvx")


class TestReferenceMaterial:
    """Reference URLs feed the guide's corpus but never become the official
    authority; without docs no page is official."""

    DOCS = "https://docs.x.com"
    BLOG = "https://blog.x.io/deep-dive"

    def _generator(self, pages: dict[str, PageContent]) -> TechGuideGenerator:
        return _reference_generator(pages)

    def _pages(self) -> dict[str, PageContent]:
        return {
            self.DOCS: PageContent(self.DOCS, "X Docs", "How to use X."),
            self.BLOG: PageContent(self.BLOG, "Deep dive", "How the scheduler works."),
        }

    async def test_references_feed_the_corpus_without_becoming_official(
        self,
    ) -> None:
        gen = self._generator(self._pages())
        material = await gen.gather(
            [self.DOCS], reference_urls=[self.BLOG], discover_subpages=False
        )
        second = gen.researcher.research.call_args_list[1]
        assert second.args[0] == [self.BLOG]
        assert second.kwargs["discover_subpages"] is False
        assert gen.official.hosts == frozenset({"docs.x.com"})
        # The gate reads the docs alone.
        gate_sources = gen.relevance_chain.ainvoke.await_args.args[0]["sources"]
        assert self.BLOG not in gate_sources
        labels = _labels(gen, material)
        assert labels == {self.DOCS: OFFICIAL_LABEL, self.BLOG: REFERENCE_LABEL}
        assert material.references == [self.BLOG]
        assert not gen.notes

    async def test_an_unreadable_reference_is_noted(self) -> None:
        gen = self._generator(self._pages())
        missing = "https://gone.example/post"
        material = await gen.gather(
            [self.DOCS], reference_urls=[missing], discover_subpages=False
        )
        assert [p.url for p in material.corpus.pages] == [self.DOCS]
        assert any(missing in n for n in gen.notes)

    async def test_no_readable_reference_and_no_docs_is_rejected(self) -> None:
        gen = self._generator({})
        with pytest.raises(NotTechnicalContentError):
            await gen.gather([], reference_urls=["https://gone.example/post"])

    async def test_one_paper_given_several_ways_is_read_once(self) -> None:
        pages = {
            self.DOCS: PageContent(self.DOCS, "X Docs", "How to use X."),
            "https://arxiv.org/abs/2309.06180": PageContent(
                "https://arxiv.org/abs/2309.06180", "Paper", "Paged KV cache."
            ),
        }
        gen = self._generator(pages)
        material = await gen.gather(
            [self.DOCS],
            reference_urls=[
                "https://arxiv.org/abs/2309.06180",
                "https://arxiv.org/pdf/2309.06180v2.pdf",
                "https://arxiv.org/abs/2309.06180v1",
            ],
            discover_subpages=False,
        )
        second = gen.researcher.research.call_args_list[1]
        assert second.args[0] == ["https://arxiv.org/abs/2309.06180"]
        assert material.references == ["https://arxiv.org/abs/2309.06180"]
        assert [p.url for p in material.corpus.pages] == [
            self.DOCS,
            "https://arxiv.org/abs/2309.06180",
        ]

    async def test_a_reference_the_docs_crawl_landed_on_is_not_added_twice(
        self,
    ) -> None:
        landed = "https://docs.x.com/en/stable/"
        pages = {
            self.DOCS: PageContent(self.DOCS, "X Docs", "Docs.", final_url=landed),
            landed: PageContent(landed, "X Docs", "Docs."),
        }
        gen = self._generator(pages)
        material = await gen.gather(
            [self.DOCS], reference_urls=[landed], discover_subpages=False
        )
        assert [p.url for p in material.corpus.pages] == [self.DOCS]
        assert not gen.notes


def _reference_generator(pages: dict[str, PageContent]) -> TechGuideGenerator:
    """A generator whose researcher returns ``pages`` for the URLs it is given
    (no search backend)."""
    gen = _make_generator()
    gen.researcher.search_provider.supports_search = False

    def research(urls: list[str], *, discover_subpages: bool) -> ResearchCorpus:
        return ResearchCorpus(pages=[pages[u] for u in urls if u in pages])

    gen.researcher.research = MagicMock(side_effect=research)
    gen.relevance_chain.ainvoke = _relevant()
    return gen


def _labels(gen: TechGuideGenerator, material: GuideMaterial) -> dict[str, str]:
    """Each corpus page's URL -> the label prompts give it."""
    return {p.url: source_label(p) for p in material.corpus.pages}


def test_planner_digest_gives_docs_and_references_fair_shares() -> None:
    docs = [PageContent(f"https://docs.x.com/{i}", "D", "d " * 400) for i in range(20)]
    blogs = [
        PageContent(f"https://blog.x.io/{i}", "Deep", "b " * 400, user_reference=True)
        for i in range(20)
    ]
    digest = TechGuideGenerator._sources_digest(
        ResearchCorpus(pages=[*docs, *blogs]), max_chars=6_000
    )
    assert digest.startswith(f"- https://blog.x.io/0 ({REFERENCE_LABEL})")
    assert "https://docs.x.com/0" in digest
    assert len(digest) <= 6_000
    # A short reference list leaves the rest to the docs.
    short = TechGuideGenerator._sources_digest(
        ResearchCorpus(pages=[*docs, blogs[0]]), max_chars=6_000
    )
    assert short.count("https://docs.x.com/") > digest.count("https://docs.x.com/")


DOCS = ["https://docs.x.com/install", "https://docs.x.com/serve"]
REFS = ["https://blog.x.io/scheduler", "https://arxiv.org/abs/2309.06180"]


@pytest.fixture
def several_inputs() -> TechGuideGenerator:
    """A generator over two docs pages and two references (a post and a
    paper)."""
    pages = {
        DOCS[0]: PageContent(DOCS[0], "Install", "pip install x"),
        DOCS[1]: PageContent(DOCS[1], "Serve", "x serve model"),
        REFS[0]: PageContent(REFS[0], "Sched", "The scheduler batches."),
        REFS[1]: PageContent(REFS[1], "Paper", "Paged KV cache blocks."),
    }
    return _reference_generator(pages)


class TestSeveralDocsAndReferences:
    async def test_every_input_is_labelled(
        self, several_inputs: TechGuideGenerator
    ) -> None:
        gen = several_inputs
        material = await gen.gather(
            list(DOCS), reference_urls=list(REFS), discover_subpages=False
        )
        assert gen.researcher.research.call_args_list[0].args[0] == DOCS
        assert gen.researcher.research.call_args_list[1].args[0] == REFS
        labels = _labels(gen, material)
        assert labels == {
            **dict.fromkeys(DOCS, OFFICIAL_LABEL),
            **dict.fromkeys(REFS, REFERENCE_LABEL),
        }
        assert material.references == REFS

    async def test_a_url_given_as_both_is_docs_and_read_once(
        self, several_inputs: TechGuideGenerator
    ) -> None:
        gen = several_inputs
        material = await gen.gather(
            list(DOCS),
            reference_urls=[DOCS[1] + "/", *REFS],
            discover_subpages=False,
        )
        assert gen.researcher.research.call_args_list[1].args[0] == REFS
        assert _labels(gen, material)[DOCS[1]] == OFFICIAL_LABEL
        assert material.references == REFS

    async def test_references_without_docs(
        self, several_inputs: TechGuideGenerator
    ) -> None:
        gen = several_inputs
        material = await gen.gather([], reference_urls=list(REFS))
        assert gen.researcher.research.call_count == 1
        assert not gen.official.hosts
        assert OFFICIAL_LABEL not in _labels(gen, material).values()
        # No docs: nothing to look up.
        gen.researcher.add_official_reference.assert_not_called()
        gate_sources = gen.relevance_chain.ainvoke.await_args.args[0]["sources"]
        assert all(url in gate_sources for url in REFS)
        assert material.references == REFS
        assert _NO_DOCS_NOTE in gen.notes

    async def test_references_are_planned_from_and_searched_beyond(
        self, several_inputs: TechGuideGenerator
    ) -> None:
        gen = several_inputs
        gen.auto_research = True
        gen.researcher.search_provider.supports_search = True
        gen.researcher.run_searches = MagicMock()
        gen.research_plan_chain.ainvoke = AsyncMock(
            return_value={"topic": "X", "queries": "x scheduler design\nx metrics"}
        )
        await gen.gather(list(DOCS), reference_urls=list(REFS), discover_subpages=False)
        digest = gen.research_plan_chain.ainvoke.await_args.args[0]["sources"]
        for url in REFS:
            assert f"{url} ({REFERENCE_LABEL})" in digest
        searched_corpus, queries = gen.researcher.run_searches.call_args.args
        assert queries == ["x scheduler design", "x metrics"]
        assert gen.researcher.run_searches.call_args.kwargs["fetch_top"] == (
            gen.fetch_top_results
        )
        held = [p.url for p in searched_corpus.pages if p.user_reference]
        assert held == REFS


class TestGlossary:
    def test_a_term_is_introduced_only_by_its_first_use_form(self) -> None:
        from scholar_lens.domain.glossary import glossary_view

        glossary = {"continuous batching": "연속 배칭", "Prefill": "프리필"}
        # A bare rendering (or the rendering inside a longer word) is no
        # introduction; "rendering(English)" is, in any case, with or without
        # a space before the parenthesis.
        bare = glossary_view(glossary, ["연속 배칭과 프리필링을 씁니다."])
        assert "[introduced]" not in bare
        given = glossary_view(
            glossary, ["연속 배칭 (Continuous Batching)과 프리필(prefill)을 씁니다."]
        )
        assert given.count("[introduced]") == 2


_STRUCTURED = (
    "## S\n\n### Part\n\nThe server listens [@a; @b]. It answers requests in "
    "order and keeps a queue of them.\n\n![Queue](q.png)\n\n$$\nx = 1\n$$\n\n"
    "| a | b |\n| --- | --- |\n| 1 | 2 |\n"
)


@pytest.mark.parametrize(
    "edited,reason",
    [
        (_STRUCTURED.replace("### Part\n\n", ""), "loses heading '### Part'"),
        (_STRUCTURED.replace("![Queue](q.png)\n\n", ""), "loses image 'q.png'"),
        (_STRUCTURED.replace("; @b", ""), "loses citation [@b]"),
        (_STRUCTURED.replace("$$\nx = 1\n$$\n\n", ""), "loses a display equation"),
        (_STRUCTURED.split("| a |")[0], "loses a table"),
        (
            _STRUCTURED.replace(
                " It answers requests in order and keeps a queue of them.", ""
            ),
            "shrinks the prose",
        ),
    ],
)
def test_a_fix_that_loses_structure_is_named(edited: str, reason: str) -> None:
    assert lost_structure(_STRUCTURED, edited).startswith(reason)


def test_a_fix_may_correct_structure_in_place() -> None:
    # A wrong formula or cell fixed in place, one sentence reworded.
    edited = (
        _STRUCTURED.replace("x = 1", "x = 2")
        .replace("| 1 |", "| 3 |")
        .replace("keeps a queue", "keeps one queue")
    )
    assert lost_structure(_STRUCTURED, edited) == ""


def test_a_targeted_fix_may_not_drop_a_code_block() -> None:
    from scholar_lens.sections_engine.section_loop import lost_structure

    draft = "## S\n\nText.\n\n```bash\nvllm serve m --bad-flag\n```\n\nMore text."
    assert lost_structure(draft, "## S\n\nText.\n\nMore text.") == "loses a code block"
    edited = draft.replace(" --bad-flag", "")
    assert lost_structure(draft, edited) == ""


@pytest.mark.parametrize("cap", [0, 1, 2])
async def test_the_configured_third_party_cap_reaches_the_guides_searches(
    cap: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``tech_guide.max_third_party_pages`` goes from the config through
    bootstrap's toolkit into the generator's searches: with a cap of N,
    N of the third-party results are admitted."""
    from scholar_lens import bootstrap
    from scholar_lens.adapters.web.research import WebResearcher
    from scholar_lens.configs import Config
    from tests.integration.test_source_ranking import _result, _Search

    monkeypatch.setattr(
        "scholar_lens.adapters.web.guide_research.BedrockLanguageModelFactory",
        lambda **_: FakeLLMFactory(),
    )
    config = Config()
    config.tech_guide.max_third_party_pages = cap
    session = boto3.Session(region_name="us-east-1")
    context = bootstrap.run_context(
        config, default_boto_session=session, bedrock_boto_session=session
    )
    search = _Search(
        {"q": [_result(f"https://blog{i}.example.org/post") for i in range(5)]}
    )
    researcher = WebResearcher(search)
    researcher.research = MagicMock(return_value=_corpus())  # type: ignore[method-assign]
    gen = bootstrap.GuideToolkit(context).research(  # type: ignore[arg-type]
        researcher, MagicMock(), title_max_length=70
    )
    # No network: no release or tag lookups, the gate answers.
    gen.release_lookup = None
    gen.repo_tags = None
    gen.relevance_chain = MagicMock()
    gen.relevance_chain.ainvoke = _relevant()
    material = await gen.gather(
        ["https://docs.x.com"], discover_subpages=False, search_queries=["q"]
    )
    blogs = [p for p in material.corpus.pages if "blog" in p.url]
    assert len(blogs) == cap
    assert all(p.authority == "third_party" for p in blogs)
    assert gen.min_secondary_pages == config.tech_guide.min_secondary_pages


def _bootstrapped_guide(
    config: object,
    monkeypatch: pytest.MonkeyPatch,
    search: object,
    seed: ResearchCorpus,
) -> TechGuideGenerator:
    """A guide generator built by bootstrap from ``config``, searching with
    ``search`` and reading ``seed`` for the requested docs (no network)."""
    from scholar_lens import bootstrap
    from scholar_lens.adapters.web.research import WebResearcher

    monkeypatch.setattr(
        "scholar_lens.adapters.web.guide_research.BedrockLanguageModelFactory",
        lambda **_: FakeLLMFactory(),
    )
    session = boto3.Session(region_name="us-east-1")
    context = bootstrap.run_context(
        config,  # type: ignore[arg-type]
        default_boto_session=session,
        bedrock_boto_session=session,
    )
    researcher = WebResearcher(search)  # type: ignore[arg-type]
    researcher.research = MagicMock(return_value=seed)  # type: ignore[method-assign]
    gen = bootstrap.GuideToolkit(context).research(  # type: ignore[arg-type]
        researcher, MagicMock(), title_max_length=70
    )
    gen.release_lookup = None
    gen.repo_tags = None
    gen.relevance_chain = MagicMock()
    gen.relevance_chain.ainvoke = _relevant()
    return gen


@pytest.mark.parametrize("reserve, cap", [(3, 4), (3, 2), (1, 4)])
async def test_the_secondary_reserve_admits_non_official_pages_end_to_end(
    reserve: int, cap: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With more official results than ``fetch_top_results`` ranking first,
    at least min(min_secondary_pages, cap) non-official pages still get in."""
    from scholar_lens.configs import Config
    from tests.integration.test_source_ranking import _result, _Search

    config = Config()
    config.tech_guide.fetch_top_results = 4
    config.tech_guide.max_third_party_pages = cap
    config.tech_guide.min_secondary_pages = reserve
    official = [_result(f"https://docs.x.com/guide/p{i}") for i in range(8)]
    others = [_result(f"https://blog{i}.example.org/post") for i in range(4)]
    gen = _bootstrapped_guide(
        config, monkeypatch, _Search({"q": official + others}), _corpus()
    )
    material = await gen.gather(
        ["https://docs.x.com"], discover_subpages=False, search_queries=["q"]
    )
    searched = material.corpus.pages[1:]
    assert len(searched) == 4
    non_official = [p for p in searched if p.authority not in ("official",)]
    assert len(non_official) >= min(reserve, cap)


async def test_the_configured_link_share_scopes_the_requested_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tech_guide.official_scope_link_share`` reaches the run's source set:
    a requested product page's own links keep another product on the host
    out of the official docs."""
    from scholar_lens.configs import Config
    from tests.integration.test_source_ranking import _Search

    config = Config()
    host = "https://cloud.x.com"
    links = [f"{host}/prod-a/docs/{p}" for p in ("a", "b", "c", "d")]
    links += [f"{host}/prod-b/docs", f"{host}/blog"]
    seed = ResearchCorpus(
        pages=[
            PageContent(
                f"{host}/prod-a/docs/start",
                "A",
                "How to use A.",
                links=links,
                content_links=links,
            )
        ]
    )
    gen = _bootstrapped_guide(config, monkeypatch, _Search({}), seed)
    assert gen.official_scope_link_share == config.tech_guide.official_scope_link_share
    await gen.gather([seed.pages[0].url], discover_subpages=False, search_queries=[])
    official = gen._sources.official
    assert official.on_host(f"{host}/prod-a/docs/e")
    assert not official.on_host(f"{host}/prod-b/docs/start")
    assert not official.on_host(f"{host}/prod-a/pricing")
