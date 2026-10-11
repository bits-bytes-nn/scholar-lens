"""Tests for the technical-guide generator (LLM chains stubbed)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from scholar_lens.src.constants import LanguageModelId
from scholar_lens.src.prompts import (
    TechGuideEvaluationPrompt,
    TechGuideResearchPlanPrompt,
    TechGuideSectionPrompt,
    TechGuideSynopsisPrompt,
)
from scholar_lens.src.tech_guide import (
    NotTechnicalContentError,
    PlannedSection,
    TechGuideGenerator,
)
from scholar_lens.src.web_research import PageContent, ResearchCorpus


class TestPromptsBuild:
    """Every declared input variable must appear in its template (get_prompt
    raises otherwise), so this catches a renamed/missing placeholder."""

    PROMPTS = [
        TechGuideResearchPlanPrompt,
        TechGuideSynopsisPrompt,
        TechGuideSectionPrompt,
        TechGuideEvaluationPrompt,
    ]

    def test_get_prompt_does_not_raise(self) -> None:
        for prompt in self.PROMPTS:
            assert prompt.get_prompt() is not None, prompt.__name__

    def test_section_prompt_carries_depth_directive(self) -> None:
        assert "depth_directive" in TechGuideSectionPrompt.input_variables
        assert "{depth_directive}" in TechGuideSectionPrompt.human_prompt_template


def _make_generator(
    *,
    verify_grounding: bool = False,
    auto_research: bool = False,
    min_quality_score: int = 75,
    max_revision_attempts: int = 0,
) -> TechGuideGenerator:
    """Construct a generator without touching Bedrock or the network.

    Defaults keep the optional stages OFF (no auto-research, no revision loop)
    so a test exercises only what it opts into; tests that want a stage set its
    flag and stub the corresponding chain.
    """
    gen = TechGuideGenerator.__new__(TechGuideGenerator)
    gen.language = "English"
    gen.max_sections = 12
    gen.verify_grounding = verify_grounding
    gen.auto_research = auto_research
    gen.max_research_queries = 6
    gen.fetch_top_results = 4
    gen.min_quality_score = min_quality_score
    gen.max_revision_attempts = max_revision_attempts
    gen.max_total_tokens = None
    gen._token_tracker = None
    gen.notes = []
    gen.researcher = MagicMock()
    gen.relevance_chain = MagicMock()
    gen.research_plan_chain = MagicMock()
    gen.synopsis_chain = MagicMock()
    gen.section_chain = MagicMock()
    gen.evaluation_chain = MagicMock()
    gen.grounding_chain = MagicMock()
    gen.synopsis_model_id = LanguageModelId.CLAUDE_V4_5_HAIKU
    gen.writing_model_id = LanguageModelId.CLAUDE_V4_5_HAIKU
    # fit_text would call Bedrock CountTokens; stub it to a pass-through.
    factory = MagicMock()
    factory.fit_text = MagicMock(side_effect=lambda model_id, text, **kw: text)
    # _write_sections composes the source + previous-sections reserves against the
    # writer's window; return a realistic 1M so the arithmetic uses a real int.
    factory.effective_context_window = MagicMock(return_value=1_000_000)
    gen.llm_factory = factory
    return gen


def _corpus() -> ResearchCorpus:
    return ResearchCorpus(
        pages=[PageContent("https://docs.x.com", "X Docs", "How to use X.")]
    )


class TestParseSynopsisSections:
    def test_numbered_list_titles(self) -> None:
        gen = _make_generator()
        synopsis = "1. Intro — overview\n2. Setup — install\n3. Usage — examples"
        titles = [s.title for s in gen._parse_synopsis_sections(synopsis)]
        assert titles == ["Intro", "Setup", "Usage"]

    def test_bulleted_list(self) -> None:
        gen = _make_generator()
        titles = [s.title for s in gen._parse_synopsis_sections("- A\n* B\n• C")]
        assert titles == ["A", "B", "C"]

    def test_caps_at_max_sections(self) -> None:
        gen = _make_generator()
        gen.max_sections = 2
        out = gen._parse_synopsis_sections("1. a\n2. b\n3. c")
        assert [s.title for s in out] == ["a", "b"]

    def test_unnumbered_synopsis_falls_back_to_lines(self) -> None:
        gen = _make_generator()
        out = gen._parse_synopsis_sections("just one line")
        assert [s.title for s in out] == ["just one line"]

    def test_parses_area_and_depth_tags(self) -> None:
        gen = _make_generator()
        synopsis = (
            "1. [CONCEPT] [deep] How GitOps works — the mental model (visuals: none)\n"
            "2. [USAGE] [brief] Quick install — one-liner (visuals: code)"
        )
        out = gen._parse_synopsis_sections(synopsis)
        assert out[0].area == "CONCEPT"
        assert out[0].depth == "deep"
        assert out[0].title == "How GitOps works"
        assert out[0].visuals == "none"
        assert out[1].area == "USAGE"
        assert out[1].depth == "brief"
        assert out[1].title == "Quick install"
        assert out[1].visuals == "code"

    def test_visual_hint_reaches_depth_directive(self) -> None:
        # P2: the planned visual is carried into the writer's depth directive
        # (it used to be parsed then discarded).
        gen = _make_generator()
        out = gen._parse_synopsis_sections("1. [DETAIL] [deep] X — y (visuals: table)")
        assert out[0].visuals == "table"
        assert "table" in out[0].depth_directive("Korean")

    def test_unknown_tags_default_gracefully(self) -> None:
        # A loosely-formatted line without recognised tags still yields a title
        # with the default depth, never crashing the parse.
        gen = _make_generator()
        out = gen._parse_synopsis_sections("1. Plain Title — desc")
        assert out[0].title == "Plain Title"
        assert out[0].depth == "standard"
        assert out[0].area == ""

    def test_depth_directive_maps_to_text(self) -> None:
        gen = _make_generator()
        out = gen._parse_synopsis_sections("1. [CONCEPT] [deep] Title")
        assert "DEEP section" in out[0].depth_directive("Korean")
        assert "about 12,500 characters" in out[0].depth_directive("English")

    def test_spaced_hyphen_title_not_truncated(self) -> None:
        # " - " inside a title must survive (only em/en dash separates desc).
        gen = _make_generator()
        out = gen._parse_synopsis_sections("1. [USAGE] [brief] Using gRPC - the basics")
        assert out[0].title == "Using gRPC - the basics"

    def test_unknown_leading_tag_stripped_from_title(self) -> None:
        gen = _make_generator()
        out = gen._parse_synopsis_sections("1. [Foo] Real Title — desc")
        assert out[0].title == "Real Title"


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
        from scholar_lens.src import guide_pipeline as tg

        for var in ("TAVILY_API_KEY", "BRAVE_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        if tavily:
            monkeypatch.setenv("TAVILY_API_KEY", tavily)
        if brave:
            monkeypatch.setenv("BRAVE_API_KEY", brave)
        return tg.build_search_provider()

    def test_prefers_tavily(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scholar_lens.src.web_research import TavilySearchProvider

        p = self._provider(monkeypatch, tavily="tvly-k", brave="brave-k")
        assert isinstance(p, TavilySearchProvider)

    def test_falls_back_to_brave(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scholar_lens.src.web_research import BraveSearchProvider

        p = self._provider(monkeypatch, brave="brave-k")
        assert isinstance(p, BraveSearchProvider)

    def test_null_when_no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scholar_lens.src.web_research import NullSearchProvider

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


class TestGenerateFlow:
    async def test_empty_urls_rejected(self) -> None:
        gen = _make_generator()
        with pytest.raises(ValueError):
            await gen.generate([])

    async def test_no_readable_pages_rejected(self) -> None:
        gen = _make_generator()
        gen.researcher.research = MagicMock(return_value=ResearchCorpus())
        with pytest.raises(NotTechnicalContentError):
            await gen.generate(["https://x.com"])

    async def test_full_flow_assembles_guide(self) -> None:
        gen = _make_generator()
        gen.researcher.research = MagicMock(return_value=_corpus())
        gen.relevance_chain.ainvoke = AsyncMock(
            return_value={"is_relevant": "yes", "topic": "Using X", "reason": "docs"}
        )
        gen.synopsis_chain.ainvoke = AsyncMock(
            return_value={"synopsis": "1. Intro - overview\n2. Usage - examples"}
        )
        gen.section_chain.ainvoke = AsyncMock(
            return_value={"section_markdown": "## Section\nbody"}
        )
        guide = await gen.generate(["https://docs.x.com"], discover_subpages=False)
        assert guide.topic == "Using X"
        assert guide.body.count("## Section") == 2  # two synopsis sections
        assert guide.source_urls == ["https://docs.x.com"]

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("<section_markdown>## A\nbody</section_markdown>", "## A\nbody"),
            ("## A\nbody without wrapper", "## A\nbody without wrapper"),
            ("<section_markdown>## A\ntruncated", "## A\ntruncated"),
        ],
    )
    def test_section_output_parsed_leniently(self, raw: str, expected: str) -> None:
        from scholar_lens.src.tech_guide import parse_section_output

        assert parse_section_output(raw) == {"section_markdown": expected}


class TestGrounding:
    async def test_grounding_pass_runs_when_enabled(self) -> None:
        gen = _make_generator(verify_grounding=True)
        gen.researcher.research = MagicMock(return_value=_corpus())
        gen.relevance_chain.ainvoke = AsyncMock(
            return_value={"is_relevant": "yes", "topic": "Using X", "reason": "docs"}
        )
        gen.synopsis_chain.ainvoke = AsyncMock(
            return_value={"synopsis": "1. Intro - overview"}
        )
        gen.section_chain.ainvoke = AsyncMock(
            return_value={"section_markdown": "## Section\nungrounded claim"}
        )
        gen.grounding_chain.ainvoke = AsyncMock(
            return_value={"grounded_markdown": "## Section\ncleaned"}
        )
        guide = await gen.generate(["https://docs.x.com"], discover_subpages=False)
        gen.grounding_chain.ainvoke.assert_awaited()
        assert "cleaned" in guide.body
        assert "ungrounded" not in guide.body

    async def test_grounding_falls_back_to_draft_when_empty(self) -> None:
        gen = _make_generator(verify_grounding=True)
        gen.grounding_chain.ainvoke = AsyncMock(
            return_value={"grounded_markdown": "   "}
        )
        out = await gen._ground_section("## Original\nbody", "src")
        assert out == "## Original\nbody"

    async def test_grounding_skipped_when_disabled(self) -> None:
        gen = _make_generator(verify_grounding=False)
        gen.researcher.research = MagicMock(return_value=_corpus())
        gen.relevance_chain.ainvoke = AsyncMock(
            return_value={"is_relevant": "yes", "topic": "Using X", "reason": "docs"}
        )
        gen.synopsis_chain.ainvoke = AsyncMock(
            return_value={"synopsis": "1. Intro - overview"}
        )
        gen.section_chain.ainvoke = AsyncMock(
            return_value={"section_markdown": "## Section\nbody"}
        )
        gen.grounding_chain.ainvoke = AsyncMock()
        await gen.generate(["https://docs.x.com"], discover_subpages=False)
        gen.grounding_chain.ainvoke.assert_not_awaited()


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
        gen.relevance_chain.ainvoke = AsyncMock(
            return_value={"is_relevant": "yes", "topic": "Using X", "reason": "ok"}
        )
        gen.synopsis_chain.ainvoke = AsyncMock(
            return_value={"synopsis": "1. [CONCEPT] [deep] Intro"}
        )
        gen.section_chain.ainvoke = AsyncMock(
            return_value={"section_markdown": "## Section\nbody"}
        )
        await gen.generate(["https://docs.x.com"], discover_subpages=False)
        gen.researcher.run_searches.assert_called_once()
        assert gen.researcher.run_searches.call_args.args[1] == ["q1", "q2"]
        # Top results are fetched into the corpus (reaches the writer, not just
        # the outline) — the C1 fix.
        assert gen.researcher.run_searches.call_args.kwargs["fetch_top"] == 4

    async def test_planning_skipped_when_backend_cannot_search(self) -> None:
        # H4: with a no-op search backend, the planner LLM call is not made.
        gen = _make_generator(auto_research=True)
        gen.researcher.search_provider.supports_search = False
        gen.researcher.research = MagicMock(return_value=_corpus())
        gen.researcher.run_searches = MagicMock()
        gen.research_plan_chain.ainvoke = AsyncMock()
        gen.relevance_chain.ainvoke = AsyncMock(
            return_value={"is_relevant": "yes", "topic": "Using X", "reason": "ok"}
        )
        gen.synopsis_chain.ainvoke = AsyncMock(return_value={"synopsis": "1. Intro"})
        gen.section_chain.ainvoke = AsyncMock(
            return_value={"section_markdown": "## S\nbody"}
        )
        await gen.generate(["https://docs.x.com"], discover_subpages=False)
        gen.research_plan_chain.ainvoke.assert_not_awaited()
        gen.researcher.run_searches.assert_not_called()

    async def test_explicit_queries_skip_planning(self) -> None:
        gen = _make_generator(auto_research=True)
        gen.researcher.research = MagicMock(return_value=_corpus())
        gen.researcher.run_searches = MagicMock()
        gen.research_plan_chain.ainvoke = AsyncMock()
        gen.relevance_chain.ainvoke = AsyncMock(
            return_value={"is_relevant": "yes", "topic": "Using X", "reason": "ok"}
        )
        gen.synopsis_chain.ainvoke = AsyncMock(return_value={"synopsis": "1. Intro"})
        gen.section_chain.ainvoke = AsyncMock(
            return_value={"section_markdown": "## S\nbody"}
        )
        await gen.generate(
            ["https://docs.x.com"],
            discover_subpages=False,
            search_queries=["explicit query"],
        )
        # Explicit queries bypass the planner but still drive the searches.
        gen.research_plan_chain.ainvoke.assert_not_awaited()
        gen.researcher.run_searches.assert_called_once()
        assert gen.researcher.run_searches.call_args.args[1] == ["explicit query"]

    async def test_planner_topic_used_when_relevance_gate_has_none(self) -> None:
        # Relevance returns no usable topic, but the planner guessed one earlier.
        gen = _make_generator(auto_research=False)
        gen.relevance_chain.ainvoke = AsyncMock(
            return_value={"is_relevant": "yes", "topic": "N/A", "reason": "ok"}
        )
        topic = await gen._assert_relevant(_corpus(), fallback_topic="Planned Topic")
        assert topic == "Planned Topic"


class TestEvaluateAndRevise:
    """The evaluate-and-revise loop (ported from the review pipeline)."""

    async def test_better_revision_is_kept(self) -> None:
        # Draft scores 50 (< 80); the revision scores higher (70), so it's kept.
        gen = _make_generator(min_quality_score=80, max_revision_attempts=1)
        gen.evaluation_chain.ainvoke = AsyncMock(
            side_effect=[
                {"quality_score": "50", "improvement_feedback": "go deeper"},
                {"quality_score": "70", "improvement_feedback": "better"},
            ]
        )
        gen.section_chain.ainvoke = AsyncMock(
            return_value={"section_markdown": "## Revised\ndeeper body"}
        )
        out = await gen._evaluate_and_revise(
            topic="X",
            planned=PlannedSection(title="Intro", depth="deep"),
            section_label="1. Intro",
            markdown="## Draft\nshallow",
            section_number=1,
            total_sections=1,
            previous_sections="",
            already_in_guide="- nothing yet",
            sources="src",
            available_images="(none)",
            final_outline="1. Intro",
        )
        gen.section_chain.ainvoke.assert_awaited()  # a revision happened
        assert out == "## Revised\ndeeper body"

    async def test_worse_revision_is_discarded(self) -> None:
        # Regression for the best-of guard: a revision that does NOT beat the
        # original (same/lower score) must be discarded, never returned.
        gen = _make_generator(min_quality_score=80, max_revision_attempts=1)
        gen.evaluation_chain.ainvoke = AsyncMock(
            side_effect=[
                {"quality_score": "50", "improvement_feedback": "go deeper"},
                {"quality_score": "30", "improvement_feedback": "now worse"},
            ]
        )
        gen.section_chain.ainvoke = AsyncMock(
            return_value={"section_markdown": "## Regressed\nworse body"}
        )
        out = await gen._evaluate_and_revise(
            topic="X",
            planned=PlannedSection(title="Intro", depth="deep"),
            section_label="1. Intro",
            markdown="## Draft\noriginal",
            section_number=1,
            total_sections=1,
            previous_sections="",
            already_in_guide="- nothing yet",
            sources="src",
            available_images="(none)",
            final_outline="1. Intro",
        )
        assert out == "## Draft\noriginal"  # original kept, not the worse rewrite

    async def test_unparsable_score_never_beats_a_scored_draft(self) -> None:
        # A one-slip 60 is kept over a revision whose score failed to parse (0).
        gen = _make_generator(min_quality_score=80, max_revision_attempts=1)
        gen.language = "Korean"
        gen.evaluation_chain.ainvoke = AsyncMock(
            side_effect=[
                {"quality_score": "60", "improvement_feedback": "깊이를 더합니다."},
                {"quality_score": "garbage"},
            ]
        )
        gen.section_chain.ainvoke = AsyncMock(
            return_value={"section_markdown": "## 설치\n\n설치합니다."}
        )
        out = await gen._evaluate_and_revise(
            topic="X",
            planned=PlannedSection(title="설치", depth="deep"),
            section_label="1. 설치",
            markdown="## 설치\n\n이 절에서 설치합니다.",
            section_number=1,
            total_sections=1,
            previous_sections="",
            already_in_guide="- nothing yet",
            sources="src",
            available_images="(none)",
            final_outline="1. 설치",
        )
        assert out == "## 설치\n\n이 절에서 설치합니다."

    async def test_high_score_accepts_without_revision(self) -> None:
        gen = _make_generator(min_quality_score=70, max_revision_attempts=2)
        gen.evaluation_chain.ainvoke = AsyncMock(
            return_value={"quality_score": "90", "improvement_feedback": "minor nit"}
        )
        gen.section_chain.ainvoke = AsyncMock()
        out = await gen._evaluate_and_revise(
            topic="X",
            planned=PlannedSection(title="Intro"),
            section_label="1. Intro",
            markdown="## Good\nbody",
            section_number=1,
            total_sections=1,
            previous_sections="",
            already_in_guide="- nothing yet",
            sources="src",
            available_images="(none)",
            final_outline="1. Intro",
        )
        gen.section_chain.ainvoke.assert_not_awaited()  # no revision needed
        assert out == "## Good\nbody"

    async def test_no_attempts_skips_evaluation_entirely(self) -> None:
        # max_revision_attempts=0 means the loop never runs — no eval call.
        gen = _make_generator(max_revision_attempts=0)
        gen.evaluation_chain.ainvoke = AsyncMock()
        out = await gen._evaluate_and_revise(
            topic="X",
            planned=PlannedSection(title="Intro"),
            section_label="1. Intro",
            markdown="## Draft\nbody",
            section_number=1,
            total_sections=1,
            previous_sections="",
            already_in_guide="- nothing yet",
            sources="src",
            available_images="(none)",
            final_outline="1. Intro",
        )
        gen.evaluation_chain.ainvoke.assert_not_awaited()
        assert out == "## Draft\nbody"

    async def test_low_score_with_empty_feedback_still_revises(self) -> None:
        # A below-threshold section with EMPTY feedback must NOT be accepted just
        # because feedback is blank (empty feedback only means "excellent" when the
        # score is also high). It should revise with a generic directive.
        gen = _make_generator(min_quality_score=80, max_revision_attempts=1)
        gen.evaluation_chain.ainvoke = AsyncMock(
            return_value={"quality_score": "40", "improvement_feedback": ""}
        )
        gen.section_chain.ainvoke = AsyncMock(
            return_value={"section_markdown": "## Revised\nmore depth"}
        )
        await gen._evaluate_and_revise(
            topic="X",
            planned=PlannedSection(title="Intro"),
            section_label="1. Intro",
            markdown="## Thin\nbody",
            section_number=1,
            total_sections=1,
            previous_sections="",
            already_in_guide="- nothing yet",
            sources="src",
            available_images="(none)",
            final_outline="1. Intro",
        )
        gen.section_chain.ainvoke.assert_awaited()  # revision happened

    async def test_rejected_revision_keeps_the_generic_directive(self) -> None:
        # Low score with empty feedback, then a worse revision: the next prompt
        # still carries the generic directive instead of an empty list.
        gen = _make_generator(min_quality_score=80, max_revision_attempts=2)
        gen.evaluation_chain.ainvoke = AsyncMock(
            side_effect=[
                {"quality_score": "50", "improvement_feedback": ""},
                {"quality_score": "30", "improvement_feedback": ""},
            ]
            + [{"quality_score": "90", "improvement_feedback": ""}] * 5
        )
        gen.section_chain.ainvoke = AsyncMock(
            return_value={"section_markdown": "## Revised\nmore depth"}
        )
        await gen._evaluate_and_revise(
            topic="X",
            planned=PlannedSection(title="Intro"),
            section_label="1. Intro",
            markdown="## Thin\nbody",
            section_number=1,
            total_sections=1,
            previous_sections="",
            already_in_guide="- nothing yet",
            sources="src",
            available_images="(none)",
            final_outline="1. Intro",
        )
        second = gen.section_chain.ainvoke.await_args_list[1].args[0]
        feedback = second["depth_directive"]
        assert "below the quality bar" in feedback
        assert feedback.count("below the quality bar") == 1
        assert "rejected (score 30" in feedback

    async def test_rejected_revision_lists_its_rule_slips(self) -> None:
        # Same feedback for the kept and the rejected draft: the note must not
        # end on a bare colon, and the rejected draft's slips are listed.
        gen = _make_generator(min_quality_score=80, max_revision_attempts=2)
        gen.evaluation_chain.ainvoke = AsyncMock(
            side_effect=[
                {"quality_score": "60", "improvement_feedback": "더 깊게"},
                {"quality_score": "40", "improvement_feedback": "더 깊게"},
            ]
            + [{"quality_score": "90", "improvement_feedback": ""}] * 5
        )
        gen.section_chain.ainvoke = AsyncMock(
            side_effect=[{"section_markdown": "## Revised\n이 절에서 설명합니다."}]
            + [{"section_markdown": "## Fine\n본문입니다."}] * 5
        )
        await gen._evaluate_and_revise(
            topic="X",
            planned=PlannedSection(title="Intro"),
            section_label="1. Intro",
            markdown="## Thin\nbody",
            section_number=1,
            total_sections=1,
            previous_sections="",
            already_in_guide="- nothing yet",
            sources="src",
            available_images="(none)",
            final_outline="1. Intro",
        )
        second = gen.section_chain.ainvoke.await_args_list[1].args[0]
        feedback = second["depth_directive"]
        assert not feedback.rstrip().endswith(":")
        assert "self-reference" in feedback

    def test_parse_quality_score_robust(self) -> None:
        # The score parser is now the shared helper used by both the review and
        # tech-guide evaluate loops.
        from scholar_lens.src.utils import parse_quality_score

        assert parse_quality_score("85") == 85
        assert parse_quality_score("85/100") == 85
        assert parse_quality_score("N/A") == 0
        assert parse_quality_score(None) == 0


class TestTokenBudget:
    def test_no_budget_never_exhausted(self) -> None:
        gen = _make_generator()
        gen.max_total_tokens = None
        gen._token_tracker = MagicMock(total_tokens=10_000_000)
        assert gen._budget_exhausted() is False

    def test_budget_exhausted_when_over(self) -> None:
        gen = _make_generator()
        gen.max_total_tokens = 1000
        gen._token_tracker = MagicMock(total_tokens=2000)
        assert gen._budget_exhausted() is True

    def test_budget_not_exhausted_when_under(self) -> None:
        gen = _make_generator()
        gen.max_total_tokens = 5000
        gen._token_tracker = MagicMock(total_tokens=2000)
        assert gen._budget_exhausted() is False

    def test_hard_budget_aborts_past_multiple(self) -> None:
        # Soft tier trips at 1x (skip refinement); the hard tier RAISES past
        # _HARD_BUDGET_MULTIPLE x so a runaway can't keep writing sections.
        from scholar_lens.src.metrics import TokenBudgetExceeded
        from scholar_lens.src.tech_guide import _HARD_BUDGET_MULTIPLE

        gen = _make_generator()
        gen.max_total_tokens = 1000
        tracker = MagicMock()
        # Real check_budget semantics: raise only when over the passed ceiling.
        tracker.total_tokens = int(1000 * _HARD_BUDGET_MULTIPLE) + 1

        def _check(ceiling: int | None) -> None:
            if ceiling and tracker.total_tokens > ceiling:
                raise TokenBudgetExceeded("over")

        tracker.check_budget.side_effect = _check
        gen._token_tracker = tracker
        with pytest.raises(TokenBudgetExceeded):
            gen._enforce_hard_token_budget()

    def test_hard_budget_ok_below_multiple(self) -> None:
        from scholar_lens.src.metrics import TokenBudgetExceeded

        gen = _make_generator()
        gen.max_total_tokens = 1000
        tracker = MagicMock()
        tracker.total_tokens = 1200  # over 1x (soft) but under 1.5x (hard)

        def _check(ceiling: int | None) -> None:
            if ceiling and tracker.total_tokens > ceiling:
                raise TokenBudgetExceeded("over")

        tracker.check_budget.side_effect = _check
        gen._token_tracker = tracker
        gen._enforce_hard_token_budget()  # must NOT raise

    async def test_over_budget_skips_eval_and_grounding(self) -> None:
        # When the budget is spent, sections are still WRITTEN but the expensive
        # evaluate/revise + grounding passes are skipped.
        gen = _make_generator(
            verify_grounding=True, min_quality_score=80, max_revision_attempts=2
        )
        gen.max_total_tokens = 1000
        gen._token_tracker = MagicMock(total_tokens=5000)  # already over
        gen.section_chain.ainvoke = AsyncMock(
            return_value={"section_markdown": "## S\nbody"}
        )
        gen.evaluation_chain.ainvoke = AsyncMock()
        gen.grounding_chain.ainvoke = AsyncMock()
        body = await gen._write_sections("X", "1. [CONCEPT] [deep] Intro", _corpus())
        assert "## S" in body
        gen.evaluation_chain.ainvoke.assert_not_awaited()
        gen.grounding_chain.ainvoke.assert_not_awaited()

    async def test_hard_abort_stops_writing_remaining_sections(self) -> None:
        # Past the 1.5x hard ceiling, _write_sections must STOP issuing writer
        # calls (not merely soft-degrade): the 2nd section's writer is never
        # awaited once the 1st pushes spend past the hard cap. This is the exact
        # guarantee the two-tier budget exists to provide.
        from scholar_lens.src.metrics import TokenBudgetExceeded

        gen = _make_generator(min_quality_score=80, max_revision_attempts=0)
        gen.max_total_tokens = 1000  # hard ceiling = 1500

        tracker = MagicMock()
        tracker.total_tokens = 0

        # The hard-budget check calls tracker.check_budget(1500); make it raise
        # only once spend has crossed the ceiling. Simulate the 1st section's
        # write pushing spend over by bumping total_tokens on each writer call.
        def check_budget(ceiling: int) -> None:
            if tracker.total_tokens > ceiling:
                raise TokenBudgetExceeded("over hard cap")

        tracker.check_budget = MagicMock(side_effect=check_budget)
        gen._token_tracker = tracker

        async def write(*args, **kwargs):
            tracker.total_tokens += 2000  # one section blows past 1.5x
            return {"section_markdown": "## S\nbody"}

        gen.section_chain.ainvoke = AsyncMock(side_effect=write)

        with pytest.raises(TokenBudgetExceeded):
            await gen._write_sections(
                "X", "1. [CONCEPT] [deep] A\n2. [CONCEPT] [deep] B", _corpus()
            )
        # Exactly one writer call: the loop aborted before section 2.
        assert gen.section_chain.ainvoke.await_count == 1


class TestPlannedScopeAndOverview:
    def test_description_parsed_and_shown_in_label(self) -> None:
        gen = _make_generator()
        [section] = gen._parse_synopsis_sections(
            "1. [CONCEPT] [deep] Reconciliation — how the controller diffs state "
            "(visuals: table)"
        )
        assert section.title == "Reconciliation"
        assert section.description == "how the controller diffs state"
        assert section.label(1) == (
            "1. Reconciliation — how the controller diffs state"
        )

    def test_label_without_description(self) -> None:
        gen = _make_generator()
        [section] = gen._parse_synopsis_sections("1. [USAGE] Install")
        assert section.label(2) == "2. Install"

    def test_overview_lists_planned_sections(self) -> None:
        gen = _make_generator()
        overview = gen._overview("1. A — x\n2. B — y\n3. C — z")
        assert overview.startswith("## In This Guide\n\n- A\n- B\n- C")

    def test_overview_skipped_for_short_guides(self) -> None:
        assert _make_generator()._overview("1. A\n2. B") == ""


class TestGuideLengthAndRules:
    def test_length_report_measures_prose_against_the_depth_target(self) -> None:
        from scholar_lens.src.tech_guide import length_report

        md = "## 제목\n\n" + "가" * 1200 + "\n\n```python\n" + "x" * 5000 + "\n```"
        report = length_report(md, "brief", "Korean")
        assert "1,206 characters" in report and "at most 1,000" in report
        assert "about 12,500" in length_report(md, "deep", "English")

    async def test_rule_violation_forces_a_revision_despite_a_high_score(self) -> None:
        gen = _make_generator(min_quality_score=75, max_revision_attempts=2)
        gen.language = "Korean"
        gen.evaluation_chain.ainvoke = AsyncMock(
            return_value={"quality_score": "90", "improvement_feedback": ""}
        )
        gen.section_chain.ainvoke = AsyncMock(
            return_value={"section_markdown": "## 설치\n\n도구를 설치합니다."}
        )
        out = await gen._evaluate_and_revise(
            topic="X",
            planned=PlannedSection(title="설치"),
            section_label="1. 설치",
            markdown="## 설치\n\n이 절에서는 도구를 install합니다.",
            section_number=1,
            total_sections=1,
            previous_sections="",
            already_in_guide="- nothing yet",
            sources="src",
            available_images="(none)",
            final_outline="1. 설치",
        )
        assert out == "## 설치\n\n도구를 설치합니다."
        directive = gen.section_chain.ainvoke.await_args.args[0]["depth_directive"]
        assert "self-reference" in directive and "install합니다" in directive
        assert gen.notes == []

    async def test_section_left_below_the_bar_is_noted(self) -> None:
        gen = _make_generator(min_quality_score=80, max_revision_attempts=1)
        gen.evaluation_chain.ainvoke = AsyncMock(
            return_value={"quality_score": "50", "improvement_feedback": "deeper"}
        )
        gen.section_chain.ainvoke = AsyncMock(return_value={"section_markdown": ""})
        await gen._evaluate_and_revise(
            topic="X",
            planned=PlannedSection(title="Intro"),
            section_label="1. Intro",
            markdown="## Intro\nbody",
            section_number=1,
            total_sections=1,
            previous_sections="",
            already_in_guide="- nothing yet",
            sources="src",
            available_images="(none)",
            final_outline="1. Intro",
        )
        assert any("'Intro' kept below the bar" in n for n in gen.notes)
