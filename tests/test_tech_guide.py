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
    gen.min_quality_score = min_quality_score
    gen.max_revision_attempts = max_revision_attempts
    gen.researcher = MagicMock()
    gen.relevance_chain = MagicMock()
    gen.research_plan_chain = MagicMock()
    gen.synopsis_chain = MagicMock()
    gen.section_chain = MagicMock()
    gen._section_str_chain = MagicMock()
    gen.evaluation_chain = MagicMock()
    gen.grounding_chain = MagicMock()
    gen.synopsis_model_id = LanguageModelId.CLAUDE_V4_5_HAIKU
    gen.writing_model_id = LanguageModelId.CLAUDE_V4_5_HAIKU
    # fit_text would call Bedrock CountTokens; stub it to a pass-through.
    factory = MagicMock()
    factory.fit_text = MagicMock(side_effect=lambda model_id, text, **kw: text)
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
        assert out[1].area == "USAGE"
        assert out[1].depth == "brief"
        assert out[1].title == "Quick install"

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
        assert "DEEP section" in out[0].depth_directive


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

    async def test_section_falls_back_to_raw_text(self) -> None:
        gen = _make_generator()
        gen.section_chain.ainvoke = AsyncMock(return_value={"section_markdown": ""})
        gen._section_str_chain.ainvoke = AsyncMock(
            return_value="<section_markdown>## Recovered\nbody</section_markdown>"
        )
        out = await gen._write_one_section(
            topic="X",
            synopsis="1. Intro",
            section="1. Intro",
            section_number=1,
            total_sections=1,
            previous_sections="",
            sources="src",
            available_images="(none)",
            depth_directive="STANDARD section.",
        )
        assert out == "## Recovered\nbody"


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
        out = await gen._ground_section("## Original\nbody", "src", 1)
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
    """The reflect-and-revise loop (ported from the review pipeline)."""

    async def test_low_score_triggers_revision(self) -> None:
        gen = _make_generator(min_quality_score=80, max_revision_attempts=1)
        gen.evaluation_chain.ainvoke = AsyncMock(
            return_value={"quality_score": "50", "improvement_feedback": "go deeper"}
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
            sources="src",
            available_images="(none)",
            final_outline="1. Intro",
        )
        gen.section_chain.ainvoke.assert_awaited()  # a revision happened
        assert out == "## Revised\ndeeper body"

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
            sources="src",
            available_images="(none)",
            final_outline="1. Intro",
        )
        gen.evaluation_chain.ainvoke.assert_not_awaited()
        assert out == "## Draft\nbody"

    def test_parse_quality_score_robust(self) -> None:
        assert TechGuideGenerator._parse_quality_score("85") == 85
        assert TechGuideGenerator._parse_quality_score("85/100") == 85
        assert TechGuideGenerator._parse_quality_score("N/A") == 0
        assert TechGuideGenerator._parse_quality_score(None) == 0


class TestMainForwardsPrUrl:
    """Regression: the guide pipeline dropped the PR url before Slack.

    `_run` returns (s3_url, pr_url); `main()` must unpack both and pass pr_url
    through to `post_slack_result` so the completion message links the PR — the
    same contract the paper review/summary path already has."""

    def test_pr_url_reaches_slack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import scholar_lens.tech_guide_main as tg

        monkeypatch.setattr(tg.Config, "load", classmethod(lambda cls: MagicMock()))
        monkeypatch.setattr(tg.boto3, "Session", lambda **kw: MagicMock())
        monkeypatch.setattr(tg, "is_running_in_aws", lambda: False)
        # GuideContext is a pydantic model; bypass its validation with a stub.
        monkeypatch.setattr(tg, "GuideContext", lambda **kw: MagicMock())
        monkeypatch.setattr(tg, "S3Handler", lambda *a, **k: MagicMock())
        monkeypatch.setattr(
            tg,
            "_run",
            AsyncMock(return_value=("s3://bucket/post.md", "https://gh/o/r/pull/7")),
        )

        captured: dict = {}
        monkeypatch.setattr(tg, "post_slack_result", lambda **kw: captured.update(kw))

        tg.main(["https://docs.x.com"], slack_channel="C1", slack_thread_ts="1.2")

        assert captured["pr_url"] == "https://gh/o/r/pull/7"
        assert captured["s3_url"] == "s3://bucket/post.md"
        assert captured["success"] is True
