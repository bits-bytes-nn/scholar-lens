"""Tests for the technical-guide generator (LLM chains stubbed)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from scholar_lens.src.constants import LanguageModelId
from scholar_lens.src.tech_guide import (
    NotTechnicalContentError,
    TechGuideGenerator,
)
from scholar_lens.src.web_research import PageContent, ResearchCorpus


def _make_generator(*, verify_grounding: bool = False) -> TechGuideGenerator:
    """Construct a generator without touching Bedrock or the network."""
    gen = TechGuideGenerator.__new__(TechGuideGenerator)
    gen.language = "English"
    gen.max_sections = 12
    gen.verify_grounding = verify_grounding
    gen.researcher = MagicMock()
    gen.relevance_chain = MagicMock()
    gen.synopsis_chain = MagicMock()
    gen.section_chain = MagicMock()
    gen._section_str_chain = MagicMock()
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
    def test_numbered_list(self) -> None:
        gen = _make_generator()
        synopsis = "1. Intro - overview\n2. Setup - install\n3. Usage - examples"
        assert gen._parse_synopsis_sections(synopsis) == [
            "Intro - overview",
            "Setup - install",
            "Usage - examples",
        ]

    def test_bulleted_list(self) -> None:
        gen = _make_generator()
        assert gen._parse_synopsis_sections("- A\n* B\n• C") == ["A", "B", "C"]

    def test_caps_at_max_sections(self) -> None:
        gen = _make_generator()
        gen.max_sections = 2
        out = gen._parse_synopsis_sections("1. a\n2. b\n3. c")
        assert out == ["a", "b"]

    def test_unnumbered_synopsis_falls_back_to_lines(self) -> None:
        gen = _make_generator()
        assert gen._parse_synopsis_sections("just one line") == ["just one line"]


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
