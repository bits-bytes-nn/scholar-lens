"""Tests for the paper summary agent (PaperSummarizer + main formatting)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from scholar_lens.configs import Github
from scholar_lens.main import Mode, _format_summary
from scholar_lens.src.constants import LanguageModelId
from scholar_lens.src.content_extractor import Attributes
from scholar_lens.src.explainer import Paper
from scholar_lens.src.parser import Content
from scholar_lens.src.prompts import PaperSummaryPrompt
from scholar_lens.src.summarizer import PaperSummarizer


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        arxiv_id="2401_06066",
        title="A Great Paper",
        authors=["Ada Lovelace"],
        published=datetime(2024, 1, 11, 9, 30, 0),
        pdf_url="https://arxiv.org/pdf/2401.06066",
        content=Content(text="The paper body text."),
        attributes=Attributes(
            affiliation="ACME Labs",
            category="Language Models",
            keywords=["scaling laws"],
        ),
    )


def _make_summarizer_with_stub_chain(chain_result: dict[str, str]) -> PaperSummarizer:
    """Build a PaperSummarizer without touching Bedrock, stubbing the chain."""
    summarizer = PaperSummarizer.__new__(PaperSummarizer)
    summarizer.language = "Korean"
    summarizer.translation_guideline = []
    summarizer.summary_model_id = LanguageModelId.CLAUDE_V4_5_HAIKU
    # fit_text would call Bedrock CountTokens; stub it to a pass-through.
    factory = MagicMock()
    factory.fit_text = MagicMock(side_effect=lambda model_id, text, **kw: text)
    summarizer.llm_factory = factory
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=chain_result)
    summarizer.summary_chain = chain
    return summarizer


class TestPaperSummaryPrompt:
    def test_prompt_builds_with_required_variables(self) -> None:
        assert PaperSummaryPrompt.input_variables == [
            "content",
            "codebase_summary",
            "language",
            "translation_guideline",
        ]
        assert PaperSummaryPrompt.output_variables == ["summary", "tags", "urls"]
        # get_prompt must not raise (validates that every var is in a template).
        assert PaperSummaryPrompt.get_prompt() is not None

    def test_five_section_headers_present(self) -> None:
        template = PaperSummaryPrompt.human_prompt_template
        for emoji in ["🔍", "💡", "⚙️", "📊", "🔮"]:
            assert emoji in template


class TestPaperSummarizer:
    async def test_summarize_returns_summary_tags_urls(
        self, sample_paper: Paper
    ) -> None:
        summarizer = _make_summarizer_with_stub_chain(
            {
                "summary": "## 🔍 ...\n\nbody",
                "tags": "Scaling Laws, Transformers",
                "urls": "[Repo](https://github.com/x/y)",
            }
        )
        result = await summarizer.summarize(sample_paper)
        assert result["summary"].startswith("## ")
        assert result["tags"] == "Scaling Laws, Transformers"
        assert result["urls"] == "[Repo](https://github.com/x/y)"

    async def test_codebase_summary_passed_to_chain(self, sample_paper: Paper) -> None:
        # When the paper carries a codebase summary, it must reach the prompt so
        # the implementation section can be grounded in the official code.
        paper = sample_paper.model_copy(
            update={"codebase_summary": "LoRA layer in lora/layers.py"}
        )
        summarizer = _make_summarizer_with_stub_chain(
            {"summary": "## 🔍 x", "tags": "", "urls": ""}
        )
        await summarizer.summarize(paper)
        payload = summarizer.summary_chain.ainvoke.await_args.args[0]
        assert payload["codebase_summary"] == "LoRA layer in lora/layers.py"

    async def test_no_codebase_summary_uses_placeholder(
        self, sample_paper: Paper
    ) -> None:
        summarizer = _make_summarizer_with_stub_chain(
            {"summary": "## 🔍 x", "tags": "", "urls": ""}
        )
        await summarizer.summarize(sample_paper)  # codebase_summary is None
        payload = summarizer.summary_chain.ainvoke.await_args.args[0]
        # Language-neutral sentinel so the prompt behaves the same regardless of
        # the target summary language.
        assert payload["codebase_summary"] == "(no code repository provided)"

    async def test_empty_summary_raises(self, sample_paper: Paper) -> None:
        summarizer = _make_summarizer_with_stub_chain(
            {"summary": "   ", "tags": "", "urls": ""}
        )
        with pytest.raises(ValueError):
            await summarizer.summarize(sample_paper)

    def test_split_tags(self) -> None:
        assert PaperSummarizer._split_tags("A, B ,  C") == ["A", "B", "C"]
        assert PaperSummarizer._split_tags("") == []


class TestFormatSummary:
    def test_summary_front_matter_and_body(self, sample_paper: Paper) -> None:
        gh = Github(
            cover_images={"language-models": "lm.jpg"}, default_cover_image="d.jpg"
        )
        result = {
            "summary": "## 🔍 motivation",
            "tags": "Scaling Laws",
            "urls": "[Repo](https://github.com/x/y)",
        }
        out = _format_summary(gh, sample_paper, result)
        assert out.startswith("---")
        # Summaries land under the "Paper Summaries" primary category.
        assert '"Paper Summaries"' in out
        assert "cover: /assets/images/lm.jpg" in out
        assert "## 🔍 motivation" in out
        assert "### References" in out
        assert "[Repo](https://github.com/x/y)" in out

    def test_summary_without_urls_still_lists_paper(self, sample_paper: Paper) -> None:
        gh = Github()
        out = _format_summary(
            gh, sample_paper, {"summary": "x", "tags": "", "urls": ""}
        )
        assert f"[{sample_paper.title}]({sample_paper.pdf_url})" in out


class TestMode:
    def test_modes(self) -> None:
        assert Mode.REVIEW == "review"
        assert Mode.SUMMARIZE == "summarize"
        assert set(Mode.ALL) == {"review", "summarize"}
