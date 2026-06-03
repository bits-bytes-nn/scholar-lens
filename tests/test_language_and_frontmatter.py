"""Regression tests for Jekyll front matter and review-prompt language wiring."""

from __future__ import annotations

from datetime import datetime

from scholar_lens.configs import Github
from scholar_lens.main import _build_front_matter
from scholar_lens.src.content_extractor import Attributes
from scholar_lens.src.explainer import Paper
from scholar_lens.src.parser import Content
from scholar_lens.src.prompts import (
    PaperFinalizationPrompt,
    PaperReflectionPrompt,
    PaperSynthesisPrompt,
)


class TestBuildFrontMatter:
    def _paper(self) -> Paper:
        return Paper(
            arxiv_id="2401_06066",
            title="A Study",
            authors=["Ada"],
            published=datetime(2021, 5, 4, 8, 0, 0),
            pdf_url="https://arxiv.org/pdf/2401.06066",
            content=Content(text="body"),
            attributes=Attributes(
                affiliation="ACME",
                category="Training & Inference Optimization",
                keywords=["speed & memory"],
            ),
        )

    def test_ampersand_entity_not_leaked(self) -> None:
        out = _build_front_matter(Github(), self._paper(), "Paper Reviews")
        assert "&amp;" not in out

    def test_date_is_today(self) -> None:
        out = _build_front_matter(Github(), self._paper(), "Paper Reviews")
        today = datetime.now().strftime("%Y-%m-%d")
        assert f"date: {today}" in out

    def test_paper_date_reflects_published_year(self) -> None:
        out = _build_front_matter(Github(), self._paper(), "Paper Reviews")
        assert "paper_date: 2021-05-04" in out


class TestReviewPromptsCarryLanguage:
    PROMPTS = [
        PaperSynthesisPrompt,
        PaperReflectionPrompt,
        PaperFinalizationPrompt,
    ]

    def test_language_in_input_variables(self) -> None:
        for prompt in self.PROMPTS:
            assert "language" in prompt.input_variables, prompt.__name__

    def test_language_placeholder_in_template(self) -> None:
        for prompt in self.PROMPTS:
            assert "{language}" in prompt.human_prompt_template, prompt.__name__
