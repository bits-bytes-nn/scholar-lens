"""Regression tests for Jekyll front matter and review-prompt language wiring."""

from __future__ import annotations

from datetime import datetime

from scholar_lens.configs import Github
from scholar_lens.main import _build_front_matter
from scholar_lens.src.content_extractor import Attributes
from scholar_lens.src.explainer import Paper
from scholar_lens.src.parser import Content
from scholar_lens.src.prompts import (
    PaperEvaluationPrompt,
    PaperFinalizationPrompt,
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

    def test_author_is_paper_author_not_affiliation(self) -> None:
        # Regression: front matter must use the real author ("Ada"), never the
        # affiliation ("ACME") as the author.
        out = _build_front_matter(Github(), self._paper(), "Paper Reviews")
        assert 'author: "Ada"' in out
        assert "ACME" not in out

    def test_author_falls_back_to_affiliation_when_no_authors(self) -> None:
        paper = self._paper().model_copy(update={"authors": ["Unknown"]})
        out = _build_front_matter(Github(), paper, "Paper Reviews")
        assert 'author: "ACME"' in out

    def test_date_is_paper_published_date(self) -> None:
        # The post date is the paper's own publication date, not "now".
        out = _build_front_matter(Github(), self._paper(), "Paper Reviews")
        assert "date: 2021-05-04 08:00:00" in out

    def test_no_separate_paper_date_field(self) -> None:
        out = _build_front_matter(Github(), self._paper(), "Paper Reviews")
        assert "paper_date:" not in out


class TestReviewPromptsCarryLanguage:
    PROMPTS = [
        PaperSynthesisPrompt,
        PaperEvaluationPrompt,
        PaperFinalizationPrompt,
    ]

    def test_language_in_input_variables(self) -> None:
        for prompt in self.PROMPTS:
            assert "language" in prompt.input_variables, prompt.__name__

    def test_language_placeholder_in_template(self) -> None:
        for prompt in self.PROMPTS:
            assert "{language}" in prompt.human_prompt_template, prompt.__name__
