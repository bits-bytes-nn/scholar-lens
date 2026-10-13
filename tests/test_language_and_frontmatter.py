"""Regression tests for Jekyll front matter and review-prompt language wiring."""

from __future__ import annotations

from datetime import datetime

from scholar_lens.configs import Github
from scholar_lens.src.content_extractor import Attributes
from scholar_lens.src.explainer import Paper
from scholar_lens.src.formatting import paper_front_matter, paper_post_meta
from scholar_lens.src.parser import Content
from scholar_lens.src.post_conventions import load_local_conventions
from scholar_lens.src.prompts import (
    PaperEditPrompt,
    PaperFinalizationPrompt,
    PaperPlanningPrompt,
    PaperSectionCheckPrompt,
    PaperSectionPrompt,
)


def _build_front_matter(_github: Github, paper: Paper, category_type: str, **kwargs):  # type: ignore[no-untyped-def]
    meta = paper_post_meta(paper, category_type, load_local_conventions())
    return paper_front_matter(paper, category_type, meta, **kwargs)


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

    def test_paper_author_is_the_organization(self) -> None:
        # The blog's feed reads `author` as the post author: paper posts name
        # the publishing organization in `paper_author` and never set `author`.
        out = _build_front_matter(Github(), self._paper(), "Paper Reviews")
        assert 'paper_author: "ACME"' in out
        assert "\nauthor:" not in out
        assert "Ada" not in out

    def test_paper_doi_is_recorded_for_the_citation_count(self) -> None:
        paper = self._paper().model_copy(update={"doi": "10.48550/arXiv.2401.06066"})
        out = _build_front_matter(Github(), paper, "Paper Reviews")
        assert 'paper_doi: "10.48550/arXiv.2401.06066"' in out
        assert "paper_doi" not in _build_front_matter(
            Github(), self._paper(), "Paper Reviews"
        )

    def test_an_arxiv_paper_uses_its_arxiv_doi(self) -> None:
        paper = self._paper().model_copy(
            update={"arxiv_id": "2305.18290v3", "doi": "10.52202/075280-2338"}
        )
        out = _build_front_matter(Github(), paper, "Paper Reviews")
        assert 'paper_doi: "10.48550/arXiv.2305.18290"' in out

    def test_placeholder_organization_omitted(self) -> None:
        paper = self._paper().model_copy(
            update={"attributes": Attributes(affiliation="N/A", category="X")}
        )
        out = _build_front_matter(Github(), paper, "Paper Reviews")
        assert "paper_author" not in out

    def test_use_math_only_with_math(self) -> None:
        assert "use_math" not in _build_front_matter(
            Github(), self._paper(), "Paper Reviews", body="plain"
        )
        assert "use_math: true" in _build_front_matter(
            Github(), self._paper(), "Paper Reviews", body="a $$x$$ b"
        )

    def test_subtitle_rendered_when_given(self) -> None:
        out = _build_front_matter(
            Github(), self._paper(), "Paper Reviews", subtitle='연구 "제목"'
        )
        assert 'subtitle: "연구 \\"제목\\""' in out

    def test_date_is_paper_published_date(self) -> None:
        # The post date is the paper's own publication date, not "now".
        out = _build_front_matter(Github(), self._paper(), "Paper Reviews")
        assert "date: 2021-05-04 08:00:00" in out

    def test_no_separate_paper_date_field(self) -> None:
        out = _build_front_matter(Github(), self._paper(), "Paper Reviews")
        assert "paper_date:" not in out


class TestReviewPromptsCarryLanguage:
    PROMPTS = [
        PaperPlanningPrompt,
        PaperSectionPrompt,
        PaperSectionCheckPrompt,
        PaperEditPrompt,
        PaperFinalizationPrompt,
    ]

    def test_language_in_input_variables(self) -> None:
        for prompt in self.PROMPTS:
            assert "language" in prompt.input_variables, prompt.__name__

    def test_language_placeholder_in_template(self) -> None:
        for prompt in self.PROMPTS:
            assert "{language}" in prompt.human_prompt_template, prompt.__name__
