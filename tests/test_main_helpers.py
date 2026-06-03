"""Tests for pure helpers in scholar_lens.main."""

from __future__ import annotations

from datetime import datetime

import pytest

from scholar_lens.configs import Github
from scholar_lens.main import (
    _enrich_content_with_figures,
    _format_explanation,
)
from scholar_lens.src.content_extractor import Attributes
from scholar_lens.src.explainer import Paper
from scholar_lens.src.parser import Content, Figure
from scholar_lens.src.publisher import slugify as _slugify


class TestSlugify:
    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Attention Is All You Need", "attention-is-all-you-need"),
            ("Foo/Bar: A Study?", "foo-bar-a-study"),
            ('LoRA "low-rank"', "lora-low-rank"),
            ("multiple   spaces", "multiple-spaces"),
        ],
    )
    def test_basic_slugs(self, title: str, expected: str) -> None:
        assert _slugify(title) == expected

    def test_path_separators_stripped(self) -> None:
        slug = _slugify("a/b\\c")
        assert "/" not in slug and "\\" not in slug

    def test_empty_or_punctuation_only_is_untitled(self) -> None:
        assert _slugify("///") == "untitled"
        assert _slugify("") == "untitled"

    def test_length_capped(self) -> None:
        assert len(_slugify("word " * 100, max_length=30)) <= 30


class TestEnrichContentWithFigures:
    def test_removes_image_markers_when_no_figures(self) -> None:
        text = "before [Image: alt=x, src=y] after"
        assert _enrich_content_with_figures(text, []) == "before  after"

    def test_injects_analysis_caption_for_matched_figure(self, tmp_path) -> None:
        fig_path = str(tmp_path / "fig1.png")
        fig = Figure(
            figure_id="0", path=fig_path, caption="Figure 1", analysis="A bar chart."
        )
        text = "[Image: alt=Figure 1, src=fig1.png]"
        out = _enrich_content_with_figures(text, [fig])
        assert "A bar chart." in out
        assert fig_path in out

    def test_unmatched_src_is_dropped(self) -> None:
        fig = Figure(figure_id="0", path="/tmp/known.png", analysis="x")
        text = "[Image: alt=, src=unknown.png]"
        assert _enrich_content_with_figures(text, [fig]) == ""


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        arxiv_id="2401_06066",
        title='A "Great" Paper: Part 1',
        authors=["Ada Lovelace"],
        published=datetime(2024, 1, 11, 9, 30, 0),
        pdf_url="https://arxiv.org/pdf/2401.06066",
        content=Content(text="body"),
        attributes=Attributes(
            affiliation="ACME Labs",
            category="Language Models",
            keywords=["large language models", "scaling"],
        ),
    )


class TestFormatExplanation:
    def test_front_matter_and_body(self, sample_paper: Paper) -> None:
        gh = Github(
            cover_images={"language-models": "language-models.jpg"},
            default_cover_image="default.jpg",
        )
        out = _format_explanation(gh, sample_paper, "EXPLANATION", "TAKEAWAYS")
        assert out.startswith("---")
        assert "layout: post" in out
        # Title quotes are escaped in the front matter.
        assert 'title: "A \\"Great\\" Paper: Part 1"' in out
        assert "cover: /assets/images/language-models.jpg" in out
        assert "### TL;DR\nTAKEAWAYS" in out
        assert "EXPLANATION" in out
        assert "use_math: true" in out

    def test_unknown_category_uses_default_cover(self, sample_paper: Paper) -> None:
        paper = sample_paper.model_copy(
            update={
                "attributes": Attributes(
                    affiliation="X", category="Obscure Topic", keywords=[]
                )
            }
        )
        gh = Github(default_cover_image="default.jpg")
        out = _format_explanation(gh, paper, "E", "T")
        assert "cover: /assets/images/default.jpg" in out

    def test_no_korean_punctuation_hack(self, sample_paper: Paper) -> None:
        # The removed locale hack converted "다:" -> "다." Ensure raw text passes through.
        gh = Github()
        out = _format_explanation(gh, sample_paper, "결론입니다: 좋다:", "요약:")
        assert "결론입니다: 좋다:" in out  # unchanged
