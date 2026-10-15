"""Regression tests for Jekyll front matter and review-prompt language wiring."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from scholar_lens.configs import Github
from scholar_lens.src.content_extractor import Attributes
from scholar_lens.src.explainer import Paper
from scholar_lens.src.formatting import (
    PostMeta,
    format_guide,
    format_summary,
    paper_front_matter,
    paper_post_meta,
)
from scholar_lens.src.parser import Content
from scholar_lens.src.post_conventions import BlogPost, load_local_conventions
from scholar_lens.src.post_validator import validate_post
from scholar_lens.src.prompts import (
    PaperEditPrompt,
    PaperFinalizationPrompt,
    PaperPlanningPrompt,
    PaperSectionCheckPrompt,
    PaperSectionPrompt,
)
from scholar_lens.src.tech_guide import TechGuide


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

    def test_a_replaced_post_keeps_its_published_date_and_topic(self) -> None:
        # The model proposes another topic this run; the published post's
        # date and categories (its URL) stay, the cover follows the topic.
        published = BlogPost(
            path="_posts/2025-10-19-a-study.md",
            date=datetime(2021, 5, 3, 23, 0, tzinfo=UTC),
            categories=["Paper Reviews", "Finetuning"],
        )
        spec = load_local_conventions()
        meta = paper_post_meta(
            self._paper(), "Paper Reviews", spec, published=published
        )
        out = paper_front_matter(self._paper(), "Paper Reviews", meta)
        assert "\ndate: 2021-05-03 23:00:00\n" in out
        assert 'categories: ["Paper Reviews", "Finetuning"]' in out
        assert "cover: /assets/images/finetuning.jpg" in out

    def test_a_regenerated_summary_keeps_date_topic_and_translation(self) -> None:
        # The Korean post of a translated pair: its rerun stays half of the
        # pair (the blog fails a translation_id on one language only).
        published = BlogPost(
            path="_posts/2021-05-04-a-study-summary.md",
            date=datetime(2021, 5, 4, 8, 0, tzinfo=UTC),
            categories=["Paper Summaries", "Finetuning"],
            lang="ko",
            translation_id="a-study-summary",
        )
        twin = BlogPost(
            path="_posts/2021-05-04-a-study-summary-en.md",
            lang="en",
            translation_id="a-study-summary",
        )
        spec = load_local_conventions()
        meta = paper_post_meta(
            self._paper(), "Paper Summaries", spec, published=published
        )
        github = Github()
        result = {
            "summary": "> **요약:** 결과를 정리합니다.\n\n## 방법\n\n본문입니다.\n"
        }
        post = format_summary(github, self._paper(), result, meta=meta)
        front = post.split("\n---\n", 1)[0]
        assert "\ndate: 2021-05-04 08:00:00\n" in front
        assert 'categories: ["Paper Summaries", "Finetuning"]' in front
        assert '\nlang: "ko"\ntranslation_id: "a-study-summary"' in front
        found = validate_post(
            post,
            spec,
            "summarize",
            file_name="2021-05-04-a-study-summary.md",
            site_posts=[published, twin],
        )
        assert [v for v in found if v.rule == "translation"] == []

    def test_a_regenerated_guide_keeps_its_translation(self) -> None:
        meta = PostMeta(
            topic="Language-Models",
            cover="/assets/images/tech-guides.jpg",
            tags=["vLLM"],
            lang="en",
            translation_id="vllm-guide",
        )
        guide = TechGuide(topic="vLLM", body="## Intro\n\nText.\n")
        out = format_guide(Github(), guide, meta=meta)
        assert '\nlang: "en"\ntranslation_id: "vllm-guide"\n---\n' in out

    def test_a_guide_carries_its_description(self) -> None:
        meta = PostMeta(topic="Language-Models", cover="/c.jpg", tags=["vLLM"])
        guide = TechGuide(topic="vLLM", body="Text.\n", description="vLLM 안내")
        assert '\ndescription: "vLLM 안내"\n' in format_guide(
            Github(), guide, meta=meta
        )
        guide.description = ""
        assert "description:" not in format_guide(Github(), guide, meta=meta)

    def test_a_new_post_has_no_translation_keys(self) -> None:
        out = _build_front_matter(Github(), self._paper(), "Paper Reviews")
        assert "lang:" not in out and "translation_id" not in out

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

    @pytest.mark.parametrize(
        ("doi", "expected"),
        [  # as the blog lint takes it (lint-posts.rb lines 133-138)
            ("https://doi.org/10.1145/3292500.3330701", "10.1145/3292500.3330701"),
            ("doi:10.18653/v1/2024.acl-long.1", "10.18653/v1/2024.acl-long.1"),
            ("10.48550/arXiv.2401.06066v2", "10.48550/arXiv.2401.06066"),
            ("10.1000/a?b", None),
        ],
    )
    def test_paper_doi_is_a_bare_doi(self, doi: str, expected: str | None) -> None:
        paper = self._paper().model_copy(update={"doi": doi})
        out = _build_front_matter(Github(), paper, "Paper Reviews")
        if expected is None:
            assert "paper_doi" not in out
        else:
            assert f'paper_doi: "{expected}"' in out
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
