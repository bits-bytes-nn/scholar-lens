"""Tests for post formatting and figure-marker helpers."""

from __future__ import annotations

from datetime import datetime

import pytest

from scholar_lens.configs import Github
from scholar_lens.src.content_extractor import Attributes
from scholar_lens.src.explainer import Paper, ReviewText
from scholar_lens.src.formatting import (
    PostMeta,
    format_guide,
    format_review,
    link_list,
    md_link_text,
    paper_post_meta,
)
from scholar_lens.src.parser import Content, Figure, attach_figure_context
from scholar_lens.src.post_conventions import load_local_conventions
from scholar_lens.src.publish_types import slugify as _slugify
from scholar_lens.src.tech_guide import TechGuide


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
        text = 'before [Image: alt="x", src="y"] after'
        assert attach_figure_context(text, [], is_pdf_parsed=False) == "before  after"

    def test_injects_analysis_caption_for_matched_figure(self, tmp_path) -> None:
        fig_path = str(tmp_path / "fig1.png")
        fig = Figure(
            figure_id="0", path=fig_path, caption="Figure 1", analysis="A bar chart."
        )
        text = '[Image: alt="Figure 1", src="fig1.png"]'
        out = attach_figure_context(text, [fig], is_pdf_parsed=False)
        assert "A bar chart." in out
        assert fig_path in out

    def test_unmatched_src_is_dropped(self) -> None:
        fig = Figure(figure_id="0", path="/tmp/known.png", analysis="x")
        text = '[Image: alt="", src="unknown.png"]'
        assert attach_figure_context(text, [fig], is_pdf_parsed=False) == ""


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


def _format_explanation(gh, paper, body, tldr, closing="", subtitle=""):  # type: ignore[no-untyped-def]
    meta = paper_post_meta(paper, gh.review_category, load_local_conventions())
    return format_review(
        gh, paper, ReviewText(body, tldr, closing, subtitle), meta=meta
    )


class TestFormatExplanation:
    def test_front_matter_and_body(self, sample_paper: Paper) -> None:
        gh = Github()
        out = _format_explanation(gh, sample_paper, "EXPLANATION", "TAKEAWAYS")
        assert out.startswith("---")
        assert "layout: post" in out
        # Title quotes are escaped in the front matter.
        assert 'title: "A \\"Great\\" Paper: Part 1"' in out
        assert "cover: /assets/images/language-models.jpg" in out
        assert "## TL;DR\nTAKEAWAYS" in out
        assert "EXPLANATION" in out
        assert "use_math" not in out  # no math in this body
        assert "description:" not in out  # none given: the blog derives one

    def test_description_goes_to_the_front_matter(self, sample_paper: Paper) -> None:
        meta = paper_post_meta(sample_paper, "Paper Reviews", load_local_conventions())
        review = ReviewText("본문", "### Q\nA", "", "부제", description='A "B" C')
        out = format_review(Github(), sample_paper, review, meta=meta)
        assert 'description: "A \\"B\\" C"\n' in out.split("---", 2)[1]

    def test_layout_order_and_closing(self, sample_paper: Paper) -> None:
        out = _format_explanation(
            Github(),
            sample_paper,
            "# 한국어 제목\n## 서론\n본문 $$x_1$$",
            "### Q\nA",
            closing="## 한계와 시사점\n한계",
        )
        assert 'subtitle: "한국어 제목"' in out  # leading H1 moved to front matter
        assert "\n# " not in out.split("---", 2)[2]
        assert out.index("## TL;DR") < out.index("## 서론") < out.index("## 한계와")
        assert out.index("## 한계와") < out.index("### References")
        assert "use_math: true" in out

    def test_subtitle_equal_to_title_falls_back_to_the_tldr(
        self, sample_paper: Paper
    ) -> None:
        out = _format_explanation(
            Github(),
            sample_paper,
            "body",
            "### Q\n결과는 기존 기법과 대등합니다. 둘째 문장.",
            subtitle=f"  {sample_paper.title.upper()} ",
        )
        assert 'subtitle: "결과는 기존 기법과 대등합니다"' in out

    def test_no_usable_subtitle_leaves_it_out(self, sample_paper: Paper) -> None:
        out = _format_explanation(
            Github(), sample_paper, "body", "### Q", subtitle=sample_paper.title
        )
        assert "subtitle:" not in out

    def test_code_repositories_listed_in_references(self, sample_paper: Paper) -> None:
        paper = sample_paper.model_copy(
            update={"repo_urls": ["https://github.com/org/repo"]}
        )
        out = _format_explanation(Github(), paper, "E", "T")
        assert "### References" in out
        # Repo appears as a plain bullet, same style as the paper entry.
        assert "* [org/repo](https://github.com/org/repo)" in out

    def test_no_repo_reference_when_no_repos(self, sample_paper: Paper) -> None:
        out = _format_explanation(Github(), sample_paper, "E", "T")
        # Only the paper bullet in References — no extra repo bullet.
        assert out.count("github.com") == 0

    def test_unknown_category_uses_default_cover(self, sample_paper: Paper) -> None:
        paper = sample_paper.model_copy(
            update={
                "attributes": Attributes(
                    affiliation="X", category="Obscure Topic", keywords=[]
                )
            }
        )
        gh = Github()
        out = _format_explanation(gh, paper, "E", "T")
        assert "cover: /assets/images/default.jpg" in out

    def test_no_punctuation_hack_passthrough(self, sample_paper: Paper) -> None:
        # A removed locale hack rewrote a trailing "다:" -> "다.". The body must
        # now pass through verbatim regardless of language — check both an
        # English colon-ending body and the original Korean one.
        gh = Github()
        out_en = _format_explanation(
            gh, sample_paper, "In conclusion: good:", "Summary:"
        )
        assert "In conclusion: good:" in out_en  # unchanged
        out_ko = _format_explanation(gh, sample_paper, "결론입니다: 좋다:", "요약:")
        assert "결론입니다: 좋다:" in out_ko  # unchanged


def test_link_text_brackets_survive_math_normalization() -> None:
    from scholar_lens.src.formatting import md_link_text
    from scholar_lens.src.markdown_math import normalize_math

    line = f"* [{md_link_text('[Re] Reproducing Foo [v2]')}](https://x.org/p)"
    assert normalize_math(line) == line
    assert "$$" not in normalize_math(line)


def test_figure_marker_stays_one_parseable_token() -> None:
    from scholar_lens.src.parser import Figure, attach_figure_context
    from scholar_lens.src.review_plan import figure_catalog

    fig = Figure(
        figure_id="1",
        path="https://arxiv.org/html/x/x1.png",
        caption="Figure 1",
        analysis='1. Shows "A".\n2. Range [0, 1].\n3. Done.',
    )
    text = attach_figure_context(
        'Text [Image: alt="", src="x1.png"] more.', [fig], is_pdf_parsed=False
    )
    assert "\n" not in text and "[0, 1]" not in text
    assert figure_catalog(text)  # the planner still sees the figure


class TestLinkLists:
    def test_titles_link_and_missing_titles_show_the_url(self) -> None:
        out = link_list(
            [
                ("Quickstart [vLLM]", "https://docs.example.org/quickstart"),
                (None, "https://example.org/a"),
                ("N/A", "https://example.org/b"),
                ("https://example.org/c", "https://example.org/c"),
            ]
        )
        assert out.splitlines() == [
            "* [Quickstart &#91;vLLM&#93;](https://docs.example.org/quickstart)",
            "* [https://example.org/a](https://example.org/a)",
            "* [https://example.org/b](https://example.org/b)",
            "* [https://example.org/c](https://example.org/c)",
        ]

    def test_one_line_per_url(self) -> None:
        out = link_list([("A", "https://x.org/p/"), ("B", "https://x.org/p")])
        assert out == "* [A](https://x.org/p/)"

    def test_title_whitespace_collapses(self) -> None:
        assert link_list([(" A\n  B ", "https://x.org")]) == "* [A B](https://x.org)"

    @pytest.mark.parametrize(
        "title,text",
        [
            # A | makes kramdown read the list item as a table.
            ("Pods | Kubernetes", "Pods &#124; Kubernetes"),
            # HTML in a page title stays text, never a live element.
            ("<img src=x onerror=alert(1)>", "&lt;img src=x onerror=alert(1)&gt;"),
            # Liquid delimiters would render as nothing (or fail the post).
            ("{{ q }} {% raw %}", "&#123;&#123; q &#125;&#125; &#123;% raw %&#125;"),
            # An entity already in the title is one character, not two.
            ("AT&amp;T & co", "AT&amp;T &amp; co"),
        ],
    )
    def test_title_markup_shows_as_text(self, title: str, text: str) -> None:
        assert link_list([(title, "https://x.org")]) == f"* [{text}](https://x.org)"

    @pytest.mark.parametrize(
        "url,target",
        [
            ("https://x.org/a b", "https://x.org/a%20b"),
            ("https://x.org/q?x=1|2&y=3", "https://x.org/q?x=1%7C2&y=3"),
            ("https://x.org/wiki/F_(b", "https://x.org/wiki/F_%28b"),
            ("https://x.org/%E3%85%8B#s", "https://x.org/%E3%85%8B#s"),
        ],
    )
    def test_url_keeps_the_link_whole(self, url: str, target: str) -> None:
        assert link_list([("A", url)]) == f"* [A]({target})"

    def test_guide_sources_link_their_page_titles(self) -> None:
        guide = TechGuide(
            topic="X 시작하기",
            body="## 설치\n본문",
            source_urls=["https://docs.x.org/install", "https://blog.x.org/post"],
            source_titles={"https://docs.x.org/install": "Installation — X"},
        )
        meta = PostMeta(topic=None, cover="/assets/images/default.jpg", tags=[])
        out = format_guide(Github(), guide, meta=meta)
        sources = out.split("### Sources\n", 1)[1]
        assert sources == (
            "* [Installation — X](https://docs.x.org/install)\n"
            "* [https://blog.x.org/post](https://blog.x.org/post)\n"
        )

    def test_paper_references_use_the_paper_title(self, sample_paper: Paper) -> None:
        out = _format_explanation(Github(), sample_paper, "E", "T")
        assert f"* [{md_link_text(sample_paper.title)}]({sample_paper.pdf_url})" in out
