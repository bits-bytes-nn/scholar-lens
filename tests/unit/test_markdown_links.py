"""The shared link parser and the check that a post links only its sources."""

from __future__ import annotations

import pytest

from scholar_lens.domain.ids import paper_link_key
from scholar_lens.domain.markdown.links import (
    LinkSources,
    link_key,
    markdown_links,
    unsourced_links,
    verify_links,
)
from scholar_lens.domain.publishing import verify_post_links

PAPER = "https://arxiv.org/abs/2106.09685"
DOCS = "https://docs.x.dev/guide/start"


class TestParser:
    def test_reads_every_link_form(self) -> None:
        text = (
            '[a [nested] one](https://a.dev/p "title") [b](<https://b.dev/q>) '
            "<https://c.dev/r> [c][Ref] [ref][] [ref]\n\n"
            "[ref]: https://d.dev/s 'title'\n"
        )
        links = [(link.form, link.text, link.target) for link in markdown_links(text)]
        assert links == [
            ("inline", "a [nested] one", "https://a.dev/p"),
            ("inline", "b", "https://b.dev/q"),
            ("autolink", "https://c.dev/r", "https://c.dev/r"),
            ("reference", "c", "https://d.dev/s"),
            ("reference", "ref", "https://d.dev/s"),
            ("reference", "ref", "https://d.dev/s"),
            ("definition", "ref", "https://d.dev/s"),
        ]

    def test_skips_images_code_math_and_undefined_references(self) -> None:
        text = (
            "![fig](https://f.dev/a.png) `[k](https://k.dev)` "
            "$$\\mathcal{F}[f](https://m.dev)$$ [@2106.09685] [^1] [undefined]\n\n"
            "```\n[c](https://c.dev)\n```\n"
        )
        assert markdown_links(text) == []

    def test_a_linked_image_is_a_link_with_the_image_as_text(self) -> None:
        (link,) = markdown_links("[![a](https://f.dev/a.png)](https://g.dev)")
        assert (
            link.text == "![a](https://f.dev/a.png)" and link.target == "https://g.dev"
        )

    def test_a_destination_keeps_balanced_parentheses(self) -> None:
        (link,) = markdown_links("[w](https://w.dev/Fourier_(transform)) after")
        assert link.target == "https://w.dev/Fourier_(transform)"


@pytest.mark.parametrize(
    "url",
    [
        "https://arxiv.org/abs/2106.09685",
        "https://arxiv.org/abs/2106.09685v2",
        "https://arxiv.org/pdf/2106.09685v1",
        "https://arxiv.org/pdf/2106.09685.pdf",
        "https://www.arxiv.org/abs/2106.09685",
        "http://export.arxiv.org/abs/2106.09685",
        "https://doi.org/10.48550/arXiv.2106.09685",
        "https://dx.doi.org/10.48550/arxiv.2106.09685",
    ],
)
def test_every_arxiv_form_names_one_paper(url: str) -> None:
    assert paper_link_key(url) == "arxiv:2106.09685" == link_key(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://doi.org/10.18653/v1/2024.acl-long.1",
        "https://dx.doi.org/10.18653/V1/2024.ACL-LONG.1",
        "https://www.doi.org/10.18653%2Fv1%2F2024.acl-long.1",
    ],
)
def test_every_doi_form_names_one_paper(url: str) -> None:
    assert paper_link_key(url) == "doi:10.18653/v1/2024.acl-long.1"


def test_other_urls_compare_without_scheme_www_slash_or_fragment() -> None:
    assert link_key("http://www.docs.x.dev/guide/start/#install") == link_key(DOCS)
    assert link_key("https://docs.x.dev/view?id=a") != link_key(
        "https://docs.x.dev/view?id=b"
    )
    assert paper_link_key("https://openreview.net/forum?id=abc") is None


class TestVerifyLinks:
    SOURCES = LinkSources.of(
        [PAPER, DOCS], stated_text="Code: https://github.com/microsoft/LoRA."
    )

    def test_links_to_the_sources_stay_in_any_form(self) -> None:
        text = (
            "[LoRA](https://arxiv.org/pdf/2106.09685v2) "
            "[DOI](https://doi.org/10.48550/arXiv.2106.09685) "
            "[start](https://docs.x.dev/guide/start/#install) "
            "[code](https://github.com/microsoft/LoRA)"
        )
        assert verify_links(text, self.SOURCES) == (text, [])

    def test_other_targets_keep_only_their_text(self) -> None:
        text = (
            "[Bradbury et al.](https://arxiv.org/abs/1912.03356), "
            "[review](https://openreview.net/forum?id=abc 'OpenReview'), "
            "<https://blog.example.org/made-up> and [fork](https://github.com/x/LoRA)"
        )
        fixed, removed = verify_links(text, self.SOURCES)
        assert fixed == (
            "Bradbury et al., review, https://blog.example.org/made-up and fork"
        )
        assert removed == [
            "https://arxiv.org/abs/1912.03356",
            "https://openreview.net/forum?id=abc",
            "https://blog.example.org/made-up",
            "https://github.com/x/LoRA",
        ]

    def test_a_stated_url_must_end_where_the_text_ends_it(self) -> None:
        fixed, removed = verify_links(
            "[c](https://github.com/microsoft/LoRAX)", self.SOURCES
        )
        assert fixed == "c" and removed == ["https://github.com/microsoft/LoRAX"]

    def test_an_arxiv_id_counts_only_as_a_source(self) -> None:
        sources = LinkSources.of([], stated_text="see arxiv.org/abs/1912.03356")
        _, removed = verify_links("[x](https://arxiv.org/abs/1912.03356)", sources)
        assert removed == ["https://arxiv.org/abs/1912.03356"]

    def test_a_definition_goes_with_its_references(self) -> None:
        text = "See [the post][p] and [p].\n\n[p]: https://blog.example.org/x\n"
        assert verify_links(text, self.SOURCES) == (
            "See the post and p.\n\n",
            ["https://blog.example.org/x"],
        )

    def test_code_math_images_anchors_and_site_paths_are_left(self) -> None:
        text = (
            "`[k](https://k.dev)` $$\\mathcal{F}[f](https://m.dev)$$ "
            "$(f * g)[n](k)$ ![fig](https://f.dev/a.png) "
            "[top](#tldr) [post](/posts/x)\n\n```\n[c](https://c.dev)\n```\n"
        )
        assert verify_links(text, LinkSources()) == (text, [])


def test_unsourced_links_drop_whole_list_items() -> None:
    sources = LinkSources.of([PAPER], stated_text="https://github.com/microsoft/LoRA")
    kept, removed = unsourced_links(
        "[Paper](https://arxiv.org/abs/2106.09685), "
        "[Code](https://github.com/microsoft/LoRA), [Made up](https://x.dev/y)",
        sources,
    )
    assert kept == (
        "[Paper](https://arxiv.org/abs/2106.09685), "
        "[Code](https://github.com/microsoft/LoRA)"
    )
    assert removed == ["https://x.dev/y"]


def test_a_post_gets_one_note_and_its_front_matter_is_left() -> None:
    post = '---\ntitle: "[x](https://made.up)"\n---\n[a](https://made.up) [b](https://b.dev)\n'
    fixed, notes = verify_post_links(post, LinkSources.of(["https://b.dev"]))
    assert fixed == '---\ntitle: "[x](https://made.up)"\n---\na [b](https://b.dev)\n'
    assert len(notes) == 1 and "https://made.up" in notes[0]
    assert verify_post_links(fixed, LinkSources.of(["https://b.dev"])) == (fixed, [])


@pytest.mark.parametrize(
    "stated, link",
    [
        # A paper prints http:// or no scheme; the writer links https.
        ("Project page: http://acme.io/proj", "https://acme.io/proj"),
        ("Code is at github.com/acme/tool.", "https://github.com/acme/tool"),
        ("See https://www.acme.io/docs", "https://acme.io/docs#install"),
    ],
)
def test_a_link_the_source_states_in_another_spelling_is_kept(
    stated: str, link: str
) -> None:
    from scholar_lens.domain.markdown.links import LinkSources, verify_links

    sources = LinkSources.of([], stated_text=stated)
    assert verify_links(f"[x]({link})", sources)[1] == []


def test_a_longer_host_does_not_match_a_stated_one() -> None:
    from scholar_lens.domain.markdown.links import LinkSources, verify_links

    sources = LinkSources.of([], stated_text="Code: github.com/acme/tool")
    assert verify_links("[x](https://notgithub.com/acme/tool)", sources)[1]


def test_a_link_between_two_prices_is_checked() -> None:
    # The text between two "$" with a URL's "_" in it is no math.
    from scholar_lens.domain.markdown.links import LinkSources, verify_links

    text = "비용 $0.5/호출, [t](https://evil.com/a_b) 및 $M$"
    assert verify_links(text, LinkSources.of([]))[1] == ["https://evil.com/a_b"]
    # Inline math that reads like a link stays math.
    math = r"변환 $\mathcal{F}[f](\omega)$ 입니다."
    assert verify_links(math, LinkSources.of([]))[1] == []


@pytest.mark.parametrize(
    "stated, link",
    [
        # A strict prefix of a stated URL is another target.
        ("Code: https://github.com/acme/tool-v2", "https://github.com/acme/tool"),
        # A target found inside another stated URL is not stated.
        (
            "Mirror: https://mirror.org/github.com/acme/tool",
            "https://github.com/acme/tool",
        ),
    ],
)
def test_a_target_only_part_of_a_stated_url_is_removed(stated: str, link: str) -> None:
    from scholar_lens.domain.markdown.links import LinkSources, verify_links

    assert verify_links(f"[x]({link})", LinkSources.of([], stated_text=stated))[1]


@pytest.mark.parametrize(
    "text, removed",
    [
        # Single "$" renders as text: the links between them are live.
        (
            "Set $AWS_PROFILE, see [docs](https://made-up.example.com/x), "
            "then $AWS_REGION.",
            ["https://made-up.example.com/x"],
        ),
        (
            "cost $x [p](https://arxiv.org/abs/2499.99999) y$",
            ["https://arxiv.org/abs/2499.99999"],
        ),
        # Display math is math: nothing in it is a link.
        ("$$\\mathcal{F}[f](\\omega)$$", []),
    ],
)
def test_links_between_single_dollars_are_verified(
    text: str, removed: list[str]
) -> None:
    from scholar_lens.domain.markdown.links import LinkSources, verify_links

    assert verify_links(text, LinkSources.of([]))[1] == removed
