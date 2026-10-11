"""arXiv HTML figure extraction against LaTeXML structures (no LLM, no network)."""

from __future__ import annotations

import asyncio
import re

import pytest
from bs4 import BeautifulSoup

from scholar_lens.src import parser as parser_module
from scholar_lens.src.parser import Figure, HTMLRichParser, attach_figure_context
from scholar_lens.src.utils import extract_text_from_html

BASE = "https://arxiv.org/html/2501.12948v2"

# Fig 1: two SVG <object> panels sharing one caption. Figs 15/16: one LaTeXML
# figure holding two captioned figures. Fig 2: a panel with its own sub-caption.
PAGE = """
<div class="ltx_page_main">
<figure class="ltx_figure">
  <div class="ltx_flex_figure">
    <figure class="ltx_figure ltx_figure_panel">
      <object type="image/svg+xml" class="ltx_graphics" data="2501.12948v2/aime.svg"></object>
    </figure>
    <figure class="ltx_figure ltx_figure_panel">
      <object type="image/svg+xml" class="ltx_graphics" data="2501.12948v2/length.svg"></object>
    </figure>
  </div>
  <figcaption class="ltx_caption">Figure 1: (a) AIME accuracy. (b) Response length.</figcaption>
</figure>
<figure class="ltx_figure">
  <img class="ltx_graphics" src="2501.12948v2/mmlu.svg" alt="">
  <figcaption class="ltx_caption">Figure 15: MMLU categories.</figcaption>
  <object type="image/svg+xml" class="ltx_graphics ltx_figure_panel" data="2501.12948v2/mmlu_pro.svg"></object>
  <figcaption class="ltx_caption">Figure 16: MMLU-Pro categories.</figcaption>
</figure>
<figure class="ltx_figure">
  <figure class="ltx_figure ltx_figure_panel">
    <img class="ltx_graphics" src="2501.12948v2/a.png" alt="">
    <figcaption class="ltx_caption">(a) Panel A</figcaption>
  </figure>
  <figcaption class="ltx_caption">Figure 2: We only train <math alttext="A"><mi>A</mi></math> and <math alttext="B"><mi>B</mi></math>.</figcaption>
</figure>
<figure id="S5.F11" class="ltx_figure">
  <figure id="S5.F11.sf5" class="ltx_figure">
    <img class="ltx_graphics" src="2501.12948v2/layers.png" alt="">
    <figcaption class="ltx_caption">(e) Layers 25-27</figcaption>
  </figure>
  <figcaption class="ltx_caption">Figure 11: Expert load per layer.</figcaption>
</figure>
<img class="ltx_graphics" src="data:image/png;base64,AAAA" alt="inline">
</div>
"""


@pytest.fixture
def figures(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    async def fake_from_llm(cls, _analyser, figure_id, path, caption=None):  # type: ignore[no-untyped-def]
        return cls(figure_id=figure_id, path=path, caption=caption)

    monkeypatch.setattr(parser_module.Figure, "from_llm", classmethod(fake_from_llm))
    parser = HTMLRichParser.__new__(HTMLRichParser)
    parser.figure_analyser = None  # type: ignore[assignment]
    parser.url = BASE
    return asyncio.run(parser._extract_figures(BeautifulSoup(PAGE, "html.parser")))


def _by_name(figures: list[Figure]) -> dict[str, str | None]:
    return {f.path.rsplit("/", 1)[-1]: f.caption for f in figures}


def test_every_panel_and_nested_figure_is_extracted(figures: list[Figure]) -> None:
    names = _by_name(figures)
    assert set(names) == {
        "aime.svg",
        "length.svg",
        "mmlu.svg",
        "mmlu_pro.svg",
        "a.png",
        "layers.png",
    }
    assert all(f.path.startswith(f"{BASE}/") for f in figures)


def test_each_image_gets_its_own_caption(figures: list[Figure]) -> None:
    names = _by_name(figures)
    assert names["length.svg"] == "Figure 1: (a) AIME accuracy. (b) Response length."
    assert names["mmlu.svg"] == "Figure 15: MMLU categories."
    assert names["mmlu_pro.svg"] == "Figure 16: MMLU-Pro categories."


def test_sub_caption_follows_the_figure_caption_and_math_is_kept(
    figures: list[Figure],
) -> None:
    assert _by_name(figures)["a.png"] == (
        "Figure 2: We only train $$A$$ and $$B$$. (a) Panel A"
    )


def test_plain_sub_figure_gets_the_parent_caption(figures: list[Figure]) -> None:
    # LaTeXML ``\subfigure``: a nested figure "<id>.sfN" without a panel class.
    assert _by_name(figures)["layers.png"] == (
        "Figure 11: Expert load per layer. (e) Layers 25-27"
    )


def test_no_marker_is_dropped(figures: list[Figure]) -> None:
    text = extract_text_from_html(PAGE)
    assert text.count("[Image:") == 6  # the inline data: bitmap has no marker
    attached = attach_figure_context(text, figures, is_pdf_parsed=False)
    assert len(re.findall(r'\[Image: alt="', attached)) == 6


def test_listing_indentation_survives_the_html_parser_path(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from scholar_lens.src.parser import HTMLParser

    listing = (
        '<div class="ltx_page_main"><p>Code below.</p>'
        '<div class="ltx_listing ltx_lstlisting">'
        '<div class="ltx_listingline"><span>def</span> <span>f(x):</span></div>'
        '<div class="ltx_listingline"><span class="ltx_text ltx_lst_space">    </span>'
        "<span>if</span> <span>x:</span></div>"
        '<div class="ltx_listingline"><span class="ltx_text ltx_lst_space">        </span>'
        "<span>return</span> <span>1</span></div>"
        "</div></div>"
    )

    async def fetch(self, arxiv_id):  # type: ignore[no-untyped-def]
        return listing, BASE

    monkeypatch.setattr(HTMLParser, "_fetch_html_with_fallback", fetch)
    result = asyncio.run(HTMLParser().parse("2501.12948v2"))
    assert "def f(x):\n    if x:\n        return 1" in result.content.text
