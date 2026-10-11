"""Deterministic helpers of the paper pipeline (no LLM calls)."""

from __future__ import annotations

from types import SimpleNamespace

from scholar_lens.src.paper_pipeline import _figure_captions, _unanalysed_figure_notes
from scholar_lens.src.parser import Figure


def test_arxiv_placeholder_caption_falls_back_to_the_analysis() -> None:
    paper = SimpleNamespace(
        figures=[
            Figure(
                figure_id="1",
                path="/p/figures/1.png",
                caption="Refer to caption",
                analysis="A bar chart.",
            ),
            Figure(
                figure_id="2",
                path="/p/figures/2.png",
                caption="Figure 2: Loss.",
                analysis="x",
            ),
        ]
    )
    captions = _figure_captions(paper)  # type: ignore[arg-type]
    assert captions["1.png"] == "A bar chart."
    assert captions["/p/figures/2.png"] == "Figure 2: Loss."


def test_unanalysed_figures_are_noted() -> None:
    figures = [
        Figure(figure_id="1", path="/p/1.png", caption="c", analysis=None),
        Figure(figure_id="2", path="/p/2.png", caption="c", analysis="a"),
    ]
    notes = _unanalysed_figure_notes(SimpleNamespace(figures=figures))  # type: ignore[arg-type]
    assert notes == ["1 figure(s) were not analysed (captions only): check them."]
    assert _unanalysed_figure_notes(SimpleNamespace(figures=figures[1:])) == []  # type: ignore[arg-type]
