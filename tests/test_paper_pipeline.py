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


def test_code_note_only_when_a_given_repository_yielded_nothing() -> None:
    from scholar_lens.src.paper_pipeline import _code_notes

    assert _code_notes(SimpleNamespace(repo_urls=["https://g/x"], codebase_summary=None))  # type: ignore[arg-type]
    assert not _code_notes(SimpleNamespace(repo_urls=["https://g/x"], codebase_summary="s"))  # type: ignore[arg-type]
    assert not _code_notes(SimpleNamespace(repo_urls=[], codebase_summary=None))  # type: ignore[arg-type]


def _context() -> SimpleNamespace:
    paper = SimpleNamespace(figure_analysis_model_id="anthropic.claude-haiku")
    code = SimpleNamespace(
        code_analysis_model_id="m",
        code_summarization_model_id="m",
        embed_model_id="e",
        chunk_size=1,
        chunk_overlap=0,
        max_files=1,
        max_chunks=1,
    )
    return SimpleNamespace(
        config=SimpleNamespace(paper=paper, code=code), bedrock_boto_session=None
    )


async def test_html_parse_failure_falls_back_to_the_pdf(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from scholar_lens.src import paper_pipeline
    from scholar_lens.src.parser import ParserError

    class FailingHTML:
        def __init__(self, **_kwargs: object) -> None: ...

        async def __aenter__(self) -> FailingHTML:
            return self

        async def __aexit__(self, *_exc: object) -> None: ...

        async def parse(self, _arxiv_id: str) -> None:
            raise ParserError("no HTML rendering")

    async def fake_pdf(source: object, kwargs: dict) -> tuple[list, str]:
        return [], "pdf content"

    monkeypatch.setattr(paper_pipeline, "LanguageModelId", lambda value: value)
    monkeypatch.setattr(paper_pipeline, "HTMLRichParser", FailingHTML)
    monkeypatch.setattr(paper_pipeline, "_parse_pdf", fake_pdf)
    source = SimpleNamespace(arxiv_html_id="2106.09685", source_id="2106_09685")
    figures, content, is_pdf = await paper_pipeline._parse_paper_content(
        _context(), source, parse_pdf=False, callbacks=[]  # type: ignore[arg-type]
    )
    assert (figures, content, is_pdf) == ([], "pdf content", True)


async def test_code_analysis_failure_degrades_to_no_codebase(  # type: ignore[no-untyped-def]
    monkeypatch, tmp_path
) -> None:
    from scholar_lens.src import paper_pipeline

    class BrokenRetriever:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("clone failed")

    monkeypatch.setattr(paper_pipeline, "LanguageModelId", lambda value: value)
    monkeypatch.setattr(paper_pipeline, "EmbeddingModelId", lambda value: value)
    monkeypatch.setattr(paper_pipeline, "CodeRetriever", BrokenRetriever)
    result = await paper_pipeline._process_code(
        _context(), tmp_path, ["https://github.com/x/y"], []  # type: ignore[arg-type]
    )
    assert result == (None, None)


def test_number_notes_name_table_rows_the_paper_never_states() -> None:
    from scholar_lens.src.paper_pipeline import _number_notes

    paper = SimpleNamespace(content=SimpleNamespace(text="respondents 272 122 199"))
    doc = "---\ntitle: x\n---\n| 비교 | A | B | C |\n| --- | --- | --- | --- |\n| n | 27 | 21 | 22 |\n"
    notes = _number_notes(paper, doc)  # type: ignore[arg-type]
    assert len(notes) == 1 and "27 / 21 / 22" in notes[0]
    assert _number_notes(paper, doc.replace("27 | 21 | 22", "272 | 122 | 199")) == []  # type: ignore[arg-type]
