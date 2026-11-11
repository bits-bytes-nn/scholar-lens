"""Deterministic helpers of the paper pipeline (no LLM calls)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import boto3
import pytest

from scholar_lens.application.use_cases.paper import (
    _figure_captions,
    _unanalysed_figure_notes,
)
from scholar_lens.domain.constants import LanguageModelId
from scholar_lens.domain.paper.models import Figure
from tests.integration.fakes import FakeLLMFactory


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


def test_unsourced_citations_reach_the_review_notes() -> None:
    from scholar_lens.sections_engine.citation_summarizer import CitationSummarizer
    from scholar_lens.sections_engine.writers import _citation_notes

    summarizer = CitationSummarizer(
        LanguageModelId.CLAUDE_V4_5_HAIKU,
        boto3.Session(region_name="us-east-1"),
        arxiv_handler=MagicMock(),
        llm_factory=FakeLLMFactory(),  # type: ignore[arg-type]
    )
    summarizer._source_cache = {"found": object(), "a": None, "b": None}  # type: ignore[dict-item]
    notes = _citation_notes(summarizer)
    assert len(notes) == 1 and notes[0].startswith("2 cited reference(s)")
    assert "'a'" in notes[0] and "'found'" not in notes[0]
    summarizer._source_cache = {"found": object()}  # type: ignore[dict-item]
    assert _citation_notes(summarizer) == []


def test_code_note_only_when_a_given_repository_yielded_nothing() -> None:
    from scholar_lens.application.use_cases.paper import _code_notes

    assert _code_notes(SimpleNamespace(repo_urls=["https://g/x"], codebase_summary=None))  # type: ignore[arg-type]
    assert not _code_notes(SimpleNamespace(repo_urls=["https://g/x"], codebase_summary="s"))  # type: ignore[arg-type]
    assert not _code_notes(SimpleNamespace(repo_urls=[], codebase_summary=None))  # type: ignore[arg-type]


def test_text_only_note_when_pdf_layout_parsing_failed() -> None:
    from scholar_lens.application.use_cases.paper import _text_only_notes
    from scholar_lens.domain.paper.models import Content

    text_only = SimpleNamespace(content=Content(text="t", text_only=True))
    assert _text_only_notes(text_only) == [  # type: ignore[arg-type]
        "PDF parsed as text only; figures and tables unavailable."
    ]
    assert _text_only_notes(SimpleNamespace(content=Content(text="t"))) == []  # type: ignore[arg-type]


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
        config=SimpleNamespace(paper=paper, code=code),
        bedrock_boto_session=None,
        papers_dir=Path("papers"),
    )


async def test_html_parse_failure_falls_back_to_the_pdf(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from scholar_lens.adapters.documents import paper_reader as paper_pipeline
    from scholar_lens.adapters.documents.parser import ParserError
    from scholar_lens.application.use_cases.paper import _text_only_notes

    class FailingHTML:
        def __init__(self, **_kwargs: object) -> None: ...

        async def __aenter__(self) -> FailingHTML:
            return self

        async def __aexit__(self, *_exc: object) -> None: ...

        async def parse(self, _arxiv_id: str) -> None:
            raise ParserError("no HTML rendering")

    from scholar_lens.domain.paper.models import Content

    async def fake_pdf(
        source: object, papers_dir: Path, kwargs: dict
    ) -> tuple[list, Content]:
        return [], Content(text="pdf content")

    monkeypatch.setattr(paper_pipeline, "LanguageModelId", lambda value: value)
    monkeypatch.setattr(paper_pipeline, "HTMLRichParser", FailingHTML)
    monkeypatch.setattr(paper_pipeline, "_parse_pdf", fake_pdf)
    source = SimpleNamespace(arxiv_html_id="2106.09685", source_id="2106_09685")
    figures, content, is_pdf = await paper_pipeline._parse_paper_content(
        _context(), source, parse_pdf=False, callbacks=[]  # type: ignore[arg-type]
    )
    assert (figures, content.text, is_pdf) == ([], "pdf content", True)
    # The downgrade is a run note with its reason.
    assert content.notes == [
        "Read from the PDF: the arXiv HTML rendering could not be used "
        "(no HTML rendering)."
    ]
    paper = SimpleNamespace(content=content)
    assert _text_only_notes(paper) == content.notes  # type: ignore[arg-type]


async def test_code_analysis_failure_degrades_to_no_codebase(  # type: ignore[no-untyped-def]
    monkeypatch, tmp_path
) -> None:
    from scholar_lens.adapters.documents import paper_reader as paper_pipeline

    class BrokenRetriever:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("clone failed")

    monkeypatch.setattr(paper_pipeline, "LanguageModelId", lambda value: value)
    monkeypatch.setattr(paper_pipeline, "CodeRetriever", BrokenRetriever)
    result = await paper_pipeline._process_code(
        _context(),  # type: ignore[arg-type]
        tmp_path,
        ["https://github.com/x/y"],
        [],
        index=True,
        index_factory=BrokenRetriever,  # type: ignore[arg-type]
    )
    assert result == (None, None)


async def test_a_failed_code_index_keeps_the_codebase_summary(  # type: ignore[no-untyped-def]
    monkeypatch, tmp_path
) -> None:
    from scholar_lens.adapters.documents import paper_reader as paper_pipeline

    class Retriever:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def download_repositories(self, urls: list[str]) -> None:
            pass

        async def build_index(self) -> None:
            raise RuntimeError("embedding model not enabled")

        async def generate_codebase_summary(self) -> str:
            return "a LoRA trainer"

    monkeypatch.setattr(paper_pipeline, "LanguageModelId", lambda value: value)
    monkeypatch.setattr(paper_pipeline, "CodeRetriever", Retriever)
    result = await paper_pipeline._process_code(
        _context(),  # type: ignore[arg-type]
        tmp_path,
        ["https://github.com/x/y"],
        [],
        index=True,
        index_factory=Retriever,  # type: ignore[arg-type]
    )
    assert result == ("a LoRA trainer", None)


async def test_refused_credentials_stop_code_analysis(  # type: ignore[no-untyped-def]
    monkeypatch, tmp_path
) -> None:
    from botocore.exceptions import NoCredentialsError

    from scholar_lens.adapters.documents import paper_reader as paper_pipeline

    class Retriever:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def download_repositories(self, urls: list[str]) -> None:
            pass

        async def generate_codebase_summary(self) -> str:
            raise NoCredentialsError()

    monkeypatch.setattr(paper_pipeline, "LanguageModelId", lambda value: value)
    monkeypatch.setattr(paper_pipeline, "CodeRetriever", Retriever)
    with pytest.raises(NoCredentialsError):
        await paper_pipeline._process_code(
            _context(),  # type: ignore[arg-type]
            tmp_path,
            ["https://github.com/x/y"],
            [],
            index=False,
        )


@pytest.mark.parametrize("index", [False, True])
async def test_the_code_index_is_built_only_when_asked(  # type: ignore[no-untyped-def]
    monkeypatch, tmp_path, index: bool
) -> None:
    from scholar_lens.adapters.documents import paper_reader as paper_pipeline

    calls: list[str] = []

    class Retriever:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def download_repositories(self, urls: list[str]) -> None:
            calls.append("download")

        async def build_index(self) -> None:
            calls.append("index")

        async def generate_codebase_summary(self) -> str:
            calls.append("summary")
            return "a LoRA trainer"

    monkeypatch.setattr(paper_pipeline, "LanguageModelId", lambda value: value)
    monkeypatch.setattr(paper_pipeline, "CodeRetriever", Retriever)
    summary, retriever = await paper_pipeline._process_code(
        _context(),  # type: ignore[arg-type]
        tmp_path,
        ["https://github.com/x/y"],
        [],
        index=index,
        index_factory=Retriever,  # type: ignore[arg-type]
    )
    assert summary == "a LoRA trainer"
    assert (retriever is not None) is index
    assert calls == ["download", "summary", *(["index"] if index else [])]


@pytest.mark.parametrize(
    "kind, engine, sections_review",
    [
        ("review", "sections", True),
        ("review", "evidence", False),
        ("summarize", "sections", False),
        ("summarize", "evidence", False),
    ],
)
def test_only_the_sections_review_needs_citations_and_the_code_index(
    kind: str, engine: str, sections_review: bool
) -> None:
    from scholar_lens.application.ports import PaperInputs
    from scholar_lens.bootstrap import PaperToolkit

    context = SimpleNamespace(
        config=SimpleNamespace(explanation=SimpleNamespace(engine=engine))
    )
    assert PaperToolkit(context).inputs(kind) == PaperInputs(  # type: ignore[arg-type]
        citations=sections_review, code_index=sections_review
    )


def test_number_notes_name_table_rows_the_paper_never_states() -> None:
    from scholar_lens.application.use_cases.paper import _number_notes

    paper = SimpleNamespace(content=SimpleNamespace(text="respondents 272 122 199"))
    doc = "---\ntitle: x\n---\n| 비교 | A | B | C |\n| --- | --- | --- | --- |\n| n | 27 | 21 | 22 |\n"
    notes = _number_notes(paper, doc)  # type: ignore[arg-type]
    assert len(notes) == 1 and "27 / 21 / 22" in notes[0]
    assert _number_notes(paper, doc.replace("27 | 21 | 22", "272 | 122 | 199")) == []  # type: ignore[arg-type]


def test_a_retitled_arxiv_version_replaces_the_published_post() -> None:
    # v2 renamed the paper: the post keeps the slug (file name, image names)
    # of the post the version-free branch published.
    from pathlib import Path

    from scholar_lens.application.use_cases.paper import (
        _build_publish_request,
        paper_branch_id,
    )
    from scholar_lens.domain.conventions import BlogPost

    paper = SimpleNamespace(
        arxiv_id="2106.09685v2",
        title="LoRA v2: A New Title",
        pdf_url="https://arxiv.org/pdf/2106.09685v2",
        is_pdf_parsed=False,
        repo_urls=[],
        content=SimpleNamespace(text=""),
    )
    branch = paper_branch_id(paper, "review")  # type: ignore[arg-type]
    assert branch == "review-2106.09685"
    assert paper_branch_id(paper, "summarize") == "summary-2106.09685"  # type: ignore[arg-type]
    request = _build_publish_request(
        paper,  # type: ignore[arg-type]
        Path("/tmp/p"),
        "---\ntitle: x\n---\n",
        "review",
        branch_id=branch,
        known_tags=set(),
        published=BlogPost(path="_posts/2025-10-19-lora-low-rank-adaptation.md"),
    )
    assert request.branch_id == branch
    assert request.slug == "lora-low-rank-adaptation"
    assert request.staged_images_dir.name == "lora-low-rank-adaptation"


async def test_the_sections_writers_get_the_configured_caps(  # type: ignore[no-untyped-def]
    monkeypatch, tmp_path: Path
) -> None:
    from scholar_lens import bootstrap
    from scholar_lens.adapters.bedrock.usage import TokenUsageTracker
    from scholar_lens.configs import Config
    from scholar_lens.domain.paper.models import ReviewText
    from scholar_lens.sections_engine import writers

    given: dict[str, dict] = {}

    class Reviewer:
        def __init__(self, paper, **kwargs) -> None:  # type: ignore[no-untyped-def]
            given["review"] = kwargs

        async def run(self) -> ReviewText:
            return ReviewText(body="b", key_takeaways="k", closing="c", subtitle="")

    class Summarizer:
        def __init__(self, model_id, session, **kwargs) -> None:  # type: ignore[no-untyped-def]
            given["summary"] = kwargs
            self.notes: list[str] = []

        async def summarize(self, paper) -> dict[str, str]:  # type: ignore[no-untyped-def]
            return {"summary": "s"}

    monkeypatch.setattr(writers, "PaperReviewer", Reviewer)
    monkeypatch.setattr(writers, "PaperSummarizer", Summarizer)
    config = Config()
    config.explanation.max_total_tokens = 1_234_567
    config.explanation.min_quality_score = 83
    config.summary.max_total_tokens = 7_654_321
    session = boto3.Session(region_name="us-east-1")
    context = bootstrap.run_context(
        config,
        default_boto_session=session,
        bedrock_boto_session=session,
        root_dir=tmp_path,
    )
    sections = bootstrap.PaperToolkit(context).sections()
    guideline = [{"term": "low-rank", "translation": "저랭크"}]
    citations = MagicMock()
    citations.unsourced_references.return_value = []
    paper = MagicMock()
    await sections.review(
        paper,
        citation_summarizer=citations,
        code_retriever=None,
        translation_guideline=guideline,
        tracker=TokenUsageTracker(),
    )
    await sections.summary(
        paper, translation_guideline=guideline, tracker=TokenUsageTracker()
    )
    review, summary = given["review"], given["summary"]
    assert review["max_total_tokens"] == 1_234_567
    assert review["min_quality_score"] == 83
    assert review["translation_guideline"] == guideline
    assert summary["max_total_tokens"] == 7_654_321
    assert summary["translation_guideline"] == guideline
