"""Reading a paper for a review or summary: the arXiv HTML rendering (else
the PDF) parsed with its figures analysed, its attributes (and, for the
sections engine, its citations) extracted, and the codebase summary of its
repositories. The sections engine's code index and citation summarizer are
built by the factories the composition root passes in."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import boto3
from pydantic import HttpUrl

from scholar_lens.adapters.code.retriever import CodeRetriever, NoPythonFilesError
from scholar_lens.adapters.documents.content_extractor import ContentExtractor
from scholar_lens.adapters.documents.parser import (
    HTMLRichParser,
    ParserError,
    PDFParser,
    attach_figure_context,
    failure_reason,
)
from scholar_lens.adapters.retry import stops_the_run
from scholar_lens.adapters.scholarly.paper_source import PaperSource
from scholar_lens.application.ports import PaperInputs
from scholar_lens.domain.constants import LanguageModelId, LocalPaths
from scholar_lens.domain.conventions import PostConventions
from scholar_lens.domain.paper.models import Content, Figure, Paper, backfill_metadata

logger = logging.getLogger(__name__)

# The blog's most-used tags shown to the tagging prompt (all of them are
# still used to match spellings).
MAX_TAG_HINTS = 200

# (paper directory, callbacks) -> the sections engine's code index (a
# CodeRetriever that also indexes) or citation summarizer.
CodeIndexFactory = Callable[[Path, list[Any]], CodeRetriever]
CitationsFactory = Callable[[Path, list[Any]], Any]


class ReaderContext(Protocol):
    """What reading a paper takes from the run: the configuration's
    ``paper``, ``code`` sections, the Bedrock session, the work directory
    and the blog's taxonomy."""

    @property
    def config(self) -> Any: ...

    @property
    def bedrock_boto_session(self) -> boto3.Session: ...

    @property
    def papers_dir(self) -> Path: ...

    def require_conventions(self) -> PostConventions: ...

    def blog_tag_hints(self, limit: int) -> list[str]: ...


async def prepare_paper_data(
    context: ReaderContext,
    paper_dir: Path,
    source: PaperSource,
    repo_urls: list[str],
    parse_pdf: bool,
    tracker: Any,
    *,
    inputs: PaperInputs,
    index_factory: CodeIndexFactory | None = None,
    citations_factory: CitationsFactory | None = None,
) -> tuple[Paper, Any, Any]:
    """Parse the paper and extract what the selected generator needs
    (``inputs``): citations and the code index are slow and costly, so they
    are built only for the generator that reads them.
    """
    callbacks = [tracker]
    figures, content, is_pdf_parsed = await _parse_paper_content(
        context, source, parse_pdf, callbacks
    )
    logger.info("Parsed the paper from %s", "PDF" if is_pdf_parsed else "HTML")
    content.text = attach_figure_context(
        content.text, figures, is_pdf_parsed=is_pdf_parsed
    )

    paper_dir.mkdir(parents=True, exist_ok=True)
    metadata = source.fetch_metadata()
    paper_cfg = context.config.paper
    content_extractor = await ContentExtractor.create(
        LanguageModelId(paper_cfg.citation_extraction_model_id),
        LanguageModelId(paper_cfg.attributes_extraction_model_id),
        context.bedrock_boto_session,
        callbacks=callbacks,
        category_options=list(context.require_conventions().topics),
        controlled_tags=context.require_conventions().tags.controlled,
        existing_keywords=context.blog_tag_hints(MAX_TAG_HINTS),
    )
    if inputs.citations:
        # The review planner reads the whole paper itself, so only citations
        # (for prerequisite summaries) are extracted beyond the attributes.
        citations, attributes = await asyncio.gather(
            content_extractor.extract_citations(content.text),
            content_extractor.extract_attributes(content.text),
        )
    else:
        attributes = await content_extractor.extract_attributes(content.text)
        citations = []
    codebase_summary, code_retriever = await _process_code(
        context,
        paper_dir,
        repo_urls,
        callbacks,
        index=inputs.code_index,
        index_factory=index_factory,
    )
    citation_summarizer = (
        citations_factory(paper_dir, callbacks)
        if inputs.citations and citations_factory is not None
        else None
    )
    meta_fields = backfill_metadata(
        metadata.model_dump(), attributes, title_is_fallback=source.title_is_fallback
    )
    paper = Paper(
        **meta_fields,
        content=content,
        attributes=attributes,
        citations=citations,
        figures=figures,
        repo_urls=[HttpUrl(url) for url in repo_urls],
        codebase_summary=codebase_summary,
        is_pdf_parsed=is_pdf_parsed,
    )
    await _write_json(
        paper_dir / LocalPaths.PAPER_FILE.value, paper.model_dump(mode="json")
    )
    logger.info("Saved paper data to '%s'", paper_dir)
    return paper, code_retriever, citation_summarizer


async def _parse_paper_content(
    context: ReaderContext,
    source: PaperSource,
    parse_pdf: bool,
    callbacks: list[Any],
) -> tuple[list[Figure], Content, bool]:
    # Callbacks carry the token tracker, so figure analysis counts toward cost.
    parser_kwargs: dict[str, Any] = {
        "figure_analysis_model_id": LanguageModelId(
            context.config.paper.figure_analysis_model_id
        ),
        "boto_session": context.bedrock_boto_session,
        "callbacks": callbacks,
    }
    arxiv_html_id = source.arxiv_html_id
    if parse_pdf or arxiv_html_id is None:
        figures, content = await _parse_pdf(source, context.papers_dir, parser_kwargs)
        return figures, content, True
    try:
        async with HTMLRichParser(**parser_kwargs) as parser:
            result = await parser.parse(arxiv_html_id)
            return result.figures, result.content, False
    except ParserError as e:
        reason = failure_reason(e)
        logger.warning(
            "HTML parsing failed for '%s' (%s); falling back to PDF.",
            source.source_id,
            reason,
        )
        figures, content = await _parse_pdf(source, context.papers_dir, parser_kwargs)
        content.notes.append(
            f"Read from the PDF: the arXiv HTML rendering could not be used ({reason})."
        )
        return figures, content, True


async def _parse_pdf(
    source: PaperSource, papers_dir: Path, parser_kwargs: dict[str, Any]
) -> tuple[list[Figure], Content]:
    pdf_path = source.download_pdf(papers_dir)
    async with PDFParser(**parser_kwargs) as parser:
        return await parser.parse(
            pdf_path, papers_dir / source.source_id / LocalPaths.FIGURES_DIR.value
        )


async def _process_code(
    context: ReaderContext,
    paper_dir: Path,
    repo_urls: list[str],
    callbacks: list[Any],
    *,
    index: bool,
    index_factory: CodeIndexFactory | None = None,
) -> tuple[str | None, Any]:
    """The codebase summary of the paper's repositories and, with ``index``,
    a retriever over its searchable code index (otherwise None: no chunk is
    analysed or embedded). Code grounding is supplementary, so a failure
    degrades to "no codebase" (or no code search, when only the index
    fails) instead of failing the whole run, except one that stops the run
    anywhere (refused or expired credentials, a spent budget)."""
    if not repo_urls:
        return None, None
    code_cfg = context.config.code
    try:
        code_index: Any = None
        code_retriever: CodeRetriever
        if index and index_factory is not None:
            code_retriever = code_index = index_factory(paper_dir, callbacks)
        else:
            code_retriever = CodeRetriever(
                LanguageModelId(code_cfg.code_summarization_model_id),
                paper_dir=paper_dir,
                boto_session=context.bedrock_boto_session,
                max_files=code_cfg.max_files,
                callbacks=callbacks,
            )
        await code_retriever.download_repositories(repo_urls)
        # The summary reads the files only: an index that cannot be built
        # (an embedding model not enabled) leaves it, and the review is
        # written without code search.
        summary = await code_retriever.generate_codebase_summary()
    except NoPythonFilesError:
        logger.warning("No Python files in the repositories; skipping code analysis.")
        return None, None
    except Exception as e:  # noqa: BLE001 - supplementary input, see docstring
        if stops_the_run(e):
            raise
        logger.warning("Code analysis failed; continuing without it: %s", e)
        return None, None
    if code_index is not None:
        try:
            await code_index.build_index()
        except Exception as e:  # noqa: BLE001 - supplementary input
            if stops_the_run(e):
                raise
            logger.warning("Code index failed; reviewing without code search: %s", e)
            code_index = None
    return summary, code_index


async def _write_json(path: Path, data: Any) -> None:
    content = json.dumps(data, indent=2, ensure_ascii=False)
    await asyncio.to_thread(path.write_text, content, encoding="utf-8")
