"""Paper review / summary pipeline: resolve → parse → extract → generate → publish."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from pydantic import HttpUrl

from .citation_summarizer import CitationSummarizer
from .code_retriever import CodeRetriever, NoPythonFilesError
from .constants import EmbeddingModelId, LanguageModelId, LocalPaths, S3Paths
from .content_extractor import Attributes, ContentExtractor
from .explainer import Paper, PaperReviewer
from .formatting import format_review, format_summary, paper_post_meta
from .jobs import JobRequest
from .logger import logger
from .markdown_lint import (
    drop_unknown_images,
    figure_markers_to_images,
    fill_image_alts,
)
from .metrics import TokenUsageTracker
from .paper_source import PaperSource, resolve_paper_source
from .parser import (
    Content,
    Figure,
    HTMLRichParser,
    ParserError,
    PDFParser,
    attach_figure_context,
)
from .post_validator import split_front_matter, unsourced_table_numbers
from .publisher import PublishRequest, build_pr_body
from .runtime import ROOT_DIR, PipelineOutcome, RunContext, publish_artifact
from .summarizer import PaperSummarizer
from .tags import unsupported_tags
from .utils import is_placeholder

# The blog's most-used tags shown to the tagging prompt (all of them are
# still used to match spellings).
MAX_TAG_HINTS = 200


async def run_paper_pipeline(
    context: RunContext, request: JobRequest, tracker: TokenUsageTracker
) -> PipelineOutcome:
    """Generate and publish a review (``kind="review"``) or summary."""
    assert request.source is not None  # guaranteed by JobRequest validation
    source = resolve_paper_source(request.source)
    code_retriever: CodeRetriever | None = None
    try:
        # Loaded first: an S3 failure here must not follow the paid stages.
        translation_guideline = await _load_translation_guideline(context)
        paper_dir = ROOT_DIR / LocalPaths.PAPERS_DIR.value / source.source_id
        # Arbitrary PDF-URL sources have no arXiv HTML rendering.
        parse_pdf = request.parse_pdf or source.arxiv_html_id is None
        paper, code_retriever, citation_summarizer = await _prepare_paper_data(
            context,
            paper_dir,
            source,
            request.repo_urls,
            parse_pdf,
            tracker,
            needs_citations=request.kind == "review",
        )
        if request.kind == "summarize":
            document, degraded = await _generate_summary(
                context, paper, translation_guideline, tracker
            )
        else:
            assert citation_summarizer is not None
            document, degraded = await _generate_review(
                context,
                paper,
                citation_summarizer,
                code_retriever,
                translation_guideline,
                tracker,
            )
        degraded += (
            _unanalysed_figure_notes(paper)
            + _code_notes(paper)
            + _number_notes(paper, document)
        )
        # Keep only images that are real figures of this paper.
        document = figure_markers_to_images(document)
        document = drop_unknown_images(document, _figure_sources(paper))
        document = fill_image_alts(document, _figure_captions(paper), paper.title)
        publish_request = _build_publish_request(
            paper,
            paper_dir,
            document,
            request.kind,
            known_tags=set(context.require_blog_tags())
            | set(context.require_conventions().tags.controlled),
        )
        publish_request.review_notes += degraded
        outcome = await publish_artifact(
            context, publish_request, create_pr=request.create_pr
        )
        return outcome
    finally:
        if code_retriever is not None:
            await code_retriever.delete_index()
        source.close()


async def _generate_review(
    context: RunContext,
    paper: Paper,
    citation_summarizer: CitationSummarizer,
    code_retriever: CodeRetriever | None,
    translation_guideline: list[dict[str, Any]] | None,
    tracker: TokenUsageTracker,
) -> tuple[str, list[str]]:
    cfg = context.config.explanation
    reviewer = PaperReviewer(
        paper,
        planning_model_id=LanguageModelId(cfg.paper_planning_model_id),
        synthesis_model_id=LanguageModelId(cfg.paper_synthesis_model_id),
        evaluation_model_id=LanguageModelId(cfg.paper_evaluation_model_id),
        finalization_model_id=LanguageModelId(cfg.paper_finalization_model_id),
        boto_session=context.bedrock_boto_session,
        citation_summarizer=citation_summarizer,
        code_retriever=code_retriever,
        translation_guideline=translation_guideline,
        language=context.config.output_language,
        min_quality_score=cfg.min_quality_score,
        max_synthesis_attempts=cfg.max_synthesis_attempts,
        max_sections=cfg.max_sections,
        max_edit_rounds=cfg.max_edit_rounds,
        thinking_effort=cfg.thinking_effort,
        planning_effort=cfg.planning_effort,
        evaluator_enable_thinking=cfg.evaluator_enable_thinking,
        synthesizer_enable_thinking=cfg.synthesizer_enable_thinking,
        callbacks=[tracker],
        max_total_tokens=cfg.max_total_tokens,
    )
    review = await reviewer.run()
    logger.info("Generated the review body, TL;DR and closing assessment.")
    github = context.config.resources.github
    meta = paper_post_meta(
        paper,
        github.review_category,
        context.require_conventions(),
        known_tags=context.require_blog_tags(),
    )
    return format_review(github, paper, review, meta=meta), [
        *meta.notes,
        *review.notes,
    ]


async def _generate_summary(
    context: RunContext,
    paper: Paper,
    translation_guideline: list[dict[str, Any]] | None,
    tracker: TokenUsageTracker,
) -> tuple[str, list[str]]:
    cfg = context.config.summary
    summarizer = PaperSummarizer(
        LanguageModelId(cfg.summary_model_id),
        context.bedrock_boto_session,
        language=context.config.output_language,
        translation_guideline=translation_guideline,
        enable_thinking=cfg.summarizer_enable_thinking,
        thinking_effort=cfg.thinking_effort,
        max_total_tokens=cfg.max_total_tokens,
        callbacks=[tracker],
    )
    result = await summarizer.summarize(paper)
    logger.info("Generated the paper summary.")
    github = context.config.resources.github
    meta = paper_post_meta(
        paper,
        github.summary_category,
        context.require_conventions(),
        known_tags=context.require_blog_tags(),
    )
    return format_summary(github, paper, result, meta=meta), [
        *meta.notes,
        *summarizer.notes,
    ]


async def _prepare_paper_data(
    context: RunContext,
    paper_dir: Path,
    source: PaperSource,
    repo_urls: list[str],
    parse_pdf: bool,
    tracker: TokenUsageTracker,
    *,
    needs_citations: bool,
) -> tuple[Paper, CodeRetriever | None, CitationSummarizer | None]:
    """Parse the paper and extract everything the generators need.

    Citations feed only the review workflow, so the summary path skips that
    (slow, costly) extraction.
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
    await asyncio.gather(
        _write_json(paper_dir / LocalPaths.CONTENT_FILE.value, content.model_dump()),
        _write_json(
            paper_dir / LocalPaths.FIGURES_FILE.value, [f.model_dump() for f in figures]
        ),
    )
    metadata = source.fetch_metadata()
    paper_cfg = context.config.paper
    content_extractor = await ContentExtractor.create(
        LanguageModelId(paper_cfg.citation_extraction_model_id),
        LanguageModelId(paper_cfg.attributes_extraction_model_id),
        context.bedrock_boto_session,
        callbacks=callbacks,
        category_options=list(context.require_conventions().topics),
        controlled_tags=context.require_conventions().tags.controlled,
        existing_keywords=context.require_blog_tags()[:MAX_TAG_HINTS],
    )
    if needs_citations:
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
        context, paper_dir, repo_urls, callbacks
    )
    citation_summarizer = (
        CitationSummarizer(
            LanguageModelId(context.config.citations.citation_summarization_model_id),
            context.bedrock_boto_session,
            paper_dir=paper_dir,
            callbacks=callbacks,
            prefer_full_text=context.config.citations.prefer_full_text,
        )
        if needs_citations
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
    context: RunContext,
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
        figures, content = await _parse_pdf(source, parser_kwargs)
        return figures, content, True
    try:
        async with HTMLRichParser(**parser_kwargs) as parser:
            result = await parser.parse(arxiv_html_id)
            return result.figures, result.content, False
    except ParserError as e:
        logger.warning(
            "HTML parsing failed for '%s' (%s); falling back to PDF.",
            source.source_id,
            e,
        )
        figures, content = await _parse_pdf(source, parser_kwargs)
        return figures, content, True


async def _parse_pdf(
    source: PaperSource, parser_kwargs: dict[str, Any]
) -> tuple[list[Figure], Content]:
    papers_dir = ROOT_DIR / LocalPaths.PAPERS_DIR.value
    pdf_path = source.download_pdf(papers_dir)
    async with PDFParser(**parser_kwargs) as parser:
        return await parser.parse(
            pdf_path, papers_dir / source.source_id / LocalPaths.FIGURES_DIR.value
        )


async def _process_code(
    context: RunContext,
    paper_dir: Path,
    repo_urls: list[str],
    callbacks: list[Any],
) -> tuple[str | None, CodeRetriever | None]:
    """Index the paper's repositories. Code grounding is supplementary, so any
    failure degrades to "no codebase" instead of failing the whole run."""
    if not repo_urls:
        return None, None
    code_cfg = context.config.code
    try:
        code_retriever = CodeRetriever(
            LanguageModelId(code_cfg.code_analysis_model_id),
            LanguageModelId(code_cfg.code_summarization_model_id),
            EmbeddingModelId(code_cfg.embed_model_id),
            paper_dir=paper_dir,
            boto_session=context.bedrock_boto_session,
            chunk_size=code_cfg.chunk_size,
            chunk_overlap=code_cfg.chunk_overlap,
            max_files=code_cfg.max_files,
            max_chunks=code_cfg.max_chunks,
            callbacks=callbacks,
        )
        await code_retriever.download_repositories(repo_urls)
        await code_retriever.build_index()
        summary = await code_retriever.generate_codebase_summary()
        return summary, code_retriever
    except NoPythonFilesError:
        logger.warning("No Python files in the repositories; skipping code analysis.")
    except Exception as e:  # noqa: BLE001 - supplementary input, see docstring
        logger.warning("Code analysis failed; continuing without it: %s", e)
    return None, None


async def _write_json(path: Path, data: Any) -> None:
    content = json.dumps(data, indent=2, ensure_ascii=False)
    await asyncio.to_thread(path.write_text, content, encoding="utf-8")


async def _load_translation_guideline(
    context: RunContext,
) -> list[dict[str, Any]] | None:
    """The glossary-style translation guideline: the local file when present,
    else the S3 copy. It is optional, so a missing object yields ``None``; any
    other S3 error still raises (before any paid stage runs)."""
    guideline_path = (
        ROOT_DIR
        / LocalPaths.ASSETS_DIR.value
        / LocalPaths.TRANSLATION_GUIDELINE_FILE.value
    )
    if guideline_path.exists():
        return json.loads(await asyncio.to_thread(guideline_path.read_text, "utf-8"))
    if not context.s3_handler:
        return None
    s3_key = (
        f"{context.config.resources.s3_prefix}/"
        f"{S3Paths.TRANSLATION_GUIDELINE.value}/{guideline_path.name}"
    ).lstrip("/")
    text = await asyncio.to_thread(context.s3_handler.get_text, s3_key)
    if text is None:
        logger.info("No translation guideline at '%s'; continuing without.", s3_key)
        return None
    return json.loads(text)


def backfill_metadata(
    meta_fields: dict[str, Any], attributes: Attributes, *, title_is_fallback: bool
) -> dict[str, Any]:
    """Prefer the title/authors parsed from the paper text when the source had
    none (a bare PDF URL yields a URL-derived title and placeholder authors)."""
    title = str(meta_fields.get("title", "")).strip()
    if not is_placeholder(attributes.title) and (title_is_fallback or not title):
        meta_fields["title"] = attributes.title
    authors = [a for a in (meta_fields.get("authors") or []) if not is_placeholder(a)]
    if not authors and attributes.authors:
        meta_fields["authors"] = attributes.authors
    return meta_fields


def _figure_sources(paper: Paper) -> set[str]:
    """Every form a figure may take in the post: as extracted and its basename."""
    sources: set[str] = set()
    for fig in paper.figures:
        sources.add(str(fig.path))
        sources.add(Path(str(fig.path)).name)
    return sources


def _figure_captions(paper: Paper) -> dict[str, str]:
    captions: dict[str, str] = {}
    for fig in paper.figures:
        caption = fig.caption if fig.caption != "Refer to caption" else ""
        caption = caption or fig.analysis or ""  # arXiv's placeholder is no caption
        captions[str(fig.path)] = caption
        captions[Path(str(fig.path)).name] = caption
    return captions


def _unanalysed_figure_notes(paper: Paper) -> list[str]:
    """A note when figure analysis gave up on some figures (they were written
    about from their captions only)."""
    count = sum(fig.analysis is None for fig in paper.figures)
    if not count:
        return []
    return [f"{count} figure(s) were not analysed (captions only): check them."]


def _code_notes(paper: Paper) -> list[str]:
    """A note when the user gave a repository but code analysis produced
    nothing (it degrades to "no codebase" instead of failing the run)."""
    if paper.repo_urls and not paper.codebase_summary:
        return [
            "Code analysis of the given repository failed or found no Python "
            "files; the post was written without it."
        ]
    return []


def _number_notes(paper: Paper, document: str) -> list[str]:
    """Table numbers the paper never states (an invented or mistyped value
    is the defect; a derived one is fine but worth a look)."""
    missing = unsourced_table_numbers(document, paper.content.text)
    if not missing:
        return []
    return [
        "Table values not found in the paper (check each is copied or derived "
        f"correctly): {', '.join(missing)}"
    ]


def _tag_notes(paper: Paper, document: str) -> list[str]:
    """Tags whose phrase or acronym the paper itself never uses: a likely
    mis-tag worth a reviewer's look (checked against the paper, not the
    translated post)."""
    front, _, _ = split_front_matter(document)
    tags = [str(t) for t in (front or {}).get("tags") or []]
    return [
        f"Tag '{tag}' (or its acronym) does not appear in the paper; check it fits."
        for tag in unsupported_tags(tags, paper.content.text)
    ]


def _new_tags(document: str, known_tags: set[str]) -> list[str]:
    front, _, _ = split_front_matter(document)
    tags = (front or {}).get("tags") or []
    return [str(t) for t in tags if str(t) not in known_tags]


def _build_publish_request(
    paper: Paper,
    paper_dir: Path,
    document: str,
    kind: str,
    *,
    known_tags: set[str],
) -> PublishRequest:
    label = "summary" if kind == "summarize" else "review"
    pr_body = build_pr_body(
        kind=f"paper {label}",
        title=paper.title,
        fields={
            "Paper ID": paper.arxiv_id,
            "PDF URL": str(paper.pdf_url),
            "PDF Parsing": "enabled" if paper.is_pdf_parsed else "disabled",
            "Characters": f"{len(document):,}",
            "Repositories": ", ".join(str(r) for r in paper.repo_urls),
            # Tags not used by any post yet: confirm they are worth creating.
            "New tags": ", ".join(_new_tags(document, known_tags)),
        },
    )
    return PublishRequest(
        title=paper.title,
        markdown=document,
        work_dir=paper_dir,
        # Version-free id: a new arXiv version updates the same branch and PR.
        branch_id=f"{label}-{re.sub(r'v\d+$', '', paper.arxiv_id)}",
        pr_title=f"Paper {label.capitalize()}: {paper.title}",
        pr_body=pr_body,
        commit_message=f"feat: add paper {label} for '{paper.title}'",
        kind="summarize" if kind == "summarize" else "review",
        review_notes=_tag_notes(paper, document),
    )
