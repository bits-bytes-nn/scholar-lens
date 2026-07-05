import argparse
import asyncio
import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import boto3
from pydantic import HttpUrl

sys.path.append(str(Path(__file__).parent.parent))

from scholar_lens.configs import Config
from scholar_lens.configs import Github as GithubConfig
from scholar_lens.slack.notifier import post_slack_result
from scholar_lens.src import (
    AppConstants,
    Attributes,
    CitationSummarizer,
    CodeRetriever,
    Content,
    ContentExtractor,
    EmbeddingModelId,
    EnvVars,
    ExplainerGraph,
    Figure,
    HTMLRichParser,
    LanguageModelId,
    LocalPaths,
    MetricsEmitter,
    NoPythonFilesError,
    Paper,
    PaperSource,
    PaperSummarizer,
    ParserError,
    PDFParser,
    Publisher,
    PublishRequest,
    S3Paths,
    SSMParams,
    TokenUsageTracker,
    arg_as_bool,
    build_pr_body,
    is_placeholder,
    is_running_in_aws,
    logger,
    plot_langchain_graph,
    resolve_paper_source,
)
from scholar_lens.src.runtime import (
    RunContext,
    build_context,
    build_publisher,
    format_alarm,
    load_secrets_from_ssm,
    publish_sns,
)

ROOT_DIR: Path = Path("/tmp") if is_running_in_aws() else Path(__file__).parent.parent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Mode:
    """Generation modes for the pipeline."""

    REVIEW = "review"
    SUMMARIZE = "summarize"

    ALL = (REVIEW, SUMMARIZE)


# The per-run context is the shared RunContext (see scholar_lens.src.runtime);
# kept under this name so the module's internal type annotations read locally.
AppContext = RunContext


def main(
    source: str,
    repo_urls: list[str] | None,
    parse_pdf: bool,
    mode: str = Mode.REVIEW,
    slack_channel: str | None = None,
    slack_thread_ts: str | None = None,
) -> None:
    context = build_context(Config.load())

    artifact_label = "summary" if mode == Mode.SUMMARIZE else "review"

    # Resolve INSIDE the try so a construction failure (e.g. SSRF rejection of a
    # bad URL in PdfUrlSource.__init__) still reports back via SNS/Slack instead
    # of vanishing. paper_source may be None in finally — guard every access.
    paper_source: PaperSource | None = None
    s3_url: str | None = None
    pr_url: str | None = None
    success = False
    error_message: str | None = None
    try:
        paper_source = resolve_paper_source(source)
        s3_url, pr_url = asyncio.run(
            _run_pipeline(context, paper_source, repo_urls, parse_pdf, mode)
        )
        success = True
    except Exception as e:
        error_message = str(e)
        logger.error("Failed to process paper '%s': %s", source, e, exc_info=True)
        raise
    finally:
        # Fall back to the raw input for notifications when resolution failed.
        notify_title = paper_source.source_id if paper_source is not None else source
        topic_arn = os.getenv(EnvVars.TOPIC_ARN.value)
        if is_running_in_aws() and topic_arn:
            _send_sns_notification(
                context.default_boto_session,
                topic_arn,
                success,
                notify_title,
                repo_urls,
                parse_pdf,
                s3_url,
                mode,
            )
        post_slack_result(
            channel=slack_channel,
            thread_ts=slack_thread_ts,
            success=success,
            artifact_label=artifact_label,
            title=notify_title,
            s3_url=s3_url,
            pr_url=pr_url,
            error=error_message,
        )
        if paper_source is not None:
            paper_source.close()


async def _run_pipeline(
    context: AppContext,
    source: PaperSource,
    repo_urls: list[str] | None,
    parse_pdf: bool,
    mode: str = Mode.REVIEW,
) -> tuple[str | None, str | None]:
    paper_dir = ROOT_DIR / LocalPaths.PAPERS_DIR.value / source.source_id
    _setup_aws_env(context)

    # Arbitrary PDF-URL sources have no arXiv HTML rendering, so force PDF parsing.
    if source.arxiv_html_id is None and not parse_pdf:
        logger.info("Non-arXiv source detected; forcing PDF parsing.")
        parse_pdf = True

    # Create the token tracker BEFORE data prep so the citation/extraction/code
    # Bedrock calls are counted and budgeted too (not just generation).
    tracker = TokenUsageTracker()
    started = time.monotonic()
    gen_success = False
    try:
        paper, code_retriever, citation_summarizer = await _prepare_paper_data(
            context, paper_dir, source, repo_urls, parse_pdf, tracker, mode
        )
        translation_guideline = await _load_translation_guideline(context)

        if mode == Mode.SUMMARIZE:
            document = await _generate_summary(
                context, paper, translation_guideline, tracker
            )
        else:
            # Review mode always builds the citation summarizer (see
            # _prepare_paper_data); assert for the type checker.
            assert citation_summarizer is not None
            document = await _generate_review(
                context,
                paper,
                citation_summarizer,
                code_retriever,
                translation_guideline,
                tracker,
            )
        gen_success = True
    finally:
        MetricsEmitter(
            context.default_boto_session, enabled=is_running_in_aws()
        ).emit_run(
            mode=mode,
            success=gen_success,
            duration_seconds=time.monotonic() - started,
            tracker=tracker,
        )

    publisher = _build_publisher(context)
    request = _build_paper_publish_request(paper, paper_dir, document, mode)
    s3_url, document_path = await publisher.publish(request)

    pr_url: str | None = None
    if context.config.resources.github.enabled:
        pr_url = await publisher.create_pull_request(request, document_path)

    if code_retriever:
        await code_retriever.delete_index()

    return s3_url, pr_url


async def _generate_review(
    context: AppContext,
    paper: Paper,
    citation_summarizer: CitationSummarizer,
    code_retriever: CodeRetriever | None,
    translation_guideline: list[dict[str, Any]] | None,
    tracker: TokenUsageTracker,
) -> str:
    explainer = ExplainerGraph(
        paper=paper,
        callbacks=[tracker],
        paper_analysis_model_id=LanguageModelId(
            context.config.explanation.paper_analysis_model_id
        ),
        paper_enrichment_model_id=LanguageModelId(
            context.config.explanation.paper_enrichment_model_id
        ),
        paper_finalization_model_id=LanguageModelId(
            context.config.explanation.paper_finalization_model_id
        ),
        paper_evaluation_model_id=LanguageModelId(
            context.config.explanation.paper_evaluation_model_id
        ),
        paper_synthesis_model_id=LanguageModelId(
            context.config.explanation.paper_synthesis_model_id
        ),
        output_fixing_model_id=LanguageModelId(
            context.config.paper.output_fixing_model_id
        ),
        boto_session=context.bedrock_boto_session,
        citation_summarizer=citation_summarizer,
        code_retriever=code_retriever,
        translation_guideline=(
            [translation_guideline] if translation_guideline else None
        ),
        enable_output_fixing=True,
        evaluator_enable_thinking=context.config.explanation.evaluator_enable_thinking,
        synthesizer_enable_thinking=context.config.explanation.synthesizer_enable_thinking,
        thinking_effort=context.config.explanation.thinking_effort,
        language=context.config.output_language,
        min_quality_score=context.config.explanation.min_quality_score,
        max_synthesis_attempts=context.config.explanation.max_synthesis_attempts,
        max_total_tokens=context.config.explanation.max_total_tokens,
    )
    _save_workflow_graph(explainer)

    explanation, key_takeaways = await explainer.run()
    logger.info("Successfully generated explanation and key takeaways.")
    return _format_explanation(
        context.config.resources.github, paper, explanation, key_takeaways
    )


async def _generate_summary(
    context: AppContext,
    paper: Paper,
    translation_guideline: list[dict[str, Any]] | None,
    tracker: TokenUsageTracker,
) -> str:
    summarizer = PaperSummarizer(
        LanguageModelId(context.config.summary.summary_model_id),
        context.bedrock_boto_session,
        language=context.config.output_language,
        translation_guideline=translation_guideline,
        enable_thinking=context.config.summary.summarizer_enable_thinking,
        thinking_effort=context.config.summary.thinking_effort,
        max_total_tokens=context.config.summary.max_total_tokens,
        callbacks=[tracker],
    )
    result = await summarizer.summarize(paper)
    logger.info("Successfully generated paper summary.")
    return _format_summary(context.config.resources.github, paper, result)


def _setup_aws_env(context: AppContext) -> None:
    load_secrets_from_ssm(
        context,
        {
            SSMParams.GITHUB_TOKEN: EnvVars.GITHUB_TOKEN,
            SSMParams.LANGCHAIN_API_KEY: EnvVars.LANGCHAIN_API_KEY,
            SSMParams.UPSTAGE_API_KEY: EnvVars.UPSTAGE_API_KEY,
            SSMParams.SLACK_BOT_TOKEN: EnvVars.SLACK_BOT_TOKEN,
        },
    )


async def _prepare_paper_data(
    context: AppContext,
    paper_dir: Path,
    source: PaperSource,
    repo_urls: list[str] | None,
    parse_pdf: bool,
    tracker: TokenUsageTracker | None = None,
    mode: str = Mode.REVIEW,
) -> tuple[Paper, CodeRetriever | None, CitationSummarizer | None]:
    callbacks = [tracker] if tracker is not None else None
    # Citations + table-of-contents feed the review's ExplainerGraph only; the
    # summary path uses neither. Skipping them for summaries avoids a slow,
    # wasteful pass (extracting/expanding 100s of references, and a full-text TOC
    # call that can read-timeout) that never affects the output.
    needs_citations_and_toc = mode != Mode.SUMMARIZE
    figures, content, is_pdf_parsed = await _parse_paper_content(
        context, source, parse_pdf
    )
    logger.info("Paper content was parsed using %s", "PDF" if is_pdf_parsed else "HTML")

    figure_iterator = iter(figures)

    def repair_repl(match: re.Match) -> str:
        fig = next(figure_iterator, None)
        if fig:
            return f"[Image: alt=, src={Path(fig.path).name}]"
        return ""

    content.text = re.sub(r"\[Image:\s*alt=,\s*src=]", repair_repl, content.text)

    content.text = _enrich_content_with_figures(content.text, figures)

    if is_pdf_parsed:
        for fig in figures:
            content.text = content.text.replace(str(fig.path), Path(fig.path).name)

    paper_dir.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(
        _write_json_to_file_async(
            paper_dir / LocalPaths.CONTENT_FILE.value, content.model_dump()
        ),
        _write_json_to_file_async(
            paper_dir / LocalPaths.FIGURES_FILE.value, [f.model_dump() for f in figures]
        ),
    )
    metadata = source.fetch_metadata()
    content_extractor = await ContentExtractor.create(
        LanguageModelId(context.config.paper.citation_extraction_model_id),
        LanguageModelId(context.config.paper.attributes_extraction_model_id),
        LanguageModelId(context.config.paper.table_of_contents_model_id),
        LanguageModelId(context.config.paper.output_fixing_model_id),
        context.bedrock_boto_session,
        root_dir=ROOT_DIR,
        bucket_name=context.config.resources.s3_bucket_name,
        s3_prefix=context.config.resources.s3_prefix,
        enable_output_fixing=True,
    )
    if needs_citations_and_toc:
        citations, attributes, table_of_contents = await asyncio.gather(
            content_extractor.extract_citations(content.text),
            content_extractor.extract_attributes(content.text),
            content_extractor.extract_table_of_contents(content.text),
        )
    else:
        attributes = await content_extractor.extract_attributes(content.text)
        citations, table_of_contents = [], {}
    codebase_summary, code_retriever = await _process_code(
        context, paper_dir, repo_urls
    )
    citation_summarizer = (
        CitationSummarizer(
            LanguageModelId(context.config.citations.citation_summarization_model_id),
            LanguageModelId(context.config.citations.citation_analysis_model_id),
            context.bedrock_boto_session,
            paper_dir=paper_dir,
            callbacks=callbacks,
            prefer_full_text=context.config.citations.prefer_full_text,
        )
        if needs_citations_and_toc
        else None
    )
    converted_repo_urls = [HttpUrl(url) for url in repo_urls] if repo_urls else []
    # Sources without real bibliographic metadata (e.g. an arbitrary PDF URL)
    # yield a placeholder title ("Pdf") and ["Unknown"] authors. Backfill those
    # from the title/authors the attribute extractor parsed out of the PDF text.
    meta_fields = metadata.model_dump()
    meta_fields = _backfill_metadata(meta_fields, attributes)
    paper = Paper(
        **meta_fields,
        content=content,
        attributes=attributes,
        citations=citations,
        table_of_contents=table_of_contents,
        figures=figures,
        repo_urls=converted_repo_urls,
        codebase_summary=codebase_summary,
        is_pdf_parsed=is_pdf_parsed,
    )
    await _write_json_to_file_async(
        paper_dir / LocalPaths.PAPER_FILE.value, paper.model_dump(mode="json")
    )
    logger.info("Saved paper data to '%s'", paper_dir)
    return paper, code_retriever, citation_summarizer


async def _parse_paper_content(
    context: AppContext, source: PaperSource, parse_pdf: bool
) -> tuple[list[Figure], Content, bool]:
    parser_kwargs: dict[str, Any] = {
        "figure_analysis_model_id": LanguageModelId(
            context.config.paper.figure_analysis_model_id
        ),
        "boto_session": context.bedrock_boto_session,
    }

    # HTML parsing relies on arXiv's HTML rendering and is only available for
    # arXiv sources; arbitrary PDF-URL sources always go through PDF parsing.
    arxiv_html_id = source.arxiv_html_id
    if parse_pdf or arxiv_html_id is None:
        figures, content = await _parse_pdf(source, parser_kwargs)
        return figures, content, True

    try:
        parser = HTMLRichParser(**parser_kwargs)
        async with parser:
            result = await parser.parse(arxiv_html_id)
            return result.figures, result.content, False
    except ParserError:
        logger.warning(
            "HTML parsing failed for '%s', falling back to PDF.", source.source_id
        )
        figures, content = await _parse_pdf(source, parser_kwargs)
        return figures, content, True


async def _parse_pdf(
    source: PaperSource, parser_kwargs: dict[str, Any]
) -> tuple[list[Figure], Content]:
    papers_dir = ROOT_DIR / LocalPaths.PAPERS_DIR.value
    paper_dir = papers_dir / source.source_id
    pdf_path = source.download_pdf(papers_dir)

    parser = PDFParser(**parser_kwargs)
    async with parser:
        figures, content = await parser.parse(
            pdf_path, paper_dir / LocalPaths.FIGURES_DIR.value
        )
    return figures, content


async def _process_code(
    context: AppContext, paper_dir: Path, repo_urls: list[str] | None
) -> tuple[str | None, CodeRetriever | None]:
    if not repo_urls:
        return None, None

    try:
        code_retriever = CodeRetriever(
            LanguageModelId(context.config.code.code_analysis_model_id),
            LanguageModelId(context.config.code.code_summarization_model_id),
            EmbeddingModelId(context.config.code.embed_model_id),
            paper_dir=paper_dir,
            boto_session=context.bedrock_boto_session,
            chunk_size=context.config.code.chunk_size,
            chunk_overlap=context.config.code.chunk_overlap,
        )

        await code_retriever.download_repositories(repo_urls)
        await code_retriever.create_or_load_index()
        summary = await code_retriever.generate_codebase_summary()
        return summary, code_retriever
    except NoPythonFilesError:
        logger.warning("No Python files in repos; skipping code analysis.")
        return None, None


async def _write_json_to_file_async(path: Path, data: Any) -> None:
    content = json.dumps(data, indent=2, ensure_ascii=False)
    await asyncio.to_thread(path.write_text, content, encoding="utf-8")


def _enrich_content_with_figures(text: str, figures: list[Figure]) -> str:
    pattern = r"\[Image:\s*alt=(.*?),\s*src=(.*?)\]"

    if not figures:
        return re.sub(pattern, "", text)

    figure_map = {Path(fig.path).name: fig for fig in figures}

    def repl(match: re.Match) -> str:
        alt_value = match.group(1).strip()
        src_value = match.group(2).strip()

        if not src_value:
            return ""

        src_filename = Path(str(src_value)).name
        matched_figure = figure_map.get(src_filename)

        if matched_figure and matched_figure.analysis:
            clean_alt = alt_value.strip("[]") or matched_figure.caption or ""
            alt_text = clean_alt.replace('"', '\\"')
            analysis_text = matched_figure.analysis.replace('"', '\\"')

            return f'[Image: alt="{alt_text}", src="{matched_figure.path}", caption="{analysis_text}"]'

        logger.warning(
            "Could not find a matching figure or analysis for src: '%s'", src_filename
        )
        return ""

    return re.sub(pattern, repl, text)


async def _load_translation_guideline(
    context: AppContext,
) -> list[dict[str, Any]] | None:
    guideline_path = (
        ROOT_DIR
        / LocalPaths.ASSETS_DIR.value
        / LocalPaths.TRANSLATION_GUIDELINE_FILE.value
    )
    if context.s3_handler:
        s3_prefix = context.config.resources.s3_prefix
        s3_key = f"{s3_prefix}/{S3Paths.TRANSLATION_GUIDELINE.value}/{guideline_path.name}".lstrip(
            "/"
        )
        if not guideline_path.exists():
            await context.s3_handler.download_file_async(s3_key, guideline_path)

    if guideline_path.exists():
        return json.loads(await asyncio.to_thread(guideline_path.read_text, "utf-8"))
    return None


def _save_workflow_graph(explainer: ExplainerGraph) -> None:
    graph_path = ROOT_DIR / LocalPaths.ASSETS_DIR.value / LocalPaths.GRAPH_FILE.value
    graph_path.parent.mkdir(exist_ok=True, parents=True)
    plot_langchain_graph(explainer.workflow, graph_path)


def _build_front_matter(
    github_config: GithubConfig, paper: Paper, primary_category: str
) -> str:
    """Jekyll front matter shared by reviews and summaries.

    ``primary_category`` is the lead category label (e.g. "Paper Reviews" or
    "Paper Summaries"); the paper's own subject category is appended after it.
    """
    front_matter_template = """---
layout: post
title: "{title}"
date: {date}
author: "{author}"
categories: [{categories}]
tags: [{tags}]
cover: /assets/images/{cover_image}
use_math: true
---
"""
    # Unescape HTML entities (e.g. "&amp;") that the extractor may leave in the
    # category/keywords before slugifying, so they never leak into the YAML.
    category = html.unescape(paper.attributes.category)
    keywords_str = ", ".join(
        [f'"{html.unescape(kw).replace(" ", "-")}"' for kw in paper.attributes.keywords]
    )
    category_str = category.replace(" ", "-")
    cover_image = github_config.cover_image_for(category)
    categories_str = f'"{primary_category}", "{category_str}"'

    # Prefer the paper's real authors ("Vaswani et al."); fall back to the
    # extracted affiliation only when no author list is available (e.g. a
    # non-arXiv PDF whose metadata yielded none). Using affiliation AS the author
    # (the old behaviour) wrongly printed "Microsoft Corporation" as the author.
    author = _format_authors(paper.authors) or html.unescape(
        paper.attributes.affiliation
    )

    return front_matter_template.format(
        title=paper.title.replace('"', '\\"'),
        # Use the paper's own publication date as the post date.
        date=paper.published.strftime("%Y-%m-%d %H:%M:%S"),
        author=author.replace('"', '\\"'),
        categories=categories_str,
        tags=keywords_str,
        cover_image=cover_image,
    )


def _backfill_metadata(
    meta_fields: dict[str, Any], attributes: Attributes
) -> dict[str, Any]:
    """Fill placeholder title/authors from PDF-parsed attributes.

    Source metadata from a bare PDF URL has no real title/authors (title is a
    URL-stem fallback like "Pdf"; authors are ``["Unknown"]``). When the
    extractor recovered them from the document text, prefer those.
    """
    title = str(meta_fields.get("title", "")).strip()
    if attributes.title and (not title or title.lower() in {"pdf", "untitled"}):
        meta_fields["title"] = attributes.title
    authors = [a for a in (meta_fields.get("authors") or []) if not is_placeholder(a)]
    if not authors and attributes.authors:
        meta_fields["authors"] = attributes.authors
    return meta_fields


def _format_authors(authors: list[str]) -> str:
    """Render an author list as front-matter text ("A", "A and B", "A et al.").

    Filters out junk placeholders ("Unknown") that the PDF-URL path emits when
    no real author metadata is available.
    """
    names = [a.strip() for a in authors if not is_placeholder(a)]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]} et al."


def _format_explanation(
    github_config: GithubConfig, paper: Paper, explanation: str, key_takeaways: str
) -> str:
    front_matter = _build_front_matter(
        github_config, paper, github_config.review_category
    )
    body = f"### TL;DR\n{key_takeaways}\n- - -\n{explanation}"
    references_lines = [f"* [{_md_link_text(paper.title)}]({paper.pdf_url})"]
    references_lines.extend(_code_repository_references(paper))
    references = "\n- - -\n### References\n" + "\n".join(references_lines)
    return f"{front_matter}{body}{references}"


def _md_link_text(text: str) -> str:
    """Escape brackets in Markdown link text so a title with ``]`` doesn't break
    the ``[text](url)`` syntax."""
    return text.replace("[", "\\[").replace("]", "\\]")


def _code_repository_references(paper: Paper) -> list[str]:
    """Reference bullets for the paper's associated code repositories, if any.

    Uses the same bullet style as the paper entry (no label); the repo URL's last
    path segment ("owner/name") is the link text so it reads as a repository
    rather than a bare URL."""
    lines = []
    for repo in paper.repo_urls:
        url = str(repo)
        name = "/".join(url.rstrip("/").split("/")[-2:]) or url
        lines.append(f"* [{_md_link_text(name)}]({url})")
    return lines


def _format_summary(
    github_config: GithubConfig, paper: Paper, result: dict[str, str]
) -> str:
    """Render a summary post: front matter + Markdown summary + extracted URLs."""
    front_matter = _build_front_matter(
        github_config, paper, github_config.summary_category
    )
    body = result["summary"]
    references_lines = [f"* [{_md_link_text(paper.title)}]({paper.pdf_url})"]
    references_lines.extend(_code_repository_references(paper))
    if urls := result.get("urls", "").strip():
        references_lines.append(urls)
    references = "\n- - -\n### References\n" + "\n".join(references_lines)
    return f"{front_matter}{body}{references}"


def _build_publisher(context: AppContext) -> Publisher:
    return build_publisher(context, ROOT_DIR)


def _build_paper_publish_request(
    paper: Paper, paper_dir: Path, document: str, mode: str
) -> PublishRequest:
    artifact_label = "summary" if mode == Mode.SUMMARIZE else "review"
    pr_body = build_pr_body(
        kind=f"paper {artifact_label}",
        title=paper.title,
        fields={
            "Paper ID": paper.arxiv_id or "",
            "PDF URL": str(paper.pdf_url) if paper.pdf_url else "",
            "PDF Parsing": "enabled" if paper.is_pdf_parsed else "disabled",
            "Characters": f"{len(document):,}",
            "Repositories": ", ".join(str(r) for r in paper.repo_urls),
        },
    )
    return PublishRequest(
        title=paper.title,
        markdown=document,
        work_dir=paper_dir,
        branch_id=paper.arxiv_id,
        pr_title=f"Paper {artifact_label.capitalize()}: {paper.title}",
        pr_body=pr_body,
        commit_message=f"feat: Add paper {artifact_label} for '{paper.title}'",
        rewrite_local_images=paper.is_pdf_parsed,
    )


def _send_sns_notification(
    session: boto3.Session,
    topic_arn: str,
    success: bool,
    source_id: str,
    repo_urls: list[str] | None,
    parse_pdf: bool,
    s3_url: str | None,
    mode: str = Mode.REVIEW,
) -> None:
    # Only failures are alarm-worthy; a successful run is reported in Slack, not SNS.
    if success:
        return

    artifact = "Summary" if mode == Mode.SUMMARIZE else "Review"
    fields = {
        "Source": source_id,
        "PDF Parsing": "enabled" if parse_pdf else "disabled",
    }
    if repo_urls:
        fields["Repositories"] = ", ".join(repo_urls)
    if s3_url:
        fields["Output"] = s3_url

    subject, message = format_alarm(
        event=f"Paper {artifact}", status="FAILED", fields=fields
    )
    publish_sns(session, topic_arn, subject=subject, message=message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scholar Lens: An AI-powered paper reviewer."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source",
        type=str,
        help="arXiv ID (e.g. 2401.06066) or a URL to a paper PDF.",
    )
    source_group.add_argument(
        "--arxiv-id",
        type=str,
        help="[Deprecated] arXiv ID of the paper. Use --source instead.",
    )
    parser.add_argument(
        "--repo-urls", type=str, nargs="*", help="Associated GitHub repository URLs."
    )
    parser.add_argument(
        "--parse-pdf", type=arg_as_bool, default=False, help="Force PDF parsing."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=Mode.ALL,
        default=Mode.REVIEW,
        help="Generation mode: 'review' (in-depth) or 'summarize' (concise).",
    )
    parser.add_argument(
        "--slack-channel",
        type=str,
        default=None,
        help="Slack channel to post the result back to (set by the bot).",
    )
    parser.add_argument(
        "--slack-thread-ts",
        type=str,
        default=None,
        help="Slack thread timestamp to reply in (set by the bot).",
    )
    args = parser.parse_args()

    # The Slack dispatcher passes repo URLs as ONE space-joined Batch parameter
    # (Ref::repo_urls), which Batch substitutes as a single argv token — so with
    # nargs="*" two repos arrive as ["urlA urlB"], one bogus string. Split on
    # whitespace and drop the NULL sentinel (parity with tech_guide_main._flatten).
    flattened_repo_urls: list[str] = []
    for value in args.repo_urls or []:
        flattened_repo_urls.extend(value.split())
    repo_urls = [
        url for url in flattened_repo_urls if url and url != AppConstants.NULL_STRING
    ] or None
    source = args.source or args.arxiv_id
    logger.info(
        "Starting paper %s process: source=%s repo_urls=%s parse_pdf=%s",
        args.mode,
        source,
        repo_urls,
        args.parse_pdf,
    )
    main(
        source,
        repo_urls,
        args.parse_pdf,
        args.mode,
        slack_channel=args.slack_channel,
        slack_thread_ts=args.slack_thread_ts,
    )
