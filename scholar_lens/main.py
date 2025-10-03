import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
from git import Repo
from github import Auth, Github, GithubException
from pydantic import BaseModel, HttpUrl

sys.path.append(str(Path(__file__).parent.parent))

from scholar_lens.configs import Config
from scholar_lens.src import (
    AppConstants,
    ArxivHandler,
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
    NoPythonFilesError,
    Paper,
    ParserError,
    PDFParser,
    S3Handler,
    S3Paths,
    SSMParams,
    arg_as_bool,
    get_ssm_param_value,
    is_running_in_aws,
    logger,
    plot_langchain_graph,
)

ROOT_DIR: Path = Path("/tmp") if is_running_in_aws() else Path(__file__).parent.parent
COVER_IMAGES_MAP: dict[str, str] = {
    "language-models": "language-models.jpg",
    "multimodal-learning": "multimodal-learning.jpg",
    "retrieval-augmented-generation": "retrieval-augmented-generation.jpg",
    # NOTE: add new cover images here
}


class AppContext(BaseModel):
    config: Config
    default_boto_session: boto3.Session
    bedrock_boto_session: boto3.Session
    s3_handler: S3Handler | None = None

    class Config:
        arbitrary_types_allowed = True


def main(arxiv_id: str, repo_urls: list[str] | None, parse_pdf: bool) -> None:
    config = Config.load()
    profile_name = (
        os.getenv(EnvVars.AWS_PROFILE_NAME.value)
        if is_running_in_aws()
        else config.resources.profile_name
    )

    context = AppContext(
        config=config,
        default_boto_session=boto3.Session(
            profile_name=profile_name, region_name=config.resources.default_region_name
        ),
        bedrock_boto_session=boto3.Session(
            profile_name=profile_name, region_name=config.resources.bedrock_region_name
        ),
    )
    if config.resources.s3_bucket_name:
        context.s3_handler = S3Handler(
            context.default_boto_session, config.resources.s3_bucket_name
        )

    s3_url: str | None = None
    success = False
    try:
        s3_url = asyncio.run(_run_pipeline(context, arxiv_id, repo_urls, parse_pdf))
        success = True
    except Exception as e:
        logger.error("Failed to process paper '%s': %s", arxiv_id, e, exc_info=True)
        raise
    finally:
        topic_arn = os.getenv(EnvVars.TOPIC_ARN.value)
        if is_running_in_aws() and topic_arn:
            _send_sns_notification(
                context.default_boto_session,
                topic_arn,
                success,
                arxiv_id,
                repo_urls,
                parse_pdf,
                s3_url,
            )


async def _run_pipeline(
    context: AppContext, arxiv_id: str, repo_urls: list[str] | None, parse_pdf: bool
) -> str | None:
    paper_dir = ROOT_DIR / LocalPaths.PAPERS_DIR.value / arxiv_id.replace(".", "_")
    _setup_aws_env(context)

    paper, code_retriever, citation_summarizer = await _prepare_paper_data(
        context, paper_dir, arxiv_id, repo_urls, parse_pdf
    )

    translation_guideline = await _load_translation_guideline(context)
    explainer = ExplainerGraph(
        paper=paper,
        paper_analysis_model_id=LanguageModelId(
            context.config.explanation.paper_analysis_model_id
        ),
        paper_enrichment_model_id=LanguageModelId(
            context.config.explanation.paper_enrichment_model_id
        ),
        paper_finalization_model_id=LanguageModelId(
            context.config.explanation.paper_finalization_model_id
        ),
        paper_reflection_model_id=LanguageModelId(
            context.config.explanation.paper_reflection_model_id
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
    )
    _save_workflow_graph(explainer)

    explanation, key_takeaways = await explainer.run()
    logger.info("Successfully generated explanation and key takeaways.")

    formatted_explanation = _format_explanation(paper, explanation, key_takeaways)
    s3_url, file_name = await _save_and_upload_results(
        context, paper, paper_dir, paper.title, formatted_explanation
    )

    explanation_path = paper_dir / f"{file_name}.md"

    if context.config.resources.github.enabled:
        if explanation_path.exists():
            await _create_github_pull_request(
                context, paper, explanation_path, len(formatted_explanation)
            )
        else:
            logger.warning("Explanation file not found, skipping PR creation.")

    if code_retriever:
        await code_retriever.delete_index()

    return s3_url


def _setup_aws_env(context: AppContext) -> None:
    if not is_running_in_aws():
        return

    base_path = (
        f"/{context.config.resources.project_name}/{context.config.resources.stage}"
    )
    ssm_map = {
        SSMParams.GITHUB_TOKEN: EnvVars.GITHUB_TOKEN,
        SSMParams.LANGCHAIN_API_KEY: EnvVars.LANGCHAIN_API_KEY,
        SSMParams.UPSTAGE_API_KEY: EnvVars.UPSTAGE_API_KEY,
    }
    for ssm_param, env_var in ssm_map.items():
        try:
            value = get_ssm_param_value(
                context.default_boto_session, f"{base_path}/{ssm_param.value}"
            )
            os.environ[env_var.value] = value
            logger.info("Set env var '%s' from SSM.", env_var.value)
        except Exception as e:
            logger.error("Failed to set env var '%s' from SSM: %s", env_var.value, e)


async def _prepare_paper_data(
    context: AppContext,
    paper_dir: Path,
    arxiv_id: str,
    repo_urls: list[str] | None,
    parse_pdf: bool,
) -> tuple[Paper, CodeRetriever | None, CitationSummarizer]:
    arxiv_handler = ArxivHandler()
    figures, content, is_pdf_parsed = await _parse_paper_content(
        context, arxiv_id, parse_pdf, arxiv_handler
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
    metadata = arxiv_handler.fetch_metadata(arxiv_id)
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
    citations, attributes, table_of_contents = await asyncio.gather(
        content_extractor.extract_citations(content.text),
        content_extractor.extract_attributes(content.text),
        content_extractor.extract_table_of_contents(content.text),
    )
    codebase_summary, code_retriever = await _process_code(
        context, paper_dir, repo_urls
    )
    citation_summarizer = CitationSummarizer(
        LanguageModelId(context.config.citations.citation_summarization_model_id),
        LanguageModelId(context.config.citations.citation_analysis_model_id),
        context.bedrock_boto_session,
        paper_dir=paper_dir,
    )
    converted_repo_urls = [HttpUrl(url) for url in repo_urls] if repo_urls else []
    paper = Paper(
        **metadata.model_dump(),
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
    context: AppContext, arxiv_id: str, parse_pdf: bool, arxiv_handler: ArxivHandler
) -> tuple[list[Figure], Content, bool]:
    parser_kwargs = {
        "figure_analysis_model_id": LanguageModelId(
            context.config.paper.figure_analysis_model_id
        ),
        "boto_session": context.bedrock_boto_session,
    }

    if parse_pdf:
        figures, content = await _parse_pdf(arxiv_id, parser_kwargs, arxiv_handler)
        return figures, content, True

    try:
        parser = HTMLRichParser(**parser_kwargs)
        async with parser:
            result = await parser.parse(arxiv_id)
            return result.figures, result.content, False
    except ParserError:
        logger.warning("HTML parsing failed for '%s', falling back to PDF.", arxiv_id)
        figures, content = await _parse_pdf(arxiv_id, parser_kwargs, arxiv_handler)
        return figures, content, True


async def _parse_pdf(
    arxiv_id: str, parser_kwargs: dict[str, Any], arxiv_handler: ArxivHandler
) -> tuple[list[Figure], Content]:
    papers_dir = ROOT_DIR / LocalPaths.PAPERS_DIR.value
    safe_id_dir = papers_dir / arxiv_id.replace(".", "_")
    pdf_path = arxiv_handler.download_paper(arxiv_id, papers_dir)

    parser = PDFParser(**parser_kwargs)
    async with parser:
        figures, content = await parser.parse(
            pdf_path, safe_id_dir / LocalPaths.FIGURES_DIR.value
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
        s3_prefix = context.config.resources.s3_prefix or ""
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


def _format_explanation(paper: Paper, explanation: str, key_takeaways: str) -> str:
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
    keywords_str = ", ".join(
        [f'"{kw.replace(" ", "-")}"' for kw in paper.attributes.keywords]
    )
    category_str = paper.attributes.category.replace(" ", "-").lower()
    cover_image = COVER_IMAGES_MAP.get(category_str, "default.jpg")

    categories_str = f'"Paper Reviews", "{paper.attributes.category}"'

    front_matter = front_matter_template.format(
        title=paper.title.replace('"', '\\"'),
        date=paper.published.strftime("%Y-%m-%d %H:%M:%S"),
        author=paper.attributes.affiliation,
        categories=categories_str,
        tags=keywords_str,
        cover_image=cover_image,
    )

    body = f"### TL;DR\n{key_takeaways}\n- - -\n{explanation}".replace(
        "다:", "다."
    ).replace("요:", "요.")
    references = f"\n- - -\n### References\n* [{paper.title}]({paper.pdf_url})"
    return f"{front_matter}{body}{references}"


async def _save_and_upload_results(
    context: AppContext, paper: Paper, paper_dir: Path, title: str, explanation: str
) -> tuple[str | None, str]:
    safe_title = re.sub(r"[\s,:?]", "-", title.lower()).strip("-")
    file_name = f"{datetime.now().strftime('%Y-%m-%d')}-{safe_title}"
    assets_path_str = f"/{S3Paths.ASSETS.value}/{file_name}"

    if paper.is_pdf_parsed:

        def replace_local_path(match: re.Match) -> str:
            alt_text = match.group(1)
            img_filename = match.group(2)
            if not img_filename.startswith("http"):
                return f"![{alt_text}]({assets_path_str}/{img_filename})"
            return match.group(0)

        pattern = r"!\[(.*?)\]\((.*?)\)"
        explanation = re.sub(pattern, replace_local_path, explanation)

    explanation_path = paper_dir / f"{file_name}.md"
    await asyncio.to_thread(explanation_path.write_text, explanation, encoding="utf-8")

    if not context.s3_handler:
        return None, file_name

    s3_prefix = context.config.resources.s3_prefix or ""
    posts_key = f"{s3_prefix}/{S3Paths.POSTS.value}".lstrip("/")
    assets_key = f"{s3_prefix}/{S3Paths.ASSETS.value}/{file_name}".lstrip("/")
    await context.s3_handler.upload_file_async(explanation_path, posts_key)
    figures_dir = paper_dir / LocalPaths.FIGURES_DIR.value
    if figures_dir.exists():
        await asyncio.to_thread(
            context.s3_handler.upload_directory,
            figures_dir,
            assets_key,
            file_extensions=[".gif", ".jpg", ".jpeg", ".png"],
            public_readable=True,
        )
    return (
        f"s3://{context.config.resources.s3_bucket_name}/{posts_key}/{file_name}.md",
        file_name,
    )


async def _create_github_pull_request(
    context: AppContext, paper: Paper, explanation_path: Path, num_characters: int
) -> None:
    repo_config = context.config.resources.github

    if not repo_config.repository:
        logger.error("GitHub repository not configured.")
        return

    token = os.getenv(EnvVars.GITHUB_TOKEN.value)
    if not token:
        logger.error(
            "GitHub token not found in environment variable '%s'.",
            EnvVars.GITHUB_TOKEN.value,
        )
        return

    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"paper/{paper.arxiv_id}-{timestamp}"
        clone_dir = ROOT_DIR / LocalPaths.GITHUB_CLONE_DIR.value

        commit_message = f"feat: Add paper review for '{paper.title}'"
        pr_title = f"Paper Review: {paper.title}"
        repo_info = (
            f"- **Repositories**: {', '.join([str(repo) for repo in paper.repo_urls])}\n"
            if paper.repo_urls
            else ""
        )
        pr_body = (
            f"This PR adds an AI-generated review for the paper: **{paper.title}**\n\n"
            f"- **ArXiv ID**: {paper.arxiv_id}\n"
            f"- **PDF URL**: {paper.pdf_url}\n"
            f"- **PDF Parsing**: {'enabled' if paper.is_pdf_parsed else 'disabled'}\n"
            f"- **Number of Characters**: {num_characters:,}\n"
            f"{repo_info}\n"
            f"This pull request was automatically generated by the Scholar-Lens system."
        )

        await asyncio.to_thread(
            _git_operations,
            context,
            clone_dir,
            branch_name,
            commit_message,
            explanation_path,
        )

        logger.info("Creating a pull request on GitHub...")

        auth = Auth.Token(token)
        g = Github(auth=auth)
        gh_repo = g.get_repo(repo_config.repository)

        try:
            gh_repo.create_pull(
                title=pr_title,
                body=pr_body,
                head=branch_name,
                base=repo_config.base_branch,
            )
            logger.info("Successfully created a pull request: '%s'", pr_title)
        except GithubException as e:
            if e.status == 422 and "A pull request already exists" in str(e.data):
                logger.warning(
                    "Pull request for branch '%s' already exists.", branch_name
                )
            else:
                raise

    except Exception as e:
        logger.error("Failed to create GitHub pull request: %s", e, exc_info=True)
    finally:
        if clone_dir.exists():
            await asyncio.to_thread(shutil.rmtree, clone_dir, ignore_errors=True)
            logger.info("Cleaned up local clone directory: '%s'", clone_dir)


def _git_operations(
    context: AppContext,
    clone_dir: Path,
    branch_name: str,
    commit_message: str,
    explanation_path: Path,
) -> None:
    repo_config = context.config.resources.github
    repo_url = f"https://oauth2:{os.getenv(EnvVars.GITHUB_TOKEN.value)}@github.com/{repo_config.repository}.git"

    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    logger.info("Cloning repository '%s' to '%s'", repo_config.repository, clone_dir)
    repo = Repo.clone_from(repo_url, clone_dir)

    if branch_name in repo.heads:
        new_branch = repo.heads[branch_name]
    else:
        new_branch = repo.create_head(
            branch_name, repo.remotes.origin.refs[repo_config.base_branch]
        )
    new_branch.checkout()

    posts_dir = clone_dir / LocalPaths.POSTS_DIR.value
    posts_dir.mkdir(exist_ok=True)
    shutil.copy(explanation_path, posts_dir)
    logger.info("Copied markdown file to '%s'", posts_dir)

    figures_dir = explanation_path.parent / LocalPaths.FIGURES_DIR.value
    if figures_dir.exists():
        file_name_stem = explanation_path.stem
        assets_target_dir = clone_dir / LocalPaths.ASSETS_DIR.value / file_name_stem
        assets_target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(figures_dir, assets_target_dir, dirs_exist_ok=True)
        logger.info("Copied figures to '%s'", assets_target_dir)

    if not repo.is_dirty(untracked_files=True):
        logger.warning("No changes to commit. Skipping push and pull request.")
        return

    logger.info("Committing changes...")
    repo.git.add(all=True)
    author_actor = (
        f"{repo_config.author_name} <{repo_config.author_email}>"
        if repo_config.author_email
        else repo_config.author_name
    )
    repo.git.commit("-m", commit_message, f"--author={author_actor}")

    logger.info("Pushing changes to branch '%s'...", branch_name)
    origin = repo.remote(name="origin")
    origin.push(refspec=f"{branch_name}:{branch_name}", force=True)
    origin.push(refspec=f"{branch_name}:{branch_name}", force=True)


def _send_sns_notification(
    session: boto3.Session,
    topic_arn: str,
    success: bool,
    arxiv_id: str,
    repo_urls: list[str] | None,
    parse_pdf: bool,
    s3_url: str | None,
) -> None:
    status = "succeeded" if success else "failed"
    subject = f"Paper Review {status.title()}"
    message_lines = [
        f"Paper review {status} for arXiv ID: {arxiv_id}",
        f"PDF Parsing: {'enabled' if parse_pdf else 'disabled'}",
    ]
    if repo_urls:
        message_lines.append(f"Repositories: {', '.join(repo_urls)}")
    if s3_url:
        message_lines.append(f"Output: {s3_url}")

    try:
        sns = session.client("sns")
        sns.publish(
            TopicArn=topic_arn, Subject=subject, Message="\n".join(message_lines)
        )
    except Exception as e:
        logger.error("Failed to send SNS notification: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scholar Lens: An AI-powered paper reviewer."
    )
    parser.add_argument(
        "--arxiv-id", type=str, required=True, help="arXiv ID of the paper."
    )
    parser.add_argument(
        "--repo-urls", type=str, nargs="*", help="Associated GitHub repository URLs."
    )
    parser.add_argument(
        "--parse-pdf", type=arg_as_bool, default=False, help="Force PDF parsing."
    )
    args = parser.parse_args()

    repo_urls = (
        args.repo_urls
        if args.repo_urls and args.repo_urls != [AppConstants.NULL_STRING]
        else None
    )
    logger.info("Starting paper review process with args: %s", vars(args))
    main(args.arxiv_id, repo_urls, args.parse_pdf)
