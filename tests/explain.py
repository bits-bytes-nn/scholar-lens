import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
from scholar_lens.configs import Config
from scholar_lens.src import (
    CitationSummarizer,
    CodeRetriever,
    EmbeddingModelId,
    ExplainerGraph,
    LanguageModelId,
    LocalPaths,
    Paper,
    S3Handler,
    S3Paths,
    logger,
    plot_langchain_graph,
)

COVER_IMAGES_MAP: dict[str, str] = {
    "language-models": "language-models.jpg",
    "retrieval-augmented-generation": "retrieval-augmented-generation.jpg",
}
ARXIV_ID: str = "2302.13971"


async def async_main(arxiv_id: str) -> None:
    config = Config.load()
    profile_name = config.resources.profile_name
    default_boto_session = boto3.Session(
        profile_name=profile_name, region_name=config.resources.default_region_name
    )
    bedrock_boto_session = boto3.Session(
        profile_name=profile_name, region_name=config.resources.bedrock_region_name
    )
    s3_handler = (
        S3Handler(default_boto_session, config.resources.s3_bucket_name)
        if config.resources.s3_bucket_name
        else None
    )

    paper_dir = _get_paper_dir(arxiv_id)
    paper_data = _load_paper_data(paper_dir)
    paper = Paper(**paper_data)

    citation_summarizer = CitationSummarizer(
        LanguageModelId(config.citations.citation_summarization_model_id),
        LanguageModelId(config.citations.citation_analysis_model_id),
        bedrock_boto_session,
        paper_dir=paper_dir,
    )

    repo_urls = [str(url) for url in paper.repo_urls]
    code_retriever = await _initialize_code_retriever(
        config,
        bedrock_boto_session,
        paper_dir=paper_dir,
        repo_urls=repo_urls,
    )

    translation_guideline = _load_translation_guideline()

    explainer = ExplainerGraph(
        paper,
        LanguageModelId(config.explanation.paper_analysis_model_id),
        LanguageModelId(config.explanation.paper_enrichment_model_id),
        LanguageModelId(config.explanation.paper_finalization_model_id),
        LanguageModelId(config.explanation.paper_reflection_model_id),
        LanguageModelId(config.explanation.paper_synthesis_model_id),
        LanguageModelId(config.paper.output_fixing_model_id),
        bedrock_boto_session,
        citation_summarizer,
        code_retriever=code_retriever,
        translation_guideline=translation_guideline,
        enable_output_fixing=True,
        reflector_enable_thinking=config.explanation.reflector_enable_thinking,
        synthesizer_enable_thinking=config.explanation.synthesizer_enable_thinking,
    )
    _save_workflow_graph(explainer)

    explanation, key_takeaways = await explainer.run()
    logger.info("Successfully generated explanation and key takeaways.")

    explanation = _format_explanation(paper, explanation, key_takeaways)

    await _save_and_upload_explanation(
        paper_dir,
        paper.title,
        explanation,
        s3_handler,
        config.resources.s3_prefix,
    )

    if code_retriever:
        await code_retriever.delete_index()


def _get_paper_dir(arxiv_id: str) -> Path:
    return (
        Path(__file__).parent.parent
        / LocalPaths.PAPERS_DIR.value
        / arxiv_id.replace(".", "_")
    )


def _load_paper_data(paper_dir: Path) -> dict[str, Any]:
    paper_path = paper_dir / LocalPaths.PAPER_FILE.value
    with open(paper_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def _initialize_code_retriever(
    config: Config,
    boto_session: boto3.Session,
    paper_dir: Path | None = None,
    repo_urls: list[str] | None = None,
) -> CodeRetriever | None:
    if not repo_urls:
        return None

    code_retriever = CodeRetriever(
        LanguageModelId(config.code.code_analysis_model_id),
        LanguageModelId(config.code.code_summarization_model_id),
        EmbeddingModelId(config.code.embed_model_id),
        paper_dir=paper_dir,
        chunk_size=config.code.chunk_size,
        chunk_overlap=config.code.chunk_overlap,
        boto_session=boto_session,
    )

    await code_retriever.download_repositories(repo_urls)
    await code_retriever.create_or_load_index()
    return code_retriever


def _load_translation_guideline() -> list[list[dict[str, str]]] | None:
    guideline_path = (
        Path(__file__).parent.parent
        / LocalPaths.ASSETS_DIR.value
        / LocalPaths.TRANSLATION_GUIDELINE_FILE.value
    )
    try:
        if guideline_path.exists():
            with open(guideline_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load translation guideline: '%s'", e)
    return None


def _save_workflow_graph(explainer: ExplainerGraph) -> None:
    graph_path = (
        Path(__file__).parent.parent
        / LocalPaths.ASSETS_DIR.value
        / LocalPaths.GRAPH_FILE.value
    )
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    plot_langchain_graph(explainer.workflow, graph_path)


def _format_explanation(paper: Paper, explanation: str, key_takeaways: str) -> str:
    front_matter_template = """---
layout: post
title: "{title}"
date: {date}
author: "{author}"
categories: "{category}"
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

    front_matter = front_matter_template.format(
        title=paper.title.replace('"', '\\"'),
        date=paper.published.strftime("%Y-%m-%d %H:%M:%S"),
        author=paper.attributes.affiliation,
        category=paper.attributes.category,
        tags=keywords_str,
        cover_image=cover_image,
    )

    body = f"### TL;DR\n{key_takeaways}\n- - -\n{explanation}".replace(
        "다:", "다."
    ).replace("요:", "요.")
    references = f"\n- - -\n### References\n* [{paper.title}]({paper.pdf_url})"
    return f"{front_matter}{body}{references}"


async def _save_and_upload_explanation(
    paper_dir: Path,
    title: str,
    explanation: str,
    s3_handler: S3Handler | None,
    s3_prefix: str | None,
) -> None:
    safe_title = re.sub(r"[\s,:?]", "-", title.lower()).strip("-")
    file_name = f"{datetime.now().strftime('%Y-%m-%d')}-{safe_title}"

    assets_path_str = f"/{S3Paths.ASSETS.value}/{file_name}"
    final_explanation = explanation.replace(
        str(paper_dir / LocalPaths.FIGURES_DIR.value), assets_path_str
    )

    explanation_path = paper_dir / f"{file_name}.md"
    await asyncio.to_thread(
        explanation_path.write_text, final_explanation, encoding="utf-8"
    )
    logger.info("Saved explanation to '%s'", explanation_path)

    if not s3_handler:
        return

    prefix = s3_prefix or ""
    posts_key = f"{prefix}/{S3Paths.POSTS.value}".lstrip("/")
    assets_key = f"{prefix}/{S3Paths.ASSETS.value}/{file_name}".lstrip("/")

    await s3_handler.upload_file_async(explanation_path, posts_key)

    figures_dir = paper_dir / LocalPaths.FIGURES_DIR.value
    if figures_dir.exists():
        await asyncio.to_thread(
            s3_handler.upload_directory,
            figures_dir,
            assets_key,
            file_extensions=[".gif", ".jpg", ".jpeg", ".png"],
            public_readable=True,
        )


if __name__ == "__main__":
    asyncio.run(async_main(ARXIV_ID))
