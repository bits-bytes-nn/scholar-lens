"""Technical-guide pipeline: research the URLs, write the guide, publish it."""

from __future__ import annotations

import os
from pathlib import Path

from .constants import EnvVars, LanguageModelId, LocalPaths
from .formatting import build_post_meta, format_guide
from .jobs import JobRequest
from .logger import logger
from .metrics import TokenUsageTracker
from .publisher import PublishRequest, build_pr_body, slugify
from .runtime import ROOT_DIR, PipelineOutcome, RunContext, publish_artifact
from .tech_guide import TechGuide, TechGuideGenerator
from .web_research import (
    BraveSearchProvider,
    NullSearchProvider,
    TavilySearchProvider,
    WebResearcher,
    WebSearchProvider,
)


async def run_guide_pipeline(
    context: RunContext, request: JobRequest, tracker: TokenUsageTracker
) -> PipelineOutcome:
    """Generate and publish a guide; raises ``NotTechnicalContentError`` when
    the URLs are not technical documentation."""
    cfg = context.config.tech_guide
    researcher = WebResearcher(search_provider=build_search_provider())
    generator = TechGuideGenerator(
        relevance_model_id=LanguageModelId(cfg.relevance_model_id),
        synopsis_model_id=LanguageModelId(cfg.synopsis_model_id),
        writing_model_id=LanguageModelId(cfg.writing_model_id),
        evaluation_model_id=LanguageModelId(cfg.evaluation_model_id),
        boto_session=context.bedrock_boto_session,
        researcher=researcher,
        language=context.config.output_language,
        enable_thinking=cfg.writer_enable_thinking,
        evaluator_enable_thinking=cfg.evaluator_enable_thinking,
        thinking_effort=cfg.thinking_effort,
        verify_grounding=cfg.verify_grounding,
        auto_research=cfg.auto_research,
        max_research_queries=cfg.max_research_queries,
        fetch_top_results=cfg.fetch_top_results,
        min_quality_score=cfg.min_quality_score,
        max_revision_attempts=cfg.max_revision_attempts,
        max_total_tokens=cfg.max_total_tokens,
        callbacks=[tracker],
    )
    try:
        guide = await generator.generate(
            request.urls,
            discover_subpages=request.discover_subpages,
            search_queries=request.search_queries or None,
        )
    finally:
        researcher.close()

    spec = context.require_conventions()
    github = context.config.resources.github
    taxonomy = await generator.classify(
        guide,
        category_options=list(spec.topics),
        controlled_tags=spec.tags.controlled,
    )
    meta = build_post_meta(
        title=guide.topic,
        category_type=github.tech_guide_category,
        proposed_topic=taxonomy.get("category", ""),
        specific_tags=_split_list(taxonomy.get("keywords", "")),
        controlled_tags=_split_list(taxonomy.get("controlled_tags", "")),
        proposed_seo_title=taxonomy.get("seo_title"),
        spec=spec,
        known_tags=context.require_blog_tags(),
    )
    slug = slugify(guide.topic, max_length=80)
    work_dir = ROOT_DIR / LocalPaths.PAPERS_DIR.value / f"guide-{slug}"
    work_dir.mkdir(parents=True, exist_ok=True)
    document = format_guide(github, guide, meta=meta)
    return await publish_artifact(
        context,
        _build_publish_request(guide, work_dir, document, slug),
        create_pr=request.create_pr,
    )


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_search_provider() -> WebSearchProvider:
    """Tavily (LLM-oriented, returns cleaned content) > Brave > no web search."""
    if os.getenv(EnvVars.TAVILY_API_KEY.value):
        logger.info("Using Tavily Search for web research.")
        return TavilySearchProvider()
    if os.getenv(EnvVars.BRAVE_API_KEY.value):
        logger.info("Using Brave Search for web research.")
        return BraveSearchProvider()
    logger.info("No search API key set; researching the supplied URLs only.")
    return NullSearchProvider()


def _build_publish_request(
    guide: TechGuide, work_dir: Path, document: str, slug: str
) -> PublishRequest:
    pr_body = build_pr_body(
        kind="technical guide",
        title=guide.topic,
        fields={
            "Sources": f"{len(guide.source_urls)} page(s)",
            "Characters": f"{len(document):,}",
        },
    )
    return PublishRequest(
        title=guide.topic,
        markdown=document,
        work_dir=work_dir,
        branch_id=f"guide-{slug}",
        pr_title=f"Tech Guide: {guide.topic}",
        pr_body=pr_body,
        commit_message=f"feat: add technical guide for '{guide.topic}'",
        kind="guide",
    )
