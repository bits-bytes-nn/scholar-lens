"""Technical-guide pipeline: research the URLs, write the guide, publish it."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

from .constants import EnvVars, LanguageModelId, LocalPaths
from .formatting import build_post_meta, cut_title, format_guide
from .jobs import JobRequest
from .logger import logger
from .markdown_lint import (
    drop_unknown_images,
    figure_markers_to_images,
    fill_image_alts,
)
from .metrics import TokenUsageTracker
from .publish_types import slugify
from .publisher import PublishRequest, build_pr_body
from .runtime import (
    ROOT_DIR,
    PipelineOutcome,
    RunContext,
    load_translation_guideline,
    publish_artifact,
)
from .tech_guide import TechGuide, TechGuideGenerator
from .web_research import (
    BraveSearchProvider,
    NullSearchProvider,
    TavilySearchProvider,
    WebResearcher,
    WebSearchProvider,
    normalize_url,
)


async def run_guide_pipeline(
    context: RunContext, request: JobRequest, tracker: TokenUsageTracker
) -> PipelineOutcome:
    """Generate and publish a guide; raises ``NotTechnicalContentError`` when
    the URLs are not technical documentation."""
    sources = [*request.urls, *request.reference_urls]
    cfg = context.config.tech_guide
    translation_guideline = await load_translation_guideline(context)
    researcher = WebResearcher(search_provider=build_search_provider())
    spec = context.require_conventions()
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
        max_sections=cfg.max_sections,
        title_max_length=spec.seo_title_max_length,
        verify_grounding=cfg.verify_grounding,
        auto_research=cfg.auto_research,
        max_research_queries=cfg.max_research_queries,
        fetch_top_results=cfg.fetch_top_results,
        min_quality_score=cfg.min_quality_score,
        max_revision_attempts=cfg.max_revision_attempts,
        max_total_tokens=cfg.max_total_tokens,
        translation_guideline=translation_guideline,
        callbacks=[tracker],
    )
    github = context.config.resources.github
    try:
        guide = await generator.generate(
            request.urls,
            reference_urls=request.reference_urls,
            discover_subpages=request.discover_subpages,
            search_queries=request.search_queries or None,
        )
        taxonomy = await generator.classify(
            guide,
            category_options=list(spec.topics),
            controlled_tags=spec.tags.controlled,
        )
    except Exception as e:
        # Keep the generator's notes (e.g. the budget note) on a failed run.
        e.run_notes = list(generator.notes)  # type: ignore[attr-defined]
        raise
    finally:
        researcher.close()
    guide.topic = guide_title(
        guide.topic, taxonomy.get("seo_title") or "", spec.seo_title_max_length
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
    document = guide_document(format_guide(github, guide, meta=meta), guide)
    publish_request = _build_publish_request(
        guide, work_dir, document, branch_id=guide_branch_id(sources)
    )
    publish_request.review_notes += guide.notes
    outcome = await publish_artifact(
        context, publish_request, create_pr=request.create_pr
    )
    return outcome


def guide_title(title: str, seo_title: str, limit: int) -> str:
    """``title`` when it fits the blog's ``seo_title`` limit, else the
    taxonomy's ``seo_title`` (when it fits) or the title cut at a word
    boundary: a guide title over the limit is a list of sections, not a name,
    and the part before its colon ("vLLM") names the tool, not the guide."""
    if len(title) <= limit:
        return title
    seo_title = " ".join(seo_title.split())
    if seo_title and len(seo_title) <= limit:
        return seo_title
    return cut_title(title, limit)


def guide_document(document: str, guide: TechGuide) -> str:
    """Copied ``[Image: …]`` markers become images, images the writer was
    never offered (a logo, a tracking pixel, an invented URL) are dropped, and
    every image gets alt text: the paper pipeline's order."""
    document = figure_markers_to_images(document)
    document = drop_unknown_images(document, set(guide.image_urls))
    return fill_image_alts(document, {}, guide.topic)


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


def guide_branch_id(urls: list[str]) -> str:
    """A guide's branch is keyed by its source URLs (documentation and
    reference material), not the model-chosen title, so rerunning the same
    request reuses the branch and pull request."""
    key = "\n".join(sorted({_source_key(url) for url in urls}))
    return f"guide-{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def _source_key(url: str) -> str:
    """:func:`normalize_url` plus the query, which can select the page
    (``view?id=intro``); the fragment never does."""
    query = urlparse(url).query
    return normalize_url(url) + (f"?{query}" if query else "")


def _build_publish_request(
    guide: TechGuide, work_dir: Path, document: str, *, branch_id: str
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
        branch_id=branch_id,
        pr_title=f"Tech Guide: {guide.topic}",
        pr_body=pr_body,
        commit_message=f"feat: add technical guide for '{guide.topic}'",
        kind="guide",
    )
