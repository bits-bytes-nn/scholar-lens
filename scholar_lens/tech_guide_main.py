"""CLI entrypoint for the technical-guide / tutorial generator.

Takes a list of source URLs (library/framework/platform docs), researches them
(with optional sub-page discovery and web search), generates a self-study guide,
and publishes it to S3 + a blog PR via the shared :class:`Publisher`. Refuses to
write when the URLs are not technical documentation.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scholar_lens.configs import Config
from scholar_lens.configs import Github as GithubConfig
from scholar_lens.slack.notifier import post_slack_result
from scholar_lens.src import (
    AppConstants,
    BraveSearchProvider,
    EnvVars,
    LanguageModelId,
    LocalPaths,
    NotTechnicalContentError,
    NullSearchProvider,
    PublishRequest,
    SSMParams,
    TavilySearchProvider,
    TechGuide,
    TechGuideGenerator,
    TokenUsageTracker,
    WebResearcher,
    WebSearchProvider,
    arg_as_bool,
    build_pr_body,
    is_running_in_aws,
    logger,
)
from scholar_lens.src.runtime import (
    RunContext,
    build_context,
    build_publisher,
    load_secrets_from_ssm,
    publish_sns,
)

ROOT_DIR: Path = Path("/tmp") if is_running_in_aws() else Path(__file__).parent.parent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(
    urls: list[str],
    *,
    discover_subpages: bool = True,
    search_queries: list[str] | None = None,
    slack_channel: str | None = None,
    slack_thread_ts: str | None = None,
) -> str | None:
    context = build_context(Config.load())

    s3_url: str | None = None
    pr_url: str | None = None
    success = False
    error_message: str | None = None
    sources = ", ".join(urls)
    # Falls back to the source URL(s) until the generated topic is known; the
    # pipeline overwrites this with the real guide title (e.g. "Getting Started
    # with Argo CD") so Slack/SNS show that rather than a raw URL.
    title = sources
    try:
        s3_url, pr_url, generated_topic = asyncio.run(
            _run(
                context,
                urls,
                discover_subpages=discover_subpages,
                search_queries=search_queries,
            )
        )
        if generated_topic:
            title = generated_topic
        success = True
        return s3_url
    except Exception as e:
        error_message = str(e)
        logger.error("Failed to generate tech guide: %s", e, exc_info=True)
        raise
    finally:
        topic_arn = os.getenv(EnvVars.TOPIC_ARN.value)
        if is_running_in_aws() and topic_arn:
            status = "succeeded" if success else "failed"
            lines = [f"Technical guide {status} for: {sources}"]
            if s3_url:
                lines.append(f"Output: {s3_url}")
            publish_sns(
                context.default_boto_session,
                topic_arn,
                subject=f"Tech Guide {status.title()}",
                lines=lines,
            )
        post_slack_result(
            channel=slack_channel,
            thread_ts=slack_thread_ts,
            success=success,
            artifact_label="guide",
            title=title,
            s3_url=s3_url,
            pr_url=pr_url,
            error=error_message,
            sources=sources,
        )


async def _run(
    context: RunContext,
    urls: list[str],
    *,
    discover_subpages: bool,
    search_queries: list[str] | None,
) -> tuple[str | None, str | None, str | None]:
    _setup_aws_env(context)
    tracker = TokenUsageTracker()
    researcher = WebResearcher(search_provider=_build_search_provider())
    generator = TechGuideGenerator(
        relevance_model_id=LanguageModelId(
            context.config.tech_guide.relevance_model_id
        ),
        synopsis_model_id=LanguageModelId(context.config.tech_guide.synopsis_model_id),
        writing_model_id=LanguageModelId(context.config.tech_guide.writing_model_id),
        boto_session=context.bedrock_boto_session,
        researcher=researcher,
        language=context.config.output_language,
        enable_thinking=context.config.tech_guide.writer_enable_thinking,
        thinking_effort=context.config.tech_guide.thinking_effort,
        verify_grounding=context.config.tech_guide.verify_grounding,
        auto_research=context.config.tech_guide.auto_research,
        max_research_queries=context.config.tech_guide.max_research_queries,
        fetch_top_results=context.config.tech_guide.fetch_top_results,
        min_quality_score=context.config.tech_guide.min_quality_score,
        max_revision_attempts=context.config.tech_guide.max_revision_attempts,
        max_total_tokens=context.config.tech_guide.max_total_tokens,
        callbacks=[tracker],
    )

    try:
        guide = await generator.generate(
            urls,
            discover_subpages=discover_subpages,
            search_queries=search_queries,
        )
    finally:
        # Release the researcher's HTTP connection pool regardless of outcome.
        researcher.close()
        logger.info(
            "Tech-guide token usage: %d in + %d out = %d total (~$%.2f), %d calls.",
            tracker.input_tokens,
            tracker.output_tokens,
            tracker.total_tokens,
            tracker.estimated_cost_usd(),
            tracker.call_count,
        )

    work_dir = ROOT_DIR / LocalPaths.PAPERS_DIR.value / _guide_slug(guide.topic)
    work_dir.mkdir(parents=True, exist_ok=True)

    document = _format_guide(context.config.resources.github, guide)
    publisher = build_publisher(context, ROOT_DIR)
    request = _build_request(guide, work_dir, document)
    s3_url, document_path = await publisher.publish(request)
    pr_url: str | None = None
    if context.config.resources.github.enabled:
        pr_url = await publisher.create_pull_request(request, document_path)
    return s3_url, pr_url, guide.topic


def _setup_aws_env(context: RunContext) -> None:
    """Load GitHub/search/Slack secrets from SSM into the environment (AWS only)."""
    load_secrets_from_ssm(
        context,
        {
            SSMParams.GITHUB_TOKEN: EnvVars.GITHUB_TOKEN,
            SSMParams.TAVILY_API_KEY: EnvVars.TAVILY_API_KEY,
            SSMParams.BRAVE_API_KEY: EnvVars.BRAVE_API_KEY,
            SSMParams.SLACK_BOT_TOKEN: EnvVars.SLACK_BOT_TOKEN,
        },
    )


def _build_search_provider() -> WebSearchProvider:
    # Prefer Tavily: it's built for LLM retrieval and returns cleaned page
    # content directly. Fall back to Brave, then to no web search.
    if os.getenv(EnvVars.TAVILY_API_KEY.value):
        logger.info("Using Tavily Search for web research.")
        return TavilySearchProvider()
    if os.getenv(EnvVars.BRAVE_API_KEY.value):
        logger.info("Using Brave Search for web research.")
        return BraveSearchProvider()
    logger.info("No search API key set; researching supplied URLs only.")
    return NullSearchProvider()


def _guide_slug(topic: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:80] or "tech-guide"


def _format_guide(github_config: GithubConfig, guide: TechGuide) -> str:
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
    cover_image = github_config.cover_image_for("tech-guides")
    category = github_config.tech_guide_category.replace('"', '\\"')
    front_matter = front_matter_template.format(
        title=guide.topic.replace('"', '\\"'),
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        author=github_config.author_name,
        categories=f'"{category}"',
        tags="",
        cover_image=cover_image,
    )
    sources = "\n".join(f"* <{url}>" for url in guide.source_urls)
    references = f"\n- - -\n### Sources\n{sources}\n"
    return f"{front_matter}{guide.body}{references}"


def _build_request(guide: TechGuide, work_dir: Path, document: str) -> PublishRequest:
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
        branch_id=_guide_slug(guide.topic),
        pr_title=f"Tech Guide: {guide.topic}",
        pr_body=pr_body,
        commit_message=f"feat: Add technical guide for '{guide.topic}'",
        rewrite_local_images=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scholar Lens: technical guide / tutorial generator."
    )
    parser.add_argument(
        "--urls",
        type=str,
        nargs="+",
        required=True,
        help="One or more documentation URLs to base the guide on.",
    )
    parser.add_argument(
        "--discover-subpages",
        type=arg_as_bool,
        default=True,
        help="Follow in-scope sub-pages of the supplied URLs.",
    )
    parser.add_argument(
        "--search-queries",
        type=str,
        nargs="*",
        help="Optional web-search queries to augment the research corpus.",
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

    # AWS Batch substitutes Ref:: parameters as a single space-joined token,
    # while an interactive CLI passes multiple args; flatten both forms.
    def _flatten(values: list[str] | None) -> list[str]:
        items: list[str] = []
        for value in values or []:
            items.extend(value.split())
        return [v for v in items if v and v != AppConstants.NULL_STRING]

    def _clean(value: str | None) -> str | None:
        if not value or value == AppConstants.NULL_STRING:
            return None
        return value

    urls = _flatten(args.urls)
    search_queries = _flatten(args.search_queries) or None
    logger.info("Starting tech-guide generation for %d URL(s).", len(urls))
    try:
        main(
            urls,
            discover_subpages=args.discover_subpages,
            search_queries=search_queries,
            slack_channel=_clean(args.slack_channel),
            slack_thread_ts=_clean(args.slack_thread_ts),
        )
    except NotTechnicalContentError as e:
        logger.error("Refusing to generate a guide: %s", e)
        sys.exit(2)
