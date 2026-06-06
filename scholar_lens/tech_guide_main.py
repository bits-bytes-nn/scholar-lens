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

import boto3
from pydantic import BaseModel, ConfigDict

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
    Publisher,
    PublishRequest,
    S3Handler,
    SSMParams,
    TechGuide,
    TechGuideGenerator,
    WebResearcher,
    WebSearchProvider,
    arg_as_bool,
    get_ssm_param_value,
    is_running_in_aws,
    logger,
)

ROOT_DIR: Path = Path("/tmp") if is_running_in_aws() else Path(__file__).parent.parent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class GuideContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: Config
    default_boto_session: boto3.Session
    bedrock_boto_session: boto3.Session
    s3_handler: S3Handler | None = None


def main(
    urls: list[str],
    *,
    discover_subpages: bool = True,
    search_queries: list[str] | None = None,
    slack_channel: str | None = None,
    slack_thread_ts: str | None = None,
) -> str | None:
    config = Config.load()
    profile_name = (
        os.getenv(EnvVars.AWS_PROFILE_NAME.value)
        if is_running_in_aws()
        else config.resources.profile_name
    )
    context = GuideContext(
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
    error_message: str | None = None
    topic = ", ".join(urls)
    try:
        s3_url = asyncio.run(
            _run(
                context,
                urls,
                discover_subpages=discover_subpages,
                search_queries=search_queries,
            )
        )
        success = True
        return s3_url
    except Exception as e:
        error_message = str(e)
        logger.error("Failed to generate tech guide: %s", e, exc_info=True)
        raise
    finally:
        topic_arn = os.getenv(EnvVars.TOPIC_ARN.value)
        if is_running_in_aws() and topic_arn:
            _send_guide_sns_notification(
                context.default_boto_session, topic_arn, success, topic, s3_url
            )
        post_slack_result(
            channel=slack_channel,
            thread_ts=slack_thread_ts,
            success=success,
            artifact_label="guide",
            title=topic,
            s3_url=s3_url,
            error=error_message,
        )


def _send_guide_sns_notification(
    session: boto3.Session,
    topic_arn: str,
    success: bool,
    sources: str,
    s3_url: str | None,
) -> None:
    status = "succeeded" if success else "failed"
    lines = [f"Technical guide {status} for: {sources}"]
    if s3_url:
        lines.append(f"Output: {s3_url}")
    try:
        session.client("sns").publish(
            TopicArn=topic_arn,
            Subject=f"Tech Guide {status.title()}",
            Message="\n".join(lines),
        )
    except Exception as e:  # noqa: BLE001 - notification failure must not fail the job
        logger.error("Failed to send SNS notification: %s", e)


async def _run(
    context: GuideContext,
    urls: list[str],
    *,
    discover_subpages: bool,
    search_queries: list[str] | None,
) -> str | None:
    _setup_aws_env(context)
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

    work_dir = ROOT_DIR / LocalPaths.PAPERS_DIR.value / _guide_slug(guide.topic)
    work_dir.mkdir(parents=True, exist_ok=True)

    document = _format_guide(context.config.resources.github, guide)
    publisher = Publisher(
        context.config.resources.github,
        root_dir=ROOT_DIR,
        s3_handler=context.s3_handler,
        s3_bucket_name=context.config.resources.s3_bucket_name,
        s3_prefix=context.config.resources.s3_prefix,
    )
    request = _build_request(guide, work_dir, document)
    s3_url, document_path = await publisher.publish(request)
    if context.config.resources.github.enabled:
        await publisher.create_pull_request(request, document_path)
    return s3_url


def _setup_aws_env(context: GuideContext) -> None:
    """Load GitHub/Brave secrets from SSM into the environment (AWS only)."""
    if not is_running_in_aws():
        return
    base = f"/{context.config.resources.project_name}/{context.config.resources.stage}"
    ssm_map = {
        SSMParams.GITHUB_TOKEN: EnvVars.GITHUB_TOKEN,
        SSMParams.BRAVE_API_KEY: EnvVars.BRAVE_API_KEY,
        SSMParams.SLACK_BOT_TOKEN: EnvVars.SLACK_BOT_TOKEN,
    }
    for ssm_param, env_var in ssm_map.items():
        try:
            os.environ[env_var.value] = get_ssm_param_value(
                context.default_boto_session, f"{base}/{ssm_param.value}"
            )
            logger.info("Set env var '%s' from SSM.", env_var.value)
        except Exception as e:  # noqa: BLE001 - optional secret
            logger.info("Could not set '%s' from SSM: %s", env_var.value, e)


def _build_search_provider() -> WebSearchProvider:
    if os.getenv(EnvVars.BRAVE_API_KEY.value):
        logger.info("Using Brave Search for web research.")
        return BraveSearchProvider()
    logger.info("No Brave API key set; researching supplied URLs only.")
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
    front_matter = front_matter_template.format(
        title=guide.topic.replace('"', '\\"'),
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        author=github_config.author_name,
        categories='"Tech Guides"',
        tags="",
        cover_image=cover_image,
    )
    sources = "\n".join(f"* <{url}>" for url in guide.source_urls)
    references = f"\n- - -\n### Sources\n{sources}\n"
    return f"{front_matter}{guide.body}{references}"


def _build_request(guide: TechGuide, work_dir: Path, document: str) -> PublishRequest:
    pr_body = (
        f"This PR adds an AI-generated technical guide: **{guide.topic}**\n\n"
        f"- **Sources**: {len(guide.source_urls)} page(s)\n"
        f"- **Number of Characters**: {len(document):,}\n\n"
        f"This pull request was automatically generated by the Scholar-Lens system."
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
