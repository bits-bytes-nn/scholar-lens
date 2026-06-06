"""Maps a parsed Slack intent to an AWS Batch job submission.

This is the "autonomous agent" surface the user drives from Slack: it turns a
classified request into a containerised job (review / summarize / guide) that
runs the existing pipeline and uploads to S3 + opens a blog PR. The dispatcher
itself only submits jobs — it performs no LLM work.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import boto3

from ..src.aws_helpers import submit_batch_job
from ..src.constants import AppConstants
from ..src.logger import logger
from .intent import ParsedIntent, SlackIntent


@dataclass
class SlackContext:
    """Where to post the result back to (the originating message)."""

    channel: str
    thread_ts: str | None = None
    user: str | None = None


@dataclass
class DispatchResult:
    job_id: str
    job_name: str
    intent: SlackIntent


class JobDispatcher:
    """Submits Batch jobs for parsed Slack intents."""

    def __init__(
        self,
        boto_session: boto3.Session,
        *,
        project_name: str,
        stage: str,
        review_job_queue: str,
        review_job_definition: str,
        guide_job_queue: str | None = None,
        guide_job_definition: str | None = None,
    ) -> None:
        self.boto_session = boto_session
        self.project_name = project_name
        self.stage = stage
        self.review_job_queue = review_job_queue
        self.review_job_definition = review_job_definition
        # Guides run a DIFFERENT container entrypoint (tech_guide_main), so they
        # require their own job definition — we must NOT silently fall back to
        # the paper-review definition (that would run the wrong program). They
        # may share the review queue (same compute environment).
        self.guide_job_queue = guide_job_queue or review_job_queue
        self.guide_job_definition = guide_job_definition

    def dispatch(
        self,
        parsed: ParsedIntent,
        *,
        timestamp: str,
        slack_context: SlackContext | None = None,
    ) -> DispatchResult:
        if not parsed.is_actionable:
            raise ValueError(
                f"Intent '{parsed.intent.value}' is not actionable: {parsed.reason}"
            )
        if parsed.intent in (SlackIntent.REVIEW, SlackIntent.SUMMARIZE):
            return self._dispatch_paper(
                parsed, timestamp=timestamp, slack_context=slack_context
            )
        if parsed.intent is SlackIntent.GUIDE:
            return self._dispatch_guide(
                parsed, timestamp=timestamp, slack_context=slack_context
            )
        raise ValueError(f"Unsupported intent: {parsed.intent.value}")

    @staticmethod
    def _slack_params(slack_context: SlackContext | None) -> dict[str, str]:
        if slack_context is None:
            return {
                "slack_channel": AppConstants.NULL_STRING,
                "slack_thread_ts": AppConstants.NULL_STRING,
            }
        return {
            "slack_channel": slack_context.channel,
            "slack_thread_ts": slack_context.thread_ts or AppConstants.NULL_STRING,
        }

    def _dispatch_paper(
        self,
        parsed: ParsedIntent,
        *,
        timestamp: str,
        slack_context: SlackContext | None = None,
    ) -> DispatchResult:
        job_name = self._job_name(parsed.intent.value, timestamp)
        repo_urls = (
            " ".join(parsed.repo_urls) if parsed.repo_urls else AppConstants.NULL_STRING
        )
        parameters = {
            "source": parsed.sources[0],
            "repo_urls": repo_urls,
            "parse_pdf": "true" if parsed.parse_pdf else "false",
            "mode": parsed.intent.value,
            **self._slack_params(slack_context),
        }
        job_id = submit_batch_job(
            self.boto_session,
            job_name,
            self.review_job_queue,
            self.review_job_definition,
            parameters=parameters,
        )
        logger.info(
            "Dispatched %s job '%s' (id=%s)", parsed.intent.value, job_name, job_id
        )
        return DispatchResult(job_id=job_id, job_name=job_name, intent=parsed.intent)

    def _dispatch_guide(
        self,
        parsed: ParsedIntent,
        *,
        timestamp: str,
        slack_context: SlackContext | None = None,
    ) -> DispatchResult:
        if not self.guide_job_definition:
            raise ValueError(
                "No technical-guide job definition is configured. Deploy the "
                "guide Batch job definition (and its SSM parameter) to enable "
                "'guide' requests."
            )
        job_name = self._job_name("guide", timestamp)
        parameters = {
            "urls": " ".join(parsed.sources),
            "discover_subpages": "true",
            "search_queries": AppConstants.NULL_STRING,
            **self._slack_params(slack_context),
        }
        job_id = submit_batch_job(
            self.boto_session,
            job_name,
            self.guide_job_queue,
            self.guide_job_definition,
            parameters=parameters,
        )
        logger.info("Dispatched guide job '%s' (id=%s)", job_name, job_id)
        return DispatchResult(job_id=job_id, job_name=job_name, intent=parsed.intent)

    def _job_name(self, action: str, timestamp: str) -> str:
        return f"{self.project_name}-{self.stage}-{action}-{timestamp}"


def utc_timestamp() -> str:
    """Compact UTC timestamp for unique Batch job names."""
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S")
