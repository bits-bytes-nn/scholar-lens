"""Job contract shared by the CLI, the AWS Batch worker, and agent callers.

A :class:`JobRequest` fully describes one artifact run. The CLI runs it locally
or serialises it into the single ``request`` parameter of the Batch job
definition (``worker --request Ref::request``), so the local and remote paths
parse exactly the same object and cannot drift. Every run ends in a
:class:`JobResult`, printed as JSON by the CLI and, for Batch runs, stored in S3
so ``status`` can report it after the container is gone.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scholar_lens.domain.constants import S3Paths
from scholar_lens.domain.survey.plan import parse_paper_id
from scholar_lens.domain.usage import RunUsage

ArtifactKind = Literal["review", "summarize", "guide", "survey_plan", "survey_write"]


JobStatus = Literal["succeeded", "failed", "rejected"]


# Batch job states that will not change any more.
TERMINAL_BATCH_STATES = frozenset({"SUCCEEDED", "FAILED"})


def _check_urls(what: str, urls: list[str]) -> None:
    for url in urls:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"{what} must be an http(s) URL: {url!r}")


class JobRequest(BaseModel):
    """One artifact run: what to read and whether to open a blog PR."""

    model_config = ConfigDict(extra="forbid")

    kind: ArtifactKind
    # review / summarize: an arXiv id or a paper PDF URL.
    source: str | None = None
    repo_urls: list[str] = Field(default_factory=list)
    parse_pdf: bool = False
    # guide: documentation URLs (the authority for commands, flags and
    # defaults), reference material to build on (papers, design docs, deep-dive
    # posts; never official) and optional extra web-search queries.
    # survey_plan: reference material too (blog posts, survey articles, paper
    # pages): read for the scope, the papers they link become candidates.
    urls: list[str] = Field(default_factory=list)
    reference_urls: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    discover_subpages: bool = True
    # guide: gather the sources again instead of reusing a saved corpus.
    fresh_sources: bool = False
    # Compute every stage again: ignore the saved stage checkpoints and the
    # reading cache (both are still written), and gather a guide's sources
    # again (implies ``fresh_sources``). Without it a rerun with the same
    # inputs resumes from the stages it saved.
    fresh: bool = False
    # survey_plan: the topic, papers the survey must include (arXiv ids or
    # URLs, DOIs), an optional year range and ``reference_urls``.
    topic: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    since_year: int | None = Field(default=None, ge=1900, le=2100)
    until_year: int | None = Field(default=None, ge=1900, le=2100)
    # survey_write: the plan file, a local path or s3://<bucket>/<key> (a
    # Batch run reads it from S3; the CLI uploads a local plan first).
    plan: str | None = None
    # Opens a blog PR when the GitHub integration is enabled in config.
    create_pr: bool = True

    @model_validator(mode="after")
    def _check_inputs(self) -> JobRequest:
        survey_fields = bool(
            self.topic
            or self.paper_ids
            or self.since_year
            or self.until_year
            or self.plan
        )
        if self.kind in ("survey_plan", "survey_write") and (
            self.parse_pdf or not self.discover_subpages or self.fresh_sources
        ):
            raise ValueError(
                f"A {self.kind} request takes no 'parse_pdf', 'discover_subpages' "
                "or 'fresh_sources'."
            )
        if self.kind == "survey_plan" and self.fresh:
            raise ValueError("A survey_plan request takes no 'fresh'.")
        if self.kind == "survey_write":
            if not (self.plan and self.plan.strip()):
                raise ValueError("A survey_write request needs a 'plan'.")
            if (
                self.source
                or self.repo_urls
                or self.urls
                or self.reference_urls
                or self.search_queries
                or self.topic
                or self.paper_ids
                or self.since_year
                or self.until_year
            ):
                raise ValueError(
                    "A survey_write request takes 'plan' and 'create_pr' only."
                )
            return self
        if self.kind == "survey_plan":
            if not (self.topic and self.topic.strip()):
                raise ValueError("A survey_plan request needs a 'topic'.")
            if (
                self.source
                or self.repo_urls
                or self.urls
                or self.search_queries
                or self.plan
            ):
                raise ValueError(
                    "A survey_plan request takes 'topic', 'paper_ids', "
                    "'reference_urls', 'since_year' and 'until_year'."
                )
            for paper_id in self.paper_ids:
                parse_paper_id(paper_id)  # SurveyInputError (a ValueError)
            _check_urls("A reference", self.reference_urls)
            if (
                self.since_year
                and self.until_year
                and self.since_year > self.until_year
            ):
                raise ValueError("'since_year' must not be after 'until_year'.")
            return self
        if survey_fields:
            raise ValueError(f"A {self.kind} request takes no survey fields.")
        if self.kind == "guide":
            if not self.urls and not self.reference_urls:
                raise ValueError(
                    "A guide request needs at least one URL (documentation "
                    "or reference material)."
                )
            if self.source or self.repo_urls:
                raise ValueError("A guide request takes 'urls', not 'source'.")
            _check_urls("A documentation page", self.urls)
            _check_urls("A reference", self.reference_urls)
        else:
            if not (self.source and self.source.strip()):
                raise ValueError(f"A {self.kind} request needs a 'source'.")
            if self.urls or self.reference_urls or self.search_queries:
                raise ValueError(f"A {self.kind} request takes 'source', not 'urls'.")
            if self.fresh_sources:
                raise ValueError(f"A {self.kind} request takes no 'fresh_sources'.")
            # Checked before any paid stage: the paper record needs URLs.
            _check_urls("A repository", self.repo_urls)
        return self

    @property
    def fresh_gathering(self) -> bool:
        """Whether a guide gathers its sources again (``fresh`` implies
        ``fresh_sources``)."""
        return self.fresh or self.fresh_sources

    def describe(self) -> str:
        """Short human label of the input, for logs and notifications."""
        return (
            self.topic
            or self.plan
            or self.source
            or ", ".join([*self.urls, *self.reference_urls])
        )


class JobResult(BaseModel):
    """Outcome of one run. ``warnings`` lists non-fatal problems (review
    notes, rule slips). A blog PR that was requested but could not be opened
    fails the run: ``pr_error`` says why, and ``markdown_path``/``s3_uri``
    still point to the saved artifact."""

    # None only for a Batch request too malformed to name its kind.
    kind: ArtifactKind | None
    status: JobStatus = "failed"
    input: str
    title: str | None = None
    markdown_path: str | None = None
    # survey_plan: the plan file of a local run (``s3_uri`` is its S3 copy,
    # the only one a Batch run leaves).
    plan_path: str | None = None
    s3_uri: str | None = None
    pr_url: str | None = None
    # Set only on a failed run whose artifact was saved but whose PR failed.
    pr_error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    # Blog-convention violations that blocked publishing (see post_validator).
    violations: list[str] = Field(default_factory=list)
    job_id: str | None = None
    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    # Calls, tokens (cache included) and cost by stage and model; None in
    # manifests written before it existed.
    usage: RunUsage | None = None
    # A failure notification was due but SNS rejected it; the worker then
    # exits with a code the stack's EventBridge rule reports instead.
    notification_failed: bool = Field(default=False, exclude=True)


class SubmittedJob(BaseModel):
    job_id: str
    job_name: str
    kind: ArtifactKind
    input: str


class JobStatusReport(BaseModel):
    """Batch state of a remote job plus its result once it has finished."""

    job_id: str
    job_name: str | None = None
    batch_status: str
    status_reason: str | None = None
    created_at: str | None = None
    stopped_at: str | None = None
    result: JobResult | None = None


def ssm_base(project_name: str, stage: str) -> str:
    """Prefix of every SSM parameter of one deployment."""
    return f"/{project_name}/{stage}"


def result_manifest_key(s3_prefix: str, job_id: str) -> str:
    return f"{s3_prefix}/{S3Paths.RESULTS.value}/{job_id}.json".lstrip("/")
