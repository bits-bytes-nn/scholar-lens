"""``scholar-lens survey plan``: references → scope → candidates →
selection → plan file.

``survey plan`` writes an editable plan (``papers/survey-<slug>/plan.json``
and its S3 copy); the write stage (``survey write --plan``) reads it. The
user's reference material (``--reference``) is read once: excerpts go to
the scope call and the papers each reference is or links to join the
candidates. The scope and selection calls are the only model calls; the
candidate step is deterministic API work and every check on the models'
answers is code (``domain.survey.plan``).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

from scholar_lens.application.ports import (
    CandidateSearch,
    ObjectStore,
    SurveyPlanning,
    SurveyPlanTools,
    UsageTracker,
)
from scholar_lens.application.run import PipelineOutcome, RunContext
from scholar_lens.domain.constants import S3Paths
from scholar_lens.domain.jobs import JobRequest
from scholar_lens.domain.survey.plan import (
    SurveyPlan,
    TimeRange,
    build_plan,
    merge_records,
    shortlist,
    survey_slug,
)
from scholar_lens.domain.survey.references import unread_note

logger = logging.getLogger(__name__)


async def plan_survey(
    request: JobRequest,
    *,
    planner: SurveyPlanning,
    finder: CandidateSearch,
    min_papers: int,
    max_papers: int,
    candidate_pool: int,
    now: datetime,
) -> SurveyPlan:
    """Read the references, scope the topic, gather and rank candidates,
    select the papers and build the plan."""
    topic = (request.topic or "").strip()
    user_range = TimeRange(since=request.since_year, until=request.until_year)
    required_records = await asyncio.to_thread(
        finder.resolve_required, request.paper_ids
    )
    required = merge_records(required_records)
    references = await asyncio.to_thread(finder.read_references, request.reference_urls)
    referenced = await asyncio.to_thread(finder.referenced_records, references)
    scope = await planner.scope(topic, user_range, required, references)
    pool = await asyncio.to_thread(
        finder.gather,
        scope,
        [*required_records, *referenced],
        time_range=user_range,
        today=now.date(),
    )
    picks = shortlist(pool.candidates, candidate_pool)
    logger.info(
        "Survey pool: %d candidates, %d citation edges; shortlist of %d.",
        len(pool.candidates),
        len(pool.edges),
        len(picks),
    )
    selection = await planner.select(
        topic,
        scope,
        pool,
        picks,
        min_papers=min_papers,
        max_papers=max_papers,
        today=now.date(),
    )
    return build_plan(
        topic=topic,
        scope=scope,
        selection=selection,
        pool=picks,
        edges=pool.edges,
        time_range=user_range,
        references=[r.plan_reference() for r in references],
        sources=pool.sources,
        notes=[*pool.notes, *unread_note(references)],
        created_at=now,
    )


def save_plan(plan: SurveyPlan, path: Path) -> None:
    """Write ``plan`` to ``path``; a different plan already there (perhaps
    edited by hand) is kept next to it as ``plan.<UTC timestamp>.json``."""
    text = plan.to_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        path.rename(path.with_name(f"{path.stem}.{stamp}{path.suffix}"))
    path.write_text(text, encoding="utf-8")


def upload_plan(plan: SurveyPlan, s3_handler: ObjectStore, key: str) -> None:
    """Upload ``plan`` to ``key``; a different plan already there (perhaps
    edited and staged by hand) is kept next to it as
    ``plan-<sha256 prefix>.json``, the name ``survey write --remote`` gives an
    uploaded plan, so the same content never takes two keys."""
    text = plan.to_json()
    old = s3_handler.get_text(key)
    if old is not None and old != text:
        digest = hashlib.sha256(old.encode()).hexdigest()[:16]
        s3_handler.put_text(key.rsplit("/", 1)[0] + f"/plan-{digest}.json", old)
    s3_handler.put_text(key, text)


async def run_survey_plan(
    context: RunContext,
    request: JobRequest,
    tracker: UsageTracker,
    tools: SurveyPlanTools,
) -> PipelineOutcome:
    limits = tools.limits
    planner = tools.planner(tracker)
    finder = tools.finder()
    try:
        plan = await plan_survey(
            request,
            planner=planner,
            finder=finder,
            min_papers=limits.min_papers,
            max_papers=limits.max_papers,
            candidate_pool=limits.candidate_pool,
            now=datetime.now(UTC),
        )
    finally:
        finder.close()
    slug = survey_slug(plan.topic)
    path = context.papers_dir / slug / "plan.json"
    await asyncio.to_thread(save_plan, plan, path)
    s3_uri = None
    if context.s3_handler is not None:
        resources = context.config.resources
        key = f"{resources.s3_prefix}/{S3Paths.SURVEYS.value}/{slug}/plan.json".lstrip(
            "/"
        )
        await asyncio.to_thread(upload_plan, plan, context.s3_handler, key)
        s3_uri = f"s3://{resources.s3_bucket_name}/{key}"
    logger.info(
        "Survey plan: %d papers in %d lineages -> %s",
        len(plan.papers),
        len(plan.lineages),
        path,
    )
    return PipelineOutcome(
        title=plan.scope,
        plan_path=path,
        s3_uri=s3_uri,
        warnings=list(plan.notes),
    )
