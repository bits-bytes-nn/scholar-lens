"""``scholar-lens survey write``: plan file → evidence engine → post.

``survey write --plan`` reads the plan (local or S3) the plan stage wrote and
the user may have edited, resolves papers added by hand (DataCite for arXiv
ids, OpenAlex for DOIs), reads the plan's reference articles again
(explanatory material, listed as further reading), writes the post with the
evidence engine (:func:`write_survey_with_evidence`) and publishes it like
the other kinds. Every check on the models' answers is code
(``domain.survey.post``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

from scholar_lens.application.ports import (
    FileLocationError,
    Files,
    ObjectStore,
    PlanIndex,
    ReferenceReader,
    SurveyWriteTools,
    UsageTracker,
)
from scholar_lens.application.run import (
    PipelineOutcome,
    RunContext,
    check_post_type,
    find_published_post,
    load_translation_guideline,
    publish_artifact,
)
from scholar_lens.application.survey_engine import EngineSettings, EvidenceSurveyEngine
from scholar_lens.domain.constants import S3Paths
from scholar_lens.domain.ids import paper_url
from scholar_lens.domain.jobs import JobRequest
from scholar_lens.domain.markdown.lint import drop_unknown_images, fill_image_alts
from scholar_lens.domain.markdown.math import has_math, normalize_math
from scholar_lens.domain.post_format import build_post_meta, render_front_matter
from scholar_lens.domain.post_names import title_slug
from scholar_lens.domain.publishing import PublishRequest, build_pr_body
from scholar_lens.domain.survey.plan import (
    SurveyInputError,
    SurveyPlan,
    survey_slug,
)
from scholar_lens.domain.survey.post import (
    credit_figures,
    fill_metadata,
    further_reading,
    needs_metadata,
    normalize_plan_data,
    plan_data,
    plan_link_sources,
    reference_list,
    render_citations,
    validate_plan,
)
from scholar_lens.domain.survey.references import reference_excerpts

logger = logging.getLogger(__name__)


def read_plan_text(location: str, files: Files) -> str:
    """The plan file's text from a local path or ``s3://<bucket>/<key>`` in
    the run's bucket; ``SurveyInputError`` when it is not there."""
    try:
        text = files.read_text(location)
    except FileLocationError as e:
        raise SurveyInputError(f"The plan {location!r}: {e}") from e
    except UnicodeDecodeError as e:
        raise SurveyInputError(f"The plan {location!r} is not UTF-8: {e}") from e
    if text is None:
        raise SurveyInputError(f"No plan at {location!r}.")
    return text


def load_plan(text: str, index: PlanIndex) -> tuple[SurveyPlan, list[str]]:
    """Validate an edited plan again: ids normalized, stale edges and empty
    lineages dropped, papers added by hand completed from the indexes
    (``SurveyInputError`` for one no index knows), then the plan schema."""
    data = plan_data(text)
    notes = normalize_plan_data(data)
    added = [p for p in data["papers"] if isinstance(p, dict) and needs_metadata(p)]
    if added:
        records = index.records([p["id"] for p in added])
        for paper in added:
            if paper["id"] not in records:
                raise SurveyInputError(
                    f"No index knows the paper {paper['id']!r} added to the plan."
                )
            fill_metadata(paper, [records[paper["id"]]])
        notes.append(
            "Papers added by hand, completed from the indexes: "
            + ", ".join(p["id"] for p in added)
        )
    return validate_plan(data), notes


def stage_plan_for_batch(
    request: JobRequest,
    s3_handler: ObjectStore | None,
    s3_prefix: str,
    *,
    files: Files,
) -> JobRequest:
    """A survey_write request whose plan a Batch job can read: a local plan
    is checked against the plan schema (``SurveyInputError`` when it would
    fail the job), uploaded to ``<prefix>/surveys/<slug>/plan-<sha256>.json``
    (one key per content, so nothing is overwritten) and the request points
    at it."""
    location = request.plan or ""
    if location.startswith("s3://"):
        return request
    if s3_handler is None:
        raise SurveyInputError(
            "--remote needs resources.s3_bucket_name to upload the plan."
        )
    text = read_plan_text(location, files)
    data = plan_data(text)
    topic = str(data.get("topic") or "survey")
    check_plan_shape(data)
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    key = (
        f"{s3_prefix}/{S3Paths.SURVEYS.value}/{survey_slug(topic)}/plan-{digest}.json"
    ).lstrip("/")
    s3_handler.put_text(key, text)
    return request.model_copy(update={"plan": f"s3://{s3_handler.bucket_name}/{key}"})


def check_plan_shape(data: dict[str, Any]) -> None:
    """Validate an edited plan without the indexes (``SurveyInputError``
    when it is unusable): a paper added by hand gets its title and URL from
    the indexes in the job, so here its id stands in for them."""
    copy = json.loads(json.dumps(data))
    normalize_plan_data(copy)
    copy["papers"] = [
        {
            **p,
            "title": p.get("title") or p["id"],
            "url": p.get("url") or paper_url(p["id"]),
        }
        for p in copy["papers"]
    ]
    validate_plan(copy)


@dataclass
class Survey:
    """The written survey before it becomes a post."""

    title: str
    body: str
    plan: SurveyPlan
    # The reference articles that were read: (title, url) for further reading.
    reading: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Problems the post may not be published with as is (listed first).
    blocking: list[str] = field(default_factory=list)
    # Figures offered to the writers: url -> (paper id, caption).
    figures: dict[str, tuple[str, str]] = field(default_factory=dict)


async def write_survey_with_evidence(
    context: RunContext,
    tools: SurveyWriteTools,
    plan: SurveyPlan,
    *,
    index: PlanIndex,
    reader: ReferenceReader,
    translation_guideline: str,
    tracker: UsageTracker,
    title_max_length: int,
) -> Survey:
    """The evidence engine: reading agents, an evidence-assigning outline,
    verified sections (docs/redesign.md)."""
    writing = tools.writing
    model = tools.models.language_model(writing.roles, tracker, writing.max_cost_usd)
    records = await asyncio.to_thread(index.records, [p.id for p in plan.papers])
    documents = tools.documents(
        reader, index, {pid: r.abstract for pid, r in records.items() if r.abstract}
    )
    language = context.config.output_language
    store = tools.evidence_store(fresh=context.fresh)
    engine = EvidenceSurveyEngine(
        model,
        documents,
        EngineSettings(
            language=language,
            title_max_length=title_max_length,
            reader_tag=writing.reader_tag,
            chars_per_core=writing.chars_per_core,
            chars_per_supporting=writing.chars_per_supporting,
            min_total_chars=writing.min_total_chars,
            max_total_chars=writing.max_total_chars,
        ),
        store=store,
        checkpoints=store,
        roles=writing.roles,
    )
    articles = [r for r in plan.references if r.kind == "article"]
    read = await asyncio.to_thread(reader.read_references, [r.url for r in articles])
    pages = [r.page for r in read if r.page]
    titles = {r.url: r.title for r in articles}
    reading = [(titles.get(r.url) or r.page.title, r.url) for r in read if r.page]
    notes: list[str] = []
    if unread := [r.url for r in read if r.page is None]:
        notes.append(
            "Reference articles that could not be read (left out): " + ", ".join(unread)
        )
    result = await engine.write(
        plan,
        references=reference_excerpts(pages, plan.topic, 8_000) if pages else "",
        guideline=translation_guideline,
    )
    # Papers read from less than their full text (known once they are read).
    notes = [*documents.notes(), *notes]
    figures = {
        e.figure_url: (e.paper_id, e.caption or "")
        for e in result.ledger
        if e.kind == "figure" and e.figure_url and e.figure_url in result.body
    }
    # arXiv HTML lists some figures it does not serve (404): an image that
    # cannot be downloaded would fail the publish step, so it is dropped
    # here (survey_body removes images without an owner).
    broken = await tools.figures.unservable(list(figures))
    for url in broken:
        figures.pop(url)
    if broken:
        notes.append("Figures arXiv does not serve were left out: " + ", ".join(broken))
    return Survey(
        title=result.title,
        body=result.body,
        plan=result.plan,
        reading=reading,
        notes=[*notes, *result.notes],
        blocking=list(result.blocking),
        figures=figures,
    )


def survey_body(survey: Survey, language: str) -> tuple[str, list[str]]:
    """The body as published: citation keys rendered (unknown ones removed),
    images the writers were not offered dropped, figures credited to their
    paper, math normalized. Returns it with notes. Links to anything but the
    plan's papers and articles are unlinked when it is published
    (:func:`plan_link_sources`)."""
    papers = {p.id: p for p in survey.plan.papers}
    body, unknown = render_citations(survey.body, papers)
    owned = {u: f for u, f in survey.figures.items() if f[0] in papers}
    owners = {url: papers[paper_id] for url, (paper_id, _) in owned.items()}
    captions = {url: caption for url, (_, caption) in owned.items()}
    body = drop_unknown_images(body, set(owners))
    body = fill_image_alts(body, captions, survey.title, korean=language == "Korean")
    body = credit_figures(body, owners, language)
    notes = []
    if unknown:
        notes.append(
            "Citations of papers outside the survey removed: "
            + ", ".join(sorted(set(unknown)))
        )
    return normalize_math(body.strip()), notes


async def run_survey_write(
    context: RunContext,
    request: JobRequest,
    tracker: UsageTracker,
    tools: SurveyWriteTools,
) -> PipelineOutcome:
    github = context.config.resources.github
    spec = context.require_conventions()
    check_post_type(spec, github.survey_category, "survey_category", artifact="survey")
    text = await asyncio.to_thread(read_plan_text, request.plan or "", context.files)
    index = tools.index()
    plan, plan_notes = await asyncio.to_thread(load_plan, text, index)
    guideline = await load_translation_guideline(context)
    writer = tools.finalizer(tracker)
    reader = tools.reader()
    papers_dir = context.papers_dir
    try:
        survey = await write_survey_with_evidence(
            context,
            tools,
            plan,
            index=index,
            reader=reader,
            # The same JSON the review and guide writers get, not a repr.
            translation_guideline=(
                json.dumps(guideline, ensure_ascii=False) if guideline else ""
            ),
            tracker=tracker,
            title_max_length=spec.seo_title_max_length,
        )
        body, body_notes = survey_body(survey, writer.language)
        taxonomy = await writer.finalize(
            survey.title,
            body,
            category_options=list(spec.topics),
            controlled_tags=spec.tags.controlled,
        )
    except Exception as e:
        # Keep the plan's and the writer's notes on a failed run.
        e.run_notes = [*plan_notes, *writer.notes]  # type: ignore[attr-defined]
        raise
    finally:
        reader.close()
    # Keyed by the plan's topic, not the model's title: a rerun of the same
    # plan updates the same branch, pull request and published post.
    slug = survey_slug(plan.topic)
    published, lookup_notes = await find_published_post(
        context,
        slug=title_slug(survey.title, "survey"),
        branch_id=slug,
        category=github.survey_category,
    )
    meta = build_post_meta(
        title=survey.title,
        category_type=github.survey_category,
        proposed_topic=taxonomy.get("category", ""),
        specific_tags=_split(taxonomy.get("keywords", "")),
        controlled_tags=_split(taxonomy.get("controlled_tags", "")),
        proposed_seo_title=None,
        spec=spec,
        known_tags=context.require_blog_tags(),
        published=published,
    )
    description = " ".join((taxonomy.get("description") or "").split())
    front_matter = render_front_matter(
        title=survey.title,
        seo_title=meta.seo_title,
        # The published post's date, else the plan's creation time, stable
        # across reruns of the plan (the file name carries the publish day,
        # which is never earlier).
        date=meta.date or plan.created_at.astimezone(UTC),
        categories=[github.survey_category] + ([meta.topic] if meta.topic else []),
        tags=meta.tags,
        cover=meta.cover,
        use_math=has_math(body),
        description=description or None,
        lang=meta.lang,
        translation_id=meta.translation_id,
    )
    document = (
        f"{front_matter}{body}\n\n{reference_list(survey.plan.papers)}"
        + further_reading(survey.reading, writer.language)
    )
    work_dir = papers_dir / slug
    await asyncio.to_thread(context.files.make_dir, str(work_dir))
    core = sum(p.role == "core" for p in survey.plan.papers)
    publish_request = PublishRequest(
        title=survey.title,
        markdown=document,
        work_dir=work_dir,
        branch_id=slug,
        pr_title=f"Paper Survey: {survey.title}",
        pr_body=build_pr_body(
            kind="paper survey",
            title=survey.title,
            fields={
                "Topic": plan.topic,
                "Papers": f"{len(survey.plan.papers)} ({core} core)",
                "Lineages": str(len(survey.plan.lineages)),
                "Characters": f"{len(document):,}",
            },
        ),
        commit_message=f"feat: add paper survey '{survey.title}'",
        kind="survey",
        published_path=published.path if published else None,
    )
    publish_request.link_sources = plan_link_sources(survey.plan)
    publish_request.blocking += survey.blocking
    publish_request.review_notes += [
        *lookup_notes,
        *plan_notes,
        *survey.notes,
        *body_notes,
        *writer.notes,
    ]
    return await publish_artifact(context, publish_request, create_pr=request.create_pr)


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
