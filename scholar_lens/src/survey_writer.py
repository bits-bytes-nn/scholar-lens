"""Paper-survey write stage: plan file → evidence engine → post.

``survey write --plan`` reads the plan (local or S3) the plan stage wrote and
the user may have edited, resolves papers added by hand (DataCite for arXiv
ids, OpenAlex for DOIs), reads the plan's reference articles again
(explanatory material, listed as further reading), writes the post with the
evidence engine (:func:`write_survey_with_evidence`, docs/redesign.md) and
publishes it like the other kinds. Every check on the models' answers is
code (``survey_post``).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC
from pathlib import Path
from typing import Any

import boto3
from langchain_core.runnables import Runnable

from ..adapters.aws.storage import S3Handler
from ..adapters.scholarly.index import (
    DataCiteIndex,
    OpenAlexIndex,
    datacite_record,
    openalex_record,
)
from ..application.usage import staged
from ..domain.markdown.lint import drop_unknown_images, fill_image_alts
from ..domain.markdown.math import has_math, normalize_math
from ..domain.post_names import slugify
from ..domain.review_plan import digest_sections
from ..domain.survey.plan import (
    SourceRecord,
    SurveyInputError,
    SurveyPlan,
    survey_slug,
    unversioned_arxiv_id,
)
from ..domain.survey.post import (
    credit_figures,
    fill_metadata,
    foreign_paper_links,
    further_reading,
    needs_metadata,
    normalize_plan_data,
    plan_data,
    reference_list,
    render_citations,
    unlink_foreign_papers,
    validate_plan,
)
from ..domain.tags import controlled_tag_problems
from ..domain.validation import (
    description_markup,
    description_problems,
    fit_description,
)
from .constants import LanguageModelId, S3Paths
from .formatting import build_post_meta, render_front_matter
from .jobs import JobRequest
from .logger import logger
from .metrics import TokenUsageTracker
from .prompt_safety import neutralize_prompt_tags
from .prompts import SurveyFinalizationPrompt
from .publish_types import title_slug
from .publisher import PublishRequest, build_pr_body
from .runtime import (
    PipelineOutcome,
    RunContext,
    find_published_post,
    load_translation_guideline,
    publish_artifact,
)
from .survey_pipeline import read_references, reference_excerpts
from .utils import BedrockLanguageModelFactory, HTMLTagOutputParser, RetryableBase
from .web_research import WebResearcher

# Images of a paper's arXiv HTML rendering offered to its reading agent.
MAX_PAPER_IMAGES = 40
# ---- reading the plan ---------------------------------------------------------------


def read_plan_text(location: str, s3_handler: S3Handler | None) -> str:
    """The plan file's text from a local path or ``s3://<bucket>/<key>`` in
    the run's bucket; ``SurveyInputError`` when it is not there."""
    if location.startswith("s3://"):
        bucket, _, key = location.removeprefix("s3://").partition("/")
        if s3_handler is None or bucket != s3_handler.bucket_name or not key:
            raise SurveyInputError(
                f"The plan {location!r} is not in this run's S3 bucket."
            )
        try:
            text = s3_handler.get_text(key)
        except UnicodeDecodeError as e:
            raise SurveyInputError(f"The plan {location!r} is not UTF-8: {e}") from e
        if text is None:
            raise SurveyInputError(f"No plan at {location!r}.")
        return text
    path = Path(location).expanduser()
    if not path.is_file():
        raise SurveyInputError(f"No plan file at {location!r}.")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise SurveyInputError(f"The plan {location!r} is not UTF-8: {e}") from e


class PaperIndex:
    """Bibliographic records and abstracts of plan papers: arXiv's own
    metadata (DataCite) for arXiv ids, OpenAlex's for DOIs. A DOI paper that
    is also on arXiv takes arXiv's abstract (OpenAlex's abstracts of arXiv
    records are unreliable) and keeps its arXiv id. Blocking."""

    def __init__(self, datacite: DataCiteIndex, openalex: OpenAlexIndex) -> None:
        self.datacite = datacite
        self.openalex = openalex

    def records(self, paper_ids: Sequence[str]) -> dict[str, SourceRecord]:
        found: dict[str, SourceRecord] = {}
        arxiv_of = {i: i for i in paper_ids if unversioned_arxiv_id(i)}
        for doi in (i for i in paper_ids if i not in arxiv_of):
            works = self.openalex.by_doi(doi)
            if records := [r for w in works if (r := openalex_record(w, "plan"))]:
                found[doi] = records[0]
                if records[0].arxiv_id:
                    arxiv_of[doi] = records[0].arxiv_id
        if not arxiv_of:
            return found
        arxiv = {
            record.arxiv_id: record
            for item in self.datacite.arxiv_papers(sorted(set(arxiv_of.values())))
            if (record := datacite_record(item, None)) and record.arxiv_id
        }
        for paper_id, arxiv_id in arxiv_of.items():
            if arxiv_id not in arxiv:
                continue
            if paper_id in found:
                found[paper_id] = replace(
                    found[paper_id], abstract=arxiv[arxiv_id].abstract
                )
            else:
                found[paper_id] = arxiv[arxiv_id]
        return found


def load_plan(text: str, index: PaperIndex) -> tuple[SurveyPlan, list[str]]:
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
    request: JobRequest, s3_handler: S3Handler | None, s3_prefix: str
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
    text = read_plan_text(location, None)
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
        {**p, "title": p.get("title") or p["id"], "url": p.get("url") or p["id"]}
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
    # Figures offered to the writers: url -> (paper id, caption).
    figures: dict[str, tuple[str, str]] = field(default_factory=dict)


class SurveyWriter(RetryableBase):
    """The write stage's finalization call (description and taxonomy of
    the finished post)."""

    def __init__(
        self,
        *,
        finalization_model_id: LanguageModelId,
        boto_session: boto3.Session,
        language: str,
        thinking_effort: str,
        callbacks: list[Any] | None = None,
        llm_factory: BedrockLanguageModelFactory | None = None,
    ) -> None:
        self.callbacks = callbacks or []
        self.language = language
        self.notes: list[str] = []
        factory = llm_factory or BedrockLanguageModelFactory(boto_session=boto_session)
        model = factory.get_model(
            finalization_model_id,
            temperature=0.0,
            callbacks=self.callbacks or None,
            enable_thinking=False,
            thinking_effort=thinking_effort,
        )
        self.finalization_chain: Runnable = (
            SurveyFinalizationPrompt.get_prompt()
            | model
            | HTMLTagOutputParser(
                tag_names=SurveyFinalizationPrompt.output_variables or []
            )
        )

    async def finalize(
        self,
        title: str,
        body: str,
        *,
        category_options: Sequence[str],
        controlled_tags: Sequence[str],
    ) -> dict[str, str]:
        """Topic, tags and the search description, checked by the post
        validator's description rules with one repair round. A description
        the validator would reject (markup, not longer than the title) is
        dropped: the blog then derives one from the first paragraph."""
        payload = {
            "title": title,
            "outline": neutralize_prompt_tags(digest_sections([body])),
            "category_options": ", ".join(category_options),
            "controlled_tags": ", ".join(controlled_tags),
            "language": self.language,
            "repair_note": "",
        }
        result: dict[str, str] = {}
        problems: list[str] = []
        description = ""
        for attempt in range(2):
            result = await self._finalize(payload)
            description = fit_description(
                " ".join((result.get("description") or "").split())
            )
            result["description"] = description
            if not (
                problems := description_problems(description, body)
                + controlled_tag_problems(
                    result.get("controlled_tags"), controlled_tags
                )
            ):
                return result
            logger.warning(
                "Survey finalization answer unusable (attempt %d): %s",
                attempt + 1,
                problems,
            )
            payload["repair_note"] = (
                "Your previous answer could not be used: "
                + "; ".join(problems)
                + ". Answer all four parts again."
            )
        self.notes.append(
            "Survey description still breaks the rules after a repair: "
            + "; ".join(problems)
        )
        if description_markup(description) or len(description) <= len(title):
            result["description"] = ""
        return result

    @RetryableBase._retry("survey_finalization")
    @staged("finalize")
    async def _finalize(self, payload: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = await self.finalization_chain.ainvoke(payload)
        return result


async def write_survey_with_evidence(
    context: RunContext,
    plan: SurveyPlan,
    *,
    index: PaperIndex,
    researcher: WebResearcher,
    papers_dir: Path,
    translation_guideline: str,
    tracker: TokenUsageTracker,
    title_max_length: int,
) -> Survey:
    """The evidence engine: reading agents, an evidence-assigning outline,
    verified sections (docs/redesign.md)."""
    from ..adapters.bedrock.strands_models import RoleModel, StrandsLanguageModel
    from ..application.survey_engine import EngineSettings, EvidenceSurveyEngine
    from .evidence_models import budgeted, role_model, unservable
    from .survey_documents import EvidenceTextStore, PaperDocuments

    cfg = context.config.survey

    def role(model_id: str, effort: str, thinking: bool) -> RoleModel:
        return role_model(context, model_id, effort, thinking)

    model = StrandsLanguageModel(
        context.bedrock_boto_session,
        {
            "read": role(cfg.card_model_id, cfg.thinking_effort, False),
            "outline": role(cfg.outline_model_id, "high", True),
            # The writer falls back to the reading model when Bedrock's
            # content filter keeps stopping it (attack-heavy sections).
            "write": dataclasses.replace(
                role(cfg.writing_model_id, cfg.thinking_effort, True),
                fallback=role(cfg.card_model_id, cfg.thinking_effort, True),
            ),
            "evaluate": role(cfg.evaluation_model_id, cfg.thinking_effort, True),
        },
        budgeted(tracker, cfg.write_max_cost_usd),
    )
    records = await asyncio.to_thread(index.records, [p.id for p in plan.papers])
    documents = PaperDocuments(
        researcher,
        index.openalex,
        {pid: r.abstract for pid, r in records.items() if r.abstract},
    )
    language = context.config.output_language
    engine = EvidenceSurveyEngine(
        model,
        documents,
        EngineSettings(
            language=language,
            title_max_length=title_max_length,
            reader_tag=slugify(LanguageModelId(cfg.card_model_id).value),
            chars_per_core=cfg.chars_per_core_paper,
            chars_per_supporting=cfg.chars_per_supporting_paper,
            min_total_chars=cfg.min_total_chars,
            max_total_chars=cfg.max_total_chars,
        ),
        store=EvidenceTextStore(
            papers_dir / "survey-evidence",
            context.s3_handler,
            context.config.resources.s3_prefix,
        ),
    )
    articles = [r for r in plan.references if r.kind == "article"]
    read = await asyncio.to_thread(
        read_references, [r.url for r in articles], researcher
    )
    pages = [r.page for r in read if r.page]
    titles = {r.url: r.title for r in articles}
    reading = [(titles.get(r.url) or r.page.title, r.url) for r in read if r.page]
    notes = []
    if unread := [r.url for r in read if r.page is None]:
        notes.append(
            "Reference articles that could not be read (left out): " + ", ".join(unread)
        )
    result = await engine.write(
        plan,
        references=reference_excerpts(pages, plan.topic, 8_000) if pages else "",
        guideline=translation_guideline,
    )
    figures = {
        e.figure_url: (e.paper_id, e.caption or "")
        for e in result.ledger
        if e.kind == "figure" and e.figure_url and e.figure_url in result.body
    }
    # arXiv HTML lists some figures it does not serve (404): an image that
    # cannot be downloaded would fail the publish step, so it is dropped
    # here (survey_body removes images without an owner).
    broken = await unservable(list(figures))
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
        figures=figures,
    )


def survey_body(survey: Survey, language: str) -> tuple[str, list[str]]:
    """The body as published: citation keys rendered (unknown ones removed),
    images the writers were not offered dropped, figures credited to their
    paper, math normalized. Returns it with notes."""
    papers = {p.id: p for p in survey.plan.papers}
    foreign = foreign_paper_links(survey.body, papers.values())
    body, unknown = render_citations(
        unlink_foreign_papers(survey.body, papers.values()), papers
    )
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
    if foreign:
        notes.append(
            "Links to papers outside the survey reduced to text: "
            + ", ".join(sorted(set(foreign)))
        )
    return normalize_math(body.strip()), notes


async def run_survey_write_pipeline(
    context: RunContext, request: JobRequest, tracker: TokenUsageTracker
) -> PipelineOutcome:
    cfg = context.config.survey
    github = context.config.resources.github
    spec = context.require_conventions()
    if github.survey_category not in spec.types:
        raise ValueError(
            f"The blog's post conventions have no '{github.survey_category}' type "
            f"(types: {', '.join(spec.types)}): add it to the blog's "
            "_data/post_conventions.yml or set resources.github.survey_category."
        )
    text = await asyncio.to_thread(
        read_plan_text, request.plan or "", context.s3_handler
    )
    index = PaperIndex(DataCiteIndex(), OpenAlexIndex())
    plan, plan_notes = await asyncio.to_thread(load_plan, text, index)
    guideline = await load_translation_guideline(context)
    writer = SurveyWriter(
        finalization_model_id=LanguageModelId(cfg.outline_model_id),
        boto_session=context.bedrock_boto_session,
        language=context.config.output_language,
        thinking_effort=cfg.thinking_effort,
        callbacks=[tracker],
    )
    researcher = WebResearcher(max_images_per_page=MAX_PAPER_IMAGES)
    papers_dir = context.papers_dir
    try:
        survey = await write_survey_with_evidence(
            context,
            plan,
            index=index,
            researcher=researcher,
            papers_dir=papers_dir,
            translation_guideline=str(guideline or ""),
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
        researcher.close()
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
    work_dir.mkdir(parents=True, exist_ok=True)
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
