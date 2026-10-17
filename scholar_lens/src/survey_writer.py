"""Paper-survey write stage: plan file → paper cards → outline → sections →
post.

``survey write --plan`` reads the plan (local or S3) the plan stage wrote and
the user may have edited, resolves papers added by hand (DataCite for arXiv
ids, OpenAlex for DOIs), turns each paper into a card (cached on disk and in
S3), reads the plan's reference articles again (explanatory material for the
sections that frame the field, listed as further reading), plans the
outline, writes every section through the guide's
write → evaluate → revise → ground loop (``section_loop.revise_section``,
``checked_grounding``) and publishes the post like the other kinds. Every
check on the models' answers is code (``survey_post``).
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC
from pathlib import Path
from typing import Any

import boto3
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from .aws_helpers import S3Handler
from .constants import LanguageModelId, LocalPaths, S3Paths
from .formatting import build_post_meta, cut_title, render_front_matter
from .jobs import JobRequest
from .logger import logger
from .markdown_lint import drop_unknown_images, fill_image_alts
from .markdown_math import has_math, normalize_math
from .metrics import TokenUsageTracker
from .post_validator import description_markup, description_problems
from .prompt_safety import neutralize_prompt_tags
from .prompts import (
    BasePrompt,
    SurveyCardPrompt,
    SurveyEvaluationPrompt,
    SurveyFinalizationPrompt,
    SurveyGroundingPrompt,
    SurveyOutlinePrompt,
    SurveySectionPrompt,
)
from .prompts.base import PROSE_UNIT, blocking_issues_block
from .prompts.survey import DEPTH_GUIDANCE, SECTION_ROLES
from .publish_types import title_slug
from .publisher import PublishRequest, build_pr_body
from .quality_gate import rule_issues
from .review_plan import (
    article_ledger,
    digest_sections,
    ensure_section_heading,
)
from .runtime import (
    ROOT_DIR,
    PipelineOutcome,
    RunContext,
    find_published_post,
    load_translation_guideline,
    publish_artifact,
)
from .scholarly_index import (
    DataCiteIndex,
    OpenAlexIndex,
    datacite_record,
    openalex_record,
)
from .section_loop import (
    EMPTY_DRAFT,
    FIXING,
    REVISING,
    checked_grounding,
    feedback_or_generic,
    glossary_view,
    measured_length,
    owned_elsewhere,
    parse_section_output,
    revise_section,
)
from .survey_pipeline import (
    SurveyCalls,
    read_references,
    reference_excerpts,
)
from .survey_plan import (
    PlanPaper,
    SourceRecord,
    SurveyInputError,
    SurveyPlan,
    SurveyPlanError,
    parse_tagged_json,
    survey_slug,
    unversioned_arxiv_id,
)
from .survey_post import (
    CardAnswer,
    CardSource,
    OutlineSection,
    PaperCard,
    SurveyOutline,
    apply_outline,
    author_year,
    build_card,
    card_cache_key,
    card_problems,
    check_outline,
    citation_issues,
    clean_outline,
    comparison_table,
    credit_figures,
    figure_list,
    fill_metadata,
    foreign_paper_links,
    further_reading,
    needs_metadata,
    normalize_plan_data,
    plan_data,
    prior_work_text,
    reference_list,
    render_card,
    render_card_brief,
    render_card_outcomes,
    render_citations,
    section_targets,
    structure_problems,
    timeline_table,
    unlink_foreign_papers,
    validate_plan,
    without_papers,
)
from .tags import controlled_tag_problems
from .utils import (
    BedrockLanguageModelFactory,
    HTMLTagOutputParser,
    parse_lines,
    parse_quality_score,
)
from .web_research import (
    PageContent,
    ResearchCorpus,
    SourceFetchError,
    WebResearcher,
)

# Card calls in flight at once (arXiv fetches are paced process-wide anyway).
CARD_CONCURRENCY = 4
# Images of a paper's arXiv HTML rendering offered to its card call.
MAX_PAPER_IMAGES = 40
# Sections that explain the papers they cover (full cards); the others place
# or compare papers already explained (brief lines).
COVERING_KINDS = ("lineage", "cross_cutting")
# Sections that frame the field or tie lineages together (cross_cutting also
# explains its papers) and so get the user's reference articles, within this
# many characters in all.
REFERENCE_KINDS = ("introduction", "cross_cutting", "comparison", "insights")
REFERENCE_MATERIAL_CHARS = 8_000


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


# ---- paper cards ------------------------------------------------------------------------


class CardStore:
    """Cards on disk (``papers/survey-cards/``) and, with a bucket, in S3
    (``<prefix>/surveys/cards/``), keyed by :func:`card_cache_key`."""

    def __init__(
        self, local_dir: Path, s3_handler: S3Handler | None, s3_prefix: str
    ) -> None:
        self.local_dir = local_dir
        self.s3_handler = s3_handler
        self.s3_prefix = f"{s3_prefix}/{S3Paths.SURVEYS.value}/cards".lstrip("/")

    def get(self, key: str) -> PaperCard | None:
        """The cached card, or None (also for a card this code cannot read,
        which is then written again)."""
        path = self.local_dir / key
        text = path.read_text(encoding="utf-8") if path.is_file() else None
        if text is None and self.s3_handler is not None:
            text = self.s3_handler.get_text(f"{self.s3_prefix}/{key}")
        if not text:
            return None
        try:
            return PaperCard.model_validate_json(text)
        except ValidationError as e:
            logger.warning("Unreadable cached card %s: %s", key, e)
            return None

    def put(self, key: str, card: PaperCard) -> None:
        text = card.model_dump_json(indent=2)
        path = self.local_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if self.s3_handler is not None:
            self.s3_handler.put_text(f"{self.s3_prefix}/{key}", text)


def card_variant(card_source_chars: int, translation_guideline: str = "") -> str:
    """A short digest of what a card is written by: the card prompt, the
    source limit and the translation guideline (a change to any writes the
    cards again)."""
    template = (
        SurveyCardPrompt.system_prompt_template
        + SurveyCardPrompt.human_prompt_template
        + str(card_source_chars)
        + translation_guideline
    )
    return hashlib.sha256(template.encode()).hexdigest()[:8]


@dataclass
class PaperText:
    """What a card is written from."""

    text: str
    kind: CardSource
    # Captioned images of the paper's arXiv HTML rendering (url -> caption).
    figures: dict[str, str] = field(default_factory=dict)


def paper_text(
    paper: PlanPaper, record: SourceRecord | None, researcher: WebResearcher
) -> PaperText | None:
    """A core paper's arXiv HTML full text (with its figures) when it is on
    arXiv, else the abstract from its index record; None when there is
    neither. Blocking."""
    arxiv_id = unversioned_arxiv_id(paper.id) or (record and record.arxiv_id)
    if paper.role == "core" and arxiv_id:
        try:
            corpus = researcher.research(
                [f"https://arxiv.org/abs/{arxiv_id}"], discover_subpages=False
            )
        except SourceFetchError as e:  # arXiv unreachable: the abstract will do
            logger.warning("Could not read arXiv %s: %s", arxiv_id, e)
            corpus = ResearchCorpus()
        page: PageContent | None = corpus.pages[0] if corpus.pages else None
        if page is not None and "/html/" in (page.final_url or page.url):
            return PaperText(page.text, "full_text", dict(page.image_captions))
    if record is not None and record.abstract:
        return PaperText(record.abstract, "abstract")
    return None


# ---- the writer -----------------------------------------------------------------------


@dataclass
class Survey:
    """The written survey before it becomes a post."""

    title: str
    body: str
    plan: SurveyPlan
    cards: dict[str, PaperCard]
    # The reference articles that were read: (title, url) for further reading.
    reading: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class SurveyWriter(SurveyCalls):
    """The write stage's model calls: cards, outline, sections (writer,
    evaluator, fact-checker) and finalization."""

    def __init__(
        self,
        *,
        card_model_id: LanguageModelId,
        outline_model_id: LanguageModelId,
        writing_model_id: LanguageModelId,
        evaluation_model_id: LanguageModelId,
        boto_session: boto3.Session,
        language: str,
        thinking_effort: str,
        card_source_chars: int,
        min_quality_score: int,
        max_revision_attempts: int,
        verify_grounding: bool,
        title_max_length: int,
        lengths: Mapping[str, int],
        max_total_tokens: int | None,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.callbacks = callbacks or []
        self._init_token_budget(self.callbacks, max_total_tokens)
        self.language = language
        self.card_model_id = card_model_id
        self.card_source_chars = card_source_chars
        self.min_quality_score = min_quality_score
        self.max_revision_attempts = max_revision_attempts
        self.verify_grounding = verify_grounding
        self.title_max_length = title_max_length
        self.lengths = dict(lengths)
        self.notes: list[str] = []
        factory = BedrockLanguageModelFactory(boto_session=boto_session)

        def model(model_id: LanguageModelId, *, thinking: bool = False) -> Any:
            return factory.get_model(
                model_id,
                temperature=0.0,
                callbacks=self.callbacks or None,
                enable_thinking=thinking,
                thinking_effort=thinking_effort,
            )

        def tagged(prompt: type[BasePrompt], model_id: LanguageModelId) -> Runnable:
            return (
                prompt.get_prompt()
                | model(model_id)
                | HTMLTagOutputParser(tag_names=prompt.output_variables or [])
            )

        self.card_chain = (
            SurveyCardPrompt.get_prompt() | model(card_model_id) | StrOutputParser()
        )
        # Grouping the papers into sections is the stage's main judgment.
        self.outline_chain = (
            SurveyOutlinePrompt.get_prompt()
            | model(outline_model_id, thinking=True)
            | StrOutputParser()
        )
        # Parsed leniently: a missing wrapper keeps the text (the guide's way).
        self.section_chain = (
            SurveySectionPrompt.get_prompt()
            | model(writing_model_id)
            | StrOutputParser()
            | RunnableLambda(parse_section_output)
        )
        self.evaluation_chain = tagged(SurveyEvaluationPrompt, evaluation_model_id)
        self.grounding_chain = tagged(SurveyGroundingPrompt, writing_model_id)
        self.finalization_chain = tagged(SurveyFinalizationPrompt, outline_model_id)

    # ---- budget ------------------------------------------------------------------

    def _refine(self) -> bool:
        """Whether a section still gets evaluation, revision and grounding
        (False once the soft token limit is reached)."""
        if not self._over_soft_token_budget():
            return True
        note = (
            "Token budget reached: later survey sections were written without "
            "evaluation, revision or grounding."
        )
        if note not in self.notes:
            self.notes.append(note)
            logger.warning(note)
        return False

    # ---- cards ---------------------------------------------------------------------

    async def card(
        self, paper: PlanPaper, text: PaperText, translation_guideline: str = ""
    ) -> PaperCard:
        """One card call (one repair round); the answer is checked against
        the paper's text in code (:func:`survey_post.build_card`). The
        card's texts render terms by ``translation_guideline`` (they fill the
        post's tables as written)."""
        figures = figure_list(text.figures) if text.kind == "full_text" else []
        payload = {
            "title": neutralize_prompt_tags(paper.title),
            "citation": neutralize_prompt_tags(
                f"{paper.authors}, {paper.year or 'n.d.'}"
            ),
            "source_kind": "full text" if text.kind == "full_text" else "abstract",
            "source": neutralize_prompt_tags(text.text[: self.card_source_chars]),
            "figures": neutralize_prompt_tags(
                "\n".join(f"- {fid}: {caption}" for fid, _, caption in figures)
            )
            or "(none)",
            "translation_guideline": translation_guideline or "- none",
            "language": self.language,
        }

        def parse(raw: str) -> tuple[CardAnswer, list[str]]:
            answer = parse_tagged_json(raw, "card", CardAnswer)
            ids = [fid for fid, _, _ in figures]
            return answer, card_problems(answer, source=text.text, figure_ids=ids)

        answer = await self._ask(
            self.card_chain, payload, parse, tag="card", accept_last=True
        )
        return build_card(
            answer,
            paper_id=paper.id,
            source=text.text,
            source_kind=text.kind,
            figures=figures,
            model_id=self.card_model_id.value,
            language=self.language,
        )

    # ---- outline -------------------------------------------------------------------

    async def outline(
        self,
        plan: SurveyPlan,
        cards: Mapping[str, PaperCard],
        translation_guideline: str,
    ) -> SurveyOutline:
        papers = {p.id: p for p in plan.papers}
        lineages = "\n".join(
            f"- {lin.id}: {lin.name} — {lin.description}\n  papers: "
            + ", ".join(
                f"{p.id} ({p.year or 'n.d.'}, {p.role})"
                for p in plan.papers
                if p.lineage == lin.id
            )
            for lin in plan.lineages
        )
        digest = "\n".join(
            render_card_brief(card, papers[pid])
            + f"\n  role: {papers[pid].role}"
            + "".join(
                f"\n  equation {e.id}: {e.meaning[:200]}" for e in card.key_equations
            )
            + "".join(f"\n  figure {f.id}: {f.caption[:200]}" for f in card.figures)
            for pid, card in cards.items()
        )
        payload = {
            "topic": neutralize_prompt_tags(plan.topic),
            "scope": neutralize_prompt_tags(plan.scope),
            "lineages": neutralize_prompt_tags(lineages),
            "cards": neutralize_prompt_tags(digest),
            "connections": neutralize_prompt_tags(connections(plan, cards, None)),
            "translation_guideline": translation_guideline,
            "title_max_length": self.title_max_length,
            "language": self.language,
        }

        def parse(raw: str) -> tuple[SurveyOutline, list[str]]:
            outline = parse_tagged_json(raw, "outline", SurveyOutline)
            return outline, check_outline(
                outline, plan, cards, title_max_length=self.title_max_length
            )

        outline = await self._ask(
            self.outline_chain, payload, parse, tag="outline", accept_last=True
        )
        if problems := structure_problems(outline, plan):
            raise SurveyPlanError(f"The survey outline is unusable: {problems}")
        if removed := clean_outline(outline, plan, cards):
            self.notes.append(
                "Outline references removed after the repair round: "
                + "; ".join(removed)
            )
        if len(outline.title) > self.title_max_length:
            outline.title = cut_title(outline.title, self.title_max_length)
        apply_outline(
            outline,
            plan,
            section_targets(
                outline,
                plan,
                per_core=self.lengths["per_core"],
                per_supporting=self.lengths["per_supporting"],
                min_total=self.lengths["min_total"],
                max_total=self.lengths["max_total"],
                language=self.language,
            ),
        )
        logger.info(
            "Survey outline: '%s', %d sections, %s characters of prose.",
            outline.title,
            len(outline.sections),
            f"{sum(s.target_chars for s in outline.sections):,}",
        )
        return outline

    # ---- sections ------------------------------------------------------------------

    async def write(
        self,
        plan: SurveyPlan,
        cards: Mapping[str, PaperCard],
        outline: SurveyOutline,
        material: Sequence[PageContent] = (),
    ) -> str:
        """Every section through write → evaluate → revise → ground; the
        code's tables appended (timeline to the introduction, comparison to
        the comparison section); ``material`` (the reference articles' pages)
        offered to the sections of :data:`REFERENCE_KINDS`."""
        papers = {p.id: p for p in plan.papers}
        lineage_titles = {
            s.lineage: s.title
            for s in outline.sections
            if s.kind == "lineage" and s.lineage
        }
        tables = {
            "introduction": timeline_table(plan, cards, self.language, lineage_titles),
            "comparison": comparison_table(plan, cards, self.language, lineage_titles),
        }
        written: list[str] = []
        for index, section in enumerate(outline.sections):
            self._enforce_hard_token_budget()
            logger.info(
                "Writing survey section %d/%d: %s",
                index + 1,
                len(outline.sections),
                section.title,
            )
            table = tables.get(section.kind, "")
            inputs = self._inputs(plan, cards, outline, index, written, table, material)
            directive = self._directive(section)
            text = await self._write_section({**inputs, "depth_directive": directive})
            if not text:
                # Once more, like the review loop, before leaving a hole.
                logger.warning(
                    "Survey section '%s' came back empty - retrying once.",
                    section.title,
                )
                text = await self._write_section(
                    {**inputs, "depth_directive": f"{directive}\n\n{EMPTY_DRAFT}"}
                )
            if not text:
                self.notes.append(f"Survey section '{section.title}' came back empty.")
                if table:  # the code's table still belongs in the post
                    written.append(f"## {section.title}\n\n{table}")
                continue
            text = ensure_section_heading(text, section.title)
            # A lineage section cites each of its papers.
            required = [p for p in section.papers if section.kind == "lineage"]

            def rules(markdown: str, required: list[str] = required) -> list[str]:
                return rule_issues(markdown) + citation_issues(
                    markdown, papers, required
                )

            blocking: list[str] = []
            if self._refine() and self.max_revision_attempts > 0:
                text, blocking = await self._revise(
                    section, inputs, directive, text, rules
                )
            if self.verify_grounding and self._refine():
                if blocking:
                    logger.info(
                        "Grounding survey section '%s' with %d leftover blocking "
                        "issue(s).",
                        section.title,
                        len(blocking),
                    )
                grounded = await checked_grounding(
                    text, functools.partial(self._ground, text, inputs, blocking)
                )
                # Grounding is the last pass: a citation it broke (e.g. the
                # only citation of a covered paper removed) keeps the draft.
                before = citation_issues(text, papers, required)
                if set(citation_issues(grounded, papers, required)) - set(before):
                    logger.info(
                        "Grounding broke survey section '%s''s citations: kept "
                        "the draft before grounding.",
                        section.title,
                    )
                    self.notes.append(
                        f"Survey section '{section.title}': grounding broke its "
                        "citations, so the draft before grounding was kept."
                    )
                else:
                    text = grounded
            # Only the section's planned figures may show in it.
            text = drop_unknown_images(text, set(_figure_urls(section, cards)))
            written.append(f"{text}\n\n{table}" if table else text)
        return "\n\n".join(written)

    def _inputs(
        self,
        plan: SurveyPlan,
        cards: Mapping[str, PaperCard],
        outline: SurveyOutline,
        index: int,
        written: Sequence[str],
        table: str,
        material: Sequence[PageContent] = (),
    ) -> dict[str, Any]:
        """What the writer, evaluator and fact-checker of section ``index``
        see (all but the depth directive)."""
        papers = {p.id: p for p in plan.papers}
        section = outline.sections[index]
        # A section that names no paper of its own (the introduction,
        # comparison, insights) draws on the core papers.
        own = [p for p in section.papers if p in cards] or [
            p.id for p in plan.papers if p.role == "core" and p.id in cards
        ]
        # Only a section that explains its papers gets their full cards; the
        # comparison gets their results and limitations, the rest a line.
        render = (
            render_card
            if section.kind in COVERING_KINDS
            else (
                render_card_outcomes
                if section.kind == "comparison"
                else render_card_brief
            )
        )
        appended = (
            f"\n(The code appends the {'timeline' if section.kind == 'introduction' else 'comparison'} "
            "table after this section's text.)"
            if table
            else ""
        )
        previous = "\n\n".join(
            part
            for part in (digest_sections(written[:-2]), "\n\n".join(written[-2:]))
            if part
        )
        return {
            "title": outline.title,
            "outline": "\n".join(
                _section_line(s, i) for i, s in enumerate(outline.sections, 1)
            ),
            "section": _section_line(section, index + 1) + appended,
            "section_number": index + 1,
            "total_sections": len(outline.sections),
            "previous_sections": previous or "(this is the first section)",
            "already_in_post": article_ledger(written),
            "owned_elsewhere": owned_elsewhere(outline.sections, index),
            "glossary": glossary_view(outline.glossary_map, list(written)),
            "cards": neutralize_prompt_tags(
                "\n\n".join(render(cards[p], papers[p]) for p in own)
            ),
            "other_papers": neutralize_prompt_tags(
                "\n".join(
                    render_card_brief(card, papers[pid])
                    for pid, card in cards.items()
                    if pid not in own
                )
                or "- none"
            ),
            "connections": neutralize_prompt_tags(
                connections(plan, cards, own or None)
            ),
            "citation_keys": neutralize_prompt_tags(
                "\n".join(
                    f"- [@{p.id}] {cards[p.id].name} ({author_year(p)}): {p.title}"
                    for p in plan.papers
                    if p.id in cards
                )
            ),
            "reference_material": (
                reference_excerpts(
                    material,
                    f"{plan.topic} {section.title} {section.description}",
                    REFERENCE_MATERIAL_CHARS,
                )
                if section.kind in REFERENCE_KINDS
                else "none"
            ),
            "figures": _figures_text(section, cards),
            "equations": _equations_text(section, cards),
            "appended_table": table or "none",
            "language": self.language,
        }

    def _directive(self, section: OutlineSection) -> str:
        role = f"Role: {SECTION_ROLES[section.kind]}"
        # A lineage section's length follows its papers, not a depth.
        directive = (
            f"LINEAGE section (about {section.target_chars:,} {PROSE_UNIT}). {role}"
            if section.kind == "lineage"
            else f"{section.depth.upper()} section (about "
            f"{section.target_chars:,} {PROSE_UNIT}): "
            f"{DEPTH_GUIDANCE[section.depth]}\n{role}"
        )
        if section.tables:
            directive += "\nTables to build from the cards: " + "; ".join(
                section.tables
            )
        return directive

    async def _revise(
        self,
        section: OutlineSection,
        inputs: dict[str, Any],
        directive: str,
        text: str,
        rules: Callable[[str], list[str]],
    ) -> tuple[str, list[str]]:
        """The best draft and the evaluator's blocking issues it still has."""

        async def evaluate(markdown: str) -> tuple[int, str, list[str], list[str]]:
            score, feedback, blocking, wording = await self._evaluate(
                markdown, inputs, directive, section
            )
            feedback = feedback_or_generic(score, feedback, self.min_quality_score)
            return score, feedback, blocking, wording

        async def write(request: str) -> str:
            revised = await self._write_section(
                {**inputs, "depth_directive": f"{directive}\n\n{request}"}
            )
            return ensure_section_heading(revised, section.title) if revised else ""

        async def revise(feedback: str, draft: str) -> str:
            return await write(REVISING.format(feedback=feedback, draft=draft))

        async def fix(request: str, draft: str) -> str:
            return await write(FIXING.format(request=request, draft=draft))

        best = await revise_section(
            text,
            evaluate=evaluate,
            revise=revise,
            length=lambda markdown: _length(markdown, section),
            min_score=self.min_quality_score,
            max_attempts=self.max_revision_attempts,
            label=f"'{section.title}'",
            rules=rules,
            fix=fix,
            # The re-score of the fix is an optional check (soft budget).
            rescore=lambda: not self._over_soft_token_budget(),
        )
        if note := best.below_bar_note("Survey", section.title, self.min_quality_score):
            self.notes.append(note)
        return best.text, best.blocking

    @SurveyCalls._retry("survey_section")
    async def _write_section(self, payload: dict[str, Any]) -> str:
        self._enforce_hard_token_budget()
        result = await self.section_chain.ainvoke(payload)
        return str((result or {}).get("section_markdown", "")).strip()

    @SurveyCalls._retry("survey_evaluation")
    async def _evaluate(
        self,
        markdown: str,
        inputs: dict[str, Any],
        directive: str,
        section: OutlineSection,
    ) -> tuple[int, str, list[str], list[str]]:
        self._enforce_hard_token_budget()
        result = await self.evaluation_chain.ainvoke(
            {
                "title": inputs["title"],
                "outline": inputs["outline"],
                "section_to_write": inputs["section"],
                "depth_directive": directive,
                "section": markdown,
                "previous_sections": inputs["previous_sections"],
                "owned_elsewhere": inputs["owned_elsewhere"],
                "glossary": inputs["glossary"],
                "length_report": _length(markdown, section),
                "cards": inputs["cards"],
                "other_papers": inputs["other_papers"],
                "connections": inputs["connections"],
                "reference_material": inputs["reference_material"],
                "appended_table": inputs["appended_table"],
                "language": self.language,
            }
        )
        score = parse_quality_score((result or {}).get("quality_score"))
        feedback = ((result or {}).get("improvement_feedback") or "").strip()
        return (
            score,
            feedback,
            parse_lines((result or {}).get("blocking_issues")),
            parse_lines((result or {}).get("wording_issues")),
        )

    @SurveyCalls._retry("survey_grounding")
    async def _ground(
        self, draft: str, inputs: dict[str, Any], blocking: list[str] | None = None
    ) -> str:
        self._enforce_hard_token_budget()
        result = await self.grounding_chain.ainvoke(
            {
                "section": draft,
                "cards": inputs["cards"],
                "other_papers": inputs["other_papers"],
                "connections": inputs["connections"],
                "reference_material": inputs["reference_material"],
                "appended_table": inputs["appended_table"],
                "glossary": inputs["glossary"],
                "language": self.language,
                "blocking_issues": blocking_issues_block(
                    blocking or [], "<cards>, <other_papers> and <appended_table>"
                ),
            }
        )
        return str((result or {}).get("grounded_markdown") or "").strip() or draft

    # ---- finalization -----------------------------------------------------------

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
            description = " ".join((result.get("description") or "").split())
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

    @SurveyCalls._retry("survey_finalization")
    async def _finalize(self, payload: dict[str, Any]) -> dict[str, str]:
        self._enforce_hard_token_budget()
        result: dict[str, str] = await self.finalization_chain.ainvoke(payload)
        return result


def _length(markdown: str, section: OutlineSection) -> str:
    kind = "lineage" if section.kind == "lineage" else section.depth
    return measured_length(
        markdown,
        label=f"this {kind} section",
        bound="about",
        target=section.target_chars,
    )


def _section_line(section: OutlineSection, number: int) -> str:
    line = f"{number}. [{section.kind}] {section.title}"
    if section.description:
        line += f" — {section.description}"
    if section.papers:
        line += f" (papers: {', '.join(section.papers)})"
    return line


def _planned(
    ids: Sequence[str], cards: Mapping[str, PaperCard], attr: str
) -> list[tuple[str, Any]]:
    """``(paper id, item)`` for each planned figure or equation id."""
    found = []
    for item_id in ids:
        pid = item_id.split("#", 1)[0]
        found += [(pid, x) for x in getattr(cards[pid], attr) if x.id == item_id]
    return found


def _figure_urls(section: OutlineSection, cards: Mapping[str, PaperCard]) -> list[str]:
    return [f.url for _, f in _planned(section.figures, cards, "figures")]


def _figures_text(section: OutlineSection, cards: Mapping[str, PaperCard]) -> str:
    return (
        "\n".join(
            f"- {f.url} — from [@{pid}] {cards[pid].name}: "
            f"{neutralize_prompt_tags(f.caption)}"
            for pid, f in _planned(section.figures, cards, "figures")
        )
        or "none"
    )


def _equations_text(section: OutlineSection, cards: Mapping[str, PaperCard]) -> str:
    return (
        "\n".join(
            f"- {e.id} of [@{pid}] {cards[pid].name}: "
            f"$${neutralize_prompt_tags(e.latex)}$$ — "
            f"{neutralize_prompt_tags(e.meaning)}"
            for pid, e in _planned(section.equations, cards, "key_equations")
        )
        or "none"
    )


def connections(
    plan: SurveyPlan, cards: Mapping[str, PaperCard], focus: Sequence[str] | None
) -> str:
    """What the text may build relations on, for the ``focus`` papers (all
    when None): the plan's citation edges (order and awareness only) and the
    prior work each card names with the relation the paper states."""
    names = {pid: card.name for pid, card in cards.items()}
    years = {p.id: p.year or "n.d." for p in plan.papers}
    wanted = set(focus) if focus is not None else {p.id for p in plan.papers}

    def label(pid: str) -> str:
        return f"[@{pid}] {names.get(pid, pid)} ({years.get(pid, 'n.d.')})"

    lines = [
        f"- {label(e.citing)} came after and cites {label(e.cited)}"
        for e in plan.edges
        if e.citing in wanted or e.cited in wanted
    ]
    lines += [
        f"- {label(pid)}, by its card: {prior_work_text(card)}"
        for pid, card in cards.items()
        if pid in wanted and card.prior_work
    ]
    return "\n".join(lines) or "- none"


# ---- the stage --------------------------------------------------------------------------


async def gather_cards(
    plan: SurveyPlan,
    writer: SurveyWriter,
    *,
    store: CardStore,
    index: PaperIndex,
    researcher: WebResearcher,
    translation_guideline: str = "",
) -> tuple[dict[str, PaperCard], list[str]]:
    """A card per plan paper (cached ones reused), in plan order, and the
    notes on papers left out (no text, or no usable card)."""
    model_id = writer.card_model_id.value
    variant = card_variant(writer.card_source_chars, translation_guideline)
    keys = {
        p.id: card_cache_key(p.id, model_id, writer.language, variant)
        for p in plan.papers
    }
    cards: dict[str, PaperCard] = {}
    for paper in plan.papers:
        cached = await asyncio.to_thread(store.get, keys[paper.id])
        # Two ids can share a file name: the card must be this paper's.
        if cached and cached.paper_id == paper.id:
            cards[paper.id] = cached
    # A card from the abstract does not do for a core paper on arXiv (it was
    # supporting in another survey, or arXiv was unreachable); a DOI paper's
    # arXiv id comes from its index record.
    abstract_core = [
        p.id
        for p in plan.papers
        if p.role == "core" and p.id in cards and cards[p.id].source == "abstract"
    ]
    looked_up = [p.id for p in plan.papers if p.id not in cards] + [
        pid for pid in abstract_core if not unversioned_arxiv_id(pid)
    ]
    records = await asyncio.to_thread(index.records, looked_up) if looked_up else {}
    for pid in abstract_core:
        record = records.get(pid)
        if unversioned_arxiv_id(pid) or (record and record.arxiv_id):
            del cards[pid]
    missing = [p for p in plan.papers if p.id not in cards]
    logger.info("Survey cards: %d cached, %d to write.", len(cards), len(missing))
    notes: list[str] = []
    limit = asyncio.Semaphore(CARD_CONCURRENCY)

    async def one(paper: PlanPaper) -> None:
        async with limit:
            text = await asyncio.to_thread(
                paper_text, paper, records.get(paper.id), researcher
            )
            if text is None:
                notes.append(f"No text found for {paper.id} ({paper.title}): left out.")
                return
            try:
                card = await writer.card(paper, text, translation_guideline)
            except SurveyPlanError as e:
                notes.append(
                    f"No usable card for {paper.id} ({paper.title}): left out ({e})."
                )
                return
            cards[paper.id] = card
            # A core arXiv paper read from its abstract may only have been
            # unreachable: write its card again next time.
            record = records.get(paper.id)
            on_arxiv = unversioned_arxiv_id(paper.id) or (record and record.arxiv_id)
            if not (paper.role == "core" and on_arxiv and text.kind == "abstract"):
                await asyncio.to_thread(store.put, keys[paper.id], card)

    try:
        async with asyncio.TaskGroup() as group:
            for paper in missing:
                group.create_task(one(paper))
    except ExceptionGroup as failed:
        # The first failure cancels the other calls: report its own error.
        raise failed.exceptions[0] from failed
    return {p.id: cards[p.id] for p in plan.papers if p.id in cards}, notes


def card_notes(plan: SurveyPlan, cards: Mapping[str, PaperCard]) -> list[str]:
    """PR checklist lines on what the cards could not back."""
    notes = []
    dropped = [
        f"{card.name} ({len(card.dropped)})" for card in cards.values() if card.dropped
    ]
    if dropped:
        notes.append(
            "Unverified numbers, equations or figures removed from paper cards: "
            + ", ".join(dropped)
        )
    abstract_only = [
        cards[p.id].name
        for p in plan.papers
        if p.role == "core" and p.id in cards and cards[p.id].source == "abstract"
    ]
    if abstract_only:
        notes.append(
            "Core papers written from their abstract only (no arXiv HTML full "
            "text): " + ", ".join(abstract_only)
        )
    return notes


async def write_survey(
    plan: SurveyPlan,
    writer: SurveyWriter,
    *,
    store: CardStore,
    index: PaperIndex,
    researcher: WebResearcher,
    translation_guideline: str,
) -> Survey:
    """Cards and reference articles → outline → sections."""
    cards, notes = await gather_cards(
        plan,
        writer,
        store=store,
        index=index,
        researcher=researcher,
        translation_guideline=translation_guideline,
    )
    articles = [r for r in plan.references if r.kind == "article"]
    read = await asyncio.to_thread(
        read_references, [r.url for r in articles], researcher
    )
    titles = {r.url: r.title for r in articles}
    reading = [(titles.get(r.url) or r.page.title, r.url) for r in read if r.page]
    if unread := [r.url for r in read if r.page is None]:
        notes.append(
            "Reference articles that could not be read (left out): " + ", ".join(unread)
        )
    plan = without_papers(plan, [p.id for p in plan.papers if p.id not in cards])
    outline = await writer.outline(plan, cards, translation_guideline)
    body = await writer.write(plan, cards, outline, [r.page for r in read if r.page])
    return Survey(
        title=outline.title,
        body=body,
        plan=plan,
        cards=cards,
        reading=reading,
        notes=[*notes, *card_notes(plan, cards)],
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
    owners = {
        f.url: papers[card.paper_id]
        for card in survey.cards.values()
        for f in card.figures
    }
    captions = {
        f.url: f.caption for card in survey.cards.values() for f in card.figures
    }
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
    resources = context.config.resources
    text = await asyncio.to_thread(
        read_plan_text, request.plan or "", context.s3_handler
    )
    index = PaperIndex(DataCiteIndex(), OpenAlexIndex())
    plan, plan_notes = await asyncio.to_thread(load_plan, text, index)
    guideline = await load_translation_guideline(context)
    writer = SurveyWriter(
        card_model_id=LanguageModelId(cfg.card_model_id),
        outline_model_id=LanguageModelId(cfg.outline_model_id),
        writing_model_id=LanguageModelId(cfg.writing_model_id),
        evaluation_model_id=LanguageModelId(cfg.evaluation_model_id),
        boto_session=context.bedrock_boto_session,
        language=context.config.output_language,
        thinking_effort=cfg.thinking_effort,
        card_source_chars=cfg.card_source_chars,
        min_quality_score=cfg.min_quality_score,
        max_revision_attempts=cfg.max_revision_attempts,
        verify_grounding=cfg.verify_grounding,
        title_max_length=spec.seo_title_max_length,
        lengths={
            "per_core": cfg.chars_per_core_paper,
            "per_supporting": cfg.chars_per_supporting_paper,
            "min_total": cfg.min_total_chars,
            "max_total": cfg.max_total_chars,
        },
        max_total_tokens=cfg.write_max_total_tokens,
        callbacks=[tracker],
    )
    researcher = WebResearcher(max_images_per_page=MAX_PAPER_IMAGES)
    papers_dir = ROOT_DIR / LocalPaths.PAPERS_DIR.value
    store = CardStore(
        papers_dir / "survey-cards", context.s3_handler, resources.s3_prefix
    )
    try:
        survey = await write_survey(
            plan,
            writer,
            store=store,
            index=index,
            researcher=researcher,
            translation_guideline=str(guideline or ""),
        )
        body, body_notes = survey_body(survey, writer.language)
        taxonomy = await writer.finalize(
            survey.title,
            body,
            category_options=list(spec.topics),
            controlled_tags=spec.tags.controlled,
        )
    except Exception as e:
        # Keep the writer's notes (e.g. the budget note) on a failed run.
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
