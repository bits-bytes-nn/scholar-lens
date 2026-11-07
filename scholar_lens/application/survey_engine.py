"""The evidence-based survey writer (docs/redesign.md, 3.2).

Read (a bounded agent per paper, records checked against the paper) →
outline (each piece of evidence to one section) → draft each section from its
evidence → verify (code checks and a claim-by-claim check) and review (located
rubric issues) → fix only the named sentences → assemble with code-built
tables. Ports only: the model, the papers' text and the evidence cache are
injected.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..domain.evidence import (
    Evidence,
    EvidenceLedger,
    SourceDocument,
)
from ..domain.evidence_draft import (
    cited_ids,
    draft_problems,
    free_numbers,
    prose_sentences,
    section_heading,
    section_numbers,
)
from ..domain.glossary import glossary_view
from ..domain.survey.evidence_post import (
    EvidenceOutline,
    OutlineSection,
    cite_papers,
    comparison_table,
    outline_problems,
    repair_outline,
    section_targets,
    timeline_table,
)
from ..domain.survey.plan import PlanPaper, SurveyPlan
from ..domain.survey.post import without_papers
from ..prompts import survey_evidence as prompts
from .evidence_writer import (
    EvidenceWriter,
    WriterPrompts,
    all_or_none,
    block,
    cached_evidence,
    evidence_lines,
    safe_key,
)
from .ports import AgentLimits, DocumentSource, LanguageModel, TextStore
from .usage import usage_stage

__all__ = [
    "EngineSettings",
    "EvidenceSurvey",
    "EvidenceSurveyEngine",
]

logger = logging.getLogger(__name__)

# Bump when the reader's prompt or the evidence schema changes meaning; the
# prompt's wording alone does not invalidate cached evidence.
READER_VERSION = 7

SURVEY_PROMPTS = WriterPrompts(
    reader=prompts.READER_SYSTEM,
    writer=prompts.WRITER_SYSTEM,
    claim_check=prompts.CLAIM_CHECK_SYSTEM,
    evidence_check=prompts.EVIDENCE_CHECK_SYSTEM,
    review=prompts.REVIEW_SYSTEM,
    fix=prompts.FIX_SYSTEM,
)


def _no_code(document: SourceDocument, kind: str, fields: object) -> list[str]:
    """A survey states what papers find; their code and settings are a
    guide's evidence."""
    if kind in ("code", "setting"):
        return [f"kind {kind!r} is not used in a survey; record a method instead"]
    # Results and limitations fill the comparison table: each needs its
    # short noun-phrase cell (a claim is a sentence, not a cell).
    cell = fields.get("cell", "") if isinstance(fields, Mapping) else ""
    if kind in ("number", "limitation") and not str(cell).strip():
        return [
            f"a {kind} needs `cell`: a short noun phrase for the comparison table "
            "(a result with its benchmark and metric, or the limitation), in the "
            "post's language and register"
        ]
    return []


@dataclass(frozen=True)
class EngineSettings:
    language: str
    title_max_length: int
    # Cache-key part naming the reading model.
    reader_tag: str
    max_fix_rounds: int = 2
    read_turns_core: int = 30
    read_turns_supporting: int = 12
    read_tokens_per_paper: int = 600_000
    concurrency: int = 4
    outline_repairs: int = 2
    # Prose characters (Korean): per paper of a lineage section, and the
    # post's total range (survey config).
    chars_per_core: int = 3_000
    chars_per_supporting: int = 800
    min_total_chars: int = 20_000
    max_total_chars: int = 70_000

    @property
    def korean(self) -> bool:
        return self.language == "Korean"


@dataclass
class EvidenceSurvey:
    title: str
    # Markdown with ``[@paper id]`` citation keys and the code-built tables.
    body: str
    outline: EvidenceOutline
    ledger: EvidenceLedger
    plan: SurveyPlan
    documents: dict[str, str] = field(default_factory=dict)  # paper id -> kind
    notes: list[str] = field(default_factory=list)


class EvidenceUnavailable(RuntimeError):
    """The papers a survey needs gave no usable evidence: a failed run (the
    sources or the reader), never a rejected plan."""


class EvidenceSurveyEngine(EvidenceWriter):
    def __init__(
        self,
        model: LanguageModel,
        documents: DocumentSource,
        settings: EngineSettings,
        store: TextStore | None = None,
    ) -> None:
        super().__init__(model, SURVEY_PROMPTS, max_fix_rounds=settings.max_fix_rounds)
        self.documents = documents
        self.settings = settings
        self.store = store

    # ---- read -------------------------------------------------------------------

    def _cache_key(self, paper: PlanPaper, document: SourceDocument) -> str:
        pid = safe_key(paper.id)
        text = hashlib.sha256(document.text.encode()).hexdigest()[:12]
        return (
            f"evidence/{pid}/{text}-v{READER_VERSION}-{self.settings.reader_tag}-"
            f"{safe_key(self.settings.language)}-{paper.role}-"
            f"{document.kind}.json"
        )

    async def read(self, plan: SurveyPlan) -> tuple[EvidenceLedger, dict[str, str]]:
        """Every paper's evidence (cached per paper, reader version, model,
        language, role and document kind), in plan order."""
        gate = asyncio.Semaphore(self.settings.concurrency)

        async def one(paper: PlanPaper) -> tuple[list[Evidence], str]:
            async with gate:
                document = await self.documents.document(paper)
                if document is None:
                    self.notes.append(
                        f"No text for {paper.id}; left out of the survey."
                    )
                    return [], "none"
                key = self._cache_key(paper, document)
                if self.store is not None and (cached := self.store.get(key)):
                    kept = cached_evidence(cached, document, key=key)
                    if kept is not None:
                        return kept, document.kind
                items = await self._read_paper(paper, document, plan)
                if self.store is not None and items:
                    self.store.put(
                        key,
                        json.dumps([e.model_dump() for e in items], ensure_ascii=False),
                    )
                return items, document.kind

        with usage_stage("read"):
            results = await all_or_none(one(p) for p in plan.papers)
        ledger = EvidenceLedger()
        kinds: dict[str, str] = {}
        for paper, (items, kind) in zip(plan.papers, results, strict=True):
            ledger.extend(items)
            kinds[paper.id] = kind
            if paper.role == "core" and kind == "abstract":
                self.notes.append(
                    f"Core paper {paper.id} was read from its abstract only."
                )
        return ledger, kinds

    async def _read_paper(
        self, paper: PlanPaper, document: SourceDocument, plan: SurveyPlan
    ) -> list[Evidence]:
        core = paper.role == "core"
        required: tuple[str, ...] = ()
        if document.kind != "abstract":
            required = ("method", "number", "limitation") if core else ("limitation",)
        prompt = "\n\n".join(
            [
                f"Survey topic: {plan.topic}\nScope: {plan.scope}",
                f"Paper: {paper.title} ({paper.authors}, {paper.year}); role: "
                f"{paper.role}; lineage: {paper.lineage}; why it is in the survey: {paper.reason}",
                f"Text available: {document.kind.replace('_', ' ')}.",
                block("outline", document.outline()),
                f"Survey language for claims and cells: {self.settings.language}.",
                (
                    "Record about 12-25 pieces of evidence."
                    if core
                    else "Record about 4-8 pieces of evidence."
                ),
            ]
        )
        return await self.read_document(
            document,
            prompt=prompt,
            limits=AgentLimits(
                turns=(
                    self.settings.read_turns_core
                    if core
                    else self.settings.read_turns_supporting
                ),
                total_tokens=self.settings.read_tokens_per_paper,
            ),
            required=required,
            rule=_no_code,
        )

    # ---- outline ----------------------------------------------------------------

    async def outline(
        self, plan: SurveyPlan, ledger: EvidenceLedger, references: str, guideline: str
    ) -> EvidenceOutline:
        papers = "\n".join(
            f"- {p.id} | {p.year} | {p.role} | lineage {p.lineage} | {p.title}"
            for p in plan.papers
        )
        lineages = "\n".join(
            f"- {lin.id}: {lin.name}: {lin.description}" for lin in plan.lineages
        )
        edges = (
            "\n".join(f"- {e.citing} cites {e.cited}" for e in plan.edges)
            if plan.edges
            else ""
        )
        base = "\n\n".join(
            [
                f"Topic: {plan.topic}\nScope: {plan.scope}\nLanguage: {self.settings.language}",
                f"Title: in English, at most {self.settings.title_max_length} characters.",
                block("lineages", lineages),
                block("papers", papers),
                block("citations", edges),
                block("evidence", evidence_lines(list(ledger))),
                block("reference_material", references),
                block("translation_guideline", guideline),
            ]
        )
        prompt = base
        with usage_stage("outline"):
            for attempt in range(self.settings.outline_repairs + 1):
                outline = await self.model.structured(
                    role="outline",
                    system=prompts.OUTLINE_SYSTEM,
                    prompt=prompt,
                    schema=EvidenceOutline,
                )
                outline, repaired = repair_outline(outline, plan, ledger)
                if repaired:
                    logger.info("Outline repaired in code: %s", "; ".join(repaired))
                problems = outline_problems(
                    outline,
                    plan,
                    ledger,
                    title_max_length=self.settings.title_max_length,
                )
                if not problems:
                    return outline
                logger.warning("Outline attempt %d unusable: %s", attempt + 1, problems)
                prompt = (
                    base
                    + "\n\n"
                    + block(
                        "previous_problems",
                        "Fix these in your next outline:\n"
                        + "\n".join(f"- {p}" for p in problems),
                    )
                )
        raise ValueError(
            "The outline breaks the rules after repairs: " + "; ".join(problems)
        )

    # ---- sections ---------------------------------------------------------------

    async def write(
        self,
        plan: SurveyPlan,
        *,
        references: str = "",
        guideline: str = "",
    ) -> EvidenceSurvey:
        ledger, kinds = await self.read(plan)
        missing = [p.id for p in plan.papers if not ledger.of_paper(p.id)]
        # Not the plan's fault: the papers' text could not be read or gave no
        # usable evidence. A failure (not a rejection) that names them.
        if len(missing) == len(plan.papers):
            raise EvidenceUnavailable(
                "No paper of the plan gave usable evidence (text unreachable or "
                "every record rejected): " + ", ".join(missing)
            )
        if not any(p.role == "core" and p.id not in missing for p in plan.papers):
            raise EvidenceUnavailable(
                "No core paper gave usable evidence: "
                + ", ".join(p.id for p in plan.papers if p.role == "core")
            )
        if missing:
            self.notes.append(
                "Papers without evidence (left out): " + ", ".join(missing)
            )
            plan = without_papers(plan, missing)
        outline = await self.outline(plan, ledger, references, guideline)
        outline = await self._check_evidence(outline, ledger)
        self._names = dict(outline.short_names)
        glossary = {t.english: t.rendering for t in outline.glossary}
        free = free_numbers(sorted({p.year for p in plan.papers if p.year}))
        written: list[tuple[OutlineSection, str]] = []
        numbers: dict[str, str] = {}
        targets = section_targets(
            outline,
            plan,
            per_core=self.settings.chars_per_core,
            per_supporting=self.settings.chars_per_supporting,
            min_total=self.settings.min_total_chars,
            max_total=self.settings.max_total_chars,
            language=self.settings.language,
        )
        for section, target in zip(outline.sections, targets, strict=True):
            markdown = await self._section(
                plan,
                outline,
                ledger,
                section,
                written,
                numbers,
                glossary,
                references,
                free,
                target,
            )
            heading = section_heading(markdown) or section.title
            for n in section_numbers(markdown, free):
                numbers.setdefault(n, heading)
            written.append((section, markdown))
        body = self._assemble(plan, outline, ledger, written)
        return EvidenceSurvey(
            title=outline.title,
            body=body,
            outline=outline,
            ledger=ledger,
            plan=plan,
            documents=kinds,
            notes=self.notes,
        )

    async def _section(
        self,
        plan: SurveyPlan,
        outline: EvidenceOutline,
        ledger: EvidenceLedger,
        section: OutlineSection,
        written: list[tuple[OutlineSection, str]],
        numbers: dict[str, str],
        glossary: dict[str, str],
        references: str,
        free: str,
        target: int,
    ) -> str:
        own = [e for eid in section.evidence if (e := ledger.get(eid))]
        # What earlier sections actually state (their final text), so a
        # back-reference never points at a removed sentence.
        earlier_ids = list(
            dict.fromkeys(
                eid
                for _, md in written
                for s in prose_sentences(md)
                for eid in cited_ids(s)
            )
        )
        earlier = [e for eid in earlier_ids if (e := ledger.get(eid))]
        # Evidence a later section already referred back to once.
        cites: dict[str, int] = {}
        for s, md in written:
            for eid in {i for snt in prose_sentences(md) for i in cited_ids(snt)}:
                if eid not in s.evidence:
                    cites[eid] = cites.get(eid, 0) + 1
        exhausted = set(cites) - set(section.evidence)
        figures = [e.figure_url for e in own if e.kind == "figure" and e.figure_url]
        sections_map = "\n".join(
            f"- {'(this) ' if s is section else ''}{s.kind}: {s.title} — {s.thesis}"
            for s in outline.sections
        )
        previous = written[-1][1][-2500:] if written else ""
        prompt = "\n\n".join(
            [
                f"Language: {self.settings.language}. Topic: {plan.topic}.",
                block("outline", sections_map),
                block(
                    "section",
                    f"kind: {section.kind}\ntitle: {section.title}\nthesis: {section.thesis}\n"
                    f"question: {section.question}\ndepth: {section.depth}\n"
                    f"length: about {target:,} characters of prose\n"
                    f"running example: {outline.running_example or 'none'}",
                ),
                block("evidence", evidence_lines(own)),
                block("already_written_evidence", evidence_lines(earlier[-60:])),
                block(
                    "paper_names",
                    "\n".join(
                        f"- {p.id}: {outline.short_names.get(p.id, p.title)} "
                        f"({p.authors}, {p.year})"
                        for p in plan.papers
                    ),
                ),
                block("glossary", glossary_view(glossary, [m for _, m in written])),
                block("previous_section_end", previous),
                block(
                    "reference_material",
                    (
                        references
                        if section.kind in ("introduction", "cross_cutting", "insights")
                        else ""
                    ),
                ),
                (
                    "The code appends this table after the section; introduce and "
                    "interpret it (its columns are exactly these), do not "
                    "reproduce it:\n"
                    + (
                        timeline_table(plan, outline, self.settings.korean)
                        if section.kind == "introduction"
                        else comparison_table(
                            plan, outline, ledger, self.settings.korean
                        )
                    )
                    if section.kind in ("introduction", "comparison")
                    else ""
                ),
            ]
        )
        draft = await self.draft(
            prompt,
            own,
            target,
            retry_hint="This is a survey of published research papers for "
            "practitioners.",
        )

        def checks(markdown: str) -> list[str]:
            return draft_problems(
                markdown,
                title=section.title,
                own=section.evidence,
                earlier=earlier_ids,
                ledger=ledger,
                earlier_numbers=numbers,
                figure_urls=figures,
                free_numbers=free,
                exhausted=exhausted,
                papers=section.papers if section.kind == "lineage" else (),
                target_chars=target,
                depth=(
                    section.depth
                    if section.kind in ("lineage", "cross_cutting")
                    else ""
                ),
            )

        return await self.verify_and_fix(
            draft,
            title=section.title,
            review_context=f"Section kind: {section.kind}; thesis: {section.thesis}; "
            f"depth: {section.depth}.",
            checks=checks,
            ledger=ledger,
            own=own,
            earlier=earlier,
            free=free,
        )

    async def _check_evidence(
        self, outline: EvidenceOutline, ledger: EvidenceLedger
    ) -> EvidenceOutline:
        """Check the claim, scope and cell of every number and limitation the
        outline uses against its quote (a scope can name the wrong benchmark
        while every number is in the quote); wrong ones leave the outline."""
        used = {eid for s in outline.sections for eid in s.evidence} | {
            eid for r in outline.table for eid in [*r.results, r.limitation] if eid
        }
        wrong = await self.wrong_evidence([e for e in ledger if e.id in used])
        if not wrong:
            return outline
        sections = [
            s.model_copy(update={"evidence": [e for e in s.evidence if e not in wrong]})
            for s in outline.sections
        ]
        table = [
            r.model_copy(
                update={
                    "results": [e for e in r.results if e not in wrong],
                    "limitation": None if r.limitation in wrong else r.limitation,
                }
            )
            for r in outline.table
        ]
        return outline.model_copy(update={"sections": sections, "table": table})

    # ---- assemble ---------------------------------------------------------------

    def _assemble(
        self,
        plan: SurveyPlan,
        outline: EvidenceOutline,
        ledger: EvidenceLedger,
        written: list[tuple[OutlineSection, str]],
    ) -> str:
        parts = []
        for section, markdown in written:
            if not markdown.lstrip().startswith("## "):
                markdown = f"## {section.title}\n\n{markdown.lstrip()}"
            parts.append(markdown)
            if section.kind == "introduction":
                parts.append(timeline_table(plan, outline, self.settings.korean))
            elif section.kind == "comparison":
                parts.append(
                    comparison_table(plan, outline, ledger, self.settings.korean)
                )
        return cite_papers("\n\n".join(p.strip() for p in parts if p.strip()), ledger)
