"""The evidence-based survey writer (docs/redesign.md, 3.2): the survey's
spec and hooks bound to the post engine.

Read (a bounded agent per paper, records checked against the paper) →
outline (each piece of evidence to one section) → draft each section from its
evidence → verify (code checks and a claim-by-claim check) and review (located
rubric issues) → fix only the named sentences → assemble with code-built
tables. Ports only: the model, the papers' text and the evidence cache are
injected.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

from ..domain.evidence import EvidenceLedger, SourceDocument
from ..domain.evidence_draft import free_numbers
from ..domain.specs.survey import REFERENCE_KINDS, SURVEY_SPEC
from ..domain.survey.evidence_post import (
    EvidenceOutline,
    OutlineSection,
    cite_papers,
    comparison_table,
    outline_problems,
    plan_targets,
    repair_outline,
    section_targets,
    table_evidence,
    timeline_table,
    with_canonical_table,
    without_table_evidence,
)
from ..domain.survey.plan import PlanPaper, SurveyPlan
from ..domain.survey.post import without_papers
from ..prompts import survey_evidence as prompts
from .evidence_writer import WriterPrompts, block, evidence_lines, safe_key
from .ports import (
    AgentLimits,
    CheckpointStore,
    DocumentSource,
    LanguageModel,
    TextStore,
)
from .post_engine import (
    PostEngine,
    PostPrompts,
    ReadJob,
    ReadStart,
    RunSettings,
    SectionContext,
)

__all__ = [
    "EngineSettings",
    "EvidenceSurvey",
    "EvidenceSurveyEngine",
]

logger = logging.getLogger(__name__)

# Bump when the reader's prompt or the evidence schema changes meaning; the
# prompt's wording alone does not invalidate cached evidence.
READER_VERSION = 8

SURVEY_PROMPTS = WriterPrompts(
    reader=prompts.READER_SYSTEM,
    writer=prompts.WRITER_SYSTEM,
    claim_check=prompts.CLAIM_CHECK_SYSTEM,
    evidence_check=prompts.EVIDENCE_CHECK_SYSTEM,
    review=prompts.REVIEW_SYSTEM,
    fix=prompts.FIX_SYSTEM,
)


@dataclass(frozen=True)
class EngineSettings:
    language: str
    title_max_length: int
    # Cache-key part naming the reading model.
    reader_tag: str
    max_fix_rounds: int = SURVEY_SPEC.limits.fix_rounds
    read_turns_core: int = 30
    read_turns_supporting: int = 12
    read_tokens_per_paper: int = 600_000
    concurrency: int = 4
    outline_repairs: int = SURVEY_SPEC.limits.outline_repairs
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
    # Problems the post may not be published with as is (the PR lists
    # them first, under Blocking).
    blocking: list[str] = field(default_factory=list)


class EvidenceUnavailable(RuntimeError):
    """The papers a survey needs gave no usable evidence: a failed run (the
    sources or the reader), never a rejected plan."""


class EvidenceSurveyEngine(PostEngine[EvidenceOutline, OutlineSection, EvidenceSurvey]):
    def __init__(
        self,
        model: LanguageModel,
        documents: DocumentSource,
        settings: EngineSettings,
        store: TextStore | None = None,
        *,
        checkpoints: CheckpointStore | None = None,
    ) -> None:
        super().__init__(
            model,
            SURVEY_SPEC,
            SURVEY_PROMPTS,
            PostPrompts(
                outline=prompts.OUTLINE_SYSTEM,
                subject=prompts.PASS_SUBJECT,
                repetition=prompts.REPETITION_EXTRA,
                tail=prompts.PASS_TAIL,
            ),
            RunSettings(
                language=settings.language,
                concurrency=settings.concurrency,
                outline_repairs=settings.outline_repairs,
                min_total_chars=settings.min_total_chars,
                max_total_chars=settings.max_total_chars,
            ),
            outline_schema=EvidenceOutline,
            store=store,
            checkpoints=checkpoints,
            max_fix_rounds=settings.max_fix_rounds,
        )
        self.documents = documents
        self.settings = settings
        self._plan: SurveyPlan  # set by write
        self._references = ""
        self._guideline = ""
        self._kinds: dict[str, str] = {}

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

        def start(paper: PlanPaper) -> ReadStart:
            async def job() -> ReadJob | None:
                document = await self.documents.document(paper)
                if document is None:
                    self.notes.append(
                        f"No text for {paper.id}; left out of the survey."
                    )
                    return None
                return self._read_job(paper, document, plan)

            return job

        results = await self.read_all([start(p) for p in plan.papers])
        ledger = EvidenceLedger()
        kinds: dict[str, str] = {}
        for paper, (job, items) in zip(plan.papers, results, strict=True):
            ledger.extend(items)
            kind = job.document.kind if job is not None else "none"
            kinds[paper.id] = kind
        return ledger, kinds

    def _read_job(
        self, paper: PlanPaper, document: SourceDocument, plan: SurveyPlan
    ) -> ReadJob:
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
        return ReadJob(
            document=document,
            key=self._cache_key(paper, document),
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
        )

    # ---- the survey ---------------------------------------------------------------

    async def write(
        self,
        plan: SurveyPlan,
        *,
        references: str = "",
        guideline: str = "",
    ) -> EvidenceSurvey:
        # Every core paper and lineage, from the plan in code: the readers
        # record their evidence first.
        self.targets = plan_targets(plan)
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
            # The papers left out take their targets with them.
            self.targets = plan_targets(plan)
        self._plan, self._kinds = plan, kinds
        self._references, self._guideline = references, guideline
        self.free = free_numbers(sorted({p.year for p in plan.papers if p.year}))
        return await self.compose(ledger)

    def checkpoint_inputs(self) -> object:
        # The plan decides the sections' lengths, named papers and tables.
        return {
            "settings": self.settings,
            "plan": self._plan,
            "references": self._references,
            "guideline": self._guideline,
            "documents": self._kinds,
        }

    # ---- hooks: outline -----------------------------------------------------------

    def outline_prompt(
        self, ledger: EvidenceLedger, duplicates: Mapping[str, str]
    ) -> str:
        plan = self._plan
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
        return "\n\n".join(
            [
                f"Topic: {plan.topic}\nScope: {plan.scope}\nLanguage: {self.settings.language}",
                f"Title: in English, at most {self.settings.title_max_length} characters.",
                block("lineages", lineages),
                block("papers", papers),
                block("citations", edges),
                block("evidence", evidence_lines(list(ledger))),
                block("reference_material", self._references),
                block("translation_guideline", self._guideline),
            ]
        )

    def repair(
        self, outline: EvidenceOutline, ledger: EvidenceLedger
    ) -> tuple[EvidenceOutline, list[str]]:
        return repair_outline(outline, self._plan, ledger)

    def outline_problems(
        self, outline: EvidenceOutline, ledger: EvidenceLedger
    ) -> list[str]:
        return outline_problems(
            outline,
            self._plan,
            ledger,
            title_max_length=self.settings.title_max_length,
        )

    # The comparison table cites evidence too: its results and limitations
    # are merged, checked and dropped like the sections'.
    def cited_elsewhere(self, outline: EvidenceOutline) -> set[str]:
        return table_evidence(outline)

    def canonical_elsewhere(
        self, outline: EvidenceOutline, duplicates: Mapping[str, str]
    ) -> EvidenceOutline:
        return with_canonical_table(outline, duplicates)

    def drop_elsewhere(
        self, outline: EvidenceOutline, wrong: Collection[str]
    ) -> EvidenceOutline:
        return without_table_evidence(outline, wrong)

    def source_names(self, outline: EvidenceOutline) -> dict[str, str]:
        return dict(outline.short_names)

    def section_targets(self, outline: EvidenceOutline) -> list[int]:
        return section_targets(
            outline,
            self._plan,
            per_core=self.settings.chars_per_core,
            per_supporting=self.settings.chars_per_supporting,
            min_total=self.settings.min_total_chars,
            max_total=self.settings.max_total_chars,
            language=self.settings.language,
        )

    # ---- hooks: sections ----------------------------------------------------------

    def prompt_head(self) -> str:
        return f"Language: {self.settings.language}. Topic: {self._plan.topic}."

    def naming_blocks(
        self, ctx: SectionContext[EvidenceOutline, OutlineSection]
    ) -> list[str]:
        return [
            block(
                "paper_names",
                "\n".join(
                    f"- {p.id}: {ctx.outline.short_names.get(p.id, p.title)} "
                    f"({p.authors}, {p.year})"
                    for p in self._plan.papers
                ),
            )
        ]

    def closing_blocks(
        self, ctx: SectionContext[EvidenceOutline, OutlineSection]
    ) -> list[str]:
        kind = ctx.section.kind
        attached = self._attachment(kind, ctx.outline, ctx.ledger)
        return [
            block(
                "reference_material",
                self._references if kind in REFERENCE_KINDS else "",
            ),
            (
                "The code appends this table after the section; introduce and "
                "interpret it (its columns are exactly these), do not "
                "reproduce it:\n" + attached
                if attached
                else ""
            ),
        ]

    def named_papers(self, section: OutlineSection) -> Sequence[str]:
        # A lineage section cites every paper it covers.
        return section.papers if section.kind == "lineage" else ()

    def finish_section(self, section: OutlineSection, markdown: str) -> str:
        # A survey section's missing heading is added at assembly.
        return markdown

    # ---- assemble ---------------------------------------------------------------

    def _attachment(
        self, kind: str, outline: EvidenceOutline, ledger: EvidenceLedger
    ) -> str:
        """The table code appends after a section of ``kind`` ("" when
        none)."""
        name = self.spec.attachments.get(kind, "")
        if name == "timeline":
            return timeline_table(self._plan, outline, self.settings.korean)
        if name == "comparison":
            return comparison_table(self._plan, outline, ledger, self.settings.korean)
        return ""

    async def assemble(
        self,
        outline: EvidenceOutline,
        ledger: EvidenceLedger,
        written: list[tuple[OutlineSection, str]],
        tldr: str | None,
    ) -> EvidenceSurvey:
        parts = []
        for section, markdown in written:
            if not markdown.lstrip().startswith("## "):
                markdown = f"## {section.title}\n\n{markdown.lstrip()}"
            parts.append(markdown)
            if attached := self._attachment(section.kind, outline, ledger):
                parts.append(attached)
        body = cite_papers("\n\n".join(p.strip() for p in parts if p.strip()), ledger)
        return EvidenceSurvey(
            title=outline.title,
            body=body,
            outline=outline,
            ledger=ledger,
            plan=self._plan,
            documents=self._kinds,
            notes=self.notes,
            blocking=self.blocking,
        )
