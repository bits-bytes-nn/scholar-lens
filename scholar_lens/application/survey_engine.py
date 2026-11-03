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
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ..domain.evidence import (
    Evidence,
    EvidenceLedger,
    SourceDocument,
    evidence_problems,
)
from ..domain.glossary import glossary_view
from ..domain.lengths import DEPTH_CHARS, NON_KOREAN_CHAR_FACTOR
from ..domain.survey.evidence_post import (
    EvidenceOutline,
    OutlineSection,
    cite_papers,
    cited_ids,
    comparison_table,
    draft_problems,
    drop_sentences,
    outline_problems,
    prose_sentences,
    section_heading,
    section_numbers,
    structure_lost,
    timeline_table,
)
from ..domain.survey.plan import PlanPaper, SurveyPlan
from ..domain.survey.post import without_papers
from ..prompts import survey_evidence as prompts
from .ports import AgentLimits, DocumentSource, LanguageModel, TextStore
from .usage import usage_stage

logger = logging.getLogger(__name__)

# Bump when the reader's prompt or the evidence schema changes meaning; the
# prompt's wording alone does not invalidate cached evidence.
READER_VERSION = 1
_SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")


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


class SentenceVerdict(BaseModel):
    n: int
    verdict: Literal["supported", "unsupported", "wrong"]
    reason: str = ""


class ClaimCheck(BaseModel):
    verdicts: list[SentenceVerdict] = Field(default_factory=list)


class ReviewIssue(BaseModel):
    sentence: str
    category: Literal["depth", "flow", "kindness", "style"]
    fix: str


class SectionReview(BaseModel):
    issues: list[ReviewIssue] = Field(default_factory=list, max_length=10)


def _prose_length(markdown: str) -> int:
    return len(" ".join(prose_sentences(markdown)))


def _block(tag: str, text: str) -> str:
    return f"<{tag}>\n{text.strip() or '(none)'}\n</{tag}>"


def _evidence_lines(items: Sequence[Evidence]) -> str:
    lines = []
    for e in items:
        extra = []
        if e.scope:
            extra.append(f"scope: {e.scope}")
        if e.latex:
            extra.append(f"latex: $${e.latex}$$")
        if e.figure_url:
            extra.append(f"figure: {e.figure_url} ({(e.caption or '')[:200]})")
        lines.append(
            f"[{e.id}] ({e.paper_id}, {e.kind}) {e.claim}"
            + (f" | {'; '.join(extra)}" if extra else "")
            + f'\n    quote: "{e.quote[:500]}"'
        )
    return "\n".join(lines)


class EvidenceSurveyEngine:
    def __init__(
        self,
        model: LanguageModel,
        documents: DocumentSource,
        settings: EngineSettings,
        store: TextStore | None = None,
    ) -> None:
        self.model = model
        self.documents = documents
        self.settings = settings
        self.store = store
        self.notes: list[str] = []

    # ---- read -------------------------------------------------------------------

    def _cache_key(self, paper: PlanPaper, kind: str) -> str:
        pid = _SAFE_KEY.sub("_", paper.id)
        return (
            f"evidence/{pid}/v{READER_VERSION}-{self.settings.reader_tag}-"
            f"{_SAFE_KEY.sub('_', self.settings.language)}-{paper.role}-{kind}.json"
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
                key = self._cache_key(paper, document.kind)
                if self.store is not None and (cached := self.store.get(key)):
                    try:
                        return [
                            Evidence.model_validate(e) for e in json.loads(cached)
                        ], document.kind
                    except (ValueError, ValidationError):
                        logger.warning(
                            "Unreadable cached evidence %s; reading again.", key
                        )
                items = await self._read_paper(paper, document, plan)
                if self.store is not None and items:
                    self.store.put(
                        key,
                        json.dumps([e.model_dump() for e in items], ensure_ascii=False),
                    )
                return items, document.kind

        with usage_stage("read"):
            results = await asyncio.gather(*(one(p) for p in plan.papers))
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
        ledger = EvidenceLedger()
        rejected = 0

        def read_section(anchor: str) -> str:
            """Return the full text of one section of the paper by its anchor
            (S1, S2, ... from the outline)."""
            section = document.section(anchor)
            if section is None:
                return f"Unknown anchor {anchor!r}. Anchors: " + ", ".join(
                    s.anchor for s in document.sections
                )
            return f"{section.anchor} | {section.heading}\n\n{section.text[:30_000]}"

        def search_paper(query: str) -> str:
            """Find passages of the paper containing every word of the query;
            returns anchor and snippet per match."""
            hits = document.search(query)
            return "\n\n".join(f"[{a}] {s}" for a, s in hits) or "No passage matches."

        def record_evidence(
            kind: str,
            claim: str,
            quote: str,
            anchor: str,
            scope: str = "",
            cell: str = "",
            figure_id: str = "",
            latex: str = "",
        ) -> str:
            """Record one piece of evidence. kind: finding, number, method,
            definition, limitation, relation, equation or figure. claim: what
            the post may state (survey language). quote: the paper's words,
            verbatim from the section at anchor. scope (numbers): model,
            benchmark, setting, metric. cell: short noun phrase for a table.
            figure_id: F-id for a figure. latex: the equation as in the paper."""
            nonlocal rejected
            problems = evidence_problems(
                document,
                kind=kind,
                claim=claim,
                quote=quote,
                anchor=anchor,
                scope=scope,
                cell=cell,
                figure_id=figure_id or None,
                latex=latex or None,
            )
            if kind not in Evidence.model_fields["kind"].annotation.__args__:  # type: ignore[union-attr]
                problems.insert(0, f"unknown kind {kind!r}")
            if problems:
                rejected += 1
                return "REJECTED: " + "; ".join(problems)
            item = ledger.add(
                document,
                kind=kind,
                claim=claim.strip(),
                quote=quote.strip(),
                anchor=anchor.strip(),
                scope=scope.strip(),
                cell=cell.strip(),
                figure_id=figure_id.strip() or None,
                latex=latex.strip() or None,
            )
            return f"recorded {item.id}"

        def finish(summary: str) -> str:
            """Call when the evidence covers the paper as the survey needs."""
            return "done"

        core = paper.role == "core"
        prompt = "\n\n".join(
            [
                f"Survey topic: {plan.topic}\nScope: {plan.scope}",
                f"Paper: {paper.title} ({paper.authors}, {paper.year}); role: "
                f"{paper.role}; lineage: {paper.lineage}; why it is in the survey: {paper.reason}",
                f"Text available: {document.kind.replace('_', ' ')}.",
                _block("outline", document.outline()),
                f"Survey language for claims and cells: {self.settings.language}.",
                (
                    "Record about 12-25 pieces of evidence."
                    if core
                    else "Record about 4-8 pieces of evidence."
                ),
            ]
        )
        outcome = await self.model.run_agent(
            role="read",
            system=prompts.READER_SYSTEM,
            prompt=prompt,
            tools=[read_section, search_paper, record_evidence, finish],
            limits=AgentLimits(
                turns=(
                    self.settings.read_turns_core
                    if core
                    else self.settings.read_turns_supporting
                ),
                total_tokens=self.settings.read_tokens_per_paper,
            ),
        )
        logger.info(
            "Read %s: %d evidence, %d rejected records (%s).",
            paper.id,
            len(ledger),
            rejected,
            outcome.stop_reason,
        )
        if outcome.stop_reason.startswith("limit"):
            self.notes.append(
                f"Reading {paper.id} stopped at its limit ({outcome.stop_reason})."
            )
        return list(ledger)

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
                _block("lineages", lineages),
                _block("papers", papers),
                _block("citations", edges),
                _block("evidence", _evidence_lines(list(ledger))),
                _block("reference_material", references),
                _block("translation_guideline", guideline),
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
                    + _block(
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
        if missing:
            self.notes.append(
                "Papers without evidence (left out): " + ", ".join(missing)
            )
            plan = without_papers(plan, missing)
        outline = await self.outline(plan, ledger, references, guideline)
        glossary = {t.english: t.rendering for t in outline.glossary}
        free = (
            " ".join({str(p.year) for p in plan.papers if p.year})
            + " 0 1 2 3 4 5 6 7 8 9 10"
        )
        written: list[tuple[OutlineSection, str]] = []
        numbers: dict[str, str] = {}
        for section in outline.sections:
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
    ) -> str:
        own = [e for eid in section.evidence if (e := ledger.get(eid))]
        earlier_ids = [eid for s, _ in written for eid in s.evidence]
        earlier = [e for eid in earlier_ids if (e := ledger.get(eid))]
        figures = [e.figure_url for e in own if e.kind == "figure" and e.figure_url]
        sections_map = "\n".join(
            f"- {'(this) ' if s is section else ''}{s.kind}: {s.title} — {s.thesis}"
            for s in outline.sections
        )
        previous = written[-1][1][-2500:] if written else ""
        prompt = "\n\n".join(
            [
                f"Language: {self.settings.language}. Topic: {plan.topic}.",
                _block("outline", sections_map),
                _block(
                    "section",
                    f"kind: {section.kind}\ntitle: {section.title}\nthesis: {section.thesis}\n"
                    f"question: {section.question}\ndepth: {section.depth}\n"
                    f"running example: {outline.running_example or 'none'}",
                ),
                _block("evidence", _evidence_lines(own)),
                _block("already_written_evidence", _evidence_lines(earlier[-60:])),
                _block(
                    "paper_names",
                    "\n".join(
                        f"- {p.id}: {outline.short_names.get(p.id, p.title)} "
                        f"({p.authors}, {p.year})"
                        for p in plan.papers
                    ),
                ),
                _block("glossary", glossary_view(glossary, [m for _, m in written])),
                _block("previous_section_end", previous),
                _block(
                    "reference_material",
                    (
                        references
                        if section.kind in ("introduction", "cross_cutting", "insights")
                        else ""
                    ),
                ),
                (
                    "The code appends the timeline table after the introduction and the "
                    "comparison table after the comparison section; introduce and "
                    "interpret them, do not reproduce them."
                    if section.kind in ("introduction", "comparison")
                    else ""
                ),
            ]
        )
        target = DEPTH_CHARS[section.depth] * (
            1 if self.settings.korean else NON_KOREAN_CHAR_FACTOR
        )
        with usage_stage("draft"):
            draft = (
                await self.model.text(
                    role="write", system=prompts.WRITER_SYSTEM, prompt=prompt
                )
            ).strip()
            if own and _prose_length(draft) < 0.3 * target:
                logger.warning(
                    "Section %r came back short (%d chars): %r",
                    section.title,
                    _prose_length(draft),
                    draft[:300],
                )
                draft = (
                    await self.model.text(
                        role="write",
                        system=prompts.WRITER_SYSTEM,
                        prompt=prompt
                        + "\n\nYour previous answer was too short or empty. This is "
                        "a survey of published research papers for practitioners: "
                        f"write the whole section (about {int(target):,} characters "
                        "of prose) from its evidence.",
                    )
                ).strip()

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
                depth=(
                    section.depth
                    if section.kind in ("lineage", "cross_cutting")
                    else ""
                ),
            )

        for round_ in range(self.settings.max_fix_rounds + 1):
            code = checks(draft)
            claims = await self._check_claims(draft, ledger)
            review = await self._review(draft, section) if round_ == 0 else []
            issues = code + [f"{v}: {s}" for s, v in claims] + review
            if not issues or round_ == self.settings.max_fix_rounds:
                break
            with usage_stage("fix"):
                fixed = (
                    await self.model.text(
                        role="write",
                        system=prompts.FIX_SYSTEM,
                        prompt="\n\n".join(
                            [
                                _block("section", draft),
                                _block(
                                    "evidence", _evidence_lines(own + earlier[-30:])
                                ),
                                _block("problems", "\n".join(f"- {i}" for i in issues)),
                            ]
                        ),
                    )
                ).strip()
            if reason := structure_lost(draft, fixed):
                logger.info("Fix of %r rejected: %s", section.title, reason)
                break
            if len(checks(fixed)) > len(code):
                logger.info("Fix of %r rejected: adds rule problems", section.title)
                break
            draft = fixed
        claims = await self._check_claims(draft, ledger)
        if claims:
            draft = drop_sentences(draft, [s for s, _ in claims])
            self.notes.append(
                f"Section '{section.title}': {len(claims)} unsupported sentence(s) removed: "
                + " | ".join(f"{s[:120]} ({v})" for s, v in claims[:5])
            )
        if leftover := checks(draft):
            self.notes.append(
                f"Section '{section.title}' still has: " + " | ".join(leftover[:5])
            )
        return draft

    async def _check_claims(
        self, draft: str, ledger: EvidenceLedger
    ) -> list[tuple[str, str]]:
        """(sentence, verdict with reason) of the sentences the evidence does
        not support."""
        sentences = prose_sentences(draft)
        if not sentences:
            return []
        lines = []
        for n, sentence in enumerate(sentences, 1):
            cited = [e for i in cited_ids(sentence) if (e := ledger.get(i))]
            lines.append(
                f"{n}. {sentence}\n   evidence: "
                + (
                    " || ".join(
                        f'[{e.id}] {e.claim} — "{e.quote[:300]}" {e.scope}'
                        for e in cited
                    )
                    or "(none)"
                )
            )
        with usage_stage("verify"):
            check = await self.model.structured(
                role="evaluate",
                system=prompts.CLAIM_CHECK_SYSTEM,
                prompt=_block("sentences", "\n".join(lines)),
                schema=ClaimCheck,
            )
        bad = []
        for v in check.verdicts:
            if v.verdict != "supported" and 1 <= v.n <= len(sentences):
                bad.append((sentences[v.n - 1], f"{v.verdict}: {v.reason}"))
        return bad

    async def _review(self, draft: str, section: OutlineSection) -> list[str]:
        with usage_stage("review"):
            review = await self.model.structured(
                role="evaluate",
                system=prompts.REVIEW_SYSTEM,
                prompt="\n\n".join(
                    [
                        f"Section kind: {section.kind}; thesis: {section.thesis}; depth: {section.depth}.",
                        _block("section", draft),
                    ]
                ),
                schema=SectionReview,
            )
        return [
            f"{i.category}: {i.fix} — in: {i.sentence[:200]}" for i in review.issues
        ]

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
            parts.append(markdown)
            if section.kind == "introduction":
                parts.append(timeline_table(plan, outline, self.settings.korean))
            elif section.kind == "comparison":
                parts.append(
                    comparison_table(plan, outline, ledger, self.settings.korean)
                )
        missing = [s.title for s, m in written if not m.lstrip().startswith("## ")]
        if missing:
            self.notes.append("Sections without their heading: " + ", ".join(missing))
        return cite_papers("\n\n".join(p.strip() for p in parts if p.strip()), ledger)
