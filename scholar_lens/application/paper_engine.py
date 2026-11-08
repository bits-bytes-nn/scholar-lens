"""The evidence-based paper review and summary writer (docs/redesign.md, 3.2,
5.2): the paper post's spec and hooks bound to the post engine.

Read (bounded agents over groups of the paper's sections, records checked
against the paper) → outline (each piece of evidence to one section, in the
post's order) → draft each section from its evidence and the paper passages
it comes from → verify and review → fix only the named sentences → the
TL;DR last → subtitle and description. Ports only: the model and the
evidence cache are injected.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from ..domain.evidence import (
    Evidence,
    EvidenceLedger,
    SourceDocument,
    duplicate_evidence,
)
from ..domain.evidence_draft import free_numbers
from ..domain.guide.evidence_guide import strip_markers
from ..domain.paper.evidence_paper import (
    PaperOutline,
    PaperSection,
    PostKind,
    paper_outline_problems,
    repair_paper_outline,
    tldr_problems,
    tldr_questions,
)
from ..domain.specs.paper import PAPER_REVIEW_SPEC, paper_spec
from ..domain.validation import description_problems, fit_description
from ..prompts import paper_evidence as prompts
from .checkpoints import input_hash
from .evidence_writer import (
    WriterPrompts,
    block,
    brief_lines,
    quote_for_prompt_code,
    safe_key,
)
from .ports import AgentLimits, CheckpointStore, LanguageModel, TextStore
from .post_engine import PostEngine, PostPrompts, ReadJob, ReadStart, RunSettings
from .usage import usage_stage

logger = logging.getLogger(__name__)

# Bump when the reader's prompt or the evidence schema changes meaning.
PAPER_READER_VERSION = 3
_FREE_NUMBERS = free_numbers()

PAPER_PROMPTS = WriterPrompts(
    reader=prompts.READER_SYSTEM,
    writer=prompts.WRITER_SYSTEM,
    claim_check=prompts.CLAIM_CHECK_SYSTEM,
    evidence_check=prompts.EVIDENCE_CHECK_SYSTEM,
    review=prompts.REVIEW_SYSTEM,
    fix=prompts.FIX_SYSTEM,
)
# A review and a summary share every limit default.
_LIMITS = PAPER_REVIEW_SPEC.limits


@dataclass(frozen=True)
class PaperSettings:
    language: str
    post: PostKind
    # Cache-key part naming the reading model.
    reader_tag: str
    max_sections: int = 12
    max_fix_rounds: int = _LIMITS.fix_rounds
    # Paper characters one reading agent covers, its turns and tokens.
    chars_per_reader: int = 15_000
    read_turns: int = 24
    read_tokens: int = 500_000
    concurrency: int = 4
    outline_repairs: int = _LIMITS.outline_repairs
    # Source text handed to one section's writer and checks.
    source_chars: int = 40_000
    # Prose characters (Korean) of the whole post.
    min_total_chars: int = 25_000
    max_total_chars: int = 45_000

    @property
    def korean(self) -> bool:
        return self.language == "Korean"


class FinalText(BaseModel):
    subtitle: str
    description: str


@dataclass
class EvidencePaper:
    outline: PaperOutline
    ledger: EvidenceLedger
    # (kind, Markdown without evidence markers), in the post's order, the
    # TL;DR first.
    sections: list[tuple[str, str]]
    subtitle: str = ""
    description: str = ""
    notes: list[str] = field(default_factory=list)
    # Problems the post may not be published with as is (the PR lists
    # them first, under Blocking).
    blocking: list[str] = field(default_factory=list)


def reading_groups(document: SourceDocument, chars: int) -> list[SourceDocument]:
    """The paper's sections in consecutive groups of about ``chars``
    characters, each as a document of its own (same id, its sections and
    the figures placed in them; figures placed nowhere go to the first)."""
    groups: list[list[str]] = [[]]
    size = 0
    for s in document.sections:
        if groups[-1] and size + len(s.text) > chars:
            groups.append([])
            size = 0
        groups[-1].append(s.anchor)
        size += len(s.text)
    out = []
    for n, anchors in enumerate(groups):
        wanted = set(anchors)
        out.append(
            document.model_copy(
                update={
                    "sections": [s for s in document.sections if s.anchor in wanted],
                    "figures": [
                        f
                        for f in document.figures
                        if f.anchor in wanted or (n == 0 and not f.anchor)
                    ],
                }
            )
        )
    return [g for g in out if g.sections]


class EvidencePaperEngine(PostEngine[PaperOutline, PaperSection, EvidencePaper]):
    def __init__(
        self,
        model: LanguageModel,
        settings: PaperSettings,
        store: TextStore | None = None,
        *,
        checkpoints: CheckpointStore | None = None,
    ) -> None:
        review = settings.post == "review"
        super().__init__(
            model,
            paper_spec(settings.post),
            PAPER_PROMPTS,
            PostPrompts(
                outline=prompts.OUTLINE_REVIEW if review else prompts.OUTLINE_SUMMARY,
                subject=prompts.PASS_SUBJECT,
                consistency=prompts.CONSISTENCY_EXTRA,
                tail=prompts.PASS_TAIL,
            ),
            RunSettings(
                language=settings.language,
                concurrency=settings.concurrency,
                outline_repairs=settings.outline_repairs,
                min_total_chars=settings.min_total_chars,
                max_total_chars=settings.max_total_chars,
            ),
            outline_schema=PaperOutline,
            store=store,
            checkpoints=checkpoints,
            max_fix_rounds=settings.max_fix_rounds,
        )
        self.settings = settings
        self.free = _FREE_NUMBERS
        self._documents: dict[str, SourceDocument] = {}
        self._title = ""
        self._abstract = ""
        self._guideline = ""
        # A digest of the documents read (set by write; checkpoint input).
        self._sources = ""

    # ---- read -------------------------------------------------------------------

    def _cache_key(self, document: SourceDocument, part: int) -> str:
        pid = safe_key(document.paper_id)
        text = hashlib.sha256(document.text.encode()).hexdigest()[:12]
        return (
            f"evidence/{pid}/{text}-p{part}-v{PAPER_READER_VERSION}-"
            f"{self.settings.reader_tag}-{safe_key(self.settings.language)}-"
            f"{self.settings.post}.json"
        )

    async def read(
        self, title: str, paper: SourceDocument, extra: Sequence[SourceDocument]
    ) -> EvidenceLedger:
        """The paper's evidence, a reading agent per group of sections, then
        the extra documents' (an official code summary); cached per group."""
        groups = reading_groups(paper, self.settings.chars_per_reader)
        jobs = [(paper, g, n) for n, g in enumerate(groups)] + [
            (d, d, 0) for d in extra
        ]

        def start(whole: SourceDocument, part: SourceDocument, n: int) -> ReadStart:
            async def job() -> ReadJob:
                anchors = ", ".join(s.anchor for s in part.sections)
                prompt = "\n\n".join(
                    [
                        f"Paper: {title}",
                        block("paper_outline", whole.outline()),
                        f"Your sections: {anchors}. Record evidence from them only.",
                        f"Post language for claims and cells: {self.settings.language}.",
                        (
                            "Record about 10-25 pieces of evidence."
                            if self.settings.post == "review"
                            else "Record about 6-12 pieces of evidence: what a "
                            "five-minute summary needs."
                        ),
                    ]
                )
                return ReadJob(
                    document=part,
                    key=self._cache_key(part, n),
                    prompt=prompt,
                    limits=AgentLimits(
                        turns=self.settings.read_turns,
                        total_tokens=self.settings.read_tokens,
                    ),
                )

            return job

        results = await self.read_all([start(w, p, n) for w, p, n in jobs])
        return self.ledger_of(items for _, items in results)

    # ---- the post -----------------------------------------------------------------

    async def write(
        self,
        title: str,
        paper: SourceDocument,
        *,
        extra: Sequence[SourceDocument] = (),
        abstract: str = "",
        guideline: str = "",
    ) -> EvidencePaper:
        self._title, self._abstract, self._guideline = title, abstract, guideline
        self._documents = {d.paper_id: d for d in (paper, *extra)}
        self._sources = input_hash([d.model_dump(mode="json") for d in (paper, *extra)])
        self.passages = {
            (d.paper_id, s.anchor): s.text for d in (paper, *extra) for s in d.sections
        }
        self._names = {d.paper_id: d.title for d in (paper, *extra)}
        # Key claims of the abstract and introduction: the readers record
        # their evidence first.
        self.targets = await self.key_targets([paper])
        ledger = await self.read(title, paper, extra)
        if not len(ledger):
            raise ValueError("The paper gave no usable evidence.")
        return await self.compose(ledger)

    def source_sections(self, own: Sequence[Evidence]) -> str:
        """The full paper sections the evidence comes from, up to
        ``source_chars``: equations, tables and context the quotes cut."""
        seen: list[tuple[str, str]] = []
        for e in own:
            if (e.paper_id, e.anchor) not in seen:
                seen.append((e.paper_id, e.anchor))
        parts: list[str] = []
        left = self.settings.source_chars
        for sid, anchor in seen:
            document = self._documents.get(sid)
            section = document.section(anchor) if document else None
            if document is None or section is None or left <= 0:
                continue
            text = quote_for_prompt_code(section.text)[:left]
            left -= len(text)
            parts.append(f"[{sid} {anchor}: {section.heading}]\n{text}")
        return "\n\n".join(parts)

    def checkpoint_inputs(self) -> object:
        return {
            "settings": self.settings,
            "sources": self._sources,
            "title": self._title,
            "abstract": self._abstract,
            "guideline": self._guideline,
        }

    # ---- hooks: outline -----------------------------------------------------------

    def offered_duplicates(self, ledger: EvidenceLedger) -> dict[str, str]:
        # A fact the paper states in several places (abstract, introduction,
        # results) and a figure two readers both recorded are offered once.
        return duplicate_evidence(ledger, within_source=True)

    def outline_prompt(
        self, ledger: EvidenceLedger, duplicates: Mapping[str, str]
    ) -> str:
        return "\n\n".join(
            [
                f"Paper: {self._title}\nLanguage: {self.settings.language}\n"
                f"At most {self.settings.max_sections} sections.",
                block(
                    "evidence",
                    brief_lines([e for e in ledger if e.id not in duplicates]),
                ),
                block("translation_guideline", self._guideline),
            ]
        )

    def repair(
        self, outline: PaperOutline, ledger: EvidenceLedger
    ) -> tuple[PaperOutline, list[str]]:
        return repair_paper_outline(
            outline, ledger, self.settings.post, self.settings.language
        )

    def outline_problems(
        self, outline: PaperOutline, ledger: EvidenceLedger
    ) -> list[str]:
        return paper_outline_problems(
            outline,
            ledger,
            self.settings.post,
            max_sections=self.settings.max_sections,
        )

    # ---- hooks: sections and the whole post ---------------------------------------

    def reference_for(self, own: Sequence[Evidence]) -> str:
        return self.source_sections(own)

    def prompt_head(self) -> str:
        return (
            f"Language: {self.settings.language}. Paper: {self._title}. Post: "
            f"{self.settings.post}."
        )

    def tldr_prompt(
        self,
        outline: PaperOutline,
        section: PaperSection,
        target: int,
        openings: str,
        evidence: str,
    ) -> str:
        questions, exact = tldr_questions(self.settings.language)
        instructions = (
            prompts.TLDR_REVIEW.format(
                questions=" | ".join(questions)
                + ("" if exact else f" (translated into {self.settings.language})")
            )
            if self.settings.post == "review"
            else prompts.TLDR_SUMMARY
        )
        return "\n\n".join(
            [
                f"Language: {self.settings.language}. Paper: {self._title}.",
                instructions,
                block("post_sections", openings),
                block("evidence", evidence),
            ]
        )

    def tldr_problems(self, markdown: str) -> list[str]:
        return tldr_problems(markdown, self.settings.post, self.settings.language)

    async def assemble(
        self,
        outline: PaperOutline,
        ledger: EvidenceLedger,
        written: list[tuple[PaperSection, str]],
        tldr: str | None,
    ) -> EvidencePaper:
        parts: list[tuple[str, str]] = [(s.kind, md) for s, md in written]
        if tldr is not None:
            parts.insert(0, ("tldr", tldr))
        sections = [(kind, strip_markers(md)) for kind, md in parts]
        final = await self._final(self._title, self._abstract, sections)
        return EvidencePaper(
            outline=outline,
            ledger=ledger,
            sections=sections,
            subtitle=final.subtitle,
            description=final.description,
            notes=self.notes,
            blocking=self.blocking,
        )

    async def _final(
        self, title: str, abstract: str, sections: Sequence[tuple[str, str]]
    ) -> FinalText:
        """Subtitle and search description, checked by the post validator's
        description rules with one repair; a description still unusable is
        dropped (the blog derives one). A checkpointed stage."""
        return await self.staged(
            "final",
            [prompts.FINAL_SYSTEM, title, abstract, list(sections), self.settings],
            lambda: self._final_text(title, abstract, sections),
            dump=lambda final: final.model_dump(mode="json"),
            load=FinalText.model_validate,
        )

    async def _final_text(
        self, title: str, abstract: str, sections: Sequence[tuple[str, str]]
    ) -> FinalText:
        tldr = next((md for kind, md in sections if kind == "tldr"), "")
        base = "\n\n".join(
            [
                f"Language: {self.settings.language}. Paper title: {title}",
                block("tldr", tldr),
                block("abstract", abstract),
            ]
        )
        prompt = base
        final = FinalText(subtitle="", description="")
        with usage_stage("finalize"):
            for _ in range(2):
                final = await self.model.structured(
                    role="evaluate",
                    system=prompts.FINAL_SYSTEM,
                    prompt=prompt,
                    schema=FinalText,
                )
                description = fit_description(" ".join(final.description.split()))
                if not (problems := description_problems(description, tldr)):
                    return final.model_copy(update={"description": description})
                prompt = (
                    base
                    + "\n\n"
                    + block("previous_problems", "\n".join(f"- {p}" for p in problems))
                )
        self.notes.append(
            "The description still broke the rules; the blog derives one."
        )
        return final.model_copy(update={"description": ""})
