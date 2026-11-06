"""The evidence-based paper review and summary writer (docs/redesign.md, 3.2,
5.2).

Read (bounded agents over groups of the paper's sections, records checked
against the paper) → outline (each piece of evidence to one section, in the
post's order) → draft each section from its evidence and the paper passages
it comes from → verify and review → fix only the named sentences → the
TL;DR last → subtitle and description. Ports only: the model and the
evidence cache are injected.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from ..domain.evidence import Evidence, EvidenceLedger, SourceDocument
from ..domain.evidence_draft import (
    cited_ids,
    draft_problems,
    prose_sentences,
    section_heading,
    section_numbers,
)
from ..domain.glossary import glossary_view
from ..domain.guide.evidence_guide import guide_draft_problems, strip_markers
from ..domain.paper.evidence_paper import (
    REVIEW_TLDR_QUESTIONS,
    TLDR_TITLE,
    PaperOutline,
    PaperSection,
    PostKind,
    paper_outline_problems,
    paper_section_targets,
    repair_paper_outline,
    tldr_problems,
)
from ..domain.validation import description_problems, fit_description
from ..prompts import paper_evidence as prompts
from .evidence_writer import (
    EvidenceWriter,
    WriterPrompts,
    block,
    brief_lines,
    evidence_lines,
    quote_for_prompt_code,
    recheck,
)
from .ports import AgentLimits, LanguageModel, TextStore
from .usage import usage_stage

logger = logging.getLogger(__name__)

# Bump when the reader's prompt or the evidence schema changes meaning.
PAPER_READER_VERSION = 1
_SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")
_FREE_NUMBERS = " ".join(str(n) for n in range(11))

PAPER_PROMPTS = WriterPrompts(
    reader=prompts.READER_SYSTEM,
    writer=prompts.WRITER_SYSTEM,
    claim_check=prompts.CLAIM_CHECK_SYSTEM,
    evidence_check=prompts.EVIDENCE_CHECK_SYSTEM,
    review=prompts.REVIEW_SYSTEM,
    fix=prompts.FIX_SYSTEM,
)


@dataclass(frozen=True)
class PaperSettings:
    language: str
    post: PostKind
    # Cache-key part naming the reading model.
    reader_tag: str
    max_sections: int = 12
    max_fix_rounds: int = 2
    # Paper characters one reading agent covers, its turns and tokens.
    chars_per_reader: int = 15_000
    read_turns: int = 24
    read_tokens: int = 500_000
    concurrency: int = 4
    outline_repairs: int = 2
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


class EvidencePaperEngine(EvidenceWriter):
    def __init__(
        self,
        model: LanguageModel,
        settings: PaperSettings,
        store: TextStore | None = None,
    ) -> None:
        super().__init__(model, PAPER_PROMPTS, max_fix_rounds=settings.max_fix_rounds)
        self.settings = settings
        self.store = store
        self._documents: dict[str, SourceDocument] = {}

    # ---- read -------------------------------------------------------------------

    def _cache_key(self, document: SourceDocument, part: int) -> str:
        pid = _SAFE_KEY.sub("_", document.paper_id)
        text = hashlib.sha256(document.text.encode()).hexdigest()[:12]
        return (
            f"evidence/{pid}/{text}-p{part}-v{PAPER_READER_VERSION}-"
            f"{self.settings.reader_tag}-{_SAFE_KEY.sub('_', self.settings.language)}-"
            f"{self.settings.post}.json"
        )

    async def read(
        self, title: str, paper: SourceDocument, extra: Sequence[SourceDocument]
    ) -> EvidenceLedger:
        """The paper's evidence, a reading agent per group of sections, then
        the extra documents' (an official code summary); cached per group."""
        gate = asyncio.Semaphore(self.settings.concurrency)
        groups = reading_groups(paper, self.settings.chars_per_reader)
        jobs = [(paper, g, n) for n, g in enumerate(groups)] + [
            (d, d, 0) for d in extra
        ]

        async def one(
            whole: SourceDocument, part: SourceDocument, n: int
        ) -> list[Evidence]:
            async with gate:
                key = self._cache_key(part, n)
                if self.store is not None and (cached := self.store.get(key)):
                    try:
                        items = [Evidence.model_validate(e) for e in json.loads(cached)]
                        return [e for e in items if not recheck(part, e)]
                    except (ValueError, ValidationError):
                        logger.warning(
                            "Unreadable cached evidence %s; reading again.", key
                        )
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
                items = await self.read_document(
                    part,
                    prompt=prompt,
                    limits=AgentLimits(
                        turns=self.settings.read_turns,
                        total_tokens=self.settings.read_tokens,
                    ),
                    rule=_no_code_or_settings,
                )
                if self.store is not None and items:
                    self.store.put(
                        key,
                        json.dumps([e.model_dump() for e in items], ensure_ascii=False),
                    )
                return items

        with usage_stage("read"):
            results = await asyncio.gather(*(one(w, p, n) for w, p, n in jobs))
        ledger = EvidenceLedger()
        for items in results:
            ledger.extend(items)
        return ledger

    # ---- outline ----------------------------------------------------------------

    async def outline(
        self, title: str, ledger: EvidenceLedger, guideline: str
    ) -> PaperOutline:
        system = (
            prompts.OUTLINE_REVIEW
            if self.settings.post == "review"
            else prompts.OUTLINE_SUMMARY
        )
        base = "\n\n".join(
            [
                f"Paper: {title}\nLanguage: {self.settings.language}\n"
                f"At most {self.settings.max_sections} sections.",
                block("evidence", brief_lines(list(ledger))),
                block("translation_guideline", guideline),
            ]
        )
        prompt = base
        problems: list[str] = []
        with usage_stage("outline"):
            for attempt in range(self.settings.outline_repairs + 1):
                outline = await self.model.structured(
                    role="outline", system=system, prompt=prompt, schema=PaperOutline
                )
                outline, repaired = repair_paper_outline(
                    outline, ledger, self.settings.post, self.settings.language
                )
                if repaired:
                    logger.info(
                        "Paper outline repaired in code: %s", "; ".join(repaired)
                    )
                problems = paper_outline_problems(
                    outline,
                    ledger,
                    self.settings.post,
                    max_sections=self.settings.max_sections,
                )
                if not problems:
                    return outline
                logger.warning(
                    "Paper outline attempt %d unusable: %s", attempt + 1, problems
                )
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
            "The paper outline breaks the rules after repairs: " + "; ".join(problems)
        )

    async def _drop_wrong(
        self, outline: PaperOutline, ledger: EvidenceLedger
    ) -> PaperOutline:
        used = {eid for s in outline.sections for eid in s.evidence}
        wrong = await self.wrong_evidence([e for e in ledger if e.id in used])
        if not wrong:
            return outline
        return outline.model_copy(
            update={
                "sections": [
                    s.model_copy(
                        update={"evidence": [e for e in s.evidence if e not in wrong]}
                    )
                    for s in outline.sections
                ]
            }
        )

    # ---- sections ---------------------------------------------------------------

    async def write(
        self,
        title: str,
        paper: SourceDocument,
        *,
        extra: Sequence[SourceDocument] = (),
        abstract: str = "",
        guideline: str = "",
    ) -> EvidencePaper:
        self._documents = {d.paper_id: d for d in (paper, *extra)}
        self.passages = {
            (d.paper_id, s.anchor): s.text for d in (paper, *extra) for s in d.sections
        }
        self._names = {d.paper_id: d.title for d in (paper, *extra)}
        ledger = await self.read(title, paper, extra)
        if not len(ledger):
            raise ValueError("The paper gave no usable evidence.")
        outline = await self.outline(title, ledger, guideline)
        outline = await self._drop_wrong(outline, ledger)
        glossary = {t.english: t.rendering for t in outline.glossary}
        targets = paper_section_targets(
            outline,
            min_total=self.settings.min_total_chars,
            max_total=self.settings.max_total_chars,
            korean=self.settings.korean,
        )
        written: list[tuple[PaperSection, str]] = []
        numbers: dict[str, str] = {}
        tldr: tuple[PaperSection, int] | None = None
        for section, target in zip(outline.sections, targets, strict=True):
            if section.kind == "tldr":
                tldr = (section, target)
                continue
            markdown = await self._section(
                title, outline, ledger, section, written, numbers, glossary, target
            )
            heading = section_heading(markdown) or section.title
            for n in section_numbers(markdown, _FREE_NUMBERS):
                numbers.setdefault(n, heading)
            written.append((section, markdown))
        parts: list[tuple[str, str]] = [(s.kind, md) for s, md in written]
        if tldr is not None:
            parts.insert(0, ("tldr", await self._tldr(title, ledger, *tldr, written)))
        sections = [(kind, strip_markers(md)) for kind, md in parts]
        final = await self._final(title, abstract, sections)
        return EvidencePaper(
            outline=outline,
            ledger=ledger,
            sections=sections,
            subtitle=final.subtitle,
            description=final.description,
            notes=self.notes,
        )

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

    async def _section(
        self,
        title: str,
        outline: PaperOutline,
        ledger: EvidenceLedger,
        section: PaperSection,
        written: list[tuple[PaperSection, str]],
        numbers: dict[str, str],
        glossary: Mapping[str, str],
        target: int,
    ) -> str:
        own = [e for eid in section.evidence if (e := ledger.get(eid))]
        earlier_ids = list(
            dict.fromkeys(
                eid
                for _, md in written
                for s in prose_sentences(md)
                for eid in cited_ids(s)
            )
        )
        earlier = [e for eid in earlier_ids if (e := ledger.get(eid))]
        exhausted = {
            eid
            for s, md in written
            for eid in {i for snt in prose_sentences(md) for i in cited_ids(snt)}
            if eid not in s.evidence
        }
        figures = [e.figure_url for e in own if e.kind == "figure" and e.figure_url]
        reference = self.source_sections(own)
        sections_map = "\n".join(
            f"- {'(this) ' if s is section else ''}{s.kind}: {s.title} — {s.thesis}"
            for s in outline.sections
        )
        prompt = "\n\n".join(
            [
                f"Language: {self.settings.language}. Paper: {title}. Post: "
                f"{self.settings.post}.",
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
                    "glossary", glossary_view(dict(glossary), [m for _, m in written])
                ),
                block(
                    "previous_section_end", written[-1][1][-2500:] if written else ""
                ),
                block("source_sections", reference),
            ]
        )
        draft = await self.draft(
            prompt,
            own,
            target,
            retry_hint="This is a post explaining a published research paper.",
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
                free_numbers=_FREE_NUMBERS,
                passages=self.passages,
                exhausted=exhausted,
                target_chars=target,
                depth=section.depth if section.kind == "method" else "",
                own_view_limit=1 if section.kind == "closing" else 0,
            ) + guide_draft_problems(markdown)

        self.reference = reference
        markdown = await self.verify_and_fix(
            draft,
            title=section.title,
            review_context=f"Section kind: {section.kind}; thesis: {section.thesis}; "
            f"depth: {section.depth}.",
            checks=checks,
            ledger=ledger,
            own=own,
            earlier=earlier,
            free=_FREE_NUMBERS,
        )
        self.reference = ""
        heading = section.title
        if not markdown.lstrip().startswith("## "):
            markdown = f"## {heading}\n\n{markdown.lstrip()}"
        elif self.settings.post == "summary" or section.kind == "closing":
            # The fixed headings of the blog's convention.
            markdown = re.sub(
                r"^## .*$", f"## {heading}", markdown.lstrip(), count=1, flags=re.M
            )
        return markdown

    async def _tldr(
        self,
        title: str,
        ledger: EvidenceLedger,
        section: PaperSection,
        target: int,
        written: Sequence[tuple[PaperSection, str]],
    ) -> str:
        """The TL;DR from the sections' cited evidence and openings."""
        cited = list(
            dict.fromkeys(
                eid
                for _, md in written
                for s in prose_sentences(md)
                for eid in cited_ids(s)
            )
        )
        context = [e for eid in cited if (e := ledger.get(eid)) and e.kind != "figure"]
        openings = "\n".join(
            f"- {section_heading(md) or s.title}: "
            + " ".join(prose_sentences(md)[:2])[:400]
            for s, md in written
        )
        lang = "Korean" if self.settings.korean else ""
        instructions = (
            prompts.TLDR_REVIEW.format(
                questions=" | ".join(REVIEW_TLDR_QUESTIONS[lang])
            )
            if self.settings.post == "review"
            else prompts.TLDR_SUMMARY
        )
        prompt = "\n\n".join(
            [
                f"Language: {self.settings.language}. Paper: {title}.",
                instructions,
                block("post_sections", openings),
                block("evidence", evidence_lines(context[:80])),
            ]
        )
        draft = await self.draft(
            prompt, context, target, retry_hint="This is the TL;DR of a paper post."
        )

        def checks(md: str) -> list[str]:
            return draft_problems(
                md,
                title=TLDR_TITLE,
                own=[e.id for e in context],
                earlier=(),
                ledger=ledger,
                earlier_numbers={},
                figure_urls=(),
                free_numbers=_FREE_NUMBERS,
                passages=self.passages,
                own_view_limit=0,
            ) + tldr_problems(md, self.settings.post, self.settings.language)

        markdown = await self.verify_and_fix(
            draft,
            title=TLDR_TITLE,
            review_context="Section kind: tldr.",
            checks=checks,
            ledger=ledger,
            own=context,
            earlier=(),
            free=_FREE_NUMBERS,
        )
        markdown = re.sub(
            r"^## .*$", f"## {TLDR_TITLE}", markdown.lstrip(), count=1, flags=re.M
        )
        if not markdown.startswith("## "):
            markdown = f"## {TLDR_TITLE}\n\n{markdown}"
        return markdown

    async def _final(
        self, title: str, abstract: str, sections: Sequence[tuple[str, str]]
    ) -> FinalText:
        """Subtitle and search description, checked by the post validator's
        description rules with one repair; a description still unusable is
        dropped (the blog derives one)."""
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


def _no_code_or_settings(
    document: SourceDocument, kind: str, fields: object
) -> list[str]:
    """A paper post states what the paper finds and does; code and settings
    are a guide's evidence."""
    if kind in ("code", "setting"):
        return [f"kind {kind!r} is not used for a paper; record a method instead"]
    return []
