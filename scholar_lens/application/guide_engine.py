"""The evidence-based guide writer (docs/redesign.md, 3.2 and 5.1).

Read (a bounded agent per source page, records checked against the page and
its authority) → outline (each piece of evidence to one section, in the
guide's order) → draft each section from its evidence → verify and review →
fix only the named sentences → the TL;DR last, from what the sections say →
assemble without the evidence markers (a guide lists its sources at the
end). Ports only: the model and the evidence cache are injected.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import ValidationError

from ..domain.evidence import (
    Evidence,
    EvidenceLedger,
    SourceDocument,
    duplicate_evidence,
    near_duplicate,
)
from ..domain.evidence_draft import (
    cited_ids,
    draft_problems,
    prose_sentences,
    section_heading,
    section_numbers,
)
from ..domain.glossary import glossary_view
from ..domain.guide.evidence_guide import (
    TLDR_TITLE,
    GuideOutline,
    GuideSection,
    cited_sources,
    guide_draft_problems,
    guide_outline_problems,
    guide_section_targets,
    repair_guide_outline,
    strip_markers,
)
from ..prompts import guide_evidence as prompts
from .evidence_writer import (
    EvidenceWriter,
    WriterPrompts,
    block,
    evidence_lines,
    recheck,
)
from .ports import AgentLimits, LanguageModel, TextStore
from .usage import usage_stage

logger = logging.getLogger(__name__)

# Bump when the reader's prompt or the evidence schema changes meaning.
GUIDE_READER_VERSION = 5
_SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")
# Small counts and ordinals a guide states without evidence.
_FREE_NUMBERS = " ".join(str(n) for n in range(11))

GUIDE_PROMPTS = WriterPrompts(
    reader=prompts.READER_SYSTEM,
    writer=prompts.WRITER_SYSTEM,
    claim_check=prompts.CLAIM_CHECK_SYSTEM,
    evidence_check=prompts.EVIDENCE_CHECK_SYSTEM,
    review=prompts.REVIEW_SYSTEM,
    fix=prompts.FIX_SYSTEM,
)
_PRIMARY = ("official", "reference")
_AUTHORITY_RANK = {"official": 0, "reference": 1, "third_party": 2, "older_docs": 3}
# Definitions in a code block: what later code of the running example calls.
_DEFINITION = re.compile(
    r"^\s*(?:(?:async\s+)?def \w+\(.*|class \w+.*|(?:export\s+)?(?:async\s+)?function \w+\(.*"
    r"|@\w[\w.]*(?:\(.*\))?)$",
    re.M,
)
_CODE_BLOCK = re.compile(r"^(`{3,}|~{3,})[^\n]*\n(.*?)^\1[ \t]*$", re.M | re.S)


def defined_in_code(markdowns: Sequence[str], limit: int = 60) -> str:
    """The definitions (functions, classes, decorators) the earlier
    sections' code shows, in order: the running example's names and
    signatures that later code must call as defined."""
    lines: list[str] = []
    for md in markdowns:
        for block_match in _CODE_BLOCK.finditer(md):
            for m in _DEFINITION.finditer(block_match.group(2)):
                line = m.group(0).strip()
                if line not in lines:
                    lines.append(line)
    return "\n".join(lines[-limit:])


def _canonical(outline: GuideOutline, duplicates: Mapping[str, str]) -> GuideOutline:
    """Each duplicate evidence id replaced by the id that keeps its fact."""
    if not duplicates:
        return outline
    return outline.model_copy(
        update={
            "sections": [
                s.model_copy(
                    update={"evidence": [duplicates.get(e, e) for e in s.evidence]}
                )
                for s in outline.sections
            ]
        }
    )


def _copied(markdown: str, body: Sequence[str]) -> list[str]:
    """TL;DR sentences that copy a body sentence instead of summing up."""
    return [
        f"copies a sentence of the body; sum it up in new words: {s[:160]}"
        for s in prose_sentences(markdown)
        if near_duplicate(s, body)
    ]


@dataclass(frozen=True)
class GuideSettings:
    language: str
    title_max_length: int
    # Cache-key part naming the reading model.
    reader_tag: str
    max_sections: int = 14
    max_fix_rounds: int = 2
    # Agent turns per page: the docs and the user's material are read in
    # depth, search results for what they add.
    read_turns_primary: int = 24
    read_turns_other: int = 10
    read_tokens_per_source: int = 400_000
    # Pages read (the docs and the user's material first).
    max_sources: int = 40
    concurrency: int = 4
    outline_repairs: int = 2
    # Prose characters (Korean) of the whole guide.
    min_total_chars: int = 30_000
    max_total_chars: int = 55_000

    @property
    def korean(self) -> bool:
        return self.language == "Korean"


@dataclass
class EvidenceGuide:
    title: str
    # Markdown without evidence markers.
    body: str
    outline: GuideOutline
    ledger: EvidenceLedger
    # The sources the body cites, in order of first citation.
    cited: list[SourceDocument] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def authority_rule(has_official: bool) -> Callable[..., list[str]]:
    """Settings (commands, flags, defaults) come from the official docs when
    there are any, and never from another version of them."""

    def rule(
        document: SourceDocument, kind: str, fields: Mapping[str, str]
    ) -> list[str]:
        if kind != "setting":
            return []
        if document.authority == "older_docs":
            return [
                "this page documents another version: its settings are not "
                "current; record how the mechanism worked (method) instead"
            ]
        if has_official and document.authority != "official":
            return [
                "settings come from the official docs only; record what this "
                "page explains (finding, method, relation) instead"
            ]
        return []

    return rule


def _outline_lines(items: Sequence[Evidence]) -> str:
    """Evidence for the planner: claims, scopes, figures and the first lines
    of code (quotes stay with the writer)."""
    lines = []
    for e in items:
        line = f"[{e.id}] ({e.paper_id}, {e.kind}) {e.claim}"
        if e.scope:
            line += f" | scope: {e.scope}"
        if e.figure_url:
            line += f" | figure: {(e.caption or '')[:160]}"
        if e.code:
            head = " ⏎ ".join(e.code.strip().split("\n")[:3])
            line += f" | code: {head[:200]}"
        lines.append(line)
    return "\n".join(lines)


class EvidenceGuideEngine(EvidenceWriter):
    def __init__(
        self,
        model: LanguageModel,
        settings: GuideSettings,
        store: TextStore | None = None,
        *,
        identifier_check: Callable[[str], list[str]] | None = None,
    ) -> None:
        super().__init__(model, GUIDE_PROMPTS, max_fix_rounds=settings.max_fix_rounds)
        self.settings = settings
        self.store = store
        # Flags and arguments of a draft that no official page mentions.
        self.identifier_check = identifier_check

    # ---- read -------------------------------------------------------------------

    def _cache_key(self, document: SourceDocument) -> str:
        page = hashlib.sha256(document.url.encode()).hexdigest()[:16]
        text = hashlib.sha256(document.text.encode()).hexdigest()[:12]
        return (
            f"evidence/{page}-{text}/v{GUIDE_READER_VERSION}-{self.settings.reader_tag}-"
            f"{_SAFE_KEY.sub('_', self.settings.language)}-{document.authority}.json"
        )

    def selected(self, documents: Sequence[SourceDocument]) -> list[SourceDocument]:
        """The pages to read: the docs and the user's material first, then
        the rest in corpus order, up to ``max_sources``."""
        ordered = [d for d in documents if d.authority in _PRIMARY] + [
            d for d in documents if d.authority not in _PRIMARY
        ]
        if dropped := ordered[self.settings.max_sources :]:
            self.notes.append(
                f"{len(dropped)} page(s) past the reading limit were not read: "
                + ", ".join(d.url for d in dropped[:10])
            )
        return ordered[: self.settings.max_sources]

    async def read(
        self, topic: str, documents: Sequence[SourceDocument]
    ) -> EvidenceLedger:
        """Every page's evidence (cached per page content, reader version,
        model, language and authority), in the given order."""
        gate = asyncio.Semaphore(self.settings.concurrency)
        has_official = any(d.authority == "official" for d in documents)
        rule = authority_rule(has_official)

        async def one(document: SourceDocument) -> list[Evidence]:
            async with gate:
                key = self._cache_key(document)
                if self.store is not None and (cached := self.store.get(key)):
                    try:
                        items = [Evidence.model_validate(e) for e in json.loads(cached)]
                        return [
                            e
                            for e in items
                            if not recheck(document, e)
                            and not rule(document, e.kind, {})
                        ]
                    except (ValueError, ValidationError):
                        logger.warning(
                            "Unreadable cached evidence %s; reading again.", key
                        )
                primary = document.authority in _PRIMARY
                prompt = "\n\n".join(
                    [
                        f"Guide topic: {topic}",
                        f"Page {document.paper_id}: {document.title} ({document.url}); "
                        f"authority: {document.authority or 'third_party'}.",
                        block("outline", document.outline()),
                        f"Guide language for claims and cells: {self.settings.language}.",
                        (
                            "Record about 10-25 pieces of evidence."
                            if primary
                            else "Record about 3-10 pieces of evidence: what this "
                            "page adds to the docs."
                        ),
                    ]
                )
                items = await self.read_document(
                    document,
                    prompt=prompt,
                    limits=AgentLimits(
                        turns=(
                            self.settings.read_turns_primary
                            if primary
                            else self.settings.read_turns_other
                        ),
                        total_tokens=self.settings.read_tokens_per_source,
                    ),
                    rule=rule,
                    # A docs page's diagrams are its most useful visuals and
                    # the reader tends to skip them: ask for one.
                    required=("figure",) if primary and document.figures else (),
                )
                if self.store is not None and items:
                    self.store.put(
                        key,
                        json.dumps([e.model_dump() for e in items], ensure_ascii=False),
                    )
                return items

        with usage_stage("read"):
            results = await asyncio.gather(*(one(d) for d in documents))
        ledger = EvidenceLedger()
        for items in results:
            ledger.extend(items)
        return ledger

    # ---- outline ----------------------------------------------------------------

    async def outline(
        self,
        topic: str,
        ledger: EvidenceLedger,
        documents: Mapping[str, SourceDocument],
        guideline: str,
    ) -> GuideOutline:
        # A fact several pages state is offered once, from the most
        # authoritative page.
        duplicates = duplicate_evidence(
            ledger,
            {
                sid: _AUTHORITY_RANK.get(d.authority or "third_party", 2)
                for sid, d in documents.items()
            },
        )
        if duplicates:
            logger.info("%d evidence items repeat another page's.", len(duplicates))
        sources = "\n".join(
            f"- {sid} | {d.authority or 'third_party'} | {d.title} | {d.url}"
            for sid, d in documents.items()
        )
        base = "\n\n".join(
            [
                f"Topic: {topic}\nLanguage: {self.settings.language}",
                f"Title: in English, at most {self.settings.title_max_length} characters. "
                f"At most {self.settings.max_sections} sections.",
                block("sources", sources),
                block(
                    "evidence",
                    _outline_lines([e for e in ledger if e.id not in duplicates]),
                ),
                block("translation_guideline", guideline),
            ]
        )
        prompt = base
        problems: list[str] = []
        with usage_stage("outline"):
            for attempt in range(self.settings.outline_repairs + 1):
                outline = await self.model.structured(
                    role="outline",
                    system=prompts.OUTLINE_SYSTEM,
                    prompt=prompt,
                    schema=GuideOutline,
                )
                outline, repaired = repair_guide_outline(
                    _canonical(outline, duplicates), ledger
                )
                if repaired:
                    logger.info(
                        "Guide outline repaired in code: %s", "; ".join(repaired)
                    )
                problems = guide_outline_problems(
                    outline,
                    ledger,
                    documents,
                    title_max_length=self.settings.title_max_length,
                    max_sections=self.settings.max_sections,
                )
                if not problems:
                    return outline
                logger.warning(
                    "Guide outline attempt %d unusable: %s", attempt + 1, problems
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
            "The guide outline breaks the rules after repairs: " + "; ".join(problems)
        )

    async def _drop_wrong(
        self, outline: GuideOutline, ledger: EvidenceLedger
    ) -> GuideOutline:
        used = {eid for s in outline.sections for eid in s.evidence}
        wrong = await self.wrong_evidence([e for e in ledger if e.id in used])
        if not wrong:
            return outline
        sections = [
            s.model_copy(update={"evidence": [e for e in s.evidence if e not in wrong]})
            for s in outline.sections
        ]
        return outline.model_copy(update={"sections": sections})

    # ---- sections ---------------------------------------------------------------

    async def write(
        self,
        topic: str,
        documents: Sequence[SourceDocument],
        *,
        guideline: str = "",
    ) -> EvidenceGuide:
        documents = self.selected(documents)
        by_id = {d.paper_id: d for d in documents}
        self._names = {d.paper_id: d.title for d in documents}
        ledger = await self.read(topic, documents)
        if not len(ledger):
            raise ValueError("No source page gave usable evidence for the guide.")
        outline = await self.outline(topic, ledger, by_id, guideline)
        outline = await self._drop_wrong(outline, ledger)
        glossary = {t.english: t.rendering for t in outline.glossary}
        targets = guide_section_targets(
            outline,
            min_total=self.settings.min_total_chars,
            max_total=self.settings.max_total_chars,
            korean=self.settings.korean,
        )
        written: list[tuple[GuideSection, str]] = []
        numbers: dict[str, str] = {}
        tldr: tuple[GuideSection, int] | None = None
        for section, target in zip(outline.sections, targets, strict=True):
            if section.kind == "tldr":
                # Written last, from what the sections came to say.
                tldr = (section, target)
                continue
            markdown = await self._section(
                topic, outline, ledger, section, written, numbers, glossary, target
            )
            heading = section_heading(markdown) or section.title
            for n in section_numbers(markdown, _FREE_NUMBERS):
                numbers.setdefault(n, heading)
            written.append((section, markdown))
        parts = [md for _, md in written]
        if tldr is not None:
            parts.insert(0, await self._tldr(topic, outline, ledger, *tldr, written))
        body = "\n\n".join(p.strip() for p in parts if p.strip())
        cited = [by_id[s] for s in cited_sources(body, ledger) if s in by_id]
        return EvidenceGuide(
            title=outline.title,
            body=strip_markers(body),
            outline=outline,
            ledger=ledger,
            cited=cited,
            notes=self.notes,
        )

    def _prompt(
        self,
        topic: str,
        outline: GuideOutline,
        section: GuideSection,
        own: Sequence[Evidence],
        earlier: Sequence[Evidence],
        written: Sequence[tuple[GuideSection, str]],
        glossary: Mapping[str, str],
        target: int,
    ) -> str:
        sections_map = "\n".join(
            f"- {'(this) ' if s is section else ''}{s.kind}: {s.title} — {s.thesis}"
            for s in outline.sections
        )
        return "\n\n".join(
            [
                f"Language: {self.settings.language}. Topic: {topic}.",
                block("outline", sections_map),
                block(
                    "section",
                    f"kind: {section.kind}\ntitle: {section.title}\nthesis: {section.thesis}\n"
                    f"question: {section.question}\ndepth: {section.depth}\n"
                    f"length: about {target:,} characters of prose\n"
                    f"running example: {outline.running_example or 'none'}\n"
                    f"code language: {outline.code_language or 'the docs’ main one'}",
                ),
                block("evidence", evidence_lines(own)),
                block("already_written_evidence", evidence_lines(list(earlier)[-60:])),
                block(
                    "source_names",
                    "\n".join(f"- {sid}: {name}" for sid, name in self._names.items()),
                ),
                block(
                    "glossary", glossary_view(dict(glossary), [m for _, m in written])
                ),
                block(
                    "previous_section_end", written[-1][1][-2500:] if written else ""
                ),
                block("code_defined_earlier", defined_in_code([m for _, m in written])),
            ]
        )

    def _checks(
        self,
        section: GuideSection,
        outline: GuideOutline,
        ledger: EvidenceLedger,
        own: Sequence[Evidence],
        earlier_ids: Sequence[str],
        numbers: Mapping[str, str],
        exhausted: set[str],
        target: int,
    ) -> Callable[[str], list[str]]:
        figures = [e.figure_url for e in own if e.kind == "figure" and e.figure_url]

        def checks(markdown: str) -> list[str]:
            problems = draft_problems(
                markdown,
                title=section.title,
                own=section.evidence,
                earlier=earlier_ids,
                ledger=ledger,
                earlier_numbers=numbers,
                figure_urls=figures,
                free_numbers=_FREE_NUMBERS,
                exhausted=exhausted,
                target_chars=target,
                depth=(
                    section.depth
                    if section.kind in ("concepts", "architecture")
                    else ""
                ),
                # Advice is stated plainly; the guide's own judgement is
                # marked only where it chooses.
                own_view_limit=1 if section.kind == "choosing" else 0,
            )
            problems += guide_draft_problems(
                markdown, code_language=outline.code_language
            )
            if self.identifier_check is not None and (
                unknown := self.identifier_check(markdown)
            ):
                problems.append(
                    "flags or arguments no official page mentions; use the "
                    "documented ones or leave them out: " + ", ".join(unknown)
                )
            return problems

        return checks

    async def _section(
        self,
        topic: str,
        outline: GuideOutline,
        ledger: EvidenceLedger,
        section: GuideSection,
        written: list[tuple[GuideSection, str]],
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
        # Evidence a later section already referred back to once.
        exhausted = {
            eid
            for s, md in written
            for eid in {i for snt in prose_sentences(md) for i in cited_ids(snt)}
            if eid not in s.evidence
        }
        prompt = self._prompt(
            topic, outline, section, own, earlier, written, glossary, target
        )
        draft = await self.draft(
            prompt,
            own,
            target,
            retry_hint="This is a technical guide to a software tool for practitioners.",
        )
        markdown = await self.verify_and_fix(
            draft,
            title=section.title,
            review_context=f"Section kind: {section.kind}; thesis: {section.thesis}; "
            f"depth: {section.depth}; running example: {outline.running_example or 'none'}.\n"
            + block("code_defined_earlier", defined_in_code([m for _, m in written])),
            checks=self._checks(
                section, outline, ledger, own, earlier_ids, numbers, exhausted, target
            ),
            ledger=ledger,
            own=own,
            earlier=earlier,
            free=_FREE_NUMBERS,
        )
        if not markdown.lstrip().startswith("## "):
            markdown = f"## {section.title}\n\n{markdown.lstrip()}"
        return markdown

    async def _tldr(
        self,
        topic: str,
        outline: GuideOutline,
        ledger: EvidenceLedger,
        section: GuideSection,
        target: int,
        written: Sequence[tuple[GuideSection, str]],
    ) -> str:
        """The TL;DR from its evidence and the sections' headings and
        opening sentences: it may state what the body states (the one place
        a fact is said twice)."""
        own = [e for eid in section.evidence if (e := ledger.get(eid))]
        cited = list(
            dict.fromkeys(
                eid
                for _, md in written
                for s in prose_sentences(md)
                for eid in cited_ids(s)
            )
        )
        body = [e for eid in cited if (e := ledger.get(eid))]
        openings = "\n".join(
            f"- {section_heading(md) or s.title}: "
            + " ".join(prose_sentences(md)[:2])[:400]
            for s, md in written
        )
        prompt = "\n\n".join(
            [
                f"Language: {self.settings.language}. Topic: {topic}.",
                block(
                    "section",
                    f"kind: tldr\ntitle: {TLDR_TITLE}\nthesis: {section.thesis}\n"
                    f"length: about {target:,} characters of prose, one or two "
                    "paragraphs, no list, no code, no figure",
                ),
                block("guide_sections", openings),
                block("evidence", evidence_lines([*own, *body][:80])),
            ]
        )
        draft = await self.draft(
            prompt,
            own,
            target,
            retry_hint="This is the TL;DR of a technical guide for practitioners.",
        )
        body_sentences = [s for _, md in written for s in prose_sentences(md)]
        # Figures stay in their sections.
        context = [e for e in (*own, *body) if e.kind != "figure"]
        markdown = await self.verify_and_fix(
            draft,
            title=TLDR_TITLE,
            review_context="Section kind: tldr: what the tool is, the problem, its "
            "key mechanism and when to use it.",
            checks=lambda md: (
                draft_problems(
                    md,
                    title=TLDR_TITLE,
                    own=[e.id for e in context],
                    earlier=(),
                    ledger=ledger,
                    earlier_numbers={},
                    figure_urls=(),
                    free_numbers=_FREE_NUMBERS,
                    target_chars=target,
                    own_view_limit=0,
                )
                + guide_draft_problems(md)
                + _copied(md, body_sentences)
            ),
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
