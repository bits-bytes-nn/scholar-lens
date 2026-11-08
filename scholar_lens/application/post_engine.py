"""The one post engine (docs/engine.md, 4 and 8 E1).

read → outline (repaired in code, asked again with its problems) → wrong
evidence left out → each section drafted from its evidence, checked and
fixed → whole-post reviews → the TL;DR last → assembly. The engine knows no
post kind: what differs comes from the kind's ``PostSpec`` (data) and the
small hooks below, which a kind's engine overrides. The shared loop pieces
(reading agent, claim check, fix) are ``EvidenceWriter``'s. Ports only: the
model and the evidence cache are injected.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel

from ..domain.evidence import (
    Evidence,
    EvidenceLedger,
    SourceDocument,
    same_fact_duplicates,
)
from ..domain.evidence_draft import (
    cited_ids,
    copied_sentences,
    free_numbers,
    prose_sentences,
    section_heading,
    section_numbers,
    section_problems,
)
from ..domain.glossary import glossary_view
from ..domain.post_spec import (
    PostOutline,
    PostSection,
    PostSpec,
    depth_targets,
    evidence_policy,
    other_sdk,
)
from ..domain.verifiers import Problem, VerifierSet, as_problems
from .evidence_writer import (
    EvidenceWriter,
    PostFinding,
    PostReview,
    WriterPrompts,
    all_or_none,
    block,
    cached_evidence,
    evidence_lines,
    post_so_far,
    with_canonical_evidence,
)
from .ports import AgentLimits, LanguageModel, TextStore
from .usage import usage_stage

logger = logging.getLogger(__name__)

# Evidence a TL;DR or a section prompt shows at most.
_TLDR_EVIDENCE = 80
_EARLIER_EVIDENCE = 60
# The end of the previous section the next section's writer reads.
_PREVIOUS_END_CHARS = 2_500


@dataclass(frozen=True)
class ReadJob:
    """One reading agent's work: a source (or part of one), the cache key
    of its evidence, the reader's prompt and limits, and the evidence kinds
    it is asked again for when it skips them."""

    document: SourceDocument
    key: str
    prompt: str
    limits: AgentLimits
    required: tuple[str, ...] = ()


# Makes a read job inside the reading gate (a survey fetches the paper's
# text first); None when there is nothing to read.
ReadStart = Callable[[], Awaitable[ReadJob | None]]


@dataclass(frozen=True)
class PostPrompts:
    """The system prompts of the outline and of each whole-post review."""

    outline: str
    passes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSettings:
    """The run's values the engine reads (from a kind's settings)."""

    language: str
    concurrency: int
    outline_repairs: int
    min_total_chars: int
    max_total_chars: int


@dataclass(frozen=True)
class SectionContext[OutlineT, SectionT]:
    """What one section's prompt, checks and review see."""

    outline: OutlineT
    ledger: EvidenceLedger
    section: SectionT
    own: list[Evidence]
    earlier_ids: list[str]
    earlier: list[Evidence]
    written: Sequence[tuple[SectionT, str]]
    glossary: Mapping[str, str]
    target: int
    reference: str


def _with_title(markdown: str, title: str) -> str:
    """``markdown``'s first ``##`` heading set to ``title`` (a function
    replacement: the title is text, not a template, "$\\mathcal{O}(n)$"
    holds escapes)."""
    return re.sub(
        r"^## .*$", lambda _m: f"## {title}", markdown.lstrip(), count=1, flags=re.M
    )


class PostEngine[OutlineT: PostOutline, SectionT: PostSection, ResultT](EvidenceWriter):
    """The generic flow over a ``PostSpec``; a kind's engine supplies its
    prompts and overrides the hooks it needs."""

    def __init__(
        self,
        model: LanguageModel,
        spec: PostSpec,
        prompts: WriterPrompts,
        post_prompts: PostPrompts,
        run: RunSettings,
        *,
        outline_schema: type[OutlineT],
        store: TextStore | None = None,
        max_fix_rounds: int,
    ) -> None:
        super().__init__(
            model, prompts, max_fix_rounds=max_fix_rounds, limits=spec.limits
        )
        self.spec = spec
        self.post_prompts = post_prompts
        self.run = run
        self.outline_schema = outline_schema
        self.store = store
        # Numbers a sentence may state without evidence.
        self.free = free_numbers()

    # ---- read -------------------------------------------------------------------

    def policy_problems(
        self,
        document: SourceDocument,
        kind: str,
        fields: Mapping[str, object],
        corpus: Collection[str] = (),
    ) -> list[str]:
        """The spec's evidence policy for one record."""
        return evidence_policy(
            self.spec.evidence, document, kind, fields, corpus=corpus
        )

    async def read_all(
        self, starts: Iterable[ReadStart], *, corpus: Collection[str] = ()
    ) -> list[tuple[ReadJob | None, list[Evidence]]]:
        """Every job's evidence, in order, at most ``concurrency`` readers at
        once; evidence is cached per job key, and a failure cancels the
        rest. ``corpus``: the authorities of the run's sources (the policy
        reads them)."""
        gate = asyncio.Semaphore(self.run.concurrency)

        def rule(
            document: SourceDocument, kind: str, fields: Mapping[str, str]
        ) -> list[str]:
            return self.policy_problems(document, kind, fields, corpus)

        async def one(start: ReadStart) -> tuple[ReadJob | None, list[Evidence]]:
            async with gate:
                job = await start()
                if job is None:
                    return None, []
                if self.store is not None and (cached := self.store.get(job.key)):
                    document = job.document
                    kept = (
                        cached_evidence(
                            cached,
                            document,
                            lambda e: rule(document, e.kind, e.model_dump(mode="json")),
                            key=job.key,
                        )
                        if self.spec.evidence.recheck_cached
                        else cached_evidence(cached, document, key=job.key)
                    )
                    if kept is not None:
                        return job, kept
                items = await self.read_document(
                    job.document,
                    prompt=job.prompt,
                    limits=job.limits,
                    required=job.required,
                    rule=rule,
                )
                if self.store is not None and items:
                    self.store.put(
                        job.key,
                        json.dumps([e.model_dump() for e in items], ensure_ascii=False),
                    )
                return job, items

        with usage_stage("read"):
            return await all_or_none(one(s) for s in starts)

    @staticmethod
    def ledger_of(results: Iterable[Sequence[Evidence]]) -> EvidenceLedger:
        """One ledger of the jobs' evidence, numbered in job order."""
        ledger = EvidenceLedger()
        for items in results:
            ledger.extend(items)
        return ledger

    # ---- outline ----------------------------------------------------------------

    async def plan(self, ledger: EvidenceLedger) -> OutlineT:
        """The outline: asked, its duplicate facts merged, repaired in code,
        and asked again with its problems up to ``outline_repairs`` times.

        Raises:
            ValueError: The outline still breaks the spec's rules.
        """
        name = self.spec.outline_name
        duplicates = self.offered_duplicates(ledger)
        if duplicates:
            logger.info("%d evidence items repeat another.", len(duplicates))
        base = self.outline_prompt(ledger, duplicates)
        prompt = base
        problems: list[str] = []
        with usage_stage("outline"):
            for attempt in range(self.run.outline_repairs + 1):
                # The schema is a pydantic model the protocol cannot name.
                outline = cast(
                    OutlineT,
                    await self.model.structured(
                        role="outline",
                        system=self.post_prompts.outline,
                        prompt=prompt,
                        schema=cast(type[BaseModel], self.outline_schema),
                    ),
                )
                same = duplicates | same_fact_duplicates(
                    outline.same_facts, ledger, self.source_rank()
                )
                outline = self.canonical_elsewhere(
                    with_canonical_evidence(outline, same), same
                )
                outline, repaired = self.repair(outline, ledger)
                if repaired:
                    logger.info(
                        "%s repaired in code: %s",
                        name.capitalize(),
                        "; ".join(repaired),
                    )
                problems = self.outline_problems(outline, ledger)
                if not problems:
                    return outline
                logger.warning(
                    "%s attempt %d unusable: %s",
                    name.capitalize(),
                    attempt + 1,
                    problems,
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
            f"The {name} breaks the rules after repairs: " + "; ".join(problems)
        )

    @staticmethod
    def _without(outline: OutlineT, ids: Collection[str]) -> OutlineT:
        return outline.model_copy(
            update={
                "sections": [
                    s.model_copy(
                        update={"evidence": [e for e in s.evidence if e not in ids]}
                    )
                    for s in outline.sections
                ]
            }
        )

    async def drop_wrong(self, outline: OutlineT, ledger: EvidenceLedger) -> OutlineT:
        """The outline without the evidence it uses (sections and what the
        kind cites elsewhere) whose claim misstates its quote."""
        used = {
            eid for s in outline.sections for eid in s.evidence
        } | self.cited_elsewhere(outline)
        wrong = await self.wrong_evidence([e for e in ledger if e.id in used])
        if not wrong:
            return outline
        return self.drop_elsewhere(self._without(outline, wrong), wrong)

    def keep_code_language(self, outline: OutlineT, ledger: EvidenceLedger) -> OutlineT:
        """Evidence the reader recorded as one other SDK language's (a
        TypeScript-only behaviour in a Python post) leaves the outline:
        stated there it reads as true for the post's language."""
        language = self.code_language(outline)
        if not (self.spec.evidence.same_code_language and language):
            return outline
        other = [
            eid
            for sec in outline.sections
            for eid in sec.evidence
            if (e := ledger.get(eid)) and other_sdk(e.sdk, language)
        ]
        if not other:
            return outline
        logger.info(
            "Evidence for another SDK than %s left out: %s", language, ", ".join(other)
        )
        return self._without(outline, set(other))

    def drop_empty(self, outline: OutlineT) -> OutlineT:
        """Sections the checks emptied, dropped with a note when the spec
        says so (the TL;DR has no evidence of its own)."""
        if self.spec.empty_sections != "drop":
            return outline
        if empty := [
            s.title
            for s in outline.sections
            if not s.evidence and not self.spec.is_tldr(s.kind)
        ]:
            self.notes.append(
                "Sections left without evidence were dropped: " + "; ".join(empty)
            )
            outline = outline.model_copy(
                update={
                    "sections": [
                        s
                        for s in outline.sections
                        if s.evidence or self.spec.is_tldr(s.kind)
                    ]
                }
            )
        return outline

    # ---- the flow ---------------------------------------------------------------

    async def compose(self, ledger: EvidenceLedger) -> ResultT:
        """From the read evidence to the assembled post."""
        outline = await self.plan(ledger)
        outline = await self.drop_wrong(outline, ledger)
        outline = self.keep_code_language(outline, ledger)
        outline = self.drop_empty(outline)
        self._names = self.source_names(outline)
        glossary = {t.english: t.rendering for t in outline.glossary}
        targets = self.section_targets(outline)
        written, tldr = await self.write_sections(outline, ledger, glossary, targets)
        written = await self.review_post(outline, ledger, written)
        opening = (
            await self.write_tldr(outline, ledger, *tldr, written)
            if tldr is not None
            else None
        )
        return await self.assemble(outline, ledger, written, opening)

    @staticmethod
    def own(section: SectionT, ledger: EvidenceLedger) -> list[Evidence]:
        return [e for eid in section.evidence if (e := ledger.get(eid))]

    @staticmethod
    def cited(written: Sequence[tuple[Any, str]]) -> list[str]:
        """The evidence ids the written sections cite, in order, once."""
        return list(
            dict.fromkeys(
                eid
                for _, md in written
                for s in prose_sentences(md)
                for eid in cited_ids(s)
            )
        )

    # ---- sections ---------------------------------------------------------------

    async def write_sections(
        self,
        outline: OutlineT,
        ledger: EvidenceLedger,
        glossary: Mapping[str, str],
        targets: Sequence[int],
    ) -> tuple[list[tuple[SectionT, str]], tuple[SectionT, int] | None]:
        """Every section in order (the TL;DR set aside, written last), and
        the numbers each states tracked by the section that states them
        first."""
        written: list[tuple[SectionT, str]] = []
        numbers: dict[str, str] = {}
        tldr: tuple[SectionT, int] | None = None
        for section, target in zip(outline.sections, targets, strict=True):
            if self.spec.is_tldr(section.kind):
                tldr = (section, target)
                continue
            markdown = await self.write_section(
                outline, ledger, section, written, numbers, glossary, target
            )
            heading = section_heading(markdown) or section.title
            for n in section_numbers(markdown, self.free):
                numbers.setdefault(n, heading)
            written.append((section, markdown))
        return written, tldr

    async def write_section(
        self,
        outline: OutlineT,
        ledger: EvidenceLedger,
        section: SectionT,
        written: list[tuple[SectionT, str]],
        numbers: dict[str, str],
        glossary: Mapping[str, str],
        target: int,
    ) -> str:
        own = self.own(section, ledger)
        # What earlier sections actually state (their final text), so a
        # back-reference never points at a removed sentence.
        earlier_ids = self.cited(written)
        earlier = [e for eid in earlier_ids if (e := ledger.get(eid))]
        # Evidence a later section already referred back to once; a section
        # may always state its own.
        exhausted = {
            eid
            for s, md in written
            for eid in {i for snt in prose_sentences(md) for i in cited_ids(snt)}
            if eid not in s.evidence
        } - set(section.evidence)
        ctx = SectionContext(
            outline=outline,
            ledger=ledger,
            section=section,
            own=own,
            earlier_ids=earlier_ids,
            earlier=earlier,
            written=written,
            glossary=glossary,
            target=target,
            reference=self.reference_for(own),
        )
        draft = await self.draft(
            self.section_prompt(ctx),
            own,
            target,
            retry_hint=self.spec.retry_hint,
            context=post_so_far([md for _, md in written]),
        )
        self.reference = self.checking_reference(ctx.reference, draft)
        markdown = await self.verify_and_fix(
            draft,
            title=section.title,
            review_context=self.review_context(ctx),
            verifiers=self.section_verifiers(ctx, numbers, exhausted),
            ledger=ledger,
            own=own,
            earlier=earlier,
            free=self.free,
            section=section,
            judgement=section.kind in self.spec.judgement_kinds,
            rest="\n\n".join(md for _, md in written),
        )
        self.reference = ""
        return self.finish_section(section, markdown)

    def section_prompt(self, ctx: SectionContext[OutlineT, SectionT]) -> str:
        """The section writer's prompt: the shared skeleton with the kind's
        blocks."""
        section, outline = ctx.section, ctx.outline
        sections_map = "\n".join(
            f"- {'(this) ' if s is section else ''}{s.kind}: {s.title} — {s.thesis}"
            + (f" [teaches: {', '.join(t)}]" if (t := self.taught(s)) else "")
            for s in outline.sections
        )
        return "\n\n".join(
            [
                self.prompt_head(),
                block("outline", sections_map),
                block(
                    "section",
                    f"kind: {section.kind}\ntitle: {section.title}\nthesis: {section.thesis}\n"
                    f"question: {section.question}\ndepth: {section.depth}\n"
                    f"length: about {ctx.target:,} characters of prose\n"
                    f"running example: {outline.running_example or 'none'}"
                    + self.section_lines(outline),
                ),
                block("evidence", evidence_lines(ctx.own)),
                block(
                    "already_written_evidence",
                    evidence_lines(ctx.earlier[-_EARLIER_EVIDENCE:]),
                ),
                *self.naming_blocks(ctx),
                block(
                    "glossary",
                    glossary_view(dict(ctx.glossary), [m for _, m in ctx.written]),
                ),
                block(
                    "previous_section_end",
                    ctx.written[-1][1][-_PREVIOUS_END_CHARS:] if ctx.written else "",
                ),
                *self.closing_blocks(ctx),
            ]
        )

    def section_verifiers(
        self,
        ctx: SectionContext[OutlineT, SectionT],
        numbers: Mapping[str, str],
        exhausted: Collection[str],
    ) -> VerifierSet:
        """The verifier set of a section, built once: the shared code checks,
        then the kind's (``extra_problems``). What it reads of the post is
        fixed here (the numbers earlier sections state), so a later run of
        the set, after the whole post is written, judges the section as its
        own fixes were judged."""
        section = ctx.section
        figures = [e.figure_url for e in ctx.own if e.kind == "figure" and e.figure_url]
        depth = section.depth if section.kind in self.spec.depth_kinds else ""
        earlier_numbers = dict(numbers)
        spent = frozenset(exhausted)
        papers = tuple(self.named_papers(section))

        def shared(markdown: str) -> list[Problem]:
            return section_problems(
                markdown,
                title=section.title,
                own=section.evidence,
                earlier=ctx.earlier_ids,
                ledger=ctx.ledger,
                earlier_numbers=earlier_numbers,
                figure_urls=figures,
                free_numbers=self.free,
                passages=self.passages,
                exhausted=spent,
                papers=papers,
                target_chars=ctx.target,
                depth=depth,
            )

        def kind(markdown: str) -> list[Problem]:
            return as_problems(section.kind, self.extra_problems(ctx, markdown))

        return VerifierSet(section.title, (shared, kind))

    # ---- whole post -------------------------------------------------------------

    async def review_post(
        self,
        outline: OutlineT,
        ledger: EvidenceLedger,
        written: list[tuple[SectionT, str]],
    ) -> list[tuple[SectionT, str]]:
        """The spec's whole-post reviews read the post in order (run
        together), the kind's code findings join them, and each section they
        name is fixed."""
        passes = self.spec.post_passes
        if not passes or not written:
            return written
        with usage_stage("review"):
            reviews = await asyncio.gather(
                *(
                    self.model.structured(
                        role="evaluate",
                        system=self.post_prompts.passes[name],
                        prompt=self.pass_prompt(name, outline, ledger, written),
                        schema=PostReview,
                    )
                    for name in passes
                )
            )
        for name, review in zip(passes, reviews, strict=True):
            if review.trace:
                logger.info("Post review %s trace: %s", name, review.trace[:600])
        found = self.code_findings(written)
        issues = [i for r in reviews for i in r.issues] + found
        if not issues:
            return written
        logger.info(
            "Post review: %d issue(s) (%s), %d from the code check.",
            len(issues),
            ", ".join(
                f"{n}: {len(r.issues)}" for n, r in zip(passes, reviews, strict=True)
            ),
            len(found),
        )
        fixed = await self.apply_findings(
            written,
            issues,
            ledger,
            title_of=lambda s: s.title,
            own_of=lambda s: self.own(s, ledger),
            reference_of=self.reference_for,
        )
        self.after_review(fixed)
        return fixed

    async def write_tldr(
        self,
        outline: OutlineT,
        ledger: EvidenceLedger,
        section: SectionT,
        target: int,
        written: Sequence[tuple[SectionT, str]],
    ) -> str:
        """The TL;DR from its evidence, the evidence the sections cite and
        their headings and opening sentences: it may state what the body
        states (the one place a fact is said twice)."""
        tldr = self.spec.tldr
        assert tldr is not None
        pool = [*self.own(section, ledger)]
        pool += [e for eid in self.cited(written) if (e := ledger.get(eid))]
        # Figures stay in their sections.
        context = [e for e in pool if e.kind != "figure"]
        shown = (pool if tldr.figures_in_prompt else context)[:_TLDR_EVIDENCE]
        openings = "\n".join(
            f"- {section_heading(md) or s.title}: "
            + " ".join(prose_sentences(md)[:2])[:400]
            for s, md in written
        )
        draft = await self.draft(
            self.tldr_prompt(outline, section, target, openings, evidence_lines(shown)),
            shown,
            target,
            retry_hint=tldr.retry_hint,
            evidence=shown,
        )
        body = [s for _, md in written for s in prose_sentences(md)]

        def shared(md: str) -> list[Problem]:
            return section_problems(
                md,
                title=tldr.title,
                own=[e.id for e in context],
                earlier=(),
                ledger=ledger,
                earlier_numbers={},
                figure_urls=(),
                free_numbers=self.free,
                passages=self.passages,
                target_chars=target if tldr.checks_length else 0,
            )

        def kind(md: str) -> list[Problem]:
            problems = as_problems(tldr.kind, self.tldr_problems(md))
            if tldr.no_body_copies:
                problems += as_problems("body_copy", copied_sentences(md, body))
            return problems

        markdown = await self.verify_and_fix(
            draft,
            title=tldr.title,
            review_context=tldr.review_context,
            verifiers=VerifierSet(tldr.title, (shared, kind)),
            ledger=ledger,
            own=context,
            earlier=(),
            free=self.free,
            judgement=tldr.kind in self.spec.judgement_kinds,
            rest="\n\n".join(md for _, md in written),
        )
        markdown = _with_title(markdown, tldr.title)
        if not markdown.startswith("## "):
            markdown = f"## {tldr.title}\n\n{markdown}"
        return markdown

    # ---- hooks: outline ---------------------------------------------------------

    def outline_prompt(
        self, ledger: EvidenceLedger, duplicates: Mapping[str, str]
    ) -> str:
        """The outline writer's prompt (``duplicates``: evidence not offered)."""
        raise NotImplementedError

    def offered_duplicates(self, ledger: EvidenceLedger) -> dict[str, str]:
        """Evidence that repeats an earlier item's fact, not offered to the
        outline: duplicate id -> the id kept."""
        return {}

    def source_rank(self) -> Mapping[str, int] | None:
        """Source id -> rank (lower keeps a fact the outline marks as one)."""
        return None

    def repair(
        self, outline: OutlineT, ledger: EvidenceLedger
    ) -> tuple[OutlineT, list[str]]:
        """What code fixes in an outline, and what it changed."""
        raise NotImplementedError

    def outline_problems(self, outline: OutlineT, ledger: EvidenceLedger) -> list[str]:
        """What makes an outline unusable (the outline writer is told)."""
        raise NotImplementedError

    def cited_elsewhere(self, outline: OutlineT) -> set[str]:
        """Evidence the outline cites outside its sections (a table)."""
        return set()

    def canonical_elsewhere(
        self, outline: OutlineT, duplicates: Mapping[str, str]
    ) -> OutlineT:
        """``cited_elsewhere`` with each duplicate replaced by its kept id."""
        return outline

    def drop_elsewhere(self, outline: OutlineT, wrong: Collection[str]) -> OutlineT:
        """``cited_elsewhere`` without the ``wrong`` evidence."""
        return outline

    def code_language(self, outline: OutlineT) -> str:
        """The SDK language the post's code is in ("" when none)."""
        return ""

    def source_names(self, outline: OutlineT) -> dict[str, str]:
        """Source id -> the name the claim check shows for it."""
        return dict(self._names)

    def section_targets(self, outline: OutlineT) -> list[int]:
        """Prose characters per section."""
        return depth_targets(
            outline.sections,
            min_total=self.run.min_total_chars,
            max_total=self.run.max_total_chars,
            language=self.run.language,
        )

    # ---- hooks: sections --------------------------------------------------------

    def reference_for(self, own: Sequence[Evidence]) -> str:
        """The source passages a section's writer, checks and fixes read
        besides its evidence ("" when none)."""
        return ""

    def checking_reference(self, reference: str, draft: str) -> str:
        """The passages the checks and fixes of ``draft`` read."""
        return reference

    def prompt_head(self) -> str:
        """The first line of a section prompt (language and subject)."""
        raise NotImplementedError

    def taught(self, section: SectionT) -> Sequence[str]:
        """What a section teaches (shown in the outline map)."""
        return ()

    def section_lines(self, outline: OutlineT) -> str:
        """Lines appended to the prompt's section block."""
        return ""

    def naming_blocks(self, ctx: SectionContext[OutlineT, SectionT]) -> list[str]:
        """Prompt blocks that name the sources, after the evidence."""
        return []

    def closing_blocks(self, ctx: SectionContext[OutlineT, SectionT]) -> list[str]:
        """The prompt's last blocks."""
        return [block("source_sections", ctx.reference)]

    def review_context(self, ctx: SectionContext[OutlineT, SectionT]) -> str:
        """What the section reviewer is told about the section."""
        s = ctx.section
        return f"Section kind: {s.kind}; thesis: {s.thesis}; depth: {s.depth}."

    def named_papers(self, section: SectionT) -> Sequence[str]:
        """Papers a section must cite."""
        return ()

    def extra_problems(
        self, ctx: SectionContext[OutlineT, SectionT], markdown: str
    ) -> list[str]:
        """The kind's own problems of a section draft."""
        return []

    def finish_section(self, section: SectionT, markdown: str) -> str:
        """A written section with its heading: the outline title when the
        draft has none, and code's title for a kind the spec fixes."""
        if not markdown.lstrip().startswith("## "):
            return f"## {section.title}\n\n{markdown.lstrip()}"
        if section.kind in self.spec.fixed_titles:
            return _with_title(markdown, section.title)
        return markdown

    # ---- hooks: whole post and assembly -----------------------------------------

    def pass_prompt(
        self,
        name: str,
        outline: OutlineT,
        ledger: EvidenceLedger,
        written: Sequence[tuple[SectionT, str]],
    ) -> str:
        """The prompt of the whole-post review ``name``."""
        raise NotImplementedError

    def code_findings(
        self, written: Sequence[tuple[SectionT, str]]
    ) -> list[PostFinding]:
        """Whole-post problems code finds, fixed with the reviews'."""
        return []

    def after_review(self, written: Sequence[tuple[SectionT, str]]) -> None:
        """Checks of the post after the whole-post fixes."""

    def tldr_prompt(
        self,
        outline: OutlineT,
        section: SectionT,
        target: int,
        openings: str,
        evidence: str,
    ) -> str:
        """The TL;DR writer's prompt."""
        raise NotImplementedError

    def tldr_problems(self, markdown: str) -> list[str]:
        """The kind's own problems of a TL;DR draft."""
        return []

    async def assemble(
        self,
        outline: OutlineT,
        ledger: EvidenceLedger,
        written: list[tuple[SectionT, str]],
        tldr: str | None,
    ) -> ResultT:
        """The kind's result from the written sections and the TL;DR."""
        raise NotImplementedError
