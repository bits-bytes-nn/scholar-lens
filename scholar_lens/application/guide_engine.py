"""The evidence-based guide writer (docs/redesign.md, 3.2 and 5.1): the
guide's spec and hooks bound to the post engine.

Read (a bounded agent per source page, records checked against the page and
its authority) → outline (each piece of evidence to one section, in the
guide's order) → draft each section from its evidence → verify and review →
fix only the named sentences → the TL;DR last, from what the sections say →
assemble without the evidence markers (a guide lists its sources at the
end). Ports only: the model and the evidence cache are injected.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from scholar_lens.application.checkpoints import input_hash
from scholar_lens.application.evidence_writer import (
    WriterPrompts,
    block,
    brief_lines,
    quote_for_prompt_code,
    safe_key,
)
from scholar_lens.application.ports import (
    AgentLimits,
    CheckpointStore,
    LanguageModel,
    TextStore,
)
from scholar_lens.application.post_engine import (
    PostEngine,
    PostPrompts,
    ReadJob,
    ReadStart,
    RunSettings,
    SectionContext,
)
from scholar_lens.domain.evidence import (
    Evidence,
    EvidenceLedger,
    SourceDocument,
    duplicate_evidence,
)
from scholar_lens.domain.evidence_draft import free_numbers
from scholar_lens.domain.guide.evidence_guide import (
    TLDR_TITLE,
    GuideOutline,
    GuideSection,
    cited_sources,
    code_definitions,
    guide_draft_problems,
    guide_outline_problems,
    name_sources,
    repair_guide_outline,
    strip_markers,
)
from scholar_lens.domain.guide.sdk_index import SdkIndex, definitions_for
from scholar_lens.domain.prompts import guide_evidence as prompts
from scholar_lens.domain.specs.guide import GUIDE_SPEC

logger = logging.getLogger(__name__)

# Bump when the reader's prompt or the evidence schema changes meaning.
GUIDE_READER_VERSION = 9
# Small counts and ordinals a guide states without evidence.
_FREE_NUMBERS = free_numbers()

GUIDE_PROMPTS = WriterPrompts(
    reader=prompts.READER_SYSTEM,
    writer=prompts.WRITER_SYSTEM,
    claim_check=prompts.CLAIM_CHECK_SYSTEM,
    evidence_check=prompts.EVIDENCE_CHECK_SYSTEM,
    review=prompts.REVIEW_SYSTEM,
    fix=prompts.FIX_SYSTEM,
)
_POST_PROMPTS = PostPrompts(
    outline=prompts.OUTLINE_SYSTEM,
    subject=prompts.PASS_SUBJECT,
    code_trace=prompts.CODE_TRACE,
    tail=prompts.PASS_TAIL,
)
_PRIMARY = ("official", "reference")
_AUTHORITY_RANK = {"official": 0, "reference": 1, "third_party": 2, "older_docs": 3}
# The product's own definitions shown to a section's checks and fixes.
_SDK_SOURCE_CHARS = 12_000


# Definitions in a code block: what later code of the running example calls.
@dataclass(frozen=True)
class GuideSettings:
    language: str
    title_max_length: int
    # Cache-key part naming the reading model.
    reader_tag: str
    max_sections: int = 14
    max_fix_rounds: int = GUIDE_SPEC.limits.fix_rounds
    # Agent turns per page: the docs and the user's material are read in
    # depth, search results for what they add.
    read_turns_primary: int = 24
    read_turns_other: int = 10
    read_tokens_per_source: int = 400_000
    # Pages read (the docs and the user's material first).
    max_sources: int = 40
    concurrency: int = 4
    outline_repairs: int = GUIDE_SPEC.limits.outline_repairs
    # Source text handed to one section's writer and checks.
    source_chars: int = 40_000
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
    # Problems the post may not be published with as is (the PR lists
    # them first, under Blocking).
    blocking: list[str] = field(default_factory=list)


def _rank(document: SourceDocument | None) -> int:
    return _AUTHORITY_RANK.get(
        (document.authority if document else "") or "third_party", 2
    )


class EvidenceGuideEngine(PostEngine[GuideOutline, GuideSection, EvidenceGuide]):
    def __init__(
        self,
        model: LanguageModel,
        settings: GuideSettings,
        store: TextStore | None = None,
        *,
        checkpoints: CheckpointStore | None = None,
        identifier_check: Callable[[str], list[str]] | None = None,
        products: Sequence[str] = (),
        package_facts: str = "",
        sdk: SdkIndex | None = None,
    ) -> None:
        super().__init__(
            model,
            GUIDE_SPEC,
            GUIDE_PROMPTS,
            _POST_PROMPTS,
            RunSettings(
                language=settings.language,
                concurrency=settings.concurrency,
                outline_repairs=settings.outline_repairs,
                min_total_chars=settings.min_total_chars,
                max_total_chars=settings.max_total_chars,
            ),
            outline_schema=GuideOutline,
            store=store,
            checkpoints=checkpoints,
            max_fix_rounds=settings.max_fix_rounds,
        )
        self.settings = settings
        self.free = _FREE_NUMBERS
        self._documents: dict[str, SourceDocument] = {}
        self._topic = ""
        self._guideline = ""
        # A digest of the pages read (set by write; checkpoint input).
        self._sources = ""
        # Flags and arguments of a draft that no official page mentions.
        self.identifier_check = identifier_check
        # The product's names (never a source's name).
        self.products: tuple[str, ...] = tuple(products)
        # The packages the docs install, their versions and extras (PyPI).
        self.package_facts = package_facts
        # The product's own code: what its definitions say decides defaults,
        # signatures and return values over a page.
        self.sdk = sdk

    # ---- source passages ------------------------------------------------------

    def _sdk_source(self, markdown: str) -> str:
        """The product's own definitions of what ``markdown`` uses, as a
        block for the checks (empty without the package's code)."""
        if self.sdk is None:
            return ""
        found = definitions_for(markdown, self.sdk, _SDK_SOURCE_CHARS)
        return f"\n\n[sdk_source: {self.sdk.version}]\n{found}" if found else ""

    def source_sections(self, own: Sequence[Evidence]) -> str:
        """The full source sections the evidence comes from, official pages
        first, up to ``source_chars``: complete code (imports, whole
        functions) and context the evidence snippets cut."""
        seen: list[tuple[str, str]] = []
        for e in own:
            if (e.paper_id, e.anchor) not in seen:
                seen.append((e.paper_id, e.anchor))
        seen.sort(key=lambda pa: _rank(self._documents.get(pa[0])))
        parts: list[str] = []
        left = self.settings.source_chars
        for sid, anchor in seen:
            document = self._documents.get(sid)
            section = document.section(anchor) if document else None
            if document is None or section is None or left <= 0:
                continue
            if document.authority not in _PRIMARY:
                # A third-party page's identifiers are often stale; its
                # evidence says what the guide may take from it.
                continue
            text = quote_for_prompt_code(section.text)[:left]
            left -= len(text)
            parts.append(
                f"[{sid} {anchor}: {document.title} | {section.heading}"
                f" | {document.label()}]\n{text}"
            )
        return "\n\n".join(parts)

    # ---- read -------------------------------------------------------------------

    def _cache_key(self, document: SourceDocument) -> str:
        page = hashlib.sha256(document.url.encode()).hexdigest()[:16]
        text = hashlib.sha256(document.text.encode()).hexdigest()[:12]
        return (
            f"evidence/{page}-{text}/v{GUIDE_READER_VERSION}-{self.settings.reader_tag}-"
            f"{safe_key(self.settings.language)}-{document.authority}.json"
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

        def start(document: SourceDocument) -> ReadStart:
            async def job() -> ReadJob:
                primary = document.authority in _PRIMARY
                main = self.limits.main_source_evidence
                minor = self.limits.minor_source_evidence
                prompt = "\n\n".join(
                    [
                        f"Guide topic: {topic}",
                        f"Page {document.paper_id}: {document.title} ({document.url}); "
                        f"authority: {document.label()}.",
                        block("outline", document.outline()),
                        f"Guide language for claims and cells: {self.settings.language}.",
                        (
                            f"Record about {main[0]}-{main[1]} pieces of evidence."
                            if primary
                            else f"Record about {minor[0]}-{minor[1]} pieces of "
                            "evidence: what this page adds to the docs."
                        ),
                    ]
                )
                return ReadJob(
                    document=document,
                    key=self._cache_key(document),
                    prompt=prompt,
                    limits=AgentLimits(
                        turns=(
                            self.settings.read_turns_primary
                            if primary
                            else self.settings.read_turns_other
                        ),
                        total_tokens=self.settings.read_tokens_per_source,
                    ),
                    # A docs page's diagrams are its most useful visuals and
                    # the reader tends to skip them: ask for one.
                    required=("figure",) if primary and document.figures else (),
                )

            return job

        results = await self.read_all(
            [start(d) for d in documents],
            corpus={d.authority for d in documents},
        )
        return self.ledger_of(items for _, items in results)

    # ---- the guide ----------------------------------------------------------------

    async def write(
        self,
        topic: str,
        documents: Sequence[SourceDocument],
        *,
        guideline: str = "",
    ) -> EvidenceGuide:
        documents = self.selected(documents)
        self._topic, self._guideline = topic, guideline
        self._documents = {d.paper_id: d for d in documents}
        self._sources = input_hash([d.model_dump(mode="json") for d in documents])
        self.passages = {
            (d.paper_id, s.anchor): s.text for d in documents for s in d.sections
        }
        # The claim check sees which kind of source a fact comes from.
        self._names = {d.paper_id: f"{d.title} [{d.label()}]" for d in documents}
        # The docs' main topics and the user's material: the readers record
        # their evidence first.
        self.targets = await self.key_targets(documents)
        ledger = await self.read(topic, documents)
        if not len(ledger):
            raise ValueError("No source page gave usable evidence for the guide.")
        return await self.compose(ledger)

    def checkpoint_inputs(self) -> object:
        # The checks read the pages (passages, the flags they mention), the
        # product's names, its packages and its code besides the prompt.
        sdk = self.sdk
        return {
            "settings": self.settings,
            "sources": self._sources,
            "guideline": self._guideline,
            "products": list(self.products),
            "package_facts": self.package_facts,
            "sdk": (
                [sdk.version, sorted(sdk.packages), sdk.partial, len(sdk.modules)]
                if sdk is not None
                else None
            ),
        }

    # ---- hooks: outline -----------------------------------------------------------

    def offered_duplicates(self, ledger: EvidenceLedger) -> dict[str, str]:
        # A fact several pages state is offered once, from the most
        # authoritative page.
        return duplicate_evidence(ledger, dict(self.source_rank()))

    def source_rank(self) -> Mapping[str, int]:
        return {sid: _rank(d) for sid, d in self._documents.items()}

    def outline_prompt(
        self, ledger: EvidenceLedger, duplicates: Mapping[str, str]
    ) -> str:
        sources = "\n".join(
            f"- {sid} | {d.label()} | {d.title} | {d.url}"
            for sid, d in self._documents.items()
        )
        return "\n\n".join(
            [
                f"Topic: {self._topic}\nLanguage: {self.settings.language}",
                f"Title: in English, at most {self.settings.title_max_length} characters. "
                f"At most {self.settings.max_sections} sections.",
                block("sources", sources),
                block(
                    "evidence",
                    brief_lines([e for e in ledger if e.id not in duplicates]),
                ),
                block("translation_guideline", self._guideline),
            ]
        )

    def repair(
        self, outline: GuideOutline, ledger: EvidenceLedger
    ) -> tuple[GuideOutline, list[str]]:
        return repair_guide_outline(outline, ledger)

    def outline_problems(
        self, outline: GuideOutline, ledger: EvidenceLedger
    ) -> list[str]:
        return guide_outline_problems(
            outline,
            ledger,
            self._documents,
            title_max_length=self.settings.title_max_length,
            max_sections=self.settings.max_sections,
        )

    def code_language(self, outline: GuideOutline) -> str:
        return outline.code_language

    # ---- hooks: sections ----------------------------------------------------------

    def reference_for(self, own: Sequence[Evidence]) -> str:
        return self.source_sections(own)

    def checking_reference(self, reference: str, draft: str) -> str:
        return reference + self._sdk_source(draft)

    def prompt_head(self) -> str:
        return f"Language: {self.settings.language}. Topic: {self._topic}."

    def taught(self, section: GuideSection) -> Sequence[str]:
        return section.topics

    def section_lines(self, outline: GuideOutline) -> str:
        return f"\ncode language: {outline.code_language or 'the docs’ main one'}"

    def naming_blocks(
        self, ctx: SectionContext[GuideOutline, GuideSection]
    ) -> list[str]:
        return [
            block(
                "source_names",
                "\n".join(f"- {sid}: {name}" for sid, name in self._names.items()),
            )
        ]

    def review_context(self, ctx: SectionContext[GuideOutline, GuideSection]) -> str:
        section, outline = ctx.section, ctx.outline
        return (
            f"Section kind: {section.kind}; thesis: {section.thesis}; "
            f"depth: {section.depth}; running example: {outline.running_example or 'none'}.\n"
            + block(
                "code_defined_earlier",
                "\n".join(code_definitions([m for _, m in ctx.written])),
            )
        )

    def extra_problems(
        self, ctx: SectionContext[GuideOutline, GuideSection], markdown: str
    ) -> list[str]:
        problems = guide_draft_problems(
            markdown, code_language=ctx.outline.code_language
        )
        if self.identifier_check is not None and (
            unknown := self.identifier_check(markdown)
        ):
            problems.append(
                "flags or arguments no official page mentions; use the "
                "documented ones or leave them out: " + ", ".join(unknown)
            )
        return problems

    # ---- hooks: whole guide -------------------------------------------------------

    def code_reference(self, ledger: EvidenceLedger, post: str) -> list[str]:
        """The official docs' code, the product's definitions of what the
        guide uses and its packages: what the consistency read checks the
        guide's code against (an import an earlier block lacked, a handler
        that never returns its result, an install extra missing)."""
        source_code = "\n\n".join(
            f"[{e.paper_id}] {e.claim}\n{quote_for_prompt_code(e.code or '')}"
            for e in ledger
            if e.kind == "code"
            and (d := self._documents.get(e.paper_id))
            and d.authority == "official"
        )[: self.settings.source_chars]
        return [
            block("source_code", source_code),
            block("sdk_source", self._sdk_source(post)),
            block("packages", self.package_facts),
        ]

    def tldr_prompt(
        self,
        outline: GuideOutline,
        section: GuideSection,
        target: int,
        openings: str,
        evidence: str,
    ) -> str:
        return "\n\n".join(
            [
                f"Language: {self.settings.language}. Topic: {self._topic}.",
                block(
                    "section",
                    f"kind: tldr\ntitle: {TLDR_TITLE}\nthesis: {section.thesis}\n"
                    f"length: about {target:,} characters of prose, one or two "
                    "paragraphs, no list, no code, no figure",
                ),
                block("guide_sections", openings),
                block("evidence", evidence),
            ]
        )

    def tldr_problems(self, markdown: str) -> list[str]:
        return guide_draft_problems(markdown)

    async def assemble(
        self,
        outline: GuideOutline,
        ledger: EvidenceLedger,
        written: list[tuple[GuideSection, str]],
        tldr: str | None,
    ) -> EvidenceGuide:
        parts = [md for _, md in written]
        if tldr is not None:
            parts.insert(0, tldr)
        body = "\n\n".join(p.strip() for p in parts if p.strip())
        by_id = self._documents
        cited = [by_id[s] for s in cited_sources(body, ledger) if s in by_id]
        return EvidenceGuide(
            title=outline.title,
            body=name_sources(
                strip_markers(body), {sid: d.title for sid, d in by_id.items()}
            ),
            outline=outline,
            ledger=ledger,
            cited=cited,
            notes=self.notes,
            blocking=self.blocking,
        )
