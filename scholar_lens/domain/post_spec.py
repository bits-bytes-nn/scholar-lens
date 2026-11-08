"""The declarative description of a post kind (docs/engine.md, 4.0).

A ``PostSpec`` is data: the section kinds and their order and counts, the
titles code fixes, the evidence each kind may use, and the limits of the
engine's loops. The engine reads a spec and never branches on the post
kind; what a kind adds beyond data is a named hook of its own module.

This module also holds the pieces every kind shares: the common section
fields, the outline checks driven by a spec, and the reader's evidence
policy.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field

from scholar_lens.domain.evidence import EvidenceLedger, SourceDocument
from scholar_lens.domain.lengths import DEPTH_CHARS, scaled_targets

Depth = Literal["deep", "standard", "brief"]

# Languages of one SDK family: evidence that holds for one of them only is
# not stated in a post whose code is in another.
SDK_LANGUAGES = {
    "python": "python",
    "py": "python",
    "typescript": "typescript",
    "ts": "typescript",
    "javascript": "javascript",
    "js": "javascript",
    "java": "java",
    "go": "go",
    "rust": "rust",
}
_CODE_SPAN = re.compile(r"`[^`\n]+`")


# ---- the shared models -------------------------------------------------------


# The fields every outline section has, in the order its schema shows them;
# a kind narrows ``kind`` to its own literal and adds its fields. No
# docstring: a subclass's schema stays its own.
class SectionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    # A claim as a noun phrase, not a topic label.
    title: str = Field(min_length=1)
    thesis: str = ""
    question: str = ""
    depth: Depth = "standard"
    evidence: list[str] = Field(default_factory=list)


class PostSection(Protocol):
    """What the engine reads of an outline section (every kind's)."""

    @property
    def kind(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def thesis(self) -> str: ...

    @property
    def question(self) -> str: ...

    @property
    def depth(self) -> str: ...

    @property
    def evidence(self) -> list[str]: ...

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self: ...


class PostOutline(Protocol):
    """What the engine reads of an outline (every kind's)."""

    @property
    def sections(self) -> Sequence[Any]: ...

    @property
    def glossary(self) -> Sequence[Any]: ...

    @property
    def running_example(self) -> str: ...

    @property
    def same_facts(self) -> list[list[str]]: ...

    @property
    def coverage(self) -> Mapping[str, list[str]]: ...

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self: ...


# ---- the spec ----------------------------------------------------------------


@dataclass(frozen=True)
class KindSpec:
    """One section kind: its rank in the post's order (kinds of one rank
    may interleave)."""

    name: str
    rank: int


@dataclass(frozen=True)
class KindCount:
    """A rule on how often a kind occurs and where; ``message`` is what the
    outline writer is told when it is broken."""

    kind: str
    count: Literal["exactly_one", "at_least_one", "at_most_one", "any"]
    message: str
    position: Literal["", "first", "last"] = ""


@dataclass(frozen=True)
class KindOrder:
    """The sections follow the kinds' ranks."""

    message: str


@dataclass(frozen=True)
class KindSequence:
    """The sections, ``skip`` kinds aside, are exactly ``kinds`` in order."""

    kinds: tuple[str, ...]
    message: str
    skip: tuple[str, ...] = ()


@dataclass(frozen=True)
class TitleLength:
    """The post title fits the configured length."""


@dataclass(frozen=True)
class MaxSections:
    """The outline has at most the configured number of sections."""


OutlineCheck = KindCount | KindOrder | KindSequence | TitleLength | MaxSections


@dataclass(frozen=True)
class Ownership:
    """How each piece of evidence belongs to one section, and what the
    outline writer is told when it does not. Fields of the messages:
    ``{eid}``, ``{title}`` and, for ``twice``, ``{first}``."""

    unknown: str = "unknown evidence {eid} in '{title}'"
    twice: str = "{eid} is in both '{first}' and '{title}'"
    # Reported for a section without evidence when not empty (kinds in
    # ``exempt`` excepted).
    empty: str = ""
    exempt: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuthorityRule:
    """Evidence of ``kinds`` (every kind when empty, ``exempt`` excepted)
    from a source whose authority is in ``rejected`` is refused with
    ``message``. ``when_corpus_has``: the rule holds only when some source
    of the run has that authority. ``names_identifier``: only when the
    claim or cell names an identifier (a code span)."""

    rejected: frozenset[str]
    message: str
    kinds: frozenset[str] = frozenset()
    exempt: frozenset[str] = frozenset()
    when_corpus_has: str = ""
    names_identifier: bool = False

    def applies(
        self,
        document: SourceDocument,
        kind: str,
        fields: Mapping[str, object],
        corpus: Collection[str],
    ) -> bool:
        if self.kinds and kind not in self.kinds:
            return False
        if kind in self.exempt or document.authority not in self.rejected:
            return False
        if self.when_corpus_has and self.when_corpus_has not in corpus:
            return False
        if self.names_identifier:
            text = f"{fields.get('claim', '')} {fields.get('cell', '')}"
            return bool(_CODE_SPAN.search(text))
        return True


@dataclass(frozen=True)
class RequiredField:
    """Evidence of ``kinds`` needs ``name`` filled; ``message`` may name the
    ``{kind}``."""

    kinds: frozenset[str]
    name: str
    message: str


@dataclass(frozen=True)
class EvidencePolicy:
    """What a reader may record for a post kind (beyond the checks every
    record passes against its source)."""

    forbidden_kinds: frozenset[str] = frozenset()
    # Told to the reader for a forbidden kind; may name the ``{kind}``.
    forbidden_message: str = ""
    required_fields: tuple[RequiredField, ...] = ()
    authority_rules: tuple[AuthorityRule, ...] = ()
    # Whether cached evidence is held to the policy again when reused.
    recheck_cached: bool = False
    # Whether evidence that holds for one SDK language only leaves an
    # outline whose code is in another.
    same_code_language: bool = False


@dataclass(frozen=True)
class TldrSpec:
    """The TL;DR, written last from what the sections came to say."""

    kind: str
    title: str
    retry_hint: str
    review_context: str
    # Whether the prompt's evidence keeps figures (the checks never do).
    figures_in_prompt: bool = False
    # Whether the TL;DR is held to its target length.
    checks_length: bool = False
    # Whether a TL;DR sentence that copies a body sentence is a problem.
    no_body_copies: bool = False


@dataclass(frozen=True)
class TargetKind:
    """A kind of key target and what it is (the extraction is told).
    ``headline``: how many targets of the kind, in the extraction's order
    (most important first), the TL;DR must state (0: none)."""

    name: str
    meaning: str
    headline: int = 0


@dataclass(frozen=True)
class CoverageSpec:
    """Where a post's key targets come from (docs/engine.md, 4.6).

    One structured call extracts ``count`` targets from the documents whose
    authority is in ``authorities`` (every document when empty), reading
    the sections whose heading names one of ``sections`` (each document's
    outline when empty); every document whose authority is in
    ``per_document`` is a target of its own, in code. A kind that derives
    its targets from its plan (no model call) extracts none: ``count`` is
    ``(0, 0)``."""

    kinds: tuple[TargetKind, ...] = ()
    count: tuple[int, int] = (0, 0)
    authorities: frozenset[str] = frozenset()
    sections: tuple[str, ...] = ()
    per_document: frozenset[str] = frozenset()
    # The kind of a per-document target, and what it is.
    document_kind: TargetKind = TargetKind(
        "document", "a document the user chose; the post uses it"
    )

    @property
    def headline_kinds(self) -> dict[str, int]:
        return {k.name: k.headline for k in self.kinds if k.headline}


PassName = Literal["consistency", "repetition", "coverage"]


@dataclass(frozen=True)
class PostPasses:
    """The whole-post steps, run in this order after the sections
    (docs/engine.md, 4.7): ``consistency`` and ``repetition`` are model
    reads of the whole post, run together; ``coverage`` is code."""

    names: tuple[PassName, ...] = ("consistency", "repetition", "coverage")
    # The consistency read also traces one run of the running example
    # through the post and checks its code against the sources' code, and
    # code checks the program the runnable sections build (a post whose
    # code a reader pastes and runs).
    code_trace: bool = False


@dataclass(frozen=True)
class Limits:
    """Defaults of the engine's loops (settings may override them)."""

    outline_repairs: int = 2
    fix_rounds: int = 2
    # The largest share of a section's prose one revision may cut: a fix
    # that cuts more rewrote the section instead of fixing it.
    max_prose_cut: float = 0.25
    # The largest share of a section's claim units code deletes as
    # unsupported. The guard catches a claim check that misread the section
    # (or its evidence) and flagged most of it, so nothing is deleted and the
    # post is blocked for a reader; at a half, a short section losing one of
    # three sentences is still an ordinary deletion.
    max_drop_share: float = 0.5
    # Earlier sections the next section's writer reads in full (the rest
    # it sees as the "already taught" list), and the characters of an
    # evidence claim that list shows.
    previous_sections: int = 1
    taught_claim_chars: int = 200
    # Source characters the key-target extraction reads.
    coverage_source_chars: int = 30_000
    # Evidence items a TL;DR prompt shows at most (the body cites more).
    tldr_evidence: int = 80
    # Earlier sections' evidence a fix sees at most besides the section's
    # own: what the section cites first, then the latest. The claim check
    # judges against the earlier evidence, so a fix must see what it cites.
    fix_earlier_evidence: int = 30
    # A draft shorter than this share of its target prose is asked again.
    short_draft_share: float = 0.3
    # Calls of a structured per-item check (claims, evidence): the first,
    # then one for the items its answer left out; an item still without a
    # verdict blocks the post.
    answer_asks: int = 2
    # The reader: turns of the follow-up that asks for a required kind it
    # skipped, how often ``finish`` refuses while one is missing, and the
    # characters ``read_section`` returns of one section.
    follow_up_turns: int = 8
    required_kind_asks: int = 2
    read_section_chars: int = 30_000
    # Evidence a reader is asked to record (low, high): of a main source
    # (the docs, a core paper, the paper) and of a minor one (another page,
    # a supporting paper).
    main_source_evidence: tuple[int, int] = (10, 25)
    minor_source_evidence: tuple[int, int] = (3, 10)
    # A section whose prose runs past this multiple of its target is too
    # long: half again leaves room for a dense section without letting one
    # take a neighbour's share of the post.
    max_length_share: float = 1.5
    # Visuals (images, display equations, tables) a section of the spec's
    # ``depth_kinds`` needs, by depth: a deep section shows its mechanism
    # and a result, a standard one one of them.
    min_visuals: tuple[tuple[str, int], ...] = (
        ("deep", 2),
        ("standard", 1),
        ("brief", 0),
    )


@dataclass(frozen=True)
class PostSpec:
    """One post kind as data (docs/engine.md, 4.0)."""

    # What the run's messages call the outline ("guide outline").
    outline_name: str
    kinds: tuple[KindSpec, ...]
    outline_checks: tuple[OutlineCheck, ...]
    ownership: Ownership
    evidence: EvidencePolicy
    # Told to the section writer when a draft comes back short.
    retry_hint: str
    tldr: TldrSpec | None = None
    # Kind -> language key ("Korean", "" for English) -> the title code
    # gives the section; a kind listed keeps its outline title as heading.
    fixed_titles: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    # Kinds whose drafts are held to their depth's visuals.
    depth_kinds: frozenset[str] = frozenset()
    # Kinds whose code a reader runs (the running example).
    runnable_kinds: frozenset[str] = frozenset()
    # Kinds whose sentences are the post's own assessment (a closing
    # judgement, when to choose a tool): an unsupported one is shown to the
    # reviewer, never deleted.
    judgement_kinds: frozenset[str] = frozenset()
    # Sections the checks emptied: kept, or dropped with a note.
    empty_sections: Literal["keep", "drop"] = "keep"
    # The whole-post steps and their options.
    post_passes: PostPasses = PostPasses()
    # Where the key targets come from.
    coverage: CoverageSpec = CoverageSpec()
    # Kind -> the block code builds and appends after the section (a table).
    attachments: Mapping[str, str] = field(default_factory=dict)
    limits: Limits = Limits()
    # The kind's name in stage checkpoint keys ("guide", "review").
    name: str = "post"

    @property
    def ranks(self) -> dict[str, int]:
        return {k.name: k.rank for k in self.kinds}

    def is_tldr(self, kind: str) -> bool:
        return self.tldr is not None and kind == self.tldr.kind


# ---- the evidence policy -----------------------------------------------------


def evidence_policy(
    policy: EvidencePolicy,
    document: SourceDocument,
    kind: str,
    fields: Mapping[str, object],
    *,
    corpus: Collection[str] = (),
) -> list[str]:
    """Why a record of ``kind`` from ``document`` is refused under
    ``policy`` (its first reason), or nothing.

    Args:
        policy: The post kind's evidence policy.
        document: The source the record comes from.
        kind: The record's evidence kind.
        fields: The record's fields (claim, cell, ...).
        corpus: The authorities of the run's sources.

    Returns:
        At most one problem, told to the reader as is.
    """
    if kind in policy.forbidden_kinds:
        return [policy.forbidden_message.format(kind=kind)]
    for required in policy.required_fields:
        if kind in required.kinds and not str(fields.get(required.name, "")).strip():
            return [required.message.format(kind=kind)]
    for rule in policy.authority_rules:
        if rule.applies(document, kind, fields, corpus):
            return [rule.message]
    return []


def other_sdk(evidence_sdk: str, code_language: str) -> bool:
    """Whether evidence recorded as holding for one SDK language holds for
    another one than the post's code language (its reader's)."""
    mine = SDK_LANGUAGES.get(code_language.strip().lower(), "")
    theirs = SDK_LANGUAGES.get(evidence_sdk.strip().lower(), "")
    return bool(mine and theirs and mine != theirs)


# ---- outline checks ----------------------------------------------------------


def _count_ok(count: str, n: int) -> bool:
    if count == "exactly_one":
        return n == 1
    if count == "at_least_one":
        return n >= 1
    if count == "at_most_one":
        return n <= 1
    return True


def structure_problems(
    spec: PostSpec,
    kinds: Sequence[str],
    *,
    title: str = "",
    title_max_length: int = 0,
    max_sections: int = 0,
) -> list[str]:
    """The outline problems ``spec.outline_checks`` find in the sections'
    ``kinds``, in the checks' order.

    Args:
        spec: The post kind.
        kinds: The outline's section kinds, in order.
        title: The post title (for ``TitleLength``).
        title_max_length: The longest title allowed.
        max_sections: The most sections allowed.

    Returns:
        One message per broken check.
    """
    problems: list[str] = []
    ranks = spec.ranks
    for check in spec.outline_checks:
        if isinstance(check, TitleLength):
            if len(title) > title_max_length:
                problems.append(
                    f"the title is longer than {title_max_length} characters"
                )
        elif isinstance(check, MaxSections):
            if len(kinds) > max_sections:
                problems.append(f"at most {max_sections} sections")
        elif isinstance(check, KindCount):
            placed = (
                check.position == "first" and (not kinds or kinds[0] != check.kind)
            ) or (check.position == "last" and (not kinds or kinds[-1] != check.kind))
            if placed or not _count_ok(check.count, list(kinds).count(check.kind)):
                problems.append(check.message)
        elif isinstance(check, KindOrder):
            order = [ranks[k] for k in kinds]
            if order != sorted(order):
                problems.append(check.message)
        elif isinstance(check, KindSequence):
            if [k for k in kinds if k not in check.skip] != list(check.kinds):
                problems.append(check.message)
    return problems


def ownership_problems(
    spec: PostSpec, sections: Sequence[PostSection], ledger: EvidenceLedger
) -> list[str]:
    """Sections without evidence (when the spec reports them), unknown
    evidence, and evidence given to two sections, in section order.

    Args:
        spec: The post kind.
        sections: The outline's sections.
        ledger: The run's evidence.

    Returns:
        One message per problem, worded by ``spec.ownership``.
    """
    rules = spec.ownership
    problems: list[str] = []
    seen: dict[str, str] = {}
    for s in sections:
        if rules.empty and not s.evidence and s.kind not in rules.exempt:
            problems.append(rules.empty.format(title=s.title))
        for eid in s.evidence:
            if ledger.get(eid) is None:
                problems.append(rules.unknown.format(eid=eid, title=s.title))
            elif eid in seen:
                problems.append(
                    rules.twice.format(eid=eid, first=seen[eid], title=s.title)
                )
            else:
                seen[eid] = s.title
    return problems


def first_owner(
    evidence: Sequence[str],
    ledger: EvidenceLedger,
    seen: set[str],
    title: str,
    notes: list[str],
) -> list[str]:
    """A section's ``evidence`` without unknown ids and ids an earlier
    section (``seen``, updated) already has; what was dropped goes to
    ``notes``."""
    kept = []
    for eid in evidence:
        if ledger.get(eid) is None:
            notes.append(f"unknown evidence {eid} dropped from '{title}'")
        elif eid in seen:
            notes.append(f"{eid} given twice; kept in its first section")
        else:
            seen.add(eid)
            kept.append(eid)
    return kept


def depth_targets(
    sections: Sequence[PostSection],
    *,
    min_total: int,
    max_total: int,
    language: str,
) -> list[int]:
    """Prose characters per section by depth, the total scaled into
    ``[min_total, max_total]`` (Korean characters).

    Args:
        sections: The outline's sections.
        min_total: The shortest post (Korean characters of prose).
        max_total: The longest post.
        language: The post's language.

    Returns:
        One target per section, in ``language``'s characters.
    """
    return scaled_targets(
        [DEPTH_CHARS[s.depth] for s in sections],
        min_total=min_total,
        max_total=max_total,
        language=language,
    )
