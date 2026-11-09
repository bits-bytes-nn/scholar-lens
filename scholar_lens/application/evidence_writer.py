"""The evidence engine's shared core (docs/redesign.md, 3.2-3.3).

Whatever the artifact (a survey, a guide), a source is read by a bounded
agent whose records are checked against the source before they enter the
ledger; a section is drafted from its evidence, checked in code and claim by
claim, reviewed for located reader problems, and fixed sentence by sentence.
The artifact engines own their outline and prompts; this module owns the
loop. Ports only: the model is injected.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import re
import threading
from collections import Counter
from collections.abc import (
    Awaitable,
    Callable,
    Collection,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError

from scholar_lens.application.checkpoints import WriterState
from scholar_lens.application.ports import (
    AgentLimits,
    LanguageModel,
    ModelText,
    was_cut,
)
from scholar_lens.domain.edit_transaction import Revision, rejudged_units, transact
from scholar_lens.domain.evidence import (
    Evidence,
    EvidenceLedger,
    SourceDocument,
    canonical_ids,
    evidence_problems,
    near_duplicate,
)
from scholar_lens.domain.evidence_draft import (
    TOO_SHORT_PROBLEM,
    add_missing_figures,
    balance_math,
    cited_ids,
    claim_units,
    drop_sentences,
    ends_mid_sentence,
    heading_names,
    is_illustration,
    merge_thin_subsections,
    prose_length,
    quote_for_prompt,
    section_heading,
    unsourced_number_sentences,
)
from scholar_lens.domain.markdown.lint import drop_empty_headings
from scholar_lens.domain.markdown.math import unglue_fences
from scholar_lens.domain.post_spec import Limits
from scholar_lens.domain.prompts.rules import FIX_ANSWER
from scholar_lens.domain.usage_context import usage_stage
from scholar_lens.domain.verifiers import Problem, VerifierSet, messages

from ..domain.prompt_safety import fenced

logger = logging.getLogger(__name__)

SectionT = TypeVar("SectionT")

_EVIDENCE_KINDS: tuple[str, ...] = Evidence.model_fields["kind"].annotation.__args__  # type: ignore[union-attr]


class SentenceVerdict(BaseModel):
    n: int = Field(description="The numbered item this verdict is for.")
    # about_sources: the sentence is about the post's inputs (its evidence,
    # the material it was given), not the subject; it says nothing to a
    # reader and is removed like a contradiction.
    verdict: Literal["supported", "unsupported", "wrong", "about_sources"]
    reason: str = ""
    # The item's atomic claims the evidence does not support, each on its
    # own (an implied cause, scope or frequency counts as a claim).
    unsupported_parts: list[str] = Field(default_factory=list, max_length=8)


class ClaimCheck(BaseModel):
    verdicts: list[SentenceVerdict] = Field(
        default_factory=list,
        description=(
            "Exactly one verdict for every numbered item, supported items "
            "included: an item without a verdict is unchecked."
        ),
    )


class EvidenceVerdict(BaseModel):
    id: str
    verdict: Literal["consistent", "wrong"]
    reason: str = ""


class EvidenceCheck(BaseModel):
    verdicts: list[EvidenceVerdict] = Field(default_factory=list)


class ReviewIssue(BaseModel):
    sentence: str
    category: Literal["depth", "flow", "kindness", "style", "code", "sources"]
    fix: str


class SectionReview(BaseModel):
    issues: list[ReviewIssue] = Field(default_factory=list, max_length=10)


class Edit(BaseModel):
    # A passage copied verbatim from the section (a sentence, a table row, a
    # whole code block), long enough to occur once.
    find: str
    # What replaces it; empty deletes it.
    replace: str = ""


class SectionEdits(BaseModel):
    """A fix as minimal edits, applied by code: every passage not named
    stays as it was (no rewrite drifts the rest of the section)."""

    edits: list[Edit] = Field(default_factory=list, max_length=30)
    notes: str = ""


class FixedSection(BaseModel):
    """A fix's answer: the whole revised section (Markdown, nothing else)
    and, apart, the writer's notes on what it changed or could not fix."""

    section: str
    notes: str = ""


class PostFinding(BaseModel):
    """A problem the whole-post review found, by the section it is in."""

    section: str
    problem: str
    fix: str


class PostReview(BaseModel):
    # What the reviewer traced before reporting (a guide: one request of the
    # final running example through every guard and policy).
    trace: str = ""
    issues: list[PostFinding] = Field(default_factory=list, max_length=30)


@dataclass(frozen=True)
class SectionChecks:
    """What every change of one section is checked with: its verifier set,
    the claim check's evidence (the section's own, and what earlier sections
    cite) and source passages, whether it is the post's own assessment, and
    the numbers it may state freely."""

    verifiers: VerifierSet
    own: tuple[Evidence, ...]
    earlier: tuple[Evidence, ...]
    reference: str
    judgement: bool
    free: str
    # Whether the section's subsections are the writer's (a deletion that
    # leaves one thin merges it), not a fixed shape (the TL;DR's questions).
    subsections: bool = True


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


VerdictKey = tuple[str, str, tuple[str, ...], str, str]


class VerdictCache:
    """Claim-check verdicts by what decides them (docs/engine.md, 4.2): the
    sentence, the sentence before it, the ids it cites, and fingerprints of
    the section's evidence context and of the source passages. A verdict is
    "" for a supported sentence."""

    def __init__(self) -> None:
        self._verdicts: dict[VerdictKey, str] = {}

    @staticmethod
    def key(
        sentence: str,
        previous: str,
        cited: Sequence[str],
        evidence: str,
        reference: str,
    ) -> VerdictKey:
        """The cache key of one claim unit.

        Args:
            sentence: The claim unit.
            previous: The unit before it ("" for the first).
            cited: The evidence ids it cites.
            evidence: The section's evidence context, as the check shows it.
            reference: The source passages the check reads.

        Returns:
            The key.
        """
        return (
            sentence,
            previous,
            tuple(cited),
            _fingerprint(evidence),
            _fingerprint(reference),
        )

    def __contains__(self, key: object) -> bool:
        return key in self._verdicts

    def __len__(self) -> int:
        return len(self._verdicts)

    def get(self, key: VerdictKey) -> str:
        """The verdict stored under ``key`` ("" when supported or unknown)."""
        return self._verdicts.get(key, "")

    def put(self, key: VerdictKey, verdict: str) -> None:
        self._verdicts[key] = verdict

    def entries(self, start: int = 0) -> list[tuple[VerdictKey, str]]:
        """The verdicts in the order they were stored, from the
        ``start``-th on (a key is stored once: the check sends only units
        without a verdict)."""
        return list(self._verdicts.items())[start:]


@dataclass(frozen=True)
class WriterPrompts:
    """The system prompts of one artifact's stages."""

    reader: str
    writer: str
    claim_check: str
    evidence_check: str
    review: str
    fix: str


# Extra admission rules of one artifact: (document, kind, fields) -> problems.
RecordRule = Callable[[SourceDocument, str, Mapping[str, str]], list[str]]


def recheck(document: SourceDocument, item: Evidence) -> list[str]:
    """Today's problems of a stored record (cached under older rules)."""
    return evidence_problems(
        document,
        kind=item.kind,
        claim=item.claim,
        quote=item.quote,
        anchor=item.anchor,
        scope=item.scope,
        cell=item.cell,
        figure_id=item.figure_id,
        latex=item.latex,
        code=item.code,
    )


def dump_read(items: Sequence[Evidence], notes: Sequence[str]) -> str:
    """A read's cache entry: its evidence and the notes the read left (a
    reader stopped at its limit), replayed when the entry is used."""
    return json.dumps(
        {"evidence": [e.model_dump() for e in items], "notes": list(notes)},
        ensure_ascii=False,
    )


def _read_entry(raw: str) -> tuple[object, list[str]]:
    """(the evidence records, the notes) of a cache entry: an object as
    :func:`dump_read` writes it, or a bare list of records (entries
    written before the notes were kept: no notes)."""
    data = json.loads(raw)
    if isinstance(data, dict):
        notes = data.get("notes", [])
        if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
            raise ValueError("notes are not a list of strings")
        return data.get("evidence"), notes
    return data, []


def cached_notes(raw: str) -> list[str]:
    """The notes a cached read left ([] when it left none or the entry is
    unreadable: :func:`cached_evidence` reads such an entry again)."""
    try:
        return _read_entry(raw)[1]
    except ValueError:
        return []


def cached_evidence(
    raw: str,
    document: SourceDocument,
    rejected: Callable[[Evidence], object] = lambda _e: False,
    *,
    key: str = "",
) -> list[Evidence] | None:
    """A document's stored evidence (:func:`dump_read`, or a bare list of
    records) still usable today, bound to the document's current source id
    (ids are numbered per run), or None when it must be read again:
    unreadable (no non-empty list of records: only a read that recorded
    something is stored), or nothing passes today's checks."""
    try:
        data, _ = _read_entry(raw)
        if not isinstance(data, list) or not data:
            raise ValueError("not a list of evidence records")
        items = [Evidence.model_validate(e) for e in data]
    except (ValueError, ValidationError):
        logger.warning("Unreadable cached evidence %s; reading again.", key)
        return None
    items = [e.model_copy(update={"paper_id": document.paper_id}) for e in items]
    kept = [e for e in items if not recheck(document, e) and not rejected(e)]
    if items and not kept:
        logger.warning("No cached evidence of %s passes today; reading again.", key)
        return None
    return kept


async def all_or_none[T](aws: Iterable[Awaitable[T]]) -> list[T]:
    """Every awaitable's result, in order; on the first failure the others
    are cancelled (no new model calls start) and the failure is raised as
    is."""
    tasks = [asyncio.ensure_future(a) for a in aws]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


@dataclass(frozen=True)
class ItemAnswers[K, V]:
    """What a structured per-item check answered: the answer of each item
    it named, the items it never answered, and whether every call gave a
    valid answer."""

    answers: dict[K, V]
    missing: list[K]
    ran: bool


async def complete_answers[K, V](
    items: Sequence[K],
    ask: Callable[[list[K], int], Awaitable[Mapping[K, V] | None]],
    *,
    asks: int,
) -> ItemAnswers[K, V]:
    """The answers of a structured check that judges each item it is sent
    (docs/engine.md, 4.10): the answered items are compared with the sent
    ones, the items left out are sent again on their own, up to ``asks``
    calls in all. An item never answered is unchecked, never passed: the
    caller blocks the post on ``missing``.

    Args:
        items: The items to judge, in order.
        ask: One call for the pending items (and the attempt, from 0): the
            answer per item it named (items it was not sent are ignored),
            or None when it gave no valid answer.
        asks: The calls at most.

    Returns:
        The answers, the items still unanswered, and whether every call ran.
    """
    answers: dict[K, V] = {}
    pending = list(dict.fromkeys(items))
    for attempt in range(asks):
        if not pending:
            break
        got = await ask(pending, attempt)
        if got is None:
            return ItemAnswers(answers, pending, False)
        answers.update((k, v) for k, v in got.items() if k in pending)
        pending = [k for k in pending if k not in got]
    return ItemAnswers(answers, pending, True)


# Appended to an artifact's fix prompt when the fix is asked as edits.
EDIT_ANSWER = """
Answer with `edits`, not the whole section: each edit replaces one passage
of the section with `replace` (empty to delete it). `find` is copied
verbatim from the section and long enough to occur once: a whole sentence,
a table row, or a whole code block. Change only what the problems need;
every passage you do not name stays exactly as it is. `notes`: only the
listed problems you could not fix, each with why (empty when you fixed
them all).
"""


def _removable(verdict: str) -> bool:
    """A sentence code removes when the writer could not fix it: one that
    contradicts its evidence or talks about the post's sources."""
    return verdict.startswith(("wrong", "about_sources"))


def with_canonical_evidence[OutlineT](
    outline: OutlineT, duplicates: Mapping[str, str]
) -> OutlineT:
    """``outline`` with every duplicate evidence id in its sections replaced
    by the id that keeps the fact; a section then lists each fact once
    (evidence a kind cites elsewhere, a survey's table, is its hook's)."""
    if not duplicates:
        return outline
    logger.info(
        "Evidence stating one fact merged: %d item(s) (%s).",
        len(duplicates),
        ", ".join(f"{d}->{k}" for d, k in list(duplicates.items())[:20]),
    )
    sections = [
        s.model_copy(update={"evidence": canonical_ids(s.evidence, duplicates)})
        for s in outline.sections  # type: ignore[attr-defined]
    ]
    return outline.model_copy(update={"sections": sections})  # type: ignore[attr-defined,no-any-return]


def _kept(answer: str) -> str:
    """A model answer stripped of surrounding whitespace, a code fence
    opener glued to a prose line moved to its own (every check then reads
    the code as code), its cut flag kept."""
    return ModelText(unglue_fences(answer.strip()), cut=was_cut(answer))


def _incomplete(text: str) -> bool:
    """Whether an answer is unfinished: stopped at the output limit, or its
    last paragraph stops mid-sentence."""
    return was_cut(text) or ends_mid_sentence(text)


_UNSAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")


def safe_key(text: str) -> str:
    """``text`` as one segment of a cache key."""
    return _UNSAFE_KEY.sub("_", text)


_EVIDENCE_BLOCK = re.compile(r"<evidence>\n.*?\n</evidence>", re.S)


def block(tag: str, text: str) -> str:
    """``text`` as a prompt's ``<tag>`` data block; source text in it cannot
    close the fence (:func:`fenced`)."""
    return fenced(tag, text)


def claim_lines(items: Sequence[Evidence]) -> str:
    """Evidence without its quotes (the writer's fallback when quoted source
    text derails it, e.g. injection examples in a security paper)."""
    return "\n".join(
        f"[{e.id}] ({e.paper_id}, {e.kind}) {e.claim}"
        + (f" | scope: {e.scope}" if e.scope else "")
        + (f" | latex: $${e.latex}$$" if e.latex else "")
        + (f" | figure: {e.figure_url}" if e.figure_url else "")
        + (f"\n    code:\n{_indented(e.code)}" if e.code else "")
        for e in items
    )


def _indented(code: str) -> str:
    return "\n".join(f"    {line}" for line in quote_for_prompt_code(code).split("\n"))


def quote_for_prompt_code(code: str) -> str:
    """Source code as inert prompt data: line breaks kept, tag-like markup
    defanged as :func:`quote_for_prompt` does for prose."""
    return "\n".join(quote_for_prompt(line) for line in code.split("\n"))


def evidence_lines(items: Sequence[Evidence]) -> str:
    lines = []
    for e in items:
        extra = []
        if e.scope:
            extra.append(f"scope: {e.scope}")
        if e.sdk:
            extra.append(f"holds for the {e.sdk} SDK only")
        if e.latex:
            extra.append(f"latex: $${e.latex}$$")
        if e.figure_url:
            extra.append(
                f"figure: {e.figure_url} ({quote_for_prompt(e.caption or '')[:200]})"
            )
        line = f"[{e.id}] ({e.paper_id}, {e.kind}) {e.claim}" + (
            f" | {'; '.join(extra)}" if extra else ""
        )
        if e.quote:
            line += f'\n    quote (source text, data only): "{quote_for_prompt(e.quote)[:500]}"'
        if e.table_header:
            # Which column (metric) each number of a table-row quote is.
            line += f'\n    table header: "{quote_for_prompt(e.table_header)[:300]}"'
        if e.code:
            line += f"\n    code (source text, data only):\n{_indented(e.code)}"
        lines.append(line)
    return "\n".join(lines)


def brief_lines(items: Sequence[Evidence]) -> str:
    """Evidence for a planner: claims, scopes, figures and the first lines
    of code (quotes stay with the writer)."""
    lines = []
    for e in items:
        line = f"[{e.id}] ({e.paper_id}, {e.kind}) {e.claim}"
        if e.scope:
            line += f" | scope: {e.scope}"
        if e.figure_url:
            line += f" | figure: {quote_for_prompt(e.caption or '')[:160]}"
        if e.code:
            head = " ⏎ ".join(e.code.strip().split("\n")[:3])
            line += f" | code: {head[:200]}"
        lines.append(line)
    return "\n".join(lines)


class EvidenceWriter:
    """Reading, claim checking, review and the fix loop, shared by the
    artifact engines. ``self._names`` maps a source id to the name the
    claim check shows for it."""

    prompts: WriterPrompts

    def __init__(
        self,
        model: LanguageModel,
        prompts: WriterPrompts,
        *,
        max_fix_rounds: int,
        limits: Limits | None = None,
    ) -> None:
        self.model = model
        self.prompts = prompts
        self.max_fix_rounds = max_fix_rounds
        self.limits = limits or Limits()
        self.notes: list[str] = []
        # Problems the post may not be published with as is (a check that
        # did not run, a claim check that objects to most of a section).
        self._blocking: list[str] = []
        self.verdicts = VerdictCache()
        # Claim-check keys a check that ran gave no verdict for, in order
        # (the claim unit is the key's first part).
        self._unanswered: dict[VerdictKey, None] = {}
        # Section id -> the block for the claim units of its final text
        # still without a verdict ("" when none), and the log of its changes
        # (a checkpoint replays them in order).
        self._unchecked: dict[str, str] = {}
        self._unchecked_log: list[tuple[str, str]] = []
        # Section title -> the checks every change of the section runs.
        # By section object (two sections may share a title).
        self.checked: dict[int, SectionChecks] = {}
        self._names: dict[str, str] = {}
        # Source passages the current section's evidence comes from (guides):
        # the claim check and the fix read them besides the evidence.
        self.reference = ""
        # (source id, anchor) -> section text: numbers a sentence states are
        # checked against the whole sections its evidence is quoted from
        # (guides and papers; a survey checks against the evidence alone).
        self.passages: dict[tuple[str, str], str] = {}

    # ---- shared state ----------------------------------------------------------

    @property
    def blocking(self) -> list[str]:
        """The problems the post may not be published with as is: the
        blocked ones in order, then each section's unchecked claim units."""
        return [*self._blocking, *(p for p in self._unchecked.values() if p)]

    @blocking.setter
    def blocking(self, problems: Sequence[str]) -> None:
        self._blocking = list(problems)
        self._unchecked.clear()

    def state_mark(self) -> tuple[int, int, int, int, int]:
        """Where the shared state stands (notes, blocking, verdicts, the
        unanswered claim units and the sections' unchecked blocks)."""
        return (
            len(self.notes),
            len(self._blocking),
            len(self.verdicts),
            len(self._unanswered),
            len(self._unchecked_log),
        )

    def state_since(self, mark: tuple[int, int, int, int, int]) -> WriterState:
        """What was added to the shared state since ``mark``, in order."""
        notes, blocking, verdicts, unanswered, unchecked = mark
        return WriterState(
            notes=self.notes[notes:],
            blocking=self._blocking[blocking:],
            verdicts=[
                (s, p, list(c), e, r, v)
                for (s, p, c, e, r), v in self.verdicts.entries(verdicts)
            ],
            unanswered=[
                (s, p, list(c), e, r)
                for s, p, c, e, r in list(self._unanswered)[unanswered:]
            ],
            unchecked=self._unchecked_log[unchecked:],
        )

    def replay_state(self, state: WriterState) -> None:
        """Add a saved stage's state changes as the stage made them: its
        notes and blocking problems after the ones already there, its
        verdicts to the cache (later checks of the same sentences in the
        same context then send nothing, as in an uninterrupted run), its
        unanswered claim units and its sections' unchecked blocks."""
        self.notes.extend(state.notes)
        for problem in state.blocking:
            self._block(problem)
        for s, p, c, e, r, v in state.verdicts:
            self.verdicts.put((s, p, tuple(c), e, r), v)
        for s, p, c, e, r in state.unanswered:
            self._unanswered[(s, p, tuple(c), e, r)] = None
        for section, problem in state.unchecked:
            self._set_unchecked(section, problem)

    # ---- read -------------------------------------------------------------------

    async def read_document(
        self,
        document: SourceDocument,
        *,
        prompt: str,
        limits: AgentLimits,
        required: Sequence[str] = (),
        rule: RecordRule | None = None,
        minimum: int = 0,
        notes: list[str] | None = None,
    ) -> list[Evidence]:
        """One agent reads ``document`` and records evidence; every record
        is checked against the document (and ``rule``) before it is kept.
        ``finish`` asks again (``limits.required_kind_asks`` times) for any
        ``required`` kind still missing. A reader that ends with a required
        kind missing or fewer than ``minimum`` records (the low end of the
        count it was asked for) is continued once, in a follow-up of
        ``limits.follow_up_turns`` turns. The read's notes go to ``notes``
        too, when given (a cache keeps them with the evidence)."""
        ledger = EvidenceLedger()
        rejected = 0
        refusals = 0
        # Rejection reasons by their first clause, for tuning the reader.
        reasons: Counter[str] = Counter()

        def read_section(anchor: str) -> str:
            """Return the full text of one section of the source by its
            anchor (S1, S2, ... from the outline)."""
            section = document.section(anchor)
            if section is None:
                return f"Unknown anchor {anchor!r}. Anchors: " + ", ".join(
                    s.anchor for s in document.sections
                )
            return f"{section.anchor} | {section.heading}\n\n{section.text[: self.limits.read_section_chars]}"

        def search_source(query: str) -> str:
            """Find passages of the source containing every word of the
            query; returns anchor and snippet per match."""
            hits = document.search(query)
            return "\n\n".join(f"[{a}] {s}" for a, s in hits) or "No passage matches."

        recording = threading.Lock()

        def record_evidence(
            kind: str,
            claim: str,
            quote: str,
            anchor: str,
            scope: str = "",
            cell: str = "",
            figure_id: str = "",
            latex: str = "",
            code: str = "",
            sdk: str = "",
        ) -> str:
            """Record one piece of evidence. kind: finding, number, method,
            definition, limitation, relation, equation, figure, code or
            setting. claim: what the post may state (the post's language).
            quote: the source's words, verbatim from the section at anchor.
            scope: what a number or setting applies to (model, benchmark,
            setting, metric; API, CLI, config file, version). cell: short noun
            phrase for a table. figure_id: F-id for a figure. latex: the
            equation as in the source. code: a code block or command copied
            exactly from the section at anchor. sdk: the one SDK language the
            fact holds for when the source says it holds for that language
            only (python, typescript, java, go, rust); empty otherwise."""
            nonlocal rejected
            fields = {
                "kind": kind,
                "claim": claim,
                "quote": quote,
                "anchor": anchor,
                "scope": scope,
                "cell": cell,
                "figure_id": figure_id,
                "latex": latex,
                "code": code,
            }
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
                code=code or None,
            )
            if kind not in _EVIDENCE_KINDS:
                problems.insert(0, f"unknown kind {kind!r}")
            if rule is not None:
                problems += rule(document, kind, fields)
            # The agent may run one turn's tool calls in parallel threads: the
            # duplicate check and the id the ledger hands out see one state.
            with recording:
                if (
                    not problems
                    and kind not in ("figure", "code")
                    and near_duplicate(
                        claim, [e.claim for e in ledger if e.kind == kind]
                    )
                ):
                    problems.append(
                        "this repeats evidence already recorded; record a "
                        "different fact or move on"
                    )
                if problems:
                    rejected += 1
                    reasons.update(f"{kind}: {p.split(':')[0][:60]}" for p in problems)
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
                    code=code.strip("\n") or None,
                    sdk=sdk.strip().lower(),
                )
            return f"recorded {item.id}"

        def finish(summary: str) -> str:
            """Call when the evidence covers the source as the post needs."""
            nonlocal refusals
            kinds = {e.kind for e in ledger}
            missing = [k for k in required if k not in kinds]
            if missing and refusals < self.limits.required_kind_asks:
                refusals += 1
                return (
                    "Not yet: this source needs at least one piece of evidence "
                    f"of each kind {missing}. Record them, or call finish again "
                    "if the source has none."
                )
            return "done"

        outcome = await self.model.run_agent(
            role="read",
            system=self.prompts.reader,
            prompt=prompt,
            tools=[read_section, search_source, record_evidence, finish],
            limits=limits,
        )
        # The agent may end its turn without calling finish, or stop at the
        # first few records: ask once, in a short follow-up, for the
        # required kinds it has not recorded and the rest of the evidence
        # its sections hold.
        missing = [k for k in required if k not in {e.kind for e in ledger}]
        short = len(ledger) < minimum
        follow_up = None
        if missing or short:
            recorded = "\n".join(f"- ({e.kind}) {e.claim[:120]}" for e in ledger)
            asks: list[str] = []
            if missing:
                asks.append(
                    f"Evidence of kind {missing} is still missing. Record "
                    "it now if the source has it (for a figure: the one "
                    "that best shows the architecture, a flow or a result, "
                    "described from the text before it)."
                )
            if short:
                asks.append(
                    f"You recorded {len(ledger)} pieces of evidence, fewer "
                    f"than the {minimum} or more asked for. Read the rest of "
                    "your sections and record the evidence they hold that "
                    "already_recorded does not: each a different fact of the "
                    "right kind. If the source holds no more, say so in "
                    "finish."
                )
            follow_up = await self.model.run_agent(
                role="read",
                system=self.prompts.reader,
                prompt="\n\n".join(
                    [
                        prompt,
                        block("already_recorded", recorded),
                        " ".join([*asks, "Then call finish."]),
                    ]
                ),
                tools=[read_section, search_source, record_evidence, finish],
                limits=AgentLimits(
                    turns=self.limits.follow_up_turns,
                    total_tokens=limits.total_tokens,
                ),
            )
        logger.info(
            "Read %s: %d evidence, %d rejected records (%s%s).",
            document.paper_id,
            len(ledger),
            rejected,
            outcome.stop_reason,
            f"; follow-up {follow_up.stop_reason}" if follow_up is not None else "",
        )
        if reasons:
            logger.info(
                "Rejections of %s: %s",
                document.paper_id,
                "; ".join(f"{r} x{n}" for r, n in reasons.most_common(6)),
            )
        stops = [("Reading", outcome.stop_reason)]
        if follow_up is not None:
            stops.append(("The follow-up read of", follow_up.stop_reason))
        for what, reason in stops:
            if reason.startswith("limit"):
                note = f"{what} {document.paper_id} stopped at its limit ({reason})."
                self.notes.append(note)
                if notes is not None:
                    notes.append(note)
        return list(ledger)

    async def wrong_evidence(self, items: Sequence[Evidence]) -> dict[str, str]:
        """Evidence whose claim, scope or cell misstates its quote (a scope
        can name the wrong benchmark while every number is in the quote):
        id -> reason. A figure's claim is checked against its caption and
        the text before it; equations and code are checked in code. A
        check without a valid answer blocks the post and drops nothing
        (docs/engine.md, 4.10)."""
        items = [e for e in items if e.kind not in ("equation", "code")]
        items = [e for e in items if e.kind != "figure" or e.caption or e.quote]
        if not items:
            return {}
        by_id = {e.id: e for e in items}

        async def ask(
            ids: list[str], attempt: int
        ) -> dict[str, EvidenceVerdict] | None:
            lines = "\n".join(
                f"[{e.id}] ({e.kind}) claim: {e.claim} | cell: {e.cell} | scope: "
                f'{e.scope}\n    quote: "{quote_for_prompt(e.quote)}"'
                + (
                    f'\n    caption: "{quote_for_prompt(e.caption)}"'
                    if e.caption
                    else ""
                )
                for e in (by_id[i] for i in ids)
            )
            what = "the evidence check"
            with usage_stage("verify"):
                check = await self._evaluated(
                    what if not attempt else f"{what} (items it left out)",
                    blocking=True,
                    unchecked=(
                        "no evidence was left out, and the post uses it unchecked"
                    ),
                    role="evaluate",
                    system=self.prompts.evidence_check,
                    prompt=block("evidence", lines),
                    schema=EvidenceCheck,
                )
            if check is None:
                return None
            return {v.id: v for v in check.verdicts}

        result = await complete_answers(list(by_id), ask, asks=self.limits.answer_asks)
        if result.ran and result.missing:
            self._block(
                f"The evidence check gave no verdict for {len(result.missing)} of "
                f"its items, even when asked again: they are unchecked, and the "
                "post uses them (" + ", ".join(result.missing) + ")."
            )
        wrong = {i: v.reason for i, v in result.answers.items() if v.verdict == "wrong"}
        if wrong:
            self.notes.append(
                "Evidence that misstates its quote was left out: "
                + "; ".join(f"{i} ({r[:120]})" for i, r in wrong.items())
            )
        return wrong

    # ---- sections ---------------------------------------------------------------

    async def draft(
        self,
        prompt: str,
        own: Sequence[Evidence],
        target: int,
        *,
        retry_hint: str,
        evidence: Sequence[Evidence] | None = None,
    ) -> str:
        """A section draft; a short or cut-off answer is asked again, then
        written from the claims alone (quoted text can derail the model):
        the prompt's ``<evidence>`` block, holding ``evidence`` (``own``
        unless given), is swapped for their claims. The prompt changes
        every section, so no message is cached."""
        with usage_stage("draft"):
            text = _kept(
                await self.model.text(
                    role="write",
                    system=self.prompts.writer,
                    prompt=prompt,
                )
            )
            if (own or evidence) and (
                prose_length(text) < self.limits.short_draft_share * target
                or _incomplete(text)
            ):
                logger.warning(
                    "Section came back short (%d chars): %r",
                    prose_length(text),
                    text[:300],
                )
                text = _kept(
                    await self.model.text(
                        role="write",
                        system=self.prompts.writer,
                        prompt=prompt
                        + "\n\nYour previous answer was too short or empty. "
                        + retry_hint
                        + f" Write the whole section (about {int(target):,} "
                        "characters of prose) from its evidence.",
                    )
                )
            if (own or evidence) and (
                prose_length(text) < self.limits.short_draft_share * target
                or _incomplete(text)
            ):
                logger.warning("Section still short; writing it from the claims alone.")
                text = _kept(
                    await self.model.text(
                        role="write",
                        system=self.prompts.writer,
                        prompt=_EVIDENCE_BLOCK.sub(
                            lambda _m: block("evidence", claim_lines(evidence or own)),
                            prompt,
                            count=1,
                        ),
                    )
                )
        return text

    async def verify_and_fix(
        self,
        draft: str,
        *,
        title: str,
        review_context: str,
        verifiers: VerifierSet,
        ledger: EvidenceLedger,
        own: Sequence[Evidence],
        earlier: Sequence[Evidence],
        free: str,
        judgement: bool = False,
        rest: str = "",
        section: object | None = None,
        subsections: bool = True,
    ) -> str:
        """The section's verifier set, the claim check and (first round) the
        review, fixed up to ``max_fix_rounds`` (each fix an edit
        transaction); then unsupported sentences narrowed once, checked
        again and deleted (:meth:`_settle`); missing figures appended and a
        lone ``$$`` removed. The section's checks are registered under
        ``title``, so every later change of it runs the same ones.

        Args:
            draft: The section draft.
            title: The section's title.
            review_context: What the reviewer is told about the section.
            verifiers: The section's verifier set.
            ledger: The run's evidence.
            own: The section's evidence.
            earlier: The evidence earlier sections cite.
            free: Numbers a sentence may state without evidence.
            judgement: Whether the section is the post's own assessment
                (its unsupported sentences are shown, not deleted).
            rest: The rest of the post (a copy of a block there lets a fix
                drop it here).
            section: The outline section (its checks are registered under it).
            subsections: Whether the section's subsections are the writer's
                (False for a fixed shape such as the TL;DR's questions).

        Returns:
            The section.
        """
        checks = SectionChecks(
            verifiers,
            tuple(own),
            tuple(earlier),
            self.reference,
            judgement,
            free,
            subsections,
        )
        if section is not None:
            self.checked[id(section)] = checks
        # The review runs once; its problems stand until a fix made for them
        # is taken (the claims and code are checked again below).
        unaddressed: list[str] = []
        for round_ in range(self.max_fix_rounds + 1):
            code = verifiers.run(draft)
            claims = await self.check_claims(draft, ledger, own, earlier)
            review = await self.review(draft, review_context) if round_ == 0 else []
            unaddressed += review
            issues = messages(code) + [f"{v}: {s}" for s, v in claims] + review
            if not issues or round_ == self.max_fix_rounds:
                break
            revision = await self._fix(
                draft,
                self.fix_evidence(draft, own, earlier),
                issues,
                verifiers=verifiers,
                rest=rest,
            )
            if revision.rejected:
                logger.info("Fix of %r rejected: %s", title, revision.rejected)
                break
            if not revision.taken or revision.text.strip() == draft.strip():
                logger.info("Fix of %r changed nothing; stopping.", title)
                break
            draft = revision.text
            unaddressed = []
        if unaddressed:
            self.notes.append(
                f"Section '{title}': review problems no fix addressed: "
                + " | ".join(i[:200] for i in unaddressed[:5])
            )
        draft = await self._settle(
            draft,
            title=title,
            ledger=ledger,
            checks=checks,
            evidence=self.fix_evidence(draft, own, earlier),
            rest=rest,
        )
        draft = add_missing_figures(draft, [e for e in own if e.kind == "figure"])
        draft = drop_empty_headings(draft)
        draft, unbalanced = balance_math(draft)
        if unbalanced:
            self.notes.append(f"Section '{title}': a lone $$ was removed.")
        self._note_leftovers(draft, title=title, verifiers=verifiers)
        self._block_unchecked(draft, title=title, checks=checks)
        return draft

    def fix_evidence(
        self, draft: str, own: Sequence[Evidence], earlier: Sequence[Evidence]
    ) -> list[Evidence]:
        """The evidence a fix of ``draft`` sees: the section's own, then the
        earlier evidence the claim check judges against, up to
        ``limits.fix_earlier_evidence`` items: what the draft cites first,
        then the latest (in their order)."""
        cap = self.limits.fix_earlier_evidence
        cited = set(cited_ids(draft))
        chosen = {i for i, e in enumerate(earlier) if e.id in cited}
        for i in range(len(earlier) - 1, -1, -1):
            if len(chosen) >= cap:
                break
            chosen.add(i)
        picked = sorted(chosen)
        if len(picked) > cap:
            # More cited than the cap: the latest of them.
            picked = picked[-cap:] if cap > 0 else []
        return [*own, *(earlier[i] for i in picked)]

    def _note_leftovers(
        self, draft: str, *, title: str, verifiers: VerifierSet
    ) -> None:
        """A PR note of the problems the section's verifier set still finds."""
        if leftover := verifiers.run(draft):
            self.notes.append(
                f"Section '{title}' still has: " + " | ".join(messages(leftover)[:5])
            )

    async def _settle(
        self,
        draft: str,
        *,
        title: str,
        ledger: EvidenceLedger,
        checks: SectionChecks,
        evidence: Sequence[Evidence],
        rest: str,
        later: str = "",
        baseline: str | None = None,
    ) -> str:
        """``draft`` with its unsupported sentences settled (docs/engine.md,
        4.5): the writer narrows them once in context (an edit
        transaction), the changed sentences are checked again, and what is
        still unsupported is deleted (:meth:`_drop_unsupported`).
        ``baseline``: only the sentences the draft added to it, and the one
        after each change, are settled (a whole-post fix)."""

        async def found(text: str) -> list[tuple[str, str]]:
            claims = await self.unsupported(
                text,
                ledger,
                checks.free,
                checks.own,
                checks.earlier,
                reference=checks.reference,
            )
            if baseline is None:
                return claims
            # What the fix changed, and the unit after each change: its
            # verdict is judged again (a new context sentence), so a new
            # objection to it counts too.
            fresh = set(rejudged_units(baseline, text))
            return [(s, v) for s, v in claims if s in fresh]

        start = draft
        claims = await found(draft)
        if claims:
            revision = await self._fix(
                draft,
                evidence,
                [
                    "These sentences are not supported by their evidence; narrow "
                    "each to what the evidence says or delete it, and mend the "
                    "sentences around a deletion:\n"
                    + "\n".join(f"- {v}: {s}" for s, v in claims)
                ],
                verifiers=checks.verifiers,
                rest=rest,
                later=later,
                # Narrowing removes unsupported content: a section it leaves
                # short is noted, not kept unsupported.
                tolerated=(TOO_SHORT_PROBLEM,),
            )
            if revision.taken and revision.text.strip() != draft.strip():
                draft = revision.text
                # The claim check reads only what the narrowing changed (the
                # verdicts of the rest are cached).
                claims = await found(draft)
            elif revision.rejected:
                logger.info("Narrowing of %r rejected: %s", title, revision.rejected)
        settled = self._drop_unsupported(draft, claims, title=title, checks=checks)
        if settled != draft:
            # A deletion gives the unit after it a new context sentence, so a
            # new verdict key: it is judged again (once; docs/engine.md 4.2),
            # and one still unanswered then blocks under its new key.
            fresh = set(rejudged_units(draft, settled))
            again = [(s, v) for s, v in await found(settled) if s in fresh]
            if again:
                settled = self._drop_unsupported(
                    settled, again, title=title, checks=checks
                )
        if settled != start and checks.verifiers.added(
            [p for p in checks.verifiers.run(start) if p.kind == TOO_SHORT_PROBLEM],
            [p for p in checks.verifiers.run(settled) if p.kind == TOO_SHORT_PROBLEM],
        ):
            self.notes.append(
                f"Section '{title}': removing its unsupported sentences left it "
                "below its minimum length; develop it from its evidence by hand."
            )
        self._block_unchecked(settled, title=title, checks=checks)
        return settled

    def _drop_unsupported(
        self,
        draft: str,
        claims: Sequence[tuple[str, str]],
        *,
        title: str,
        checks: SectionChecks,
    ) -> str:
        """``draft`` without the sentences the claim check still objects to,
        their neighbours mended: contradicting ones and ones about the
        post's sources always, unsupported ones unless they are table rows,
        the writer's marked illustrations or the post's own assessment (a
        judgement section), which the PR checklist lists instead. Code is
        never a claim unit. When the deletions would exceed
        ``limits.max_drop_share`` of the section's claim units, the check
        misread the section: nothing is deleted and the post is blocked."""
        doubtful = [(s, v) for s, v in claims if not _removable(v)]
        shown = [
            (s, v)
            for s, v in doubtful
            if checks.judgement or s.startswith("|") or is_illustration(s)
        ]
        deleted = [(s, v) for s, v in claims if (s, v) not in shown]
        units = claim_units(draft)
        if deleted and len(deleted) > self.limits.max_drop_share * len(units):
            self._block(
                f"Section '{title}': the claim check objects to {len(deleted)} of "
                f"{len(units)} sentences (more than "
                f"{self.limits.max_drop_share:.0%}); none was deleted. Check the "
                "section against its evidence: "
                + " | ".join(f"{s[:120]} ({v[:80]})" for s, v in deleted[:5])
            )
            return draft
        if deleted:
            before = draft
            draft = drop_sentences(draft, [s for s, _ in deleted])
            if checks.subsections:
                # A subsection the deletions leave with one sentence joins
                # the text above it instead of standing as a stub.
                draft = merge_thin_subsections(before, draft)
            left = set(claim_units(draft))
            if removed := [(s, v) for s, v in deleted if s not in left]:
                self.notes.append(
                    f"Section '{title}': {len(removed)} sentence(s) removed that "
                    "contradict their evidence, talk about the post's sources or "
                    "that their evidence does not support: "
                    + " | ".join(f"{s[:120]} ({v[:80]})" for s, v in removed[:5])
                )
            if kept := [(s, v) for s, v in deleted if s in left]:
                self.notes.append(
                    f"Section '{title}': fix by hand, {len(kept)} sentence(s) "
                    "their evidence does not support could not be removed: "
                    + " | ".join(f"{s[:120]} ({v[:80]})" for s, v in kept[:5])
                )
        if shown:
            self.notes.append(
                f"Section '{title}': check {len(shown)} sentence(s) "
                "the evidence may not support: "
                + " | ".join(f"{s[:120]} ({v})" for s, v in shown[:5])
            )
        return draft

    def _block(self, problem: str) -> None:
        """A problem the post may not be published with as is (once)."""
        if problem not in self._blocking:
            logger.warning("Blocking: %s", problem)
            self._blocking.append(problem)

    def _set_unchecked(self, section: str, problem: str) -> None:
        """The section's unchecked block, replacing its earlier one."""
        if self._unchecked.get(section, "") == problem:
            return
        if problem:
            logger.warning("Blocking: %s", problem)
        self._unchecked[section] = problem
        self._unchecked_log.append((section, problem))

    def _block_unchecked(
        self, draft: str, *, title: str, checks: SectionChecks
    ) -> None:
        """Block the post on the claim units of ``draft``, the section's
        final text, that a claim check asked without getting a verdict and
        that have none yet; this replaces the section's earlier block, so a
        unit judged later, rewritten or deleted no longer blocks."""
        _, keys = self._claim_keys(draft, checks.own, checks.earlier, checks.reference)
        left = list(
            dict.fromkeys(
                key[0]
                for key in keys
                if key in self._unanswered and key not in self.verdicts
            )
        )
        # The section by its title and what its checks read (two sections
        # may share a title; an object id does not survive a checkpoint).
        context = evidence_lines([*checks.earlier, *checks.own]) + checks.reference
        section = f"{title}\n{_fingerprint(context)}"
        self._set_unchecked(
            section,
            (
                (
                    f"The claim check of '{title}' gave no verdict for {len(left)} "
                    "of its items, even when asked again: they are unchecked ("
                    + "; ".join(s[:80] for s in left)
                    + ")."
                )
                if left
                else ""
            ),
        )

    async def apply_findings(
        self,
        written: Sequence[tuple[SectionT, str]],
        findings: Sequence[PostFinding],
        ledger: EvidenceLedger,
        *,
        title_of: Callable[[SectionT], str],
        own_of: Callable[[SectionT], list[Evidence]],
        reference_of: Callable[[Sequence[Evidence]], str],
        post_check: (
            Callable[[Sequence[tuple[SectionT, str]]], list[Problem]] | None
        ) = None,
    ) -> list[tuple[SectionT, str]]:
        """Each section a whole-post review names, fixed with its evidence
        and source passages as an edit transaction that runs the section's
        registered verifier set (a fix that adds a problem, or loses
        structure, is not taken); the sentences the fix adds are settled
        like a section's (narrowed, checked, deleted), and what the
        section's verifier set still finds is noted. A code block may go
        only when a copy stays in the section or an earlier one (code is
        run in reading order). ``post_check``: problems of the whole post
        (the program a reader pastes in order), run in each fix's
        transaction with the section's checks, so a fix that adds one to
        any section is not taken."""
        out: list[tuple[SectionT, str]] = []
        # A section is named by its title or heading, or by a subsection
        # title no other section has (a shared one names no single section).
        shared = Counter(n for _, md in written for n in heading_names(md))

        def names_of(section: SectionT, md: str) -> set[str]:
            own = {n for n in heading_names(md) if shared[n] == 1}
            return {title_of(section), section_heading(md), *own}

        names_seen = {n for section, md in written for n in names_of(section, md)}
        if unmatched := [
            f
            for f in findings
            if f.section.strip().lstrip("#").strip() not in names_seen
        ]:
            self.notes.append(
                "Whole-post review problems naming no section of the post (or "
                "a subsection title several sections share), not applied: "
                + " | ".join(f"{f.section}: {f.problem}"[:160] for f in unmatched[:3])
            )
        for i, (section, md) in enumerate(written):
            title = title_of(section)
            names = names_of(section, md)
            issues = [
                f"{f.problem} Fix: {f.fix}"
                for f in findings
                if f.section.strip().lstrip("#").strip() in names
            ]
            if not issues:
                out.append((section, md))
                continue
            checks = self.checked.get(id(section))
            if checks is None:
                # A fix no check would see is not taken.
                self.notes.append(
                    f"Section '{title}': whole-post review problems left as they "
                    "were (the section has no registered checks): "
                    + " | ".join(i[:160] for i in issues[:3])
                )
                out.append((section, md))
                continue
            own = own_of(section)
            self.reference = reference_of(own)
            later = "\n\n".join(m for _, m in written[i + 1 :])
            rest = "\n\n".join([m for _, m in out] + [m for _, m in written[i + 1 :]])
            verifiers = checks.verifiers
            if post_check is not None:
                verifiers = self._with_post_check(
                    verifiers, post_check, out, section, written[i + 1 :]
                )
            fixing = dataclasses.replace(checks, verifiers=verifiers)
            revision = await self._fix(
                md, own, issues, verifiers=verifiers, rest=rest, later=later
            )
            if not revision.taken or revision.text.strip() == md.strip():
                self.notes.append(
                    f"Section '{title}': whole-post review problems left as they "
                    "were"
                    + (f" ({revision.rejected})" if revision.rejected else "")
                    + ": "
                    + " | ".join(i[:160] for i in issues[:3])
                )
            else:
                md = await self._settle(
                    revision.text,
                    title=title,
                    ledger=ledger,
                    checks=fixing,
                    evidence=own,
                    rest=rest,
                    later=later,
                    baseline=md,
                )
                self._note_leftovers(md, title=title, verifiers=checks.verifiers)
            self.reference = ""
            out.append((section, md))
        return out

    @staticmethod
    def _with_post_check(
        verifiers: VerifierSet,
        post_check: Callable[[Sequence[tuple[SectionT, str]]], list[Problem]],
        before: Sequence[tuple[SectionT, str]],
        section: SectionT,
        after: Sequence[tuple[SectionT, str]],
    ) -> VerifierSet:
        """``verifiers`` with the whole post's check of the post with this
        section's text in its place."""

        def whole(markdown: str) -> list[Problem]:
            return post_check([*before, (section, markdown), *after])

        return dataclasses.replace(verifiers, checks=(*verifiers.checks, whole))

    async def _fix(
        self,
        draft: str,
        evidence: Sequence[Evidence],
        issues: Sequence[str],
        *,
        verifiers: VerifierSet,
        rest: str = "",
        later: str = "",
        tolerated: Collection[str] = (),
    ) -> Revision:
        """The section revised for ``issues`` by minimal edits (each an
        exact passage replaced or deleted), applied as one edit transaction;
        when none of the edits applies, the whole revised section instead,
        as one edit. The answers are structured, so no note reaches the
        post. An answer that does not parse leaves the draft as it was."""
        heading = section_heading(draft) or "a section"
        context = "\n\n".join(
            [
                block("evidence", evidence_lines(evidence)),
                *([block("source_sections", self.reference)] if self.reference else []),
            ]
        )
        prompt = "\n\n".join(
            [
                block("section", draft),
                block("problems", "\n".join(f"- {i}" for i in issues)),
            ]
        )

        def transaction(edits: Sequence[tuple[str, str]]) -> Revision:
            return transact(
                draft,
                edits,
                verifiers=verifiers,
                max_prose_cut=self.limits.max_prose_cut,
                rest=rest,
                later=later,
                tolerated=tolerated,
                max_edits=self.limits.max_edits,
            )

        with usage_stage("fix"):
            try:
                edits = await self.model.structured(
                    role="write",
                    system=self.prompts.fix,
                    schema=SectionEdits,
                    # The evidence and passages repeat across a section's
                    # fixes (and its fallback): only what follows changes.
                    context=context,
                    prompt=prompt + "\n\n" + EDIT_ANSWER,
                )
            except ValueError as e:
                logger.warning("Edit answer unusable (%s); asking for the section.", e)
                edits = None
            if edits is not None:
                revision = transaction([(e.find, e.replace) for e in edits.edits])
                self._writer_notes(heading, edits.notes)
                if revision.skipped:
                    self.notes.append(
                        f"Section '{heading}': {len(revision.skipped)} edit(s) not "
                        "applied: " + " | ".join(revision.skipped[:3])
                    )
                if revision.applied:
                    return revision
                logger.info(
                    "No edit of %r applied (%d proposed); asking for the section.",
                    heading,
                    len(edits.edits),
                )
            try:
                answer = await self.model.structured(
                    role="write",
                    system=self.prompts.fix,
                    schema=FixedSection,
                    context=context,
                    prompt=prompt + "\n\n" + FIX_ANSWER,
                )
            except ValueError as e:
                logger.warning("Fix answer unusable (%s); the draft stands.", e)
                self.notes.append(
                    f"Section '{heading}': a fix could not be made ({e}); "
                    "the draft stands."
                )
                return Revision(draft)
        self._writer_notes(heading, answer.notes)
        if not answer.section.strip():
            return Revision(draft)
        return transaction([(draft, answer.section.strip())])

    def _writer_notes(self, heading: str, notes: str) -> None:
        """What the writer could not fix, on the PR checklist, once."""
        if text := " ".join(notes.split()):
            note = f"Section '{heading}': the writer could not fix: {text[:300]}"
            if note not in self.notes:
                self.notes.append(note)

    async def unsupported(
        self,
        draft: str,
        ledger: EvidenceLedger,
        free: str,
        own: Sequence[Evidence] = (),
        earlier: Sequence[Evidence] = (),
        *,
        reference: str | None = None,
    ) -> list[tuple[str, str]]:
        """The claim check's unsupported sentences (cached verdicts: a
        sentence already judged in the same context is not sent again) and
        the sentences whose numbers their evidence does not state."""
        found = await self.check_claims(
            draft, ledger, own, earlier, reference=reference
        )
        known = {s for s, _ in found}
        found += [
            (s, "numbers not in its evidence")
            for s in unsourced_number_sentences(draft, ledger, free, self.passages)
            if s not in known
        ]
        return found

    async def check_claims(
        self,
        draft: str,
        ledger: EvidenceLedger,
        own: Sequence[Evidence] = (),
        earlier: Sequence[Evidence] = (),
        *,
        reference: str | None = None,
    ) -> list[tuple[str, str]]:
        """(sentence, verdict with reason) of the sentences the evidence does
        not support. Only claim units without a cached verdict are sent
        (each with the sentence before it, so a pronoun reads in context);
        the cached verdicts are merged in. When the model gives no valid
        answer, the unsent units stay unchecked and the post is blocked.
        Units an answer still leaves out after ``limits.answer_asks`` calls
        are recorded, not cached: the post is blocked on those left in the
        section's final text (:meth:`_block_unchecked`).

        The shared context is two cached parts (docs/engine.md, 4.10): the
        evidence earlier sections cite first (it only grows, so it is the
        most stable prefix), then this section's evidence and source
        passages.

        Args:
            draft: The section.
            ledger: The run's evidence.
            own: The section's evidence.
            earlier: The evidence earlier sections cite.
            reference: The source passages the check reads (the writer's
                current ones when None).

        Returns:
            The units not supported, in the section's order.
        """
        reference = self.reference if reference is None else reference
        sentences, keys = self._claim_keys(draft, own, earlier, reference)
        if not sentences:
            return []
        earlier_text = evidence_lines(list(earlier))
        own_text = evidence_lines(list(own))
        sent: list[int] = []
        queued = set()
        for n, key in enumerate(keys):
            if key not in self.verdicts and key not in queued:
                queued.add(key)
                sent.append(n)
        heading = section_heading(draft) or "a section"
        context = (
            block("earlier_evidence", earlier_text) if earlier else "",
            block("section_evidence", own_text)
            + ("\n\n" + block("source_sections", reference) if reference else ""),
        )
        # An item the answer leaves out is unchecked, not supported: it is
        # asked again on its own, and never cached.

        async def ask(pending: list[int], attempt: int) -> dict[int, str] | None:
            return await self._claim_verdicts(
                heading, pending, sentences, ledger, context, attempt
            )

        result = await complete_answers(sent, ask, asks=self.limits.answer_asks)
        for n, verdict in result.answers.items():
            self.verdicts.put(keys[n], verdict)
        # Blocked by the section's final text (:meth:`_block_unchecked`),
        # whether the check left them out or did not run at all: a later
        # check may judge them, a fix rewrite or delete them.
        for n in result.missing:
            self._unanswered[keys[n]] = None
        return [
            (sentence, verdict)
            for sentence, key in zip(sentences, keys, strict=True)
            if (verdict := self.verdicts.get(key))
        ]

    def _claim_keys(
        self,
        draft: str,
        own: Sequence[Evidence],
        earlier: Sequence[Evidence],
        reference: str,
    ) -> tuple[list[str], list[VerdictKey]]:
        """The claim units of ``draft`` and their verdict-cache keys in the
        section's context (each unit with the one before it)."""
        # Table rows are claims too (a wrong cell is as wrong as a sentence).
        sentences = claim_units(draft)
        # What the verdicts depend on: both parts of the context.
        evidence_context = (
            f"{evidence_lines(list(earlier))}\n\n{evidence_lines(list(own))}"
        )
        keys = [
            self.verdicts.key(
                sentence,
                sentences[n - 1] if n else "",
                cited_ids(sentence),
                evidence_context,
                reference,
            )
            for n, sentence in enumerate(sentences)
        ]
        return sentences, keys

    async def _claim_verdicts(
        self,
        heading: str,
        sent: list[int],
        sentences: list[str],
        ledger: EvidenceLedger,
        context: tuple[str, str],
        attempt: int,
    ) -> dict[int, str] | None:
        """One claim-check call for the units ``sent``: each answered unit's
        verdict ("" when supported), keyed by its index in ``sentences``;
        None when the check gave no valid answer."""
        lines = []
        for number, n in enumerate(sent, 1):
            sentence = sentences[n]
            cited = [e for i in cited_ids(sentence) if (e := ledger.get(i))]
            lines.append(
                f"{number}. {sentence}\n"
                + (f"   follows: {sentences[n - 1]}\n" if n else "")
                + "   evidence: "
                + (
                    " || ".join(
                        f"[{e.id}, from {self._names.get(e.paper_id, e.paper_id)}] "
                        f'{e.claim} — "{quote_for_prompt(e.quote)[:300]}" {e.scope}'
                        + (f" code: {e.code[:300]}" if e.code else "")
                        for e in cited
                    )
                    or "(none)"
                )
            )
        what = f"the claim check of '{heading}'"
        with usage_stage("verify"):
            check = await self._evaluated(
                what if not attempt else f"{what} (items it left out)",
                # Its unanswered items block by the final text instead.
                quiet=True,
                role="evaluate",
                system=self.prompts.claim_check,
                # Shared by every claim check of the section (each round);
                # the earlier part by the later sections' checks too.
                context=context,
                prompt=block("sentences", "\n".join(lines)),
                schema=ClaimCheck,
            )
        if check is None:
            return None
        answered: dict[int, str] = {}
        for v in check.verdicts:
            if not 1 <= v.n <= len(sent):
                continue
            parts = [p for p in v.unsupported_parts if p.strip()]
            # A part not supported makes the item unsupported, whatever the
            # item's own label says.
            verdict = "unsupported" if v.verdict == "supported" and parts else v.verdict
            if verdict == "supported":
                answered.setdefault(sent[v.n - 1], "")
                continue
            reason = v.reason + (
                " (not supported: " + "; ".join(parts) + ")" if parts else ""
            )
            answered[sent[v.n - 1]] = f"{verdict}: {reason.strip()}"
        return answered

    async def _evaluated(
        self,
        what: str,
        *,
        blocking: bool = False,
        quiet: bool = False,
        unchecked: str = "its sentences are unchecked",
        **call: Any,
    ) -> Any:
        """A structured check's answer, or None when the model gave no valid
        answer even on the fallback model (a content filter, a cut answer):
        the run goes on with the draft as it is. A ``blocking`` check that
        did not run blocks the post (``unchecked`` says what is left
        unchecked); a ``quiet`` one leaves that to its caller (the claim
        check blocks on the items its final text still has unjudged);
        another one leaves a note on the PR checklist."""
        try:
            return await self.model.structured(**call)
        except ValueError as e:
            logger.warning("%s did not run: %s", what, e)
            if quiet:
                return None
            if blocking:
                self._block(
                    f"{what[0].upper()}{what[1:]} could not run ({e}): {unchecked}."
                )
            else:
                self.notes.append(f"Check by hand: {what} could not run ({e}).")
            return None

    async def review(self, draft: str, context: str) -> list[str]:
        """The section review's located reader problems. A review that gives
        no valid answer blocks the post, as a failed claim check does: the
        section would go out unreviewed."""
        with usage_stage("review"):
            review = await self._evaluated(
                f"the review of '{section_heading(draft) or 'a section'}'",
                blocking=True,
                unchecked="the section is unreviewed",
                role="evaluate",
                system=self.prompts.review,
                prompt="\n\n".join([context, block("section", draft)]),
                schema=SectionReview,
            )
        if review is None:
            return []
        return [
            f"{i.category}: {i.fix} — in: {i.sentence[:200]}" for i in review.issues
        ]
