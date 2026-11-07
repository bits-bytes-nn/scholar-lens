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
import json
import logging
import re
import threading
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError

from ..domain.evidence import (
    Evidence,
    EvidenceLedger,
    SourceDocument,
    evidence_problems,
    near_duplicate,
)
from ..domain.evidence_draft import (
    add_missing_figures,
    apply_edits,
    balance_math,
    cited_ids,
    claim_units,
    drop_empty_headings,
    drop_sentences,
    ends_mid_sentence,
    prose_sentences,
    quote_for_prompt,
    section_heading,
    strip_revision_notes,
    structure_lost,
    unsourced_number_sentences,
)
from ..prompts.rules import FIX_ANSWER
from .ports import AgentLimits, LanguageModel, ModelText, was_cut
from .usage import usage_stage

logger = logging.getLogger(__name__)

SectionT = TypeVar("SectionT")

_EVIDENCE_KINDS: tuple[str, ...] = Evidence.model_fields["kind"].annotation.__args__  # type: ignore[union-attr]


# Turns of the follow-up that asks a reader for a required kind it skipped.
FOLLOW_UP_TURNS = 8


class SentenceVerdict(BaseModel):
    n: int
    # about_sources: the sentence is about the post's inputs (its evidence,
    # the material it was given), not the subject; it says nothing to a
    # reader and is removed like a contradiction.
    verdict: Literal["supported", "unsupported", "wrong", "about_sources"]
    reason: str = ""
    # The item's atomic claims the evidence does not support, each on its
    # own (an implied cause, scope or frequency counts as a claim).
    unsupported_parts: list[str] = Field(default_factory=list, max_length=8)


class ClaimCheck(BaseModel):
    verdicts: list[SentenceVerdict] = Field(default_factory=list)


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


def cached_evidence(
    raw: str,
    document: SourceDocument,
    rejected: Callable[[Evidence], object] = lambda _e: False,
    *,
    key: str = "",
) -> list[Evidence] | None:
    """A document's stored evidence still usable today, bound to the
    document's current source id (ids are numbered per run), or None when it
    must be read again: unreadable, or nothing passes today's checks."""
    try:
        items = [Evidence.model_validate(e) for e in json.loads(raw)]
    except (ValueError, ValidationError):
        logger.warning("Unreadable cached evidence %s; reading again.", key)
        return None
    items = [e.model_copy(update={"paper_id": document.paper_id}) for e in items]
    kept = [e for e in items if not recheck(document, e) and not rejected(e)]
    if items and not kept:
        logger.warning("No cached evidence of %s passes today; reading again.", key)
        return None
    return kept


def prose_length(markdown: str) -> int:
    return len(" ".join(prose_sentences(markdown)))


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


POST_SO_FAR_CHARS = 60_000


def post_so_far(sections: Sequence[str], limit: int = POST_SO_FAR_CHARS) -> str:
    """The sections already written, in order, as the next section's writer
    reads them (evidence markers kept: what is cited is visible), the most
    recent kept whole when the post outgrows ``limit``."""
    text = "\n\n".join(md.strip() for md in sections if md.strip())
    if not text:
        return ""
    if len(text) > limit:
        text = "[…]\n" + text[-limit:]
    return block("post_so_far", text)


def _kept(answer: str) -> str:
    """A model answer stripped of surrounding whitespace, its cut flag kept."""
    return ModelText(answer.strip(), cut=was_cut(answer))


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
    return f"<{tag}>\n{text.strip() or '(none)'}\n</{tag}>"


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
        self, model: LanguageModel, prompts: WriterPrompts, *, max_fix_rounds: int
    ) -> None:
        self.model = model
        self.prompts = prompts
        self.max_fix_rounds = max_fix_rounds
        self.notes: list[str] = []
        self._names: dict[str, str] = {}
        # Source passages the current section's evidence comes from (guides):
        # the claim check and the fix read them besides the evidence.
        self.reference = ""
        # (source id, anchor) -> section text: numbers a sentence states are
        # checked against the whole sections its evidence is quoted from
        # (guides and papers; a survey checks against the evidence alone).
        self.passages: dict[tuple[str, str], str] = {}

    # ---- read -------------------------------------------------------------------

    async def read_document(
        self,
        document: SourceDocument,
        *,
        prompt: str,
        limits: AgentLimits,
        required: Sequence[str] = (),
        rule: RecordRule | None = None,
    ) -> list[Evidence]:
        """One agent reads ``document`` and records evidence; every record
        is checked against the document (and ``rule``) before it is kept.
        ``finish`` asks twice for any ``required`` kind still missing."""
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
            return f"{section.anchor} | {section.heading}\n\n{section.text[:30_000]}"

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
            if missing and refusals < 2:
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
        # The agent may end its turn without calling finish: ask once, in a
        # short follow-up, for the required kinds it has not recorded.
        if missing := [k for k in required if k not in {e.kind for e in ledger}]:
            recorded = "\n".join(f"- ({e.kind}) {e.claim[:120]}" for e in ledger)
            await self.model.run_agent(
                role="read",
                system=self.prompts.reader,
                prompt="\n\n".join(
                    [
                        prompt,
                        block("already_recorded", recorded),
                        f"Evidence of kind {missing} is still missing. Record "
                        "it now if the source has it (for a figure: the one "
                        "that best shows the architecture, a flow or a result, "
                        "described from the text before it), then call finish.",
                    ]
                ),
                tools=[read_section, search_source, record_evidence, finish],
                limits=AgentLimits(
                    turns=FOLLOW_UP_TURNS, total_tokens=limits.total_tokens
                ),
            )
        logger.info(
            "Read %s: %d evidence, %d rejected records (%s).",
            document.paper_id,
            len(ledger),
            rejected,
            outcome.stop_reason,
        )
        if reasons:
            logger.info(
                "Rejections of %s: %s",
                document.paper_id,
                "; ".join(f"{r} x{n}" for r, n in reasons.most_common(6)),
            )
        if outcome.stop_reason.startswith("limit"):
            self.notes.append(
                f"Reading {document.paper_id} stopped at its limit "
                f"({outcome.stop_reason})."
            )
        return list(ledger)

    async def wrong_evidence(self, items: Sequence[Evidence]) -> dict[str, str]:
        """Evidence whose claim, scope or cell misstates its quote (a scope
        can name the wrong benchmark while every number is in the quote):
        id -> reason. A figure's claim is checked against its caption and
        the text before it; equations and code are checked in code."""
        items = [e for e in items if e.kind not in ("equation", "code")]
        items = [e for e in items if e.kind != "figure" or e.caption or e.quote]
        if not items:
            return {}
        lines = "\n".join(
            f"[{e.id}] ({e.kind}) claim: {e.claim} | cell: {e.cell} | scope: "
            f'{e.scope}\n    quote: "{quote_for_prompt(e.quote)}"'
            + (f'\n    caption: "{quote_for_prompt(e.caption)}"' if e.caption else "")
            for e in items
        )
        with usage_stage("verify"):
            check = await self.model.structured(
                role="evaluate",
                system=self.prompts.evidence_check,
                prompt=block("evidence", lines),
                schema=EvidenceCheck,
            )
        wrong = {v.id: v.reason for v in check.verdicts if v.verdict == "wrong"}
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
        context: str = "",
    ) -> str:
        """A section draft; a short or cut-off answer is asked again, then
        written from the claims alone (quoted text can derail the model):
        the prompt's ``<evidence>`` block, holding ``evidence`` (``own``
        unless given), is swapped for their claims. ``context`` (the post
        so far) goes before the cache point."""
        with usage_stage("draft"):
            text = _kept(
                await self.model.text(
                    role="write",
                    system=self.prompts.writer,
                    prompt=prompt,
                    context=context,
                )
            )
            if (own or evidence) and (
                prose_length(text) < 0.3 * target or _incomplete(text)
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
                        context=context,
                        prompt=prompt
                        + "\n\nYour previous answer was too short or empty. "
                        + retry_hint
                        + f" Write the whole section (about {int(target):,} "
                        "characters of prose) from its evidence.",
                    )
                )
            if (own or evidence) and (
                prose_length(text) < 0.3 * target or _incomplete(text)
            ):
                logger.warning("Section still short; writing it from the claims alone.")
                text = _kept(
                    await self.model.text(
                        role="write",
                        system=self.prompts.writer,
                        context=context,
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
        checks: Callable[[str], list[str]],
        ledger: EvidenceLedger,
        own: Sequence[Evidence],
        earlier: Sequence[Evidence],
        free: str,
    ) -> str:
        """Code checks, the claim check and (first round) the review, fixed
        up to ``max_fix_rounds``; then unsupported sentences narrowed in
        context, contradicting ones removed, the rest noted; missing figures
        appended and a lone ``$$`` removed."""
        context = [*own, *earlier]
        for round_ in range(self.max_fix_rounds + 1):
            code = checks(draft)
            claims = await self.check_claims(draft, ledger, context)
            review = await self.review(draft, review_context) if round_ == 0 else []
            issues = code + [f"{v}: {s}" for s, v in claims] + review
            if not issues or round_ == self.max_fix_rounds:
                break
            fixed = await self._fix(draft, [*own, *earlier[-30:]], issues)
            if reason := structure_lost(draft, fixed) or (
                "it stops mid-sentence" if _incomplete(fixed) else ""
            ):
                logger.info("Fix of %r rejected: %s", title, reason)
                break
            if len(checks(fixed)) > len(code):
                logger.info("Fix of %r rejected: adds rule problems", title)
                break
            if fixed.strip() == draft.strip():
                logger.info("Fix of %r changed nothing; stopping.", title)
                break
            draft = fixed
        # The loop ends on the draft its last claim check read: reused, not
        # paid for again.
        claims = await self.unsupported(draft, ledger, free, context, checked=claims)
        if claims:
            # Let the writer narrow or delete them in context first; only what
            # is still unsupported after that is cut.
            fixed = await self._fix(
                draft,
                [*own, *earlier[-30:]],
                [
                    "These sentences are not supported by their evidence; narrow "
                    "each to what the evidence says or delete it, and mend the "
                    "sentences around a deletion:\n"
                    + "\n".join(f"- {v}: {s}" for s, v in claims)
                ],
            )
            if (
                fixed.strip() != draft.strip()
                and not structure_lost(draft, fixed)
                and not _incomplete(fixed)
                and len(checks(fixed)) <= len(checks(draft))
            ):
                draft = fixed
                claims = await self.unsupported(draft, ledger, free, context)
        wrong = [(s, v) for s, v in claims if _removable(v)]
        if wrong:
            draft = drop_sentences(draft, [s for s, _ in wrong])
            left = set(claim_units(draft))
            removed = [(s, v) for s, v in wrong if s not in left]
            if removed:
                self.notes.append(
                    f"Section '{title}': {len(removed)} sentence(s) removed that "
                    "contradict their evidence or talk about the post's sources: "
                    + " | ".join(f"{s[:120]} ({v})" for s, v in removed[:5])
                )
            if kept := [(s, v) for s, v in wrong if s in left]:
                self.notes.append(
                    f"Section '{title}': fix by hand, {len(kept)} sentence(s) "
                    "contradict their evidence: "
                    + " | ".join(f"{s[:120]} ({v})" for s, v in kept[:5])
                )
        if unsupported := [(s, v) for s, v in claims if not _removable(v)]:
            self.notes.append(
                f"Section '{title}': check {len(unsupported)} sentence(s) "
                "the evidence may not support: "
                + " | ".join(f"{s[:120]} ({v})" for s, v in unsupported[:5])
            )
        draft = add_missing_figures(draft, [e for e in own if e.kind == "figure"])
        draft = drop_empty_headings(draft)
        draft, unbalanced = balance_math(draft)
        if unbalanced:
            self.notes.append(f"Section '{title}': a lone $$ was removed.")
        if leftover := checks(draft):
            self.notes.append(
                f"Section '{title}' still has: " + " | ".join(leftover[:5])
            )
        return draft

    async def apply_findings(
        self,
        written: Sequence[tuple[SectionT, str]],
        findings: Sequence[PostFinding],
        ledger: EvidenceLedger,
        *,
        title_of: Callable[[SectionT], str],
        own_of: Callable[[SectionT], list[Evidence]],
        reference_of: Callable[[Sequence[Evidence]], str],
    ) -> list[tuple[SectionT, str]]:
        """Each section a whole-post review names, fixed with its evidence
        and source passages; a fix that loses structure is not taken, and
        sentences it makes that contradict the evidence are removed."""
        out: list[tuple[SectionT, str]] = []
        names_seen = {
            n
            for section, md in written
            for n in (title_of(section), section_heading(md))
        }
        if unmatched := [
            f
            for f in findings
            if f.section.strip().lstrip("#").strip() not in names_seen
        ]:
            self.notes.append(
                "Whole-post review problems naming no section of the post, not "
                "applied: "
                + " | ".join(f"{f.section}: {f.problem}"[:160] for f in unmatched[:3])
            )
        for section, md in written:
            names = {title_of(section), section_heading(md)}
            issues = [
                f"{f.problem} Fix: {f.fix}"
                for f in findings
                if f.section.strip().lstrip("#").strip() in names
            ]
            if not issues:
                out.append((section, md))
                continue
            own = own_of(section)
            self.reference = reference_of(own)
            fixed = await self._fix(md, own, issues)
            if (
                structure_lost(md, fixed)
                or _incomplete(fixed)
                or fixed.strip() == md.strip()
            ):
                self.notes.append(
                    f"Section '{title_of(section)}': whole-post review problems "
                    "left as they were: " + " | ".join(i[:160] for i in issues[:3])
                )
            else:
                wrong = [
                    s
                    for s, v in await self.check_claims(fixed, ledger, own)
                    if _removable(v)
                ]
                md = drop_sentences(fixed, wrong) if wrong else fixed
                left = set(claim_units(md))
                if removed := [w for w in wrong if w not in left]:
                    self.notes.append(
                        f"Section '{title_of(section)}': {len(removed)} sentence(s) "
                        "from a whole-post fix removed (they contradict their "
                        "evidence or talk about the post's sources): "
                        + " | ".join(w[:120] for w in removed[:3])
                    )
                if kept := [w for w in wrong if w in left]:
                    self.notes.append(
                        f"Section '{title_of(section)}': fix by hand, "
                        f"{len(kept)} sentence(s) contradict their evidence: "
                        + " | ".join(w[:120] for w in kept[:3])
                    )
            self.reference = ""
            out.append((section, md))
        return out

    async def _fix(
        self, draft: str, evidence: Sequence[Evidence], issues: Sequence[str]
    ) -> str:
        """The section revised for ``issues`` by minimal edits (each an
        exact passage replaced or deleted, applied by code); when none of
        the edits applies, the whole revised section instead. The answers
        are structured, so no note reaches the post. An answer that does not
        parse leaves the draft as it was."""
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
        with usage_stage("fix"):
            try:
                edits = await self.model.structured(
                    role="write",
                    system=self.prompts.fix + EDIT_ANSWER,
                    schema=SectionEdits,
                    # The evidence and passages repeat across a section's
                    # fixes; only the draft and its problems change.
                    context=context,
                    prompt=prompt,
                )
            except ValueError as e:
                logger.warning("Edit answer unusable (%s); asking for the section.", e)
                edits = None
            if edits is not None:
                fixed, failed = apply_edits(
                    draft, [(e.find, e.replace) for e in edits.edits]
                )
                self._writer_notes(heading, edits.notes)
                if not edits.edits or len(failed) < len(edits.edits):
                    if failed:
                        logger.info(
                            "%d of %d edits of %r matched nothing.",
                            len(failed),
                            len(edits.edits),
                            heading,
                        )
                    return strip_revision_notes(fixed, draft)
            try:
                answer = await self.model.structured(
                    role="write",
                    system=self.prompts.fix + FIX_ANSWER,
                    schema=FixedSection,
                    context=context,
                    prompt=prompt,
                )
            except ValueError as e:
                logger.warning("Fix answer unusable (%s); the draft stands.", e)
                self.notes.append(
                    f"Section '{heading}': a fix could not be made ({e}); "
                    "the draft stands."
                )
                return draft
        self._writer_notes(heading, answer.notes)
        return strip_revision_notes(answer.section.strip(), draft)

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
        context: Sequence[Evidence] = (),
        *,
        checked: list[tuple[str, str]] | None = None,
    ) -> list[tuple[str, str]]:
        """The claim check's unsupported sentences (``checked``: the result
        of one already run on this very draft) and the sentences whose
        numbers their evidence does not state."""
        found = (
            list(checked)
            if checked is not None
            else await self.check_claims(draft, ledger, context)
        )
        known = {s for s, _ in found}
        found += [
            (s, "numbers not in its evidence")
            for s in unsourced_number_sentences(draft, ledger, free, self.passages)
            if s not in known
        ]
        return found

    async def check_claims(
        self, draft: str, ledger: EvidenceLedger, context: Sequence[Evidence] = ()
    ) -> list[tuple[str, str]]:
        """(sentence, verdict with reason) of the sentences the evidence does
        not support."""
        # Table rows are claims too (a wrong cell is as wrong as a sentence).
        sentences = claim_units(draft)
        if not sentences:
            return []
        lines = []
        for n, sentence in enumerate(sentences, 1):
            cited = [e for i in cited_ids(sentence) if (e := ledger.get(i))]
            lines.append(
                f"{n}. {sentence}\n   evidence: "
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
        with usage_stage("verify"):
            check = await self._evaluated(
                f"the claim check of '{section_heading(draft) or 'a section'}'",
                role="evaluate",
                system=self.prompts.claim_check,
                # Shared by every claim check of the section (each round).
                context=block("section_evidence", evidence_lines(list(context)))
                + (
                    "\n\n" + block("source_sections", self.reference)
                    if self.reference
                    else ""
                ),
                prompt=block("sentences", "\n".join(lines)),
                schema=ClaimCheck,
            )
        if check is None:
            return []
        bad = []
        for v in check.verdicts:
            if not 1 <= v.n <= len(sentences):
                continue
            parts = [p for p in v.unsupported_parts if p.strip()]
            # A part not supported makes the item unsupported, whatever the
            # item's own label says.
            verdict = "unsupported" if v.verdict == "supported" and parts else v.verdict
            if verdict != "supported":
                reason = v.reason + (
                    " (not supported: " + "; ".join(parts) + ")" if parts else ""
                )
                bad.append((sentences[v.n - 1], f"{verdict}: {reason.strip()}"))
        return bad

    async def _evaluated(self, what: str, **call: Any) -> Any:
        """A structured check's answer, or None when the model gave no valid
        answer even on the fallback model (a content filter, a cut answer):
        the run goes on with the draft as it is, and the run notes say which
        check did not run, so the PR checklist shows it."""
        try:
            return await self.model.structured(**call)
        except ValueError as e:
            logger.warning("%s did not run: %s", what, e)
            self.notes.append(f"Check by hand: {what} could not run ({e}).")
            return None

    async def review(self, draft: str, context: str) -> list[str]:
        with usage_stage("review"):
            review = await self._evaluated(
                f"the review of '{section_heading(draft) or 'a section'}'",
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
