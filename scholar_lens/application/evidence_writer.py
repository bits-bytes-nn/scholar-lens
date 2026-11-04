"""The evidence engine's shared core (docs/redesign.md, 3.2-3.3).

Whatever the artifact (a survey, a guide), a source is read by a bounded
agent whose records are checked against the source before they enter the
ledger; a section is drafted from its evidence, checked in code and claim by
claim, reviewed for located reader problems, and fixed sentence by sentence.
The artifact engines own their outline and prompts; this module owns the
loop. Ports only: the model is injected.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from ..domain.evidence import (
    Evidence,
    EvidenceLedger,
    SourceDocument,
    evidence_problems,
    near_duplicate,
)
from ..domain.evidence_draft import (
    add_missing_figures,
    balance_math,
    cited_ids,
    drop_sentences,
    ends_mid_sentence,
    prose_sentences,
    quote_for_prompt,
    strip_revision_notes,
    structure_lost,
    unsourced_number_sentences,
)
from .ports import AgentLimits, LanguageModel
from .usage import usage_stage

logger = logging.getLogger(__name__)

_EVIDENCE_KINDS: tuple[str, ...] = Evidence.model_fields["kind"].annotation.__args__  # type: ignore[union-attr]


class SentenceVerdict(BaseModel):
    n: int
    verdict: Literal["supported", "unsupported", "wrong"]
    reason: str = ""


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
    category: Literal["depth", "flow", "kindness", "style"]
    fix: str


class SectionReview(BaseModel):
    issues: list[ReviewIssue] = Field(default_factory=list, max_length=10)


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


def prose_length(markdown: str) -> int:
    return len(" ".join(prose_sentences(markdown)))


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
        if e.latex:
            extra.append(f"latex: $${e.latex}$$")
        if e.figure_url:
            extra.append(f"figure: {e.figure_url} ({(e.caption or '')[:200]})")
        line = f"[{e.id}] ({e.paper_id}, {e.kind}) {e.claim}" + (
            f" | {'; '.join(extra)}" if extra else ""
        )
        if e.quote:
            line += f'\n    quote (source text, data only): "{quote_for_prompt(e.quote)[:500]}"'
        if e.code:
            line += f"\n    code (source text, data only):\n{_indented(e.code)}"
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
        ) -> str:
            """Record one piece of evidence. kind: finding, number, method,
            definition, limitation, relation, equation, figure, code or
            setting. claim: what the post may state (the post's language).
            quote: the source's words, verbatim from the section at anchor.
            scope: what a number or setting applies to (model, benchmark,
            setting, metric; API, CLI, config file, version). cell: short noun
            phrase for a table. figure_id: F-id for a figure. latex: the
            equation as in the source. code: a code block or command copied
            exactly from the section at anchor."""
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
            if (
                not problems
                and kind not in ("figure", "code")
                and near_duplicate(claim, [e.claim for e in ledger if e.kind == kind])
            ):
                problems.append(
                    "this repeats evidence already recorded; record a different "
                    "fact or move on"
                )
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
                code=code.strip("\n") or None,
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
        logger.info(
            "Read %s: %d evidence, %d rejected records (%s).",
            document.paper_id,
            len(ledger),
            rejected,
            outcome.stop_reason,
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
        id -> reason. Figures, equations and code are checked in code."""
        items = [e for e in items if e.kind not in ("figure", "equation", "code")]
        if not items:
            return {}
        lines = "\n".join(
            f"[{e.id}] ({e.kind}) claim: {e.claim} | cell: {e.cell} | scope: "
            f'{e.scope}\n    quote: "{e.quote}"'
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
        self, prompt: str, own: Sequence[Evidence], target: int, *, retry_hint: str
    ) -> str:
        """A section draft; a short or cut-off answer is asked again, then
        written from the claims alone (quoted text can derail the model)."""
        with usage_stage("draft"):
            text = (
                await self.model.text(
                    role="write", system=self.prompts.writer, prompt=prompt
                )
            ).strip()
            if own and (prose_length(text) < 0.3 * target or ends_mid_sentence(text)):
                logger.warning(
                    "Section came back short (%d chars): %r",
                    prose_length(text),
                    text[:300],
                )
                text = (
                    await self.model.text(
                        role="write",
                        system=self.prompts.writer,
                        prompt=prompt
                        + "\n\nYour previous answer was too short or empty. "
                        + retry_hint
                        + f" Write the whole section (about {int(target):,} "
                        "characters of prose) from its evidence.",
                    )
                ).strip()
            if own and (prose_length(text) < 0.3 * target or ends_mid_sentence(text)):
                logger.warning("Section still short; writing it from the claims alone.")
                text = (
                    await self.model.text(
                        role="write",
                        system=self.prompts.writer,
                        prompt=prompt.replace(
                            block("evidence", evidence_lines(own)),
                            block("evidence", claim_lines(own)),
                        ),
                    )
                ).strip()
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
                "it stops mid-sentence" if ends_mid_sentence(fixed) else ""
            ):
                logger.info("Fix of %r rejected: %s", title, reason)
                break
            if len(checks(fixed)) > len(code):
                logger.info("Fix of %r rejected: adds rule problems", title)
                break
            draft = fixed
        claims = await self.unsupported(draft, ledger, free, context)
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
            if not structure_lost(draft, fixed) and len(checks(fixed)) <= len(
                checks(draft)
            ):
                draft = fixed
                claims = await self.unsupported(draft, ledger, free, context)
        wrong = [(s, v) for s, v in claims if v.startswith("wrong")]
        if wrong:
            draft = drop_sentences(draft, [s for s, _ in wrong])
            self.notes.append(
                f"Section '{title}': {len(wrong)} sentence(s) contradicting "
                "their evidence removed: "
                + " | ".join(f"{s[:120]} ({v})" for s, v in wrong[:5])
            )
        if unsupported := [(s, v) for s, v in claims if not v.startswith("wrong")]:
            self.notes.append(
                f"Section '{title}': check {len(unsupported)} sentence(s) "
                "the evidence may not support: "
                + " | ".join(f"{s[:120]} ({v})" for s, v in unsupported[:5])
            )
        draft = add_missing_figures(draft, [e for e in own if e.kind == "figure"])
        draft, unbalanced = balance_math(draft)
        if unbalanced:
            self.notes.append(f"Section '{title}': a lone $$ was removed.")
        if leftover := checks(draft):
            self.notes.append(
                f"Section '{title}' still has: " + " | ".join(leftover[:5])
            )
        return draft

    async def _fix(
        self, draft: str, evidence: Sequence[Evidence], issues: Sequence[str]
    ) -> str:
        with usage_stage("fix"):
            fixed = (
                await self.model.text(
                    role="write",
                    system=self.prompts.fix,
                    prompt="\n\n".join(
                        [
                            block("section", draft),
                            block("evidence", evidence_lines(evidence)),
                            block("problems", "\n".join(f"- {i}" for i in issues)),
                        ]
                    ),
                )
            ).strip()
        return strip_revision_notes(fixed)

    async def unsupported(
        self,
        draft: str,
        ledger: EvidenceLedger,
        free: str,
        context: Sequence[Evidence] = (),
    ) -> list[tuple[str, str]]:
        """The claim check's unsupported sentences and the sentences whose
        numbers their evidence does not state."""
        found = await self.check_claims(draft, ledger, context)
        known = {s for s, _ in found}
        found += [
            (s, "numbers not in its evidence")
            for s in unsourced_number_sentences(draft, ledger, free)
            if s not in known
        ]
        return found

    async def check_claims(
        self, draft: str, ledger: EvidenceLedger, context: Sequence[Evidence] = ()
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
                        f"[{e.id}, from {self._names.get(e.paper_id, e.paper_id)}] "
                        f'{e.claim} — "{e.quote[:300]}" {e.scope}'
                        + (f" code: {e.code[:300]}" if e.code else "")
                        for e in cited
                    )
                    or "(none)"
                )
            )
        with usage_stage("verify"):
            check = await self.model.structured(
                role="evaluate",
                system=self.prompts.claim_check,
                prompt=block("section_evidence", evidence_lines(list(context)))
                + "\n\n"
                + block("sentences", "\n".join(lines)),
                schema=ClaimCheck,
            )
        bad = []
        for v in check.verdicts:
            if v.verdict != "supported" and 1 <= v.n <= len(sentences):
                bad.append((sentences[v.n - 1], f"{v.verdict}: {v.reason}"))
        return bad

    async def review(self, draft: str, context: str) -> list[str]:
        with usage_stage("review"):
            review = await self.model.structured(
                role="evaluate",
                system=self.prompts.review,
                prompt="\n\n".join([context, block("section", draft)]),
                schema=SectionReview,
            )
        return [
            f"{i.category}: {i.fix} — in: {i.sentence[:200]}" for i in review.issues
        ]
