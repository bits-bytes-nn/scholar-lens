"""The section loop shared by the guide and survey writers: draft →
evaluate → revise (keeping the best draft) → one re-scored targeted fix →
ground.

Each writer supplies its own calls (evaluate, revise, fix, ground) and length
target; this module holds the loop, the draft checks it applies (rule slips,
lost structure) and the helpers both writers' prompts are built from (the
glossary view, the concepts owned elsewhere, the measured prose length).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .logger import logger
from .markdown_lint import TABLE_DELIMITER, TABLE_ROW, close_open_fence
from .markdown_math import (
    DISPLAY_MATH,
    IMAGE,
    image_sources,
    segments,
    without_fences,
)
from .prompts.base import PROSE_UNIT
from .quality_gate import added_slips, publish_blockers, rule_issues
from .review_plan import draft_rank

# Evaluator feedback logged per section (the full text goes to the writer).
_LOGGED_FEEDBACK_CHARS: int = 300

_QUOTE_PREFIX = re.compile(r"^[ \t]*(?:>[ \t]?)+")
_LINKED_IMAGE = re.compile(rf"\[{IMAGE.pattern}\]\([^)\s]*\)")


def prose_chars(markdown: str) -> int:
    """Characters of prose (:data:`PROSE_UNIT`): whitespace collapsed to
    single spaces; fenced code, table rows (in a quote too, and a table
    written without outer pipes) and images (a linked one too) excluded, the
    text beside an image kept."""
    lines = [
        _QUOTE_PREFIX.sub("", line) for line in without_fences(markdown).splitlines()
    ]
    table = [bool(TABLE_ROW.match(line)) for line in lines]
    for i, line in enumerate(lines):
        # A pipe-less table: its header, its "--- | ---" row and the "a | b"
        # rows under it.
        if "|" in line and TABLE_DELIMITER.match(line) and i and "|" in lines[i - 1]:
            table[i - 1] = True
            end = i
            while end < len(lines) and "|" in lines[end]:
                table[end] = True
                end += 1
    kept = [
        IMAGE.sub("", _LINKED_IMAGE.sub("", line))
        for line, is_table in zip(lines, table, strict=True)
        if not is_table
    ]
    return len(" ".join(" ".join(kept).split()))


def measured_length(markdown: str, *, label: str, bound: str, target: int) -> str:
    """Measured prose length against a target, for an evaluator or a
    revision ("Target for <label>: <bound> <target>")."""
    return (
        f"Measured: {prose_chars(markdown):,} {PROSE_UNIT}. Target for "
        f"{label}: {bound} {target:,}."
    )


def parse_section_output(raw: str) -> dict[str, str]:
    """Extract ``<section_markdown>`` from the writer output; if the wrapper is
    missing or unbalanced, keep the text and drop stray tag fragments."""
    match = re.search(r"<section_markdown>(.*?)</section_markdown>", raw, re.DOTALL)
    text = match.group(1) if match else raw
    return {"section_markdown": re.sub(r"</?section_markdown>", "", text).strip()}


class OwningSection(Protocol):
    """A planned section as :func:`owned_elsewhere` reads it."""

    @property
    def title(self) -> str: ...

    @property
    def owns(self) -> list[str]: ...


def owned_elsewhere(sections: Sequence[OwningSection], index: int) -> str:
    """The concepts the other sections own (``index`` is 0-based), for the
    writer of section ``index``: used there, explained only by their owner."""
    own = set(sections[index].owns) if 0 <= index < len(sections) else set()
    lines = []
    for i, s in enumerate(sections):
        # A concept this section owns too is its own to explain.
        owns = [concept for concept in s.owns if concept not in own]
        if i != index and owns:
            where = "earlier" if i < index else "later"
            lines.append(f"- {where}, '{s.title}': {'; '.join(owns)}")
    return "\n".join(lines) or "- none"


_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")


def parse_glossary(raw: str) -> dict[str, str]:
    """English term -> rendering from the synopsis's ``<glossary>`` lines
    ("English term | rendering"); a parenthetical left in a rendering is
    dropped, since the writer adds the English at first use."""
    glossary: dict[str, str] = {}
    for line in raw.splitlines():
        english, sep, rendering = line.strip().lstrip("-* ").partition("|")
        english = english.strip()
        rendering = _PARENTHETICAL.sub("", rendering).strip()
        if sep and english and rendering:
            glossary.setdefault(english, rendering)
    return glossary


def glossary_view(glossary: dict[str, str], earlier: list[str]) -> str:
    """The glossary for one section: each rendering with its English, marked
    [introduced] when an ``earlier`` section already writes its first-use
    form "rendering(English)" (case-insensitive, a space before the
    parenthesis allowed), else [new]; a term written in English alone."""
    folded = [section.casefold() for section in earlier]
    lines = []
    for english, rendering in glossary.items():
        if rendering.casefold() == english.casefold():
            lines.append(f"- {rendering}")
        else:
            forms = [f"{rendering}{gap}({english})".casefold() for gap in ("", " ")]
            used = any(form in section for form in forms for section in folded)
            status = "introduced" if used else "new"
            lines.append(f"- {rendering} = {english} [{status}]")
    return "\n".join(lines) or "- none"


@dataclass
class ReviewedDraft:
    """The best draft of a section after :func:`revise_section`."""

    text: str
    score: int
    feedback: str
    # The evaluator's blocking issues, then its wording issues, then the
    # rule issues found in code.
    issues: list[str]
    # How many of ``issues`` are the evaluator's blocking (factual) issues.
    factual: int = 0
    # The targeted fix of the issues left (see :func:`revise_section`):
    # "applied and re-scored" (score and issues are the fixed draft's),
    # "reverted after re-scoring" (it ranked lower; the pre-fix draft kept),
    # "applied, not re-scored" (token budget reached; the issues stay to
    # check), "rejected" or "" (none ran).
    fix: str = ""

    @property
    def blocking(self) -> list[str]:
        """The evaluator's blocking (factual) issues alone."""
        return self.issues[: self.factual]

    def below_bar_note(self, kind: str, title: str, min_score: int) -> str | None:
        """The PR note for a draft left below ``min_score`` or with issues,
        naming each issue (truncated) for the reviewer; ``None`` when it
        passed."""
        if self.score >= min_score and not self.issues:
            return None
        fix = f"; targeted fix {self.fix}" if self.fix else ""
        listed = "; ".join(
            f'"{" ".join(i.split())[:_NOTED_ISSUE_CHARS]}"' for i in self.issues
        )
        return (
            f"{kind} section '{title}' kept below the bar (score {self.score}, "
            f"{len(self.issues)} blocking issue(s){fix}): check it"
            + (f": {listed}" if listed else ".")
        )


# Each issue in a below-the-bar PR note is cut to this many characters.
_NOTED_ISSUE_CHARS = 200

# How a revision and a targeted fix frame the draft for the section writer:
# a revision rewrites it, a fix edits it minimally.
REVISING = (
    "You are REVISING an existing draft of this section that did not pass "
    "review.\n\n{feedback}\n\nCURRENT DRAFT to improve (rewrite it, keeping "
    "what works):\n{draft}"
)
FIXING = "{request}\n\nTHE DRAFT to edit:\n{draft}"

TARGETED_FIX = (
    "TARGETED FIX: this draft is final except for these blocking issues. "
    "Make the smallest edit that resolves each one: rewrite or remove exactly "
    "the sentence(s) it names, and copy every other sentence, heading, "
    "citation, equation, figure, table and code block unchanged (edit a code "
    "block in place, never drop it). Do not rewrite, reorder, "
    "shorten or extend anything else.\n"
)
# Added to the feedback when a revision came back empty (retried once).
EMPTY_DRAFT = "The previous draft was empty. Write the full section."
# A targeted fix may shrink the prose by at most this share.
_FIX_MAX_PROSE_LOSS = 0.25
_HEADING_LINE = re.compile(r"^#{1,6}[ \t]+\S.*$", re.MULTILINE)
_CITATION_GROUP = re.compile(r"\[(@[^\[\]\n]+)\]")


def _citation_keys(text: str) -> set[str]:
    return {
        key.strip().removeprefix("@").strip()
        for group in _CITATION_GROUP.findall(text)
        for key in group.split(";")
    }


def _fence_count(text: str) -> int:
    return sum(seg.kind == "fence" for seg in segments(text))


def lost_structure(draft: str, edited: str) -> str:
    """What ``edited`` lost of ``draft``'s structure ("" when nothing): a
    heading, an image or a ``[@…]`` citation key it no longer has, fewer
    code blocks, display equations or tables (one may be corrected in place, not
    removed), or more than a quarter of the prose (:func:`prose_chars`).
    These are invariants of a minimal edit, not judgements of its quality."""
    if _fence_count(edited) < _fence_count(draft):
        return "loses a code block"
    before, after = without_fences(draft), without_fences(edited)
    for kind, items in (
        ("heading", lambda text: [h.strip() for h in _HEADING_LINE.findall(text)]),
        ("image", image_sources),
    ):
        if lost := list((Counter(items(before)) - Counter(items(after))).elements()):
            return f"loses {kind} {lost[0]!r}"
    if lost_keys := sorted(_citation_keys(before) - _citation_keys(after)):
        return f"loses citation [@{lost_keys[0]}]"
    for kind, count in (
        ("display equation", lambda text: len(DISPLAY_MATH.findall(text))),
        (
            "table",
            lambda text: sum(
                bool(TABLE_DELIMITER.match(line)) for line in text.splitlines()
            ),
        ),
    ):
        if count(after) < count(before):
            return f"loses a {kind}"
    old, new = prose_chars(draft), prose_chars(edited)
    if old and new < old * (1 - _FIX_MAX_PROSE_LOSS):
        return f"shrinks the prose from {old} to {new} characters"
    return ""


def feedback_or_generic(score: int, feedback: str, min_score: int) -> str:
    """Evaluator feedback, or a generic directive when a below-bar score
    came without any (the prompts only omit feedback for 90+ sections, so
    empty feedback with a low score is a model slip)."""
    if feedback or score >= min_score:
        return feedback
    return (
        "The section scored below the quality bar. Increase technical "
        "depth and concrete detail, tighten structure, and remove any "
        "redundancy with earlier sections."
    )


async def revise_section(
    markdown: str,
    *,
    evaluate: Callable[[str], Awaitable[tuple[int, str, list[str], list[str]]]],
    revise: Callable[[str, str], Awaitable[str]],
    length: Callable[[str], str],
    min_score: int,
    max_attempts: int,
    label: str,
    rules: Callable[[str], list[str]] = rule_issues,
    fix: Callable[[str, str], Awaitable[str]] | None = None,
    rescore: Callable[[], bool] = lambda: True,
) -> ReviewedDraft:
    """Score a section and revise it while it stays below ``min_score`` or
    has an issue: a blocking or wording issue ``evaluate`` names (it returns
    the score, the feedback, the blocking issues and the wording issues) or
    a rule ``rules`` finds in code.

    Keeps the BEST draft: a revision (``revise(feedback, draft)``) is
    re-scored and kept only if it ranks above the draft it revises
    (:func:`review_plan.draft_rank`, with the evaluator's blocking issues
    ahead of the score since they are about facts; wording issues and rule
    slips come after the score, as in the review loop), so a feedback-driven
    rewrite can never make the section worse. An empty revision is asked
    for once more. The feedback lists the issues first, then the measured
    ``length``, then the evaluator's other feedback. ``label`` names the
    section in the log.

    Issues left after the revisions get ONE targeted fix (``fix(request,
    draft)``, ``revise`` when not given, with only those issues and an order
    to edit minimally). The fixed draft is dropped when ``rules`` find
    something new in it or it loses the draft's structure
    (:func:`lost_structure`); otherwise it is re-scored with ONE ``evaluate``
    call and kept only if it ranks no lower than the pre-fix draft, so the
    result reports the fixed draft's score and issues. ``rescore()`` False
    (the token budget reached) skips that call: the fix is applied unscored
    and the pre-fix issues stay for the reviewer to check.
    """

    async def assess(text: str) -> tuple[int, str, list[str], int]:
        score, feedback, blocking, wording = await evaluate(text)
        # The issues to fix, and how many of them are about facts.
        return score, feedback, blocking + wording + rules(text), len(blocking)

    def rank(
        score: int, text: str, issues: list[str], factual: int
    ) -> tuple[bool, bool, int, int, int]:
        passes, publishable, score, slips = draft_rank(
            passes=score >= min_score and not issues,
            publishable=not publish_blockers(text),
            blocking=len(issues) - factual,
            score=score,
        )
        return passes, publishable, -factual, score, slips

    best_markdown = markdown
    best_score, best_feedback, best_issues, best_factual = await assess(markdown)
    for attempt in range(1, max_attempts + 1):
        # Accept only a score at the bar with no issue. Empty feedback with a
        # low score is a model slip (the prompt omits it only for 90+), so it
        # still revises with a generic directive.
        if best_score >= min_score and not best_issues:
            break
        # Issues first (they alone decide acceptance), then the measured
        # length, then the evaluator's other feedback.
        feedback = "\n\n".join(
            part
            for part in (
                (
                    "First fix every one of these blocking issues:\n"
                    + "\n".join(f"- {i}" for i in best_issues)
                    if best_issues
                    else ""
                ),
                f"Length: {length(best_markdown)}",
                (
                    f"Then apply this feedback:\n{best_feedback}"
                    if best_feedback
                    else ""
                ),
            )
            if part
        )
        logger.info(
            "Section %s scored %d (min %d), %d blocking issue(s) - revising "
            "(attempt %d/%d).",
            label,
            best_score,
            min_score,
            len(best_issues),
            attempt,
            max_attempts,
        )
        revised = await revise(feedback, best_markdown)
        if not revised:
            logger.warning(
                "Section %s revision came back empty - retrying once.", label
            )
            revised = await revise(f"{feedback}\n\n{EMPTY_DRAFT}", best_markdown)
        if not revised:
            break
        score, revised_feedback, issues, factual = await assess(revised)
        if rank(score, revised, issues, factual) <= rank(
            best_score, best_markdown, best_issues, best_factual
        ):
            # A rewrite of the same draft would mostly repeat this one.
            logger.info(
                "Section %s revision scored %d (discarded, best %d) - "
                "stopping revisions.",
                label,
                score,
                best_score,
            )
            break
        best_markdown, best_score = revised, score
        best_feedback, best_issues = revised_feedback, issues
        best_factual = factual
    logger.info(
        "Section %s final: score %d, %d blocking issue(s). Evaluator: %s",
        label,
        best_score,
        len(best_issues),
        " ".join(best_feedback.split())[:_LOGGED_FEEDBACK_CHARS] or "(none)",
    )
    outcome = ""
    if best_issues and max_attempts > 0:
        fixed = await (fix or revise)(
            TARGETED_FIX + "\n".join(f"- {i}" for i in best_issues), best_markdown
        )
        if not fixed or fixed == best_markdown:
            reason = "no change"
        elif added := added_slips(best_markdown, fixed, rules):
            reason = f"adds {added}"
        else:
            reason = lost_structure(best_markdown, fixed)
        if reason:
            outcome = "rejected"
            logger.info("Section %s targeted fix rejected (%s).", label, reason)
        elif not rescore():
            outcome = "applied, not re-scored"
            logger.info(
                "Section %s targeted fix of %d issue(s) applied (not re-scored: "
                "token budget reached).",
                label,
                len(best_issues),
            )
            best_markdown = fixed
        else:
            score, fixed_feedback, issues, factual = await assess(fixed)
            if rank(score, fixed, issues, factual) < rank(
                best_score, best_markdown, best_issues, best_factual
            ):
                outcome = "reverted after re-scoring"
                logger.info(
                    "Section %s targeted fix re-scored %d with %d issue(s) - "
                    "ranks lower, kept the pre-fix draft.",
                    label,
                    score,
                    len(issues),
                )
            else:
                outcome = "applied and re-scored"
                logger.info(
                    "Section %s targeted fix applied: re-scored %d, %d issue(s) "
                    "left (was %d).",
                    label,
                    score,
                    len(issues),
                    len(best_issues),
                )
                best_markdown, best_score = fixed, score
                best_feedback, best_issues = fixed_feedback, issues
                best_factual = factual
    return ReviewedDraft(
        best_markdown, best_score, best_feedback, best_issues, best_factual, outcome
    )


async def checked_grounding(draft: str, ground: Callable[[], Awaitable[str]]) -> str:
    """The grounded section (``ground()`` fact-checks ``draft``): grounding
    is the last pass, so one retry when it broke a rule (e.g. hedged a claim
    instead of removing it), and the ungrounded draft when both attempts
    broke something the validator rejects."""
    grounded = close_open_fence(await ground())
    if added_slips(draft, grounded):
        retried = close_open_fence(await ground())
        if len(added_slips(draft, retried)) < len(added_slips(draft, grounded)):
            grounded = retried
        if len(publish_blockers(grounded)) > len(publish_blockers(draft)):
            logger.info(
                "Grounding added publish blockers on both attempts: kept the draft "
                "before grounding."
            )
            grounded = draft
    return grounded
