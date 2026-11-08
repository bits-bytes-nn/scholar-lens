"""The edit transaction every change of a section goes through
(docs/engine.md, 4.4).

A fix, a whole-post fix and the narrowing of unsupported sentences all
hand a batch of (find, replace) edits to :func:`transact`. The batch is
applied and its final text checked (fences, the ``##`` headings, the only
copy of a figure, equation, table or code block, writing-rule slips), edit
by edit when the final text fails; then the section's verifier set runs
again, and a batch that adds a normalised problem is not taken. What the
transaction created or changed is returned for the claim check.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from scholar_lens.domain.evidence_draft import (
    VISUALS_PROBLEM,
    apply_edits,
    claim_units,
    ends_mid_sentence,
    removed_duplicate_visuals,
    strip_revision_notes,
    structure_lost,
)
from scholar_lens.domain.verifiers import Problem, VerifierSet


@dataclass(frozen=True)
class Revision:
    """What a transaction made of a section.

    Attributes:
        text: The section after the transaction (the draft as it was when
            the batch was not taken).
        applied: How many edits applied.
        skipped: Why the other edits did not.
        rejected: Why the applied edits were not taken ("" when they were,
            or when none applied).
        added: The new problems a rejected batch would have brought.
        changed: The claim units of ``text`` the draft did not have (new or
            changed sentences and table rows).
    """

    text: str
    applied: int = 0
    skipped: tuple[str, ...] = ()
    rejected: str = ""
    added: tuple[Problem, ...] = ()
    changed: tuple[str, ...] = ()

    @property
    def taken(self) -> bool:
        """Whether the section changed."""
        return self.applied > 0 and not self.rejected


def changed_units(before: str, after: str) -> list[str]:
    """The claim units of ``after`` that ``before`` does not have (as often
    as ``after`` has them more), in order.

    Args:
        before: A section.
        after: The section revised.

    Returns:
        The new or changed sentences and table rows.
    """
    left = Counter(claim_units(before))
    out = []
    for unit in claim_units(after):
        if left[unit] > 0:
            left[unit] -= 1
        else:
            out.append(unit)
    return out


def rejudged_units(before: str, after: str) -> list[str]:
    """The claim units of ``after`` whose claim check reads differently from
    ``before``'s: new or changed units and the unit right after a change
    (an insertion, a rewrite or a deletion before it), whose context
    sentence the verdict depends on (docs/engine.md, 4.2), in order.

    Args:
        before: A section.
        after: The section revised.

    Returns:
        The units whose (previous unit, unit) pair ``before`` does not have
        (as often as ``after`` has it more).
    """

    def pairs(units: list[str]) -> list[tuple[str, str]]:
        return [(units[n - 1] if n else "", u) for n, u in enumerate(units)]

    left = Counter(pairs(claim_units(before)))
    out = []
    for pair in pairs(claim_units(after)):
        if left[pair] > 0:
            left[pair] -= 1
        else:
            out.append(pair[1])
    return out


def transact(
    before: str,
    edits: Sequence[tuple[str, str]],
    *,
    verifiers: VerifierSet,
    max_prose_cut: float,
    rest: str = "",
    later: str = "",
    tolerated: Collection[str] = (),
) -> Revision:
    """``before`` revised by ``edits``, or as it was when they may not stand.

    Args:
        before: The section.
        edits: (find, replace) pairs; an empty replace deletes.
        verifiers: The section's verifier set.
        max_prose_cut: The largest share of the prose one revision may cut.
        rest: The other sections of the post: a figure, equation or table
            may go when an equivalent copy stays there, a code block when
            one stays in a section before this one.
        later: The part of ``rest`` after this section ("": all of it
            comes before).
        tolerated: Problem kinds a new problem of which does not refuse
            the batch (a deletion of unsupported content may leave the
            section short; the caller notes it).

    Returns:
        The revision (``taken`` when the section changed).
    """
    owned = verifiers.figures
    fixed, applied, skipped = apply_edits(
        before, edits, rest=rest, owned_figures=owned, later=later
    )
    if not applied:
        return Revision(before, skipped=tuple(skipped))
    fixed = strip_revision_notes(fixed, before)
    reason = structure_lost(
        before,
        fixed,
        rest=rest,
        max_prose_cut=max_prose_cut,
        owned_figures=owned,
        later=later,
    ) or ("it stops mid-sentence" if ends_mid_sentence(fixed) else "")
    if reason:
        return Revision(before, applied, tuple(skipped), reason)
    added = verifiers.added(before, fixed)
    if rest and removed_duplicate_visuals(before, fixed, rest):
        # A repeated table or figure deleted where another section keeps
        # it: the visuals minimum asks the writer for visuals of the
        # section's own, it does not keep a copy another section owns.
        added = [p for p in added if p.kind != VISUALS_PROBLEM]
    added = [p for p in added if p.kind not in tolerated]
    if added:
        return Revision(
            before,
            applied,
            tuple(skipped),
            "adds problems: " + " | ".join(p.text[:120] for p in added[:3]),
            tuple(added),
        )
    return Revision(
        fixed, applied, tuple(skipped), changed=tuple(changed_units(before, fixed))
    )
