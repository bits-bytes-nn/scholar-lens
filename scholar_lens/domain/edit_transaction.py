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
from collections.abc import Sequence
from dataclasses import dataclass

from scholar_lens.domain.evidence_draft import (
    apply_edits,
    claim_units,
    ends_mid_sentence,
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


def transact(
    before: str,
    edits: Sequence[tuple[str, str]],
    *,
    verifiers: VerifierSet,
    max_prose_cut: float,
    rest: str = "",
) -> Revision:
    """``before`` revised by ``edits``, or as it was when they may not stand.

    Args:
        before: The section.
        edits: (find, replace) pairs; an empty replace deletes.
        verifiers: The section's verifier set.
        max_prose_cut: The largest share of the prose one revision may cut.
        rest: The other sections of the post: a figure, equation, table or
            code block may go when an equivalent copy stays there.

    Returns:
        The revision (``taken`` when the section changed).
    """
    fixed, applied, skipped = apply_edits(before, edits, rest=rest)
    if not applied:
        return Revision(before, skipped=tuple(skipped))
    fixed = strip_revision_notes(fixed, before)
    reason = structure_lost(before, fixed, rest=rest, max_prose_cut=max_prose_cut) or (
        "it stops mid-sentence" if ends_mid_sentence(fixed) else ""
    )
    if reason:
        return Revision(before, applied, tuple(skipped), reason)
    if added := verifiers.added(before, fixed):
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
