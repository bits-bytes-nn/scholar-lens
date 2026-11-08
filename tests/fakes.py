"""Test doubles shared across the suite that need no adapter (the unit
tests import only these; adapter-backed doubles are in
``tests/integration/fakes.py``)."""

from __future__ import annotations

import re
from collections.abc import Callable

from scholar_lens.application.evidence_writer import ClaimCheck, SentenceVerdict


def with_context(context: str | tuple[str, ...], prompt: str) -> str:
    """A scripted model's view of a call: its context parts (one string, or
    the parts a cached call passes) before its prompt."""
    parts = (context,) if isinstance(context, str) else context
    return "\n\n".join([*parts, prompt])


_ITEM = re.compile(r"^(\d+)\. (.*)$", re.M)

Judge = Callable[[int, str], "tuple[str, str] | None"]


def claim_check(prompt: str, judge: Judge | None = None) -> ClaimCheck:
    """A claim check that answers every numbered item of ``prompt``, as the
    contract asks: ``judge(n, sentence)`` gives an item's (verdict, reason),
    None (and no judge) for supported."""
    verdicts = []
    for number, sentence in _ITEM.findall(prompt):
        found = judge(int(number), sentence) if judge else None
        verdict, reason = found or ("supported", "")
        verdicts.append(
            SentenceVerdict(n=int(number), verdict=verdict, reason=reason)  # type: ignore[arg-type]
        )
    return ClaimCheck(verdicts=verdicts)
