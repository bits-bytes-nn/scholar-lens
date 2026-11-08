"""What the next section's writer knows of the post so far (docs/engine.md,
4.8): an "already taught" list code builds from every earlier section (its
heading, the claims of the evidence it cites, and for a section whose code
a reader runs, what that code defines), and the full text of the last few
sections. Nothing is cut silently: the list covers every earlier section."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from scholar_lens.domain.evidence import EvidenceLedger
from scholar_lens.domain.evidence_draft import (
    cited_ids,
    prose_sentences,
    section_heading,
)
from scholar_lens.domain.program import code_definitions


@dataclass(frozen=True)
class WrittenSection:
    """An earlier section as the list reads it: its outline title, its
    Markdown, and whether a reader runs its code."""

    title: str
    markdown: str
    runnable: bool = False


def taught_list(
    sections: Sequence[WrittenSection],
    ledger: EvidenceLedger,
    *,
    claim_chars: int,
) -> str:
    """One entry per earlier section, in order: its heading, the claim of
    each piece of evidence its prose cites (by id, once per section), and,
    for a runnable section, each definition its Python code makes (read
    with ``ast``).

    Args:
        sections: The sections written so far, in order.
        ledger: The run's evidence.
        claim_chars: The characters of a claim shown.

    Returns:
        The list as lines ("" before the first section).
    """
    lines: list[str] = []
    for s in sections:
        lines.append(f"- {section_heading(s.markdown) or s.title}")
        ids = dict.fromkeys(
            eid
            for sentence in prose_sentences(s.markdown)
            for eid in cited_ids(sentence)
        )
        for eid in ids:
            if (e := ledger.get(eid)) is not None:
                lines.append(f"  - [{e.id}] {e.claim[:claim_chars]}")
        if s.runnable:
            lines += [f"  - code: {d}" for d in code_definitions([s.markdown])]
    return "\n".join(lines)


def previous_sections(sections: Sequence[WrittenSection], count: int) -> str:
    """The full Markdown of the last ``count`` sections, in order ("" for
    none)."""
    if count <= 0:
        return ""
    return "\n\n".join(
        s.markdown.strip() for s in sections[-count:] if s.markdown.strip()
    )
