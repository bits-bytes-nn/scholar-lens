"""Source documents and the evidence ledger (docs/redesign.md, 3.3).

A reading agent records what a post may say about a paper as evidence: a
claim with the verbatim passage that supports it and where it is. Code checks
every record against the document before it enters the ledger, so a post can
only state what its sources show, and the outline can give each piece of
evidence to one section.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .survey.post import equation_in_source
from .validation import unsourced_numbers

DocumentKind = Literal["full_text", "pdf_text", "abstract"]
EvidenceKind = Literal[
    "finding",  # a result, observation or conclusion the paper reports
    "number",  # a measured value, with its model, setting and metric
    "method",  # how the approach works
    "definition",  # a term or quantity the paper defines
    "limitation",  # a limitation the paper states or its results show
    "relation",  # how the paper builds on, compares with or replaces another
    "equation",  # an equation, copied as LaTeX from the paper
    "figure",  # a figure of the paper, by its id
]

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
_SPACE = re.compile(r"\s+")
# Quotes shorter than this rarely pin a claim; longer ones are excerpts.
MIN_QUOTE_CHARS = 12
MAX_QUOTE_CHARS = 800
# Short noun-phrase cells for the code-built tables.
MAX_CELL_CHARS = 90
# The paper's own numbering means nothing in a post ("Fig. 4", "Table 2").
_NUMBERED_REFERENCE = re.compile(
    r"\b(?:fig(?:ure)?s?|tab(?:le)?s?|eq(?:uation)?s?|sec(?:tion)?s?|appendix|§)\.?\s*[A-Z]?\d",
    re.IGNORECASE,
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceSection(_Strict):
    anchor: str
    heading: str
    text: str


class SourceFigure(_Strict):
    id: str
    url: str
    caption: str


class SourceDocument(_Strict):
    """A paper as the reading agent sees it: its sections in order and its
    figures; ``kind`` says how much of the paper this is."""

    paper_id: str
    title: str
    kind: DocumentKind
    sections: list[SourceSection]
    figures: list[SourceFigure] = Field(default_factory=list)

    @classmethod
    def from_markdown(
        cls,
        paper_id: str,
        title: str,
        kind: DocumentKind,
        markdown: str,
        figures: Iterable[tuple[str, str]] = (),
    ) -> SourceDocument:
        """Split Markdown at its headings (``S1``, ``S2``, … in order; text
        before the first heading is ``S0``). ``figures`` are (url, caption)
        pairs, numbered ``F1``, ``F2``, …"""
        sections: list[SourceSection] = []
        matches = list(_HEADING.finditer(markdown))
        if not matches or matches[0].start() > 0:
            head = markdown[: matches[0].start() if matches else len(markdown)]
            if head.strip():
                sections.append(
                    SourceSection(anchor="S0", heading="", text=head.strip())
                )
        for n, m in enumerate(matches, 1):
            end = matches[n].start() if n < len(matches) else len(markdown)
            body = markdown[m.end() : end].strip()
            sections.append(
                SourceSection(anchor=f"S{n}", heading=m.group(2).strip(), text=body)
            )
        return cls(
            paper_id=paper_id,
            title=title,
            kind=kind,
            sections=sections,
            figures=[
                SourceFigure(id=f"F{i}", url=url, caption=caption)
                for i, (url, caption) in enumerate(figures, 1)
            ],
        )

    @property
    def text(self) -> str:
        return "\n\n".join(
            (f"## {s.heading}\n\n" if s.heading else "") + s.text for s in self.sections
        )

    def section(self, anchor: str) -> SourceSection | None:
        return next((s for s in self.sections if s.anchor == anchor.strip()), None)

    def figure(self, figure_id: str) -> SourceFigure | None:
        return next((f for f in self.figures if f.id == figure_id.strip()), None)

    def outline(self) -> str:
        """One line per section (anchor, heading, length) and per figure."""
        lines = [
            f"{s.anchor} | {s.heading or '(untitled)'} | {len(s.text):,} chars"
            for s in self.sections
        ]
        lines += [f"{f.id} | figure | {f.caption[:160]}" for f in self.figures]
        return "\n".join(lines)

    def search(self, query: str, limit: int = 8) -> list[tuple[str, str]]:
        """(anchor, snippet) of the passages that contain every word of
        ``query`` (case-insensitive), best first by how many times."""
        words = [w for w in _SPACE.split(query.casefold()) if w]
        found: list[tuple[int, str, str]] = []
        for s in self.sections:
            for paragraph in s.text.split("\n\n"):
                folded = paragraph.casefold()
                if words and all(w in folded for w in words):
                    hits = sum(folded.count(w) for w in words)
                    found.append((hits, s.anchor, paragraph.strip()[:600]))
        found.sort(key=lambda item: -item[0])
        return [(anchor, snippet) for _, anchor, snippet in found[:limit]]


class Evidence(_Strict):
    """One thing a post may say about a paper, and where the paper says it."""

    id: str
    paper_id: str
    kind: EvidenceKind
    # What the post may state, in the post's language.
    claim: str
    # The paper's words, copied verbatim from the section at ``anchor``.
    quote: str
    anchor: str
    # For numbers: model or variant, benchmark, setting and metric.
    scope: str = ""
    # A noun phrase for a table cell (numbers and limitations).
    cell: str = ""
    figure_id: str | None = None
    figure_url: str | None = None
    caption: str | None = None
    latex: str | None = None


def _squash(text: str) -> str:
    return _SPACE.sub(" ", text).strip().casefold()


def evidence_problems(
    document: SourceDocument,
    *,
    kind: str,
    claim: str,
    quote: str,
    anchor: str,
    scope: str = "",
    cell: str = "",
    figure_id: str | None = None,
    latex: str | None = None,
) -> list[str]:
    """Why a record cannot enter the ledger: its quote is not the paper's
    words at ``anchor``, it states numbers the quote does not, a number has
    no scope, an equation is not the paper's LaTeX, a figure is unknown."""
    problems: list[str] = []
    section = document.section(anchor)
    if section is None:
        return [f"unknown anchor {anchor!r}: use an anchor from the outline"]
    if not claim.strip():
        problems.append("the claim is empty")
    if kind == "figure":
        if figure_id is None or document.figure(figure_id) is None:
            problems.append(
                f"unknown figure {figure_id!r}: use an F-id from the outline"
            )
        return problems
    if not MIN_QUOTE_CHARS <= len(quote.strip()) <= MAX_QUOTE_CHARS:
        problems.append(
            f"the quote must be {MIN_QUOTE_CHARS}-{MAX_QUOTE_CHARS} characters of "
            "the paper's text"
        )
    elif _squash(quote) not in _squash(section.text):
        problems.append(
            f"the quote is not verbatim in {anchor}: copy the paper's words exactly"
        )
    if missing := unsourced_numbers(f"{claim} {cell}", f"{quote} {scope}"):
        problems.append(
            "the claim or cell states numbers the quote does not: " + ", ".join(missing)
        )
    if kind == "number" and not scope.strip():
        problems.append(
            "a number needs its scope: model or variant, benchmark, setting, metric"
        )
    if kind == "equation" and not (latex and equation_in_source(latex, section.text)):
        problems.append("the equation is not the paper's LaTeX: copy it exactly")
    if len(cell) > MAX_CELL_CHARS:
        problems.append(f"the cell is longer than {MAX_CELL_CHARS} characters")
    if "?" in f"{claim} {scope} {cell}":
        problems.append("a guess ('?') is not evidence: state only what the quote says")
    if _NUMBERED_REFERENCE.search(f"{claim} {scope} {cell}"):
        problems.append(
            "claim, scope and cell must not name the paper's figure, table, "
            "equation or section numbers; say what it is"
        )
    return problems


class EvidenceLedger(BaseModel):
    """All evidence of a survey, numbered ``E1``, ``E2``, … in the order it
    was recorded (paper by paper)."""

    items: list[Evidence] = Field(default_factory=list)

    def add(self, document: SourceDocument, **fields: object) -> Evidence:
        """Record evidence the caller has already checked
        (:func:`evidence_problems`); a figure gets its URL and caption."""
        figure = None
        if fields.get("kind") == "figure" and fields.get("figure_id"):
            figure = document.figure(str(fields["figure_id"]))
        evidence = Evidence(
            id=f"E{len(self.items) + 1}",
            paper_id=document.paper_id,
            figure_url=figure.url if figure else None,
            caption=figure.caption if figure else None,
            **fields,  # type: ignore[arg-type]
        )
        self.items.append(evidence)
        return evidence

    def extend(self, items: Iterable[Evidence]) -> None:
        """Append another ledger's evidence (a cached paper), renumbered."""
        for item in items:
            self.items.append(item.model_copy(update={"id": f"E{len(self.items) + 1}"}))

    def get(self, evidence_id: str) -> Evidence | None:
        return next((e for e in self.items if e.id == evidence_id), None)

    def of_paper(self, paper_id: str) -> list[Evidence]:
        return [e for e in self.items if e.paper_id == paper_id]

    def __iter__(self) -> Iterator[Evidence]:  # type: ignore[override]
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)
