"""Source documents and the evidence ledger (docs/redesign.md, 3.3).

A reading agent records what a post may say about a source (a paper, a
documentation page, a design post) as evidence: a claim with the verbatim
passage that supports it and where it is. Code checks every record against
the document before it enters the ledger, so a post can only state what its
sources show, and the outline can give each piece of evidence to one section.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .markdown.math import fence_roles, open_fence_line
from .survey.post import equation_in_source
from .validation import unsourced_numbers

DocumentKind = Literal["full_text", "pdf_text", "abstract"]
# Who decides what a guide may state (``web_research.source_label``): official
# pages own commands, flags and defaults; the rest explain. "" for papers.
Authority = Literal["", "official", "reference", "third_party", "older_docs"]
EvidenceKind = Literal[
    "finding",  # a result, observation or conclusion the paper reports
    "number",  # a measured value, with its model, setting and metric
    "method",  # how the approach works
    "definition",  # a term or quantity the paper defines
    "limitation",  # a limitation the paper states or its results show
    "relation",  # how the paper builds on, compares with or replaces another
    "equation",  # an equation, copied as LaTeX from the paper
    "figure",  # a figure of the paper, by its id
    "code",  # a code block or command, copied from the source
    "setting",  # a flag, parameter, config key, env variable or default
]

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.M)
_SPACE = re.compile(r"\s+")
# Quotes shorter than this rarely pin a claim; longer ones are excerpts.
MIN_QUOTE_CHARS = 12
MAX_QUOTE_CHARS = 800
# Short noun-phrase cells for the code-built tables.
MAX_CELL_CHARS = 90
MAX_CODE_CHARS = 6_000
_CODE_SPAN = re.compile(r"`([^`\n]+)`")
# The paper's own numbering means nothing in a post ("Fig. 4", "Table 2").
_NUMBERED_REFERENCE = re.compile(
    r"\b(?:fig(?:ure)?s?|tab(?:le)?s?|eq(?:uation)?s?|sec(?:tion)?s?|appendix|§)\.?\s*[A-Z]?\d",
    re.IGNORECASE,
)


def _headings(markdown: str) -> list[re.Match[str]]:
    """The Markdown headings outside fenced code (a ``# comment`` line in a
    shell or Python block is code, not a heading)."""
    lines = markdown.split("\n")
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    roles = fence_roles(lines)
    # A fence that never closes (a stray "~~~~" in PDF text, a listing cut
    # by a page break) is not code to the end: its lines stay prose here.
    if (opened := open_fence_line(lines)) is not None:
        roles[opened:] = ["prose"] * (len(roles) - opened)
    code = {starts[i] for i, role in enumerate(roles) if role != "prose"}
    return [m for m in _HEADING.finditer(markdown) if m.start() not in code]


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
    # Where the figure sits (its section) and the text just before it: what
    # the figure shows when its caption is only a file name.
    anchor: str = ""
    context: str = ""


class SourceDocument(_Strict):
    """A paper as the reading agent sees it: its sections in order and its
    figures; ``kind`` says how much of the paper this is."""

    paper_id: str
    title: str
    kind: DocumentKind
    sections: list[SourceSection]
    figures: list[SourceFigure] = Field(default_factory=list)
    url: str = ""
    authority: Authority = ""

    @classmethod
    def from_markdown(
        cls,
        paper_id: str,
        title: str,
        kind: DocumentKind,
        markdown: str,
        figures: Iterable[tuple[str, str]] = (),
        *,
        url: str = "",
        authority: Authority = "",
    ) -> SourceDocument:
        """Split Markdown at its headings (``S1``, ``S2``, … in order; text
        before the first heading is ``S0``). ``figures`` are (url, caption)
        pairs, numbered ``F1``, ``F2``, …"""
        sections: list[SourceSection] = []
        matches = _headings(markdown)
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
        placed = []
        for i, (image, caption) in enumerate(figures, 1):
            anchor, context = _figure_place(sections, image)
            placed.append(
                SourceFigure(
                    id=f"F{i}",
                    url=image,
                    caption="" if _NOT_A_CAPTION.match(caption.strip()) else caption,
                    anchor=anchor,
                    context=context,
                )
            )
        return cls(
            paper_id=paper_id,
            title=title,
            kind=kind,
            sections=sections,
            figures=placed,
            url=url,
            authority=authority,
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
        for f in self.figures:
            line = f"{f.id} | figure" + (f" in {f.anchor}" if f.anchor else "")
            if f.caption:
                line += f" | caption: {f.caption[:160]}"
            if f.context:
                line += f" | text before it: {f.context}"
            lines.append(line)
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
    code: str | None = None


# Alt text that says nothing about the figure: a file path or name, a
# placeholder word.
_NOT_A_CAPTION = re.compile(
    r"^(?:|[\w./-]+\.(?:png|jpe?g|gif|svg|webp)|pic|image|img|figure|diagram|screenshot)$",
    re.IGNORECASE,
)
_IMAGE_MARKER = re.compile(r"\[Image:[^\]]*\](?:\s*\([^)]*\))?|!\[[^\]]*\]\([^)]*\)")


def _figure_place(sections: list[SourceSection], url: str) -> tuple[str, str]:
    """The anchor of the section that shows ``url`` and up to 240 characters
    of its text before the image (other image markers removed)."""
    for s in sections:
        if (at := s.text.find(url)) >= 0:
            before = _IMAGE_MARKER.sub(" ", s.text[:at].rsplit("[Image:", 1)[0])
            words = " ".join(before.split())
            return s.anchor, (s.heading + ": " if s.heading else "") + words[-240:]
    return "", ""


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
    code: str | None = None,
) -> list[str]:
    """Why a record cannot enter the ledger: its quote is not the source's
    words at ``anchor``, it states numbers or names identifiers the quote
    does not, a number or setting has no scope, an equation is not the
    source's LaTeX, code is not the source's code, a figure is unknown."""
    problems: list[str] = []
    if kind == "figure" and figure_id and (figure := document.figure(figure_id)):
        # A figure is placed by its id; its section is known.
        if figure.anchor or document.section(anchor) is None:
            anchor = figure.anchor or (
                document.sections[0].anchor if document.sections else anchor
            )
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
    if kind == "code":
        if not code or not code.strip():
            problems.append("a code record needs `code` copied from the source")
        elif len(code) > MAX_CODE_CHARS:
            problems.append(f"the code is longer than {MAX_CODE_CHARS} characters")
        elif _squash(code) not in _squash(section.text):
            problems.append(f"the code is not verbatim in {anchor}: copy it exactly")
        if quote.strip() and _squash(quote) not in _squash(section.text):
            problems.append(
                f"the quote is not verbatim in {anchor}: copy the source's words "
                "exactly, or leave it empty"
            )
    elif not MIN_QUOTE_CHARS <= len(quote.strip()) <= MAX_QUOTE_CHARS:
        problems.append(
            f"the quote must be {MIN_QUOTE_CHARS}-{MAX_QUOTE_CHARS} characters of "
            "the paper's text"
        )
    elif _squash(quote) not in _squash(section.text):
        problems.append(
            f"the quote is not verbatim in {anchor}: copy the paper's words exactly"
        )
    support = f"{quote} {scope} {code or ''} {latex or ''}"
    if missing := unsourced_numbers(f"{claim} {cell}", support):
        problems.append(
            "the claim or cell states numbers the quote does not: " + ", ".join(missing)
        )
    if names := [
        n
        for n in _CODE_SPAN.findall(f"{claim} {cell}")
        if _squash(n) not in _squash(support)
    ]:
        problems.append(
            "the claim names identifiers the quote and code do not: " + ", ".join(names)
        )
    if kind == "number" and not scope.strip():
        problems.append(
            "a number needs its scope: model or variant, benchmark, setting, metric"
        )
    if kind == "setting" and not scope.strip():
        problems.append(
            "a setting needs its scope: the API, CLI, config file or version it "
            "belongs to"
        )
    if kind == "equation" and not (latex and equation_in_source(latex, section.text)):
        problems.append("the equation is not the paper's LaTeX: copy it exactly")
    if len(cell) > MAX_CELL_CHARS:
        problems.append(f"the cell is longer than {MAX_CELL_CHARS} characters")
    # A sentence ends in a period or a formal verb ending; a noun phrase
    # ("AWS 람다", "61% 승률") does not.
    if cell and re.search(r"(?:\.|니다)\s*$", cell.strip()):
        problems.append("the cell is a sentence: write it as a noun phrase")
    if "?" in _CODE_OR_URL.sub("", f"{claim} {scope} {cell}"):
        problems.append("a guess ('?') is not evidence: state only what the quote says")
    if _NUMBERED_REFERENCE.search(f"{claim} {scope} {cell}"):
        problems.append(
            "claim, scope and cell must not name the paper's figure, table, "
            "equation or section numbers; say what it is"
        )
    return problems


_CODE_OR_URL = re.compile(r"`[^`]*`|https?://\S+")
_FACT_TOKEN = re.compile(r"`[^`]+`|\d+(?:[.,]\d+)*")


def _facts(text: str) -> set[str]:
    """The numbers and code spans a text states: two texts that differ in
    them state different facts however alike their words are."""
    return set(_FACT_TOKEN.findall(text))


def _same_fact(a: str, b: str, threshold: float) -> bool:
    if _facts(a) != _facts(b):
        return False
    matcher = SequenceMatcher(None, _squash(a), _squash(b))
    return matcher.quick_ratio() >= threshold and matcher.ratio() >= threshold


def near_duplicate(claim: str, others: Iterable[str], threshold: float = 0.85) -> bool:
    """Whether ``claim`` says nearly what one of ``others`` says: the same
    numbers and code spans, and similar characters (which Korean particles
    do not break as they break word overlap). Two records of one fact make
    the post repeat it; "DPO 61%" and "PPO 57%" are two facts."""
    return any(_same_fact(claim, other, threshold) for other in others)


def duplicate_evidence(
    items: Iterable[Evidence],
    rank: dict[str, int] | None = None,
    *,
    within_source: bool = False,
) -> dict[str, str]:
    """Evidence that states what an earlier item of the same kind states
    (two pages documenting one flag): duplicate id -> the id kept. With
    ``rank`` (source id -> lower is more authoritative) the most
    authoritative source keeps the fact. With ``within_source`` two items of
    one source count too (a paper states its main result in the abstract,
    the introduction and the experiments; parallel readers record each). A
    figure recorded twice (the same image) is a duplicate; otherwise figures
    and code are never duplicates of each other by their claims."""
    order = sorted(
        enumerate(items),
        key=lambda pair: ((rank or {}).get(pair[1].paper_id, 0), pair[0]),
    )
    kept: dict[str, list[Evidence]] = {}
    duplicates: dict[str, str] = {}
    shown: dict[str, str] = {}
    for _, item in order:
        if item.kind == "figure" and item.figure_url:
            if item.figure_url in shown:
                duplicates[item.id] = shown[item.figure_url]
            else:
                shown[item.figure_url] = item.id
            continue
        if item.kind in ("figure", "code"):
            continue
        for other in kept.get(item.kind, []):
            if other.paper_id == item.paper_id and not within_source:
                continue
            if other.paper_id == item.paper_id and not (
                _facts(item.claim) and item.scope.strip() == other.scope.strip()
            ):
                # Within one source only a repeated number or code span in the
                # same scope is a repeat ("긴/짧은 문맥에서" differ by a word).
                continue
            if _same_fact(item.claim, other.claim, 0.85) and _facts(
                item.cell
            ) == _facts(other.cell):
                duplicates[item.id] = other.id
                break
        else:
            kept.setdefault(item.kind, []).append(item)
    return duplicates


def canonical_ids(ids: Iterable[str], duplicates: Mapping[str, str]) -> list[str]:
    """``ids`` with each duplicate replaced by the id that keeps its fact,
    each once, in order."""
    return list(dict.fromkeys(duplicates.get(i, i) for i in ids))


class EvidenceLedger(BaseModel):
    """All evidence of a survey, numbered ``E1``, ``E2``, … in the order it
    was recorded (paper by paper)."""

    items: list[Evidence] = Field(default_factory=list)

    def add(self, document: SourceDocument, **fields: object) -> Evidence:
        """Record evidence the caller has already checked
        (:func:`evidence_problems`); a figure gets its URL, caption and the
        section it sits in."""
        figure = None
        if fields.get("kind") == "figure" and fields.get("figure_id"):
            figure = document.figure(str(fields["figure_id"]))
            if figure is not None and figure.anchor:
                fields["anchor"] = figure.anchor
            if figure is not None and not str(fields.get("quote") or "").strip():
                # What the figure shows, as far as the text says: the check
                # of its claim reads it.
                fields["quote"] = figure.context
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
