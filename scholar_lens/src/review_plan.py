"""Deterministic parts of the paper-review workflow.

The planner LLM proposes a :class:`ReviewPlan`; everything here checks and
repairs that plan in code so the writer always gets a consistent brief:
sentence anchors are clamped to the paper, uncovered paper text is attached to
a section (nothing is silently dropped), each figure goes to at most one
section, each concept and prerequisite has one owning section, and target
lengths follow from the difficulty of the items a section must cover. Section
sources are sliced from the original text (structure intact; tables, code and
display math never cut). The editor's targeted edits are applied here,
exact-match only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .logger import logger

Difficulty = Literal["easy", "medium", "hard"]
Importance = Literal["core", "supporting", "context"]

# Words of prose per must-cover item, by its difficulty for a general DS/MLE
# reader; a section's target is the sum, scaled by how central the section is.
# Hard, core material gets the most room; a section never drops below a floor.
_ITEM_WORDS: dict[str, int] = {"easy": 60, "medium": 200, "hard": 450}
_IMPORTANCE_FACTOR: dict[str, float] = {"context": 0.8, "supporting": 1.0, "core": 1.3}
MIN_SECTION_WORDS = 250
# Sentences of context added on each side of a planned range, so evidence just
# outside an LLM-estimated anchor still reaches the writer.
RANGE_PADDING = 3
# A long paper planned as only a couple of sections, or one section carrying
# most of the coverage, is a broken plan worth one repair round.
LONG_PAPER_SENTENCES = 200
MIN_SECTIONS_FOR_LONG_PAPER = 4
MAX_SECTION_COVERAGE_SHARE = 0.4
MIN_ITEMS_FOR_SHARE_CHECK = 10
# Paper text left out of every section is attached to the nearest section once
# it is at least this many sentences long; shorter gaps are stray captions etc.
MIN_UNCOVERED_SENTENCES = 3

_UNIT_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
# The whole heading must be a closing title ("8 Conclusion and Future Work",
# "Limitations", "Appendix A Limitations"); a heading that merely contains the
# word ("2.2 Limitations of Existing Methods", a paper title) is not one.
_CLOSING_HEADING = re.compile(
    r"^(?:(?:appendix\s+)?(?:[ivx]+|[a-z]|\d+)(?:\.\d+)*\.?\s+)?"
    r"(?:(?:discussion|summary)(?:\s*,|\s+and|\s*&)\s*)?"
    r"(?:conclusions?|concluding remarks|limitations?|future (?:work|directions|research))"
    r"(?:(?:\s*,|\s+and|\s*&)+\s*(?:conclusions?|limitations?|future (?:work|directions|research)"
    r"|discussion|outlook|broader impacts?))*\.?$"
    r"|^결론|^향후 연구",
    re.IGNORECASE,
)
_IMAGE_MARKER = re.compile(r'\[Image:[^\n]*?src="([^"]+)"[^\n]*?"\]')
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_HANGUL = re.compile(r"[가-힣]")
_LATIN_WORD = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def coerce_ranges(value: Any) -> list[tuple[int, int]]:
    """Accept [3, 9], [[3, 9]], ["3-9"] or [{"start": 3, "end": 9}]."""
    if not isinstance(value, list):
        return []
    if len(value) == 2 and all(isinstance(v, int) for v in value):
        value = [value]
    ranges = []
    for item in value:
        if isinstance(item, dict):
            item = [item.get("start"), item.get("end")]
        elif isinstance(item, str) and "-" in item:
            item = item.split("-", 1)
        try:
            start, end = (int(str(v).strip()) for v in item)
        except (TypeError, ValueError):
            continue
        ranges.append((start, end))
    return ranges


class PlanError(ValueError):
    """The planner output could not be parsed into a usable plan."""


class Prerequisite(BaseModel):
    model_config = ConfigDict(extra="ignore")

    concept: str = Field(min_length=1)
    why: str = ""
    # "R<n>" from the reference list shown to the planner, if one would help.
    reference: str | None = None


class CoverItem(BaseModel):
    """One thing a section must cover, with its own difficulty."""

    model_config = ConfigDict(extra="ignore")

    item: str = Field(min_length=1)
    difficulty: Difficulty = "medium"

    def __str__(self) -> str:
        return self.item


class PlannedSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1)
    purpose: str = ""
    sentence_ranges: list[tuple[int, int]] = Field(default_factory=list)
    must_cover: list[CoverItem] = Field(min_length=1)
    difficulty: Difficulty = "medium"
    difficulty_reason: str = ""
    importance: Importance = "supporting"
    prerequisites: list[Prerequisite] = Field(default_factory=list)
    owned_concepts: list[str] = Field(default_factory=list)
    # "F<n>" ids from the figure list shown to the planner, resolved to srcs.
    figures: list[str] = Field(default_factory=list)
    needs_code: bool = False

    @field_validator("sentence_ranges", mode="before")
    @classmethod
    def _coerce_ranges(cls, value: Any) -> Any:
        return coerce_ranges(value)

    @field_validator("must_cover", mode="before")
    @classmethod
    def _coerce_items(cls, value: Any) -> Any:
        # Accept plain strings (difficulty inherited from the section later).
        if isinstance(value, list):
            return [{"item": v} if isinstance(v, str) else v for v in value]
        return value

    @property
    def items(self) -> list[str]:
        return [c.item for c in self.must_cover]

    @property
    def target_words(self) -> int:
        return target_words(self.must_cover, self.importance)


class ReviewPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    paper_type: str = ""
    narrative: str = ""
    glossary: dict[str, str] = Field(default_factory=dict)
    sections: list[PlannedSection] = Field(min_length=1)
    excluded_ranges: list[tuple[int, int]] = Field(default_factory=list)
    # The paper's own limitations and future work, for the closing assessment
    # (the body never gets a limitations/conclusion section of its own).
    closing_points: list[str] = Field(default_factory=list)

    @field_validator("excluded_ranges", mode="before")
    @classmethod
    def _coerce_excluded(cls, value: Any) -> Any:
        return coerce_ranges(value)

    @field_validator("glossary", mode="before")
    @classmethod
    def _coerce_glossary(cls, value: Any) -> Any:
        # Accept a mapping or a list of {"term": ..., "rendering": ...}.
        if isinstance(value, list):
            return {
                str(item.get("term")): str(item.get("rendering") or item.get("term"))
                for item in value
                if isinstance(item, dict) and item.get("term")
            }
        return value or {}


def target_words(items: Sequence[CoverItem], importance: str) -> int:
    """Section length from what it must carry: per-item budgets by
    difficulty, scaled by importance, never below the floor."""
    words = sum(_ITEM_WORDS[i.difficulty] for i in items)
    return max(MIN_SECTION_WORDS, round(words * _IMPORTANCE_FACTOR[importance]))


def approx_words(text: str) -> int:
    """Length in English-word equivalents: Latin words plus Hangul syllables
    / 2.5 (a Korean sentence carrying the same content), code fences excluded."""
    prose = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    hangul = len(_HANGUL.findall(prose))
    latin = len(_LATIN_WORD.findall(_HANGUL.sub(" ", prose)))
    return latin + round(hangul / 2.5)


_FENCE_LINE = re.compile(r"^\s*(```|~~~)")
_HEADING_LINE = re.compile(r"^#{1,6}\s")


def _line_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for line in text.splitlines(keepends=True):
        spans.append((pos, pos + len(line.rstrip("\n"))))
        pos += len(line)
    return spans


def split_units(
    text: str, sentence_spans: Callable[[str], list[tuple[int, int]]]
) -> list[tuple[int, int]]:
    """Character spans of the anchorable units of ``text``: one per heading
    line, table, fenced code block or display-math block (never split), and
    one per sentence of ordinary paragraphs (via ``sentence_spans``)."""
    lines = _line_spans(text)
    units: list[tuple[int, int]] = []
    paragraph: list[tuple[int, int]] = []

    def flush() -> None:
        if paragraph:
            start, end = paragraph[0][0], paragraph[-1][1]
            for a, b in sentence_spans(text[start:end]):
                if text[start + a : start + b].strip():
                    units.append((start + a, start + b))
            paragraph.clear()

    i = 0
    while i < len(lines):
        start, end = lines[i]
        line = text[start:end]
        stripped = line.strip()
        block_end: int | None = None
        if _FENCE_LINE.match(line):
            fence = _FENCE_LINE.match(line).group(1)  # type: ignore[union-attr]
            j = i + 1
            while j < len(lines) and not text[
                lines[j][0] : lines[j][1]
            ].strip().startswith(fence):
                j += 1
            block_end = min(j, len(lines) - 1)
        elif stripped == "$$":
            j = i + 1
            while j < len(lines) and text[lines[j][0] : lines[j][1]].strip() != "$$":
                j += 1
            block_end = min(j, len(lines) - 1)
        elif stripped.startswith("|"):
            j = i
            while j + 1 < len(lines) and text[
                lines[j + 1][0] : lines[j + 1][1]
            ].strip().startswith("|"):
                j += 1
            block_end = j
        elif (
            _HEADING_LINE.match(stripped)
            or (
                stripped.startswith("$$")
                and stripped.endswith("$$")
                and len(stripped) > 4
            )
            # A line carrying a figure marker stays whole so no range boundary
            # can cut the marker apart.
            or "[Image:" in stripped
        ):
            block_end = i
        if block_end is not None:
            flush()
            units.append((start, lines[block_end][1]))
            i = block_end + 1
            continue
        if not stripped:
            flush()
        else:
            paragraph.append((start, end))
        i += 1
    flush()
    return units


@dataclass
class SourceText:
    """The paper text plus its anchorable units (character spans)."""

    text: str
    spans: list[tuple[int, int]]

    @classmethod
    def from_text(
        cls, text: str, sentence_spans: Callable[[str], list[tuple[int, int]]]
    ) -> SourceText:
        return cls(text, split_units(text, sentence_spans))

    @classmethod
    def from_sentences(cls, sentences: Sequence[str]) -> SourceText:
        """Units given directly (one per sentence), joined by spaces."""
        text, spans, pos = "", [], 0
        for sentence in sentences:
            if text:
                text += " "
                pos += 1
            spans.append((pos, pos + len(sentence)))
            text += sentence
            pos += len(sentence)
        return cls(text, spans)

    def __len__(self) -> int:
        return len(self.spans)

    @property
    def sentences(self) -> list[str]:
        return [self.text[a:b] for a, b in self.spans]

    def slice(self, first: int, last: int) -> str:
        """Original text from unit ``first`` through ``last`` (inclusive)."""
        return self.text[self.spans[first][0] : self.spans[last][1]]

    def closing_ranges(self) -> list[tuple[int, int]]:
        """Unit ranges of the paper's own conclusion / future-work sections.

        Their content belongs to the review's single closing section, never to
        a body section (otherwise limitations and future work appear twice).
        """
        ranges: list[tuple[int, int]] = []
        units = self.sentences
        i = 0
        while i < len(units):
            match = _UNIT_HEADING.match(units[i].strip())
            if match and _CLOSING_HEADING.match(match.group(2).strip()):
                level = len(match.group(1))
                j = i + 1
                while j < len(units):
                    nxt = _UNIT_HEADING.match(units[j].strip())
                    if nxt and len(nxt.group(1)) <= level:
                        break
                    j += 1
                ranges.append((i, j - 1))
                i = j
                continue
            i += 1
        return ranges


def number_sentences(source: SourceText | Sequence[str]) -> str:
    """One ``[i] unit`` line per anchorable unit (whitespace collapsed)."""
    units = source.sentences if isinstance(source, SourceText) else list(source)
    return "\n".join(f"[{i}] {' '.join(u.split())}" for i, u in enumerate(units))


def figure_catalog(text: str) -> list[tuple[str, str]]:
    """``(figure_id, marker)`` for each distinct figure marker, in order of
    appearance (``F1``, ``F2`` …)."""
    seen: dict[str, str] = {}
    for match in _IMAGE_MARKER.finditer(text):
        seen.setdefault(match.group(1), match.group(0))
    return [(f"F{i}", marker) for i, marker in enumerate(seen.values(), start=1)]


def marker_src(marker: str) -> str:
    match = _IMAGE_MARKER.search(marker)
    return match.group(1) if match else ""


def parse_plan(raw: str) -> ReviewPlan:
    """Parse the planner's ``<plan>{json}</plan>`` output."""
    match = re.search(r"<plan>(.*?)</plan>", raw, re.DOTALL)
    body = (match.group(1) if match else raw).strip()
    body = _JSON_FENCE.sub("", body).strip()
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        raise PlanError("no JSON object in the planner output")
    try:
        return ReviewPlan.model_validate(json.loads(body[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as e:
        raise PlanError(str(e)) from e


def check_plan_shape(plan: ReviewPlan, *, sentence_count: int) -> None:
    """Reject plans that cannot yield a complete review (triggers the repair
    round): a long paper squeezed into a few sections, or one section holding
    most of the coverage."""
    if (
        sentence_count > LONG_PAPER_SENTENCES
        and len(plan.sections) < MIN_SECTIONS_FOR_LONG_PAPER
    ):
        raise PlanError(
            f"{len(plan.sections)} sections for a {sentence_count}-unit paper; "
            f"plan at least {MIN_SECTIONS_FOR_LONG_PAPER}."
        )
    total = sum(len(s.must_cover) for s in plan.sections)
    if total >= MIN_ITEMS_FOR_SHARE_CHECK and len(plan.sections) > 1:
        for section in plan.sections:
            if len(section.must_cover) > MAX_SECTION_COVERAGE_SHARE * total:
                raise PlanError(
                    f"section '{section.title}' holds {len(section.must_cover)} of "
                    f"{total} must_cover items; split it."
                )


def _clamp(ranges: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    clamped = []
    for start, end in ranges:
        start, end = sorted((start, end))
        start, end = max(0, start), min(n - 1, end)
        if start <= end:
            clamped.append((start, end))
    return sorted(set(clamped))


def _covered(ranges: list[tuple[int, int]]) -> set[int]:
    return {i for start, end in ranges for i in range(start, end + 1)}


def _uncovered_runs(covered: set[int], n: int) -> list[tuple[int, int]]:
    runs, start = [], None
    for i in range(n + 1):
        if i < n and i not in covered:
            start = i if start is None else start
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


def normalize_plan(
    plan: ReviewPlan,
    *,
    sentence_count: int,
    figure_ids: dict[str, str],
    max_sections: int,
    closing_ranges: list[tuple[int, int]] | None = None,
) -> ReviewPlan:
    """Return a plan the workflow can rely on (see the module docstring).

    ``figure_ids`` maps the planner's ``F<n>`` ids to figure srcs; sections end
    up holding srcs. Overflow sections beyond ``max_sections`` are merged into
    the last kept one.
    """
    n = max(sentence_count, 1)
    sections = [s.model_copy(deep=True) for s in plan.sections]
    if len(sections) > max_sections:
        logger.warning(
            "Plan has %d sections; merging the tail into section %d.",
            len(sections),
            max_sections,
        )
        keep, tail = sections[: max_sections - 1], sections[max_sections - 1 :]
        merged = tail[0].model_copy(deep=True)
        for extra in tail[1:]:
            merged.sentence_ranges += extra.sentence_ranges
            merged.must_cover += extra.must_cover
            merged.owned_concepts += extra.owned_concepts
            merged.figures += extra.figures
            merged.prerequisites += extra.prerequisites
            merged.needs_code = merged.needs_code or extra.needs_code
        sections = [*keep, merged]

    known_srcs = set(figure_ids.values())
    used_figures: set[str] = set()
    owners: set[str] = set()
    seen_items: set[str] = set()
    for section in sections:
        # Items without their own difficulty inherit the section's; exact
        # duplicates (across sections too) are dropped.
        items = []
        for cover in section.must_cover:
            key = " ".join(cover.item.lower().split())
            if key in seen_items:
                continue
            seen_items.add(key)
            if "difficulty" not in cover.model_fields_set:
                cover = cover.model_copy(update={"difficulty": section.difficulty})
            items.append(cover)
        section.must_cover = items or section.must_cover[:1]
        section.sentence_ranges = _clamp(section.sentence_ranges, n)
        srcs = []
        for fig in section.figures:
            fig = fig.strip()
            src = figure_ids.get(fig) or (fig if fig in known_srcs else None)
            if src and src not in used_figures:
                used_figures.add(src)
                srcs.append(src)
        section.figures = srcs
        owned = []
        for concept in section.owned_concepts:
            key = concept.strip().lower()
            if key and key not in owners:
                owners.add(key)
                owned.append(concept.strip())
        section.owned_concepts = owned
    _dedupe_prerequisites(sections)

    closing = _clamp(closing_ranges or [], n)
    if closing:
        # The paper's conclusion feeds the closing section only.
        closed = _covered(closing)
        for section in sections:
            kept = _subtract(section.sentence_ranges, closed)
            if kept:
                section.sentence_ranges = kept
    excluded = _clamp([*plan.excluded_ranges, *closing], n)
    covered = _covered([r for s in sections for r in s.sentence_ranges] + excluded)
    for run in _uncovered_runs(covered, n):
        if run[1] - run[0] + 1 < MIN_UNCOVERED_SENTENCES:
            continue
        owner = _section_before(sections, run[0])
        logger.info(
            "Sentences %d-%d were in no section; adding them to '%s'.",
            run[0],
            run[1],
            owner.title,
        )
        owner.sentence_ranges = sorted([*owner.sentence_ranges, run])
    return plan.model_copy(update={"sections": sections, "excluded_ranges": excluded})


def _dedupe_prerequisites(sections: list[PlannedSection]) -> None:
    """Each prerequisite is explained once, in the first section that needs it,
    which then owns it; later sections only refer back. A prerequisite an
    earlier section already owns is dropped; ownership claimed by a later
    section moves to the first section that needs it."""
    owner_of = {c.lower(): i for i, s in enumerate(sections) for c in s.owned_concepts}
    taken: set[str] = set()
    for index, section in enumerate(sections):
        kept = []
        for prereq in section.prerequisites:
            key = prereq.concept.strip().lower()
            owner = owner_of.get(key)
            if key in taken or (owner is not None and owner < index):
                continue
            taken.add(key)
            kept.append(prereq)
            if owner is None or owner > index:
                if owner is not None:
                    later = sections[owner]
                    later.owned_concepts = [
                        c for c in later.owned_concepts if c.lower() != key
                    ]
                section.owned_concepts.append(prereq.concept.strip())
                owner_of[key] = index
        section.prerequisites = kept


def _subtract(
    ranges: list[tuple[int, int]], removed: set[int]
) -> list[tuple[int, int]]:
    """``ranges`` minus the unit indices in ``removed`` (as sorted ranges)."""
    kept = sorted({i for a, b in ranges for i in range(a, b + 1)} - removed)
    out: list[tuple[int, int]] = []
    for i in kept:
        if out and i == out[-1][1] + 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def _section_before(sections: list[PlannedSection], sentence: int) -> PlannedSection:
    """The section whose anchors end closest before ``sentence`` (else the
    first section with anchors, else the first section)."""
    best, best_end = None, -1
    for section in sections:
        for _, end in section.sentence_ranges:
            if best_end < end < sentence:
                best, best_end = section, end
    if best is not None:
        return best
    return next((s for s in sections if s.sentence_ranges), sections[0])


def _padded(
    ranges: list[tuple[int, int]], n: int, blocked: set[int], pad: int
) -> list[tuple[int, int]]:
    """Grow each range by ``pad`` units on both sides, never into ``blocked``
    (excluded) units or past the paper."""
    grown = []
    for start, end in ranges:
        a = start
        while a > 0 and start - a < pad and (a - 1) not in blocked:
            a -= 1
        b = end
        while b < n - 1 and b - end < pad and (b + 1) not in blocked:
            b += 1
        grown.append((a, b))
    merged: list[tuple[int, int]] = []
    for a, b in sorted(grown):
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def section_source(
    section: PlannedSection,
    source: SourceText | Sequence[str],
    figure_markers: dict[str, str],
    *,
    excluded: list[tuple[int, int]] | None = None,
    pad: int = 0,
) -> str:
    """The paper text a section draws on, with only ITS figures left in.

    Text is sliced from the original (line breaks, tables and code intact).
    Ranges are padded by ``pad`` units, never into ``excluded`` units. Markers
    of figures assigned elsewhere are removed so a figure appears once; an
    assigned figure whose marker lies outside the anchors is appended.
    """
    if not isinstance(source, SourceText):
        source = SourceText.from_sentences(source)
    n = len(source)
    blocked = _covered(excluded or [])
    ranges = (
        _padded(section.sentence_ranges, n, blocked, pad)
        if pad
        else section.sentence_ranges
    )
    parts = [source.slice(a, b) for a, b in ranges if 0 <= a <= b < n]
    text = "\n\n[…]\n\n".join(parts)
    keep = set(section.figures)
    text = _IMAGE_MARKER.sub(lambda m: m.group(0) if m.group(1) in keep else "", text)
    for src in section.figures:
        if src not in text and src in figure_markers:
            text += "\n\n" + figure_markers[src]
    return text


def render_outline(plan: ReviewPlan) -> str:
    return "\n".join(
        f"{i}. {s.title} — {s.purpose or s.must_cover[0].item}"
        for i, s in enumerate(plan.sections, start=1)
    )


def render_brief(plan: ReviewPlan, index: int, figure_markers: dict[str, str]) -> str:
    section = plan.sections[index]
    figures = "\n".join(figure_markers.get(src, src) for src in section.figures)
    prerequisites = "\n".join(
        f"- {p.concept}: {p.why}".rstrip(": ") for p in section.prerequisites
    )
    return "\n".join(
        [
            f"Section {index + 1} of {len(plan.sections)}: {section.title}",
            f"Purpose: {section.purpose}",
            f"Difficulty for a general DS/MLE reader: {section.difficulty}"
            + (f" ({section.difficulty_reason})" if section.difficulty_reason else ""),
            f"Importance: {section.importance}",
            f"Target length: about {section.target_words} English words "
            f"(about {round(section.target_words * 2.5)} characters in Korean); "
            "spend them on the hard items; never drop a required item to "
            "meet it.",
            "Must cover (every item; its difficulty sets its depth):",
            *(f"- [{c.difficulty}] {c.item}" for c in section.must_cover),
            "Prerequisites to explain briefly where first needed:",
            prerequisites or "- none",
            "Concepts this section explains (others only refer to them):",
            *(f"- {c}" for c in section.owned_concepts or ["-"]),
            "Figures to place in this section (use only these):",
            figures or "- none",
        ]
    )


def concepts_owned_elsewhere(plan: ReviewPlan, index: int) -> str:
    lines = []
    for i, s in enumerate(plan.sections):
        if i != index and s.owned_concepts:
            where = "earlier" if i < index else "later"
            lines.append(f"- {where}, '{s.title}': {'; '.join(s.owned_concepts)}")
    return "\n".join(lines) or "- none"


def render_glossary(plan: ReviewPlan) -> str:
    return "\n".join(f"- {k} → {v}" for k, v in plan.glossary.items()) or "- none"


def render_coverage(plan: ReviewPlan) -> str:
    lines = []
    for i, s in enumerate(plan.sections, start=1):
        lines.append(f"{i}. {s.title}")
        lines.extend(f"   - {c.item}" for c in s.must_cover)
    return "\n".join(lines)


def ensure_section_heading(text: str, title: str) -> str:
    """A section starts with exactly its ``##`` heading."""
    body = text.strip()
    if body.startswith("## "):
        return body
    body = re.sub(r"^#{1,6}\s+[^\n]*\n+", "", body) if body.startswith("#") else body
    return f"## {title}\n\n{body}"


class Edit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    find: str
    replace: str = ""
    reason: str = ""


_EDIT_BLOCK = re.compile(
    r"<edit>\s*<find>(.*?)</find>\s*<replace>(.*?)</replace>"
    r"(?:\s*<reason>(.*?)</reason>)?\s*</edit>",
    re.DOTALL,
)


def _trim_newlines(value: str) -> str:
    # The tag body is copied verbatim except for the line breaks a model puts
    # right after an opening tag or before a closing one.
    return value.strip("\n").strip("\r")


def parse_edits(raw: str) -> list[Edit]:
    """Parse ``<edit><find>…</find><replace>…</replace></edit>`` blocks from the
    editor's raw output. Text is taken verbatim (no JSON or HTML unescaping),
    so LaTeX backslashes and ``&`` survive exactly."""
    return [
        Edit(
            find=_trim_newlines(m.group(1)),
            replace=_trim_newlines(m.group(2)),
            reason=(m.group(3) or "").strip(),
        )
        for m in _EDIT_BLOCK.finditer(raw or "")
    ]


def tag_body(raw: str, tag: str) -> str | None:
    """Raw body of the last ``<tag>…</tag>`` in ``raw`` (no unescaping)."""
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", raw or "", re.DOTALL)
    return matches[-1] if matches else None


def apply_edits(
    text: str, edits: list[Edit], *, max_edits: int
) -> tuple[str, int, list[str]]:
    """Apply edits whose ``find`` occurs exactly once in the current text.

    Returns ``(text, applied, skipped_reasons)``. Ambiguous or missing anchors
    are skipped rather than guessed, so an edit can never land in the wrong
    place; headings cannot be removed (a replacement must keep every ``##``
    line the anchor contained).
    """
    applied, skipped = 0, []
    for edit in edits[:max_edits]:
        if not edit.find.strip():
            skipped.append("empty find")
            continue
        count = text.count(edit.find)
        if count != 1:
            skipped.append(f"find occurs {count} times: {edit.find[:60]!r}")
            continue
        headings = re.findall(r"^#{2,6} .*$", edit.find, re.MULTILINE)
        if any(h not in edit.replace for h in headings):
            skipped.append(f"would remove a heading: {edit.find[:60]!r}")
            continue
        text = text.replace(edit.find, edit.replace, 1)
        applied += 1
    if len(edits) > max_edits:
        skipped.append(f"{len(edits) - max_edits} edits over the cap")
    return text, applied, skipped


def parse_lines(raw: str | None) -> list[str]:
    """Non-empty bullet/plain lines of a tag body ("none" means empty)."""
    lines = []
    for line in (raw or "").splitlines():
        item = line.strip().lstrip("-*•").strip()
        if item and item.lower() not in {"none", "n/a", "없음", "-"}:
            lines.append(item)
    return lines


# A display block: "$$" on its own line … "$$", or one whole-line "$$…$$".
# A prose line that merely starts with inline math ("$$W_0$$는 …") is not one.
_DISPLAY_MATH = re.compile(
    r"^\$\$[ \t]*\n.*?\n[ \t]*\$\$[ \t]*$|^\$\$(?:(?!\$\$)[^\n])+\$\$[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_TABLE_HEADER = re.compile(r"^(\|[^\n]*\|)\n\|\s*:?-{3,}", re.MULTILINE)
_IMAGE_SRC = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
_SECTION_HEADING = re.compile(r"^#{2,3} [^\n]+", re.MULTILINE)


def article_ledger(sections: Sequence[str]) -> str:
    """What the article already shows, extracted in code: headings, display
    equations, table headers and images — none may appear again."""
    text = "\n\n".join(sections)
    lines = [f"- heading: {h}" for h in _SECTION_HEADING.findall(text)]
    lines += [
        f"- equation: {' '.join(m.group(0).split())[:160]}"
        for m in _DISPLAY_MATH.finditer(text)
    ]
    lines += [f"- table: {h[:160]}" for h in _TABLE_HEADER.findall(text)]
    lines += [f"- figure: {src}" for src in _IMAGE_SRC.findall(text)]
    return "\n".join(lines) or "- nothing yet"


def _first_sentence(paragraph: str) -> str:
    flat = " ".join(paragraph.split())
    match = re.search(r"(.+?[.!?。])(\s|$)", flat)
    return (match.group(1) if match else flat)[:200]


def digest_sections(sections: Sequence[str]) -> str:
    """Compact view of earlier sections: headings plus the first sentence of
    each prose paragraph (tables, code and display math left out)."""
    out = []
    for section in sections:
        for block in re.split(r"\n\s*\n", section):
            block = block.strip()
            if not block or block.startswith(("|", "```", "$$", "!", ">")):
                continue
            out.append(
                block if block.startswith("#") else f"  {_first_sentence(block)}"
            )
    return "\n".join(out)


def previous_context(sections: Sequence[str], *, full: int = 2) -> str:
    """The article so far for the next writer: earlier sections as a digest,
    the last ``full`` sections verbatim (keeps context linear, not quadratic)."""
    if not sections:
        return "(this is the first section)"
    head, tail = sections[:-full] if full else sections, (
        sections[-full:] if full else []
    )
    parts = []
    if head:
        parts.append("Earlier sections (digest):\n" + digest_sections(head))
    parts.extend(tail)
    return "\n\n".join(parts)


def render_closing_points(plan: ReviewPlan) -> str:
    return "\n".join(f"- {p}" for p in plan.closing_points) or "- none"
