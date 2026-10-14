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
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .logger import logger
from .markdown_lint import (
    FIGURE_MARKER,
    callout_lines,
    close_open_fence,
    publish_body,
    writing_issues,
)
from .markdown_math import (
    DISPLAY_MATH,
    HANGUL,
    fence_roles,
    image_sources,
    open_fence,
    without_fences,
)
from .post_validator import body_errors
from .utils.helpers import first_sentence

Difficulty = Literal["easy", "medium", "hard"]
Importance = Literal["core", "supporting", "context"]

# Words of prose per must-cover item, by its difficulty for a general DS/MLE
# reader; a section's target is the sum, scaled by how central the section is.
# Hard, core material gets the most room; a section never drops below a floor.
_ITEM_WORDS: dict[str, int] = {"easy": 60, "medium": 200, "hard": 450}
_IMPORTANCE_FACTOR: dict[str, float] = {"context": 0.8, "supporting": 1.0, "core": 1.3}
MIN_SECTION_WORDS = 250
# Whole-review budget in English-word equivalents (≈2.5 Hangul syllables
# each), growing smoothly with the share of hard items. Easy and medium items
# are shortened first, never below half their size. Hard items keep their full
# build-up unless the review would still run past 1.3x the budget; then they
# shrink toward that bound, never below 0.8x, so the overrun stays bounded.
REVIEW_WORD_BUDGET = 7_000
_HARD_SHARE_BONUS = 0.6
MIN_LENGTH_SCALE = 0.5
HARD_OVERRUN = 1.3
MIN_HARD_SCALE = 0.8
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
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_LATIN_WORD = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def _is_closing_heading(title: str) -> bool:
    """Closing-section heading, tolerating an OCR split such as "FUTURE W ORK"
    (a lone letter glued back onto the next word)."""
    if _CLOSING_HEADING.match(title):
        return True
    words = title.split()
    for k, word in enumerate(words[:-1]):
        if len(word) == 1 and word.isalpha():
            joined = " ".join(words[:k] + [word + words[k + 1]] + words[k + 2 :])
            if _CLOSING_HEADING.match(joined):
                return True
    return False


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
    # Set by normalize_plan so the whole review fits its budget.
    length_scale: float = 1.0
    hard_scale: float = 1.0

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
        return target_words(
            self.must_cover, self.importance, self.length_scale, self.hard_scale
        )


class ReviewPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    narrative: str = ""
    glossary: dict[str, str] = Field(default_factory=dict)
    sections: list[PlannedSection] = Field(min_length=1)
    excluded_ranges: list[tuple[int, int]] = Field(default_factory=list)
    # The paper's own limitations and future work, for the closing assessment
    # (the body never gets a limitations/conclusion section of its own).
    closing_points: list[str] = Field(default_factory=list)

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


def target_words(
    items: Sequence[CoverItem],
    importance: str,
    scale: float = 1.0,
    hard_scale: float = 1.0,
) -> int:
    """Section length from what it must carry: per-item budgets by
    difficulty (easy and medium ones times ``scale``, hard ones times
    ``hard_scale``), scaled by importance, never below the floor."""
    words = sum(
        _ITEM_WORDS[i.difficulty] * (hard_scale if i.difficulty == "hard" else scale)
        for i in items
    )
    return max(MIN_SECTION_WORDS, round(words * _IMPORTANCE_FACTOR[importance]))


def review_word_budget(plan: ReviewPlan) -> int:
    """The whole review's length budget, larger when more items are hard."""
    items = [c for s in plan.sections for c in s.must_cover]
    hard_share = sum(c.difficulty == "hard" for c in items) / len(items) if items else 0
    return round(REVIEW_WORD_BUDGET * (1 + _HARD_SHARE_BONUS * hard_share))


def _fit_budget(plan: ReviewPlan) -> None:
    """Shorten the easy and medium items (one scale for the whole review,
    floors included in the total) until the targets fit the budget or reach
    :data:`MIN_LENGTH_SCALE`; if the review is still over
    :data:`HARD_OVERRUN` times the budget, shorten the hard items toward that
    bound, down to :data:`MIN_HARD_SCALE`."""

    def total(scale: float, hard: float = 1.0) -> int:
        for section in plan.sections:
            section.length_scale, section.hard_scale = scale, hard
        return sum(s.target_words for s in plan.sections)

    def fit(set_total: Callable[[float], int], low: float, cap: float) -> float:
        # Bisection for the largest scale in [low, 1] whose total fits ``cap``
        # (the total is monotone in the scale); ``low`` if none does.
        if set_total(low) > cap:
            return low
        high = 1.0
        for _ in range(20):
            mid = (low + high) / 2
            low, high = (mid, high) if set_total(mid) <= cap else (low, mid)
        return low

    budget = review_word_budget(plan)
    if total(1.0) <= budget:
        return
    scale = fit(total, MIN_LENGTH_SCALE, budget)
    hard = 1.0
    if total(scale) > HARD_OVERRUN * budget:
        hard = fit(lambda h: total(scale, h), MIN_HARD_SCALE, HARD_OVERRUN * budget)
    fitted = total(scale, hard)
    logger.info(
        "Review targets: easy/medium items x%.2f, hard items x%.2f, %d words "
        "for a %d-word budget.",
        scale,
        hard,
        fitted,
        budget,
    )


def approx_words(text: str) -> int:
    """Length in English-word equivalents: Latin words plus Hangul syllables
    / 2.5 (a Korean sentence carrying the same content), code fences excluded."""
    prose = without_fences(text)
    hangul = len(HANGUL.findall(prose))
    latin = len(_LATIN_WORD.findall(HANGUL.sub(" ", prose)))
    return latin + round(hangul / 2.5)


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
    roles = fence_roles([text[a:b] for a, b in lines])
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
        if roles[i] == "open":
            j = i + 1
            while j < len(lines) and roles[j] == "code":
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
            if match and _is_closing_heading(match.group(2).strip()):
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
    for match in FIGURE_MARKER.finditer(text):
        seen.setdefault(match.group("src"), match.group(0))
    return [(f"F{i}", marker) for i, marker in enumerate(seen.values(), start=1)]


def marker_src(marker: str) -> str:
    match = FIGURE_MARKER.search(marker)
    return match.group("src") if match else ""


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
    normalized = plan.model_copy(
        update={"sections": sections, "excluded_ranges": excluded}
    )
    _fit_budget(normalized)
    return normalized


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
    text = FIGURE_MARKER.sub(
        lambda m: m.group(0) if m.group("src") in keep else "", text
    )
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
            f"(about {round(section.target_words * 2.5)} Hangul syllables in Korean); "
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
    """A section starts with exactly its ``##`` heading (and closes any code
    fence it left open)."""
    body = close_open_fence(text.strip())
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


def draft_rank(
    *, passes: bool, publishable: bool, blocking: int, score: int
) -> tuple[bool, bool, int, int]:
    """Order of two drafts of a review section (higher is better): passing
    first, then publishable (no error the validator would reject), then the
    score, then fewer blocking defects (missing items, must-fix, rule slips).
    A writing slip never outweighs the score: a faithful draft with one slip
    beats an invented one. The guide and survey loop
    (:func:`section_loop.revise_section`) puts fewer factual issues ahead of
    the score and passes only wording issues and rule slips as ``blocking``."""
    return passes, publishable, score, -blocking


def rejected_revision_note(score: int, blocking: int) -> str:
    """Feedback line for a revision that did not replace the kept draft."""
    return (
        f"A previous revision of this draft was rejected (score {score}, "
        f"{blocking} blocking problem(s)); avoid its problems:"
    )


def rule_issues(text: str) -> list[str]:
    """Deterministic rule hits a writer can fix: writing slips plus the
    publish-blocking body checks (tables, math, code fences) that are left
    after the publisher's own fixes (math delimiters, blank lines, headings),
    so a draft is never revised for what the publisher repairs."""
    return writing_issues(text) + publish_blockers(text)


def publish_blockers(text: str) -> list[str]:
    """The body errors the validator would still report once the section is
    closed (an open fence) and the publisher has normalized math and linted
    the Markdown."""
    fixed = publish_body(text, warn=False)
    return body_errors(fixed)


def added_slips(
    before: str, after: str, rules: Callable[[str], list[str]] = rule_issues
) -> list[str]:
    """Rule hits (rule and offending text) an edit would add, even when it
    also fixes another; ``rules`` finds them (:func:`rule_issues` by
    default)."""

    def keys(text: str) -> Counter[str]:
        return Counter(issue.split(":", 1)[0] for issue in rules(text))

    return list((keys(after) - keys(before)).elements())


def _is_subsequence(items: list[str], within: list[str]) -> bool:
    remaining = iter(within)
    return all(item in remaining for item in items)


def apply_edits(
    text: str, edits: list[Edit], *, max_edits: int
) -> tuple[str, int, list[str]]:
    """Apply edits whose ``find`` occurs exactly once in the current text.

    Returns ``(text, applied, skipped_reasons)``. Ambiguous or missing anchors
    are skipped rather than guessed, so an edit can never land in the wrong
    place. Top-level ``## `` headings cannot be added, removed, demoted or
    renamed (the edited text must keep the same list of them); a
    redundant ``###``/``####`` subsection may be (level jumps are repaired at
    publish time). An edit may drop a repeated figure, display equation or
    table, never its only copy, and may not add a writing-rule slip.
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
        edited = text.replace(edit.find, edit.replace, 1)
        # Closing the article's one unclosed fence: the edit only adds the
        # closing fence line and changes nothing else.
        closes_fence = (
            bool(open_fence(text))
            and not open_fence(edited)
            and _without_bare_fences(edit.replace) == _without_bare_fences(edit.find)
        )
        if open_fence(edit.find) != open_fence(edit.replace) and not closes_fence:
            # Adding or dropping a lone fence line opens or closes a code
            # block elsewhere (prose would turn into code, or code into
            # prose), unless it closes the article's one unclosed fence.
            # Changing an info string or moving a fence keeps the balance.
            skipped.append(f"would unbalance a code fence: {edit.find[:60]!r}")
            continue
        # On the text outside code, so an edit that drops a closing fence (and
        # turns the following sections into code) is caught too.
        before = _TOP_HEADING.findall(without_fences(text))
        after = _TOP_HEADING.findall(without_fences(edited))
        # Closing the article's unclosed fence may only bring back headings
        # the open code block had swallowed; it may not lose or rename one.
        if after != before and not (closes_fence and _is_subsequence(before, after)):
            skipped.append(f"would change a heading: {edit.find[:60]!r}")
            continue
        if lost := _lost_assets(edit.find, text, edited):
            skipped.append(f"would remove the only {lost}: {edit.find[:60]!r}")
            continue
        if added := added_slips(text, edited):
            skipped.append(
                f"would add a writing-rule slip ({added[0]}): {edit.find[:60]!r}"
            )
            continue
        text = edited
        applied += 1
    if len(edits) > max_edits:
        skipped.append(f"{len(edits) - max_edits} edits over the cap")
    return text, applied, skipped


_BARE_FENCE_LINE = re.compile(r"(?m)^[ \t]*(?:`{3,}|~{3,})[ \t]*\n?")


def _without_bare_fences(text: str) -> str:
    """``text`` without lines that are only a fence (a closing fence)."""
    return _BARE_FENCE_LINE.sub("", text).strip()


def _lost_assets(find: str, text: str, edited: str) -> str:
    """The kind of figure, display equation or table the edit removes without
    another copy left ("" when none): fixing one in place keeps the count and
    is fine; dropping a repeated copy leaves the other in ``edited``."""
    for kind, items in (
        ("figure", image_sources),
        ("equation", DISPLAY_MATH.findall),
        ("table", _TABLE_HEADER.findall),
    ):
        fewer = len(items(edited)) < len(items(text))
        if fewer and any(item not in edited for item in items(find)):
            return kind
    return ""


_TABLE_HEADER = re.compile(r"^(\|[^\n]*\|)[ \t]*\n\|[ \t]*:?-+", re.MULTILINE)
_TOP_HEADING = re.compile(r"^## .*$", re.MULTILINE)
_SECTION_HEADING = re.compile(r"^#{2,3} [^\n]+", re.MULTILINE)


def _code_first_lines(text: str) -> list[str]:
    """The first non-empty line of every fenced code block."""
    lines = text.splitlines()
    firsts, first, inside = [], "", False
    for line, role in zip(lines, fence_roles(lines), strict=True):
        if role == "open":
            first, inside = "", True
        elif role == "code" and not first and line.strip():
            first = line.strip()
        elif role == "close":
            inside = False
            if first:
                firsts.append(first)
    if inside and first:
        firsts.append(first)
    return firsts


def article_ledger(sections: Sequence[str]) -> str:
    """What the article already shows, extracted in code: headings, display
    equations, table headers, images, code blocks and callouts — none may
    appear again (callouts are listed so writers can keep within the cap)."""
    text = "\n\n".join(sections)
    lines = [f"- heading: {h}" for h in _SECTION_HEADING.findall(text)]
    lines += [
        f"- equation: {' '.join(m.group(0).split())[:160]}"
        for m in DISPLAY_MATH.finditer(text)
    ]
    lines += [f"- table: {h[:160]}" for h in _TABLE_HEADER.findall(text)]
    lines += [f"- figure: {src}" for src in image_sources(text)]
    lines += [f"- code: {first[:160]}" for first in _code_first_lines(text)]
    lines += [f"- callout: {c[:160]}" for c in callout_lines(text)]
    return "\n".join(lines) or "- nothing yet"


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
                block
                if block.startswith("#")
                else f"  {first_sentence(' '.join(block.split()))[:200]}"
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
