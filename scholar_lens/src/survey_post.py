"""Deterministic parts of the paper-survey write stage.

``survey write`` reads a plan file (``survey_plan.SurveyPlan``) that the user
may have edited, turns each paper into a card (one model call per paper,
checked here against the paper's own text), plans an outline (one model call,
checked here against the plan and the cards), writes the sections and
assembles the post. This module holds everything that is code, not model
judgment: re-reading the plan, the card and outline schemas and their checks,
the length targets, citation keys, the timeline and comparison tables, figure
credits and the reference list. Nothing here does I/O.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .formatting import link_list, md_link_text, md_link_url
from .markdown_lint import writing_issues
from .markdown_math import segments
from .post_validator import (
    FURTHER_READING_HEADINGS,
    REFERENCES_HEADING,
    unsourced_numbers,
)
from .prompts.tech_guide import DEPTH_CHARS, NON_KOREAN_CHAR_FACTOR
from .publish_types import slugify
from .survey_plan import (
    PlanPaper,
    SourceRecord,
    SurveyInputError,
    SurveyPlan,
    merge_records,
    parse_paper_id,
)
from .tech_guide import parse_glossary
from .utils.helpers import _escape_cell_pipes

# Cards are cached by (paper, CARD_SCHEMA_VERSION, model, language, card
# prompt and source limit): bump the version when the card fields or their
# meaning change.
CARD_SCHEMA_VERSION = 2
# Figures a core paper's card may offer the post.
MAX_CARD_FIGURES = 2
# Equations a card may offer the post.
MAX_CARD_EQUATIONS = 3
# Results shown per paper in the comparison table.
TABLE_RESULTS = 2
_SENTENCE_END = re.compile(r"(?<=[.!?。])\s+")
_MATH_DELIMS = re.compile(r"^\$\$|\$\$$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- the plan, read again ------------------------------------------------------


def plan_data(text: str) -> dict[str, Any]:
    """The plan file as JSON data; ``SurveyInputError`` when it is not a
    JSON object."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SurveyInputError(f"The plan is not valid JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("papers"), list):
        raise SurveyInputError("The plan must be a JSON object with a 'papers' list.")
    for key in ("papers", "lineages", "edges"):
        items = data.get(key) or []
        if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
            raise SurveyInputError(f"The plan's '{key}' must be a list of objects.")
    if not all(isinstance(p.get("id"), str) for p in data["papers"]):
        raise SurveyInputError("Every paper of the plan needs an 'id'.")
    return data


def needs_metadata(paper: Mapping[str, Any]) -> bool:
    """A paper the user added by hand: no title or URL yet."""
    return not (paper.get("title") and paper.get("url"))


def normalize_plan_data(data: dict[str, Any]) -> list[str]:
    """Bring an edited plan back to a consistent state, in place, before it
    is validated: paper ids as plan ids (an arXiv URL or doi.org link
    becomes the bare id), edges that name a removed paper dropped, lineages
    left without papers dropped. Returns a note per change."""
    notes: list[str] = []
    for paper in data["papers"]:
        paper["id"] = parse_paper_id(paper["id"])
    ids = {p["id"] for p in data["papers"]}
    edges = data.get("edges") or []
    kept = [
        e
        for e in edges
        if e.get("from", e.get("citing")) in ids and e.get("to", e.get("cited")) in ids
    ]
    if len(kept) < len(edges):
        notes.append(
            f"{len(edges) - len(kept)} citation edge(s) of papers removed from "
            "the plan were dropped."
        )
        data["edges"] = kept
    used = {p.get("lineage") for p in data["papers"]}
    lineages = data.get("lineages") or []
    empty = [x.get("id") for x in lineages if x.get("id") not in used]
    if empty:
        notes.append(f"Lineages without papers were dropped: {', '.join(empty)}.")
        data["lineages"] = [x for x in lineages if x.get("id") in used]
    return notes


def fill_metadata(paper: dict[str, Any], records: Sequence[SourceRecord]) -> None:
    """Fill a hand-added paper's missing bibliographic fields from its index
    records (arXiv's at DataCite, or OpenAlex's); fields the user wrote stay."""
    candidate = merge_records(list(records))[0]
    found = {
        "title": candidate.title,
        "authors": candidate.authors_label,
        "year": candidate.year,
        "date": candidate.published.isoformat() if candidate.published else None,
        "venue": candidate.venue,
        "url": candidate.url,
    }
    for key, value in found.items():
        if not paper.get(key) and value:
            paper[key] = value


def validate_plan(data: dict[str, Any]) -> SurveyPlan:
    """The plan, its papers in date order (a paper added by hand goes where
    its date puts it; one without a date goes last)."""
    try:
        plan = SurveyPlan.model_validate(data)
    except ValidationError as e:
        raise SurveyInputError(f"The plan is not valid: {e}") from e
    plan.papers.sort(
        key=lambda p: (not (p.date or p.year), p.date or str(p.year or ""), p.id)
    )
    return plan


def without_papers(plan: SurveyPlan, dropped: Iterable[str]) -> SurveyPlan:
    """``plan`` without the ``dropped`` papers, their edges and the lineages
    they leave empty."""
    gone = set(dropped)
    data = plan.model_dump(by_alias=True)
    data["papers"] = [p for p in data["papers"] if p["id"] not in gone]
    if not data["papers"]:
        raise SurveyInputError("No paper of the plan has any text to write from.")
    normalize_plan_data(data)
    return SurveyPlan.model_validate(data)


# ---- paper cards ---------------------------------------------------------------------

CardSource = Literal["full_text", "abstract"]


class CardResult(_Strict):
    claim: str = Field(min_length=1)
    value: str = Field(min_length=1)
    setting: str = ""


class CardFigureChoice(_Strict):
    id: str
    reason: str = ""


class CardEquationAnswer(_Strict):
    latex: str = Field(min_length=1)
    meaning: str = ""


# How a paper relates to a prior work it names (its own words decide).
PriorRelation = Literal["builds_on", "compares_against", "replaces", "other"]
PRIOR_RELATIONS: dict[str, str] = {
    "builds_on": "builds on",
    "compares_against": "compares against",
    "replaces": "replaces",
    "other": "relates to",
}


class CardPriorWork(_Strict):
    name: str = Field(min_length=1)
    relation: PriorRelation = "other"


class CardAnswer(_Strict):
    """The card call's output."""

    name: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    key_idea: str = Field(min_length=1)
    method: str = Field(min_length=1)
    key_equations: list[CardEquationAnswer] = Field(default_factory=list, max_length=6)
    results: list[CardResult] = Field(default_factory=list, max_length=8)
    limitations: list[str] = Field(default_factory=list, max_length=6)
    prior_work: list[CardPriorWork] = Field(default_factory=list, max_length=12)
    figures: list[CardFigureChoice] = Field(default_factory=list)


class CardFigure(_Strict):
    # "<paper id>#F<n>", as the outline names it.
    id: str
    url: str
    caption: str


class CardEquation(_Strict):
    # "<paper id>#E<n>", as the outline names it.
    id: str
    latex: str
    meaning: str = ""


class PaperCard(_Strict):
    """A paper as the writers see it: the card call's answer with every
    number, equation and figure checked against the paper in code."""

    schema_version: Literal[2] = 2  # CARD_SCHEMA_VERSION
    paper_id: str
    model_id: str
    language: str
    source: CardSource
    name: str
    problem: str
    key_idea: str
    method: str
    key_equations: list[CardEquation] = Field(default_factory=list)
    results: list[CardResult] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    prior_work: list[CardPriorWork] = Field(default_factory=list)
    figures: list[CardFigure] = Field(default_factory=list)
    # What the checks removed (for the PR checklist).
    dropped: list[str] = Field(default_factory=list)


def card_cache_key(paper_id: str, model_id: str, language: str, variant: str) -> str:
    """``<paper>/v<schema>-<model>-<language>-<variant>.json``: a card is
    reused by every survey that covers the paper with the same model,
    language and ``variant`` (a digest of the card prompt and source limit,
    so changing either writes the card again)."""
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id)
    return (
        f"{safe_id}/v{CARD_SCHEMA_VERSION}-{slugify(model_id)}-"
        f"{slugify(language)}-{variant}.json"
    )


PANEL_NOTE = "(one panel of a multi-panel figure; other panels not shown)"


def figure_list(captions: Mapping[str, str]) -> list[tuple[str, str, str]]:
    """``(F<n>, url, caption)`` for each captioned image of a paper's arXiv
    HTML rendering, in page order (an image without a caption is no figure
    a survey can explain). The panels of a multi-panel figure share its
    caption: only the first image of such a run is listed, its caption
    marked as one panel."""
    figures: list[tuple[str, str]] = []
    run: str | None = None
    for url, caption in captions.items():
        if caption and caption == run:
            if not figures[-1][1].endswith(PANEL_NOTE):
                figures[-1] = (figures[-1][0], f"{caption} {PANEL_NOTE}")
        elif caption:
            figures.append((url, caption))
        run = caption or None
    return [(f"F{n}", url, caption) for n, (url, caption) in enumerate(figures, 1)]


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", _MATH_DELIMS.sub("", text.strip()))


def equation_in_source(equation: str, source: str) -> bool:
    """Whether ``equation`` is LaTeX the source shows (whitespace and ``$$``
    delimiters ignored)."""
    return bool(equation.strip()) and _squash(equation) in re.sub(r"\s+", "", source)


def _prose_fields(answer: CardAnswer) -> dict[str, str]:
    return {
        "problem": answer.problem,
        "key_idea": answer.key_idea,
        "method": answer.method,
    }


def card_problems(
    answer: CardAnswer, *, source: str, figure_ids: Iterable[str]
) -> list[str]:
    """What the card call must fix (empty when the card is fine): numbers
    the paper never states, an equation it does not show, unknown or too
    many figures, a reference to the paper's numbering ("표 1") the post's
    tables would carry."""
    problems = []
    texts = [
        *_prose_fields(answer).values(),
        *(f"{r.claim} {r.value} {r.setting}" for r in answer.results),
        *answer.limitations,
        *(e.meaning for e in answer.key_equations),
    ]
    for text in texts:
        problems += [
            f"{issue.split(' (', 1)[0]}: refer to the paper's tables, figures, "
            "equations and sections by what they show, never by number"
            for issue in writing_issues(text)
            if issue.startswith("numbered reference")
        ]
    for name, text in _prose_fields(answer).items():
        if missing := unsourced_numbers(text, source):
            problems.append(f"{name} has numbers the paper never states: {missing}")
    for n, result in enumerate(answer.results, 1):
        text = f"{result.claim} {result.value} {result.setting}"
        if missing := unsourced_numbers(text, source):
            problems.append(f"result {n} has numbers the paper never states: {missing}")
    for n, item in enumerate(answer.limitations, 1):
        if missing := unsourced_numbers(item, source):
            problems.append(
                f"limitation {n} has numbers the paper never states: {missing}"
            )
    for n, equation in enumerate(answer.key_equations, 1):
        if not equation_in_source(equation.latex, source):
            problems.append(
                f"key equation {n} is not copied from the paper's LaTeX; copy "
                "it exactly or leave it out"
            )
        if missing := unsourced_numbers(equation.meaning, source):
            problems.append(
                f"key equation {n}'s meaning has numbers the paper never "
                f"states: {missing}"
            )
    if len(answer.key_equations) > MAX_CARD_EQUATIONS:
        problems.append(f"give at most {MAX_CARD_EQUATIONS} key equations")
    known = set(figure_ids)
    problems += [
        f"unknown figure id {f.id!r}" for f in answer.figures if f.id not in known
    ]
    if len(answer.figures) > MAX_CARD_FIGURES:
        problems.append(f"choose at most {MAX_CARD_FIGURES} figures")
    return problems


def _verified_sentences(text: str, source: str, dropped: list[str]) -> str:
    kept = []
    for sentence in _SENTENCE_END.split(text.strip()):
        if missing := unsourced_numbers(sentence, source):
            dropped.append(
                f"sentence with unverified number(s) {missing}: {sentence[:80]}"
            )
        else:
            kept.append(sentence)
    return " ".join(kept)


def build_card(
    answer: CardAnswer,
    *,
    paper_id: str,
    source: str,
    source_kind: CardSource,
    figures: Sequence[tuple[str, str, str]],
    model_id: str,
    language: str,
) -> PaperCard:
    """The card from the call's answer, with what the paper does not back
    removed: sentences and results whose numbers the paper never states, an
    equation it does not show, figures not in its figure list (at most
    :data:`MAX_CARD_FIGURES`)."""
    dropped: list[str] = []
    prose = {
        name: _verified_sentences(text, source, dropped)
        for name, text in _prose_fields(answer).items()
    }
    results = []
    for result in answer.results:
        text = f"{result.claim} {result.value} {result.setting}"
        if missing := unsourced_numbers(text, source):
            dropped.append(
                f"result with unverified number(s) {missing}: {result.claim}"
            )
        else:
            results.append(result)
    limitations = []
    for item in answer.limitations:
        if missing := unsourced_numbers(item, source):
            dropped.append(f"limitation with unverified number(s) {missing}")
        else:
            limitations.append(item)
    equations = []
    for equation in answer.key_equations:
        if equation_in_source(equation.latex, source):
            equations.append(equation)
        else:
            dropped.append("key equation not found in the paper")
    by_id = {fid: (url, caption) for fid, url, caption in figures}
    chosen = [f.id for f in answer.figures if f.id in by_id]
    if len(chosen) < len(answer.figures):
        dropped.append("figure ids not in the paper's figure list")
    return PaperCard(
        paper_id=paper_id,
        model_id=model_id,
        language=language,
        source=source_kind,
        name=answer.name.strip(),
        problem=prose["problem"],
        # The tables need a key idea even when its every sentence had an
        # unverified number: fall back to the paper's name.
        key_idea=prose["key_idea"] or answer.name.strip(),
        method=prose["method"],
        key_equations=[
            CardEquation(
                id=f"{paper_id}#E{n}",
                latex=_MATH_DELIMS.sub("", e.latex.strip()).strip(),
                meaning=_verified_sentences(e.meaning, source, dropped),
            )
            for n, e in enumerate(equations[:MAX_CARD_EQUATIONS], 1)
        ],
        results=results,
        limitations=limitations,
        prior_work=[
            CardPriorWork(name=p.name.strip(), relation=p.relation)
            for p in answer.prior_work
            if p.name.strip()
        ],
        figures=[
            CardFigure(id=f"{paper_id}#{fid}", url=by_id[fid][0], caption=by_id[fid][1])
            for fid in dict.fromkeys(chosen)
        ][:MAX_CARD_FIGURES],
        dropped=dropped,
    )


def _result_line(result: CardResult) -> str:
    return f"{result.claim}: {result.value}" + (
        f" ({result.setting})" if result.setting else ""
    )


def prior_work_text(card: PaperCard) -> str:
    """ "builds on: PPO; compares against: RLHF" (the card's own relations)."""
    by_relation: dict[str, list[str]] = {}
    for item in card.prior_work:
        by_relation.setdefault(item.relation, []).append(item.name)
    return "; ".join(
        f"{PRIOR_RELATIONS[relation]}: {', '.join(names)}"
        for relation, names in by_relation.items()
    )


def render_card(card: PaperCard, paper: PlanPaper) -> str:
    """A card in full, for the writer, evaluator and fact-checker of a
    section that covers the paper (its figures are offered per section)."""
    lines = [
        f"### [@{paper.id}] {card.name} — {paper.title} ({paper.authors}, "
        f"{paper.year or 'n.d.'}; {paper.role}; read from its "
        f"{'full text' if card.source == 'full_text' else 'abstract'})",
        f"- Problem: {card.problem}",
        f"- Key idea: {card.key_idea}",
        f"- Method: {card.method}",
    ]
    lines += [
        f"- Equation {e.id}: $${e.latex}$$ — {e.meaning}" for e in card.key_equations
    ]
    lines += [f"- Result: {_result_line(r)}" for r in card.results]
    lines += [f"- Limitation: {item}" for item in card.limitations]
    if card.prior_work:
        lines.append(f"- Prior work it names: {prior_work_text(card)}")
    return "\n".join(lines)


def render_card_brief(card: PaperCard, paper: PlanPaper) -> str:
    """One line per card, for sections that only mention the paper."""
    return f"- [@{paper.id}] {card.name} ({paper.year or 'n.d.'}): {card.key_idea}"


def render_card_outcomes(card: PaperCard, paper: PlanPaper) -> str:
    """The brief line with the paper's results and limitations, for the
    comparison section."""
    lines = [render_card_brief(card, paper)]
    lines += [f"  - Result: {_result_line(r)}" for r in card.results]
    lines += [f"  - Limitation: {item}" for item in card.limitations]
    return "\n".join(lines)


# ---- outline -------------------------------------------------------------------------

SectionKind = Literal[
    "introduction", "lineage", "cross_cutting", "comparison", "insights", "conclusion"
]
Depth = Literal["deep", "standard", "brief"]


class OutlineSection(_Strict):
    kind: SectionKind
    title: str = Field(min_length=1)
    description: str = ""
    depth: Depth = "standard"
    # A lineage section's lineage id; its papers are the lineage's (in code).
    lineage: str | None = None
    papers: list[str] = Field(default_factory=list)
    # "<paper id>#F<n>" ids from the cards.
    figures: list[str] = Field(default_factory=list)
    # "<paper id>#E<n>" ids from the cards.
    equations: list[str] = Field(default_factory=list)
    # Tables the writer builds from the cards (one line each).
    tables: list[str] = Field(default_factory=list)
    # Concepts the section explains; the others only use them.
    owns: list[str] = Field(default_factory=list)
    # Set in code (:func:`apply_outline`).
    target_chars: int = 0


class SurveyOutline(_Strict):
    """The outline call's output."""

    title: str = Field(min_length=1)
    sections: list[OutlineSection] = Field(min_length=4)
    # "English term | rendering" lines (the guide's glossary format).
    glossary: list[str] = Field(default_factory=list, max_length=20)

    @property
    def glossary_map(self) -> dict[str, str]:
        return parse_glossary("\n".join(self.glossary))


_ORDERED_TAIL = ("comparison", "insights", "conclusion")


def check_outline(
    outline: SurveyOutline,
    plan: SurveyPlan,
    cards: Mapping[str, PaperCard],
    *,
    title_max_length: int,
) -> list[str]:
    """Every problem of ``outline`` (empty when fine), for the repair round:
    a title over the limit, :func:`structure_problems` and the references
    :func:`clean_outline` would remove."""
    problems = []
    if len(outline.title) > title_max_length:
        problems.append(
            f"title has {len(outline.title)} characters; at most {title_max_length}"
        )
    problems += structure_problems(outline, plan)
    return problems + clean_outline(outline.model_copy(deep=True), plan, cards)


def structure_problems(outline: SurveyOutline, plan: SurveyPlan) -> list[str]:
    """What makes the outline unusable: the sections' kinds, order and one
    lineage section per plan lineage."""
    problems = []
    kinds = [s.kind for s in outline.sections]
    if kinds[0] != "introduction" or kinds.count("introduction") != 1:
        problems.append("the first section, and only it, is the introduction")
    for kind in ("comparison", "insights"):
        if kinds.count(kind) != 1:
            problems.append(f"plan exactly one {kind} section")
    if kinds.count("conclusion") > 1 or (
        "conclusion" in kinds and kinds[-1] != "conclusion"
    ):
        problems.append("at most one conclusion section, and it comes last")
    body = [i for i, k in enumerate(kinds) if k in ("lineage", "cross_cutting")]
    tail = [i for i, k in enumerate(kinds) if k in _ORDERED_TAIL]
    if body and tail and max(body) > min(tail):
        problems.append(
            "lineage and cross-cutting sections come before the comparison, "
            "insights and conclusion"
        )
    lineage_ids = [lin.id for lin in plan.lineages]
    planned = [s.lineage for s in outline.sections if s.kind == "lineage"]
    for lineage in lineage_ids:
        if planned.count(lineage) != 1:
            problems.append(f"plan exactly one lineage section for {lineage!r}")
    problems += [
        f"unknown lineage {x!r} in a lineage section"
        for x in planned
        if x not in lineage_ids
    ]
    return problems


def clean_outline(
    outline: SurveyOutline, plan: SurveyPlan, cards: Mapping[str, PaperCard]
) -> list[str]:
    """Remove, in place, the references a section cannot use (unknown
    papers, unknown figures or equations, one of a paper it does not cover
    or planned earlier); one line per removal."""
    removed = []
    paper_ids = {p.id for p in plan.papers}
    used_figures: set[str] = set()
    used_equations: set[str] = set()
    for section in outline.sections:
        unknown = [p for p in section.papers if p not in paper_ids]
        removed += [
            f"section {section.title!r} names unknown paper {p!r}" for p in unknown
        ]
        section.papers = [p for p in section.papers if p not in unknown]
        papers = section_papers(section, plan)
        kept = []
        for fid in section.figures:
            owner = fid.split("#", 1)[0]
            if not any(f.id == fid for f in getattr(cards.get(owner), "figures", [])):
                removed.append(f"unknown figure {fid!r}")
            elif owner not in papers:
                removed.append(
                    f"figure {fid!r} is in section {section.title!r}, which does "
                    f"not cover paper {owner!r}"
                )
            elif fid in used_figures:
                removed.append(f"figure {fid!r} is planned twice")
            else:
                kept.append(fid)
                used_figures.add(fid)
        section.figures = kept
        equations = []
        for eid in section.equations:
            owner = eid.split("#", 1)[0]
            card = cards.get(owner)
            if not any(e.id == eid for e in getattr(card, "key_equations", [])):
                removed.append(f"unknown equation {eid!r}")
            elif owner not in papers:
                removed.append(
                    f"equation {eid!r} is in section {section.title!r}, which "
                    f"does not cover paper {owner!r}"
                )
            elif eid in used_equations:
                removed.append(f"equation {eid!r} is planned twice")
            else:
                equations.append(eid)
                used_equations.add(eid)
        section.equations = equations
    return removed


def section_papers(section: OutlineSection, plan: SurveyPlan) -> list[str]:
    """The papers a section covers: a lineage section its lineage's papers
    (the plan's chronological order), any other section the ones it names."""
    if section.kind == "lineage":
        return [p.id for p in plan.papers if p.lineage == section.lineage]
    return list(section.papers)


def section_targets(
    outline: SurveyOutline,
    plan: SurveyPlan,
    *,
    per_core: int,
    per_supporting: int,
    min_total: int,
    max_total: int,
    language: str,
) -> list[int]:
    """Prose characters per section: a lineage section the sum of its
    papers' shares, any other section its depth's size; the total scaled
    into ``[min_total, max_total]`` (Korean characters; other languages
    need :data:`NON_KOREAN_CHAR_FACTOR` times as many)."""
    share = {"core": per_core, "supporting": per_supporting}
    role = {p.id: p.role for p in plan.papers}
    raw = [
        (
            sum(share[role[p]] for p in section_papers(s, plan))
            if s.kind == "lineage"
            else DEPTH_CHARS[s.depth]
        )
        for s in outline.sections
    ]
    total = sum(raw)
    scale = min(max(total, min_total), max_total) / total
    factor = 1.0 if language == "Korean" else NON_KOREAN_CHAR_FACTOR
    return [round(r * scale * factor) for r in raw]


def apply_outline(
    outline: SurveyOutline, plan: SurveyPlan, targets: Sequence[int]
) -> None:
    """Fix each section's papers and target length in the outline."""
    for section, target in zip(outline.sections, targets, strict=True):
        section.papers = section_papers(section, plan)
        section.target_chars = target


# ---- citations -----------------------------------------------------------------------

# "[@2305.18290]" or "[@2305.18290; @10.18653/v1/x]".
_CITATION = re.compile(r"\[(@[^\[\]\n]+)\]")


def _keys(group: str) -> list[str]:
    return [k.strip().removeprefix("@").strip() for k in group.split(";")]


def _outside_code(text: str) -> list[re.Match[str]]:
    code = [(s.start, s.end) for s in segments(text) if s.kind in ("fence", "icode")]
    return [
        m
        for m in _CITATION.finditer(text)
        if not any(a <= m.start() < b for a, b in code)
    ]


def citation_keys(text: str) -> list[str]:
    """The paper ids ``text`` cites with ``[@id]`` keys, in order."""
    return [key for m in _outside_code(text) for key in _keys(m.group(1))]


def citation_issues(
    text: str, papers: Mapping[str, PlanPaper], required: Iterable[str]
) -> list[str]:
    """Rule issues of a section's citations: a key no plan paper has (a
    citation the model made up), a paper link typed by hand to a paper
    outside the survey, and a paper the section covers that it never
    cites."""
    keys = citation_keys(text)
    issues = [
        f"citation: [@{key}] is no paper of this survey; cite only the listed keys"
        for key in dict.fromkeys(keys)
        if key not in papers
    ]
    issues += [
        f"citation: {url} links a paper outside this survey; name it in plain text"
        for url in dict.fromkeys(foreign_paper_links(text, papers.values()))
    ]
    issues += [
        f"citation: the section covers [@{pid}] but never cites it"
        for pid in required
        if pid not in keys
    ]
    return issues


# A link to a paper's own page (what a citation renders to).
_PAPER_LINK = re.compile(
    r"\[([^\]\n]+)\]\((https?://(?:arxiv\.org/abs/|doi\.org/)[^)\s]+)\)"
)


def _plan_urls(papers: Iterable[PlanPaper]) -> set[str]:
    return {u for p in papers for u in (p.url, md_link_url(p.url))}


def foreign_paper_links(text: str, papers: Iterable[PlanPaper]) -> list[str]:
    """Links to papers outside the survey that a writer typed by hand
    (outside code); the validator rejects them, since References lists only
    the survey's papers."""
    known = _plan_urls(papers)
    code = [(s.start, s.end) for s in segments(text) if s.kind in ("fence", "icode")]
    return [
        m.group(2)
        for m in _PAPER_LINK.finditer(text)
        if m.group(2) not in known and not any(a <= m.start() < b for a, b in code)
    ]


def unlink_foreign_papers(text: str, papers: Iterable[PlanPaper]) -> str:
    """``text`` with hand-typed links to papers outside the survey reduced
    to their link text."""
    foreign = set(foreign_paper_links(text, papers))
    return _PAPER_LINK.sub(
        lambda m: m.group(1) if m.group(2) in foreign else m.group(0), text
    )


def surname(authors: str) -> str:
    """The first author's family name from a plan's authors label."""
    first = authors.removesuffix(" et al.").split(",")[0].strip()
    return first.split()[-1] if first else ""


def author_year(paper: PlanPaper) -> str:
    """ "Rafailov et al., 2023" (one author: "Rafailov, 2023")."""
    names = [n for n in paper.authors.removesuffix(" et al.").split(",") if n.strip()]
    lead = surname(paper.authors) or paper.title
    who = (
        f"{lead} et al." if len(names) > 1 or paper.authors.endswith("et al.") else lead
    )
    return f"{who}, {paper.year or 'n.d.'}"


def render_citations(
    text: str, papers: Mapping[str, PlanPaper]
) -> tuple[str, list[str]]:
    """``[@id]`` keys as "([Rafailov et al., 2023](url))" links; a key no
    paper has is removed. Returns the text and the removed keys."""
    unknown: list[str] = []
    out, pos = [], 0
    for m in _outside_code(text):
        links = []
        for key in _keys(m.group(1)):
            paper = papers.get(key)
            if paper is None:
                unknown.append(key)
            else:
                links.append(
                    f"[{md_link_text(author_year(paper))}]({md_link_url(paper.url)})"
                )
        before = text[pos : m.start()]
        # A citation removed whole takes the space before it along.
        out.append(f"{before}({'; '.join(links)})" if links else before.rstrip(" \t"))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), unknown


# ---- tables, figures and references ------------------------------------------------

_LABELS: dict[str, dict[str, str]] = {
    "Korean": {
        "year": "연도",
        "paper": "논문",
        "lineage": "계보",
        "key_idea": "핵심 아이디어",
        "results": "주요 결과",
        "limitations": "한계",
        "source": "출처",
    },
}
_ENGLISH_LABELS = {
    "year": "Year",
    "paper": "Paper",
    "lineage": "Lineage",
    "key_idea": "Key idea",
    "results": "Main results",
    "limitations": "Limitations",
    "source": "Source",
}


def labels(language: str) -> dict[str, str]:
    return _LABELS.get(language, _ENGLISH_LABELS)


def _cell(text: str) -> str:
    return _escape_cell_pipes(" ".join(text.split())) or "-"


def _paper_link(card: PaperCard, paper: PlanPaper) -> str:
    return f"[{md_link_text(card.name)}]({md_link_url(paper.url)})"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# The tables' cells are the cards' words. Cards are written (and cached)
# before the outline's glossary exists, so they cannot follow it: the card
# call writes plain terms as TERMINOLOGY_RULES say, without English
# parentheticals (the post adds those in its prose).


def timeline_table(
    plan: SurveyPlan,
    cards: Mapping[str, PaperCard],
    language: str,
    lineage_titles: Mapping[str, str],
) -> str:
    """Year · paper · lineage · key idea for every paper, oldest first; a
    lineage is named by its section's title (``lineage_titles``: lineage id
    -> the outline's title)."""
    label = labels(language)
    return _table(
        [label["year"], label["paper"], label["lineage"], label["key_idea"]],
        (
            [
                str(p.year or "-"),
                _paper_link(cards[p.id], p),
                lineage_titles[p.lineage],
                cards[p.id].key_idea,
            ]
            for p in plan.papers
            if p.id in cards
        ),
    )


def comparison_table(
    plan: SurveyPlan,
    cards: Mapping[str, PaperCard],
    language: str,
    lineage_titles: Mapping[str, str],
) -> str:
    """Paper · lineage · main results with their setting (checked numbers
    only) · limitation for the core papers, oldest first (the key ideas are
    in the timeline)."""
    label = labels(language)
    return _table(
        [label["paper"], label["lineage"], label["results"], label["limitations"]],
        (
            [
                _paper_link(cards[p.id], p),
                lineage_titles[p.lineage],
                "; ".join(_result_line(r) for r in cards[p.id].results[:TABLE_RESULTS]),
                cards[p.id].limitations[0] if cards[p.id].limitations else "",
            ]
            for p in plan.papers
            if p.role == "core" and p.id in cards
        ),
    )


_IMAGE_LINE = re.compile(r"^!\[[^\]]*\]\((?P<src>[^)\s]+)\)[ \t]*$", re.MULTILINE)


def credit_figures(
    markdown: str, owners: Mapping[str, PlanPaper], language: str
) -> str:
    """Under each paper figure, a line naming the paper it comes from
    ("*출처: [Rafailov et al., 2023](url)*"); ``owners`` maps figure URLs to
    their papers."""
    word = labels(language)["source"]

    def repl(m: re.Match[str]) -> str:
        paper = owners.get(m.group("src"))
        if paper is None:
            return m.group(0)
        link = f"[{md_link_text(author_year(paper))}]({md_link_url(paper.url)})"
        return f"{m.group(0)}\n\n*{word}: {link}*"

    return _IMAGE_LINE.sub(repl, markdown)


# Emphasis and math markers in plain reference text, as entities.
_PLAIN_TEXT_ENTITIES = str.maketrans({"*": "&#42;", "_": "&#95;", "$": "&#36;"})


def _plain(text: str) -> str:
    return md_link_text(" ".join(text.split())).translate(_PLAIN_TEXT_ENTITIES)


def reference_list(papers: Iterable[PlanPaper]) -> str:
    """``### References``: "Authors (year). [Title](url). Venue." per paper
    (the venue when known), markup characters in the text escaped."""
    lines = []
    for p in papers:
        line = (
            f"* {_plain(p.authors)} ({p.year or 'n.d.'}). "
            f"[{_plain(p.title)}]({md_link_url(p.url)})."
        )
        if p.venue:
            line += f" {_plain(p.venue).rstrip('.')}."
        lines.append(line)
    return f"- - -\n{REFERENCES_HEADING}\n" + "\n".join(lines) + "\n"


def further_reading(links: Sequence[tuple[str, str]], language: str) -> str:
    """The user's reference articles as ``* [title](url)`` lines under
    "### 참고 자료" (Korean) or "### Further reading", for after the
    reference list; "" without any. They are no papers of the survey: the
    validator does not ask the body to cite them."""
    if not links:
        return ""
    heading = FURTHER_READING_HEADINGS.get(language, FURTHER_READING_HEADINGS[""])
    return f"\n{heading}\n{link_list(links)}\n"
