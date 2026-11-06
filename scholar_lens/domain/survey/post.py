"""Deterministic parts of the paper-survey write stage.

``survey write`` reads a plan file (``survey_plan.SurveyPlan``) that the user
may have edited; the evidence engine writes the post. This module holds
everything that is code, not model judgment: re-reading the plan, citation
keys, figure credits and the reference list. Nothing here does I/O.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from ..markdown.links import link_list, md_link_text, md_link_url
from ..markdown.math import segments
from ..validation import FURTHER_READING_HEADINGS, REFERENCES_HEADING
from ..venues import reference_venue
from .plan import (
    PlanPaper,
    SourceRecord,
    SurveyInputError,
    SurveyPlan,
    merge_records,
    parse_paper_id,
)

_MATH_DELIMS = re.compile(r"^\$\$|\$\$$")


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
    # normalize_plan_data compares and hashes these before the schema check.
    for key, items, fields in (
        ("papers", data["papers"], ("lineage",)),
        ("lineages", data.get("lineages") or [], ("id",)),
        ("edges", data.get("edges") or [], ("from", "to", "citing", "cited")),
    ):
        for item in items:
            wrong = [f for f in fields if not isinstance(item.get(f, ""), str)]
            if wrong:
                raise SurveyInputError(
                    f"The plan's '{key}' entries need text in {', '.join(wrong)}."
                )
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


# ---- equations -----------------------------------------------------------------------


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", _MATH_DELIMS.sub("", text.strip()))


def equation_in_source(equation: str, source: str) -> bool:
    """Whether ``equation`` is LaTeX the source shows (whitespace and ``$$``
    delimiters ignored)."""
    return bool(equation.strip()) and _squash(equation) in re.sub(r"\s+", "", source)


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


# ---- figures and references -------------------------------------------------------

# The figure credit's word ("*출처: …*"), by output language.
_SOURCE_WORD: dict[str, str] = {"Korean": "출처"}
_IMAGE_LINE = re.compile(r"^!\[[^\]]*\]\((?P<src>[^)\s]+)\)[ \t]*$", re.MULTILINE)


def credit_figures(
    markdown: str, owners: Mapping[str, PlanPaper], language: str
) -> str:
    """Under each paper figure, a line naming the paper it comes from
    ("*출처: [Rafailov et al., 2023](url)*"); ``owners`` maps figure URLs to
    their papers."""
    word = _SOURCE_WORD.get(language, "Source")

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
        if venue := reference_venue(p.venue):
            line += f" {_plain(venue).rstrip('.')}."
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
