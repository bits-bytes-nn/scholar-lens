"""Deterministic parts of the paper-survey plan stage.

The plan stage turns a topic into an editable :class:`SurveyPlan`: a scope
call (LLM) proposes lineages, search queries and landmark titles; the
candidate step (APIs, see ``scholarly_index``) gathers paper records; this
module merges those records into one candidate per paper, links them by
citations, ranks them transparently, and checks the selection call's (LLM)
choice against the pool before the plan is built. Nothing here does I/O.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .citation_metadata import titles_match
from .constants import arxiv_id_from_reference

# Plan files carry schema_version 2 (``SurveyPlan``); bump it with the
# generator when the fields or their meaning change. Version 1 plans also
# had ``references`` (never read); they load with it ignored.
GENERATOR = "scholar-lens survey-plan/2"

# Shown to the selection call per candidate.
ABSTRACT_EXCERPT_CHARS = 400
# Notable candidates the plan lists as left out (for the user's edits).
MAX_EXCLUDED = 10
# Authors named before "et al.".
MAX_AUTHORS = 3

_DOI = re.compile(r"^10\.\d{4,9}/\S+$")
_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", re.IGNORECASE)
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_SLUG = r"^[a-z0-9][a-z0-9-]*$"


class SurveyInputError(ValueError):
    """The request cannot be planned (unknown paper id, nothing found)."""


class SurveyPlanError(ValueError):
    """A model output that cannot be used, even after the repair round."""


# ---- identifiers -----------------------------------------------------------


def normalize_doi(value: str | None) -> str | None:
    """The bare, lower-cased DOI in ``value`` (``https://doi.org/`` or
    ``doi:`` prefix stripped), or None when it is not a DOI."""
    if not value:
        return None
    doi = _DOI_PREFIX.sub("", value.strip()).lower()
    return doi if _DOI.match(doi) else None


def unversioned_arxiv_id(value: str | None) -> str | None:
    """The arXiv id in an id, ``arXiv:`` reference, arXiv URL or arXiv DOI,
    without its version."""
    doi = normalize_doi(value)
    if doi and (match := _ARXIV_DOI.match(doi)):
        value = match.group(1)
    arxiv_id = arxiv_id_from_reference(value)
    return re.sub(r"v\d+$", "", arxiv_id) if arxiv_id else None


def parse_paper_id(value: str) -> str:
    """A user-supplied paper (arXiv id or URL, DOI or doi.org URL) as a plan
    id: the unversioned arXiv id, else the bare DOI."""
    if arxiv_id := unversioned_arxiv_id(value):
        return arxiv_id
    if doi := normalize_doi(value):
        return doi
    raise SurveyInputError(
        f"Not an arXiv id, arXiv URL or DOI: {value!r} (use e.g. 2305.18290, "
        "https://arxiv.org/abs/2305.18290 or 10.18653/v1/2024.emnlp-main.626)."
    )


# ---- source records and merged candidates ----------------------------------

Provider = Literal["openalex", "arxiv"]


@dataclass
class SourceRecord:
    """One paper as one provider returned it."""

    provider: Provider
    title: str
    authors: list[str] = field(default_factory=list)
    published: date | None = None
    venue: str | None = None
    arxiv_id: str | None = None
    # A DOI other than the arXiv one (``arxiv_id`` holds that).
    doi: str | None = None
    openalex_id: str | None = None
    citation_count: int | None = None
    abstract: str | None = None
    # OpenAlex ids of the works this record cites.
    references: list[str] = field(default_factory=list)
    # How the pipeline found it: "query:<q>", "landmark:<title>", "required",
    # "reference-of:<id>", "cites:<id>".
    found_by: set[str] = field(default_factory=set)

    def keys(self) -> set[str]:
        keys = set()
        if self.arxiv_id:
            keys.add(f"arxiv:{self.arxiv_id}")
        if self.doi:
            keys.add(f"doi:{self.doi}")
        if self.openalex_id:
            keys.add(f"openalex:{self.openalex_id}")
        return keys


@dataclass
class Candidate:
    """One paper merged from every record of it."""

    id: str
    title: str
    authors: list[str]
    published: date | None
    venue: str | None
    arxiv_id: str | None
    doi: str | None
    abstract: str | None
    openalex_citations: int | None
    keys: set[str]
    # Identity keys of the works it cites ("openalex:W…", "arxiv:…", "doi:…").
    reference_keys: set[str]
    found_by: set[str]
    semantic_scholar_citations: int | None = None
    required: bool = False

    @property
    def year(self) -> int | None:
        return self.published.year if self.published else None

    @property
    def citation_count(self) -> int | None:
        counts = [
            c
            for c in (self.openalex_citations, self.semantic_scholar_citations)
            if c is not None
        ]
        return max(counts) if counts else None

    @property
    def search_hits(self) -> int:
        """Distinct topic queries that returned the paper."""
        return sum(1 for f in self.found_by if f.startswith("query:"))

    @property
    def landmark(self) -> bool:
        """Whether its title matched a landmark title the scope proposed."""
        return any(f.startswith("landmark:") for f in self.found_by)

    @property
    def url(self) -> str:
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return f"https://openalex.org/{self.id}"

    @property
    def authors_label(self) -> str:
        names = ", ".join(self.authors[:MAX_AUTHORS])
        return f"{names} et al." if len(self.authors) > MAX_AUTHORS else names


def _arxiv_ids(records: Iterable[SourceRecord]) -> set[str]:
    return {r.arxiv_id for r in records if r.arxiv_id}


def merge_records(records: Sequence[SourceRecord]) -> list[Candidate]:
    """One candidate per paper. Records join when they share an arXiv id, a
    DOI or an OpenAlex id, or when their titles match
    (``citation_metadata.titles_match``); a record that matches several
    papers joins them into one. Never when that would put two different arXiv
    ids in one paper. Candidates follow the order of each paper's first
    record, so the result is deterministic for a given input order."""
    # Cluster id (its first record's position) -> (position, record) pairs.
    clusters: dict[int, list[tuple[int, SourceRecord]]] = {}
    by_key: dict[str, int] = {}
    for n, record in enumerate(records):
        by_id = {by_key[k] for k in record.keys() if k in by_key}
        by_title = {
            i
            for i, members in clusters.items()
            if i not in by_id
            and any(titles_match(record.title, r.title) for _, r in members)
        }
        members, arxiv_ids = [(n, record)], _arxiv_ids([record])
        for i in sorted(by_id | by_title):
            joined = _arxiv_ids(r for _, r in clusters[i])
            if len(arxiv_ids | joined) > 1:
                continue
            arxiv_ids |= joined
            members += clusters.pop(i)
        members.sort(key=lambda m: m[0])
        target = members[0][0]
        clusters[target] = members
        for _, r in members:
            for key in r.keys():
                by_key[key] = target
    return [_candidate([r for _, r in clusters[i]]) for i in sorted(clusters)]


def _first[T](values: Iterable[T | None]) -> T | None:
    return next((v for v in values if v), None)


def _candidate(cluster: list[SourceRecord]) -> Candidate:
    arxiv = [r for r in cluster if r.provider == "arxiv"]
    openalex = [r for r in cluster if r.provider == "openalex"]
    arxiv_id = _first(r.arxiv_id for r in cluster)
    doi = _first(r.doi for r in cluster)
    openalex_ids = sorted(r.openalex_id for r in openalex if r.openalex_id)
    # arXiv's record is the paper as its authors posted it: its title,
    # authors, abstract and first-version date win. OpenAlex records of
    # proceedings often carry only the year (YYYY-01-01).
    published = _first(r.published for r in arxiv) or min(
        (r.published for r in openalex if r.published), default=None
    )
    counts = [r.citation_count for r in openalex if r.citation_count is not None]
    return Candidate(
        id=arxiv_id or doi or openalex_ids[0],
        title=(_first(r.title for r in arxiv) or cluster[0].title).strip(),
        authors=_first(r.authors for r in [*arxiv, *openalex]) or [],
        published=published,
        venue=_first(r.venue for r in openalex) or ("arXiv" if arxiv_id else None),
        arxiv_id=arxiv_id,
        doi=doi,
        abstract=_first(r.abstract for r in [*arxiv, *openalex]),
        openalex_citations=max(counts) if counts else None,
        keys=set().union(*(r.keys() for r in cluster)),
        reference_keys={f"openalex:{w}" for r in cluster for w in r.references},
        found_by=set().union(*(r.found_by for r in cluster)),
    )


@dataclass
class CitationData:
    """Semantic Scholar's view of one paper: it counts arXiv-era ML papers
    far more fully than OpenAlex and lists their references."""

    citation_count: int | None
    venue: str | None
    # Identity keys ("arxiv:…", "doi:…") of the works it cites.
    reference_keys: set[str] = field(default_factory=set)


def semantic_scholar_id(candidate: Candidate) -> str | None:
    if candidate.arxiv_id:
        return f"ARXIV:{candidate.arxiv_id}"
    return f"DOI:{candidate.doi}" if candidate.doi else None


def add_citation_data(candidate: Candidate, data: CitationData | None) -> None:
    """Fold Semantic Scholar's count, venue and references into a candidate
    (its venue names the proceedings more cleanly than OpenAlex's)."""
    if data is None:
        return
    candidate.semantic_scholar_citations = data.citation_count
    if data.venue:
        candidate.venue = data.venue
    candidate.reference_keys |= data.reference_keys


# ---- citation edges and ranking ---------------------------------------------


def citation_edges(candidates: Sequence[Candidate]) -> list[tuple[str, str]]:
    """``(citing, cited)`` candidate-id pairs within ``candidates``, sorted.
    A key several candidates hold (a DOI two arXiv papers share, so
    :func:`merge_records` kept them apart) names none of them."""
    holders: dict[str, set[str]] = {}
    for c in candidates:
        for key in c.keys:
            holders.setdefault(key, set()).add(c.id)
    owner = {key: ids.pop() for key, ids in holders.items() if len(ids) == 1}
    edges = {
        (c.id, owner[key])
        for c in candidates
        for key in c.reference_keys
        if key in owner and owner[key] != c.id
    }
    return sorted(edges)


def in_degrees(edges: Iterable[tuple[str, str]]) -> dict[str, int]:
    degrees: dict[str, int] = {}
    for _, cited in edges:
        degrees[cited] = degrees.get(cited, 0) + 1
    return degrees


def citations_per_year(candidate: Candidate, current_year: int) -> float:
    if candidate.citation_count is None or candidate.year is None:
        return 0.0
    return candidate.citation_count / max(1, current_year - candidate.year + 1)


def rank_candidates(
    candidates: Sequence[Candidate], edges: Sequence[tuple[str, str]], *, today: date
) -> list[Candidate]:
    """Most supported first: topic evidence (in-pool citations received plus
    distinct topic queries that found the paper), then citations per year
    since publication, then id (a stable tie-break)."""
    degrees = in_degrees(edges)
    return sorted(
        candidates,
        key=lambda c: (
            -(degrees.get(c.id, 0) + c.search_hits),
            -citations_per_year(c, today.year),
            c.id,
        ),
    )


def in_time_range(candidate: Candidate, since: int | None, until: int | None) -> bool:
    """Whether a candidate's year lies in the range; a required paper and
    any paper when no range is set always do."""
    if candidate.required or (since is None and until is None):
        return True
    if candidate.year is None:
        return False
    return (since is None or candidate.year >= since) and (
        until is None or candidate.year <= until
    )


def shortlist(ranked: Sequence[Candidate], size: int) -> list[Candidate]:
    """The required papers and the landmark-title matches, then the
    best-ranked others, ``size`` in all (or more when the first two alone
    exceed it), in rank order."""
    sure = {c.id for c in ranked if c.required or c.landmark}
    room = max(0, size - len(sure))
    others = [c.id for c in ranked if c.id not in sure][:room]
    keep = sure | set(others)
    return [c for c in ranked if c.id in keep]


# ---- model outputs ------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScopeLineage(_Strict):
    id: str = Field(pattern=_SLUG, max_length=40)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class TimeRange(_Strict):
    since: int | None = Field(default=None, ge=1900, le=2100)
    until: int | None = Field(default=None, ge=1900, le=2100)

    @model_validator(mode="after")
    def _ordered(self) -> TimeRange:
        if self.since and self.until and self.since > self.until:
            raise ValueError(f"since ({self.since}) is after until ({self.until})")
        return self


class SurveyScope(_Strict):
    """The scope call's output."""

    topic: str = Field(min_length=1)
    lineages: list[ScopeLineage] = Field(min_length=2, max_length=6)
    queries: list[str] = Field(min_length=4, max_length=8)
    # Titles from the model's memory: search queries only, never facts.
    landmarks: list[str] = Field(default_factory=list, max_length=10)


class SelectedPaper(_Strict):
    id: str
    lineage: str
    role: Literal["core", "supporting"]
    reason: str = Field(min_length=1)


class ExcludedPaper(_Strict):
    id: str
    reason: str = Field(min_length=1)


class SurveySelection(_Strict):
    """The selection call's output."""

    paper_count_reason: str = Field(min_length=1)
    lineages: list[ScopeLineage] = Field(min_length=1)
    papers: list[SelectedPaper] = Field(min_length=1)
    excluded: list[ExcludedPaper] = Field(default_factory=list)


def parse_tagged_json[M: BaseModel](raw: str, tag: str, model: type[M]) -> M:
    """Parse ``<tag>{json}</tag>`` (the last such block, fences allowed) into
    ``model``; ``SurveyPlanError`` when it is missing or invalid."""
    blocks = re.findall(rf"<{tag}>(.*?)</{tag}>", raw, re.DOTALL)
    body = _JSON_FENCE.sub("", (blocks[-1] if blocks else raw).strip()).strip()
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        raise SurveyPlanError(f"no JSON object inside <{tag}>")
    try:
        return model.model_validate(json.loads(body[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as e:
        raise SurveyPlanError(str(e)) from e


NO_CORE_PAPER = "no paper has role 'core'; mark at least one paper core"


def check_selection(
    selection: SurveySelection,
    pool: Sequence[Candidate],
    *,
    min_papers: int,
    max_papers: int,
) -> list[str]:
    """Every problem that makes ``selection`` unusable (empty when fine):
    ids outside the pool, duplicates, a missing required paper, a count
    outside the bounds, no core paper, an unknown lineage. A pool smaller
    than ``min_papers`` lowers the lower bound to the pool size. An excluded
    id outside the pool is not a problem: :func:`build_plan` drops it."""
    ids = {c.id for c in pool}
    chosen = [p.id for p in selection.papers]
    problems = [f"unknown paper id {i!r}" for i in chosen if i not in ids]
    duplicates = sorted({i for i in chosen if chosen.count(i) > 1})
    problems += [f"paper {i!r} is selected twice" for i in duplicates]
    missing = sorted(c.id for c in pool if c.required and c.id not in chosen)
    problems += [f"required paper {i!r} is not selected" for i in missing]
    low = min(min_papers, len(ids))
    count = len(set(chosen) & ids)
    if not low <= count <= max_papers:
        problems.append(
            f"{count} papers selected; select between {low} and {max_papers}"
        )
    if not any(p.role == "core" for p in selection.papers):
        problems.append(NO_CORE_PAPER)
    lineage_ids = [lineage.id for lineage in selection.lineages]
    if len(set(lineage_ids)) != len(lineage_ids):
        problems.append("lineage ids are not unique")
    problems += [
        f"paper {p.id!r} has unknown lineage {p.lineage!r}"
        for p in selection.papers
        if p.lineage not in lineage_ids
    ]
    return problems


# ---- the plan file -------------------------------------------------------------


class PlanLineage(_Strict):
    id: str = Field(pattern=_SLUG)
    name: str = Field(min_length=1)
    description: str


class PlanPaper(_Strict):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: str = ""
    year: int | None = None
    date: str | None = None
    venue: str | None = None
    citation_count: int | None = None
    url: str
    lineage: str
    role: Literal["core", "supporting"]
    reason: str = ""


class PlanEdge(_Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    citing: str = Field(alias="from")
    cited: str = Field(alias="to")


class PlanExclusion(_Strict):
    id: str
    title: str
    year: int | None = None
    reason: str = ""


class SurveyPlan(_Strict):
    """The plan file (``plan.json``): what the write stage reads. Users may
    edit it (drop, add or move papers, rename lineages); the write stage
    validates it again."""

    schema_version: Literal[2] = 2
    generator: str = GENERATOR
    created_at: datetime
    topic: str = Field(min_length=1)
    scope: str
    # The user's year range (``--since``/``--until``); open ends are None.
    time_range: TimeRange
    paper_count_reason: str = ""
    lineages: list[PlanLineage] = Field(min_length=1)
    papers: list[PlanPaper] = Field(min_length=1)
    edges: list[PlanEdge] = Field(default_factory=list)
    excluded: list[PlanExclusion] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _upgrade(cls, data: object) -> object:
        """A version 1 plan is a version 2 plan plus ``references``."""
        if isinstance(data, dict) and data.get("schema_version") == 1:
            data = {k: v for k, v in data.items() if k != "references"}
            data["schema_version"] = 2
        return data

    @model_validator(mode="after")
    def _consistent(self) -> SurveyPlan:
        paper_ids = [p.id for p in self.papers]
        lineage_ids = [lineage.id for lineage in self.lineages]
        problems = []
        for label, values in (("paper", paper_ids), ("lineage", lineage_ids)):
            problems += sorted(
                {f"duplicate {label} id {v!r}" for v in values if values.count(v) > 1}
            )
        problems += [
            f"paper {p.id!r} has unknown lineage {p.lineage!r}"
            for p in self.papers
            if p.lineage not in lineage_ids
        ]
        problems += [
            f"lineage {i!r} has no papers"
            for i in lineage_ids
            if all(p.lineage != i for p in self.papers)
        ]
        problems += [
            f"edge {e.citing!r} -> {e.cited!r} names a paper not in the plan"
            for e in self.edges
            if e.citing not in paper_ids or e.cited not in paper_ids
        ]
        if not any(p.role == "core" for p in self.papers):
            problems.append(NO_CORE_PAPER)
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, by_alias=True) + "\n"


def chronological(papers: Iterable[Candidate]) -> list[Candidate]:
    """Oldest first; a paper without a date goes last; ties by id."""
    return sorted(
        papers, key=lambda c: (c.published is None, c.published or date.min, c.id)
    )


def build_plan(
    *,
    topic: str,
    scope: SurveyScope,
    selection: SurveySelection,
    pool: Sequence[Candidate],
    edges: Sequence[tuple[str, str]],
    time_range: TimeRange,
    sources: Sequence[str],
    notes: Sequence[str],
    created_at: datetime,
) -> SurveyPlan:
    """The plan from a selection :func:`check_selection` accepted: papers in
    chronological order with their pool metadata, citation edges among them,
    lineages that hold at least one paper, at most ``MAX_EXCLUDED`` notable
    exclusions (an excluded id outside ``pool`` is dropped with a note)."""
    by_id = {c.id: c for c in pool}
    picks = {p.id: p for p in selection.papers}
    papers = [
        PlanPaper(
            id=c.id,
            title=c.title,
            authors=c.authors_label,
            year=c.year,
            date=c.published.isoformat() if c.published else None,
            venue=c.venue,
            citation_count=c.citation_count,
            url=c.url,
            lineage=picks[c.id].lineage,
            role=picks[c.id].role,
            reason=picks[c.id].reason,
        )
        for c in chronological(by_id[i] for i in picks)
    ]
    used = {p.lineage for p in papers}
    excluded = [e for e in selection.excluded if e.id in by_id and e.id not in picks]
    unknown = sorted({e.id for e in selection.excluded if e.id not in by_id})
    if unknown:
        notes = [
            *notes,
            f"Dropped {len(unknown)} excluded id(s) not among the candidates: "
            + ", ".join(unknown),
        ]
    return SurveyPlan(
        created_at=created_at,
        topic=topic,
        scope=scope.topic,
        time_range=time_range,
        paper_count_reason=selection.paper_count_reason,
        lineages=[
            PlanLineage(
                id=lineage.id, name=lineage.name, description=lineage.description
            )
            for lineage in selection.lineages
            if lineage.id in used
        ],
        papers=papers,
        edges=[
            PlanEdge.model_validate({"from": a, "to": b})
            for a, b in edges
            if a in picks and b in picks
        ],
        excluded=[
            PlanExclusion(
                id=e.id,
                title=by_id[e.id].title,
                year=by_id[e.id].year,
                reason=e.reason,
            )
            for e in excluded[:MAX_EXCLUDED]
        ],
        sources=list(sources),
        notes=list(notes),
    )


def shortlist_digest(
    shortlist_: Sequence[Candidate], edges: Sequence[tuple[str, str]], today: date
) -> str:
    """One block per candidate for the selection call: id, year, title,
    venue, citations, in-pool citations received, topic queries that found
    it, required and landmark flags and an abstract excerpt."""
    degrees = in_degrees(edges)
    blocks = []
    for c in shortlist_:
        cites = "unknown" if c.citation_count is None else str(c.citation_count)
        flags = (" [REQUIRED]" if c.required else "") + (
            " [LANDMARK]" if c.landmark else ""
        )
        abstract = " ".join((c.abstract or "").split())[:ABSTRACT_EXCERPT_CHARS]
        blocks.append(
            f"- id: {c.id}{flags}\n  title: {c.title}\n  year: {c.year or 'unknown'}"
            f" | venue: {c.venue or 'unknown'} | citations: {cites}"
            f" (~{citations_per_year(c, today.year):.0f}/yr)"
            f" | cited by {degrees.get(c.id, 0)} pool papers"
            f" | found by {c.search_hits} topic queries\n"
            f"  abstract: {abstract or '(none)'}"
        )
    return "\n".join(blocks)
