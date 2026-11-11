"""Coverage: the key targets a post must state (docs/engine.md, 4.6).

A post kind names where its targets come from (``PostSpec.coverage``): an
extraction from some of its sources, one target per document a user chose,
or its plan. The reader records their evidence first, the outline maps each
target to that evidence, the TL;DR cites the headline ones, and the body is
checked at the end for targets it never cites. What a target means is the
model's judgement; mapping, counting and checking are code here.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from scholar_lens.domain.evidence import Evidence, SourceDocument
from scholar_lens.domain.evidence_draft import quote_for_prompt
from scholar_lens.domain.post_spec import CoverageSpec

# How many targets one extraction answer may hold: a bound of the answer's
# schema against a runaway list, above every spec's count (which caps it
# again in code).
_MAX_EXTRACTED = 30


class CoverageTarget(BaseModel):
    """One thing the post must state. ``source``: the id of the source it
    comes from (a document, a paper, a lineage of a plan)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str
    source: str
    kind: str = ""
    # Whether the TL;DR must state it.
    headline: bool = False
    # The sources whose evidence can state it, when ``source`` is not one
    # (a survey lineage: its papers); empty: ``source`` itself.
    evidence_sources: list[str] = Field(default_factory=list)

    def stated_by(self) -> set[str]:
        """The source ids whose evidence can state this target."""
        return set(self.evidence_sources) or {self.source}


class ExtractedTarget(BaseModel):
    statement: str = Field(min_length=1)
    # The id of the source passage it comes from, as listed.
    source: str = ""
    kind: str = ""


class ExtractedTargets(BaseModel):
    """The key-target extraction's answer, most important first."""

    targets: list[ExtractedTarget] = Field(
        default_factory=list, max_length=_MAX_EXTRACTED
    )


def extraction_documents(
    spec: CoverageSpec, documents: Iterable[SourceDocument]
) -> list[SourceDocument]:
    """The documents the spec's extraction reads (none when it extracts
    nothing)."""
    if spec.count[1] <= 0:
        return []
    return [
        d for d in documents if not spec.authorities or d.authority in spec.authorities
    ]


def _named(heading: str, words: Sequence[str]) -> bool:
    lowered = heading.lower()
    return any(w in lowered for w in words)


def extraction_text(
    spec: CoverageSpec, documents: Sequence[SourceDocument], budget: int
) -> str:
    """The source text the extraction reads, each document under its id:
    the sections whose heading names one of ``spec.sections`` (the first
    two when none does), or, when the spec names none, the document's
    headings and opening. Each document gets an equal share of ``budget``
    characters; the text is defanged as prompt data.

    Args:
        spec: The post kind's coverage spec.
        documents: The documents to read (:func:`extraction_documents`).
        budget: The characters of source text in all.

    Returns:
        The text, empty without documents.
    """
    if not documents or budget <= 0:
        return ""
    share = max(budget // len(documents), 1)
    parts: list[str] = []
    for d in documents:
        if spec.sections:
            chosen = [s for s in d.sections if _named(s.heading, spec.sections)]
            chosen = chosen or list(d.sections[:2])
            text = "\n\n".join(f"{s.heading}\n{s.text}".strip() for s in chosen)
        else:
            headings = "; ".join(s.heading for s in d.sections if s.heading)
            opening = d.sections[0].text[: spec.opening_chars] if d.sections else ""
            text = f"Headings: {headings}\n{opening}".strip()
        header = f"[{d.paper_id}] {d.title}" + (
            f" ({d.authority})" if d.authority else ""
        )
        parts.append(f"{header}\n{quote_for_prompt(text[:share])}")
    return "\n\n".join(parts)


def targets_from(
    answer: ExtractedTargets,
    spec: CoverageSpec,
    sources: Sequence[str],
    *,
    start: int = 1,
) -> list[CoverageTarget]:
    """The extraction's answer as targets numbered from ``T{start}``: at
    most ``spec.count[1]``, empty statements dropped, an unknown kind read
    as the spec's first, an unknown source as the first of ``sources``, and
    the first targets of each headline kind (up to its ``headline``)
    marked as headline.

    Args:
        answer: The extraction's answer.
        spec: The post kind's coverage spec.
        sources: The ids of the documents the extraction read.
        start: The number of the first target.

    Returns:
        The targets, in the answer's order.
    """
    kinds = [k.name for k in spec.kinds]
    headline = spec.headline_kinds
    taken: dict[str, int] = {}
    out: list[CoverageTarget] = []
    for item in answer.targets:
        if len(out) >= spec.count[1]:
            break
        statement = " ".join(item.statement.split())
        if not statement:
            continue
        kind = item.kind if item.kind in kinds else (kinds[0] if kinds else "")
        source = (
            item.source if item.source in sources else (sources[0] if sources else "")
        )
        taken[kind] = taken.get(kind, 0) + 1
        out.append(
            CoverageTarget(
                id=f"T{start + len(out)}",
                statement=statement,
                source=source,
                kind=kind,
                headline=taken[kind] <= headline.get(kind, 0),
            )
        )
    return out


def document_targets(
    spec: CoverageSpec, documents: Iterable[SourceDocument], *, start: int = 1
) -> list[CoverageTarget]:
    """One target per document whose authority is in ``spec.per_document``
    (the post uses every document the user chose), numbered from
    ``T{start}``."""
    out: list[CoverageTarget] = []
    for d in documents:
        if d.authority and d.authority in spec.per_document:
            out.append(
                CoverageTarget(
                    id=f"T{start + len(out)}",
                    statement=f"{spec.document_kind.meaning}: {d.title}",
                    source=d.paper_id,
                    kind=spec.document_kind.name,
                )
            )
    return out


def target_lines(targets: Sequence[CoverageTarget]) -> str:
    """The targets as prompt lines: id, kind, source and statement."""
    return "\n".join(
        f"- {t.id} ({t.kind or 'target'}{', headline' if t.headline else ''}; "
        f"source {t.source}): {t.statement}"
        for t in targets
    )


def repair_coverage(
    coverage: Mapping[str, Sequence[str]],
    targets: Sequence[CoverageTarget],
    known: Collection[str],
    duplicates: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """An outline's coverage map with what code can fix: unknown target ids
    and evidence ids dropped, a duplicate evidence id replaced by the id
    that keeps its fact, each id once.

    Args:
        coverage: Target id -> evidence ids, as the outline gave them.
        targets: The run's targets.
        known: The evidence ids of the ledger.
        duplicates: Duplicate evidence id -> the id kept.

    Returns:
        The map, in the targets' order.
    """
    duplicates = duplicates or {}
    out: dict[str, list[str]] = {}
    for t in targets:
        ids = [duplicates.get(e, e) for e in coverage.get(t.id, ())]
        out[t.id] = [e for e in dict.fromkeys(ids) if e in known]
    return out


def without_evidence(
    coverage: Mapping[str, Sequence[str]], ids: Collection[str]
) -> dict[str, list[str]]:
    """``coverage`` without the evidence ``ids`` (evidence the outline lost)."""
    return {t: [e for e in mapped if e not in ids] for t, mapped in coverage.items()}


def unassigned(
    targets: Sequence[CoverageTarget],
    coverage: Mapping[str, Sequence[str]],
    used: Collection[str],
) -> list[CoverageTarget]:
    """The targets no evidence the outline uses (``used``: given to a
    section or cited elsewhere) is mapped to."""
    return [t for t in targets if not set(coverage.get(t.id, ())) & set(used)]


def unsatisfiable(
    targets: Sequence[CoverageTarget], ledger: Iterable[Evidence]
) -> list[CoverageTarget]:
    """The targets no evidence of ``ledger`` can state: none of the sources
    that state them (:meth:`CoverageTarget.stated_by`) has evidence."""
    have = {e.paper_id for e in ledger}
    return [t for t in targets if not t.stated_by() & have]


def coverage_problems(
    targets: Sequence[CoverageTarget],
    coverage: Mapping[str, Sequence[str]],
    used: Collection[str],
    ledger: Iterable[Evidence] | None = None,
) -> list[str]:
    """One outline problem per target :func:`unassigned` finds, worded for
    the outline writer; with ``ledger``, not for the targets it cannot
    state (:func:`unsatisfiable`): no outline can map them."""
    hopeless = (
        {t.id for t in unsatisfiable(targets, ledger)} if ledger is not None else set()
    )
    return [
        f"key target {t.id} ({t.statement[:160]}) has no evidence in `coverage` "
        "that a section uses: map it to the ids of the evidence that states it "
        "and give that evidence to a section"
        for t in unassigned(targets, coverage, used)
        if t.id not in hopeless
    ]


def headline_evidence(
    targets: Sequence[CoverageTarget],
    coverage: Mapping[str, Sequence[str]],
    available: Collection[str],
) -> list[tuple[CoverageTarget, list[str]]]:
    """(target, its mapped evidence among ``available``) of each headline
    target that has such evidence: what the TL;DR must cite."""
    out = []
    for t in targets:
        if t.headline and (
            ids := [e for e in coverage.get(t.id, ()) if e in available]
        ):
            out.append((t, ids))
    return out


def headline_problems(
    headline: Sequence[tuple[CoverageTarget, Sequence[str]]], cited: Collection[str]
) -> list[str]:
    """One TL;DR problem per headline target none of whose evidence the
    TL;DR (``cited``) cites."""
    have = set(cited)
    return [
        f"the TL;DR does not state the key target {t.id} ({t.statement[:160]}): "
        f"state it and cite {', '.join(ids)}"
        for t, ids in headline
        if not have & set(ids)
    ]


def uncited(
    targets: Sequence[CoverageTarget],
    coverage: Mapping[str, Sequence[str]],
    cited: Collection[str],
) -> list[CoverageTarget]:
    """The targets none of whose mapped evidence the body cites."""
    have = set(cited)
    return [t for t in targets if not set(coverage.get(t.id, ())) & have]


# A source section the readers recorded at least this many items from is
# part of what the post covers.
MIN_SECTION_EVIDENCE = 2


def unused_source_sections(
    documents: Iterable[SourceDocument],
    ledger: Iterable[Evidence],
    used: Collection[str],
    same: Mapping[str, str],
    *,
    min_items: int = MIN_SECTION_EVIDENCE,
) -> list[str]:
    """Source sections the readers found worth ``min_items`` or more pieces
    of evidence from, none of which the outline uses (an item counts as used
    when it, or the item it duplicates, is): what a post built only from
    its outline would leave out of the source. One problem per section, for
    the outline writer."""
    # Each fact once: a duplicate counts as the item it repeats (the one
    # the outline was shown).
    recorded: dict[tuple[str, str], list[str]] = {}
    for e in ledger:
        kept = same.get(e.id, e.id)
        ids = recorded.setdefault((e.paper_id, e.anchor), [])
        if kept not in ids:
            ids.append(kept)
    problems = []
    for document in documents:
        for section in document.sections:
            ids = recorded.get((document.paper_id, section.anchor), [])
            if len(ids) < min_items or any(i in used for i in ids):
                continue
            name = section.heading or section.anchor
            problems.append(
                f"source section {section.anchor} '{name[:80]}' of {document.paper_id} "
                f"has {len(ids)} evidence items ({', '.join(ids[:6])}) and no "
                "section uses any: give its main point to the section it belongs to"
            )
    return problems
