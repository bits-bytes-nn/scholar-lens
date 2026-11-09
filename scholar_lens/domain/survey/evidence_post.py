"""The evidence-based survey (docs/redesign.md, 3.2): an outline that gives
each piece of evidence to one section, the code-built tables, and the
deterministic checks of a section draft whose sentences cite evidence as
``[[E12]]``."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scholar_lens.domain.coverage import CoverageTarget
from scholar_lens.domain.evidence import EvidenceLedger, canonical_ids
from scholar_lens.domain.evidence_draft import MARKER, block_lines, cited_ids
from scholar_lens.domain.glossary import GlossaryTerm
from scholar_lens.domain.lengths import DEPTH_CHARS, scaled_targets
from scholar_lens.domain.markdown.lint import ends_in_plain_declarative
from scholar_lens.domain.markdown.tables import escape_cell_pipes
from scholar_lens.domain.post_spec import (
    Depth,
    ownership_problems,
    structure_problems,
)
from scholar_lens.domain.specs.survey import SURVEY_SPEC
from scholar_lens.domain.survey.plan import SurveyPlan

SectionKind = Literal[
    "introduction", "lineage", "cross_cutting", "comparison", "insights", "conclusion"
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# A survey section: the common section fields (``post_spec.SectionBase``) and
# the papers a lineage covers, in the order its schema has always shown them
# (so not a subclass). No docstring: it would enter the outline's schema.
class OutlineSection(_Strict):
    kind: SectionKind
    # A claim, not a topic label ("깊어질수록 흐려지는 노드 표현").
    title: str = Field(min_length=1)
    # The one sentence the section argues.
    thesis: str = Field(min_length=1)
    # The reader's question the section answers.
    question: str = ""
    lineage: str | None = None
    papers: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    depth: Depth = "standard"


class TableRow(_Strict):
    paper: str
    results: list[str] = Field(default_factory=list, max_length=2)
    limitation: str | None = None


class EvidenceOutline(_Strict):
    title: str = Field(min_length=1)
    glossary: list[GlossaryTerm] = Field(default_factory=list, max_length=20)
    # One concrete case threaded through the post, when the topic has one.
    running_example: str = ""
    # Paper id -> its short name in tables ("GraphSAGE", "GCN").
    short_names: dict[str, str] = Field(default_factory=dict)
    # Lineage id -> a short name for the tables.
    lineage_names: dict[str, str] = Field(default_factory=dict)
    # Paper id -> its idea in one noun phrase, for the timeline.
    ideas: dict[str, str] = Field(default_factory=dict)
    # The comparison table: core papers, their key results and a limitation
    # (evidence ids).
    table: list[TableRow] = Field(default_factory=list)
    sections: list[OutlineSection] = Field(min_length=3)
    # Groups of evidence ids that state one fact in different words; code
    # keeps one of each, so the post states the fact once.
    same_facts: list[list[str]] = Field(default_factory=list, max_length=60)
    # Key target id -> the evidence ids that state it (docs/engine.md, 4.6).
    coverage: dict[str, list[str]] = Field(default_factory=dict)


def plan_targets(plan: SurveyPlan) -> list[CoverageTarget]:
    """The survey's key targets, derived from its plan in code (no model
    call): every core paper (its title and why it is in the survey), then
    every lineage (its name and description). Their kinds are the survey
    spec's coverage kinds."""
    paper, lineage = (k.name for k in SURVEY_SPEC.coverage.kinds)
    found: list[tuple[str, str, str]] = [
        (f"{p.title}: {p.reason}" if p.reason else p.title, p.id, paper)
        for p in plan.papers
        if p.role == "core"
    ]
    found += [
        (
            f"{lin.name}: {lin.description}" if lin.description else lin.name,
            lin.id,
            lineage,
        )
        for lin in plan.lineages
    ]
    # A lineage is stated by its papers' evidence.
    members: dict[str, list[str]] = {}
    for p in plan.papers:
        members.setdefault(p.lineage, []).append(p.id)
    return [
        CoverageTarget(
            id=f"T{n}",
            statement=statement,
            source=source,
            kind=kind,
            evidence_sources=members.get(source, []) if kind == lineage else [],
        )
        for n, (statement, source, kind) in enumerate(found, 1)
    ]


def outline_problems(
    outline: EvidenceOutline,
    plan: SurveyPlan,
    ledger: EvidenceLedger,
    *,
    title_max_length: int,
    merged: Mapping[str, str] | None = None,
) -> list[str]:
    """What makes an outline unusable: the section order, a lineage without
    its one section, unknown papers or evidence, evidence given twice or to
    a section that does not cover its paper, papers without evidence, and
    missing table, timeline or name entries."""
    problems: list[str] = []
    papers = {p.id: p for p in plan.papers}
    lineages = {lin.id for lin in plan.lineages}
    problems += structure_problems(
        SURVEY_SPEC,
        [s.kind for s in outline.sections],
        title=outline.title,
        title_max_length=title_max_length,
    )
    covered = [s.lineage for s in outline.sections if s.kind == "lineage"]
    for lin in sorted(lineages):
        if covered.count(lin) != 1:
            problems.append(f"lineage {lin!r} needs exactly one lineage section")
    if stray := set(covered) - lineages:
        problems.append(f"unknown lineages: {', '.join(sorted(map(str, stray)))}")
    problems += ownership_problems(SURVEY_SPEC, outline.sections, ledger)
    owner = {eid for s in outline.sections for eid in s.evidence}
    in_sections: dict[str, set[str]] = {pid: set() for pid in papers}
    for s in outline.sections:
        if strangers := [p for p in s.papers if p not in papers]:
            problems.append(f"section {s.title!r} names unknown papers: {strangers}")
        if s.kind == "lineage":
            expected = {p.id for p in plan.papers if p.lineage == s.lineage}
            if missing := sorted(expected - set(s.papers)):
                problems.append(f"lineage section {s.title!r} leaves out {missing}")
        for eid in s.evidence:
            item = ledger.get(eid)
            if item is None:
                continue
            if s.kind == "lineage" and item.paper_id not in s.papers:
                problems.append(
                    f"{eid} is about {item.paper_id}, which {s.title!r} does not cover"
                )
            if item.paper_id in in_sections:
                in_sections[item.paper_id].add(eid)

    # A paper's share counts its evidence in sections and its table row,
    # each item once; a core paper's sections still need one item when it
    # has evidence the table cannot hold (its method, figures, findings).
    # Facts, not items: evidence merged as the same fact counts once.
    def fact(eid: str) -> str:
        return (merged or {}).get(eid, eid)

    shares = {pid: {fact(i) for i in ids} for pid, ids in in_sections.items()}
    for row in outline.table:
        if row.paper in shares:
            shares[row.paper] |= {
                fact(i)
                for i in [*row.results, *([row.limitation] if row.limitation else [])]
                if (item := ledger.get(i)) and item.paper_id == row.paper
            }
    for pid, ids in shares.items():
        need = 2 if papers[pid].role == "core" else 1
        facts = {fact(e.id) for e in ledger.of_paper(pid)}
        if len(ids) < need and len(facts) >= need:
            problems.append(
                f"{pid} needs at least {need} pieces of evidence in sections "
                "or its table row"
            )
        prose_only = any(
            e.kind not in ("number", "limitation") for e in ledger.of_paper(pid)
        )
        if papers[pid].role == "core" and prose_only and not in_sections[pid]:
            problems.append(
                f"core paper {pid} needs evidence in a section: the table's "
                "results and limitation do not count; give a section one of its "
                "other items (a method, finding, definition, equation or figure)"
            )
    for name, mapping, keys in (
        ("short_names", outline.short_names, papers),
        ("ideas", outline.ideas, papers),
        ("lineage_names", outline.lineage_names, lineages),
    ):
        if missing := sorted(set(keys) - set(mapping)):
            problems.append(f"{name} misses {missing}")
    core = {p.id for p in plan.papers if p.role == "core"}
    # The evidence the table can show (the repair's rule: other numbers and
    # limitations stay in their sections).
    with_numbers = {
        e.paper_id for e in ledger if e.kind == "number" and result_cell(e.cell)
    }
    with_limits = {
        e.paper_id for e in ledger if e.kind == "limitation" and limitation_cell(e.cell)
    }
    for row in outline.table:
        if row.paper in with_numbers and not row.results:
            problems.append(
                f"table row {row.paper} has number evidence: give its results"
            )
        if row.paper in with_limits and not row.limitation:
            problems.append(
                f"table row {row.paper} has limitation evidence: give its limitation"
            )
        if row.paper not in papers:
            problems.append(f"table row {row.paper} is not a survey paper")
        for eid in [*row.results, *([row.limitation] if row.limitation else [])]:
            item = ledger.get(eid)
            if item is None or item.paper_id != row.paper:
                problems.append(f"table row {row.paper}: {eid} is not its evidence")
            elif eid in row.results and item.kind != "number":
                problems.append(f"table row {row.paper}: {eid} is not a number")
            elif eid == row.limitation and item.kind != "limitation":
                problems.append(f"table row {row.paper}: {eid} is not a limitation")
    wanted = core | (with_numbers & set(papers))
    if missing := sorted(wanted - {r.paper for r in outline.table}):
        problems.append(
            f"the table misses {missing} (every core paper and every paper with "
            "number evidence)"
        )
    for item in ledger:
        if (
            item.paper_id in core
            and item.kind in ("figure", "equation")
            and item.id not in owner
        ):
            problems.append(
                f"{item.paper_id}'s {item.kind} {item.id} is given to no section: "
                "give it to the section that explains the paper"
            )
    return problems


def _cell(text: str) -> str:
    return escape_cell_pipes(" ".join(text.split()))


def timeline_table(plan: SurveyPlan, outline: EvidenceOutline, korean: bool) -> str:
    """Every paper by date: year, paper (short name), lineage, idea."""
    head = (
        ["연도", "논문", "계보", "핵심 아이디어"]
        if korean
        else ["Year", "Paper", "Lineage", "Key idea"]
    )
    rows = [
        [
            str(p.year or ""),
            f"{outline.short_names.get(p.id, p.title)} [@{p.id}]",
            outline.lineage_names.get(p.lineage, p.lineage),
            outline.ideas.get(p.id, ""),
        ]
        for p in plan.papers
    ]
    return _table(head, rows)


def limitation_cell(cell: str) -> bool:
    """Whether a limitation's cell can stand in the comparison table: a
    noun phrase or an "입니다" sentence, no plain-style one."""
    return bool(cell) and not ends_in_plain_declarative(cell)


def result_cell(cell: str) -> bool:
    """Whether a number evidence's cell can stand in the comparison table's
    results column: it states a number and is no plain-style sentence (a
    sentence in another register reads as prose in a table)."""
    return (
        bool(cell)
        and any(c.isdigit() for c in cell)
        and not (ends_in_plain_declarative(cell))
    )


def comparison_table(
    plan: SurveyPlan, outline: EvidenceOutline, ledger: EvidenceLedger, korean: bool
) -> str:
    """Core papers: lineage, key results and limitation, from the evidence's
    noun-phrase cells only (a claim is a sentence in another register); an
    empty cell says so instead of a dash."""
    none_result = "정량 수치 없음" if korean else "No quantitative result"
    none_limit = "명시 없음" if korean else "Not stated"
    head = (
        ["논문", "계보", "주요 결과", "한계"]
        if korean
        else ["Paper", "Lineage", "Key results", "Limitation"]
    )
    papers = {p.id: p for p in plan.papers}
    rows = []
    for row in outline.table:
        paper = papers.get(row.paper)
        if paper is None:
            continue
        results = [e for eid in row.results if (e := ledger.get(eid))]
        limit = ledger.get(row.limitation) if row.limitation else None
        rows.append(
            [
                outline.short_names.get(paper.id, paper.title),
                outline.lineage_names.get(paper.lineage, paper.lineage),
                "; ".join(e.cell for e in results if result_cell(e.cell))
                or none_result,
                limit.cell if limit and limitation_cell(limit.cell) else none_limit,
            ]
        )
    return _table(head, rows)


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join("---" for _ in head) + " |",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# ---- section drafts ---------------------------------------------------------


def cite_papers(markdown: str, ledger: EvidenceLedger) -> str:
    """``[[E12, E15]]`` markers as the papers' citation keys (``[@id]``), a
    paper cited once per marker; unknown ids dropped."""

    def repl(m: re.Match[str]) -> str:
        pids: list[str] = []
        for eid in cited_ids(m.group(0)):
            item = ledger.get(eid)
            if item is not None and item.paper_id not in pids:
                pids.append(item.paper_id)
        keys = "".join(f"[@{p}]" for p in pids)
        # A marker that cites nothing ([[EX]]) goes with the space before it.
        return f"{m.group('space')}{keys}" if keys else ""

    out = re.sub(r"(?P<space>[ \t]*)" + MARKER.pattern, repl, markdown, flags=re.I)
    # The same paper cited twice in a row reads as a slip.
    out = re.sub(r"(\[@[^\]]+\])(\s*\1)+", r"\1", out)
    return _one_citation_per_run(out)


# A sentence's citation keys ("[@a][@b]").
_CITATION_GROUP = re.compile(r"(?:[ \t]*\[@[^\]]+\])+")


def _one_citation_per_run(markdown: str) -> str:
    """Within one prose line (a paragraph or list item), a run of sentences
    citing the same papers keeps the citation after the run's last sentence
    only: the same key after every sentence reads as noise. The claim check
    judged each sentence on its own markers before this conversion, so no
    check loses its evidence. Code, math, tables and headings are left
    alone."""
    lines = markdown.split("\n")
    for i, (line, role) in enumerate(zip(lines, block_lines(markdown), strict=True)):
        if role != "prose" or line.lstrip().startswith(("|", "#")):
            continue
        groups = list(_CITATION_GROUP.finditer(line))
        drop = [
            g
            for g, nxt in zip(groups, groups[1:], strict=False)
            if g.group(0).strip() == nxt.group(0).strip()
        ]
        for g in reversed(drop):
            line = line[: g.start()] + line[g.end() :]
        lines[i] = line
    return "\n".join(lines)


def repair_outline(
    outline: EvidenceOutline,
    plan: SurveyPlan,
    ledger: EvidenceLedger,
    *,
    merged: Mapping[str, str] | None = None,
) -> tuple[EvidenceOutline, list[str]]:
    """Fix what code can fix in an outline, deterministically, by fact
    (``merged``: evidence merged as the same fact -> the item kept):

    - evidence given twice keeps its first section; evidence a lineage
      section does not cover goes to its paper's lineage section (it stays
      when the paper has none, for the outline check to name);
    - a core paper's unassigned figures and equations go to its section;
    - table rows: a row naming its paper by its short name is that paper's,
      a paper's later rows add to its first, a row for a paper outside the
      plan is dropped; a row shows its own paper's evidence the table can
      show (``result_cell``, ``limitation_cell``; a merged fact by its
      showable copy), filled from the paper's first such items, and every id
      any row gave that the table does not show goes back to its paper's
      lineage section;
    - a fact the table shows leaves the sections (said once).

    Returns the outline and what was changed; what is left for the model is
    structure."""
    merged = merged or {}
    notes: list[str] = []
    papers = {p.id: p for p in plan.papers}
    home = {
        pid: i
        for i, s in enumerate(outline.sections)
        if s.kind == "lineage"
        for pid in s.papers
    }
    sections = [s.model_copy(deep=True) for s in outline.sections]
    seen: set[str] = set()
    # Evidence in a lineage section that does not cover its paper moves to
    # its paper's section after the pass (a later section that already
    # lists it keeps that copy, once).
    moves: list[tuple[int, str]] = []
    for i, s in enumerate(sections):
        kept = []
        for eid in s.evidence:
            item = ledger.get(eid)
            if item is None or eid in seen:
                continue
            if s.kind == "lineage" and item.paper_id not in s.papers:
                target = home.get(item.paper_id)
                if target is not None and target != i:
                    moves.append((target, eid))
                    continue
                # Its paper has no lineage section: it stays, and the outline
                # check names the section that does not cover it.
            kept.append(eid)
            seen.add(eid)
        s.evidence = kept
    for target, eid in moves:
        if eid not in seen:
            sections[target].evidence.append(eid)
            seen.add(eid)
            notes.append(f"moved {eid} to its paper's lineage section")
    for item in ledger:
        paper = papers.get(item.paper_id)
        if (
            paper is not None
            and paper.role == "core"
            and item.kind in ("figure", "equation")
            and item.id not in seen
            and (target := home.get(item.paper_id)) is not None
        ):
            sections[target].evidence.append(item.id)
            seen.add(item.id)
            notes.append(f"gave {item.id} ({item.kind}) to its paper's section")
    # Rows by paper: a row that names its paper by its short name is that
    # paper's, and a paper's later rows add to its first. Every id any row
    # gave is accounted for below (shown, or given back to its section).
    by_short = {name: pid for pid, name in outline.short_names.items() if pid in papers}
    rows: dict[str, TableRow] = {}
    asked: list[str] = []
    for r in outline.table:
        asked += [*r.results, *([r.limitation] if r.limitation else [])]
        pid = r.paper if r.paper in papers else by_short.get(r.paper)
        if pid is None:
            notes.append(f"dropped table row {r.paper!r} (not a survey paper)")
        elif pid in rows:
            rows[pid].results += r.results
            rows[pid].limitation = rows[pid].limitation or r.limitation
            notes.append(f"merged the table rows of {pid}")
        else:
            rows[pid] = r.model_copy(deep=True, update={"paper": pid})

    def fact(eid: str) -> str:
        # Evidence merged as the same fact counts as the item it was merged
        # into (the one sections hold).
        return merged.get(eid, eid)

    def shown_copy(eid: str, kind: str, showable: Callable[[str], bool]) -> str | None:
        """The copy of ``eid``'s fact the table can show: itself, else
        another item merged with it, else None."""
        group = [eid, *(e.id for e in ledger if fact(e.id) == fact(eid))]
        for candidate in dict.fromkeys(group):
            item = ledger.get(candidate)
            if item is not None and item.kind == kind and showable(item.cell):
                return candidate
        return None

    def first_copies(pid: str, kind: str, showable: Callable[[str], bool]) -> list[str]:
        """The paper's evidence of ``kind`` the table can show, one copy per
        fact, in ledger order."""
        out: dict[str, str] = {}
        for e in ledger.of_paper(pid):
            if e.kind == kind and showable(e.cell):
                out.setdefault(fact(e.id), e.id)
        return list(out.values())

    in_sections = {fact(eid) for s in sections for eid in s.evidence}
    for pid, paper in papers.items():
        # Only evidence whose cell the table can show goes there, one copy
        # per fact; the rest stays in (or goes back to) its section, so no
        # fact leaves the post and none is said twice.
        numbers = first_copies(pid, "number", result_cell)
        limits = first_copies(pid, "limitation", limitation_cell)
        if pid not in rows and (paper.role == "core" or numbers):
            rows[pid] = TableRow(paper=pid)
            notes.append(f"added {pid} to the table")
        if pid not in rows:
            continue
        row = rows[pid]

        def own(eid: str | None, pid: str = pid) -> bool:
            item = ledger.get(eid) if eid else None
            return item is not None and item.paper_id == pid

        results: dict[str, str] = {}
        for eid in row.results:
            copy = shown_copy(eid, "number", result_cell)
            if copy is not None and own(copy):
                results.setdefault(fact(copy), copy)
        row.results = list(results.values())[:2] or numbers[:2]
        limitation = (
            shown_copy(row.limitation, "limitation", limitation_cell)
            if row.limitation
            else None
        )
        row.limitation = (
            limitation if own(limitation) else (limits[0] if limits else None)
        )
    shown = {
        (r.paper, fact(e))
        for r in rows.values()
        for e in [*r.results, r.limitation]
        if e
    }
    for eid in dict.fromkeys(fact(e) for e in asked):
        item = ledger.get(eid)
        if item is None or (item.paper_id, eid) in shown or eid in in_sections:
            continue
        if (target := home.get(item.paper_id)) is not None:
            sections[target].evidence.append(eid)
            in_sections.add(eid)
            notes.append(f"gave {eid} back to {item.paper_id}'s lineage section")
        else:
            notes.append(
                f"dropped {eid} from the table: {item.paper_id} has no lineage "
                "section to take it"
            )
    order = [p.id for p in plan.papers]
    table = sorted(rows.values(), key=lambda r: order.index(r.paper))
    # The table owns the facts it shows: a result or limitation shown there
    # is not restated in a section (facts said once; the survey's numbers in
    # one table).
    in_table = {
        (r.paper, fact(eid))
        for r in table
        for eid in [*r.results, *([r.limitation] if r.limitation else [])]
    }

    def shown_in_table(eid: str) -> bool:
        item = ledger.get(eid)
        return item is not None and (item.paper_id, fact(eid)) in in_table

    for s in sections:
        moved = [eid for eid in s.evidence if shown_in_table(eid)]
        if moved:
            s.evidence = [eid for eid in s.evidence if not shown_in_table(eid)]
            notes.append(f"left {moved} to the table")
    # A core paper the table left with nothing in any section gets its first
    # unused prose item (a method, finding, definition) in its lineage
    # section, as its figures and equations do.
    placed = {eid for s in sections for eid in s.evidence}
    for pid, paper in papers.items():
        if paper.role != "core" or (target := home.get(pid)) is None:
            continue
        mine = [e for e in ledger.of_paper(pid)]
        if any(e.id in placed for e in mine):
            continue
        prose = next(
            (
                e
                for e in mine
                if e.kind not in ("number", "limitation") and fact(e.id) == e.id
            ),
            None,
        )
        if prose is not None:
            sections[target].evidence.append(prose.id)
            placed.add(prose.id)
            notes.append(f"gave {prose.id} to {pid}'s lineage section (its only prose)")
        # Synthesis sections interpret; they are not retellings.
        if s.kind in ("comparison", "insights", "conclusion") and s.depth == "deep":
            s.depth = "standard"
        if s.kind == "conclusion":
            s.depth = "brief"
    return outline.model_copy(update={"sections": sections, "table": table}), notes


def section_targets(
    outline: EvidenceOutline,
    plan: SurveyPlan,
    *,
    per_core: int,
    per_supporting: int,
    min_total: int,
    max_total: int,
    language: str,
) -> list[int]:
    """Prose characters per section: a lineage section its papers' shares,
    any other section its depth's size; the total scaled into
    ``[min_total, max_total]`` (Korean characters, converted by
    :func:`~scholar_lens.domain.lengths.char_factor`)."""
    role = {p.id: p.role for p in plan.papers}
    share = {"core": per_core, "supporting": per_supporting}
    raw = [
        (
            sum(share[role[p]] for p in s.papers if p in role) or DEPTH_CHARS[s.depth]
            if s.kind == "lineage"
            else DEPTH_CHARS[s.depth]
        )
        for s in outline.sections
    ]
    return scaled_targets(
        raw, min_total=min_total, max_total=max_total, language=language
    )


# ---- the table's evidence ---------------------------------------------------


def table_evidence(outline: EvidenceOutline) -> set[str]:
    """The evidence ids the comparison table cites (its results and
    limitations): checked like the sections' evidence."""
    return {eid for r in outline.table for eid in [*r.results, r.limitation] if eid}


def with_canonical_table(
    outline: EvidenceOutline, duplicates: Mapping[str, str]
) -> EvidenceOutline:
    """``outline`` with every duplicate evidence id of its table replaced by
    the id that keeps the fact."""
    if not duplicates:
        return outline
    table = [
        r.model_copy(
            update={
                "results": canonical_ids(r.results, duplicates),
                "limitation": (
                    duplicates.get(r.limitation, r.limitation)
                    if r.limitation
                    else r.limitation
                ),
            }
        )
        for r in outline.table
    ]
    return outline.model_copy(update={"table": table})


def without_table_evidence(
    outline: EvidenceOutline, wrong: Collection[str]
) -> EvidenceOutline:
    """``outline`` with the ``wrong`` evidence left out of its table."""
    table = [
        r.model_copy(
            update={
                "results": [e for e in r.results if e not in wrong],
                "limitation": None if r.limitation in wrong else r.limitation,
            }
        )
        for r in outline.table
    ]
    return outline.model_copy(update={"table": table})
