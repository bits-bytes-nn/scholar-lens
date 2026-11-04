"""The evidence-based survey (docs/redesign.md, 3.2): an outline that gives
each piece of evidence to one section, the code-built tables, and the
deterministic checks of a section draft whose sentences cite evidence as
``[[E12]]``."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..evidence import Evidence, EvidenceLedger
from ..evidence_draft import MARKER, cited_ids
from ..lengths import DEPTH_CHARS, NON_KOREAN_CHAR_FACTOR
from ..markdown.tables import escape_cell_pipes
from .plan import SurveyPlan

SectionKind = Literal[
    "introduction", "lineage", "cross_cutting", "comparison", "insights", "conclusion"
]
Depth = Literal["deep", "standard", "brief"]


_KIND_ORDER = {
    "introduction": 0,
    "lineage": 1,
    "cross_cutting": 2,
    "comparison": 3,
    "insights": 4,
    "conclusion": 5,
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutlineSection(_Strict):
    kind: SectionKind
    # A claim, not a topic label ("보상 모델은 정책보다 먼저 무너집니다").
    title: str = Field(min_length=1)
    # The one sentence the section argues.
    thesis: str = Field(min_length=1)
    # The reader's question the section answers.
    question: str = ""
    lineage: str | None = None
    papers: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    depth: Depth = "standard"


class GlossaryTerm(_Strict):
    english: str
    rendering: str


class TableRow(_Strict):
    paper: str
    results: list[str] = Field(default_factory=list, max_length=2)
    limitation: str | None = None


class EvidenceOutline(_Strict):
    title: str = Field(min_length=1)
    glossary: list[GlossaryTerm] = Field(default_factory=list, max_length=20)
    # One concrete case threaded through the post, when the topic has one.
    running_example: str = ""
    # Paper id -> its short name in tables ("DPO", "InstructGPT").
    short_names: dict[str, str] = Field(default_factory=dict)
    # Lineage id -> a short name for the tables.
    lineage_names: dict[str, str] = Field(default_factory=dict)
    # Paper id -> its idea in one noun phrase, for the timeline.
    ideas: dict[str, str] = Field(default_factory=dict)
    # The comparison table: core papers, their key results and a limitation
    # (evidence ids).
    table: list[TableRow] = Field(default_factory=list)
    sections: list[OutlineSection] = Field(min_length=3)


def outline_problems(
    outline: EvidenceOutline,
    plan: SurveyPlan,
    ledger: EvidenceLedger,
    *,
    title_max_length: int,
) -> list[str]:
    """What makes an outline unusable: the section order, a lineage without
    its one section, unknown papers or evidence, evidence given twice or to
    a section that does not cover its paper, papers without evidence, and
    missing table, timeline or name entries."""
    problems: list[str] = []
    papers = {p.id: p for p in plan.papers}
    lineages = {lin.id for lin in plan.lineages}
    if len(outline.title) > title_max_length:
        problems.append(f"the title is longer than {title_max_length} characters")
    kinds: list[str] = [s.kind for s in outline.sections]
    if kinds[0] != "introduction":
        problems.append("the first section must be the introduction")
    if [_KIND_ORDER[k] for k in kinds] != sorted(_KIND_ORDER[k] for k in kinds):
        problems.append(
            "sections must go introduction, lineages, cross-cutting, comparison, "
            "insights, conclusion"
        )
    for kind, count in (("comparison", 1), ("insights", 1)):
        if kinds.count(kind) != count:
            problems.append(f"the outline needs exactly {count} {kind} section")
    if kinds.count("conclusion") > 1:
        problems.append("at most one conclusion section")
    if kinds.count("cross_cutting") > 1:
        problems.append("at most one cross-cutting section")
    covered = [s.lineage for s in outline.sections if s.kind == "lineage"]
    for lin in sorted(lineages):
        if covered.count(lin) != 1:
            problems.append(f"lineage {lin!r} needs exactly one lineage section")
    if stray := set(covered) - lineages:
        problems.append(f"unknown lineages: {', '.join(sorted(map(str, stray)))}")
    owner: dict[str, str] = {}
    used: dict[str, int] = dict.fromkeys(papers, 0)
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
                problems.append(f"section {s.title!r} uses unknown evidence {eid}")
                continue
            if eid in owner:
                problems.append(
                    f"{eid} is given to both {owner[eid]!r} and {s.title!r}: "
                    "give each piece of evidence to one section"
                )
            owner[eid] = s.title
            if s.kind == "lineage" and item.paper_id not in s.papers:
                problems.append(
                    f"{eid} is about {item.paper_id}, which {s.title!r} does not cover"
                )
            if item.paper_id in used:
                used[item.paper_id] += 1
    for pid, count in used.items():
        need = 2 if papers[pid].role == "core" else 1
        if count < need and len(ledger.of_paper(pid)) >= need:
            problems.append(
                f"{pid} needs at least {need} pieces of evidence in sections"
            )
    for name, mapping, keys in (
        ("short_names", outline.short_names, papers),
        ("ideas", outline.ideas, papers),
        ("lineage_names", outline.lineage_names, lineages),
    ):
        if missing := sorted(set(keys) - set(mapping)):
            problems.append(f"{name} misses {missing}")
    core = {p.id for p in plan.papers if p.role == "core"}
    with_numbers = {e.paper_id for e in ledger if e.kind == "number"}
    with_limits = {e.paper_id for e in ledger if e.kind == "limitation"}
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


def comparison_table(
    plan: SurveyPlan, outline: EvidenceOutline, ledger: EvidenceLedger, korean: bool
) -> str:
    """Core papers: lineage, key results (evidence cells with their scope),
    limitation; an empty cell says so instead of a dash."""
    none_result = "정량 수치 없음" if korean else "No quantitative result"
    none_limit = "-"
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
                "; ".join(_result_cell(e) for e in results) or none_result,
                (limit.cell or limit.claim) if limit else none_limit,
            ]
        )
    return _table(head, rows)


def _result_cell(item: Evidence) -> str:
    """The evidence's noun-phrase cell as written (it names its model and
    benchmark), else its claim with the scope."""
    if item.cell:
        return _korean_units(item.cell)
    text = f"{item.claim} ({item.scope})" if item.scope else item.claim
    return _korean_units(text)


def _korean_units(text: str) -> str:
    return re.sub(r"\s+vs\.?\s+", " 대비 ", text)


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
        return "".join(f"[@{p}]" for p in pids)

    out = MARKER.sub(repl, markdown)
    # The same paper cited twice in a row reads as a slip.
    return re.sub(r"(\[@[^\]]+\])(\s*\1)+", r"\1", out)


def repair_outline(
    outline: EvidenceOutline, plan: SurveyPlan, ledger: EvidenceLedger
) -> tuple[EvidenceOutline, list[str]]:
    """Fix what code can fix in an outline, deterministically: evidence given
    twice keeps its first section; evidence a lineage section does not cover
    goes to its paper's lineage section; a core paper's unassigned figures
    and equations go there too; missing table rows and cells are filled from
    the paper's first number and limitation evidence. Returns the outline
    and what was changed; what is left for the model is structure."""
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
    for i, s in enumerate(sections):
        kept = []
        for eid in s.evidence:
            item = ledger.get(eid)
            if item is None or eid in seen:
                continue
            if s.kind == "lineage" and item.paper_id not in s.papers:
                target = home.get(item.paper_id)
                if target is not None and target != i:
                    sections[target].evidence.append(eid)
                    notes.append(f"moved {eid} to its paper's lineage section")
                    seen.add(eid)
                continue
            kept.append(eid)
            seen.add(eid)
        s.evidence = kept
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
    rows = {
        r.paper: r.model_copy(deep=True) for r in outline.table if r.paper in papers
    }
    for pid, paper in papers.items():
        numbers = [e.id for e in ledger.of_paper(pid) if e.kind == "number"]
        limits = [e.id for e in ledger.of_paper(pid) if e.kind == "limitation"]
        if pid not in rows and (paper.role == "core" or numbers):
            rows[pid] = TableRow(paper=pid)
            notes.append(f"added {pid} to the table")
        if pid in rows:
            row = rows[pid]
            row.results = [e for e in row.results if e in numbers][:2] or numbers[:2]
            if row.limitation not in limits:
                row.limitation = limits[0] if limits else None
    order = [p.id for p in plan.papers]
    table = sorted(rows.values(), key=lambda r: order.index(r.paper))
    # The table owns its results: a number shown there is not restated in
    # a section (facts said once; the survey's numbers in one table).
    in_table = {eid for r in table for eid in r.results}
    for s in sections:
        moved = [eid for eid in s.evidence if eid in in_table]
        if moved:
            s.evidence = [eid for eid in s.evidence if eid not in in_table]
            notes.append(f"left {moved} to the table")
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
    korean: bool,
) -> list[int]:
    """Prose characters per section: a lineage section its papers' shares,
    any other section its depth's size; the total scaled into
    ``[min_total, max_total]`` (Korean characters; other languages need
    :data:`NON_KOREAN_CHAR_FACTOR` times as many)."""
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
    total = sum(raw)
    scale = min(max(total, min_total), max_total) / total
    factor = 1.0 if korean else NON_KOREAN_CHAR_FACTOR
    return [round(r * scale * factor) for r in raw]
