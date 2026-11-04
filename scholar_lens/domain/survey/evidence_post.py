"""The evidence-based survey (docs/redesign.md, 3.2): an outline that gives
each piece of evidence to one section, the code-built tables, and the
deterministic checks of a section draft whose sentences cite evidence as
``[[E12]]``."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..evidence import Evidence, EvidenceLedger
from ..markdown.tables import escape_cell_pipes
from ..validation import unsourced_numbers
from .plan import SurveyPlan

SectionKind = Literal[
    "introduction", "lineage", "cross_cutting", "comparison", "insights", "conclusion"
]
Depth = Literal["deep", "standard", "brief"]

MARKER = re.compile(r"\[\[\s*(E\d+(?:\s*,\s*E\d+)*)\s*\]\]")
_ID = re.compile(r"E\d+")
_FENCE = re.compile(r"^(`{3,}|~{3,}).*?^\1[ \t]*$", re.M | re.S)
_DISPLAY_MATH = re.compile(r"^\s*\$\$.*?\$\$\s*$", re.M | re.S)
_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)[^)]*\)")
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$", re.M)
_SENTENCE_END = re.compile(r"(?<=[.!?。])\s+|\n+")
_CITATION_KEY = re.compile(r"\[@[^\]]+\]")
# An arXiv id or DOI written as a paper's name in prose.
_PAPER_ID = re.compile(r"(?<![\d./])(?:\d{4}\.\d{4,5}|10\.\d{4,9}/[^\s,;)]+)(?![\d])")
# How a post describes its own making: the input it was given (the abstract,
# a card, a summary) instead of the papers. Closed list from past posts.
META_SENTENCE = re.compile(
    r"초록(?:에서|만|에 따르면|에 기대|과 공개|수준|에는|으로는)|공개 요약|카드(?:에|가|만|로|의)|"
    r"입력 자료|제공된 (?:자료|문서|출처|근거)|근거 자료(?:에|가|로)|"
    r"(?:이|본) 글(?:에서)?(?:는|은) .{0,20}(?:살펴봅니다|다룹니다|정리합니다)|이 절에서는|"
    r"\b(?:the|this) (?:abstract|card|summary) (?:says|states|reports)|based on the abstract",
    re.IGNORECASE,
)

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
    cell = item.cell or item.claim
    return f"{cell} ({item.scope})" if item.scope and item.scope not in cell else cell


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join("---" for _ in head) + " |",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# ---- section drafts ---------------------------------------------------------


def prose_sentences(markdown: str) -> list[str]:
    """The prose sentences of a draft: code, display math, tables, headings
    and images left out."""
    text = _FENCE.sub("\n", markdown)
    text = _DISPLAY_MATH.sub("\n", text)
    text = _TABLE_LINE.sub("\n", text)
    text = _IMAGE.sub(" ", text)
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^(?:>\s*)+|^(?:[-*+]|\d+[.)])\s+", "", line)
        out += [s.strip() for s in _SENTENCE_END.split(line) if s.strip()]
    return out


def cited_ids(sentence: str) -> list[str]:
    return [i for m in MARKER.finditer(sentence) for i in _ID.findall(m.group(1))]


def _plain(sentence: str) -> str:
    return _CITATION_KEY.sub("", MARKER.sub("", sentence))


def stated_numbers(sentence: str, free: str) -> list[str]:
    """The numbers a sentence states beyond ``free`` (years, small counts)."""
    return unsourced_numbers(_plain(sentence), free)


# Visuals a lineage or cross-cutting section needs by depth (images, display
# equations and tables); the code-built tables follow other kinds.
MIN_VISUALS = {"deep": 2, "standard": 1, "brief": 0}
# A sentence the writer marks as its own illustration: its numbers need no
# evidence (the claim check still reads it).
_EXAMPLE = re.compile(
    r"예를 들어|가령|가상의|예시로|(?:이)?라고 (?:해 ?보면|하면|가정)|가정해|"
    r"for example|e\.g\.|suppose",
    re.IGNORECASE,
)
# A Korean heading in the plain register ("…했다"), not the post's "…입니다".
_DA_ENDING = re.compile(r"(?<!니)다[.!]?$")
_OWN_VIEW = re.compile(
    r"이 글의 (?:관점|판단|해석|견해)|이 글은 .{0,12}(?:본다|봅니다|판단합니다)"
)
# A number with a size unit names a model ("Llama 2 7B"), not a result.
_MODEL_SIZE = re.compile(r"^\d+(?:\.\d+)?[BMK]$", re.IGNORECASE)


def visual_count(markdown: str) -> int:
    body = _FENCE.sub("\n", markdown)
    tables = len(re.findall(r"^\s*\|?\s*:?-{3,}", body, re.M))
    return len(_IMAGE.findall(body)) + len(_DISPLAY_MATH.findall(body)) + tables


def draft_problems(
    markdown: str,
    *,
    title: str,
    own: Iterable[str],
    earlier: Iterable[str],
    ledger: EvidenceLedger,
    earlier_numbers: Mapping[str, str],
    figure_urls: Iterable[str],
    free_numbers: str,
    depth: str = "",
) -> list[str]:
    """Deterministic problems of a section draft (each quotes its sentence):
    a missing heading, unknown or foreign evidence ids, a number without
    evidence or not in the cited evidence, a number an earlier section
    already stated, a sentence about the post's own inputs, an unassigned or
    missing figure."""
    problems: list[str] = []
    for heading in re.findall(r"^#{2,4} (.+?)\s*$", markdown, re.M):
        if _DA_ENDING.search(heading):
            problems.append(
                f"heading ends in '~다': write it as a '~입니다/~합니다' sentence "
                f"or a noun phrase: {heading}"
            )
    if len(_OWN_VIEW.findall(_FENCE.sub("", markdown))) > 1:
        problems.append(
            "marks the post's own view more than once; keep one marker in a "
            "section and state the rest plainly or leave it out"
        )
    if not markdown.lstrip().startswith("## "):
        problems.append(f"the section must start with its '## ' heading ({title})")
    allowed = set(own) | set(earlier)
    for sentence in prose_sentences(markdown):
        quoted = _plain(sentence)[:160]
        ids = cited_ids(sentence)
        if unknown := [i for i in ids if ledger.get(i) is None]:
            problems.append(f"unknown evidence {unknown} in: {quoted}")
            continue
        if foreign := [i for i in ids if i not in allowed]:
            problems.append(
                f"evidence {foreign} belongs to another section; leave it out: {quoted}"
            )
        numbers = stated_numbers(sentence, free_numbers)
        example = bool(_EXAMPLE.search(sentence))
        if numbers and not ids and not example:
            problems.append(f"numbers {numbers} without evidence: {quoted}")
        elif numbers and not example:
            support = " ".join(
                f"{e.quote} {e.scope} {e.claim} {e.cell}"
                for i in ids
                if (e := ledger.get(i))
            )
            if missing := unsourced_numbers(
                _plain(sentence), f"{support} {free_numbers}"
            ):
                problems.append(
                    f"numbers {missing} are not in the cited evidence: {quoted}"
                )
        if repeated := [
            n for n in numbers if n in earlier_numbers and not _MODEL_SIZE.match(n)
        ]:
            problems.append(
                f"{repeated} already stated in '{earlier_numbers[repeated[0]]}': "
                f"refer back without repeating: {quoted}"
            )
        if _PAPER_ID.search(_plain(sentence)):
            problems.append(
                f"names a paper by its id; use its name or its authors: {quoted}"
            )
        if META_SENTENCE.search(_plain(sentence)):
            problems.append(
                "describes the post's inputs instead of the papers; state what "
                f"the paper does or leave it out: {quoted}"
            )
    if (need := MIN_VISUALS.get(depth, 0)) and visual_count(markdown) < need:
        problems.append(
            f"a {depth} section needs at least {need} visuals (its figures, an "
            "equation, or a small table from its number evidence)"
        )
    expected = set(figure_urls)
    shown = set(_IMAGE.findall(markdown))
    if extra := sorted(shown - expected):
        problems.append(f"images not given to this section: {extra}")
    if missing := sorted(expected - shown):
        problems.append(f"figures given to this section are missing: {missing}")
    return problems


def section_heading(markdown: str) -> str:
    """The text of the draft's first ``## `` heading ("" without one)."""
    m = re.search(r"^## (.+?)\s*$", markdown, re.M)
    return m.group(1) if m else ""


def unsourced_number_sentences(
    markdown: str, ledger: EvidenceLedger, free_numbers: str
) -> list[str]:
    """Sentences stating numbers their cited evidence does not (or citing
    none); a sentence marked as the writer's own example is left alone."""
    found = []
    for sentence in prose_sentences(markdown):
        numbers = stated_numbers(sentence, free_numbers)
        if not numbers or _EXAMPLE.search(sentence):
            continue
        support = " ".join(
            f"{e.quote} {e.scope} {e.claim} {e.cell}"
            for i in cited_ids(sentence)
            if (e := ledger.get(i))
        )
        if unsourced_numbers(_plain(sentence), f"{support} {free_numbers}"):
            found.append(sentence)
    return found


def section_numbers(markdown: str, free_numbers: str) -> set[str]:
    return {
        n for s in prose_sentences(markdown) for n in stated_numbers(s, free_numbers)
    }


def cite_papers(markdown: str, ledger: EvidenceLedger) -> str:
    """``[[E12, E15]]`` markers as the papers' citation keys (``[@id]``), a
    paper cited once per marker; unknown ids dropped."""

    def repl(m: re.Match[str]) -> str:
        pids: list[str] = []
        for eid in _ID.findall(m.group(1)):
            item = ledger.get(eid)
            if item is not None and item.paper_id not in pids:
                pids.append(item.paper_id)
        return "".join(f"[@{p}]" for p in pids)

    out = MARKER.sub(repl, markdown)
    # The same paper cited twice in a row reads as a slip.
    return re.sub(r"(\[@[^\]]+\])(\s*\1)+", r"\1", out)


_FRAGMENT = re.compile(r"^(?:[-*+]|\d+[.)])?\s*[^\s#|>!$`].{0,24}$")
_SENTENCE_CLOSE = re.compile(r"[.!?。:)\]]\s*$")


def drop_sentences(markdown: str, sentences: Iterable[str]) -> str:
    """``markdown`` without the given sentences (as :func:`prose_sentences`
    returns them), each removed once, and without the fragments a removal
    leaves (a short line that ends no sentence, such as "첫째")."""
    for sentence in sentences:
        markdown = markdown.replace(sentence, "", 1)
    lines = []
    for line in markdown.split("\n"):
        bare = line.strip()
        if bare and _FRAGMENT.match(bare) and not _SENTENCE_CLOSE.search(bare):
            continue
        lines.append(line.rstrip())
    out = "\n".join(lines)
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"\n{3,}", "\n\n", out))


_HEADINGS = re.compile(r"^#{2,4} .+$", re.M)


def structure_lost(before: str, after: str) -> str:
    """Why a revision lost the draft's structure ("" when it did not): a
    heading, an image or display math gone, or more than a quarter of the
    prose."""
    if set(_HEADINGS.findall(before)) - set(_HEADINGS.findall(after)):
        return "a heading was lost"
    if set(_IMAGE.findall(before)) - set(_IMAGE.findall(after)):
        return "a figure was lost"
    if before.count("$$") > after.count("$$"):
        return "an equation was lost"
    if len(" ".join(prose_sentences(after))) < 0.75 * len(
        " ".join(prose_sentences(before))
    ):
        return "more than a quarter of the prose was cut"
    return ""


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
    return outline.model_copy(update={"sections": sections, "table": table}), notes
