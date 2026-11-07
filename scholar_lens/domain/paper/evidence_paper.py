"""The evidence-based paper review and summary (docs/redesign.md, 3.2 and 5.2):
an outline that gives each piece of the paper's evidence to one section, in
the post's order, and the fixed parts of each kind.

A review goes TL;DR (four fixed questions, written last), the problem, the
method in depth, the experiments, optional analysis, and the post's own
assessment ("한계와 시사점"). A summary is a TL;DR paragraph and five fixed
sections. Neither restates the abstract after the TL;DR: each fact is said
once, in the section its evidence belongs to.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..evidence import EvidenceLedger
from ..lengths import DEPTH_CHARS, scaled_targets

PaperKind = Literal[
    "tldr", "problem", "method", "implementation", "experiments", "analysis", "closing"
]
PostKind = Literal["review", "summary"]
Depth = Literal["deep", "standard", "brief"]
TLDR_TITLE = "TL;DR"

# The review's TL;DR answers these, in order (blog convention).
REVIEW_TLDR_QUESTIONS: dict[str, tuple[str, ...]] = {
    "Korean": (
        "이 연구를 시작하게 된 배경과 동기는 무엇입니까?",
        "이 연구에서 제시하는 새로운 해결 방법은 무엇입니까?",
        "제안된 방법은 어떻게 구현되었습니까?",
        "이 연구의 결과가 가지는 의미는 무엇입니까?",
    ),
    "": (
        "What motivated this research?",
        "What new solution does it propose?",
        "How is the proposed method implemented?",
        "What do the results mean?",
    ),
}
CLOSING_TITLES = {"Korean": "한계와 시사점", "": "Limitations and Takeaways"}
# A summary's five sections, in order (blog convention).
SUMMARY_SECTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "Korean": (
        ("problem", "연구 배경"),
        ("method", "제안 방법"),
        ("implementation", "구현 방식"),
        ("experiments", "실험 결과"),
        ("closing", "의의와 한계"),
    ),
    "": (
        ("problem", "Background"),
        ("method", "Proposed Method"),
        ("implementation", "Implementation"),
        ("experiments", "Results"),
        ("closing", "Significance and Limitations"),
    ),
}
_RANK = {
    "tldr": 0,
    "problem": 1,
    "method": 2,
    "implementation": 2,
    "experiments": 3,
    "analysis": 4,
    "closing": 5,
}


def _lang(language: str) -> str | None:
    """The key of the blog's fixed wording in ``language``: "Korean",
    "" (English), or None when the blog has none (the outline's titles stand
    and the TL;DR questions are written in the post's language)."""
    name = language.strip().lower()
    if name == "korean":
        return "Korean"
    if name in ("", "english"):
        return ""
    return None


def tldr_questions(language: str) -> tuple[tuple[str, ...], bool]:
    """The review's four TL;DR questions and whether they are the blog's
    exact wording (False: English, for the writer to translate)."""
    lang = _lang(language)
    return REVIEW_TLDR_QUESTIONS[lang or ""], lang is not None


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PaperSection(_Strict):
    kind: PaperKind
    # A claim as a noun phrase ("메시지 전달을 줄인 그래프 학습").
    title: str = Field(min_length=1)
    thesis: str = ""
    question: str = ""
    depth: Depth = "standard"
    evidence: list[str] = Field(default_factory=list)


class GlossaryTerm(_Strict):
    english: str
    rendering: str


class PaperOutline(_Strict):
    glossary: list[GlossaryTerm] = Field(default_factory=list, max_length=20)
    # One concrete input or case threaded through the post, when the paper
    # has one.
    running_example: str = ""
    sections: list[PaperSection] = Field(min_length=3)
    # Groups of evidence ids that state one fact in different words; code
    # keeps one of each, so the post states the fact once.
    same_facts: list[list[str]] = Field(default_factory=list, max_length=60)


def fixed_title(kind: str, post: PostKind, language: str) -> str | None:
    """The title code gives a section: the TL;DR, the review's closing, and
    every summary section; None when the outline's title stands."""
    if kind == "tldr":
        return TLDR_TITLE
    lang = _lang(language)
    if lang is None:
        return None
    if post == "summary":
        return dict(SUMMARY_SECTIONS[lang]).get(kind)
    if kind == "closing":
        return CLOSING_TITLES[lang]
    return None


def repair_paper_outline(
    outline: PaperOutline, ledger: EvidenceLedger, post: PostKind, language: str
) -> tuple[PaperOutline, list[str]]:
    """What code can fix: unknown evidence dropped, evidence given twice kept
    in its first section, the TL;DR without evidence of its own (written
    last), fixed titles set, sections sorted into the post's order (a
    summary's kinds into its five), and a summary's duplicate kinds merged."""
    notes: list[str] = []
    seen: set[str] = set()
    sections: list[PaperSection] = []
    for s in outline.sections:
        update: dict[str, object] = {}
        if (title := fixed_title(s.kind, post, language)) is not None:
            update["title"] = title
        if s.kind == "tldr":
            update |= {"evidence": [], "depth": "brief"}
            sections.append(s.model_copy(update=update))
            continue
        kept = []
        for eid in s.evidence:
            if ledger.get(eid) is None:
                notes.append(f"unknown evidence {eid} dropped from '{s.title}'")
            elif eid in seen:
                notes.append(f"{eid} given twice; kept in its first section")
            else:
                seen.add(eid)
                kept.append(eid)
        update["evidence"] = kept
        sections.append(s.model_copy(update=update))
    if post == "summary":
        merged: dict[str, PaperSection] = {}
        for s in sections:
            kind = "method" if s.kind == "analysis" else s.kind
            if kind in merged:
                first = merged[kind]
                merged[kind] = first.model_copy(
                    update={"evidence": first.evidence + s.evidence}
                )
                notes.append(f"summary sections of kind {kind} merged")
            else:
                merged[kind] = s.model_copy(update={"kind": kind})
        sections = list(merged.values())
    ordered = sorted(sections, key=lambda s: _RANK[s.kind])
    if [s.kind for s in ordered] != [s.kind for s in sections]:
        notes.append("sections sorted into the post's order")
    return outline.model_copy(update={"sections": ordered}), notes


def paper_outline_problems(
    outline: PaperOutline,
    ledger: EvidenceLedger,
    post: PostKind,
    *,
    max_sections: int,
) -> list[str]:
    """What makes an outline unusable: the kinds and their order, a missing
    method or closing, too many sections, evidence unknown or given twice, a
    section without evidence."""
    problems: list[str] = []
    kinds = [s.kind for s in outline.sections]
    if kinds[0] != "tldr" or kinds.count("tldr") != 1:
        problems.append("the post starts with exactly one tldr section")
    if kinds[-1] != "closing" or kinds.count("closing") != 1:
        problems.append("the post ends with exactly one closing section")
    if "method" not in kinds:
        problems.append("at least one method section")
    if kinds.count("problem") != 1:
        problems.append("exactly one problem section")
    if [_RANK[k] for k in kinds] != sorted(_RANK[k] for k in kinds):
        problems.append(
            "sections must go tldr, problem, method and implementation, "
            "experiments, analysis, closing"
        )
    if post == "summary":
        wanted = [k for k, _ in SUMMARY_SECTIONS[""]]
        if [k for k in kinds if k != "tldr"] != wanted:
            problems.append(
                "a summary has exactly the sections problem, method, "
                "implementation, experiments, closing"
            )
    if len(outline.sections) > max_sections:
        problems.append(f"at most {max_sections} sections")
    if any(e.kind == "number" for e in ledger) and "experiments" not in kinds:
        problems.append("the paper reports results: add an experiments section")
    seen: dict[str, str] = {}
    for s in outline.sections:
        if s.kind not in ("tldr",) and not s.evidence and post == "review":
            problems.append(f"section '{s.title}' has no evidence")
        for eid in s.evidence:
            if ledger.get(eid) is None:
                problems.append(f"unknown evidence {eid} in '{s.title}'")
            elif eid in seen:
                problems.append(f"{eid} is in both '{seen[eid]}' and '{s.title}'")
            else:
                seen[eid] = s.title
    return problems


def paper_section_targets(
    outline: PaperOutline, *, min_total: int, max_total: int, language: str
) -> list[int]:
    """Prose characters per section by depth, the total scaled into
    ``[min_total, max_total]`` (Korean characters)."""
    return scaled_targets(
        [DEPTH_CHARS[s.depth] for s in outline.sections],
        min_total=min_total,
        max_total=max_total,
        language=language,
    )


_H3 = re.compile(r"^### (.+?)\s*$", re.M)


def tldr_problems(markdown: str, post: PostKind, language: str) -> list[str]:
    """A review's TL;DR answers the four fixed questions under ``###``
    headings, in order, each with an answer; a summary's is one or two plain
    paragraphs without headings, math, lists or images."""
    body = re.sub(r"^## .*\n?", "", markdown.strip(), count=1).strip()
    if not body:
        return ["the TL;DR is empty"]
    if post == "review":
        questions, exact = tldr_questions(language)
        headings = tuple(h.strip() for h in _H3.findall(body))
        if headings != questions if exact else len(headings) != len(questions):
            return [
                "the TL;DR must be exactly these four ### headings, in order, "
                "each followed by its answer"
                + ("" if exact else f" (in {language})")
                + ": "
                + " | ".join(questions)
            ]
        parts = re.split(r"^### .+$", body, flags=re.M)[1:]
        if any(not p.strip() for p in parts):
            return ["every TL;DR question needs an answer under it"]
        return []
    problems = []
    if re.search(r"^#{2,6} |^\s*[-*] |\$|!\[", body, re.M):
        problems.append(
            "the summary's TL;DR is plain prose: no headings, lists, math or images"
        )
    return problems


def assemble_review(sections: Sequence[tuple[str, str]]) -> tuple[str, str, str]:
    """(key takeaways, body, closing) from (kind, markdown) in order: the
    TL;DR without its heading, the sections between, the closing section."""
    key = next((md for kind, md in sections if kind == "tldr"), "")
    key = re.sub(r"^## .*\n+", "", key.strip(), count=1)
    closing = next((md for kind, md in sections if kind == "closing"), "")
    body = "\n\n".join(
        md.strip() for kind, md in sections if kind not in ("tldr", "closing")
    )
    return key.strip(), body.strip(), closing.strip()


def assemble_summary(sections: Sequence[tuple[str, str]]) -> str:
    """The summary body: ``## TL;DR`` then the five sections."""
    return "\n\n".join(md.strip() for _, md in sections if md.strip()) + "\n"
