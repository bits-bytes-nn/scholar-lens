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
from ..glossary import GlossaryTerm
from ..post_spec import (
    SectionBase,
    first_owner,
    ownership_problems,
    structure_problems,
)
from ..specs.paper import paper_spec

PaperKind = Literal[
    "tldr", "problem", "method", "implementation", "experiments", "analysis", "closing"
]
PostKind = Literal["review", "summary"]

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


class PaperSection(SectionBase):
    kind: PaperKind


class PaperOutline(_Strict):
    glossary: list[GlossaryTerm] = Field(default_factory=list, max_length=20)
    # One concrete input or case threaded through the post, when the paper
    # has one.
    running_example: str = ""
    sections: list[PaperSection] = Field(min_length=3)
    # Groups of evidence ids that state one fact in different words; code
    # keeps one of each, so the post states the fact once.
    same_facts: list[list[str]] = Field(default_factory=list, max_length=60)
    # Key target id -> the evidence ids that state it (docs/engine.md, 4.6).
    coverage: dict[str, list[str]] = Field(default_factory=dict)


def fixed_title(kind: str, post: PostKind, language: str) -> str | None:
    """The title code gives a section: the TL;DR, the review's closing, and
    every summary section; None when the outline's title stands."""
    spec = paper_spec(post)
    if spec.tldr is not None and kind == spec.tldr.kind:
        return spec.tldr.title
    lang = _lang(language)
    if lang is None:
        return None
    return spec.fixed_titles.get(kind, {}).get(lang)


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
        update["evidence"] = first_owner(s.evidence, ledger, seen, s.title, notes)
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
    rank = paper_spec(post).ranks
    ordered = sorted(sections, key=lambda s: rank[s.kind])
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
    spec = paper_spec(post)
    kinds = [s.kind for s in outline.sections]
    problems = structure_problems(spec, kinds, max_sections=max_sections)
    if any(e.kind == "number" for e in ledger) and "experiments" not in kinds:
        problems.append("the paper reports results: add an experiments section")
    problems += ownership_problems(spec, outline.sections, ledger)
    return problems


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
