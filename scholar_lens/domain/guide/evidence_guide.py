"""The evidence-based guide (docs/redesign.md, 3.2 and 5.1): an outline that
gives each piece of evidence to one section in the guide's order (TL;DR, the
problem, concepts and architecture before usage and operations, a closing on
when to choose it and how it fails), and the guide-only checks of a section
draft."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..evidence import EvidenceLedger, SourceDocument
from ..evidence_draft import MARKER, drop_empty_headings, prose_sentences
from ..lengths import DEPTH_CHARS, NON_KOREAN_CHAR_FACTOR

GuideKind = Literal[
    "tldr", "overview", "concepts", "architecture", "usage", "operations", "choosing"
]
Depth = Literal["deep", "standard", "brief"]
TLDR_TITLE = "TL;DR"

# The guide's order: what it is, the problem, how it works, how to use and
# run it, when to choose it. Kinds of one rank may interleave.
_RANK: dict[str, int] = {
    "tldr": 0,
    "overview": 1,
    "concepts": 2,
    "architecture": 2,
    "usage": 3,
    "operations": 3,
    "choosing": 4,
}
_SENTENCE_TITLE = re.compile(r"(?:다|요)[.!]?$")
# Explaining where a fact came from instead of stating it.
_SOURCE_NARRATION = re.compile(
    r"(?:공식 )?(?:문서|가이드|레퍼런스|튜토리얼)(?:에 따르면|에서는|에 (?:나온|나와 있는|적힌|소개된))|"
    r"(?:공식 )?문서(?:가|는) .{0,20}(?:설명|안내|권장)합니다|according to the (?:docs|documentation)",
    re.IGNORECASE,
)
_FENCE_OPEN = re.compile(
    r"^(`{3,}|~{3,})[ \t]*([\w+#-]*)[^\n]*\n(.*?)^\1[ \t]*$", re.M | re.S
)
# Languages of one SDK family: a guide shows one of them, not each twice.
_SDK_LANGUAGES = {
    "python": "python",
    "py": "python",
    "typescript": "typescript",
    "ts": "typescript",
    "javascript": "javascript",
    "js": "javascript",
    "java": "java",
    "go": "go",
    "rust": "rust",
}
MAX_CODE_LINES = 40
# The engine's source ids (D1 official, R1 reference, W1 third-party, O1
# older docs) never belong in the text.
_SOURCE_ID = re.compile(r"(?<![\w`./-])[DRWO]\d{1,3}(?![\w`./-])")
# Hedging about what the writer did not settle, instead of leaving it out.
_HEDGE = re.compile(
    r"(?:버전|환경|문서)에서 (?:직접 )?확인(?:해야|하세요|하십시오|해 보세요|이 필요)|"
    r"확정하지 (?:않았|못했)|확인하지 못했|확인되지 않았|(?:여기서는|이 가이드에서는) (?:다루지|확정하지)"
)
# An evidence marker standing for a word ("근거는 [[E3]]입니다"): removing
# it breaks the sentence.
_MARKER_AS_WORD = re.compile(
    MARKER.pattern + r"\s*(?:입니다|이다|이며|이고|은|는|이|가|을|를|의|에|와|과|로)"
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GuideSection(_Strict):
    kind: GuideKind
    # A claim as a noun phrase ("요청마다 다시 짜이는 배치"), not a topic label.
    title: str = Field(min_length=1)
    thesis: str = ""
    question: str = ""
    depth: Depth = "standard"
    evidence: list[str] = Field(default_factory=list)


class GlossaryTerm(_Strict):
    english: str
    rendering: str


class GuideOutline(_Strict):
    title: str = Field(min_length=1)
    glossary: list[GlossaryTerm] = Field(default_factory=list, max_length=20)
    # One application or workload threaded through the guide.
    running_example: str = ""
    # The SDK language the code uses ("python"); others get a sentence.
    code_language: str = ""
    sections: list[GuideSection] = Field(min_length=4)


def repair_guide_outline(
    outline: GuideOutline, ledger: EvidenceLedger
) -> tuple[GuideOutline, list[str]]:
    """What code can fix without asking again: unknown evidence ids dropped,
    evidence given twice kept in its first section, sections stably sorted
    into the guide's order, and the TL;DR titled, brief and without evidence
    of its own (it is written last, from what the sections cite)."""
    notes: list[str] = []
    seen: set[str] = set()
    sections = []
    for s in outline.sections:
        if s.kind == "tldr":
            sections.append(
                s.model_copy(
                    update={"title": TLDR_TITLE, "depth": "brief", "evidence": []}
                )
            )
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
        sections.append(s.model_copy(update={"evidence": kept}))
    ordered = sorted(sections, key=lambda s: _RANK[s.kind])
    if [s.kind for s in ordered] != [s.kind for s in sections]:
        notes.append("sections sorted into the guide's order")
    return outline.model_copy(update={"sections": ordered}), notes


def guide_outline_problems(
    outline: GuideOutline,
    ledger: EvidenceLedger,
    documents: Mapping[str, SourceDocument],
    *,
    title_max_length: int,
    max_sections: int,
) -> list[str]:
    """What makes a guide outline unusable: the order of its kinds, a
    missing TL;DR, architecture or closing section, too many sections,
    evidence unknown or given twice, a section without evidence, sentence
    titles, and the user's reference material left unused."""
    problems: list[str] = []
    kinds = [s.kind for s in outline.sections]
    if len(outline.title) > title_max_length:
        problems.append(f"the title is longer than {title_max_length} characters")
    if len(outline.sections) > max_sections:
        problems.append(f"at most {max_sections} sections")
    if kinds[0] != "tldr" or kinds.count("tldr") != 1:
        problems.append("the guide starts with exactly one tldr section")
    if kinds.count("overview") != 1:
        problems.append("exactly one overview section (the problem it solves)")
    if "architecture" not in kinds:
        problems.append("at least one architecture section (how it works inside)")
    if kinds[-1] != "choosing" or kinds.count("choosing") != 1:
        problems.append(
            "the guide ends with exactly one choosing section (when to choose it, "
            "the alternatives, how it fails)"
        )
    if [_RANK[k] for k in kinds] != sorted(_RANK[k] for k in kinds):
        problems.append(
            "sections must go tldr, overview, concepts and architecture, usage "
            "and operations, choosing"
        )
    seen: dict[str, str] = {}
    for s in outline.sections:
        if s.kind != "tldr" and _SENTENCE_TITLE.search(s.title.strip()):
            problems.append(f"title '{s.title}' is a sentence; write a noun phrase")
        if not s.evidence and s.kind != "tldr":
            problems.append(f"section '{s.title}' has no evidence")
        for eid in s.evidence:
            if ledger.get(eid) is None:
                problems.append(f"unknown evidence {eid} in '{s.title}'")
            elif eid in seen:
                problems.append(f"{eid} is in both '{seen[eid]}' and '{s.title}'")
            else:
                seen[eid] = s.title
    used_sources = {e.paper_id for eid in seen if (e := ledger.get(eid))}
    for sid, document in documents.items():
        if (
            document.authority == "reference"
            and ledger.of_paper(sid)
            and sid not in used_sources
        ):
            problems.append(
                f"the user's reference material {sid} ({document.title}) has "
                "evidence but no section uses it"
            )
    return problems


def guide_section_targets(
    outline: GuideOutline, *, min_total: int, max_total: int, korean: bool
) -> list[int]:
    """Prose characters per section by depth, the total scaled into
    ``[min_total, max_total]`` (Korean characters)."""
    raw = [DEPTH_CHARS[s.depth] for s in outline.sections]
    total = sum(raw)
    scale = min(max(total, min_total), max_total) / total
    factor = 1.0 if korean else NON_KOREAN_CHAR_FACTOR
    return [round(r * scale * factor) for r in raw]


def guide_draft_problems(markdown: str, *, code_language: str = "") -> list[str]:
    """Guide-only problems of a section draft: narrating which page a fact
    came from, the same code in two SDK languages, and code blocks longer
    than the paragraph around them can explain."""
    problems: list[str] = []
    for sentence in prose_sentences(markdown):
        plain = MARKER.sub("", sentence)
        if _SOURCE_ID.search(plain):
            problems.append(
                "names a source by its internal id (D1, W8); name the page or "
                f"leave it out: {plain[:160]}"
            )
        if _MARKER_AS_WORD.search(sentence):
            problems.append(
                "an evidence marker stands for a word; state the fact and put "
                f"the marker at the end of the sentence: {sentence[:160]}"
            )
        if _HEDGE.search(plain):
            problems.append(
                "hedges about what the guide did not settle; state what the "
                f"evidence shows or leave the point out: {plain[:160]}"
            )
        if _SOURCE_NARRATION.search(plain):
            problems.append(
                f"names the page a fact came from; state the fact itself: {plain[:160]}"
            )
    languages: set[str] = set()
    for m in _FENCE_OPEN.finditer(markdown):
        if family := _SDK_LANGUAGES.get(m.group(2).lower()):
            languages.add(family)
        if (lines := m.group(3).count("\n")) > MAX_CODE_LINES:
            problems.append(
                f"a {lines}-line code block; show the {MAX_CODE_LINES} or fewer "
                "lines the text explains and say what the rest does"
            )
    if len(languages) > 1:
        keep = code_language or "one language"
        problems.append(
            f"code in {sorted(languages)}: show it in {keep} and name the other "
            "SDK's difference in a sentence"
        )
    return problems


_LANGUAGE_ONLY = re.compile(
    r"\b(python|typescript|javascript|java|go|rust)\b\s*(?:sdk\s*)?(?:only|전용|에서만|만)"
    r"|(?:only|전용)\s*(?:in|for)?\s*\b(python|typescript|javascript|java|go|rust)\b",
    re.IGNORECASE,
)


def other_language_only(scope_and_claim: str, code_language: str) -> bool:
    """Whether evidence holds only for another SDK language than the
    guide's code ("TypeScript only" in a Python guide): stated in the
    guide's sections it reads as true for the language the guide uses."""
    family = _SDK_LANGUAGES.get(code_language.strip().lower(), "")
    if not family:
        return False
    for m in _LANGUAGE_ONLY.finditer(scope_and_claim):
        named = _SDK_LANGUAGES.get((m.group(1) or m.group(2)).lower(), "")
        if named and named != family:
            return True
    return False


def drop_other_language(
    outline: GuideOutline, ledger: EvidenceLedger
) -> tuple[GuideOutline, list[str]]:
    """The outline without evidence that holds only for another SDK
    language than its code language; returns the dropped ids."""
    dropped: list[str] = []
    sections = []
    for s in outline.sections:
        kept = []
        for eid in s.evidence:
            e = ledger.get(eid)
            if e is not None and other_language_only(
                f"{e.scope} {e.claim}", outline.code_language
            ):
                dropped.append(eid)
            else:
                kept.append(eid)
        sections.append(s.model_copy(update={"evidence": kept}))
    return outline.model_copy(update={"sections": sections}), dropped


_IMPORT = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import\s+([\w, ()*]+)|import\s+([\w.]+))", re.M
)


def _package_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def unverified_imports(
    markdown: str, official_text: str, products: Iterable[str]
) -> list[str]:
    """Import lines of the guide's Python code that load a module of the
    product (its top-level package starts with a product name: ``vllm``,
    ``ray``, ``strands`` and ``strands_tools``) under a path the official
    pages never show; an invented path fails as pasted. Empty
    ``official_text`` checks nothing."""
    if not official_text.strip():
        return []
    keys = [k for p in products if (k := _package_key(p))]
    known = official_text.lower()
    found: list[str] = []
    for m in _FENCE_OPEN.finditer(markdown):
        if m.group(2).lower() not in ("python", "py", ""):
            continue
        for imp in _IMPORT.finditer(m.group(3)):
            module = imp.group(1) or imp.group(3)
            top = _package_key(module.split(".")[0])
            if not any(top.startswith(k) for k in keys):
                continue
            if module.lower() not in known and module not in found:
                found.append(module)
    return found


def cited_sources(markdown: str, ledger: EvidenceLedger) -> list[str]:
    """Source ids the draft's markers cite, in order of first citation."""
    ids: list[str] = []
    for m in MARKER.finditer(markdown):
        for eid in re.findall(r"E\d+", m.group(1)):
            if (e := ledger.get(eid)) and e.paper_id not in ids:
                ids.append(e.paper_id)
    return ids


def strip_markers(markdown: str) -> str:
    """The body without evidence markers (a guide lists its sources at the
    end instead of citing each sentence), and without the lines and headings
    that only held markers."""
    out = re.sub(r"[ \t]*" + MARKER.pattern, "", markdown)
    out = re.sub(r"^[ \t]*[.,]?[ \t]*$\n?", "\n", out, flags=re.M)
    return drop_empty_headings(re.sub(r"\n{3,}", "\n\n", out)).rstrip() + "\n"
