"""The evidence-based guide (docs/redesign.md, 3.2 and 5.1): an outline that
gives each piece of evidence to one section in the guide's order (TL;DR, the
problem, concepts and architecture before usage and operations, a closing on
when to choose it and how it fails), and the guide-only checks of a section
draft."""

from __future__ import annotations

import ast
import builtins
import re
import symtable
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..evidence import EvidenceLedger, SourceDocument
from ..evidence_draft import MARKER, block_lines, drop_empty_headings, prose_sentences
from ..glossary import GlossaryTerm
from ..post_spec import (
    SDK_LANGUAGES,
    SectionBase,
    first_owner,
    ownership_problems,
    structure_problems,
)
from ..post_spec import other_sdk as other_sdk  # re-exported
from ..specs.guide import GUIDE_SPEC, TLDR_TITLE

GuideKind = Literal[
    "tldr", "overview", "concepts", "architecture", "usage", "operations", "choosing"
]

# The guide's order: what it is, the problem, how it works, how to use and
# run it, when to choose it. Kinds of one rank may interleave.
_RANK = GUIDE_SPEC.ranks
_FENCE_OPEN = re.compile(
    r"^(`{3,}|~{3,})[ \t]*([\w+#-]*)[^\n]*\n(.*?)^\1[ \t]*$", re.M | re.S
)
MAX_CODE_LINES = 40
# The engine's source ids ("src-D1" official, "src-R1" reference, "src-W1"
# third-party, "src-O1" older docs: a form no product or model name takes)
# never belong in the text.
SOURCE_ID_PREFIX = "src-"
_SOURCE_ID = re.compile(r"(?<![\w`./-])src-[DRWO]\d{1,3}(?![\w`./-])")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GuideSection(SectionBase):
    kind: GuideKind
    # The features or concepts this section teaches ("structured output",
    # "interventions"); each is taught in one section, others refer back.
    topics: list[str] = Field(default_factory=list, max_length=8)


class GuideOutline(_Strict):
    title: str = Field(min_length=1)
    glossary: list[GlossaryTerm] = Field(default_factory=list, max_length=20)
    # One application or workload threaded through the guide.
    running_example: str = ""
    # The SDK language the code uses ("python"); others get a sentence.
    code_language: str = ""
    sections: list[GuideSection] = Field(min_length=4)
    # Groups of evidence ids that state one fact in different words; code
    # keeps one of each, so the post states the fact once.
    same_facts: list[list[str]] = Field(default_factory=list, max_length=60)


def repair_guide_outline(
    outline: GuideOutline, ledger: EvidenceLedger
) -> tuple[GuideOutline, list[str]]:
    """What code can fix without asking again: unknown evidence ids dropped,
    evidence given twice kept in its first section, sections stably sorted
    into the guide's order, and the TL;DR titled, brief and without evidence
    of its own (it is written last, from what the sections cite)."""
    notes: list[str] = []
    seen: set[str] = set()
    owned: set[str] = set()
    sections = []
    for s in outline.sections:
        if s.kind == "tldr":
            sections.append(
                s.model_copy(
                    update={"title": TLDR_TITLE, "depth": "brief", "evidence": []}
                )
            )
            continue
        kept = first_owner(s.evidence, ledger, seen, s.title, notes)
        topics = []
        for topic in s.topics:
            key = " ".join(topic.lower().split())
            if key and key not in owned:
                owned.add(key)
                topics.append(topic)
            elif key:
                notes.append(f"topic {topic!r} kept in its first section")
        sections.append(s.model_copy(update={"evidence": kept, "topics": topics}))
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
    problems = structure_problems(
        GUIDE_SPEC,
        [s.kind for s in outline.sections],
        title=outline.title,
        title_max_length=title_max_length,
        max_sections=max_sections,
    )
    problems += ownership_problems(GUIDE_SPEC, outline.sections, ledger)
    seen = {eid for s in outline.sections for eid in s.evidence}
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


def guide_draft_problems(markdown: str, *, code_language: str = "") -> list[str]:
    """Guide-only problems of a section draft: an internal source id in the
    text, the same code in two SDK languages, and code blocks longer than
    the paragraph around them can explain."""
    problems: list[str] = []
    for line, role in zip(markdown.split("\n"), block_lines(markdown), strict=True):
        if role == "code" and MARKER.search(line):
            problems.append(
                "an evidence marker inside code; markers go at the end of prose "
                f"sentences only: {line.strip()[:120]}"
            )
    for sentence in prose_sentences(markdown):
        plain = MARKER.sub("", sentence)
        if _SOURCE_ID.search(plain):
            problems.append(
                "names a source by its internal id; name the page or leave it "
                f"out: {plain[:160]}"
            )
    languages: set[str] = set()
    for m in _FENCE_OPEN.finditer(markdown):
        if family := SDK_LANGUAGES.get(m.group(2).lower()):
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


# "pip install pkg[extra,extra2] other" (uv pip install too), outside prose.
_PIP_INSTALL = re.compile(r"(?:uv\s+)?pip3?\s+install\s+([^\n#;&|]+)")
_REQUIREMENT = re.compile(r"""^['"]?([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[([^\]]*)\])?""")


_PYTHON_INFO = ("python", "py", "python3")


def _bound_names(table: symtable.SymbolTable) -> set[str]:
    return {
        sym.get_name()
        for sym in table.get_symbols()
        if sym.is_assigned() or sym.is_imported() or sym.is_namespace()
    }


def _free_globals(table: symtable.SymbolTable) -> set[str]:
    """Names a block's code reads from module scope (directly, or in a
    function or class body) without binding them there."""
    names: set[str] = set()
    for sym in table.get_symbols():
        name = sym.get_name()
        if table.get_type() == "module":
            if sym.is_referenced() and not (
                sym.is_assigned() or sym.is_imported() or sym.is_namespace()
            ):
                names.add(name)
        elif sym.is_global() and sym.is_referenced():
            names.add(name)
    for child in table.get_children():
        names |= _free_globals(child)
    return names


# A fenced block at any indentation (a list item's code too).
_ANY_FENCE = re.compile(
    r"^([ \t]*)(`{3,}|~{3,})[ \t]*([\w+#-]*)[^\n]*\n(.*?)^\1\2[ \t]*$", re.M | re.S
)
_PROMPT = re.compile(r"^(?:>>> |\.\.\. )")
# Module-level definitions of a block that does not parse (best effort).
_DEFINED_LINE = re.compile(
    r"^(?:async\s+)?(?:def|class)\s+(\w+)|^(\w+)\s*(?::[^=]+)?=(?!=)|"
    r"^(?:from\s+\S+\s+)?import\s+(.+)$",
    re.M,
)


def python_blocks(markdown: str) -> list[str]:
    """The draft's Python code blocks (at any indentation), dedented."""
    return [
        textwrap.dedent(m.group(4))
        for m in _ANY_FENCE.finditer(markdown)
        if m.group(3).lower() in _PYTHON_INFO
    ]


def _session_code(code: str) -> str | None:
    """A REPL transcript or notebook cell as the code it runs (prompts and
    ``%``/``!`` lines left out); None for a plain script."""
    lines = code.split("\n")
    if not any(ln.lstrip().startswith((">>>", "%", "!")) for ln in lines):
        return None
    return "\n".join(
        _PROMPT.sub("", ln)
        for ln in lines
        if not ln.lstrip().startswith(("%", "!"))
        and (_PROMPT.match(ln) or not ln.strip())
    )


def _recovered_names(code: str) -> set[str]:
    names: set[str] = set()
    for m in _DEFINED_LINE.finditer(code):
        if m.group(1) or m.group(2):
            names.add(m.group(1) or m.group(2))
        elif m.group(3):
            for part in m.group(3).split(","):
                words = part.split()
                if words:
                    names.add(words[-1].split(".")[0].strip("()"))
    return names


def python_code_problems(
    sections: Sequence[tuple[str, str, bool]],
) -> list[tuple[str, str]]:
    """(section title, problem) for the guide's Python code read in order,
    as a reader pastes it: a block that does not parse, and a name a block
    uses that no block so far (nor Python itself) defines. ``sections`` are
    (title, markdown, checked): every block's definitions count (a REPL or
    notebook block's too, and a best-effort reading of a block that does not
    parse), problems are reported for ``checked`` sections only (the running
    example's usage, not an internals excerpt). After a star import later
    names are unknowable and are not reported. Parsing and scopes come from
    the standard library (``ast``, ``symtable``)."""
    known = set(dir(builtins)) | {"__name__", "__file__"}
    reliable = True
    problems: list[tuple[str, str]] = []
    for title, markdown, checked in sections:
        for block in python_blocks(markdown):
            session = _session_code(block)
            code = block if session is None else session
            report = checked and session is None
            try:
                tree = ast.parse(code)
                table = symtable.symtable(code, "<guide>", "exec")
            except (SyntaxError, ValueError) as e:
                known |= _recovered_names(code)
                if report:
                    reason = e.msg if isinstance(e, SyntaxError) else str(e)
                    problems.append(
                        (
                            title,
                            f"a python block does not parse ({reason}): "
                            f"{code.strip()[:120]}",
                        )
                    )
                continue
            if any(
                isinstance(node, ast.ImportFrom)
                and any(alias.name == "*" for alias in node.names)
                for node in ast.walk(tree)
            ):
                reliable = False
            bound = _bound_names(table)
            missing = sorted(_free_globals(table) - bound - known)
            if report and reliable and missing:
                problems.append(
                    (
                        title,
                        "a python block uses names no earlier block or import "
                        f"defines: {', '.join(missing)}; define or import them "
                        "where a reader pasting the blocks in order has them",
                    )
                )
            known |= bound
    return problems


def install_requirements(text: str) -> list[tuple[str, frozenset[str]]]:
    """(package, extras) of every ``pip install`` the text shows, options
    and URLs left out."""
    found: list[tuple[str, frozenset[str]]] = []
    for m in _PIP_INSTALL.finditer(text):
        for arg in m.group(1).split():
            if arg.startswith("-") or "/" in arg or arg.startswith("."):
                continue
            if r := _REQUIREMENT.match(arg):
                extras = frozenset(
                    e.strip().lower()
                    for e in (r.group(2) or "").split(",")
                    if e.strip()
                )
                found.append((r.group(1).lower(), extras))
    return found


def unknown_extras(markdown: str, provided: Mapping[str, frozenset[str]]) -> list[str]:
    """``pkg[extra]`` the guide installs that the package index does not
    provide (known packages only)."""
    bad: list[str] = []
    for package, extras in install_requirements(markdown):
        if package in provided:
            for extra in sorted(extras - provided[package]):
                if (item := f"{package}[{extra}]") not in bad:
                    bad.append(item)
    return bad


def cited_sources(markdown: str, ledger: EvidenceLedger) -> list[str]:
    """Source ids the draft's markers cite, in order of first citation."""
    ids: list[str] = []
    for m in MARKER.finditer(markdown):
        for eid in re.findall(r"E\d+", m.group(1)):
            if (e := ledger.get(eid)) and e.paper_id not in ids:
                ids.append(e.paper_id)
    return ids


def name_sources(markdown: str, titles: Mapping[str, str]) -> str:
    """Internal source ids the writer let into the text ("W2 페이지") as the
    source's title, outside code; unknown ids are left alone."""

    def repl(m: re.Match[str]) -> str:
        return titles.get(m.group(0), m.group(0))

    parts = re.split(
        r"(^(?:`{3,}|~{3,}).*?^(?:`{3,}|~{3,})[ \t]*$|`[^`\n]+`)",
        markdown,
        flags=re.M | re.S,
    )
    return "".join(
        part if i % 2 else _SOURCE_ID.sub(repl, part) for i, part in enumerate(parts)
    )


_MARKER_RUN = re.compile(r"[ \t]*" + MARKER.pattern, MARKER.flags)
_ONLY_PUNCTUATION = re.compile(r"^[ \t]*[.,]?[ \t]*$")


def strip_markers(markdown: str) -> str:
    """The body without evidence markers (a guide lists its sources at the
    end instead of citing each sentence), and without the lines and headings
    that only held markers."""
    out: list[str] = []
    for line, role in zip(markdown.split("\n"), block_lines(markdown), strict=True):
        if role == "code":  # a "." tree root or double blank lines stay
            out.append(line)
            continue
        bare = _MARKER_RUN.sub("", line)
        if bare != line and _ONLY_PUNCTUATION.match(bare):
            bare = ""
        if not bare.strip() and out and not out[-1].strip():
            continue  # one blank line between prose blocks
        out.append(bare)
    return drop_empty_headings("\n".join(out)).rstrip() + "\n"
