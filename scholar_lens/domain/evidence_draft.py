"""Deterministic checks and edits of an evidence-cited section draft
(docs/redesign.md, 3.3), shared by every artifact the evidence engine writes.

A draft is Markdown whose sentences cite their evidence as ``[[E12]]``; the
checks quote the sentence they object to, so their messages can go back to
the writer as concrete problems.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

from scholar_lens.domain.evidence import Evidence, EvidenceLedger, near_duplicate
from scholar_lens.domain.markdown.lint import drop_empty_headings
from scholar_lens.domain.markdown.math import DISPLAY_MATH, fence_roles
from scholar_lens.domain.post_spec import Limits
from scholar_lens.domain.prompt_safety import ZERO_WIDTH_SPACE, strip_defanging
from scholar_lens.domain.quality_gate import rule_issues
from scholar_lens.domain.validation import unsourced_numbers
from scholar_lens.domain.verifiers import Problem, item_problems

# An evidence marker ("[[E12]]", "[[E12, E15]]") or the illustration mark
# below; code removes both before publishing.
MARKER = re.compile(
    r"\[\[\s*((?:E\d+|EX)(?:\s*,\s*(?:E\d+|EX))*)\s*\]\]", re.IGNORECASE
)


# The writer's mark on a sentence that is its own illustration ("가상의 배치
# 32라면 …"): its numbers need no evidence; code removes the mark with the
# evidence markers.
EXAMPLE_MARK = "[[EX]]"
_EXAMPLE_MARK = re.compile(r"\[\[[^\]]*\bEX\b[^\]]*\]\]", re.IGNORECASE)


_ID = re.compile(r"E\d+", re.IGNORECASE)


# A display-math line: ``$$…$$`` alone on its line, or a ``$$`` that opens or
# closes a multi-line display (a list or quote prefix allowed).
_INLINE_CODE = re.compile(r"(`+)(?:(?!\1).)+?\1")
_MATH_PREFIX = re.compile(r"^[ \t]*(?:>[ \t]?)*(?:(?:[-*+]|\d+[.)])[ \t]+)?")
_ONE_LINE_MATH = re.compile(r"^\$\$(?:(?!\$\$).)+\$\$$")


def block_lines(markdown: str) -> list[str]:
    """Each line's role: ``"code"`` (a fenced block, fences included, at any
    indentation), ``"math"`` (a display equation) or ``"prose"``."""
    lines = markdown.split("\n")
    roles: list[str] = []
    in_math = False
    for line, fence in zip(lines, fence_roles(lines), strict=True):
        if fence != "prose":
            roles.append("code")
            continue
        bare = line[_MATH_PREFIX.match(line).end() :].strip()  # type: ignore[union-attr]
        marks = _INLINE_CODE.sub("", bare).count("$$")
        if in_math:
            # Any "$$" closes the display ("$$." and "$$ 여기서 …" too).
            roles.append("math")
            in_math = marks % 2 == 0
        elif _ONE_LINE_MATH.match(bare):
            roles.append("math")
        elif marks % 2 == 1:
            # An odd "$$" opens a display ("$$a$$ 는 …" is inline).
            roles.append("math")
            in_math = True
        else:
            roles.append("prose")
    return roles


def _blanked(markdown: str, roles: Iterable[str]) -> str:
    """``markdown`` with the lines of the given roles emptied (line count
    kept)."""
    drop = set(roles)
    return "\n".join(
        "" if role in drop else line
        for line, role in zip(markdown.split("\n"), block_lines(markdown), strict=True)
    )


def without_code(markdown: str) -> str:
    return _blanked(markdown, ("code",))


def without_blocks(markdown: str) -> str:
    """``markdown`` without code and display math (empty lines instead)."""
    return _blanked(markdown, ("code", "math"))


_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)[^)]*\)")


_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$", re.M)


_SENTENCE_END = re.compile(r"(?<=[.!?。])\s+|\n+")


_CITATION_KEY = re.compile(r"\[@[^\]]+\]")


# An arXiv id or DOI written as a paper's name in prose.
_PAPER_ID = re.compile(r"(?<![\d./])(?:\d{4}\.\d{4,5}|10\.\d{4,9}/[^\s,;)]+)(?![\d])")


def prose_sentences(markdown: str) -> list[str]:
    """The prose sentences of a draft: code, display math, tables, headings
    and images left out."""
    text = _TABLE_LINE.sub("\n", without_blocks(markdown))
    text = _IMAGE.sub(" ", text)
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^(?:>\s*)+|^(?:[-*+]|\d+[.)])\s+", "", line)
        out += [s.strip() for s in _SENTENCE_END.split(line) if s.strip()]
    return out


def prose_length(markdown: str) -> int:
    """The characters of prose every length check counts (one definition:
    :func:`prose_sentences` joined by single spaces)."""
    return len(" ".join(prose_sentences(markdown)))


def table_rows(markdown: str) -> list[str]:
    """The data rows of the draft's tables (header and rule rows left out),
    as written: what a claim check reads besides the prose."""
    rows = []
    for block in re.split(r"\n(?!\s*\|)", without_code(markdown)):
        lines = [ln.strip() for ln in block.split("\n") if _TABLE_LINE.match(ln)]
        if len(lines) >= 3 and re.match(r"^\|?\s*:?-{3,}", lines[1]):
            rows += lines[2:]
    return rows


def claim_units(markdown: str) -> list[str]:
    """What a claim check judges: the prose sentences, then the table rows."""
    return prose_sentences(markdown) + table_rows(markdown)


def cited_ids(sentence: str) -> list[str]:
    return [
        i.upper() for m in MARKER.finditer(sentence) for i in _ID.findall(m.group(1))
    ]


def _plain(sentence: str) -> str:
    return _CITATION_KEY.sub("", MARKER.sub("", _EXAMPLE_MARK.sub("", sentence)))


def is_illustration(sentence: str) -> bool:
    """Whether the writer marked ``sentence`` as its own illustration."""
    return bool(_EXAMPLE_MARK.search(sentence))


class FreeNumbers(str):
    """Numbers a sentence may state without evidence (the string the checks
    read), and the name-version pairs the sources write (``names``,
    lower-cased "llama 3.1"): such a pair in a sentence names a model or a
    library, it states no number."""

    names: frozenset[str] = frozenset()

    def with_names(self, names: Iterable[str]) -> FreeNumbers:
        out = FreeNumbers(self)
        out.names = self.names | {n.lower() for n in names}
        return out


def free_numbers(extra: Iterable[object] = ()) -> FreeNumbers:
    """Numbers a sentence may state without evidence: small counts (0-10)
    and ``extra`` (a survey's publication years)."""
    return FreeNumbers(
        " ".join([*(str(n) for n in range(11)), *(str(x) for x in extra)])
    )


# Numbers glued into a name, not a stated quantity: digits joined to a
# Latin word (ResNet-50, GPT-4o, T5, Llama-3.1-8B). A spaced version
# ("Llama 3.1") is told from a score ("MMLU 86.4") by the sources:
# :func:`named_versions`.
_NAMED_NUMBER = re.compile(
    r"\b(?:[A-Za-z]{2,}[A-Za-z0-9]*-?|[A-Z])\d[\w.]*(?:-[A-Za-z0-9.]+)*"
)
# A capitalised Latin name and the dotted version after it, as sources
# write models and libraries ("Llama 3.1", "Python 3.12").
_NAME_VERSION = re.compile(r"\b[A-Z][A-Za-z]+ \d+\.\d+(?:\.\d+)?\b")


def named_versions(texts: Iterable[str]) -> list[str]:
    """The name-version pairs the sources write ("Llama 3.1"): a sentence
    that writes the same pair names that model; the version alone, or the
    pair's number elsewhere, is still a stated number."""
    found: dict[str, None] = {}
    for text in texts:
        for m in _NAME_VERSION.finditer(text):
            found.setdefault(m.group(0), None)
    return list(found)


def _claim_text(sentence: str, free: str = "") -> str:
    """``sentence`` as its numbers are checked: markers, keys, glued names
    and the sources' name-version pairs (``free.names``) out."""
    text = _NAMED_NUMBER.sub(" ", _plain(sentence))
    for name in getattr(free, "names", ()):
        text = re.sub(rf"(?i)\b{re.escape(name)}(?![\d.])", " ", text)
    return text


def stated_numbers(sentence: str, free: str) -> list[str]:
    """The numbers a sentence states beyond ``free`` (years, small counts),
    names left out (:func:`_claim_text`)."""
    return unsourced_numbers(_claim_text(sentence, free), free)


# A table cell a writer left blank: fill it from the evidence or drop the
# row or column.
_BLANK_CELL = re.compile(r"\|\s*(?:—|–|-|N/A|n/a)?\s*(?=\|)")


# A marker written as a word of its sentence ("근거는 [[E3]]입니다"): removing it
# breaks the sentence.
_MARKER_AS_WORD = re.compile(MARKER.pattern + r"(?=[^\W\d_])")


# A parameter count names a model ("Llama 2 7B", "350M"), not a result; a
# token count ("115k") is a fact like any other and is not repeated.
_MODEL_SIZE = re.compile(r"^\d+(?:\.\d+)?[BM]$")


def visual_count(markdown: str) -> int:
    """The section's images, display equations and tables (a header row
    over a delimiter row: a ``---`` rule is not a table)."""
    body = without_code(markdown)
    tables = len(assets(markdown)["table"])
    roles = block_lines(body)
    equations = sum(
        1
        for i, role in enumerate(roles)
        if role == "math" and (i == 0 or roles[i - 1] != "math")
    )
    return len(_IMAGE.findall(body)) + equations + tables


# A heading of a subsection (``###`` and deeper) outside code.
_SUBHEADING = re.compile(r"^(#{3,6}) (.+?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")
# The problem kind of a subsection too thin to stand under its own heading.
THIN_SUBSECTION_PROBLEM = "thin_subsection"


def _thin_headings(markdown: str) -> list[tuple[int, int, str]]:
    """``(ordinal, line, title)`` of each thin subsection's heading: its
    place among the draft's headings, its line and its text (see
    :func:`thin_subsections`)."""
    lines = markdown.split("\n")
    roles = block_lines(markdown)
    headings = [
        i
        for i, (line, role) in enumerate(zip(lines, roles, strict=True))
        if role == "prose" and re.match(r"^#{1,6} ", line)
    ]
    thin = []
    for ordinal, i in enumerate(headings):
        m = _SUBHEADING.match(lines[i])
        if not m:
            continue
        level = len(m.group(1))
        end = headings[ordinal + 1] if ordinal + 1 < len(headings) else len(lines)
        if end < len(lines):
            child = re.match(r"^(#+) ", lines[end])
            if child and len(child.group(1)) > level:
                continue
        body = "\n".join(lines[i + 1 : end])
        own = any(r != "prose" for r in roles[i + 1 : end])
        own = own or visual_count(body) > 0
        own = own or any(_LIST_ITEM.match(ln) for ln in lines[i + 1 : end])
        if not own and len(prose_sentences(body)) <= 1:
            thin.append((ordinal, i, m.group(2)))
    return thin


def thin_subsections(markdown: str) -> list[str]:
    """The headings of subsections (``###`` and deeper) with at most one
    prose sentence and nothing else of their own (no figure, equation,
    table, code block or list) before the next heading: what deleting
    sentences leaves behind, or a heading over a single remark. A
    subsection whose content is its own child subsections is not thin."""
    return [title for _, _, title in _thin_headings(markdown)]


def merge_thin_subsections(before: str, after: str) -> str:
    """``after`` (``before`` with sentences deleted, its headings all kept)
    without the headings of the subsections the deletions left thin
    (:func:`thin_subsections`): the sentence left joins the text above it.
    Headings are matched by their place, so a subsection already thin in
    ``before`` (the writer's) stays, and so does another of the same
    title."""
    was_thin = {ordinal for ordinal, _, _ in _thin_headings(before)}
    drop = {
        line for ordinal, line, _ in _thin_headings(after) if ordinal not in was_thin
    }
    if not drop:
        return after
    kept = [line for i, line in enumerate(after.split("\n")) if i not in drop]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept))


# The problem kind of a section below its depth's visuals minimum.
VISUALS_PROBLEM = "visuals"
# The problem kind of a section whose prose is below its minimum length.
TOO_SHORT_PROBLEM = "too_short"
# The asset kinds (:func:`assets`) that count as a section's visuals.
_VISUAL_ASSETS = ("table", "figure", "equation")


def section_problems(
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
    exhausted: Iterable[str] = (),
    papers: Iterable[str] = (),
    target_chars: int = 0,
    passages: Mapping[tuple[str, str], str] | None = None,
    limits: Limits | None = None,
    minimum_length: bool = True,
    subsections: bool = True,
) -> list[Problem]:
    """Deterministic problems of a section draft (each quotes its sentence):
    a missing heading, unknown or foreign evidence ids, a number without
    evidence or not in the cited evidence, a number an earlier section
    already stated, a sentence about the post's own inputs, an unassigned or
    missing figure. What a sentence means (talk about the post's sources, a
    claim that a source leaves something out, style) is the claim check's
    and the review's, in any language. ``limits`` gives the length shares
    and the visuals by depth (the spec's); ``minimum_length``: whether the
    section has a minimum length (the TL;DR has none); ``subsections``:
    whether its subsections are the writer's (checked for thin ones), not a
    fixed shape (the TL;DR's questions)."""
    limits = limits or Limits()
    problems: list[Problem] = []

    def add(kind: str, text: str, where: str = "") -> None:
        problems.append(Problem(kind, text, where))

    def each(
        kind: str, items: Iterable[str], before: str, after: str, where: str = ""
    ) -> None:
        problems.extend(item_problems(kind, items, before, after, where))

    if drop_empty_headings(markdown) != markdown:
        add("empty_heading", "a heading with nothing under it; delete it or fill it")
    if not markdown.lstrip().startswith("## "):
        add("no_heading", f"the section must start with its '## ' heading ({title})")
    allowed = set(own) | set(earlier)
    for sentence in prose_sentences(markdown):
        quoted = _plain(sentence)[:160]
        ids = cited_ids(sentence)
        if unknown := [i for i in ids if ledger.get(i) is None]:
            each(
                "unknown_evidence",
                unknown,
                "unknown evidence ",
                f" in: {quoted}",
                quoted,
            )
            continue
        if _MARKER_AS_WORD.search(sentence):
            add(
                "marker_as_word",
                "an evidence marker stands for a word; state the fact and put "
                f"the marker at the end of the sentence: {sentence[:160]}",
                sentence[:160],
            )
        if spent := [i for i in ids if i in set(exhausted)]:
            each(
                "exhausted_evidence",
                spent,
                "evidence ",
                f" was already referred back to once; do not restate it again: "
                f"{quoted}",
                quoted,
            )
        if foreign := [i for i in ids if i not in allowed]:
            each(
                "foreign_evidence",
                foreign,
                "evidence ",
                f" belongs to another section; leave it out: {quoted}",
                quoted,
            )
        numbers = stated_numbers(sentence, free_numbers)
        example = is_illustration(sentence)
        if numbers and not ids and not example:
            each(
                "numbers_uncited",
                numbers,
                "numbers ",
                f" without evidence (cite it; an illustration of your own ends "
                f"with {EXAMPLE_MARK}): {quoted}",
                quoted,
            )
        elif numbers and not example:
            support = _support(ids, ledger, passages)
            if missing := unsourced_numbers(
                _claim_text(sentence, free_numbers), f"{support} {free_numbers}"
            ):
                each(
                    "numbers_not_in_evidence",
                    missing,
                    "numbers ",
                    f" are not in the cited evidence: {quoted}",
                    quoted,
                )
        if not example:
            for n in dict.fromkeys(numbers):
                if n in earlier_numbers and not _MODEL_SIZE.match(n):
                    problems.append(
                        Problem(
                            "numbers_repeated",
                            f"[{n}] already stated in '{earlier_numbers[n]}': "
                            f"refer back without repeating: {quoted}",
                            quoted,
                            item=n,
                        )
                    )
        if _PAPER_ID.search(_plain(sentence)):
            add(
                "paper_id",
                f"names a paper by its id; use its name or its authors: {quoted}",
                quoted,
            )
    for row in table_rows(markdown):
        plain_row = _plain(row)
        if _BLANK_CELL.search(plain_row):
            add(
                "blank_cell",
                "a table cell is blank ('—', or only evidence markers, which code "
                "removes): give the value the evidence or the passages state, or "
                "drop the row or column it belongs to; markers go after a cell's "
                "content: "
                f"{plain_row[:160]}",
                plain_row[:160],
            )
    cited_papers = {
        e.paper_id
        for s in prose_sentences(markdown)
        for i in cited_ids(s)
        if (e := ledger.get(i))
    }
    if uncovered := [p for p in papers if p not in cited_papers]:
        each(
            "papers_uncovered",
            uncovered,
            "papers ",
            " belong to this section but no sentence cites their evidence: "
            "cover each",
        )
    length = prose_length(markdown)
    if target_chars and length > limits.max_length_share * target_chars:
        add(
            "too_long",
            f"{length:,} characters of prose for a target of {target_chars:,}: "
            "cut background, repetition and minor detail",
        )
    if (
        minimum_length
        and target_chars >= limits.min_length_floor
        and length < limits.min_length_share * target_chars
    ):
        add(
            TOO_SHORT_PROBLEM,
            f"{length:,} characters of prose for a target of {target_chars:,} "
            f"(below {limits.min_length_share:.0%}): develop the section from "
            "its evidence (the mechanism, an example, what it means for the "
            "reader)",
        )
    if ends_mid_sentence(markdown):
        add("mid_sentence", "the section stops mid-sentence: finish it")
    each(
        THIN_SUBSECTION_PROBLEM,
        thin_subsections(markdown) if subsections else (),
        "subsections with one sentence or less and no figure, table, code or "
        "list of their own: develop each from its evidence or drop its heading "
        "so the text joins the part above: ",
        "",
    )
    # The blog's Markdown and math rules (an unbalanced $$ breaks every
    # later equation of the post).
    # A rule hit is "rule 'offending text': context"; the context is its
    # position (what quality_gate.added_slips also leaves out), so an edit
    # nearby that leaves the hit as it is adds no problem.
    problems += [
        Problem("rule", issue, issue.split(":", 1)[1] if ":" in issue else "")
        for issue in rule_issues(MARKER.sub("", markdown))
    ]
    if (need := dict(limits.min_visuals).get(depth, 0)) and visual_count(
        markdown
    ) < need:
        add(
            VISUALS_PROBLEM,
            f"a {depth} section needs at least {need} visuals (its figures, an "
            "equation, or a small table from its number evidence)",
        )
    expected = set(figure_urls)
    shown = set(_IMAGE.findall(markdown))
    each(
        "extra_images",
        sorted(shown - expected),
        "images not given to this section: ",
        "",
    )
    each(
        "missing_figures",
        sorted(expected - shown),
        "figures given to this section are missing: ",
        "",
    )
    return problems


def draft_problems(markdown: str, **checks: Any) -> list[str]:
    """The messages of :func:`section_problems` (same arguments)."""
    return [p.text for p in section_problems(markdown, **checks)]


def section_heading(markdown: str) -> str:
    """The text of the draft's first ``## `` heading ("" without one)."""
    m = re.search(r"^## (.+?)\s*$", markdown, re.M)
    return m.group(1) if m else ""


def heading_names(markdown: str) -> set[str]:
    """The text of every heading (``##`` and deeper) of the draft outside
    code: a whole-post finding may name the section or any of its
    subsections."""
    return {
        m.group(1)
        for m in re.finditer(r"^#{2,6} (.+?)\s*#*\s*$", without_code(markdown), re.M)
    }


def _support(
    ids: Iterable[str],
    ledger: EvidenceLedger,
    passages: Mapping[tuple[str, str], str] | None,
) -> str:
    """What the cited evidence states: its quote, LaTeX, code and table
    header, its claim and cell (their numbers are checked against the quote
    when recorded) and, with ``passages``, the whole source section each item
    is quoted from (a table's other values, a number in the next sentence).
    The scope is the reader's own description and supports nothing."""
    parts = []
    for i in ids:
        if (e := ledger.get(i)) is not None:
            parts.append(
                f"{e.quote} {e.latex} {e.code} {e.table_header} {e.claim} {e.cell}"
            )
            if passages:
                parts.append(passages.get((e.paper_id, e.anchor), ""))
    return " ".join(parts)


def unsourced_number_sentences(
    markdown: str,
    ledger: EvidenceLedger,
    free_numbers: str,
    passages: Mapping[tuple[str, str], str] | None = None,
) -> list[str]:
    """Sentences stating numbers their cited evidence does not (or citing
    none); a sentence marked as the writer's own example is left alone."""
    found = []
    for sentence in prose_sentences(markdown):
        numbers = stated_numbers(sentence, free_numbers)
        if not numbers or is_illustration(sentence):
            continue
        support = _support(cited_ids(sentence), ledger, passages)
        if unsourced_numbers(
            _claim_text(sentence, free_numbers), f"{support} {free_numbers}"
        ):
            found.append(sentence)
    return found


def add_missing_figures(markdown: str, figures: Sequence[Evidence]) -> str:
    """The section with each assigned figure the writer left out appended
    as an image (its claim as alt text)."""
    shown = set(_IMAGE.findall(markdown))
    # The alt text says what it shows; the paper's caption is not copied
    # (it names the paper's own figure numbers).
    extra = [
        f"![{' '.join(f.claim.split())}]({f.figure_url})"
        for f in figures
        if f.figure_url and f.figure_url not in shown
    ]
    return markdown.rstrip() + ("\n\n" + "\n\n".join(extra) + "\n" if extra else "\n")


def section_numbers(markdown: str, free_numbers: str) -> set[str]:
    return {
        n for s in prose_sentences(markdown) for n in stated_numbers(s, free_numbers)
    }


# What is left of a line once its only sentence is removed: a list marker,
# a quote mark, punctuation.
_LEFTOVER = re.compile(r"^[\s>*+\-\d.)]*[.,;:!?。]?\s*$")


def drop_sentences(markdown: str, sentences: Iterable[str]) -> str:
    """``markdown`` without the given sentences (as :func:`prose_sentences`
    and :func:`table_rows` return them), each removed once from the first
    prose line (or table row) that holds it; a line it leaves empty goes
    with it. Code and display math are never touched."""
    # A line a removal empties is None until the sentence's pass ends.
    lines: list[str | None] = list(markdown.split("\n"))
    roles = block_lines(markdown)
    for sentence in sentences:
        row = sentence.startswith("|")
        for i, (line, role) in enumerate(zip(lines, roles, strict=True)):
            # A heading is never a sentence (prose_sentences skips it), even
            # when a claim unit repeats its words.
            if role != "prose" or line is None or line.lstrip().startswith("#"):
                continue
            if sentence not in line:
                # A sentence around an inline image reads with the image
                # blanked (as prose_sentences gives it): the line's text goes,
                # its images stay.
                if not row and sentence in _IMAGE.sub(" ", line):
                    rest = _IMAGE.sub(" ", line).replace(sentence, "", 1)
                    if _LEFTOVER.match(rest):
                        lines[i] = " ".join(m.group(0) for m in _IMAGE.finditer(line))
                        break
                continue
            if row:
                if line.strip() == sentence:
                    lines[i] = None
                    break
                continue
            rest = line.replace(sentence, "", 1)
            if _LEFTOVER.match(rest):
                lines[i] = None
            else:
                lead = line[: len(line) - len(line.lstrip())]
                lines[i] = lead + re.sub(r"[ \t]{2,}", " ", rest.strip())
            break
        kept = [(ln, r) for ln, r in zip(lines, roles, strict=True) if ln is not None]
        lines = [ln for ln, _ in kept]
        roles = [r for _, r in kept]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(ln for ln in lines if ln is not None))


_HEADINGS = re.compile(r"^#{2,4} .+$", re.M)
# A table's delimiter row ("| --- | :-: |", "|:-|-:|").
_DELIMITER_ROW = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$")


def _squeezed(text: str) -> str:
    return " ".join(text.split())


def assets(markdown: str) -> dict[str, list[str]]:
    """What a revision may not lose without a copy left, by kind, each item
    with its whitespace collapsed: the code blocks (their body), tables
    (their header row), figures (their address) and display equations.

    Args:
        markdown: A section or a whole post.

    Returns:
        Kind ("code block", "table", "figure", "equation") -> its items, in
        order (an item occurs as often as the text has it).
    """
    lines = markdown.split("\n")
    blocks: list[str] = []
    body: list[str] | None = None
    for line, role in zip(lines, fence_roles(lines), strict=True):
        if role == "open":
            body = []
        elif role == "code" and body is not None:
            body.append(line)
        elif role == "close" and body is not None:
            blocks.append(_squeezed("\n".join(body)))
            body = None
    if body is not None:
        blocks.append(_squeezed("\n".join(body)))
    prose = without_code(markdown)
    rows = prose.split("\n")
    tables = [
        _squeezed(rows[i])
        for i in range(len(rows) - 1)
        if _TABLE_LINE.match(rows[i])
        and "|" in rows[i + 1]
        and _DELIMITER_ROW.match(rows[i + 1])
    ]
    return {
        "code block": blocks,
        "table": tables,
        "figure": _IMAGE.findall(prose),
        "equation": [_squeezed(m.group(0)) for m in DISPLAY_MATH.finditer(prose)],
    }


# The asset kind a reader runs in order: its copy must come first.
_CODE_BLOCK = "code block"


def lost_assets(
    before: str,
    after: str,
    rest: str = "",
    owned_figures: Collection[str] | None = None,
    later: str = "",
) -> str:
    """The first kind of asset ``after`` lost with no copy left ("" when
    none): a revision may fix an item in place (the count stays), and may
    drop a repeated one when an equivalent copy stays in ``after`` or in
    ``rest`` (the rest of the post). A code block is run in reading order:
    its copy must stay in ``after`` or in a section before this one (a copy
    only in ``later`` comes after the code that uses it).

    Args:
        before: The text before the revision.
        after: The text after it.
        rest: The other sections of the post.
        owned_figures: The figures given to the section; a figure not among
            them may go (None: every figure is the section's).
        later: The part of ``rest`` after this section ("": all of
            ``rest`` comes before it).

    Returns:
        "code block", "table", "figure", "equation" or "".
    """
    old, new = assets(before), assets(after)
    elsewhere = assets(rest) if rest else {}
    if later and elsewhere:
        before_this = Counter(elsewhere[_CODE_BLOCK]) - Counter(
            assets(later)[_CODE_BLOCK]
        )
        elsewhere[_CODE_BLOCK] = list(before_this.elements())
    for kind, items in old.items():
        if len(new[kind]) >= len(items):
            continue
        kept = set(new[kind]) | set(elsewhere.get(kind, ()))
        gone = [i for i in Counter(items) - Counter(new[kind]) if i not in kept]
        if kind == "figure" and owned_figures is not None:
            gone = [i for i in gone if i in owned_figures]
        if gone:
            return kind
    return ""


def removed_duplicate_visuals(before: str, after: str, rest: str) -> bool:
    """Whether ``after`` removed visuals of ``before`` and each one removed
    has a copy in ``rest`` (another section owns it): a repetition's
    deletion, which takes nothing from the post.

    Args:
        before: The section before the revision.
        after: The section after it.
        rest: The other sections of the post.

    Returns:
        True when visuals were removed, all with a copy in ``rest``.
    """
    old, new, elsewhere = assets(before), assets(after), assets(rest)
    removed = [
        (kind, item)
        for kind in _VISUAL_ASSETS
        for item in Counter(old[kind]) - Counter(new[kind])
    ]
    return bool(removed) and all(
        item in elsewhere.get(kind, ()) for kind, item in removed
    )


def structure_lost(
    before: str,
    after: str,
    *,
    rest: str = "",
    max_prose_cut: float,
    owned_figures: Collection[str] | None = None,
    later: str = "",
) -> str:
    """Why a revision lost the draft's structure ("" when it did not): a
    heading gone, a code block, table, figure or display equation gone with
    no copy left in the section or in ``rest`` (:func:`lost_assets`), or
    more than ``max_prose_cut`` of the prose cut.

    Args:
        before: The draft.
        after: The revision.
        rest: The other sections of the post.
        max_prose_cut: The largest share of the prose a revision may cut.
        owned_figures: The figures given to the section.
        later: The part of ``rest`` after this section.

    Returns:
        The reason, or "".
    """
    if set(_HEADINGS.findall(without_code(before))) - set(
        _HEADINGS.findall(without_code(after))
    ):
        return "a heading was lost"
    if kind := lost_assets(before, after, rest, owned_figures, later):
        return f"{'an' if kind[0] in 'aeiou' else 'a'} {kind} was lost"
    if prose_length(after) < (1 - max_prose_cut) * prose_length(before):
        return f"more than {max_prose_cut:.0%} of the prose was cut"
    return ""


# The end of a complete paragraph line: a sentence end, a closing mark, a
# citation or evidence marker, inline code or math, or a Korean sentence
# ending written without its period.
_COMPLETE_END = re.compile(r"(?:[.!?。:)\]」』\"'*`$]|\]\]|(?:다|요|죠|까))\s*$")
_LIST_LINE = re.compile(r"^(?:[-*+]|\d+[.)])\s")


def ends_mid_sentence(markdown: str) -> bool:
    """Whether the draft's last prose paragraph line stops mid-sentence (a
    cut-off answer); a list item, heading, table, image or block counts as
    complete."""
    for line in reversed(without_blocks(markdown).strip().split("\n")):
        bare = line.strip()
        if not bare:
            continue
        if bare.startswith(("#", "|", "!", "<", "- - -")) or _LIST_LINE.match(bare):
            return False
        return not _COMPLETE_END.search(bare)
    return False


def balance_math(markdown: str) -> tuple[str, bool]:
    """An odd ``$$`` count outside code makes every later equation of the
    post flip into prose: the last lone ``$$`` is removed. Returns the text
    and whether it changed."""
    lines = markdown.split("\n")
    marks: list[tuple[int, int]] = []
    for i, (line, role) in enumerate(zip(lines, block_lines(markdown), strict=True)):
        if role == "code":
            continue
        code = [(m.start(), m.end()) for m in _INLINE_CODE.finditer(line)]
        marks += [
            (i, m.start())
            for m in re.finditer(r"\$\$", line)
            if not any(s <= m.start() < e for s, e in code)
        ]
    if len(marks) % 2 == 0:
        return markdown, False
    i, at = marks[-1]
    lines[i] = lines[i][:at] + lines[i][at + 2 :]
    return "\n".join(lines), True


# A block of notes a writer appends about its own edits: a heading, bold
# label or rule-led line naming them, after which nothing is the section's.
_NOTE_LABEL = (
    r"(?:수정 ?(?:내역|사항|요약)|변경 ?(?:내역|사항)|고친 (?:부분|내용)|"
    r"revision notes?|changes made|summary of changes)"
)
# A heading or bold label naming the notes, or a line that is only the label.
_REVISION_NOTE = re.compile(
    rf"^(?:(?:#{{1,6}}[ \t]+|\*\*)[ \t]*{_NOTE_LABEL}\b.*|{_NOTE_LABEL}[ \t]*:?[ \t]*\**)$",
    re.IGNORECASE,
)


def strip_revision_notes(markdown: str, before: str = "") -> str:
    """The section without a block of notes a writer appended about its own
    edits: from a "## 수정 내역"/"**Changes made**"-style label line (outside
    code) to the end. A label line the section had ``before`` the edit is its
    content ("### 변경 사항" in a migration guide) and stays."""
    kept = {line.strip() for line in before.split("\n")}
    lines = markdown.split("\n")
    for i, (line, role) in enumerate(zip(lines, block_lines(markdown), strict=True)):
        if (
            role == "prose"
            and i > 0
            and line.strip() not in kept
            and _REVISION_NOTE.match(line.strip())
        ):
            lines = lines[:i]
            break
    out = "\n".join(lines).rstrip()
    # A rule left in front of the cut notes.
    return re.sub(r"(?:\n\s*(?:-{3,}|\*{3,}|_{3,})\s*)+$", "", out) + "\n"


def quote_for_prompt(quote: str) -> str:
    """A paper's words as inert data in a prompt: tag-like markup defanged,
    so an injection example quoted from a security paper reads as text. The
    mark is the one publishing removes (:func:`strip_defanging`), so code
    the writer copies from it is published as written."""
    return re.sub(
        r"<(/?)([A-Za-z_][\w-]*)",
        lambda m: f"<{ZERO_WIDTH_SPACE}{m.group(1)}{m.group(2)}",
        " ".join(quote.split()),
    )


_BARE_ITEM = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]*$")


def apply_edits(
    markdown: str,
    edits: Sequence[tuple[str, str]],
    *,
    rest: str = "",
    owned_figures: Collection[str] | None = None,
    later: str = "",
    max_edits: int = Limits().max_edits,
) -> tuple[str, int, list[str]]:
    """``markdown`` with the (find, replace) edits whose ``find`` occurs
    exactly once applied in order (an empty ``replace`` deletes, at most
    ``max_edits`` of them: ``Limits.max_edits``), with the
    guards of the review editor on the final text (edits one by one when it
    fails): no edit may unbalance a code fence, change a ``## `` heading,
    remove the only copy of a figure, equation, table or code block (a copy
    in ``rest``, the other sections, counts; a code block's only before
    ``later``, the sections after this one), or add a writing-rule slip.
    Returns the text, the number applied and why the others were skipped.
    A list item an edit emptied goes with its marker; nothing else is
    touched."""
    from scholar_lens.domain.review_plan import Edit
    from scholar_lens.domain.review_plan import apply_edits as guarded

    text, applied, skipped = guarded(
        markdown,
        # A model copies passages from its prompt, where tags are defanged:
        # the marks never reach the draft, and a find copied with them still
        # finds the draft's own text.
        [
            Edit(
                find=f if f in markdown else strip_defanging(f),
                replace=strip_defanging(r),
            )
            for f, r in edits
        ],
        max_edits=max_edits,
        rest=rest,
        owned_figures=owned_figures,
        later=later,
    )
    lines = text.split("\n")
    roles = block_lines(text)
    kept = [
        line
        for line, role in zip(lines, roles, strict=True)
        if not (role == "prose" and _BARE_ITEM.match(line))
    ]
    return "\n".join(kept), applied, skipped


def copied_sentences(markdown: str, body: Sequence[str]) -> list[Problem]:
    """Sentences of a summary (a TL;DR) that copy a ``body`` sentence
    instead of summing up, each a ``body_copy`` problem quoting it."""
    return [
        Problem(
            "body_copy",
            f"copies a sentence of the body; sum it up in new words: {s[:160]}",
            s[:160],
        )
        for s in prose_sentences(markdown)
        if near_duplicate(s, body)
    ]
