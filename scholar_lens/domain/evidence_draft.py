"""Deterministic checks and edits of an evidence-cited section draft
(docs/redesign.md, 3.3), shared by every artifact the evidence engine writes.

A draft is Markdown whose sentences cite their evidence as ``[[E12]]``; the
checks quote the sentence they object to, so their messages can go back to
the writer as concrete problems.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from .evidence import Evidence, EvidenceLedger
from .markdown.lint import drop_empty_headings
from .markdown.math import fence_roles
from .quality_gate import rule_issues
from .validation import unsourced_numbers

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


def _example(sentence: str) -> bool:
    return bool(_EXAMPLE_MARK.search(sentence))


def free_numbers(extra: Iterable[object] = ()) -> str:
    """Numbers a sentence may state without evidence: small counts (0-10)
    and ``extra`` (a survey's publication years)."""
    return " ".join([*(str(n) for n in range(11)), *(str(x) for x in extra)])


def stated_numbers(sentence: str, free: str) -> list[str]:
    """The numbers a sentence states beyond ``free`` (years, small counts)."""
    return unsourced_numbers(_plain(sentence), free)


# Visuals a lineage or cross-cutting section needs by depth (images, display
# equations and tables); the code-built tables follow other kinds.
MIN_VISUALS = {"deep": 2, "standard": 1, "brief": 0}


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
    body = without_code(markdown)
    tables = len(re.findall(r"^\s*\|?\s*:?-{3,}", body, re.M))
    roles = block_lines(body)
    equations = sum(
        1
        for i, role in enumerate(roles)
        if role == "math" and (i == 0 or roles[i - 1] != "math")
    )
    return len(_IMAGE.findall(body)) + equations + tables


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
    exhausted: Iterable[str] = (),
    papers: Iterable[str] = (),
    target_chars: int = 0,
    passages: Mapping[tuple[str, str], str] | None = None,
) -> list[str]:
    """Deterministic problems of a section draft (each quotes its sentence):
    a missing heading, unknown or foreign evidence ids, a number without
    evidence or not in the cited evidence, a number an earlier section
    already stated, a sentence about the post's own inputs, an unassigned or
    missing figure. What a sentence means (talk about the post's sources, a
    claim that a source leaves something out, style) is the claim check's
    and the review's, in any language."""
    problems: list[str] = []
    if drop_empty_headings(markdown) != markdown:
        problems.append("a heading with nothing under it; delete it or fill it")
    if not markdown.lstrip().startswith("## "):
        problems.append(f"the section must start with its '## ' heading ({title})")
    allowed = set(own) | set(earlier)
    for sentence in prose_sentences(markdown):
        quoted = _plain(sentence)[:160]
        ids = cited_ids(sentence)
        if unknown := [i for i in ids if ledger.get(i) is None]:
            problems.append(f"unknown evidence {unknown} in: {quoted}")
            continue
        if _MARKER_AS_WORD.search(sentence):
            problems.append(
                "an evidence marker stands for a word; state the fact and put "
                f"the marker at the end of the sentence: {sentence[:160]}"
            )
        if spent := [i for i in ids if i in set(exhausted)]:
            problems.append(
                f"evidence {spent} was already referred back to once; do not "
                f"restate it again: {quoted}"
            )
        if foreign := [i for i in ids if i not in allowed]:
            problems.append(
                f"evidence {foreign} belongs to another section; leave it out: {quoted}"
            )
        numbers = stated_numbers(sentence, free_numbers)
        example = _example(sentence)
        if numbers and not ids and not example:
            problems.append(
                f"numbers {numbers} without evidence (cite it; an illustration "
                f"of your own ends with {EXAMPLE_MARK}): {quoted}"
            )
        elif numbers and not example:
            support = _support(ids, ledger, passages)
            if missing := unsourced_numbers(
                _plain(sentence), f"{support} {free_numbers}"
            ):
                problems.append(
                    f"numbers {missing} are not in the cited evidence: {quoted}"
                )
        if not example and (
            repeated := [
                n for n in numbers if n in earlier_numbers and not _MODEL_SIZE.match(n)
            ]
        ):
            problems.append(
                f"{repeated} already stated in '{earlier_numbers[repeated[0]]}': "
                f"refer back without repeating: {quoted}"
            )
        if _PAPER_ID.search(_plain(sentence)):
            problems.append(
                f"names a paper by its id; use its name or its authors: {quoted}"
            )
    for row in table_rows(markdown):
        plain_row = _plain(row)
        if _BLANK_CELL.search(plain_row):
            problems.append(
                "a table cell is blank ('—', or only evidence markers, which code "
                "removes): give the value the evidence or the passages state, or "
                "drop the row or column it belongs to; markers go after a cell's "
                "content: "
                f"{plain_row[:160]}"
            )
    cited_papers = {
        e.paper_id
        for s in prose_sentences(markdown)
        for i in cited_ids(s)
        if (e := ledger.get(i))
    }
    if uncovered := [p for p in papers if p not in cited_papers]:
        problems.append(
            f"papers {uncovered} belong to this section but no sentence cites "
            "their evidence: cover each"
        )
    if (
        target_chars
        and (length := len(" ".join(prose_sentences(markdown)))) > 1.5 * target_chars
    ):
        problems.append(
            f"{length:,} characters of prose for a target of {target_chars:,}: "
            "cut background, repetition and minor detail"
        )
    if ends_mid_sentence(markdown):
        problems.append("the section stops mid-sentence: finish it")
    # The blog's Markdown and math rules (an unbalanced $$ breaks every
    # later equation of the post).
    problems += rule_issues(MARKER.sub("", markdown))
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


def _support(
    ids: Iterable[str],
    ledger: EvidenceLedger,
    passages: Mapping[tuple[str, str], str] | None,
) -> str:
    """What the cited evidence states: its quote, scope, claim and cell and,
    with ``passages``, the whole source section each item is quoted from (a
    table's other values, a number in the next sentence)."""
    parts = []
    for i in ids:
        if (e := ledger.get(i)) is not None:
            parts.append(f"{e.quote} {e.scope} {e.claim} {e.cell}")
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
        if not numbers or _example(sentence):
            continue
        support = _support(cited_ids(sentence), ledger, passages)
        if unsourced_numbers(_plain(sentence), f"{support} {free_numbers}"):
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
    lines = markdown.split("\n")
    roles = block_lines(markdown)
    for sentence in sentences:
        row = sentence.startswith("|")
        for i, (line, role) in enumerate(zip(lines, roles, strict=True)):
            if role != "prose":
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
                    lines[i] = None  # type: ignore[call-overload]
                    break
                continue
            rest = line.replace(sentence, "", 1)
            if _LEFTOVER.match(rest):
                lines[i] = None  # type: ignore[call-overload]
            else:
                lead = line[: len(line) - len(line.lstrip())]
                lines[i] = lead + re.sub(r"[ \t]{2,}", " ", rest.strip())
            break
        kept = [(ln, r) for ln, r in zip(lines, roles, strict=True) if ln is not None]
        lines = [ln for ln, _ in kept]
        roles = [r for _, r in kept]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


_HEADINGS = re.compile(r"^#{2,4} .+$", re.M)


def _code_blocks(markdown: str) -> int:
    return fence_roles(markdown.split("\n")).count("open")


def _tables(markdown: str) -> int:
    # A delimiter row has a pipe; a bare "---" is a rule, not a table.
    return len(re.findall(r"^\s*\|?\s*:?-{3,}.*\|", markdown, re.M))


def structure_lost(before: str, after: str) -> str:
    """Why a revision lost the draft's structure ("" when it did not): a
    heading, an image, a code block, a table or display math gone, or more
    than a quarter of the prose."""
    if _code_blocks(after) < _code_blocks(before):
        return "a code block was lost"
    if _tables(without_code(after)) < _tables(without_code(before)):
        return "a table was lost"
    if set(_HEADINGS.findall(without_code(before))) - set(
        _HEADINGS.findall(without_code(after))
    ):
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


# The end of a complete paragraph line: a sentence end, a closing mark, a
# citation or evidence marker, inline code or math, or a Korean sentence
# ending written without its period.
_COMPLETE_END = re.compile(r"(?:[.!?。:)\]」』\"'*`$]|\]\]|(?:다|요|죠|까))\s*$")
_LIST_ITEM = re.compile(r"^(?:[-*+]|\d+[.)])\s")


def ends_mid_sentence(markdown: str) -> bool:
    """Whether the draft's last prose paragraph line stops mid-sentence (a
    cut-off answer); a list item, heading, table, image or block counts as
    complete."""
    for line in reversed(without_blocks(markdown).strip().split("\n")):
        bare = line.strip()
        if not bare:
            continue
        if bare.startswith(("#", "|", "!", "<", "- - -")) or _LIST_ITEM.match(bare):
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
    so an injection example quoted from a security paper reads as text."""
    return re.sub(r"<(/?)([A-Za-z_][\w-]*)", r"‹\1\2", " ".join(quote.split()))


_BARE_ITEM = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]*$")
MAX_EDITS = 30


def apply_edits(
    markdown: str, edits: Sequence[tuple[str, str]]
) -> tuple[str, int, list[str]]:
    """``markdown`` with the (find, replace) edits whose ``find`` occurs
    exactly once applied in order (an empty ``replace`` deletes), with the
    guards of the review editor: no edit may unbalance a code fence, change
    a ``## `` heading, remove the only copy of a figure, equation or table,
    or add a writing-rule slip. Returns the text, the number applied and why
    the others were skipped. A list item an edit emptied goes with its
    marker; nothing else is touched."""
    from .review_plan import Edit
    from .review_plan import apply_edits as guarded

    text, applied, skipped = guarded(
        markdown, [Edit(find=f, replace=r) for f, r in edits], max_edits=MAX_EDITS
    )
    lines = text.split("\n")
    roles = block_lines(text)
    kept = [
        line
        for line, role in zip(lines, roles, strict=True)
        if not (role == "prose" and _BARE_ITEM.match(line))
    ]
    return "\n".join(kept), applied, skipped
