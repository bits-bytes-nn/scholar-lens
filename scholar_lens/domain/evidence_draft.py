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
from .quality_gate import rule_issues
from .validation import unsourced_numbers

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
    r"입력 자료|제공된 (?:자료|문서|출처|근거)|원래 예시|원문(?:의)? 예시|"
    r"근거 자료(?:에|가|로)|"
    r"(?:다루는|주어진|받은) (?:자료|근거)|자료에(?:는)? (?:없|나오지)|근거에(?:는)? 없|"
    r"(?:이|본) 글(?:에서)?(?:는|은) .{0,20}(?:살펴봅니다|다룹니다|정리합니다)|이 절에서는|"
    r"\b(?:the|this) (?:abstract|card|summary) (?:says|states|reports)|based on the abstract",
    re.IGNORECASE,
)


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


def table_rows(markdown: str) -> list[str]:
    """The data rows of the draft's tables (header and rule rows left out),
    as written: what a claim check reads besides the prose."""
    rows = []
    for block in re.split(r"\n(?!\s*\|)", _FENCE.sub("\n", markdown)):
        lines = [ln.strip() for ln in block.split("\n") if _TABLE_LINE.match(ln)]
        if len(lines) >= 3 and re.match(r"^\|?\s*:?-{3,}", lines[1]):
            rows += lines[2:]
    return rows


def claim_units(markdown: str) -> list[str]:
    """What a claim check judges: the prose sentences, then the table rows."""
    return prose_sentences(markdown) + table_rows(markdown)


_EMPTY_HEADING = re.compile(r"^(#{2,4}) [^\n]+\n\s*\n?(?=#{2,4} |\Z)", re.M)


def drop_empty_headings(markdown: str) -> str:
    """The draft without headings that head nothing (a duplicate left in
    front of the real one, a heading at the very end)."""
    out = markdown
    while (new := _EMPTY_HEADING.sub("", out)) != out:
        out = new
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
_SENTENCE_HEADING = re.compile(r"(?:다|요)[.!]?$")


# Formulas a writer falls into, with how often one section may use them.
_FORMULAS = ((re.compile(r"보면 됩니다"), 1), (re.compile(r"다는 점"), 2))


_OWN_VIEW = re.compile(
    r"이 (?:글|가이드)의 (?:관점|판단|해석|견해)|(?:이 )?가이드의 (?:관점|판단|해석|견해)|"
    r"이 (?:글|가이드)는 .{0,12}(?:본다|봅니다|판단합니다)"
)


# A sentence about what a paper leaves out ("저자들은 … 적지 않습니다").
_ABSENCE_CLAIM = re.compile(
    r"(?:적지|밝히지|제시하지|명시하지|보고하지|구분하지|언급하지|다루지|측정하지|"
    r"정량화하지|수치화하지)\s*(?:는\s*)?않(?:습니다|았습니다|으며|고)"
)


# A sentence pointing back at what the post said before.
_BACK_REFERENCE = re.compile(
    r"앞서 (?:본|봤던|언급한|소개한|다룬|설명한|살펴본|정리한|나온)|앞에서 (?:본|다룬|설명한)"
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
    exhausted: Iterable[str] = (),
    papers: Iterable[str] = (),
    target_chars: int = 0,
    own_view_limit: int = 1,
) -> list[str]:
    """Deterministic problems of a section draft (each quotes its sentence):
    a missing heading, unknown or foreign evidence ids, a number without
    evidence or not in the cited evidence, a number an earlier section
    already stated, a sentence about the post's own inputs, an unassigned or
    missing figure."""
    problems: list[str] = []
    for heading in re.findall(r"^#{2,4} (.+?)\s*$", markdown, re.M):
        if _SENTENCE_HEADING.search(heading):
            problems.append(
                "heading is a sentence: write the claim as a noun phrase "
                f"('정책보다 먼저 무너지는 보상 모델'): {heading}"
            )
    plain = _FENCE.sub("", markdown)
    for formula, limit in _FORMULAS:
        if (count := len(formula.findall(plain))) > limit:
            problems.append(
                f"'{formula.pattern}' {count} times in one section; vary the "
                "sentence or drop the frame"
            )
    if (views := len(_OWN_VIEW.findall(_FENCE.sub("", markdown)))) > own_view_limit:
        problems.append(
            f"marks the post's own view {views} times (at most {own_view_limit} "
            "here); state the advice plainly or leave the marker out"
        )
    for line in _FENCE.sub("", markdown).split("\n"):
        if _REVISION_PHRASE.search(line):
            problems.append(
                "a note about the writing itself, not content for the reader; "
                f"delete it: {line.strip()[:160]}"
            )
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
        if _ABSENCE_CLAIM.search(_plain(sentence)):
            problems.append(
                "says what a paper does not state or report; that is rarely "
                "evidence (the reader saw only part of it): leave it out or "
                f"state what the paper does say: {quoted}"
            )
        if _BACK_REFERENCE.search(sentence) and not set(ids) & set(earlier):
            problems.append(
                "points back to something earlier without citing the earlier "
                f"evidence; cite it or drop the back-reference: {quoted}"
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
        if META_SENTENCE.search(_plain(sentence)):
            problems.append(
                "describes the post's inputs instead of the papers; state what "
                f"the paper does or leave it out: {quoted}"
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


_FRAGMENT = re.compile(r"^(?:[-*+]|\d+[.)])?\s*[^\s#|>!$`].{0,24}$")


_SENTENCE_CLOSE = re.compile(r"[.!?。:)\]]\s*$")


def drop_sentences(markdown: str, sentences: Iterable[str]) -> str:
    """``markdown`` without the given sentences (as :func:`prose_sentences`
    returns them), each removed once, and without the fragments a removal
    leaves (a short line that ends no sentence, such as "첫째")."""
    for sentence in sentences:
        if sentence.startswith("|"):
            # A table row goes with its line, or the table splits in two.
            markdown = re.sub(
                r"^[ \t]*" + re.escape(sentence) + r"[ \t]*\n?",
                "",
                markdown,
                count=1,
                flags=re.M,
            )
        else:
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


# The end of a complete paragraph line: a sentence end, a closing mark, a
# citation or an evidence marker.
_COMPLETE_END = re.compile(r"(?:[.!?。:)\]」』\"'*]|\]\])\s*$")


def ends_mid_sentence(markdown: str) -> bool:
    """Whether the draft's last prose line stops mid-sentence (a cut-off
    answer)."""
    text = _FENCE.sub("\n", markdown)
    text = _DISPLAY_MATH.sub("\n", text)
    for line in reversed(text.strip().split("\n")):
        bare = line.strip()
        if not bare or bare.startswith(("#", "|", "!", "$$", "<", "- - -")):
            continue
        return not _COMPLETE_END.search(bare)
    return False


def balance_math(markdown: str) -> tuple[str, bool]:
    """An odd ``$$`` count outside code makes every later equation of the
    post flip into prose: the last lone ``$$`` is removed. Returns the text
    and whether it changed."""
    code = [(m.start(), m.end()) for m in _FENCE.finditer(markdown)]
    marks = [
        m.start()
        for m in re.finditer(r"\$\$", markdown)
        if not any(s <= m.start() < e for s, e in code)
    ]
    if len(marks) % 2 == 0:
        return markdown, False
    last = marks[-1]
    return markdown[:last] + markdown[last + 2 :], True


# Where a writer starts describing its own edits instead of the section.
_REVISION_NOTE = re.compile(
    r"^(?:#+\s*|\*\*|-{3,}\s*$)?\s*(?:수정 ?(?:내역|사항|요약)|변경 ?(?:내역|사항)|"
    r"고친 (?:부분|내용)|revision notes?|changes made|summary of changes)",
    re.IGNORECASE | re.M,
)


_REVISION_PHRASE = re.compile(
    r"(?:제시하신|요청하신|지적하신|말씀하신)|\bE\d+ (?:반복 )?(?:제거|삭제|수정)|"
    r"\bnotes? on (?:the )?fix|\bI (?:could not|couldn't|have|did not|didn't|kept|removed|added)\b|"
    r"\bas requested\b",
    re.IGNORECASE,
)


def strip_revision_notes(markdown: str) -> str:
    """The section without a writer's notes about its own edits: everything
    from a "수정 내역"-style heading or line on, and any line that addresses
    the requester or names evidence ids it removed."""
    if m := _REVISION_NOTE.search(markdown):
        markdown = markdown[: m.start()]
    lines = [line for line in markdown.split("\n") if not _REVISION_PHRASE.search(line)]
    out = "\n".join(lines).rstrip()
    # A rule left in front of the cut notes.
    return re.sub(r"(?:\n\s*(?:-{3,}|\*{3,}|_{3,})\s*)+$", "", out) + "\n"


def quote_for_prompt(quote: str) -> str:
    """A paper's words as inert data in a prompt: tag-like markup defanged,
    so an injection example quoted from a security paper reads as text."""
    return re.sub(r"<(/?)([A-Za-z_][\w-]*)", r"‹\1\2", " ".join(quote.split()))
