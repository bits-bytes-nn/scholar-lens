"""Whether a checker's finding still applies after the text it quotes was
edited.

A section checker quotes the passage each defect refers to, and the TL;DR
claim check copies the sentence it disputes. Once an edit pass has run, a
finding whose quoted text is gone from the result was acted on; a finding
that quotes nothing from the checked text (it quotes the paper, or nothing)
cannot be settled this way and stays open. The rule reads "acted on", not
"fixed": an edit that rewords the passage for another reason (a gloss added
inside it) also settles the finding.
"""

from __future__ import annotations

import re

# A double-quoted span (backslash escapes allowed) or a curly-quoted one.
_QUOTED = re.compile(r'"((?:[^"\\]|\\.)+)"|“([^”]+)”')
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
# Shorter fragments ("a", "그") match by chance anywhere.
_MIN_FRAGMENT = 6


# Markup an edit may add or drop without touching the words: emphasis, code
# and math marks, and a link's target. Dropping a mark on both sides only
# makes more text match, so it never settles a finding wrongly.
_LINK = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")
_MARKS = re.compile(r"[*_`$]")


def squash(text: str) -> str:
    """``text``'s words: whitespace folded; emphasis, code and math marks
    and link targets dropped (on both sides of a comparison)."""
    return " ".join(_MARKS.sub("", _LINK.sub(r"\1", text)).split())


def _fragments(span: str) -> list[str]:
    parts = (squash(p) for p in _ELLIPSIS.split(span.replace('\\"', '"')))
    return [p for p in parts if len(p) >= _MIN_FRAGMENT]


def quoted_passages(finding: str, checked: str) -> list[list[str]]:
    """The passages ``finding`` quotes from ``checked``: each quoted span
    split at its ellipses, kept only when every fragment is in ``checked``
    (whitespace folded), so a quote of the paper is not one."""
    text = squash(checked)
    passages: list[list[str]] = []
    for match in _QUOTED.finditer(finding):
        fragments = _fragments(match.group(1) or match.group(2))
        if fragments and all(f in text for f in fragments):
            passages.append(fragments)
    return passages


def gone_after_edit(passages: list[list[str]], edited: str) -> bool:
    """Whether none of ``passages`` is still whole in ``edited``; False when
    there is nothing to look for."""
    if not passages:
        return False
    text = squash(edited)
    return not any(all(f in text for f in passage) for passage in passages)


def finding_fixed(finding: str, checked: str, edited: str) -> bool:
    """Whether ``finding`` quotes ``checked`` and its quoted text is gone from
    ``edited``."""
    return gone_after_edit(quoted_passages(finding, checked), edited)


def claim_fixed(problem: str, checked: str, edited: str) -> bool:
    """Whether a claim-check line (``<the sentence, copied exactly> ||
    <wrong|unsupported>: …``) disputes a sentence of ``checked`` that is no
    longer in ``edited``."""
    sentence = problem.split("||", 1)[0].strip().removeprefix("-").strip(' "“”')
    fragments = _fragments(sentence)
    if not fragments or not all(f in squash(checked) for f in fragments):
        return False
    return gone_after_edit([fragments], edited)


_WRONG = re.compile(r"\|\|[\s*_`]*wrong(?![A-Za-z])", re.IGNORECASE)
# A claim-check line's verdict: the word after ``||``.
_VERDICT = re.compile(r"\|\|[\s*_`]*([A-Za-z]+)")


def reported_claims(lines: list[str]) -> list[str]:
    """The claim-check ``lines`` that report a problem: a line whose verdict
    after ``||`` is neither "wrong" nor "unsupported" (``|| none``, ``||
    fine``: the checker withdrew it) is dropped; a line without a verdict is
    kept."""
    kept = []
    for line in lines:
        verdict = _VERDICT.search(line)
        if verdict is None or verdict.group(1).lower() in {"wrong", "unsupported"}:
            kept.append(line)
    return kept


def fix_kept(
    before: list[str],
    after: list[str],
    old: dict[str, str],
    new: dict[str, str],
) -> bool:
    """Whether a corrected text (``new``, its parts) replaces ``old`` after a
    claim check found ``before`` in it and a recheck ``after`` in ``new``:
    no more of them wrong (rather than unsupported), and fewer problems, or
    as many with every sentence the first check disputed in ``old`` gone
    from ``new``. The recheck's count alone is noisy: a second reading
    finds another problem even in a fixed text. A reworded sentence still
    flagged also counts as gone; the notes then list the recheck's problems,
    which quote the new text."""

    def wrong(problems: list[str]) -> int:
        return sum(bool(_WRONG.search(p)) for p in problems)

    if wrong(after) > wrong(before):
        return False  # the fix made a claim the paper contradicts
    # What would be noted: the recheck's problems and the first ones it
    # missed that are still there word for word.
    if len(kept_problems(before, after, new)) < len(before):
        return True
    old_text, new_text = "\n".join(old.values()), "\n".join(new.values())
    return len(after) == len(before) and all(
        claim_fixed(problem, old_text, new_text) for problem in before
    )


def disputed(problem: str) -> list[str]:
    """The fragments of the sentence a claim-check line copies (before
    ``||``), split at its ellipses."""
    sentence = problem.split("||", 1)[0].strip().removeprefix("-").strip(' "“”')
    return _fragments(sentence)


def kept_problems(
    before: list[str], after: list[str], new: dict[str, str]
) -> list[str]:
    """What to note once a fix is kept: the recheck's problems (``after``)
    and every first-check problem (``before``) whose disputed sentence is
    still in ``new`` word for word and the recheck did not report again (a
    second reading misses some it saw the first time)."""
    text = squash("\n".join(new.values()))
    reported = [" ".join(disputed(p)).rstrip(".!?。 ") for p in after]
    kept = list(after)
    for problem in before:
        fragments = disputed(problem)
        sentence = " ".join(fragments).rstrip(".!?。 ")
        again = any(
            sentence and r and (sentence in r or r in sentence) for r in reported
        )
        if fragments and all(f in text for f in fragments) and not again:
            kept.append(problem)
    return kept
