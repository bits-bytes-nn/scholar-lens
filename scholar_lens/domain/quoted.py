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


def _squash(text: str) -> str:
    """``text``'s words: whitespace folded; emphasis, code and math marks
    and link targets dropped (on both sides of a comparison)."""
    return " ".join(_MARKS.sub("", _LINK.sub(r"\1", text)).split())


def _fragments(span: str) -> list[str]:
    parts = (_squash(p) for p in _ELLIPSIS.split(span.replace('\\"', '"')))
    return [p for p in parts if len(p) >= _MIN_FRAGMENT]


def quoted_passages(finding: str, checked: str) -> list[list[str]]:
    """The passages ``finding`` quotes from ``checked``: each quoted span
    split at its ellipses, kept only when every fragment is in ``checked``
    (whitespace folded), so a quote of the paper is not one."""
    text = _squash(checked)
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
    text = _squash(edited)
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
    if not fragments or not all(f in _squash(checked) for f in fragments):
        return False
    return gone_after_edit([fragments], edited)


_WRONG = re.compile(r"\|\|\s*wrong\b", re.IGNORECASE)


def fix_kept(
    before: list[str],
    after: list[str],
    old: dict[str, str],
    new: dict[str, str],
) -> bool:
    """Whether a corrected text (``new``, its parts) replaces ``old`` after a
    claim check found ``before`` in it and a recheck ``after`` in ``new``:
    fewer problems, or as many, no more of them wrong (rather than
    unsupported), and every sentence the first check disputed in ``old``
    gone from ``new``. The recheck's count alone is noisy: a second reading
    finds another problem even in a fixed text. A reworded sentence still
    flagged also counts as gone; the notes then list the recheck's problems,
    which quote the new text."""
    if len(after) < len(before):
        return True

    def wrong(problems: list[str]) -> int:
        return sum(bool(_WRONG.search(p)) for p in problems)

    old_text, new_text = "\n".join(old.values()), "\n".join(new.values())
    return (
        len(after) == len(before)
        and wrong(after) <= wrong(before)
        and all(claim_fixed(problem, old_text, new_text) for problem in before)
    )
