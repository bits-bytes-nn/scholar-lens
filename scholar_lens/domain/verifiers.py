"""Problems and the verifier set of a section (docs/engine.md, 4.1).

A verifier reads a section's Markdown and reports :class:`Problem` s. The
engine builds one :class:`VerifierSet` per section and runs the same set at
every point that changes the section: the draft, each fix, the whole-post
fixes. Two runs are compared as multisets of normalised problems (the
sentence a problem quotes and every digit left out): a problem that moved
or whose passage was rephrased is the same problem, while one more problem
of a normalised form than before is a new one. A check that finds several
items wrong in one place (numbers, papers, images, flags) reports one problem
per item, so a fix that settles some of them adds none; :func:`messages`
shows the writer such problems together again.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

_DIGITS = re.compile(r"\d+")


@dataclass(frozen=True)
class Problem:
    """One problem a verifier found.

    Attributes:
        kind: What is wrong, as a stable name ("numbers_uncited").
        text: The message the writer is shown (it may quote ``where``).
        where: The passage the problem is in ("" when it is the section's).
        blocking: Whether the post may not be published with it as is.
        item: The one listed item the problem is about (a number, a paper,
            an image), shown in ``text`` as ``[item]`` ("" when none).
    """

    kind: str
    text: str
    where: str = ""
    blocking: bool = False
    item: str = ""

    @property
    def key(self) -> tuple[str, str]:
        """The problem without its position: its kind and its message with
        the quoted passage and every digit removed."""
        text = self.text.replace(self.where, "") if self.where else self.text
        return self.kind, " ".join(_DIGITS.sub("", text).split())


Check = Callable[[str], Sequence[Problem]]


def as_problems(kind: str, found: Iterable[str | Problem]) -> list[Problem]:
    """What a check reports, as problems: a message is a problem of ``kind``
    that quotes no passage, a problem is kept as it is."""
    return [p if isinstance(p, Problem) else Problem(kind, p) for p in found]


def item_problems(
    kind: str, items: Iterable[str], before: str, after: str, where: str = ""
) -> list[Problem]:
    """One problem per listed item (in order, each once), its message
    ``f"{before}[{item}]{after}"``: a fix that settles some items leaves the
    others' problems as they were, and one more item is one more problem."""
    return [
        Problem(kind, f"{before}[{i}]{after}", where, item=i)
        for i in dict.fromkeys(items)
    ]


def messages(problems: Iterable[Problem]) -> list[str]:
    """The messages the writer is shown: the problems of one kind and
    passage that differ only in their item are one message listing the
    items (in the first one's place); the rest as they are."""
    groups: dict[tuple[str, str, str], list[Problem]] = {}
    order: list[tuple[str, str, str] | Problem] = []
    for p in problems:
        if not p.item or f"[{p.item}]" not in p.text:
            order.append(p)
            continue
        shape = (p.kind, p.where, p.text.replace(f"[{p.item}]", "\0", 1))
        if shape not in groups:
            groups[shape] = []
            order.append(shape)
        groups[shape].append(p)
    out = []
    for entry in order:
        if isinstance(entry, Problem):
            out.append(entry.text)
        else:
            listed = ", ".join(p.item for p in groups[entry])
            out.append(entry[2].replace("\0", f"[{listed}]", 1))
    return out


@dataclass(frozen=True)
class VerifierSet:
    """The named checks of one section, run together in order."""

    name: str
    checks: tuple[Check, ...]
    # The figures given to the section (None: every figure shown is its
    # own). Only these are the section's to keep: a figure given to another
    # section is restored there, one given to none belongs nowhere.
    figures: frozenset[str] | None = None

    def run(self, markdown: str) -> list[Problem]:
        """Every check's problems of ``markdown``, in the checks' order."""
        return [p for check in self.checks for p in check(markdown)]

    def added(
        self, before: str | Sequence[Problem], after: str | Sequence[Problem]
    ) -> list[Problem]:
        """The problems of ``after`` beyond those of ``before``, compared as
        multisets of normalised forms (either side: a text to check or its
        problems): ``after`` adds a problem for each time it has a
        normalised form more often than ``before``.

        Args:
            before: The text before an edit, or its problems.
            after: The text after it, or its problems.

        Returns:
            The new problems, in order: of a form that grew, those whose
            passage ``before`` does not have come first.
        """
        old = before if not isinstance(before, str) else self.run(before)
        new = after if not isinstance(after, str) else self.run(after)
        # A problem at the same passage is the same one; the rest of a form
        # are matched by count (moved or reworded passages).
        exact = Counter((p.key, p.where) for p in old)
        same = []
        for p in new:
            same.append(exact[(p.key, p.where)] > 0)
            if same[-1]:
                exact[(p.key, p.where)] -= 1
        left = Counter(p.key for p in old)
        for p, kept in zip(new, same, strict=True):
            if kept:
                left[p.key] -= 1
        out = []
        for p, kept in zip(new, same, strict=True):
            if kept:
                continue
            if left[p.key] > 0:
                left[p.key] -= 1
            else:
                out.append(p)
        return out


def no_checks(name: str = "none") -> VerifierSet:
    """A verifier set that finds nothing."""
    return VerifierSet(name, ())
