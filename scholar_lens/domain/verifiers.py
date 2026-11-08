"""Problems and the verifier set of a section (docs/engine.md, 4.1).

A verifier reads a section's Markdown and reports :class:`Problem` s. The
engine builds one :class:`VerifierSet` per section and runs the same set at
every point that changes the section: the draft, each fix, the whole-post
fixes. Two runs are compared as sets of normalised problems (the sentence a
problem quotes and every digit left out), so a batch of edits is judged by
the kinds of problem it adds, not by how many it leaves.
"""

from __future__ import annotations

import re
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
    """

    kind: str
    text: str
    where: str = ""
    blocking: bool = False

    @property
    def key(self) -> tuple[str, str]:
        """The problem without its position: its kind and its message with
        the quoted passage and every digit removed."""
        text = self.text.replace(self.where, "") if self.where else self.text
        return self.kind, " ".join(_DIGITS.sub("", text).split())


Check = Callable[[str], Sequence[Problem]]


def as_problems(kind: str, texts: Iterable[str]) -> list[Problem]:
    """Messages of a check that reports strings, as problems of ``kind``."""
    return [Problem(kind, t) for t in texts]


@dataclass(frozen=True)
class VerifierSet:
    """The named checks of one section, run together in order."""

    name: str
    checks: tuple[Check, ...]

    def run(self, markdown: str) -> list[Problem]:
        """Every check's problems of ``markdown``, in the checks' order."""
        return [p for check in self.checks for p in check(markdown)]

    def added(
        self, before: str | Sequence[Problem], after: str | Sequence[Problem]
    ) -> list[Problem]:
        """The problems of ``after`` whose normalised form ``before`` does
        not have (either side: a text to check or its problems).

        Args:
            before: The text before an edit, or its problems.
            after: The text after it, or its problems.

        Returns:
            The new problems, once per normalised form, in order.
        """
        old = before if not isinstance(before, str) else self.run(before)
        new = after if not isinstance(after, str) else self.run(after)
        seen = {p.key for p in old}
        out = []
        for p in new:
            if p.key not in seen:
                seen.add(p.key)
                out.append(p)
        return out


def no_checks(name: str = "none") -> VerifierSet:
    """A verifier set that finds nothing."""
    return VerifierSet(name, ())
