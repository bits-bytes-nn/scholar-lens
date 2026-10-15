"""Deterministic draft gates shared by every writing loop (review, summary,
guide and survey): the rule slips a writer can fix and the body errors the
publisher's validator would still report."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from .markdown_lint import publish_body, writing_issues
from .post_validator import body_errors


def rule_issues(text: str) -> list[str]:
    """Deterministic rule hits a writer can fix: writing slips plus the
    publish-blocking body checks (tables, math, code fences) that are left
    after the publisher's own fixes (math delimiters, blank lines, headings),
    so a draft is never revised for what the publisher repairs."""
    return writing_issues(text) + publish_blockers(text)


def publish_blockers(text: str) -> list[str]:
    """The body errors the validator would still report once the section is
    closed (an open fence) and the publisher has normalized math and linted
    the Markdown."""
    fixed = publish_body(text, warn=False)
    return body_errors(fixed)


def added_slips(
    before: str, after: str, rules: Callable[[str], list[str]] = rule_issues
) -> list[str]:
    """Rule hits (rule and offending text) an edit would add, even when it
    also fixes another; ``rules`` finds them (:func:`rule_issues` by
    default)."""

    def keys(text: str) -> Counter[str]:
        return Counter(issue.split(":", 1)[0] for issue in rules(text))

    return list((keys(after) - keys(before)).elements())
