"""Defang prompt-fence tags in untrusted text (papers, web pages, code).

Prompts wrap untrusted input in XML-like fences (``<source_text>``,
``<sources>``, ...). Text containing a literal ``</source_text>`` could close the
fence early and inject instructions, so the "treat as untrusted DATA" framing is
only sound if the delimiters cannot appear in the data. Each fence tag is
neutralised by inserting a zero-width space after ``<``: it reads the same but no
longer matches the delimiter. Only fence tag names are touched, so other markup
and code in the source stay intact.

The tag set is derived from the prompt templates themselves (every tag that
wraps a ``{placeholder}``), so a new fence is protected without anyone
remembering to list it. ``strip_defanging`` removes the marks again from
generated Markdown, in case a model copied a defanged tag into its output.
"""

from __future__ import annotations

import functools
import re

from .prompts import BasePrompt

# A fence is a tag whose body holds a ``{placeholder}`` the input is filled
# into; output-format tags (``<plan>``, ``<edit>``, ...) carry no input.
_FENCED_PLACEHOLDER = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>(.*?)</\1>", re.DOTALL)
_PLACEHOLDER = re.compile(r"(?<!\{)\{[A-Za-z_]\w*\}(?!\})")
_ZERO_WIDTH_SPACE = "\u200b"


def _prompt_classes() -> list[type]:
    found: list[type] = []
    pending = list(BasePrompt.__subclasses__())
    while pending:
        cls = pending.pop()
        found.append(cls)
        pending.extend(cls.__subclasses__())
    return found


@functools.cache
def prompt_fence_tags() -> frozenset[str]:
    """Names of the tags that wrap an input placeholder in any prompt template."""
    tags: set[str] = set()
    for cls in _prompt_classes():
        for attr in ("system_prompt_template", "human_prompt_template"):
            template = str(getattr(cls, attr, "") or "")
            tags.update(
                match.group(1).lower()
                for match in _FENCED_PLACEHOLDER.finditer(template)
                if _PLACEHOLDER.search(match.group(2))
            )
    return frozenset(tags)


@functools.cache
def _fence_pattern() -> re.Pattern[str]:
    names = sorted(prompt_fence_tags(), key=len, reverse=True)
    # Spacing variants ("</ paper>", "< /paper >") and attributes still read
    # as the fence to a model, so they are defanged too.
    return re.compile(
        r"<\s*/?\s*(?:" + "|".join(map(re.escape, names)) + r")(?=[\s/>])[^<>]*>",
        re.IGNORECASE,
    )


def neutralize_prompt_tags(text: str) -> str:
    """Defang prompt-fence tags in untrusted text so it can't break the fence."""
    return _fence_pattern().sub(lambda m: f"<{_ZERO_WIDTH_SPACE}{m.group(0)[1:]}", text)


def strip_defanging(text: str) -> str:
    """Undo :func:`neutralize_prompt_tags` marks a model copied into its output."""
    return text.replace(f"<{_ZERO_WIDTH_SPACE}", "<")
