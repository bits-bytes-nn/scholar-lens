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
remembering to list it: each template class registers its templates when it
is defined (:func:`register_prompt_templates`). ``strip_defanging`` removes the
marks again from generated Markdown, in case a model copied a defanged tag into
its output.
"""

from __future__ import annotations

import functools
import re

# A fence is a tag whose body holds a ``{placeholder}`` the input is filled
# into; output-format tags (``<plan>``, ``<edit>``, ...) carry no input.
_FENCED_PLACEHOLDER = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>(.*?)</\1>", re.DOTALL)
_PLACEHOLDER = re.compile(r"(?<!\{)\{[A-Za-z_]\w*\}(?!\})")
_ZERO_WIDTH_SPACE = "\u200b"

_TAGS: set[str] = set()


def register_prompt_templates(*templates: str) -> None:
    """Protect the fences of ``templates`` (a prompt's system and human
    templates) from now on."""
    found = {
        match.group(1).lower()
        for template in templates
        for match in _FENCED_PLACEHOLDER.finditer(template or "")
        if _PLACEHOLDER.search(match.group(2))
    }
    if not found <= _TAGS:
        _TAGS.update(found)
        _fence_pattern.cache_clear()


def prompt_fence_tags() -> frozenset[str]:
    """Names of the tags that wrap an input placeholder in any registered
    prompt template."""
    return frozenset(_TAGS)


@functools.cache
def _fence_pattern() -> re.Pattern[str]:
    names = sorted(_TAGS, key=len, reverse=True)
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
