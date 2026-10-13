"""Jekyll front matter: the one place the ``---`` delimiters are found (a leaf
module, so the validator, the publisher and the conventions loader share it
without an import cycle)."""

from __future__ import annotations

from typing import Any

import yaml


def split_front_matter_text(markdown: str) -> tuple[str, str]:
    """``(front_matter_with_delimiters, body)``; the front matter is empty when
    the post has none."""
    if markdown.startswith("---\n"):
        end = markdown.find("\n---\n", 4)
        if end != -1:
            return markdown[: end + 5], markdown[end + 5 :]
    return "", markdown


def split_front_matter(markdown: str) -> tuple[dict[str, Any] | None, str, int]:
    """Return ``(front_matter, body, body_start_line)``; ``None`` if absent or
    unparsable."""
    front, body = split_front_matter_text(markdown)
    if not front:
        return None, markdown, 1
    try:
        data = yaml.safe_load(front[4:-5])
    except yaml.YAMLError:
        return None, markdown, 1
    return (data if isinstance(data, dict) else None), body, front.count("\n") + 1
