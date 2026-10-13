"""Jekyll front matter: the one place the ``---`` delimiters are found (a leaf
module, so the validator, the publisher and the conventions loader share it
without an import cycle)."""

from __future__ import annotations


def split_front_matter_text(markdown: str) -> tuple[str, str]:
    """``(front_matter_with_delimiters, body)``; the front matter is empty when
    the post has none."""
    if markdown.startswith("---\n"):
        end = markdown.find("\n---\n", 4)
        if end != -1:
            return markdown[: end + 5], markdown[end + 5 :]
    return "", markdown
