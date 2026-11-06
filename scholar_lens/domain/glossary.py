"""The article glossary: terms and their renderings, and which a section
has yet to introduce."""

from __future__ import annotations


def glossary_view(glossary: dict[str, str], earlier: list[str]) -> str:
    """The glossary for one section: each rendering with its English, marked
    [introduced] when an ``earlier`` section already writes its first-use
    form "rendering(English)" (case-insensitive, a space before the
    parenthesis allowed), else [new]; a term written in English alone."""
    folded = [section.casefold() for section in earlier]
    lines = []
    for english, rendering in glossary.items():
        if rendering.casefold() == english.casefold():
            lines.append(f"- {rendering}")
        else:
            forms = [f"{rendering}{gap}({english})".casefold() for gap in ("", " ")]
            used = any(form in section for form in forms for section in folded)
            status = "introduced" if used else "new"
            lines.append(f"- {rendering} = {english} [{status}]")
    return "\n".join(lines) or "- none"
