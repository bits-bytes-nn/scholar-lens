"""Markdown table cells."""

from __future__ import annotations

import re

_MATH_SPAN = re.compile(r"\$\$.+?\$\$", re.DOTALL)


def escape_cell_pipes(cell: str) -> str:
    """Make a table cell pipe-safe without changing its meaning: prose pipes are
    escaped (``\\|``); inside ``$$…$$`` a norm ``\\|`` becomes ``\\Vert`` and a
    bare ``|`` becomes ``\\vert`` (escaping them would turn ``|x|`` into a norm
    or ``\\|W\\|`` into a line break)."""
    out, pos = [], 0
    for m in _MATH_SPAN.finditer(cell):
        out.append(cell[pos : m.start()].replace("|", "\\|"))
        math = m.group(0).replace("\\|", "\\Vert ")
        out.append(re.sub(r"(?<!\\)\|", r"\\vert ", math))
        pos = m.end()
    out.append(cell[pos:].replace("|", "\\|"))
    return "".join(out)
