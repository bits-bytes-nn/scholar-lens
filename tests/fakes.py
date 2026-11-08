"""Test doubles shared across the suite that need no adapter (the unit
tests import only these; adapter-backed doubles are in
``tests/integration/fakes.py``)."""

from __future__ import annotations


def with_context(context: str | tuple[str, ...], prompt: str) -> str:
    """A scripted model's view of a call: its context parts (one string, or
    the parts a cached call passes) before its prompt."""
    parts = (context,) if isinstance(context, str) else context
    return "\n\n".join([*parts, prompt])
