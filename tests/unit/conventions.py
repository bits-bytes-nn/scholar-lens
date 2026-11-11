"""The bundled post conventions (``assets/post_conventions.yml``) as the unit
tests read them: the file parsed by the domain model, without the adapter
that loads it at run time."""

from __future__ import annotations

from pathlib import Path

from scholar_lens.domain.conventions import PostConventions

BUNDLED = Path(__file__).resolve().parents[2] / "assets" / "post_conventions.yml"


def bundled_conventions() -> PostConventions:
    """The conventions of the bundled copy."""
    return PostConventions.from_yaml(BUNDLED.read_text(encoding="utf-8"))
