"""The bundled copy of the blog's post conventions
(``assets/post_conventions.yml``), for tests, offline runs and the fallback
when the blog cannot be read. The canonical file is the blog's; CI fails
when the copy drifts."""

from __future__ import annotations

from pathlib import Path

from scholar_lens.domain.conventions import PostConventions

LOCAL_CONVENTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "post_conventions.yml"
)


def load_local_conventions(path: Path = LOCAL_CONVENTIONS_PATH) -> PostConventions:
    """The conventions of the bundled copy (or of ``path``)."""
    return PostConventions.from_yaml(path.read_text(encoding="utf-8"))
