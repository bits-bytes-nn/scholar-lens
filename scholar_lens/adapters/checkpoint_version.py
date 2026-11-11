"""Stage checkpoints keyed by the code that made them.

A stage checkpoint (docs/engine.md, 4.9) is keyed by its inputs. The checks,
repairs and edit rules that turn those inputs into the stage's output are
code: when they change, a saved stage would carry the old code's result
into a resumed run. :func:`check_code_version` hashes that code's sources,
and :class:`VersionedCheckpoints` puts the hash in every checkpoint key; the
reading cache (keyed by the reader's version) is not affected.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from functools import cache
from importlib import resources
from importlib.resources.abc import Traversable

from scholar_lens.application.ports import TextStore
from scholar_lens.domain.constants import CHECKPOINT_AREA

# The engine modules whose code shapes a stage's output, besides the domain.
_ENGINE_MODULES = (
    "post_engine.py",
    "evidence_writer.py",
    "paper_engine.py",
    "guide_engine.py",
    "survey_engine.py",
)


@cache
def check_code_version() -> str:
    """A hash of the domain package's sources and the engine modules', by
    their paths inside the package (the same code gives the same version on
    every machine)."""
    root = resources.files("scholar_lens")
    files = sorted(_python_files(root / "domain", "domain"))
    files += [
        (f"application/{name}", root / "application" / name) for name in _ENGINE_MODULES
    ]
    digest = hashlib.sha256()
    for name, path in files:
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _python_files(node: Traversable, name: str) -> Iterator[tuple[str, Traversable]]:
    for child in node.iterdir():
        path = f"{name}/{child.name}"
        if child.is_dir():
            if child.name != "__pycache__":
                yield from _python_files(child, path)
        elif child.name.endswith(".py"):
            yield path, child


class VersionedCheckpoints:
    """A :class:`TextStore` whose checkpoint keys (``checkpoints/...``) sit
    under the check code's version (``checkpoints/<version>/...``); other
    keys (the reading cache) pass through unchanged."""

    def __init__(self, store: TextStore, version: str | None = None) -> None:
        self.store = store
        self.version = version or check_code_version()

    def _key(self, key: str) -> str:
        prefix = f"{CHECKPOINT_AREA}/"
        if key.startswith(prefix):
            return f"{prefix}{self.version}/{key.removeprefix(prefix)}"
        return key

    def get(self, key: str) -> str | None:
        return self.store.get(self._key(key))

    def put(self, key: str, text: str) -> None:
        self.store.put(self._key(key), text)

    def __getattr__(self, name: str) -> object:
        # The wrapped store's other attributes (its local directory, area).
        return getattr(self.store, name)
