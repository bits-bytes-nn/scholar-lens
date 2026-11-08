"""Stage checkpoints of the post engine (docs/engine.md, 4.9).

A stage's output is saved under a hash of everything that decides it: the
checkpoint format version, the post kind (``PostSpec.name``), the stage, the
output of the stage before it (a chain, so a recomputed stage invalidates
every later one and nothing earlier), and the stage's own inputs (its
prompts, the system prompts it uses, the settings it reads, and the model,
effort and fallback of each role that produces it). A rerun with
the same inputs loads each saved stage instead of calling the model; a
changed input misses its stage's key and every later one. Keys are input
hashes, so resuming is never unsafe: a key either names this exact work or
is never looked up.

What a stage changed in the writer's shared state (its PR notes, blocking
problems, claim-check verdicts and unchecked claim units) is saved with its output and replayed on
load (:class:`WriterState`), so a resumed run's notes, blocking list and
verdict cache match an uninterrupted run's.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scholar_lens.application.ports import CheckpointStore, TextStore

logger = logging.getLogger(__name__)

# Bump when what a stage saves, or how the engine replays it, changes
# meaning (a prompt change needs no bump: the prompts are hashed).
CHECKPOINT_VERSION = 2
# The store's area the checkpoints live in.
CHECKPOINT_AREA = "checkpoints"


def _jsonable(value: object) -> Any:
    """``json.dumps``'s fallback: dataclasses and pydantic models as their
    fields, sets sorted. Anything else (a function, an object whose text
    holds its address) is not an input that hashes the same twice."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, set | frozenset):
        return sorted(value, key=repr)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Not a checkpoint input: {type(value).__name__}")


def input_hash(*parts: object) -> str:
    """A stable hash of ``parts`` (JSON with sorted keys).

    Args:
        *parts: Strings, numbers, lists, mappings, dataclasses, pydantic
            models and sets.

    Returns:
        The hex SHA-256 of their canonical JSON.

    Raises:
        TypeError: A part has no stable JSON form.
    """
    text = json.dumps(
        parts, sort_keys=True, ensure_ascii=False, default=_jsonable, allow_nan=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class WriterState(BaseModel):
    """What one stage added to the writer's shared state, in order: PR
    notes, blocking problems, claim-check verdicts (each the five parts of
    its :class:`~.evidence_writer.VerdictCache` key, then the verdict), the
    keys of claim units a check gave no verdict for, and the changes of the
    sections' unchecked blocks (section id, problem; "" lifts it)."""

    model_config = ConfigDict(extra="forbid")

    notes: list[str] = Field(default_factory=list)
    blocking: list[str] = Field(default_factory=list)
    verdicts: list[tuple[str, str, list[str], str, str, str]] = Field(
        default_factory=list
    )
    unanswered: list[tuple[str, str, list[str], str, str]] = Field(default_factory=list)
    unchecked: list[tuple[str, str]] = Field(default_factory=list)


class Checkpoint(BaseModel):
    """One saved stage: its output (JSON) and its writer-state changes."""

    model_config = ConfigDict(extra="forbid")

    version: int
    stage: str
    value: Any
    state: WriterState = Field(default_factory=WriterState)


class Checkpoints:
    """The engine's view of a :class:`~.ports.CheckpointStore`: keys under
    ``checkpoints/<kind>/<stage>/<hash>.json``."""

    def __init__(self, store: CheckpointStore, kind: str) -> None:
        self.store = store
        self.kind = kind

    def key(self, stage: str, digest: str) -> str:
        return f"{CHECKPOINT_AREA}/{self.kind}/{stage}/{digest[:32]}.json"

    def load(self, key: str) -> Checkpoint | None:
        """The checkpoint saved under ``key``; None when there is none, or
        it is unreadable or of another format version (the stage then runs
        again and saves over it)."""
        raw = self.store.get(key)
        if raw is None:
            return None
        try:
            saved = Checkpoint.model_validate_json(raw)
        except ValidationError as e:
            logger.warning("Unreadable checkpoint %s (%s); running again.", key, e)
            return None
        if saved.version != CHECKPOINT_VERSION:
            return None
        return saved

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        self.store.put(key, checkpoint.model_dump_json())


class FreshStore:
    """A store that is written but never read: a ``--fresh`` run computes
    every stage (and reads every source) again, and its results still
    serve the next run."""

    def __init__(self, store: TextStore) -> None:
        self.store = store

    def get(self, key: str) -> str | None:
        return None

    def put(self, key: str, text: str) -> None:
        self.store.put(key, text)
