"""Ports: what the use cases need from the outside world."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

from scholar_lens.domain.costs import RunRecord
from scholar_lens.domain.evidence import SourceDocument
from scholar_lens.domain.survey.plan import PlanPaper


class RunLedger(Protocol):
    """Finished runs, from their result manifests."""

    def runs(self, since: datetime, until: datetime) -> Iterable[RunRecord]: ...


class BilledCosts(Protocol):
    """What the provider billed for the project, by service."""

    def billed(self, since: datetime, until: datetime) -> dict[str, float]: ...


# ---- the evidence engine's ports -----------------------------------------------

ModelRole = Literal["read", "outline", "write", "evaluate"]
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ModelText(str):
    """A text answer that knows whether the model stopped at its output
    limit (``cut``): a cut answer is incomplete however it ends."""

    cut: bool

    def __new__(cls, text: str, *, cut: bool = False) -> ModelText:
        obj = super().__new__(cls, text)
        obj.cut = cut
        return obj


def was_cut(text: str) -> bool:
    return bool(getattr(text, "cut", False))


@dataclass(frozen=True)
class AgentLimits:
    turns: int
    total_tokens: int | None = None


@dataclass(frozen=True)
class AgentOutcome:
    text: str
    stop_reason: str


# What repeated calls share, before the call's own prompt: one part, or
# parts ordered from the most stable (shared by the most calls) to the least.
ModelContext = str | tuple[str, ...]
# Cache points the adapter places in a call's messages; parts past it join
# the last. One: the calls that repeat a context (a section's check rounds)
# repeat all of it, and a section's earlier evidence grows from one section
# to the next, so a second point would not be read back.
MAX_CONTEXT_PARTS = 1


class LanguageModel(Protocol):
    """Model calls by role (the adapter maps a role to a model and effort).
    ``context``: see :data:`ModelContext`; empty parts are left out, and a
    call without context writes no message to the cache."""

    async def structured(
        self,
        *,
        role: ModelRole,
        system: str,
        prompt: str,
        schema: type[SchemaT],
        context: ModelContext = "",
    ) -> SchemaT: ...

    async def text(
        self, *, role: ModelRole, system: str, prompt: str, context: ModelContext = ""
    ) -> str: ...

    async def run_agent(
        self,
        *,
        role: ModelRole,
        system: str,
        prompt: str,
        tools: list[Callable[..., str]],
        limits: AgentLimits,
    ) -> AgentOutcome: ...


class DocumentSource(Protocol):
    """A survey paper's text, as complete as can be had (full text, else PDF
    text, else the abstract); None when there is none."""

    async def document(self, paper: PlanPaper) -> SourceDocument | None: ...


class TextStore(Protocol):
    """A cache of text blobs by key (local files, S3)."""

    def get(self, key: str) -> str | None: ...

    def put(self, key: str, text: str) -> None: ...


class CheckpointStore(Protocol):
    """Saved stage outputs of the post engine, by a hash of each stage's
    inputs (docs/engine.md, 4.9). The adapter is a :class:`TextStore` (the
    reading cache's local directory and S3 prefix): the engine keeps its
    checkpoints under its ``checkpoints/`` area. ``get`` is None for a key
    never saved; a ``put`` that could not be kept anywhere raises."""

    def get(self, key: str) -> str | None: ...

    def put(self, key: str, text: str) -> None: ...
