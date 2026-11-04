"""Ports: what the use cases need from the outside world."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

from ..domain.costs import RunRecord
from ..domain.evidence import SourceDocument
from ..domain.survey.plan import PlanPaper


class RunLedger(Protocol):
    """Finished runs, from their result manifests."""

    def runs(self, since: datetime, until: datetime) -> Iterable[RunRecord]: ...


class BilledCosts(Protocol):
    """What the provider billed for the project, by service."""

    def billed(self, since: datetime, until: datetime) -> dict[str, float]: ...


# ---- the evidence engine's ports -----------------------------------------------

ModelRole = Literal["read", "outline", "write", "evaluate"]
SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True)
class AgentLimits:
    turns: int
    total_tokens: int | None = None


@dataclass(frozen=True)
class AgentOutcome:
    text: str
    stop_reason: str


class LanguageModel(Protocol):
    """Model calls by role (the adapter maps a role to a model and effort)."""

    async def structured(
        self, *, role: ModelRole, system: str, prompt: str, schema: type[SchemaT]
    ) -> SchemaT: ...

    async def text(self, *, role: ModelRole, system: str, prompt: str) -> str: ...

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
