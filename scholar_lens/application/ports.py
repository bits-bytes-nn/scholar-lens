"""Ports: what the use cases need from the outside world."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

from ..domain.costs import RunRecord


class RunLedger(Protocol):
    """Finished runs, from their result manifests."""

    def runs(self, since: datetime, until: datetime) -> Iterable[RunRecord]: ...


class BilledCosts(Protocol):
    """What the provider billed for the project, by service."""

    def billed(self, since: datetime, until: datetime) -> dict[str, float]: ...
