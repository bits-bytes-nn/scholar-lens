"""Model-call usage: one record per call and the run's breakdown.

The record is the unit of cost tracking (docs/redesign.md, section 4.4): every
model call yields one, the run's ``JobResult.usage`` sums them by stage and
model, and the structured log keeps each one for later reporting.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

UNSTAGED = "unstaged"

CallOutcome = Literal["ok", "failed"]


class UsageEvent(BaseModel):
    """Usage of one model call. Token counts are as the provider reported
    them; ``input_tokens`` excludes the cache read and write tokens."""

    stage: str = UNSTAGED
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int | None = None
    outcome: CallOutcome = "ok"

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


class UsageTotals(BaseModel):
    """Summed usage of a group of calls."""

    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def add(self, event: UsageEvent) -> None:
        self.calls += 1
        self.failed_calls += event.outcome == "failed"
        self.input_tokens += event.input_tokens
        self.output_tokens += event.output_tokens
        self.cache_read_tokens += event.cache_read_tokens
        self.cache_write_tokens += event.cache_write_tokens
        self.cost_usd = round(self.cost_usd + event.cost_usd, 6)


class RunUsage(BaseModel):
    """A run's usage in total, by stage and by model; the price table the
    costs were computed with is named so old estimates can be re-priced."""

    price_version: str
    total: UsageTotals = Field(default_factory=UsageTotals)
    by_stage: dict[str, UsageTotals] = Field(default_factory=dict)
    by_model: dict[str, UsageTotals] = Field(default_factory=dict)

    @classmethod
    def of(cls, events: Iterable[UsageEvent], price_version: str) -> RunUsage:
        usage = cls(price_version=price_version)
        for event in events:
            usage.total.add(event)
            usage.by_stage.setdefault(event.stage, UsageTotals()).add(event)
            usage.by_model.setdefault(event.model_id, UsageTotals()).add(event)
        return usage

    def top_stages(self, n: int = 3) -> list[tuple[str, float]]:
        """The ``n`` most expensive stages as (stage, cost), costliest first."""
        ranked = sorted(self.by_stage.items(), key=lambda kv: -kv[1].cost_usd)
        return [(stage, totals.cost_usd) for stage, totals in ranked[:n]]


class TokenBudgetExceeded(RuntimeError):
    """Raised when a run exceeds its configured total-token budget."""


def cost_ceiling_message(cost_usd: float, calls: int, max_cost_usd: float) -> str:
    """Why a run stopped at its cost ceiling, and how to go on."""
    return (
        f"Cost ceiling passed: ${cost_usd:.2f} > ${max_cost_usd:.2f} "
        f"(across {calls} calls). Raise the kind's ceiling "
        "(explanation.max_cost_usd, summary.max_cost_usd, "
        "tech_guide.max_cost_usd or survey.write_max_cost_usd; null for none). "
        "On the evidence engine, a rerun with the same inputs resumes from its "
        "saved stages."
    )
