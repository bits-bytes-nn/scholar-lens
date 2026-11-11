"""Cost reporting over finished runs: totals grouped by kind, stage, model or
day, with what failed and rejected runs spent (docs/redesign.md, 4.4)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from scholar_lens.domain.usage import RunUsage

GroupBy = Literal["kind", "stage", "model", "day"]
# The stage of a run whose manifest predates per-stage usage.
UNRECORDED = "unrecorded"


class RunRecord(BaseModel):
    """What the report needs from one run's result manifest."""

    job_id: str
    kind: str
    status: str
    finished_at: datetime
    cost_usd: float
    usage: RunUsage | None = None


class CostRow(BaseModel):
    key: str
    runs: int = 0
    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    # Spend of runs that did not produce an artifact (failed or rejected).
    wasted_usd: float = 0.0
    max_run_usd: float = 0.0


class CostReport(BaseModel):
    since: datetime
    until: datetime
    group_by: GroupBy
    runs: int = 0
    cost_usd: float = 0.0
    wasted_usd: float = 0.0
    rows: list[CostRow] = Field(default_factory=list)
    # Billed cost by AWS service for the project's tag (Cost Explorer).
    billed_usd: dict[str, float] | None = None


def _parts(record: RunRecord, by: GroupBy) -> list[tuple[str, int, int, float]]:
    """(key, calls, tokens, cost) parts of one run for the grouping."""
    if by == "kind":
        return [(record.kind, _calls(record), _tokens(record), record.cost_usd)]
    if by == "day":
        day = record.finished_at.date().isoformat()
        return [(day, _calls(record), _tokens(record), record.cost_usd)]
    if record.usage is None:
        return [(UNRECORDED, 0, 0, record.cost_usd)]
    groups = record.usage.by_stage if by == "stage" else record.usage.by_model
    return [
        (
            key,
            t.calls,
            t.input_tokens
            + t.output_tokens
            + t.cache_read_tokens
            + t.cache_write_tokens,
            t.cost_usd,
        )
        for key, t in groups.items()
    ]


def _calls(record: RunRecord) -> int:
    return record.usage.total.calls if record.usage else 0


def _tokens(record: RunRecord) -> int:
    if record.usage is None:
        return 0
    t = record.usage.total
    return t.input_tokens + t.output_tokens + t.cache_read_tokens + t.cache_write_tokens


def cost_report(
    records: Iterable[RunRecord], *, since: datetime, until: datetime, by: GroupBy
) -> CostReport:
    """Group the runs that finished in [since, until); rows costliest first."""
    report = CostReport(since=since, until=until, group_by=by)
    rows: dict[str, CostRow] = {}
    for record in records:
        if not since <= record.finished_at < until:
            continue
        wasted = record.status != "succeeded"
        report.runs += 1
        report.cost_usd += record.cost_usd
        report.wasted_usd += record.cost_usd if wasted else 0.0
        for key, calls, tokens, cost in _parts(record, by):
            row = rows.setdefault(key, CostRow(key=key))
            row.runs += 1
            row.calls += calls
            row.tokens += tokens
            row.cost_usd += cost
            row.wasted_usd += cost if wasted else 0.0
            row.max_run_usd = max(row.max_run_usd, cost)
    report.cost_usd = round(report.cost_usd, 4)
    report.wasted_usd = round(report.wasted_usd, 4)
    for row in rows.values():
        row.cost_usd = round(row.cost_usd, 4)
        row.wasted_usd = round(row.wasted_usd, 4)
        row.max_run_usd = round(row.max_run_usd, 4)
    report.rows = sorted(rows.values(), key=lambda r: (-r.cost_usd, r.key))
    return report
