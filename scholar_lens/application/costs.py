"""Use case: report what the runs of a period cost."""

from __future__ import annotations

from datetime import datetime

from ..domain.costs import CostReport, GroupBy, cost_report
from .ports import BilledCosts, RunLedger


def report_costs(
    ledger: RunLedger,
    *,
    since: datetime,
    until: datetime,
    by: GroupBy,
    billing: BilledCosts | None = None,
) -> CostReport:
    report = cost_report(ledger.runs(since, until), since=since, until=until, by=by)
    if billing is not None:
        report.billed_usd = billing.billed(since, until)
    return report
