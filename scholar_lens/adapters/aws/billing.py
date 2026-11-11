"""Billed cost of the project's tagged resources from Cost Explorer."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


class CostExplorerBilling:
    """Unblended cost by service for resources tagged ``tag_key=tag_value``
    (a cost allocation tag, active in the billing console). Cost Explorer
    is daily; the period is widened to whole days."""

    def __init__(self, ce_client: Any, tag_key: str, tag_value: str) -> None:
        self._ce = ce_client
        self._tag = {"Key": tag_key, "Values": [tag_value], "MatchOptions": ["EQUALS"]}

    def billed(self, since: datetime, until: datetime) -> dict[str, float]:
        end = until.date() + (
            timedelta(days=1) if until.time() != datetime.min.time() else timedelta()
        )
        params: dict[str, Any] = {
            "TimePeriod": {"Start": since.date().isoformat(), "End": end.isoformat()},
            "Granularity": "MONTHLY",
            "Metrics": ["UnblendedCost"],
            "Filter": {"Tags": self._tag},
            "GroupBy": [{"Type": "DIMENSION", "Key": "SERVICE"}],
        }
        totals: dict[str, float] = {}
        while True:
            response = self._ce.get_cost_and_usage(**params)
            for period in response.get("ResultsByTime", []):
                for group in period.get("Groups", []):
                    service = group["Keys"][0]
                    amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
                    totals[service] = round(totals.get(service, 0.0) + amount, 4)
            token = response.get("NextPageToken")
            if not token:
                return dict(sorted(totals.items(), key=lambda kv: -kv[1]))
            params["NextPageToken"] = token
