"""Cost report: grouping, wasted spend, manifests on S3 and Cost Explorer."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from scholar_lens.adapters.aws.billing import CostExplorerBilling
from scholar_lens.adapters.aws.run_ledger import S3RunLedger
from scholar_lens.application.use_cases.costs import report_costs
from scholar_lens.domain.costs import UNRECORDED, RunRecord, cost_report
from scholar_lens.domain.usage import RunUsage, UsageEvent

DAY = datetime(2026, 11, 3, tzinfo=UTC)
NEXT = datetime(2026, 11, 4, tzinfo=UTC)


def _usage(*events: tuple[str, str, float]) -> RunUsage:
    return RunUsage.of(
        [
            UsageEvent(stage=s, model_id=m, input_tokens=10, cost_usd=c)
            for s, m, c in events
        ],
        "v1",
    )


RECORDS = [
    RunRecord(
        job_id="a",
        kind="guide",
        status="succeeded",
        finished_at=DAY,
        cost_usd=3.0,
        usage=_usage(("draft", "opus", 2.0), ("verify", "haiku", 1.0)),
    ),
    RunRecord(
        job_id="b",
        kind="guide",
        status="failed",
        finished_at=DAY,
        cost_usd=1.0,
        usage=_usage(("draft", "opus", 1.0)),
    ),
    RunRecord(
        job_id="c", kind="review", status="succeeded", finished_at=DAY, cost_usd=5.0
    ),
    RunRecord(
        job_id="d", kind="review", status="succeeded", finished_at=NEXT, cost_usd=9.0
    ),
]


def test_by_kind_counts_wasted_spend_and_skips_other_days() -> None:
    report = cost_report(RECORDS, since=DAY, until=NEXT, by="kind")
    assert (report.runs, report.cost_usd, report.wasted_usd) == (3, 9.0, 1.0)
    assert [
        (r.key, r.runs, r.cost_usd, r.wasted_usd, r.max_run_usd) for r in report.rows
    ] == [
        ("review", 1, 5.0, 0.0, 5.0),
        ("guide", 2, 4.0, 1.0, 3.0),
    ]


def test_by_stage_uses_recorded_usage_and_marks_old_manifests() -> None:
    report = cost_report(RECORDS, since=DAY, until=NEXT, by="stage")
    rows = {r.key: r.cost_usd for r in report.rows}
    assert rows == {"draft": 3.0, "verify": 1.0, UNRECORDED: 5.0}


@mock_aws
def test_s3_ledger_reads_manifests_in_the_period() -> None:
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="results-bucket")
    usage = _usage(("draft", "opus", 0.5))
    s3.put_object(
        Bucket="results-bucket",
        Key="sl/results/j1.json",
        Body=json.dumps(
            {
                "job_id": "j1",
                "kind": "summarize",
                "status": "succeeded",
                "input": "x",
                "estimated_cost_usd": 0.5,
                "usage": json.loads(usage.model_dump_json()),
            }
        ),
    )
    s3.put_object(
        Bucket="results-bucket", Key="sl/results/broken.json", Body=b"{not json"
    )
    s3.put_object(Bucket="results-bucket", Key="sl/posts/other.md", Body=b"x")
    # Not manifests: a list at the top level, and anything in a sub-folder.
    s3.put_object(Bucket="results-bucket", Key="sl/results/list.json", Body=b"[1]")
    s3.put_object(Bucket="results-bucket", Key="sl/results/evidence/x.json", Body=b"[]")
    # moto stamps objects with the current time; any later bound will do.
    later = datetime.now(UTC) + timedelta(days=366)
    ledger = S3RunLedger(s3, "results-bucket", "sl/results")
    (record,) = list(ledger.runs(datetime(2000, 1, 1, tzinfo=UTC), later))
    assert (record.job_id, record.kind, record.cost_usd) == ("j1", "summarize", 0.5)
    assert record.usage == usage
    report = report_costs(
        ledger,
        since=datetime(2000, 1, 1, tzinfo=UTC),
        until=later,
        by="model",
    )
    assert [(r.key, r.cost_usd) for r in report.rows] == [("opus", 0.5)]


class _StubS3:
    """Lists ``bodies`` (key -> manifest, or the error code ``get_object``
    raises) under ``sl/results/``, all finished on ``DAY``."""

    def __init__(self, bodies: dict[str, bytes | str]) -> None:
        self.bodies = bodies

    def get_paginator(self, _name: str) -> Any:
        contents = [{"Key": k, "LastModified": DAY} for k in self.bodies]
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": contents}]
        return paginator

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        body = self.bodies[Key]
        if isinstance(body, str):
            raise ClientError({"Error": {"Code": body}}, "GetObject")
        return {"Body": io.BytesIO(body)}


def _manifest(job_id: str, **fields: Any) -> bytes:
    return json.dumps({"job_id": job_id, "kind": "guide", **fields}).encode()


def test_a_refused_manifest_read_fails_the_report() -> None:
    s3 = _StubS3(
        {
            "sl/results/a.json": _manifest("a", estimated_cost_usd=1.0),
            "sl/results/b.json": "AccessDenied",
        }
    )
    ledger = S3RunLedger(s3, "results-bucket", "sl/results")
    with pytest.raises(ClientError, match="AccessDenied"):
        report_costs(ledger, since=DAY, until=NEXT, by="kind")


def test_a_manifest_gone_after_listing_is_skipped_and_no_status_is_waste() -> None:
    s3 = _StubS3(
        {
            "sl/results/a.json": _manifest(
                "a", status="succeeded", estimated_cost_usd=2.0
            ),
            "sl/results/gone.json": "NoSuchKey",
            # Written by a run that died before recording its status.
            "sl/results/c.json": _manifest("c", estimated_cost_usd=1.0),
        }
    )
    ledger = S3RunLedger(s3, "results-bucket", "sl/results")
    report = report_costs(ledger, since=DAY, until=NEXT, by="kind")
    assert (report.runs, report.cost_usd, report.wasted_usd) == (2, 3.0, 1.0)


def test_billing_sums_services_across_pages_for_the_tag() -> None:
    ce = MagicMock()

    def group(svc, amt):
        return {
            "Keys": [svc],
            "Metrics": {"UnblendedCost": {"Amount": amt}},
        }

    ce.get_cost_and_usage.side_effect = [
        {
            "ResultsByTime": [{"Groups": [group("Amazon EC2", "1.5")]}],
            "NextPageToken": "t",
        },
        {
            "ResultsByTime": [
                {"Groups": [group("Amazon EC2", "0.5"), group("AWS Batch", "0")]}
            ]
        },
    ]
    billed = CostExplorerBilling(ce, "ProjectName", "scholar-lens").billed(DAY, NEXT)
    assert billed == {"Amazon EC2": 2.0, "AWS Batch": 0.0}
    first = ce.get_cost_and_usage.call_args_list[0].kwargs
    assert first["TimePeriod"] == {"Start": "2026-11-03", "End": "2026-11-04"}
    assert first["Filter"]["Tags"]["Key"] == "ProjectName"
    assert ce.get_cost_and_usage.call_args_list[1].kwargs["NextPageToken"] == "t"


@pytest.mark.parametrize("argv", [["costs", "--since", "2026-13-01"], ["costs"]])
def test_costs_arguments_are_validated(argv: list[str]) -> None:
    from scholar_lens.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)
