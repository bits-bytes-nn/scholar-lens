"""Finished runs from the Batch result manifests on S3."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from ...domain.costs import RunRecord
from ...domain.usage import RunUsage

logger = logging.getLogger(__name__)


class S3RunLedger:
    """Reads ``<prefix>/results/<job_id>.json``; a manifest's finish time is
    its object's last-modified time (manifests are written once, at the end
    of the run)."""

    def __init__(self, s3_client: Any, bucket: str, results_prefix: str) -> None:
        self._s3 = s3_client
        self._bucket = bucket
        self._prefix = results_prefix.rstrip("/") + "/"

    def runs(self, since: datetime, until: datetime) -> Iterator[RunRecord]:
        pages = self._s3.get_paginator("list_objects_v2").paginate(
            Bucket=self._bucket, Prefix=self._prefix
        )
        for page in pages:
            for obj in page.get("Contents", []):
                finished = obj["LastModified"]
                if obj["Key"].endswith(".json") and since <= finished < until:
                    record = self._record(obj["Key"], finished)
                    if record is not None:
                        yield record

    def _record(self, key: str, finished: datetime) -> RunRecord | None:
        body = self._s3.get_object(Bucket=self._bucket, Key=key)["Body"].read()
        try:
            return _to_record(body, finished)
        except (ValidationError, ValueError, KeyError) as e:
            logger.warning("Skipping unreadable result manifest %s: %s", key, e)
            return None


def _to_record(body: bytes, finished: datetime) -> RunRecord:
    import json

    data = json.loads(body)
    usage = data.get("usage")
    return RunRecord(
        job_id=data["job_id"],
        kind=data.get("kind") or "unknown",
        status=data.get("status") or "failed",
        finished_at=finished,
        cost_usd=float(data.get("estimated_cost_usd") or 0.0),
        usage=RunUsage.model_validate(usage) if usage else None,
    )
