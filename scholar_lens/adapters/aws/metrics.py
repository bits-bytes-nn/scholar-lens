"""Run metrics on CloudWatch (and an EMF-style line on stdout), so cost and
latency are queryable and alarmable. Best-effort: nothing here may fail the
job."""

from __future__ import annotations

import logging

import boto3

from scholar_lens.adapters.bedrock.usage import TokenUsageTracker
from scholar_lens.domain.usage import RunUsage

logger = logging.getLogger(__name__)


METRICS_NAMESPACE = "ScholarLens"


# The per-run cost metric; the stack's cost alarm watches this exact name.
COST_METRIC_NAME = "EstimatedCostUSD"


class MetricsEmitter:
    """Publishes run metrics to CloudWatch (best-effort) and stdout."""

    def __init__(
        self,
        boto_session: boto3.Session | None = None,
        *,
        namespace: str = METRICS_NAMESPACE,
        enabled: bool = True,
    ) -> None:
        self.namespace = namespace
        self.enabled = enabled
        self._client = None
        if enabled and boto_session is not None:
            try:
                self._client = boto_session.client("cloudwatch")
            except Exception as e:  # noqa: BLE001
                logger.debug("CloudWatch client unavailable: %s", e)

    def emit_run(
        self,
        *,
        mode: str,
        success: bool,
        duration_seconds: float,
        tracker: TokenUsageTracker,
    ) -> None:
        dims = [{"Name": "Mode", "Value": mode}]
        cost = tracker.estimated_cost_usd()
        metrics = {
            "InputTokens": tracker.input_tokens,
            "OutputTokens": tracker.output_tokens,
            "TotalTokens": tracker.total_tokens,
            "LLMCalls": tracker.call_count,
            COST_METRIC_NAME: cost,
            "DurationSeconds": round(duration_seconds, 2),
            "Success": 1 if success else 0,
            "Failure": 0 if success else 1,
        }
        # Always log a structured summary line (queryable even without metrics).
        logger.info(
            "run_metrics mode=%s success=%s tokens_in=%d tokens_out=%d calls=%d "
            "cost_usd=%.4f duration_s=%.1f",
            mode,
            success,
            tracker.input_tokens,
            tracker.output_tokens,
            tracker.call_count,
            cost,
            duration_seconds,
        )
        if not self._client:
            return
        try:
            # Emit each metric TWICE: once with the Mode dimension (per-pipeline
            # breakdown) and once with NO dimensions (a stack-wide aggregate). A
            # CloudWatch alarm can only watch a single time series — it can't
            # SEARCH/SUM across dimensions — so the dimensionless copy is what the
            # cost/error alarms actually alarm on.
            self._client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    entry
                    for name, value in metrics.items()
                    for entry in (
                        {
                            "MetricName": name,
                            "Value": float(value),
                            "Dimensions": dims,
                        },
                        {"MetricName": name, "Value": float(value)},
                    )
                ],
            )
        except Exception as e:  # noqa: BLE001 - metric emission must not fail the job
            logger.warning("Failed to publish CloudWatch metrics: %s", e)
        self._emit_stages(mode, tracker.run_usage())

    def _emit_stages(self, mode: str, usage: RunUsage) -> None:
        """Cost, tokens and calls per stage (dimensions Mode and Stage)."""
        if not self._client or not usage.by_stage:
            return
        data = [
            {
                "MetricName": name,
                "Value": float(value),
                "Dimensions": [
                    {"Name": "Mode", "Value": mode},
                    {"Name": "Stage", "Value": stage},
                ],
            }
            for stage, totals in usage.by_stage.items()
            for name, value in (
                ("StageCostUSD", totals.cost_usd),
                ("StageTokens", totals.total_tokens),
                ("StageCalls", totals.calls),
                ("StageFailedCalls", totals.failed_calls),
            )
        ]
        try:
            for i in range(0, len(data), 1000):  # the API's per-request limit
                self._client.put_metric_data(
                    Namespace=self.namespace, MetricData=data[i : i + 1000]
                )
        except Exception as e:  # noqa: BLE001 - metric emission must not fail the job
            logger.warning("Failed to publish stage metrics: %s", e)
