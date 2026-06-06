"""Token-usage accounting, cost estimation, and CloudWatch metric emission.

The pipeline makes many Bedrock calls (especially the per-paragraph synthesis
loop with Opus 4.8 + thinking). Without instrumentation this is the most
expensive — and previously invisible — part of the system. This module adds:

* :class:`TokenUsageTracker` — a LangChain callback that accumulates input/output
  token counts across every model call in a run.
* :class:`TokenBudgetExceeded` + ``check_budget`` — a hard guardrail so a
  pathological paper or a retry storm cannot run up unbounded spend.
* :class:`MetricsEmitter` — publishes ``InputTokens`` / ``OutputTokens`` /
  ``EstimatedCostUSD`` / ``DurationSeconds`` / ``Success`` to CloudWatch (and an
  EMF-style line to stdout) so cost and latency are queryable and alarmable.

All of it degrades to a no-op off-AWS or when boto/CloudWatch is unavailable;
nothing here may fail the job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import boto3
from langchain_core.callbacks import BaseCallbackHandler

from .logger import logger

METRICS_NAMESPACE = "ScholarLens"

# Rough USD per 1K tokens by model family (input, output). Used only for an
# *estimate* metric/log line — billing is authoritative. Tune as pricing changes.
_DEFAULT_RATE = (0.003, 0.015)
_COST_PER_1K: dict[str, tuple[float, float]] = {
    "opus": (0.015, 0.075),
    "sonnet": (0.003, 0.015),
    "haiku": (0.0008, 0.004),
}


class TokenBudgetExceeded(RuntimeError):
    """Raised when a run exceeds its configured total-token budget."""


@dataclass
class TokenUsageTracker(BaseCallbackHandler):
    """Accumulates Bedrock token usage across all LLM calls in a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0
    per_model: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: ANN401
        in_tok, out_tok, model = _extract_usage(response)
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.call_count += 1
        bucket = self.per_model.setdefault(
            model, {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        )
        bucket["input_tokens"] += in_tok
        bucket["output_tokens"] += out_tok
        bucket["calls"] += 1

    def estimated_cost_usd(self) -> float:
        total = 0.0
        for model, usage in self.per_model.items():
            rate_in, rate_out = _rate_for(model)
            total += usage["input_tokens"] / 1000 * rate_in
            total += usage["output_tokens"] / 1000 * rate_out
        return round(total, 4)

    def check_budget(self, max_total_tokens: int | None) -> None:
        """Raise :class:`TokenBudgetExceeded` if over the configured budget."""
        if max_total_tokens and self.total_tokens > max_total_tokens:
            raise TokenBudgetExceeded(
                f"Token budget exceeded: {self.total_tokens} > {max_total_tokens} "
                f"(across {self.call_count} calls)."
            )


def _extract_usage(response: Any) -> tuple[int, int, str]:  # noqa: ANN401
    """Pull (input, output, model) from a LangChain LLMResult, defensively."""
    in_tok = out_tok = 0
    model = "unknown"
    try:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("usage") or llm_output.get("token_usage") or {}
        model = str(llm_output.get("model_id") or llm_output.get("model_name") or model)
        # Fall back to per-generation usage_metadata (Converse via langchain-aws).
        if not usage:
            for gen_list in getattr(response, "generations", []) or []:
                for gen in gen_list:
                    meta = getattr(
                        getattr(gen, "message", None), "usage_metadata", None
                    )
                    if meta:
                        in_tok += int(meta.get("input_tokens", 0))
                        out_tok += int(meta.get("output_tokens", 0))
        else:
            in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            out_tok = int(
                usage.get("output_tokens") or usage.get("completion_tokens") or 0
            )
    except Exception as e:  # noqa: BLE001 - accounting must never break a call
        logger.debug("Could not extract token usage: %s", e)
    return in_tok, out_tok, model


def _rate_for(model_id: str) -> tuple[float, float]:
    lowered = model_id.lower()
    for family, rate in _COST_PER_1K.items():
        if family in lowered:
            return rate
    return _DEFAULT_RATE


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
            "EstimatedCostUSD": cost,
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
            self._client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {"MetricName": name, "Value": float(value), "Dimensions": dims}
                    for name, value in metrics.items()
                ],
            )
        except Exception as e:  # noqa: BLE001 - metric emission must not fail the job
            logger.warning("Failed to publish CloudWatch metrics: %s", e)
