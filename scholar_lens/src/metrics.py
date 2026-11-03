"""Token-usage accounting, cost estimation, and CloudWatch metric emission.

The pipelines make many Bedrock calls (especially the per-section writing
loops), the most expensive part of the system. This module provides:

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

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import boto3
from langchain_core.callbacks import BaseCallbackHandler

from ..application.usage import current_stage
from ..domain.usage import RunUsage, UsageEvent
from .constants import HARD_BUDGET_MULTIPLE
from .logger import logger
from .utils.models import (
    CACHE_READ_PRICE_MULTIPLE,
    CACHE_WRITE_PRICE_MULTIPLE,
    FALLBACK_PRICE_PER_MTOK,
    PRICE_VERSION,
    model_info_for_invoked_id,
    profile_source,
)

METRICS_NAMESPACE = "ScholarLens"
# The per-run cost metric; the stack's cost alarm watches this exact name.
COST_METRIC_NAME = "EstimatedCostUSD"


class TokenBudgetExceeded(RuntimeError):
    """Raised when a run exceeds its configured total-token budget."""


@dataclass
class TokenUsageTracker(BaseCallbackHandler):
    """Accumulates Bedrock token usage across all LLM calls in a run, and
    records each call (:class:`UsageEvent`, under the current
    :func:`~scholar_lens.application.usage.usage_stage`) for cost tracking:
    one ``llm_usage`` JSON log line per call and ``JobResult.usage``."""

    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0
    per_model: dict[str, dict[str, int]] = field(default_factory=dict)
    # Priced per call, so tiered (long-prompt) rates apply to the right calls.
    cost_usd: float = 0.0
    events: list[UsageEvent] = field(default_factory=list)
    _started: dict[Any, float] = field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def on_chat_model_start(
        self, serialized: Any, messages: Any, **kwargs: Any
    ) -> None:  # noqa: ANN401
        self._started[kwargs.get("run_id")] = time.monotonic()

    def on_llm_start(
        self, serialized: Any, prompts: Any, **kwargs: Any
    ) -> None:  # noqa: ANN401
        self._started[kwargs.get("run_id")] = time.monotonic()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: ANN401
        usage = _extract_usage(response)
        # The budget counts every token the model read, cached or not.
        read = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
        self.input_tokens += read
        self.output_tokens += usage.output_tokens
        self.call_count += 1
        bucket = self.per_model.setdefault(
            usage.model_id, {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        )
        bucket["input_tokens"] += read
        bucket["output_tokens"] += usage.output_tokens
        bucket["calls"] += 1
        cost = event_cost_usd(usage)
        self.cost_usd += cost
        self._record(
            usage.model_copy(update={"cost_usd": round(cost, 6)}),
            kwargs.get("run_id"),
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        # Tokens of a failed call are not reported; the call is still counted
        # so failure-heavy stages show up.
        self._record(
            UsageEvent(stage=current_stage(), model_id="unknown", outcome="failed"),
            kwargs.get("run_id"),
        )

    def _record(self, event: UsageEvent, run_id: Any) -> None:  # noqa: ANN401
        started = self._started.pop(run_id, None)
        if started is not None:
            event.latency_ms = int((time.monotonic() - started) * 1000)
        self.events.append(event)
        logger.info("llm_usage %s", event.model_dump_json())

    def run_usage(self) -> RunUsage:
        return RunUsage.of(self.events, PRICE_VERSION)

    def estimated_cost_usd(self) -> float:
        """Estimated spend from list prices of the models actually called.
        Bedrock bills at its own rates, so treat this as an estimate."""
        return round(self.cost_usd, 4)

    def check_budget(self, max_total_tokens: int | None) -> None:
        """Raise :class:`TokenBudgetExceeded` if over the configured budget."""
        if max_total_tokens and self.total_tokens > max_total_tokens:
            raise TokenBudgetExceeded(
                f"Token budget exceeded: {self.total_tokens} > {max_total_tokens} "
                f"(across {self.call_count} calls)."
            )


class TokenBudgetGuard:
    """Shared total-token budget wiring for the generation pipelines.

    All three generators (``PaperReviewer``, ``PaperSummarizer``,
    ``TechGuideGenerator``) locate the run's :class:`TokenUsageTracker` in their
    callback list and abort once cumulative spend passes ``max_total_tokens``.
    Subclasses call :meth:`_init_token_budget` from
    ``__init__`` and :meth:`_enforce_token_budget` at each expensive step.
    The guide and survey pipelines use the two-tier budget instead: past the
    soft limit (:meth:`_over_soft_token_budget`) they skip the optional
    checks, past the hard ceiling (:meth:`_enforce_hard_token_budget`) they
    abort.
    """

    _token_tracker: TokenUsageTracker | None
    max_total_tokens: int | None

    def _init_token_budget(
        self,
        callbacks: Sequence[BaseCallbackHandler] | None,
        max_total_tokens: int | None,
    ) -> None:
        self._token_tracker = next(
            (cb for cb in (callbacks or []) if isinstance(cb, TokenUsageTracker)),
            None,
        )
        self.max_total_tokens = max_total_tokens

    def _enforce_token_budget(self) -> None:
        """Abort the run if the configured total-token budget is exceeded."""
        if self._token_tracker is not None and self.max_total_tokens:
            self._token_tracker.check_budget(self.max_total_tokens)

    def _enforce_hard_token_budget(self) -> None:
        """Abort the run (``TokenBudgetExceeded``) once spend passes
        :data:`~scholar_lens.src.constants.HARD_BUDGET_MULTIPLE` times the
        budget: the soft limit only skips the optional checks, this is the
        true cap so a runaway can't keep issuing writer calls."""
        if self._token_tracker is not None and self.max_total_tokens:
            self._token_tracker.check_budget(
                int(self.max_total_tokens * HARD_BUDGET_MULTIPLE)
            )

    def _over_soft_token_budget(self) -> bool:
        """Whether spend has reached the budget (the soft limit)."""
        return bool(
            self._token_tracker is not None
            and self.max_total_tokens
            and self._token_tracker.total_tokens >= self.max_total_tokens
        )


def _extract_usage(response: Any) -> UsageEvent:  # noqa: ANN401
    """The usage of a LangChain LLMResult as an event under the current
    stage, defensively. langchain-aws reports cached tokens inside
    ``input_tokens`` (``input_token_details``); they are split out here."""
    in_tok = out_tok = cache_read = cache_write = 0
    model = "unknown"
    try:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("usage") or llm_output.get("token_usage") or {}
        model = str(llm_output.get("model_id") or llm_output.get("model_name") or model)
        # Fall back to per-generation usage_metadata (Converse via langchain-aws).
        # ChatBedrockConverse._generate returns a ChatResult with NO llm_output,
        # so both the tokens AND the model id must come from the generation: the
        # id lives in message.response_metadata["model_name"] (set by langchain-
        # aws). Without this the whole run buckets under "unknown", and the cost
        # estimate falls back to the Opus rate.
        if not usage:
            for gen_list in getattr(response, "generations", []) or []:
                for gen in gen_list:
                    message = getattr(gen, "message", None)
                    meta = getattr(message, "usage_metadata", None)
                    if meta:
                        details = meta.get("input_token_details") or {}
                        read = int(details.get("cache_read") or 0)
                        write = int(details.get("cache_creation") or 0)
                        cache_read += read
                        cache_write += write
                        in_tok += max(
                            int(meta.get("input_tokens", 0)) - read - write, 0
                        )
                        out_tok += int(meta.get("output_tokens", 0))
                    if model == "unknown":
                        resp_meta = getattr(message, "response_metadata", None) or {}
                        gen_model = resp_meta.get("model_name") or resp_meta.get(
                            "model_id"
                        )
                        if gen_model:
                            model = str(gen_model)
        else:
            in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            out_tok = int(
                usage.get("output_tokens") or usage.get("completion_tokens") or 0
            )
    except Exception as e:  # noqa: BLE001 - accounting must never break a call
        logger.debug("Could not extract token usage: %s", e)
    return UsageEvent(
        stage=current_stage(),
        # Reported by model, not by the kind's application profile ARN.
        model_id=profile_source(model),
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def event_cost_usd(event: UsageEvent) -> float:
    """List-price cost of one call, cache tokens at their multiples of the
    input price (the long-prompt tier is decided on all tokens read)."""
    read = event.input_tokens + event.cache_read_tokens + event.cache_write_tokens
    price_in, price_out = _prices(event.model_id, read)
    return (
        event.input_tokens * price_in
        + event.cache_read_tokens * price_in * CACHE_READ_PRICE_MULTIPLE
        + event.cache_write_tokens * price_in * CACHE_WRITE_PRICE_MULTIPLE
        + event.output_tokens * price_out
    ) / 1e6


def _prices(model_id: str, input_tokens: int) -> tuple[float, float]:
    info = model_info_for_invoked_id(model_id)
    if (
        info is not None
        and info.long_prompt_threshold is not None
        and info.long_prompt_price_per_mtok is not None
        and input_tokens > info.long_prompt_threshold
    ):
        return info.long_prompt_price_per_mtok
    return price_per_mtok(model_id)


def call_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """List-price cost of one uncached call, honouring tiered pricing."""
    return event_cost_usd(
        UsageEvent(
            model_id=model_id, input_tokens=input_tokens, output_tokens=output_tokens
        )
    )


def price_per_mtok(model_id: str) -> tuple[float, float]:
    """(input, output) USD per million tokens for an invoked model id."""
    info = model_info_for_invoked_id(model_id)
    if info is None:
        logger.debug("No price for model '%s'; using the fallback rate.", model_id)
        return FALLBACK_PRICE_PER_MTOK
    return info.input_price_per_mtok, info.output_price_per_mtok


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
                (
                    "StageTokens",
                    totals.input_tokens
                    + totals.output_tokens
                    + totals.cache_read_tokens
                    + totals.cache_write_tokens,
                ),
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
