"""Tests for token accounting, cost estimation, and metric emission.

All paths degrade to no-ops off-AWS; these tests assert that accounting never
raises and that cost math is correct, without any AWS or network access.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from scholar_lens.src.metrics import (
    METRICS_NAMESPACE,
    MetricsEmitter,
    TokenBudgetExceeded,
    TokenBudgetGuard,
    TokenUsageTracker,
    _extract_usage,
    price_per_mtok,
)


def _llm_output_response(model_id: str, input_tokens: int, output_tokens: int) -> Any:
    return SimpleNamespace(
        llm_output={
            "model_id": model_id,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }
    )


def _usage_metadata_response(
    input_tokens: int, output_tokens: int, model_name: str | None = None
) -> Any:
    message = SimpleNamespace(
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        response_metadata=({"model_name": model_name} if model_name else {}),
    )
    gen = SimpleNamespace(message=message)
    return SimpleNamespace(llm_output={}, generations=[[gen]])


class TestOnLLMEnd:
    def test_llm_output_usage_accumulates(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_llm_output_response("anthropic.claude-opus-4-8", 100, 50))
        assert tracker.input_tokens == 100
        assert tracker.output_tokens == 50
        assert tracker.total_tokens == 150
        assert tracker.call_count == 1

    def test_usage_metadata_path_accumulates(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_llm_output_response("anthropic.claude-opus-4-8", 100, 50))
        tracker.on_llm_end(_usage_metadata_response(40, 10))
        assert tracker.input_tokens == 140
        assert tracker.output_tokens == 60
        assert tracker.total_tokens == 200
        assert tracker.call_count == 2

    def test_converse_model_attributed_from_response_metadata(self) -> None:
        # Regression (code-0): the Converse path returns no llm_output, so the
        # model id must be recovered from message.response_metadata["model_name"].
        # Otherwise everything buckets under "unknown" and the cost estimate falls
        # back to the Sonnet rate — ~5x under-reporting Opus spend.
        tracker = TokenUsageTracker()
        tracker.on_llm_end(
            _usage_metadata_response(1000, 1000, model_name="claude-opus-4-8")
        )
        assert "unknown" not in tracker.per_model
        assert "claude-opus-4-8" in tracker.per_model
        # Opus 4.8 list price $5 in + $25 out per MTok -> 0.03.
        assert tracker.estimated_cost_usd() == pytest.approx(0.03)


class TestEstimatedCost:
    def test_opus_known_calculation(self) -> None:
        tracker = TokenUsageTracker()
        tracker.on_llm_end(
            _llm_output_response("anthropic.claude-opus-4-8", 1000, 1000)
        )
        assert tracker.estimated_cost_usd() == pytest.approx(0.03)

    def test_opus_costs_more_than_haiku(self) -> None:
        opus = TokenUsageTracker()
        opus.on_llm_end(_llm_output_response("claude-opus-4-8", 1000, 1000))
        haiku = TokenUsageTracker()
        haiku.on_llm_end(_llm_output_response("claude-haiku-5-5", 1000, 1000))
        assert opus.estimated_cost_usd() > haiku.estimated_cost_usd()


class TestCheckBudget:
    def test_over_raises(self) -> None:
        tracker = TokenUsageTracker(input_tokens=900, output_tokens=200)
        with pytest.raises(TokenBudgetExceeded):
            tracker.check_budget(1000)

    def test_under_no_raise(self) -> None:
        tracker = TokenUsageTracker(input_tokens=100, output_tokens=100)
        tracker.check_budget(1000)

    def test_none_no_raise(self) -> None:
        tracker = TokenUsageTracker(input_tokens=10_000, output_tokens=10_000)
        tracker.check_budget(None)


class TestTokenBudgetGuard:
    """The mixin shared verbatim by all three generators (extract tracker from
    callbacks, then abort past the ceiling)."""

    def _guard(
        self, callbacks: list[Any] | None, max_total_tokens: int | None
    ) -> TokenBudgetGuard:
        guard = TokenBudgetGuard()
        guard._init_token_budget(callbacks, max_total_tokens)
        return guard

    def test_finds_tracker_among_callbacks(self) -> None:
        tracker = TokenUsageTracker()
        guard = self._guard([MagicMock(), tracker, MagicMock()], 1000)
        assert guard._token_tracker is tracker
        assert guard.max_total_tokens == 1000

    def test_no_tracker_present(self) -> None:
        guard = self._guard([MagicMock()], 1000)
        assert guard._token_tracker is None
        guard._enforce_token_budget()  # no tracker → no-op, must not raise

    def test_enforce_raises_when_over(self) -> None:
        tracker = TokenUsageTracker(input_tokens=900, output_tokens=200)
        guard = self._guard([tracker], 1000)
        with pytest.raises(TokenBudgetExceeded):
            guard._enforce_token_budget()

    def test_enforce_noop_when_under(self) -> None:
        tracker = TokenUsageTracker(input_tokens=100, output_tokens=100)
        guard = self._guard([tracker], 1000)
        guard._enforce_token_budget()

    def test_enforce_noop_when_no_ceiling(self) -> None:
        tracker = TokenUsageTracker(input_tokens=10_000, output_tokens=10_000)
        guard = self._guard([tracker], None)
        guard._enforce_token_budget()  # ceiling None → no-op

    def test_two_tier_budget(self) -> None:
        from scholar_lens.src.constants import HARD_BUDGET_MULTIPLE

        tracker = TokenUsageTracker(input_tokens=1000)
        guard = self._guard([tracker], 1000)
        # At the budget: soft limit reached, hard ceiling not.
        assert guard._over_soft_token_budget()
        guard._enforce_hard_token_budget()
        tracker.input_tokens = int(1000 * HARD_BUDGET_MULTIPLE) + 1
        with pytest.raises(TokenBudgetExceeded):
            guard._enforce_hard_token_budget()

    def test_two_tier_budget_without_ceiling(self) -> None:
        tracker = TokenUsageTracker(input_tokens=10_000)
        guard = self._guard([tracker], None)
        assert not guard._over_soft_token_budget()
        guard._enforce_hard_token_budget()  # ceiling None → no-op


class TestMetricsEmitter:
    def test_no_session_does_not_raise_and_no_client(self) -> None:
        emitter = MetricsEmitter(boto_session=None)
        assert emitter._client is None
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_llm_output_response("opus", 100, 50))
        emitter.emit_run(
            mode="review",
            success=True,
            duration_seconds=1.23,
            tracker=tracker,
        )  # no raise

    def test_disabled_skips_client(self) -> None:
        session = MagicMock()
        emitter = MetricsEmitter(boto_session=session, enabled=False)
        assert emitter._client is None
        session.client.assert_not_called()

    def test_session_publishes_to_namespace(self) -> None:
        session = MagicMock()
        client = session.client.return_value
        emitter = MetricsEmitter(boto_session=session)
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_llm_output_response("opus", 100, 50))
        emitter.emit_run(
            mode="summary",
            success=False,
            duration_seconds=2.0,
            tracker=tracker,
        )
        client.put_metric_data.assert_called_once()
        assert client.put_metric_data.call_args.kwargs["Namespace"] == METRICS_NAMESPACE

    def test_cost_emitted_both_dimensioned_and_dimensionless(self) -> None:
        # The alarm watches a dimensionless series (a CloudWatch alarm can't
        # SEARCH across dimensions), so every metric must be emitted BOTH with the
        # Mode dimension and without.
        session = MagicMock()
        client = session.client.return_value
        emitter = MetricsEmitter(boto_session=session)
        tracker = TokenUsageTracker()
        tracker.on_llm_end(_llm_output_response("opus", 100, 50))
        emitter.emit_run(
            mode="guide",
            success=True,
            duration_seconds=1.0,
            tracker=tracker,
        )
        data = client.put_metric_data.call_args.kwargs["MetricData"]
        cost_entries = [e for e in data if e["MetricName"] == "EstimatedCostUSD"]
        assert len(cost_entries) == 2
        assert any("Dimensions" not in e for e in cost_entries)  # aggregate
        assert any(
            e.get("Dimensions") == [{"Name": "Mode", "Value": "guide"}]
            for e in cost_entries
        )  # per-mode

    def test_client_failure_is_swallowed(self) -> None:
        session = MagicMock()
        client = session.client.return_value
        client.put_metric_data.side_effect = RuntimeError("boom")
        emitter = MetricsEmitter(boto_session=session)
        tracker = TokenUsageTracker()
        emitter.emit_run(
            mode="review",
            success=True,
            duration_seconds=1.0,
            tracker=tracker,
        )  # no raise


class TestRateAndExtractUsageEdges:
    @pytest.mark.parametrize(
        ("model_id", "price"),
        [
            ("anthropic.claude-fable-5-1", (10.0, 50.0)),
            ("global.anthropic.claude-opus-5-5", (4.0, 20.0)),
            ("us.anthropic.claude-sonnet-5-5", (2.0, 10.0)),
            ("claude-haiku-5-5", (0.10, 0.50)),
            ("apac.anthropic.claude-opus-4-8", (5.0, 25.0)),
            ("anthropic.claude-opus-4-1-20250805-v1:0", (15.0, 75.0)),
        ],
    )
    def test_price_from_registry_for_invoked_ids(
        self, model_id: str, price: tuple[float, float]
    ) -> None:
        assert price_per_mtok(model_id) == price

    def test_unknown_model_uses_conservative_fallback(self) -> None:
        assert price_per_mtok("some-random-model") == (5.0, 25.0)

    def test_malformed_response_returns_zeros(self) -> None:
        in_tok, out_tok, model = _extract_usage(object())
        assert (in_tok, out_tok) == (0, 0)
        assert model == "unknown"

    def test_none_response_returns_zeros(self) -> None:
        in_tok, out_tok, model = _extract_usage(None)
        assert (in_tok, out_tok, model) == (0, 0, "unknown")


class TestTieredPricing:
    def test_haiku_55_long_prompts_use_the_higher_tier(self) -> None:
        from scholar_lens.src.metrics import call_cost_usd

        short = call_cost_usd("us.anthropic.claude-haiku-5-5", 100_000, 1_000)
        long = call_cost_usd("us.anthropic.claude-haiku-5-5", 200_000, 1_000)
        assert short == pytest.approx(100_000 / 1e6 * 0.10 + 1_000 / 1e6 * 0.50)
        assert long == pytest.approx(200_000 / 1e6 * 0.50 + 1_000 / 1e6 * 2.50)

    def test_tracker_prices_each_call(self) -> None:
        from scholar_lens.src.metrics import TokenUsageTracker

        tracker = TokenUsageTracker()
        tracker.on_llm_end(
            _llm_output_response("anthropic.claude-haiku-5-5", 50_000, 0)
        )
        tracker.on_llm_end(
            _llm_output_response("anthropic.claude-haiku-5-5", 150_000, 0)
        )
        # Aggregating first (200K at one rate) would misprice both calls.
        assert tracker.estimated_cost_usd() == pytest.approx(0.005 + 0.075)
