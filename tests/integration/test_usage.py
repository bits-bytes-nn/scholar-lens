"""Per-call usage records: stage tagging, cache tokens, failures, totals."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from scholar_lens.adapters.bedrock.usage import TokenUsageTracker
from scholar_lens.domain.model_registry import event_cost_usd
from scholar_lens.domain.usage import UNSTAGED, RunUsage, UsageEvent
from scholar_lens.domain.usage_context import current_stage, staged, usage_stage

OPUS = "us.anthropic.claude-opus-5-5"


def _converse_response(input_tokens: int, output_tokens: int, read=0, write=0):
    message = SimpleNamespace(
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_token_details": {"cache_read": read, "cache_creation": write},
        },
        response_metadata={"model_name": OPUS},
    )
    return SimpleNamespace(
        llm_output=None, generations=[[SimpleNamespace(message=message)]]
    )


def test_stage_follows_tasks_and_threads() -> None:
    @staged("outline")
    async def outline() -> tuple[str, str, str]:
        task = await asyncio.create_task(asyncio.sleep(0, current_stage()))
        thread = await asyncio.to_thread(current_stage)
        with usage_stage("verify"):
            inner = current_stage()
        return task, thread, inner

    assert asyncio.run(outline()) == ("outline", "outline", "verify")
    assert current_stage() == UNSTAGED


def test_cached_tokens_are_split_out_and_priced_at_their_multiples() -> None:
    tracker = TokenUsageTracker()
    with usage_stage("draft"):
        tracker.on_llm_end(_converse_response(10_000, 500, read=8_000, write=1_000))
    (event,) = tracker.events
    assert (event.stage, event.model_id) == ("draft", OPUS)
    assert (event.input_tokens, event.cache_read_tokens, event.cache_write_tokens) == (
        1_000,
        8_000,
        1_000,
    )
    # Opus 5.5: $4 in / $20 out per MTok; reads 0.1x, writes 1.25x the input.
    expected = (1_000 * 4 + 8_000 * 0.4 + 1_000 * 5 + 500 * 20) / 1e6
    assert event.cost_usd == pytest.approx(expected)
    # The budget still counts every token the model read.
    assert tracker.input_tokens == 10_000


def test_failed_calls_are_counted_by_stage() -> None:
    tracker = TokenUsageTracker()
    with usage_stage("review"):
        tracker.on_llm_error(RuntimeError("throttled"), run_id="r1")
        tracker.on_llm_end(_converse_response(100, 10), run_id="r2")
    usage = tracker.run_usage()
    assert usage.by_stage["review"].calls == 2
    assert usage.by_stage["review"].failed_calls == 1


def test_latency_is_measured_from_the_start_callback() -> None:
    tracker = TokenUsageTracker()
    tracker.on_chat_model_start({}, [], run_id="r1")
    tracker.on_llm_end(_converse_response(100, 10), run_id="r1")
    assert tracker.events[0].latency_ms is not None


def test_run_usage_totals_and_top_stages() -> None:
    events = [
        UsageEvent(stage="draft", model_id=OPUS, input_tokens=100, cost_usd=0.5),
        UsageEvent(stage="draft", model_id=OPUS, output_tokens=10, cost_usd=0.25),
        UsageEvent(stage="verify", model_id="haiku", input_tokens=5, cost_usd=0.1),
    ]
    usage = RunUsage.of(events, "v1")
    assert usage.total.calls == 3 and usage.total.cost_usd == pytest.approx(0.85)
    assert usage.by_model["haiku"].calls == 1
    assert usage.top_stages(1) == [("draft", 0.75)]
    # A manifest round-trips through JSON (JobResult.usage).
    assert RunUsage.model_validate_json(usage.model_dump_json()) == usage


def test_uncached_event_cost_matches_list_prices() -> None:
    event = UsageEvent(model_id=OPUS, input_tokens=1_000_000, output_tokens=1_000_000)
    assert event_cost_usd(event) == pytest.approx(24.0)


@pytest.mark.parametrize("stop", ["end_turn", "max_tokens"])
def test_a_response_cut_at_its_limit_is_still_priced(
    monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    # Bedrock bills a response cut at max_tokens: its usage counts toward
    # the run's cost and budgets even though the call raises.
    from langchain_aws import ChatBedrockConverse
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from scholar_lens.adapters.bedrock.chat_models import (
        CheckedChatBedrockConverse,
        OutputTruncatedError,
    )

    def generate(self, messages, stop_=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
        message = AIMessage(
            content="partial",
            usage_metadata={
                "input_tokens": 1_000,
                "output_tokens": 2_000,
                "total_tokens": 3_000,
            },
            response_metadata={"stopReason": stop, "model_name": "m"},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    monkeypatch.setattr(ChatBedrockConverse, "_generate", generate)
    tracker = TokenUsageTracker()
    model = CheckedChatBedrockConverse(
        model="anthropic.claude-test",
        region_name="us-west-2",
        max_tokens=2_000,
        callbacks=[tracker],
    )
    if stop == "max_tokens":
        with pytest.raises(OutputTruncatedError):
            model.invoke([HumanMessage(content="hi")])
    else:
        model.invoke([HumanMessage(content="hi")])
    assert tracker.output_tokens == 2_000 and tracker.cost_usd > 0
    assert tracker.events[-1].outcome == ("failed" if stop == "max_tokens" else "ok")
