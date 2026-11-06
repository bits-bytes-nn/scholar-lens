"""The Strands adapter: usage per cycle, content-filtered text calls
repeated, and an answer cut at the output limit returned with its usage (no
Bedrock: Agent is replaced)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from strands.types.exceptions import MaxTokensReachedException

from scholar_lens.adapters.bedrock import strands_models
from scholar_lens.adapters.bedrock.strands_models import RoleModel, StrandsLanguageModel


class FakeResult:
    def __init__(self, stop_reason: str, text: str) -> None:
        # "max_tokens": the agent raises, as Strands does.
        self.stop_reason = stop_reason
        self._text = text
        usage = {"inputTokens": 10, "outputTokens": 5}
        self.metrics = SimpleNamespace(
            agent_invocations=[SimpleNamespace(cycles=[SimpleNamespace(usage=usage)])],
            accumulated_usage=usage,
        )

    def __str__(self) -> str:
        return self._text


def _model(
    monkeypatch: pytest.MonkeyPatch, results: list[FakeResult]
) -> tuple[StrandsLanguageModel, list]:
    seen: list = []

    class FakeAgent:
        def __init__(self, *, model, **kwargs) -> None:
            assert kwargs["retry_strategy"] is None
            seen.append(model)
            self.messages: list = []

        async def invoke_async(self, prompt, **_):
            result = results.pop(0)
            self.event_loop_metrics = result.metrics
            if result.stop_reason == "max_tokens":
                self.messages.append(
                    {"role": "assistant", "content": [{"text": str(result)}]}
                )
                raise MaxTokensReachedException("cut")
            return result

    monkeypatch.setattr(strands_models, "Agent", FakeAgent)
    monkeypatch.setattr(strands_models, "BedrockModel", lambda **kw: kw)
    events: list = []
    model = StrandsLanguageModel(
        None,  # type: ignore[arg-type]
        {
            "write": RoleModel(
                "arn:x", "global.anthropic.claude-opus-5-5", thinking=True
            )
        },
        events.append,
    )
    model._events = events  # type: ignore[attr-defined]
    return model, seen


def test_content_filtered_calls_are_repeated_last_without_thinking(monkeypatch) -> None:
    results = [FakeResult("content_filtered", "부분")] * 3 + [
        FakeResult("end_turn", "완성된 절입니다.")
    ]
    model, seen = _model(monkeypatch, list(results))
    assert (
        asyncio.run(model.text(role="write", system="s", prompt="p"))
        == "완성된 절입니다."
    )
    assert ["thinking" in m["additional_request_fields"] for m in seen] == [
        True,
        True,
        True,
        False,
    ]
    assert len(model._events) == 4  # type: ignore[attr-defined]


def test_a_plain_answer_is_not_repeated(monkeypatch) -> None:
    model, seen = _model(monkeypatch, [FakeResult("end_turn", "답입니다.")])
    assert asyncio.run(model.text(role="write", system="s", prompt="p")) == "답입니다."
    assert len(seen) == 1


def test_a_filtered_call_ends_on_the_fallback_model(monkeypatch) -> None:
    results = [FakeResult("content_filtered", "부분")] * 4 + [
        FakeResult("end_turn", "다른 모델의 절입니다.")
    ]
    seen: list = []

    class FakeAgent:
        def __init__(self, *, model, **_) -> None:
            seen.append(model["model_id"])

        async def invoke_async(self, prompt, **_):
            result = results.pop(0)
            self.event_loop_metrics = result.metrics
            return result

    monkeypatch.setattr(strands_models, "Agent", FakeAgent)
    monkeypatch.setattr(strands_models, "BedrockModel", lambda **kw: kw)
    events: list = []
    fallback = RoleModel("sonnet-arn", "global.anthropic.claude-sonnet-5-5")
    model = StrandsLanguageModel(
        None,  # type: ignore[arg-type]
        {
            "write": RoleModel(
                "opus-arn", "global.anthropic.claude-opus-5-5", fallback=fallback
            )
        },
        events.append,
    )
    assert (
        asyncio.run(model.text(role="write", system="s", prompt="p"))
        == "다른 모델의 절입니다."
    )
    assert seen == ["opus-arn"] * 4 + ["sonnet-arn"]
    assert events[-1].model_id == "global.anthropic.claude-sonnet-5-5"


def test_shared_context_goes_before_a_cache_point() -> None:
    from scholar_lens.adapters.bedrock.strands_models import _content

    assert _content("", "check these") == "check these"
    assert _content("evidence E1", "check these") == [
        {"text": "evidence E1"},
        {"cachePoint": {"type": "default"}},
        {"text": "check these"},
    ]


def test_an_answer_cut_at_the_output_limit_is_returned_with_its_usage(
    monkeypatch,
) -> None:
    model, _ = _model(monkeypatch, [FakeResult("max_tokens", "잘린 절")])
    text = asyncio.run(model.text(role="write", system="s", prompt="p"))
    assert text == "잘린 절"
    assert [e.output_tokens for e in model._events] == [5]  # type: ignore[attr-defined]


def test_an_agent_cut_at_the_output_limit_keeps_its_outcome(monkeypatch) -> None:
    from scholar_lens.application.ports import AgentLimits

    model, _ = _model(monkeypatch, [FakeResult("max_tokens", "")])
    outcome = asyncio.run(
        model.run_agent(
            role="write", system="s", prompt="p", tools=[], limits=AgentLimits(turns=3)
        )
    )
    assert outcome.stop_reason == "max_tokens"
    assert len(model._events) == 1  # type: ignore[attr-defined]


def test_an_evidence_run_stops_at_its_cost_ceiling() -> None:
    from scholar_lens.domain.usage import UsageEvent
    from scholar_lens.src.evidence_models import budgeted
    from scholar_lens.src.metrics import TokenBudgetExceeded, TokenUsageTracker

    tracker = TokenUsageTracker()
    record = budgeted(tracker, 0.01)
    event = UsageEvent(
        stage="write",
        model_id="global.anthropic.claude-opus-5-5",
        input_tokens=1_000,
        output_tokens=100,
    )
    record(event)
    with pytest.raises(TokenBudgetExceeded):
        record(event.model_copy(update={"input_tokens": 100_000}))
    assert tracker.call_count == 2
    budgeted(tracker, None)(event)
