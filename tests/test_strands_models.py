"""The Strands adapter: usage per cycle, content-filtered text calls
repeated, and an answer cut at the output limit returned with its usage (no
Bedrock: Agent is replaced)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import boto3
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
            assert kwargs["retry_strategy"] is not None
            seen.append(model)
            self.messages: list = []

        def add_middleware(self, *_) -> None:
            pass

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

        def add_middleware(self, *_) -> None:
            pass

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
    from scholar_lens.adapters.bedrock.usage import TokenUsageTracker
    from scholar_lens.domain.usage import TokenBudgetExceeded, UsageEvent
    from scholar_lens.src.evidence_models import budgeted

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


def test_a_content_filtered_structured_call_moves_to_the_fallback_model(
    monkeypatch,
) -> None:
    from pydantic import BaseModel

    class Answer(BaseModel):
        section: str

    class Result:
        def __init__(self, stop: str, output: object) -> None:
            self.stop_reason = stop
            self.structured_output = output
            usage = {"inputTokens": 1, "outputTokens": 1}
            self.metrics = SimpleNamespace(
                agent_invocations=[
                    SimpleNamespace(cycles=[SimpleNamespace(usage=usage)])
                ],
                accumulated_usage=usage,
            )

    answers = [Result("content_filtered", None)] * 4 + [
        Result("end_turn", Answer(section="ok"))
    ]
    seen: list = []

    class FakeAgent:
        def __init__(self, *, model, **_) -> None:
            seen.append(model["model_id"])
            self.messages: list = []

        def add_middleware(self, *_) -> None:
            pass

        async def invoke_async(self, prompt, **_):
            result = answers.pop(0)
            self.event_loop_metrics = result.metrics
            return result

    monkeypatch.setattr(strands_models, "Agent", FakeAgent)
    monkeypatch.setattr(strands_models, "BedrockModel", lambda **kw: kw)
    fallback = RoleModel("sonnet-arn", "global.anthropic.claude-sonnet-5-5")
    model = StrandsLanguageModel(
        None,  # type: ignore[arg-type]
        {"write": RoleModel("opus-arn", "opus", fallback=fallback)},
        lambda _e: None,
    )
    out = asyncio.run(
        model.structured(role="write", system="s", prompt="p", schema=Answer)
    )
    assert out.section == "ok" and seen[-1] == "sonnet-arn"


class _StubBedrock:
    """Answers ConverseStream with no network and records each request: text,
    or the structured answer the request asks for (a native output schema, or
    the tool it offers)."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def converse_stream(self, **request):
        self.requests.append(request)
        tools = request.get("toolConfig", {}).get("tools", [])
        names = [t["toolSpec"]["name"] for t in tools if "toolSpec" in t]
        answer = json.dumps({"section": "ok"})
        if "outputConfig" in request or not names:
            text = answer if "outputConfig" in request else "답입니다."
            blocks: list[dict] = [
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": text},
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 0}},
            ]
            stop = "end_turn"
        else:
            start = {"toolUse": {"toolUseId": "t1", "name": names[0]}}
            blocks = [
                {"contentBlockStart": {"contentBlockIndex": 0, "start": start}},
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"toolUse": {"input": answer}},
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 0}},
            ]
            stop = "tool_use"
        usage = {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}
        return {
            "stream": [
                {"messageStart": {"role": "assistant"}},
                *blocks,
                {"messageStop": {"stopReason": stop}},
                {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}},
            ]
        }


def _stubbed(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[StrandsLanguageModel, _StubBedrock]:
    """The adapter on real Strands agents and BedrockModel, whose Bedrock
    client is replaced by a stub (static fake credentials: no AWS call)."""
    stub = _StubBedrock()
    real = strands_models.BedrockModel

    def model(**kwargs):
        built = real(**kwargs)
        built.client.converse_stream = stub.converse_stream
        return built

    monkeypatch.setattr(strands_models, "BedrockModel", model)
    session = boto3.Session(
        aws_access_key_id="stub",
        aws_secret_access_key="stub",
        region_name="us-east-1",
    )
    profile = "arn:aws:bedrock:us-east-1:000000000000:application-inference-profile/x"
    adapter = StrandsLanguageModel(
        session, {"write": RoleModel(profile, "opus")}, lambda _e: None
    )
    return adapter, stub


def _message_cache_points(request: dict) -> int:
    return sum(
        "cachePoint" in block
        for message in request["messages"]
        for block in message["content"]
    )


def test_a_one_shot_text_call_writes_no_message_to_the_cache(monkeypatch) -> None:
    model, stub = _stubbed(monkeypatch)
    asyncio.run(model.text(role="write", system="s", prompt="p"))
    (request,) = stub.requests
    assert _message_cache_points(request) == 0
    assert "cachePoint" in request["system"][-1]


def test_a_text_call_with_context_caches_only_the_shared_prefix(monkeypatch) -> None:
    model, stub = _stubbed(monkeypatch)
    asyncio.run(model.text(role="write", system="s", prompt="p", context="evidence"))
    (request,) = stub.requests
    assert request["messages"][-1]["content"] == [
        {"text": "evidence"},
        {"cachePoint": {"type": "default"}},
        {"text": "p"},
    ]
    assert "cachePoint" in request["system"][-1]


@pytest.mark.parametrize("context, points", [("", 0), ("evidence", 1)])
def test_a_structured_call_caches_a_message_only_with_context(
    monkeypatch, context: str, points: int
) -> None:
    from pydantic import BaseModel

    class Answer(BaseModel):
        section: str

    model, stub = _stubbed(monkeypatch)
    out = asyncio.run(
        model.structured(
            role="write", system="s", prompt="p", schema=Answer, context=context
        )
    )
    assert out.section == "ok"
    assert stub.requests
    assert [_message_cache_points(r) for r in stub.requests] == [points] * len(
        stub.requests
    )
    assert all("cachePoint" in r["system"][-1] for r in stub.requests)


def test_an_agent_conversation_keeps_its_message_cache_point(monkeypatch) -> None:
    from scholar_lens.application.ports import AgentLimits

    model, stub = _stubbed(monkeypatch)
    asyncio.run(
        model.run_agent(
            role="write", system="s", prompt="p", tools=[], limits=AgentLimits(turns=3)
        )
    )
    assert [_message_cache_points(r) for r in stub.requests] == [1]


def test_context_parts_share_one_cache_point() -> None:
    from scholar_lens.adapters.bedrock.strands_models import _content

    point = {"cachePoint": {"type": "default"}}
    # Parts join, most stable first, before the one cache point.
    assert _content(("earlier", "section"), "check") == [
        {"text": "earlier\n\nsection"},
        point,
        {"text": "check"},
    ]
    # Empty parts are left out; no part, no cache point.
    assert _content(("", "section"), "check") == [
        {"text": "section"},
        point,
        {"text": "check"},
    ]
    assert _content(("", " "), "check") == "check"


@pytest.mark.parametrize(
    "context, points", [(("", ""), 0), (("earlier", ""), 1), (("earlier", "own"), 1)]
)
def test_a_structured_call_caches_its_context_once(
    monkeypatch, context: tuple[str, ...], points: int
) -> None:
    from pydantic import BaseModel

    class Answer(BaseModel):
        section: str

    model, stub = _stubbed(monkeypatch)
    asyncio.run(
        model.structured(
            role="write", system="s", prompt="p", schema=Answer, context=context
        )
    )
    assert stub.requests
    assert [_message_cache_points(r) for r in stub.requests] == [points] * len(
        stub.requests
    )
    first = stub.requests[0]["messages"][0]["content"]
    assert first[-1] == {"text": "p"}
    # The request stays within Bedrock's four cache points.
    assert all(
        _message_cache_points(r)
        + sum("cachePoint" in b for b in r["system"])
        + sum("cachePoint" in t for t in r.get("toolConfig", {}).get("tools", []))
        <= 4
        for r in stub.requests
    )
