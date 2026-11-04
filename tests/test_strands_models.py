"""The Strands adapter: usage per cycle, and content-filtered text calls
repeated (no Bedrock: Agent is replaced)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from scholar_lens.adapters.bedrock import strands_models
from scholar_lens.adapters.bedrock.strands_models import RoleModel, StrandsLanguageModel


class FakeResult:
    def __init__(self, stop_reason: str, text: str) -> None:
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
        def __init__(self, *, model, **_) -> None:
            seen.append(model)

        async def invoke_async(self, prompt, **_):
            return results.pop(0)

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
