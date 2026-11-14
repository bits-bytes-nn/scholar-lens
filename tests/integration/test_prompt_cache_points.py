"""The sections engine's prompt caching: where cache points go in the
request langchain-aws sends, and that the text the model reads is unchanged."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda

from scholar_lens.adapters.bedrock.chat_models import CheckedChatBedrockConverse
from scholar_lens.adapters.bedrock.prompts.base import with_cache_points
from scholar_lens.sections_engine.prompts.review import (
    PaperFinalizationPrompt,
    PaperSectionCheckPrompt,
    PaperSectionPrompt,
)

POINT = {"cachePoint": {"type": "default"}}


class _Client:
    """A bedrock-runtime client that records the Converse request."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def converse(self, **request: Any) -> dict[str, Any]:
        self.requests.append(request)
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "metrics": {"latencyMs": 1},
        }


def _request(prompt_cls: Any, *, cached: bool = True) -> dict[str, Any]:
    client = _Client()
    llm = CheckedChatBedrockConverse(
        model_id="us.anthropic.claude-opus-5-5",
        region_name="us-west-2",
        client=client,
        provider="anthropic",
    )
    # Blank and padded values too: the text must be the same to the character.
    values = {v: f"  [{v}]\n" for v in prompt_cls.input_variables}
    values[prompt_cls.input_variables[-1]] = ""
    chain = prompt_cls.get_prompt()
    if cached:
        chain = chain | RunnableLambda(with_cache_points(prompt_cls))
    (chain | llm).invoke(values)
    (request,) = client.requests
    return request


def _text(blocks: list[dict[str, Any]]) -> str:
    return "".join(b["text"] for b in blocks if "text" in b)


def _points(request: dict[str, Any]) -> int:
    blocks = [
        *request["system"],
        *(b for m in request["messages"] for b in m["content"]),
        *request.get("toolConfig", {}).get("tools", []),
    ]
    return sum(b == POINT for b in blocks)


@pytest.mark.parametrize(
    "prompt_cls, points",
    [
        (PaperSectionPrompt, 2),
        (PaperSectionCheckPrompt, 1),
        (PaperFinalizationPrompt, 0),
    ],
)
def test_cache_points_split_the_request_without_changing_its_text(
    prompt_cls: Any, points: int
) -> None:
    cached, plain = _request(prompt_cls), _request(prompt_cls, cached=False)
    assert _points(cached) == points <= 4
    assert _points(plain) == 0
    assert _text(cached["system"]) == _text(plain["system"])
    assert [_text(m["content"]) for m in cached["messages"]] == [
        _text(m["content"]) for m in plain["messages"]
    ]
    rest = {k: v for k, v in cached.items() if k not in ("system", "messages")}
    assert rest == {k: v for k, v in plain.items() if k not in ("system", "messages")}


def test_the_writer_caches_its_system_prompt_and_the_run_prefix() -> None:
    request = _request(PaperSectionPrompt)
    assert request["system"][-1] == POINT
    first, point, rest = request["messages"][0]["content"]
    assert point == POINT
    assert first["text"].endswith("</translation_guideline>")
    assert "[plan_outline]" in first["text"] and "[glossary]" in first["text"]
    # What changes per section or call follows the point.
    assert "[section_brief]" in rest["text"] and "[current_explanation]" in rest["text"]
