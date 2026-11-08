"""Test doubles shared across the suite (no AWS, no network)."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from scholar_lens.src.utils import BedrockLanguageModelFactory


class FakeLLMFactory:
    """Stands in for :class:`BedrockLanguageModelFactory`: no Bedrock client or
    inference-profile lookup, every model answers ``reply`` and texts fit
    unchanged. Tests replace the chains they drive."""

    DEFAULT_CONTEXT_RESERVE_TOKENS = (
        BedrockLanguageModelFactory.DEFAULT_CONTEXT_RESERVE_TOKENS
    )

    def __init__(self, reply: str = "", *, context_window: int = 200_000) -> None:
        self.reply = reply
        self.context_window = context_window
        self.models: list[tuple[Any, dict[str, Any]]] = []

    def get_model(self, model_id: Any, **kwargs: Any) -> Runnable:
        self.models.append((model_id, kwargs))
        return RunnableLambda(lambda _: self.reply)

    def fit_text(self, model_id: Any, text: str, **_: Any) -> str:
        return text

    def effective_context_window(self, model_id: Any) -> int:
        return self.context_window


class FakeEmbeddingFactory:
    """Stands in for :class:`BedrockEmbeddingModelFactory` (no Bedrock client)."""

    def __init__(self, max_sequence_length: int = 8192) -> None:
        self.max_sequence_length = max_sequence_length

    def get_model(self, model_id: Any, **_: Any) -> Any:
        return None

    def get_max_sequence_length(self, model_id: Any) -> int:
        return self.max_sequence_length


def with_context(context: str | tuple[str, ...], prompt: str) -> str:
    """A scripted model's view of a call: its context parts (one string, or
    the parts a cached call passes) before its prompt."""
    parts = (context,) if isinstance(context, str) else context
    return "\n\n".join([*parts, prompt])
