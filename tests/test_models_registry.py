"""Tests for the language/embedding model registries and Opus 4.8 wiring."""

from __future__ import annotations

import pytest

from scholar_lens.src.constants import EmbeddingModelId, LanguageModelId
from scholar_lens.src.utils.models import (
    _EMBEDDING_MODEL_INFO,
    _LANGUAGE_MODEL_INFO,
)


class TestLanguageModelRegistry:
    def test_every_enum_member_has_registry_entry(self) -> None:
        missing = [m.name for m in LanguageModelId if m not in _LANGUAGE_MODEL_INFO]
        assert not missing, f"models missing registry info: {missing}"

    def test_opus_4_8_present_and_consistent(self) -> None:
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_8_OPUS]
        assert info.supports_1m_context_window is True
        assert info.supports_thinking is True
        assert info.supports_prompt_caching is True

    @pytest.mark.parametrize(
        "model",
        [
            LanguageModelId.CLAUDE_V4_6_SONNET,
            LanguageModelId.CLAUDE_V4_6_OPUS,
            LanguageModelId.CLAUDE_V4_7_OPUS,
            LanguageModelId.CLAUDE_V4_8_OPUS,
        ],
    )
    def test_v4_6_plus_cohort_is_self_consistent(self, model: LanguageModelId) -> None:
        # Regression: the 4.6+ cohort previously had contradictory flags
        # (1M context_window_size but supports_1m_context_window=False).
        info = _LANGUAGE_MODEL_INFO[model]
        assert info.supports_1m_context_window is True
        assert info.supports_prompt_caching is True
        assert info.supports_thinking is True

    @pytest.mark.parametrize(
        ("model", "model_id"),
        [
            (LanguageModelId.CLAUDE_V5_5_OPUS, "anthropic.claude-opus-5-5"),
            (LanguageModelId.CLAUDE_V5_5_SONNET, "anthropic.claude-sonnet-5-5"),
            (LanguageModelId.CLAUDE_V5_5_HAIKU, "anthropic.claude-haiku-5-5"),
            (LanguageModelId.CLAUDE_V5_1_FABLE, "anthropic.claude-fable-5-1"),
        ],
    )
    def test_latest_models_registered(
        self, model: LanguageModelId, model_id: str
    ) -> None:
        # Bedrock base ids carry no date/version suffix for these models.
        assert model.value == model_id
        assert LanguageModelId(model_id) is model
        info = _LANGUAGE_MODEL_INFO[model]
        assert info.uses_adaptive_thinking is True
        assert info.supports_thinking is True
        assert info.supports_1m_context_window is True
        assert info.max_output_tokens == 128000
        # Not CountTokens-capable; counted via the same-family proxy.
        assert info.count_tokens_proxy == LanguageModelId.CLAUDE_V4_6_SONNET.value

    def test_positive_token_limits(self) -> None:
        for model, info in _LANGUAGE_MODEL_INFO.items():
            assert info.context_window_size > 0, model
            assert info.max_output_tokens > 0, model


class TestEmbeddingModelRegistry:
    def test_every_enum_member_has_registry_entry(self) -> None:
        missing = [m.name for m in EmbeddingModelId if m not in _EMBEDDING_MODEL_INFO]
        assert not missing, f"embedding models missing info: {missing}"


class TestRequestShapeFlags:
    @pytest.mark.parametrize(
        ("model", "price"),
        [
            (LanguageModelId.CLAUDE_V5_5_OPUS, (4.0, 20.0)),
            (LanguageModelId.CLAUDE_V5_5_SONNET, (2.0, 10.0)),
            (LanguageModelId.CLAUDE_V5_5_HAIKU, (0.10, 0.50)),
            (LanguageModelId.CLAUDE_V5_1_FABLE, (10.0, 50.0)),
            (LanguageModelId.CLAUDE_V4_8_OPUS, (5.0, 25.0)),
            (LanguageModelId.CLAUDE_V5_SONNET, (2.0, 10.0)),
        ],
    )
    def test_price(self, model: LanguageModelId, price: tuple[float, float]) -> None:
        info = _LANGUAGE_MODEL_INFO[model]
        assert (info.input_price_per_mtok, info.output_price_per_mtok) == price

    def test_every_model_is_priced(self) -> None:
        for model, info in _LANGUAGE_MODEL_INFO.items():
            assert (
                info.input_price_per_mtok > 0 and info.output_price_per_mtok > 0
            ), model
