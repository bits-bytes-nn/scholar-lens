"""Tests for utils.factories pure/offline logic (no live boto3/Bedrock).

The boto3 ``Session`` is replaced with a ``MagicMock`` so the factory can be
constructed without any network access; ``Session.client`` returns a mock and
is never exercised against AWS here. Only deterministic, offline logic is
tested: ``max_tokens`` validation, thinking/performance flag logic, config
construction (including the 1M-context-window beta injection), token
truncation via tiktoken, and ``get_model_info`` lookup.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from scholar_lens.src.constants import LanguageModelId
from scholar_lens.src.utils.factories import (
    BaseBedrockWrapper,
    BedrockLanguageModelFactory,
)
from scholar_lens.src.utils.models import _LANGUAGE_MODEL_INFO


@pytest.fixture(autouse=True)
def _silence_logging() -> None:
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _make_factory(profile_name: str = "default") -> BedrockLanguageModelFactory:
    """Build a factory backed by a fully mocked boto3 session (no network)."""
    session = MagicMock()
    session.profile_name = profile_name
    session.region_name = "us-east-1"
    session.client.return_value = MagicMock()
    return BedrockLanguageModelFactory(boto_session=session, region_name="us-east-1")


# Reference model infos used across tests.
_HAIKU_V3 = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V3_HAIKU]  # 4096, no thinking
_HAIKU_V3_5 = _LANGUAGE_MODEL_INFO[
    LanguageModelId.CLAUDE_V3_5_HAIKU
]  # perf optimization
_OPUS_V4_8 = _LANGUAGE_MODEL_INFO[
    LanguageModelId.CLAUDE_V4_8_OPUS
]  # 64000, adaptive thinking, 1m, temperature deprecated
_SONNET_V4_5 = _LANGUAGE_MODEL_INFO[
    LanguageModelId.CLAUDE_V4_5_SONNET
]  # 64000, legacy (enabled+budget) thinking, 1m


class TestValidateMaxTokens:
    def test_caps_when_over_model_max(self) -> None:
        assert (
            BedrockLanguageModelFactory._validate_max_tokens(999_999, _HAIKU_V3)
            == _HAIKU_V3.max_output_tokens
        )

    def test_returns_requested_when_under(self) -> None:
        assert BedrockLanguageModelFactory._validate_max_tokens(100, _HAIKU_V3) == 100

    def test_defaults_to_model_max_when_none(self) -> None:
        assert (
            BedrockLanguageModelFactory._validate_max_tokens(None, _HAIKU_V3)
            == _HAIKU_V3.max_output_tokens
        )


class TestShouldEnableThinking:
    def test_enabled_when_flag_and_support(self) -> None:
        assert (
            BedrockLanguageModelFactory._should_enable_thinking(True, _OPUS_V4_8)
            is True
        )

    def test_disabled_without_flag(self) -> None:
        assert (
            BedrockLanguageModelFactory._should_enable_thinking(False, _OPUS_V4_8)
            is False
        )

    def test_disabled_when_unsupported(self) -> None:
        assert (
            BedrockLanguageModelFactory._should_enable_thinking(True, _HAIKU_V3)
            is False
        )


class TestShouldEnablePerformanceOptimization:
    def test_enabled_when_flag_support_and_not_cross_region(self) -> None:
        assert (
            BedrockLanguageModelFactory._should_enable_performance_optimization(
                True, _HAIKU_V3_5, False
            )
            is True
        )

    def test_disabled_when_cross_region(self) -> None:
        assert (
            BedrockLanguageModelFactory._should_enable_performance_optimization(
                True, _HAIKU_V3_5, True
            )
            is False
        )

    def test_disabled_when_unsupported(self) -> None:
        assert (
            BedrockLanguageModelFactory._should_enable_performance_optimization(
                True, _HAIKU_V3, False
            )
            is False
        )


class TestBuildBaseConfig:
    def test_default_profile_omits_credentials(self) -> None:
        factory = _make_factory(profile_name="default")
        config = factory._build_base_config("m", is_cross_region=False)
        assert "credentials_profile_name" not in config
        assert config["model_id"] == "m"
        assert config["region_name"] == "us-east-1"

    def test_named_profile_sets_credentials(self) -> None:
        factory = _make_factory(profile_name="myprofile")
        config = factory._build_base_config("m", is_cross_region=False)
        assert config["credentials_profile_name"] == "myprofile"

    def test_non_cross_region_uses_model_kwargs(self) -> None:
        factory = _make_factory()
        config = factory._build_base_config("m", is_cross_region=False)
        assert "model_kwargs" in config
        assert config["model_kwargs"]["top_k"] == factory.DEFAULT_TOP_K
        assert config["model_kwargs"]["stop_sequences"] == ["\n\nHuman:"]

    def test_cross_region_inlines_common_params(self) -> None:
        factory = _make_factory()
        config = factory._build_base_config("m", is_cross_region=True)
        assert "model_kwargs" not in config
        assert config["stop_sequences"] == ["\n\nHuman:"]


class TestBuildModelConfig1MContextWindow:
    def test_cross_region_beta_header_present(self) -> None:
        factory = _make_factory()
        config = factory._build_model_config(
            _OPUS_V4_8,
            "global.anthropic.claude-opus-4-8",
            is_cross_region=True,
            supports_1m_context_window=True,
        )
        assert config["additional_model_request_fields"]["anthropic_beta"] == [
            "context-1m-2025-08-07"
        ]

    def test_non_cross_region_beta_header_present(self) -> None:
        factory = _make_factory()
        config = factory._build_model_config(
            _OPUS_V4_8,
            "anthropic.claude-opus-4-8",
            is_cross_region=False,
            supports_1m_context_window=True,
        )
        beta = config["model_kwargs"]["additionalModelRequestFields"]["anthropic_beta"]
        assert beta == ["context-1m-2025-08-07"]

    def test_beta_header_absent_when_flag_off(self) -> None:
        factory = _make_factory()
        config = factory._build_model_config(
            _OPUS_V4_8,
            "anthropic.claude-opus-4-8",
            is_cross_region=False,
            supports_1m_context_window=False,
        )
        assert "additionalModelRequestFields" not in config["model_kwargs"]

    def test_beta_header_absent_when_model_unsupported(self) -> None:
        # Even with the kwarg on, a model that does not support 1M context
        # must not get the beta header.
        factory = _make_factory()
        config = factory._build_model_config(
            _HAIKU_V3,
            "anthropic.claude-3-haiku",
            is_cross_region=False,
            supports_1m_context_window=True,
        )
        assert "additionalModelRequestFields" not in config["model_kwargs"]


class TestBuildModelConfigFeatures:
    def test_legacy_thinking_sets_temperature_and_budget_cross_region(self) -> None:
        # Legacy thinking models (Sonnet 4.5) use enabled+budget and DO send
        # temperature (forced to 1.0 in thinking mode).
        factory = _make_factory()
        config = factory._build_model_config(
            _SONNET_V4_5,
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            is_cross_region=True,
            enable_thinking=True,
        )
        assert config["temperature"] == 1.0
        assert config["max_tokens"] == _SONNET_V4_5.max_output_tokens
        thinking = config["additional_model_request_fields"]["thinking"]
        assert thinking["type"] == "enabled"
        assert (
            thinking["budget_tokens"]
            == BedrockLanguageModelFactory.DEFAULT_THINKING_BUDGET_TOKENS
        )

    def test_adaptive_thinking_omits_temperature_cross_region(self) -> None:
        # Opus 4.8 uses adaptive thinking + effort and DEPRECATES temperature,
        # so the config must NOT include a temperature key.
        factory = _make_factory()
        config = factory._build_model_config(
            _OPUS_V4_8,
            "us.anthropic.claude-opus-4-8",
            is_cross_region=True,
            enable_thinking=True,
        )
        assert "temperature" not in config
        assert config["max_tokens"] == _OPUS_V4_8.max_output_tokens
        fields = config["additional_model_request_fields"]
        assert fields["thinking"] == {"type": "adaptive"}
        assert fields["output_config"]["effort"] == (
            BedrockLanguageModelFactory.DEFAULT_THINKING_EFFORT
        )

    def test_no_thinking_keeps_default_temperature(self) -> None:
        factory = _make_factory()
        config = factory._build_model_config(
            _HAIKU_V3, "anthropic.claude-3-haiku", is_cross_region=False
        )
        assert (
            config["model_kwargs"]["temperature"]
            == BedrockLanguageModelFactory.DEFAULT_TEMPERATURE
        )

    def test_performance_optimization_applied(self) -> None:
        factory = _make_factory()
        config = factory._build_model_config(
            _HAIKU_V3_5,
            "anthropic.claude-3-5-haiku",
            is_cross_region=False,
            enable_performance_optimization=True,
        )
        assert config["performanceConfig"]["latency"] == (
            BedrockLanguageModelFactory.DEFAULT_LATENCY_MODE
        )


class TestGetModelInfo:
    def test_known_model_returns_info(self) -> None:
        factory = _make_factory()
        info = factory.get_model_info(LanguageModelId.CLAUDE_V4_8_OPUS)
        assert info is _OPUS_V4_8

    def test_supported_models_listed(self) -> None:
        factory = _make_factory()
        models = factory.get_supported_models()
        assert LanguageModelId.CLAUDE_V4_8_OPUS in models
        assert len(models) == len(_LANGUAGE_MODEL_INFO)


class TestBaseBedrockWrapperTruncation:
    def test_no_limits_returns_unchanged(self) -> None:
        wrapper = BaseBedrockWrapper()
        assert wrapper._truncate_text("abc", None, None, "doc") == "abc"

    def test_char_truncation(self) -> None:
        wrapper = BaseBedrockWrapper()
        out = wrapper._truncate_text(
            "abcdefghij", max_chars=4, max_tokens=None, text_type="doc"
        )
        assert out == "abcd"

    def test_char_limit_not_exceeded_returns_unchanged(self) -> None:
        wrapper = BaseBedrockWrapper()
        out = wrapper._truncate_text(
            "abc", max_chars=100, max_tokens=None, text_type="doc"
        )
        assert out == "abc"

    def test_token_truncation_crashes_on_bare_wrapper(self) -> None:
        # POTENTIAL SOURCE BUG: BaseBedrockWrapper declares
        # ``buffer_tokens: int = Field(default=128, ge=0)`` but is a plain
        # (non-pydantic) class on its own; pydantic field resolution only
        # happens when it is mixed with BedrockEmbeddings. On a bare instance
        # ``self.buffer_tokens`` stays a FieldInfo object, so the token
        # truncation path raises TypeError on ``max_tokens - self.buffer_tokens``.
        # See scholar_lens/src/utils/factories.py:44.
        wrapper = BaseBedrockWrapper()
        # Inject a fake tokenizer so the token path is reached without
        # downloading the real BPE vocab (network-free / CI-safe).

        class _FakeTok:
            def encode(self, text: str, allowed_special: str = "all") -> list[int]:
                return list(range(len(text.split())))

            def decode(self, ids: list[int]) -> str:
                return "x"

        wrapper._tokenizer = _FakeTok()  # type: ignore[assignment]
        long_text = "word " * 100
        with pytest.raises(TypeError):
            wrapper._truncate_text(
                long_text, max_chars=None, max_tokens=10, text_type="doc"
            )
