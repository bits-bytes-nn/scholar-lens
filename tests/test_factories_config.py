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

    def test_max_tokens_is_ignored_only_char_cap_applies(self) -> None:
        # Embedding truncation is char-only now (Bedrock CountTokens doesn't
        # support embedding models); max_tokens is accepted but not enforced.
        wrapper = BaseBedrockWrapper()
        out = wrapper._truncate_text(
            "abcdefghij", max_chars=None, max_tokens=2, text_type="doc"
        )
        assert out == "abcdefghij"


class TestFitText:
    """fit_text/count_tokens against a fake CountTokens (no network).

    The fake counts 1 token per 4 characters and, like the real API, rejects
    inputs above a hard limit with a 'too long' ValidationException so the
    binary-search shrink path is exercised."""

    _HARD_LIMIT_TOKENS = 200_000

    def _factory_with_counter(self) -> BedrockLanguageModelFactory:
        factory = _make_factory()

        def fake_count_tokens(*, modelId: str, input: dict) -> dict:  # noqa: N803
            text = input["converse"]["messages"][0]["content"][0]["text"]
            tokens = -(-len(text) // 4)  # ceil(len/4)
            if tokens > self._HARD_LIMIT_TOKENS:
                raise RuntimeError(
                    "ValidationException: prompt is too long: "
                    f"{tokens} tokens > {self._HARD_LIMIT_TOKENS} maximum"
                )
            return {"inputTokens": tokens}

        factory._client.count_tokens = MagicMock(side_effect=fake_count_tokens)
        # Cross-region resolution would hit the network; keep the plain id.
        factory._client.list_inference_profiles = MagicMock(
            return_value={"inferenceProfileSummaries": []}
        )
        return factory

    def test_short_text_returned_unchanged(self) -> None:
        factory = self._factory_with_counter()
        text = "hello world"
        assert factory.fit_text(LanguageModelId.CLAUDE_V4_5_HAIKU, text) == text

    def test_empty_text_is_noop(self) -> None:
        factory = self._factory_with_counter()
        assert factory.fit_text(LanguageModelId.CLAUDE_V4_5_HAIKU, "") == ""

    def test_long_text_fitted_under_budget(self) -> None:
        factory = self._factory_with_counter()
        model = LanguageModelId.CLAUDE_V4_5_HAIKU  # 200k window
        # ~300k tokens of text (1.2M chars) — well over the budget.
        text = "x" * 1_200_000
        fitted = factory.fit_text(model, text)
        assert len(fitted) < len(text)
        budget = factory.effective_context_window(model) - (
            factory.DEFAULT_CONTEXT_RESERVE_TOKENS
        )
        # The fitted text's exact token count must be within budget.
        assert factory.count_tokens(model, fitted) <= budget

    def test_counter_error_leaves_text_intact(self) -> None:
        factory = self._factory_with_counter()
        factory._client.count_tokens = MagicMock(
            side_effect=RuntimeError("transient throttling")
        )
        text = "x" * 1_200_000
        # A non-"too long" error must not silently gut the content.
        assert factory.fit_text(LanguageModelId.CLAUDE_V4_5_HAIKU, text) == text

    def test_count_tokens_uses_base_model_id(self) -> None:
        # Regression: CountTokens rejects cross-region profile ids (apac./global.)
        # — it must be called with the BASE model id, not the resolved one.
        captured: dict = {}
        factory = _make_factory()
        factory._client.count_tokens = MagicMock(
            side_effect=lambda **kw: captured.update(kw) or {"inputTokens": 1}
        )
        factory.count_tokens(LanguageModelId.CLAUDE_V4_6_SONNET, "hi")
        assert captured["modelId"] == LanguageModelId.CLAUDE_V4_6_SONNET.value
        assert not captured["modelId"].startswith(("apac.", "global.", "us."))
