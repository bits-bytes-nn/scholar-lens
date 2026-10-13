"""Tests for utils.factories offline logic (mocked boto3, no Bedrock calls)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from scholar_lens.src.constants import LanguageModelId
from scholar_lens.src.utils import factories as fac
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


@pytest.fixture(autouse=True)
def _clear_profile_cache() -> None:
    fac._resolve_inference_profile.cache_clear()
    yield
    fac._resolve_inference_profile.cache_clear()


def _make_factory(profile_name: str = "default") -> BedrockLanguageModelFactory:
    """Build a factory backed by a fully mocked boto3 session (no network)."""
    session = MagicMock()
    session.profile_name = profile_name
    session.region_name = "us-east-1"
    session.client.return_value = MagicMock()
    return BedrockLanguageModelFactory(boto_session=session, region_name="us-east-1")


def _config(model: LanguageModelId, **kwargs) -> dict:  # type: ignore[no-untyped-def]
    return _make_factory()._build_model_config(
        _LANGUAGE_MODEL_INFO[model], model.value, **kwargs
    )


ALWAYS_ON = [
    LanguageModelId.CLAUDE_V5_5_OPUS,
    LanguageModelId.CLAUDE_V5_5_SONNET,
    LanguageModelId.CLAUDE_V5_5_HAIKU,
    LanguageModelId.CLAUDE_V5_1_FABLE,
]


class TestMaxTokens:
    def test_default_is_bounded_not_model_max(self) -> None:
        config = _config(LanguageModelId.CLAUDE_V5_5_OPUS)
        assert config["max_tokens"] == BedrockLanguageModelFactory.DEFAULT_MAX_TOKENS
        assert config["max_tokens"] < 128_000

    def test_request_capped_at_model_max(self) -> None:
        assert (
            _config(LanguageModelId.CLAUDE_V3_HAIKU, max_tokens=999_999)["max_tokens"]
            == 4096
        )


class TestAdaptiveModels:
    @pytest.mark.parametrize("model", ALWAYS_ON)
    def test_effort_sent_without_thinking_and_no_sampling(
        self, model: LanguageModelId
    ) -> None:
        config = _config(model, enable_thinking=False, thinking_effort="low")
        fields = config["additional_model_request_fields"]
        assert fields["output_config"] == {"effort": "low"}
        assert "thinking" not in fields  # never send type='disabled'
        assert "temperature" not in config

    @pytest.mark.parametrize("model", ALWAYS_ON)
    def test_thinking_requested_is_adaptive(self, model: LanguageModelId) -> None:
        fields = _config(model, enable_thinking=True, thinking_effort="high")[
            "additional_model_request_fields"
        ]
        assert fields["thinking"] == {"type": "adaptive"}
        assert fields["output_config"] == {"effort": "high"}

    def test_default_effort_when_unset(self) -> None:
        fields = _config(LanguageModelId.CLAUDE_V5_5_SONNET)[
            "additional_model_request_fields"
        ]
        assert fields["output_config"] == {"effort": "medium"}

    def test_temperature_ignored_for_adaptive(self) -> None:
        assert "temperature" not in _config(
            LanguageModelId.CLAUDE_V4_8_OPUS, temperature=0.0
        )


class TestLegacyModels:
    def test_thinking_uses_budget_and_temperature_one(self) -> None:
        config = _config(LanguageModelId.CLAUDE_V4_5_HAIKU, enable_thinking=True)
        assert config["temperature"] == 1.0
        assert config["additional_model_request_fields"]["thinking"]["type"] == (
            "enabled"
        )

    def test_no_thinking_keeps_requested_temperature(self) -> None:
        config = _config(LanguageModelId.CLAUDE_V3_HAIKU, temperature=0.2)
        assert config["temperature"] == 0.2
        assert "additional_model_request_fields" not in config

    def test_thinking_ignored_when_unsupported(self) -> None:
        config = _config(LanguageModelId.CLAUDE_V3_HAIKU, enable_thinking=True)
        assert config["temperature"] == 0.0


class TestLongContextBeta:
    def test_beta_sent_when_requested_and_supported(self) -> None:
        fields = _config(
            LanguageModelId.CLAUDE_V5_5_OPUS, supports_1m_context_window=True
        )["additional_model_request_fields"]
        assert fields["anthropic_beta"] == [
            BedrockLanguageModelFactory.ANTHROPIC_1M_BETA
        ]

    def test_beta_absent_when_not_requested(self) -> None:
        fields = _config(LanguageModelId.CLAUDE_V5_5_OPUS)[
            "additional_model_request_fields"
        ]
        assert "anthropic_beta" not in fields

    def test_beta_absent_for_unsupported_model(self) -> None:
        config = _config(
            LanguageModelId.CLAUDE_V3_HAIKU, supports_1m_context_window=True
        )
        assert "additional_model_request_fields" not in config


class TestCredentials:
    def test_default_profile_omits_credentials(self) -> None:
        assert "credentials_profile_name" not in _config(
            LanguageModelId.CLAUDE_V3_HAIKU
        )

    def test_named_profile_sets_credentials(self) -> None:
        factory = _make_factory("research")
        config = factory._build_model_config(
            _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V3_HAIKU], "id"
        )
        assert config["credentials_profile_name"] == "research"


class TestInferenceProfiles:
    def _session(self, profiles: list[str] | Exception) -> MagicMock:
        client = MagicMock()
        if isinstance(profiles, Exception):
            client.list_inference_profiles.side_effect = profiles
        else:
            client.list_inference_profiles.return_value = {
                "inferenceProfileSummaries": [
                    {"inferenceProfileId": p} for p in profiles
                ]
            }
        session = MagicMock()
        session.client.return_value = client
        return session

    def test_prefers_global_profile(self) -> None:
        model = LanguageModelId.CLAUDE_V5_5_OPUS
        session = self._session([f"us.{model.value}", f"global.{model.value}"])
        assert fac._resolve_inference_profile(session, model, "us-west-2") == (
            f"global.{model.value}"
        )

    def test_regional_then_base(self) -> None:
        model = LanguageModelId.CLAUDE_V5_5_HAIKU
        session = self._session([f"apac.{model.value}"])
        assert (
            fac._resolve_inference_profile(session, model, "ap-northeast-2")
            == f"apac.{model.value}"
        )
        assert (
            fac._resolve_inference_profile(self._session([]), model, "eu-west-1")
            == model.value
        )

    def test_lookup_is_cached(self) -> None:
        model = LanguageModelId.CLAUDE_V5_5_SONNET
        session = self._session([])
        fac._resolve_inference_profile(session, model, "us-west-2")
        fac._resolve_inference_profile(session, model, "us-west-2")
        assert session.client.return_value.list_inference_profiles.call_count == 1

    def test_lookup_failure_fails_closed(self) -> None:
        error = ClientError({"Error": {"Code": "AccessDeniedException"}}, "List")
        with pytest.raises(RuntimeError, match="inference profiles"):
            fac._resolve_inference_profile(
                self._session(error), LanguageModelId.CLAUDE_V5_5_OPUS, "us-west-2"
            )


class TestGetModelInfo:
    def test_known_model_returns_info(self) -> None:
        factory = _make_factory()
        info = factory.get_model_info(LanguageModelId.CLAUDE_V5_5_OPUS)
        assert info is not None and info.max_output_tokens == 128_000
        assert info.uses_adaptive_thinking


class TestBaseBedrockWrapperTruncation:
    def test_no_limit_returns_unchanged(self) -> None:
        assert BaseBedrockWrapper()._truncate_text("abc", None, "doc") == "abc"

    def test_char_truncation(self) -> None:
        assert BaseBedrockWrapper()._truncate_text("abcdefghij", 4, "doc") == "abcd"


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
                # Mirror the real API: a structured ValidationException, not a
                # bare RuntimeError — _within_budget keys off the error CODE.
                raise ClientError(
                    {
                        "Error": {
                            "Code": "ValidationException",
                            "Message": (
                                f"prompt is too long: {tokens} tokens > "
                                f"{self._HARD_LIMIT_TOKENS} maximum"
                            ),
                        }
                    },
                    "CountTokens",
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

    def test_count_tokens_uses_proxy_for_unsupported_model(self) -> None:
        # Regression: Opus 4.8's base id isn't supported by Bedrock CountTokens,
        # so it must count via its same-family proxy (Sonnet 4.6), not its own id.
        captured: dict = {}
        factory = _make_factory()
        factory._client.count_tokens = MagicMock(
            side_effect=lambda **kw: captured.update(kw) or {"inputTokens": 1}
        )
        factory.count_tokens(LanguageModelId.CLAUDE_V4_8_OPUS, "hi")
        assert captured["modelId"] == LanguageModelId.CLAUDE_V4_6_SONNET.value
        assert "opus" not in captured["modelId"]

    def test_within_budget_estimates_when_counting_unsupported(self) -> None:
        # If CountTokens reports the model is unsupported (and there's no proxy),
        # _within_budget falls back to a char estimate instead of re-raising, so
        # the text is still bounded rather than passed through unfitted.
        factory = _make_factory()

        def unsupported(**kw):  # noqa: ANN003, ANN202
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "This model doesn't support counting tokens",
                    }
                },
                "CountTokens",
            )

        factory._client.count_tokens = MagicMock(side_effect=unsupported)
        factory._client.list_inference_profiles = MagicMock(
            return_value={"inferenceProfileSummaries": []}
        )
        # 35 chars / 3.5 = 10 est tokens: within a 20-token budget, over a 5-token one.
        assert factory._within_budget(LanguageModelId.CLAUDE_V3_HAIKU, "x" * 35, 20)
        assert not factory._within_budget(LanguageModelId.CLAUDE_V3_HAIKU, "x" * 35, 5)


class TestGetModel:
    def test_builds_converse_model_for_resolved_profile(self) -> None:
        factory = _make_factory()
        model = LanguageModelId.CLAUDE_V5_5_OPUS
        factory.boto_session.client.return_value.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": [
                {"inferenceProfileId": f"global.{model.value}"}
            ]
        }
        llm = factory.get_model(model, thinking_effort="low")
        assert llm.model_id == f"global.{model.value}"
        assert llm.temperature is None
        assert llm.additional_model_request_fields == {
            "output_config": {"effort": "low"}
        }


class TestTruncationGuard:
    """A response that stopped at max_tokens must raise, never be parsed."""

    def _result(self, stop_reason: str):  # type: ignore[no-untyped-def]
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        message = AIMessage(
            content="partial", response_metadata={"stopReason": stop_reason}
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def test_max_tokens_stop_raises(self) -> None:
        import pytest

        from scholar_lens.src.utils.factories import (
            OutputTruncatedError,
            _raise_if_truncated,
        )

        with pytest.raises(OutputTruncatedError, match="max_tokens=64000"):
            _raise_if_truncated(
                self._result("max_tokens"), "anthropic.claude-opus-5-5", 64000
            )
        _raise_if_truncated(self._result("end_turn"), "m", 10)  # no error

    async def test_model_raises_through_a_chain(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import pytest
        from langchain_aws import ChatBedrockConverse
        from langchain_core.output_parsers import StrOutputParser

        from scholar_lens.src.utils.factories import (
            CheckedChatBedrockConverse,
            OutputTruncatedError,
        )

        monkeypatch.setattr(
            ChatBedrockConverse,
            "_generate",
            lambda self, messages, stop=None, run_manager=None, **kw: self_result,
        )
        self_result = self._result("max_tokens")
        model = CheckedChatBedrockConverse(
            model_id="anthropic.claude-opus-5-5",
            region_name="us-west-2",
            max_tokens=100,
        )
        chain = model | StrOutputParser()
        with pytest.raises(OutputTruncatedError):
            await chain.ainvoke("hi")

    def test_truncation_is_not_retried(self) -> None:
        from scholar_lens.src.utils.factories import OutputTruncatedError
        from scholar_lens.src.utils.retry import is_transient_error

        assert is_transient_error(OutputTruncatedError("x")) is False
