from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, ClassVar, Generic, TypeVar

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import ChatBedrockConverse
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field

from ..constants import EmbeddingModelId, LanguageModelId
from ..logger import logger
from .models import (
    _EMBEDDING_MODEL_INFO,
    _LANGUAGE_MODEL_INFO,
    EmbeddingModelInfo,
    LanguageModelInfo,
)

ModelIdT = TypeVar("ModelIdT")
ModelInfoT = TypeVar("ModelInfoT")
WrapperT = TypeVar("WrapperT")


class BaseBedrockWrapper:
    def _truncate_text(self, text: str, max_chars: int | None, text_type: str) -> str:
        """Cap ``text`` to ``max_chars`` for an embedding model.

        Embedding models are not supported by Bedrock CountTokens, so inputs are
        bounded by the model's character limit (chunks are ~1k chars, far below
        any token limit)."""
        if max_chars and len(text) > max_chars:
            logger.warning(
                "%s character count (%d) exceeds maximum (%d). Truncating.",
                text_type.capitalize(),
                len(text),
                max_chars,
            )
            return text[:max_chars]
        return text


class BaseBedrockModelFactory(Generic[ModelIdT, ModelInfoT, WrapperT], ABC):
    # A non-streaming call returns nothing until generation ends, so the read
    # timeout must cover the longest response asked for (128k output tokens,
    # thinking included, at a few dozen tokens per second), not a socket gap.
    BOTO_READ_TIMEOUT: ClassVar[int] = 3600
    BOTO_CONNECT_TIMEOUT: ClassVar[int] = 30
    # One attempt: RetryableBase (utils.retry) is the single retry layer, so
    # botocore must not multiply it.
    BOTO_MAX_ATTEMPTS: ClassVar[int] = 1
    MAX_POOL_CONNECTIONS: ClassVar[int] = 50

    def __init__(
        self,
        boto_session: boto3.Session | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        self.boto_session = boto_session or boto3.Session(profile_name=profile_name)
        self.region_name = region_name or self.boto_session.region_name
        boto_config = BotoConfig(
            read_timeout=self.BOTO_READ_TIMEOUT,
            connect_timeout=self.BOTO_CONNECT_TIMEOUT,
            retries={"total_max_attempts": self.BOTO_MAX_ATTEMPTS, "mode": "standard"},
            max_pool_connections=self.MAX_POOL_CONNECTIONS,
        )
        self._client = self.boto_session.client(
            self._get_boto_service_name(),
            region_name=self.region_name,
            config=boto_config,
        )
        logger.debug(
            "Initialized %s for region: '%s'", self.__class__.__name__, self.region_name
        )

    @abstractmethod
    def _get_boto_service_name(self) -> str: ...

    @abstractmethod
    def _get_model_info_dict(self) -> dict[ModelIdT, ModelInfoT]: ...

    @abstractmethod
    def get_model(self, model_id: ModelIdT, **kwargs: Any) -> WrapperT: ...

    def get_model_info(self, model_id: ModelIdT) -> ModelInfoT | None:
        return self._get_model_info_dict().get(model_id)


class BedrockCrossRegionModelHelper:
    """Resolves the inference profile (``global.`` or regional prefix) to call
    a model through; the result is cached per (session, region, model)."""

    @staticmethod
    def get_cross_region_model_id(
        boto_session: boto3.Session,
        model_id: LanguageModelId,
        region_name: str,
    ) -> str:
        return _resolve_inference_profile(boto_session, model_id, region_name)

    @staticmethod
    def _build_cross_region_model_id(
        model_id: LanguageModelId, region_name: str, is_global: bool = False
    ) -> str:
        if is_global:
            return f"global.{model_id.value}"
        prefix = "apac" if region_name.startswith("ap-") else region_name[:2]
        return f"{prefix}.{model_id.value}"

    @staticmethod
    def _list_system_profiles(bedrock_client: Any) -> set[str]:
        profiles: set[str] = set()
        next_token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"typeEquals": "SYSTEM_DEFINED"}
            if next_token:
                kwargs["nextToken"] = next_token
            response = bedrock_client.list_inference_profiles(**kwargs)
            profiles.update(
                p["inferenceProfileId"]
                for p in response.get("inferenceProfileSummaries", [])
            )
            next_token = response.get("nextToken")
            if not next_token:
                return profiles


@lru_cache(maxsize=64)
def _resolve_inference_profile(
    boto_session: boto3.Session, model_id: LanguageModelId, region_name: str
) -> str:
    """Prefer the global profile, then the regional one, then the base id.

    A failed lookup raises: silently falling back to the base id would surface
    later as an opaque ValidationException on every call.
    """
    helper = BedrockCrossRegionModelHelper
    try:
        profiles = helper._list_system_profiles(
            boto_session.client("bedrock", region_name=region_name)
        )
    except ClientError as e:
        raise RuntimeError(
            f"Could not list Bedrock inference profiles in '{region_name}': {e}"
        ) from e
    for is_global in (True, False):
        candidate = helper._build_cross_region_model_id(
            model_id, region_name, is_global
        )
        if candidate in profiles:
            logger.debug("Using inference profile '%s'", candidate)
            return candidate
    logger.debug("No inference profile for '%s'; using the base id.", model_id.value)
    return model_id.value


class BedrockEmbeddingsWrapper(BaseBedrockWrapper, BedrockEmbeddings):
    max_sequence_length: int | None = Field(default=None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        truncated_texts = [
            self._truncate_text(text, self.max_sequence_length, "document")
            for text in texts
        ]
        return super().embed_documents(truncated_texts)

    def embed_query(self, text: str) -> list[float]:
        truncated_text = self._truncate_text(text, self.max_sequence_length, "query")
        return super().embed_query(truncated_text)

    def _embedding_func(self, text: str) -> list[float]:
        # langchain_community wraps every endpoint error in a bare ValueError,
        # which the retry policy treats as deterministic; surface the botocore
        # error instead so throttling and timeouts are classified (and retried)
        # like every other Bedrock call. It is re-raised outside the handler so
        # the ValueError is not chained back onto it as its context.
        try:
            return super()._embedding_func(text)
        except ValueError as e:
            original = e.__cause__ or e.__context__
            if not isinstance(original, (ClientError, BotoCoreError)):
                raise
        raise original


class BedrockEmbeddingModelFactory(
    BaseBedrockModelFactory[
        EmbeddingModelId, EmbeddingModelInfo, BedrockEmbeddingsWrapper
    ]
):
    def _get_boto_service_name(self) -> str:
        return "bedrock-runtime"

    def _get_model_info_dict(self) -> dict[EmbeddingModelId, EmbeddingModelInfo]:
        return _EMBEDDING_MODEL_INFO

    def get_max_sequence_length(self, model_id: EmbeddingModelId) -> int | None:
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Unsupported embedding model ID: '{model_id.value}'")
        return model_info.max_sequence_length

    def get_model(
        self, model_id: EmbeddingModelId, **kwargs: Any
    ) -> BedrockEmbeddingsWrapper:
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Unsupported embedding model ID: '{model_id.value}'")

        model = BedrockEmbeddingsWrapper(
            client=self._client,
            model_id=model_id.value,
            max_sequence_length=model_info.max_sequence_length,
            **kwargs,
        )
        logger.debug("Created embedding model: '%s'", model_id.value)
        return model


class OutputTruncatedError(RuntimeError):
    """The model stopped at ``max_tokens``: its output is incomplete, so it must
    never be parsed as if it were whole. Not retried (the same request would
    stop at the same place); raise the call's ``max_tokens`` instead."""


def _raise_if_truncated(
    result: ChatResult, model_id: str, max_tokens: int | None
) -> None:
    for generation in result.generations:
        metadata = getattr(generation.message, "response_metadata", {}) or {}
        reason = metadata.get("stopReason") or metadata.get("stop_reason")
        if reason == "max_tokens":
            raise OutputTruncatedError(
                f"'{model_id}' hit max_tokens={max_tokens}; the output is incomplete."
            )


class CheckedChatBedrockConverse(ChatBedrockConverse):
    """``ChatBedrockConverse`` that fails loudly on a truncated response instead
    of handing partial text to the output parsers. (The async path runs
    ``_generate`` in an executor, so both are covered.)"""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = super()._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        _raise_if_truncated(result, self.model_id, self.max_tokens)
        return result


class BedrockLanguageModelFactory(
    BaseBedrockModelFactory[
        LanguageModelId, LanguageModelInfo, CheckedChatBedrockConverse
    ]
):
    """Builds Converse chat models with the request shape each model accepts.

    ``get_model`` options: ``enable_thinking``, ``thinking_effort``,
    ``max_tokens`` (default :attr:`DEFAULT_MAX_TOKENS`), ``temperature``
    (legacy models only), ``supports_1m_context_window``, ``callbacks``.
    """

    DEFAULT_TEMPERATURE: ClassVar[float] = 0.0
    DEFAULT_THINKING_BUDGET_TOKENS: ClassVar[int] = 2048
    DEFAULT_THINKING_EFFORT: ClassVar[str] = "medium"
    # Per-call output ceiling (capped at the model's own maximum). Thinking
    # counts against it on the always-thinking 5.x models, so it is generous;
    # asking for the full 128k would reserve that much tokens-per-minute quota
    # up front. A response that still hits it raises OutputTruncatedError.
    DEFAULT_MAX_TOKENS: ClassVar[int] = 64_000
    ANTHROPIC_1M_BETA: ClassVar[str] = "context-1m-2025-08-07"
    LARGE_CONTEXT_WINDOW: ClassVar[int] = 1_000_000
    # Tokens held back from the context window for the prompt template wrapping
    # the fitted text plus the model's own response.
    DEFAULT_CONTEXT_RESERVE_TOKENS: ClassVar[int] = 16_000
    PROXY_COUNT_SAFETY_FACTOR: ClassVar[float] = 1.35

    def _get_boto_service_name(self) -> str:
        return "bedrock-runtime"

    def _get_model_info_dict(self) -> dict[LanguageModelId, LanguageModelInfo]:
        return _LANGUAGE_MODEL_INFO

    def get_model(
        self, model_id: LanguageModelId, **kwargs: Any
    ) -> CheckedChatBedrockConverse:
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Unsupported language model ID: '{model_id.value}'")
        resolved_model_id = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            self.boto_session, model_id, self.region_name or ""
        )
        config = self._build_model_config(model_info, resolved_model_id, **kwargs)
        logger.debug("Created language model '%s'", resolved_model_id)
        return CheckedChatBedrockConverse(**config)

    def effective_context_window(self, model_id: LanguageModelId) -> int:
        """Usable context window in tokens (1M-beta models are always invoked
        with the beta; unknown models fall back to a conservative 200k)."""
        info = self.get_model_info(model_id)
        if not info:
            return 200_000
        if info.supports_1m_context_window:
            return max(info.context_window_size, self.LARGE_CONTEXT_WINDOW)
        return info.context_window_size

    def count_tokens(self, model_id: LanguageModelId, text: str) -> int:
        """Exact input-token count for ``text`` via Bedrock CountTokens (the
        model's own tokenizer).

        Uses the BASE model id: CountTokens rejects cross-region inference-profile
        ids (``apac.``/``global.`` → ValidationException), unlike Converse. Some
        newer models (e.g. Opus 4.6+) aren't supported by CountTokens at all; for
        those a same-family ``count_tokens_proxy`` is used so the count stays
        accurate. For 1M-capable models it sends the long-context beta flag so
        inputs above the base 200k limit can still be measured. Raises on API
        error (callers that must not fail catch it)."""
        converse: dict[str, Any] = {
            "messages": [{"role": "user", "content": [{"text": text}]}]
        }
        info = self.get_model_info(model_id)
        if info and info.supports_1m_context_window:
            converse["additionalModelRequestFields"] = {
                "anthropic_beta": [self.ANTHROPIC_1M_BETA]
            }
        # Count with the proxy model when the target isn't CountTokens-capable.
        count_model_id = (info.count_tokens_proxy if info else None) or model_id.value
        response = self._client.count_tokens(
            modelId=count_model_id, input={"converse": converse}
        )
        return int(response["inputTokens"])

    # Conservative chars-per-token used only as a last-resort estimate when a
    # model is not supported by CountTokens and has no proxy (English/Korean
    # technical text runs ~3.5-4 chars/token; 3.5 errs toward over-counting).
    _FALLBACK_CHARS_PER_TOKEN: float = 3.5

    def _within_budget(self, model_id: LanguageModelId, text: str, budget: int) -> bool:
        """Whether ``text`` fits ``budget`` tokens. CountTokens itself rejects
        inputs beyond the model's measurable limit with a ValidationException
        ("prompt is too long") — that unambiguously means over budget, so it is
        treated as a False (shrink) signal rather than an error. If the model is
        simply not supported by CountTokens (and has no proxy), fall back to a
        deterministic char estimate so the text is still bounded rather than
        passed through unfitted."""
        try:
            return self.count_tokens(model_id, text) <= budget
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            message = e.response.get("Error", {}).get("Message", "").lower()
            if code == "ValidationException" and "too long" in message:
                return False
            if code == "ValidationException" and (
                "support" in message and "counting tokens" in message
            ):
                est_tokens = len(text) / self._FALLBACK_CHARS_PER_TOKEN
                return est_tokens <= budget
            raise

    def fit_text(
        self,
        model_id: LanguageModelId,
        text: str,
        *,
        reserve_tokens: int | None = None,
        label: str = "input",
    ) -> str:
        """Truncate ``text`` so it fits ``model_id``'s context window, measured
        exactly with Bedrock CountTokens (no chars-per-token estimation).

        Returns ``text`` unchanged when it already fits; otherwise binary-searches
        the longest prefix whose exact token count is within budget. A genuine
        "prompt too long" from the counter shrinks the search; any other API error
        leaves the text intact so a transient issue can't silently gut content
        (the caller's retry/raise path then handles a real overflow)."""
        if not text:
            return text
        reserve = (
            reserve_tokens
            if reserve_tokens is not None
            else self.DEFAULT_CONTEXT_RESERVE_TOKENS
        )
        budget = max(self.effective_context_window(model_id) - reserve, 1_000)
        info = self.get_model_info(model_id)
        if info and info.count_tokens_proxy:
            # Counted with another model's tokenizer; newer tokenizers (Opus 4.7+
            # and the 5.x family) emit up to ~1.35x as many tokens for the same
            # text, so shrink the budget instead of risking an overflow.
            budget = int(budget / self.PROXY_COUNT_SAFETY_FACTOR)
        try:
            if self._within_budget(model_id, text, budget):
                return text
            lo, hi, best = 0, len(text), 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if mid > 0 and self._within_budget(model_id, text[:mid], budget):
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
        except (
            Exception
        ) as e:  # noqa: BLE001 - never fail the pipeline on a counter error
            logger.warning(
                "CountTokens failed while fitting %s for '%s' (%s); leaving text intact.",
                label,
                model_id.value,
                e,
            )
            return text
        logger.info(
            "Fitted %s for '%s': %d -> %d chars to stay within %d tokens.",
            label,
            model_id.value,
            len(text),
            best,
            budget,
        )
        return text[:best]

    def _build_model_config(
        self, model_info: LanguageModelInfo, resolved_model_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "model_id": resolved_model_id,
            "region_name": self.region_name,
            "client": self._client,
            "callbacks": kwargs.get("callbacks"),
            "max_tokens": min(
                kwargs.get("max_tokens") or self.DEFAULT_MAX_TOKENS,
                model_info.max_output_tokens,
            ),
        }
        if (
            self.boto_session.profile_name
            and self.boto_session.profile_name != "default"
        ):
            config["credentials_profile_name"] = self.boto_session.profile_name
        request_fields: dict[str, Any] = {}
        thinking = bool(kwargs.get("enable_thinking")) and model_info.supports_thinking
        if model_info.uses_adaptive_thinking:
            # Effort is always sent: it is the cost/depth control whether or not
            # thinking is requested, and models whose thinking is always on
            # would otherwise run at their own (higher) default. Sampling
            # parameters and thinking.type='disabled' are rejected, so neither
            # is ever sent.
            effort = kwargs.get("thinking_effort") or self.DEFAULT_THINKING_EFFORT
            request_fields["output_config"] = {"effort": effort}
            if thinking:
                request_fields["thinking"] = {"type": "adaptive"}
        else:
            # Legacy models: thinking needs temperature 1.0 and a token budget.
            config["temperature"] = (
                1.0 if thinking else kwargs.get("temperature", self.DEFAULT_TEMPERATURE)
            )
            if thinking:
                request_fields["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.DEFAULT_THINKING_BUDGET_TOKENS,
                }
        if (
            kwargs.get("supports_1m_context_window")
            and model_info.supports_1m_context_window
        ):
            request_fields["anthropic_beta"] = [self.ANTHROPIC_1M_BETA]
        if request_fields:
            config["additional_model_request_fields"] = request_fields
        return config
