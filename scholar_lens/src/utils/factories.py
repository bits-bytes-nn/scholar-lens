from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

import boto3
from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_community.embeddings import BedrockEmbeddings
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
    def _truncate_text(
        self, text: str, max_chars: int | None, max_tokens: int | None, text_type: str
    ) -> str:
        """Cap ``text`` to ``max_chars`` for an embedding model.

        Embedding models (Titan/Cohere) are NOT supported by Bedrock CountTokens,
        and the project policy is to use CountTokens for token counting rather
        than a foreign tokenizer estimate. So embedding inputs are bounded only
        by the model's character limit; ``max_tokens`` is accepted for interface
        compatibility but intentionally not enforced here (embedding chunks are
        ~1k chars, far below any model's token limit, so the char cap suffices)."""
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
    BOTO_READ_TIMEOUT: ClassVar[int] = 300
    BOTO_CONNECT_TIMEOUT: ClassVar[int] = 60
    BOTO_MAX_ATTEMPTS: ClassVar[int] = 3
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
            retries={"max_attempts": self.BOTO_MAX_ATTEMPTS, "mode": "adaptive"},
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

    def get_supported_models(self) -> list[ModelIdT]:
        return list(self._get_model_info_dict().keys())


class BedrockCrossRegionModelHelper:
    @staticmethod
    def get_cross_region_model_id(
        boto_session: boto3.Session,
        model_id: LanguageModelId,
        region_name: str,
    ) -> str:
        try:
            bedrock_client = boto_session.client("bedrock", region_name=region_name)
            global_model_id = (
                BedrockCrossRegionModelHelper._build_cross_region_model_id(
                    model_id, region_name, is_global=True
                )
            )
            if BedrockCrossRegionModelHelper._is_cross_region_model_available(
                bedrock_client, global_model_id
            ):
                logger.debug("Using global cross-region model: '%s'", global_model_id)
                return global_model_id
            regional_model_id = (
                BedrockCrossRegionModelHelper._build_cross_region_model_id(
                    model_id, region_name, is_global=False
                )
            )
            if BedrockCrossRegionModelHelper._is_cross_region_model_available(
                bedrock_client, regional_model_id
            ):
                logger.debug(
                    "Using regional cross-region model: '%s'", regional_model_id
                )
                return regional_model_id
            logger.debug(
                "Cross-region models not available, using standard model: '%s'",
                model_id.value,
            )
            return model_id.value
        except Exception as e:
            logger.warning(
                "Failed to resolve cross-region model for '%s': %s. Falling back to standard model.",
                model_id.value,
                e,
            )
            return model_id.value

    @staticmethod
    def _build_cross_region_model_id(
        model_id: LanguageModelId, region_name: str, is_global: bool = False
    ) -> str:
        if is_global:
            return f"global.{model_id.value}"
        prefix = "apac" if region_name.startswith("ap-") else region_name[:2]
        return f"{prefix}.{model_id.value}"

    @staticmethod
    def _is_cross_region_model_available(
        bedrock_client: Any, cross_region_id: str
    ) -> bool:
        try:
            # Paginate: a single maxResults page can silently truncate the
            # catalog (and miss the profile we're checking) as AWS adds models.
            next_token: str | None = None
            while True:
                kwargs: dict[str, Any] = {"typeEquals": "SYSTEM_DEFINED"}
                if next_token:
                    kwargs["nextToken"] = next_token
                response = bedrock_client.list_inference_profiles(**kwargs)
                for profile in response.get("inferenceProfileSummaries", []):
                    if profile["inferenceProfileId"] == cross_region_id:
                        return True
                next_token = response.get("nextToken")
                if not next_token:
                    return False
        except Exception as e:
            raise RuntimeError(
                f"Failed to check cross-region model availability: {e}"
            ) from e


class BedrockEmbeddingsWrapper(BaseBedrockWrapper, BedrockEmbeddings):
    max_sequence_length: int | None = Field(default=None)
    max_sequence_tokens: int | None = Field(default=None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        truncated_texts = [
            self._truncate_text(
                text, self.max_sequence_length, self.max_sequence_tokens, "document"
            )
            for text in texts
        ]
        return super().embed_documents(truncated_texts)

    def embed_query(self, text: str) -> list[float]:
        truncated_text = self._truncate_text(
            text, self.max_sequence_length, self.max_sequence_tokens, "query"
        )
        return super().embed_query(truncated_text)


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

        model_kwargs = {}
        dimensions = kwargs.pop("dimensions", None)

        if dimensions:
            supported_dims = model_info.dimensions
            is_supported = False
            if isinstance(supported_dims, list):
                is_supported = dimensions in supported_dims
            elif isinstance(supported_dims, int):
                is_supported = dimensions == supported_dims

            if not is_supported:
                raise ValueError(
                    f"Dimension {dimensions} is not supported by model '{model_id.value}'. "
                    f"Supported dimensions: {supported_dims}"
                )
            if isinstance(supported_dims, list):
                model_kwargs["dimensions"] = dimensions

        model = BedrockEmbeddingsWrapper(
            client=self._client,
            model_id=model_id.value,
            model_kwargs=model_kwargs,
            max_sequence_length=model_info.max_sequence_length,
            max_sequence_tokens=model_info.max_sequence_tokens,
            **kwargs,
        )
        logger.debug(f"Created embedding model: '{model_id.value}'")
        return model


class BedrockLanguageModelFactory(
    BaseBedrockModelFactory[
        LanguageModelId, LanguageModelInfo, ChatBedrock | ChatBedrockConverse
    ]
):
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.0
    DEFAULT_TOP_K: ClassVar[int] = 50
    DEFAULT_THINKING_BUDGET_TOKENS: ClassVar[int] = 2048
    DEFAULT_THINKING_EFFORT: ClassVar[str] = "medium"
    DEFAULT_LATENCY_MODE: ClassVar[str] = "normal"
    ANTHROPIC_1M_BETA: ClassVar[str] = "context-1m-2025-08-07"
    LARGE_CONTEXT_WINDOW: ClassVar[int] = 1_000_000
    # Tokens held back from the context window for the prompt template wrapping
    # the fitted text plus the model's own response.
    DEFAULT_CONTEXT_RESERVE_TOKENS: ClassVar[int] = 16_000

    def _get_boto_service_name(self) -> str:
        return "bedrock-runtime"

    def _get_model_info_dict(self) -> dict[LanguageModelId, LanguageModelInfo]:
        return _LANGUAGE_MODEL_INFO

    def get_model(
        self, model_id: LanguageModelId, **kwargs: Any
    ) -> ChatBedrock | ChatBedrockConverse:
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Unsupported language model ID: '{model_id.value}'")
        resolved_model_id = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            self.boto_session, model_id, self.region_name or ""
        )
        is_cross_region = resolved_model_id != model_id.value
        enable_thinking = kwargs.get("enable_thinking", False)
        want_1m = kwargs.get("supports_1m_context_window", False)
        # The 1M-context beta is passed via additionalModelRequestFields, which is
        # a Converse-only field — on the legacy ChatBedrock (InvokeModel) path it
        # is silently dropped, so the window would NOT actually be enabled and
        # fit_text (trusting a 1M budget) would overflow the real 200k limit.
        # Force Converse whenever the 1M window is requested and supported.
        use_converse = (
            is_cross_region
            or (enable_thinking and model_info.supports_thinking)
            or (want_1m and model_info.supports_1m_context_window)
        )
        model_config = self._build_model_config(
            model_info, resolved_model_id, use_converse, **kwargs
        )
        model_class = ChatBedrockConverse if use_converse else ChatBedrock
        model = model_class(**model_config)
        logger.debug(
            "Created language model: '%s' with class %s",
            resolved_model_id,
            model_class.__name__,
        )
        return model

    def effective_context_window(self, model_id: LanguageModelId) -> int:
        """The usable context window for ``model_id`` in tokens.

        Models flagged for the 1M window are always invoked with it enabled, so
        their effective window is the large one; otherwise it is the registered
        window. Unknown models fall back to a conservative 200k."""
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
        ids (``apac.``/``global.`` → ValidationException), unlike Converse. For
        1M-capable models it sends the long-context beta flag so inputs above the
        base 200k limit can still be measured. Raises on API error (callers that
        must not fail catch it)."""
        converse: dict[str, Any] = {
            "messages": [{"role": "user", "content": [{"text": text}]}]
        }
        info = self.get_model_info(model_id)
        if info and info.supports_1m_context_window:
            converse["additionalModelRequestFields"] = {
                "anthropic_beta": [self.ANTHROPIC_1M_BETA]
            }
        response = self._client.count_tokens(
            modelId=model_id.value, input={"converse": converse}
        )
        return int(response["inputTokens"])

    def _within_budget(self, model_id: LanguageModelId, text: str, budget: int) -> bool:
        """Whether ``text`` fits ``budget`` tokens. CountTokens itself rejects
        inputs beyond the model's measurable limit with a "too long"
        ValidationException — that unambiguously means over budget, so it is
        treated as a False (shrink) signal rather than an error."""
        try:
            return self.count_tokens(model_id, text) <= budget
        except Exception as e:  # noqa: BLE001
            if "too long" in str(e).lower():
                return False
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
        self,
        model_info: LanguageModelInfo,
        resolved_model_id: str,
        is_cross_region: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        enable_thinking = kwargs.get("enable_thinking", False)
        supports_1m_context_window = kwargs.get("supports_1m_context_window", False)
        temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)
        final_temperature = (
            1.0
            if self._should_enable_thinking(enable_thinking, model_info)
            else temperature
        )
        if final_temperature != temperature:
            logger.debug("Adjusting temperature to 1.0 for thinking mode")
        final_max_tokens = self._validate_max_tokens(
            kwargs.get("max_tokens"), model_info
        )
        # Newer models (Opus 4.8+) deprecate the temperature parameter entirely;
        # only send max_tokens for them.
        token_config: dict[str, Any] = {"max_tokens": final_max_tokens}
        if not model_info.uses_adaptive_thinking:
            token_config["temperature"] = final_temperature
        config = self._build_base_config(resolved_model_id, is_cross_region, **kwargs)
        if is_cross_region:
            config.update(token_config)
        else:
            config["model_kwargs"].update(token_config)
        if supports_1m_context_window and model_info.supports_1m_context_window:
            # get_model forces the Converse path when the 1M window is requested,
            # so is_cross_region is True here and the beta goes through Converse's
            # additional_model_request_fields (the InvokeModel branch below can't
            # actually deliver it).
            if is_cross_region:
                config.setdefault("additional_model_request_fields", {}).update(
                    {"anthropic_beta": [self.ANTHROPIC_1M_BETA]}
                )
            else:
                config["model_kwargs"].setdefault(
                    "additionalModelRequestFields", {}
                ).update({"anthropic_beta": [self.ANTHROPIC_1M_BETA]})
            logger.debug("Applied 1M context window support")
        self._apply_model_features(config, model_info, is_cross_region, **kwargs)
        return config

    def _build_base_config(
        self, resolved_model_id: str, is_cross_region: bool, **kwargs: Any
    ) -> dict[str, Any]:
        config = {
            "model_id": resolved_model_id,
            "region_name": self.region_name,
            "client": self._client,
            "callbacks": kwargs.get("callbacks"),
        }
        if (
            self.boto_session.profile_name
            and self.boto_session.profile_name != "default"
        ):
            config["credentials_profile_name"] = self.boto_session.profile_name
        common_params = {
            "stop_sequences": ["\n\nHuman:"],
        }
        if is_cross_region:
            config.update(common_params)
        else:
            config["model_kwargs"] = {
                "top_k": kwargs.get("top_k", self.DEFAULT_TOP_K),
                **common_params,
            }
        return config

    def _apply_model_features(
        self,
        config: dict[str, Any],
        model_info: LanguageModelInfo,
        is_cross_region: bool,
        **kwargs: Any,
    ) -> None:
        enable_perf = kwargs.get("enable_performance_optimization", False)
        enable_think = kwargs.get("enable_thinking", False)
        if self._should_enable_performance_optimization(
            enable_perf, model_info, is_cross_region
        ):
            latency = kwargs.get("latency_mode", self.DEFAULT_LATENCY_MODE)
            config.setdefault("performanceConfig", {}).update({"latency": latency})
            logger.debug(
                "Applied performance optimization (latency_mode='%s')", latency
            )
        if self._should_enable_thinking(enable_think, model_info):
            if model_info.uses_adaptive_thinking:
                # Opus 4.8+ API: thinking.type='adaptive' + output_config.effort.
                effort = kwargs.get("thinking_effort", self.DEFAULT_THINKING_EFFORT)
                think_config = {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": effort},
                }
                detail = f"effort='{effort}'"
            else:
                # Legacy API: thinking.type='enabled' + budget_tokens.
                budget = kwargs.get(
                    "thinking_budget_tokens", self.DEFAULT_THINKING_BUDGET_TOKENS
                )
                think_config = {
                    "thinking": {"type": "enabled", "budget_tokens": budget}
                }
                detail = f"budget_tokens={budget}"
            if is_cross_region:
                config.setdefault("additional_model_request_fields", {}).update(
                    think_config
                )
            else:
                config.setdefault("model_kwargs", {}).update(think_config)
            logger.debug("Applied thinking mode (%s)", detail)

    @staticmethod
    def _validate_max_tokens(
        max_tokens: int | None, model_info: LanguageModelInfo
    ) -> int:
        final_max_tokens = max_tokens or model_info.max_output_tokens
        if final_max_tokens > model_info.max_output_tokens:
            logger.warning(
                "Requested max_tokens (%d) exceeds model's maximum (%d). Adjusting.",
                final_max_tokens,
                model_info.max_output_tokens,
            )
            return model_info.max_output_tokens
        return final_max_tokens

    @staticmethod
    def _should_enable_performance_optimization(
        enable: bool, model_info: LanguageModelInfo, is_cross_region: bool
    ) -> bool:
        return (
            enable
            and model_info.supports_performance_optimization
            and not is_cross_region
        )

    @staticmethod
    def _should_enable_thinking(enable: bool, model_info: LanguageModelInfo) -> bool:
        return enable and model_info.supports_thinking
