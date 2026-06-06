from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

import boto3
from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_community.embeddings import BedrockEmbeddings
from pydantic import Field, PrivateAttr
from tiktoken import Encoding, get_encoding

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
    buffer_tokens: int = Field(default=128, ge=0)
    _tokenizer: Encoding | None = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # The tokenizer is loaded lazily: get_encoding("cl100k_base") downloads
        # the BPE vocab from the network on a cold cache, so doing it at
        # construction would make every wrapper instantiation hit the network
        # (breaking offline/test/Batch environments). Load on first real use.
        self._tokenizer = None

    @property
    def tokenizer(self) -> Encoding:
        if self._tokenizer is None:
            self._tokenizer = get_encoding("cl100k_base")
        return self._tokenizer

    def _truncate_text(
        self, text: str, max_chars: int | None, max_tokens: int | None, text_type: str
    ) -> str:
        if not max_chars and not max_tokens:
            return text

        final_text = text
        truncated = False

        # Only tokenize when a token limit is actually requested — a char-only
        # truncation must not pull in the tokenizer (and its network download).
        if max_tokens:
            token_ids = self.tokenizer.encode(text, allowed_special="all")
            if len(token_ids) > max_tokens:
                effective_tokens = max_tokens - self.buffer_tokens
                truncated_token_ids = token_ids[:effective_tokens]
                final_text = self.tokenizer.decode(truncated_token_ids)
                logger.warning(
                    f"{text_type.capitalize()} token count ({len(token_ids)}) exceeds maximum ({max_tokens}). Truncating."
                )
                truncated = True

        if max_chars and len(text) > max_chars:
            if not truncated or len(text[:max_chars]) < len(final_text):
                final_text = text[:max_chars]
                logger.warning(
                    f"{text_type.capitalize()} character count ({len(text)}) exceeds maximum ({max_chars}). Truncating."
                )

        return final_text


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
        use_converse = is_cross_region or (
            enable_thinking and model_info.supports_thinking
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
            if is_cross_region:
                config.setdefault("additional_model_request_fields", {}).update(
                    {"anthropic_beta": ["context-1m-2025-08-07"]}
                )
            else:
                config["model_kwargs"].setdefault(
                    "additionalModelRequestFields", {}
                ).update({"anthropic_beta": ["context-1m-2025-08-07"]})
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
