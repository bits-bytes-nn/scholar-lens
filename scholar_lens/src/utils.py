import argparse
import ast
import functools
import json
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, TypeVar, Generic, ClassVar

import asyncio
import boto3
import math
import tenacity
from botocore.config import Config as BotoConfig
from bs4 import BeautifulSoup, NavigableString, Tag
from langchain.output_parsers import OutputFixingParser
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.output_parsers import XMLOutputParser
from langchain_core.runnables import Runnable
from langchain_core.runnables.graph import MermaidDrawMethod
from lxml import etree
from pydantic import BaseModel, Field, PrivateAttr
from tiktoken import Encoding, get_encoding
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

from .constants import EmbeddingModelId, LanguageModelId
from .logger import logger

MAX_RETRIES: int = 5
RETRY_MAX_WAIT: int = 120
RETRY_MULTIPLIER: int = 30


class EmbeddingModelInfo(BaseModel):
    dimensions: int | list[int] | None = Field(
        default=None,
        description="The embedding dimensions. Can be a single value or list of supported dimensions.",
    )
    max_sequence_length: int | None = Field(
        default=None,
        description="Maximum sequence length in characters that the model can process.",
    )
    max_sequence_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens the model can process in a single sequence.",
    )


class LanguageModelInfo(BaseModel):
    context_window_size: int = Field(
        description="Maximum context window size in tokens that the model can handle."
    )
    max_output_tokens: int = Field(
        description="Maximum number of tokens the model can generate in a single response."
    )
    supports_performance_optimization: bool = Field(
        default=False,
        description="Whether the model supports performance optimization features.",
    )
    supports_prompt_caching: bool = Field(
        default=False,
        description="Whether the model supports prompt caching to improve performance.",
    )
    supports_thinking: bool = Field(
        default=False,
        description="Whether the model supports thinking/reasoning capabilities.",
    )
    supports_1m_context_window: bool = Field(
        default=False,
        description="Whether the model supports 1M context window.",
    )


_EMBEDDING_MODEL_INFO: dict[EmbeddingModelId, EmbeddingModelInfo] = {
    EmbeddingModelId.TITAN_EMBED_V1: EmbeddingModelInfo(
        dimensions=1536, max_sequence_length=50000, max_sequence_tokens=8192
    ),
    EmbeddingModelId.TITAN_EMBED_V2: EmbeddingModelInfo(
        dimensions=[256, 512, 1024], max_sequence_length=50000, max_sequence_tokens=8192
    ),
    EmbeddingModelId.EMBED_ENGLISH_V3: EmbeddingModelInfo(
        dimensions=1024, max_sequence_length=2048, max_sequence_tokens=512
    ),
    EmbeddingModelId.EMBED_MULTILINGUAL_V3: EmbeddingModelInfo(
        dimensions=1024, max_sequence_length=2048, max_sequence_tokens=512
    ),
    # NOTE: add new models here
}

_LANGUAGE_MODEL_INFO: dict[LanguageModelId, LanguageModelInfo] = {
    LanguageModelId.CLAUDE_V3_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=4096,
        supports_prompt_caching=True,
    ),
    LanguageModelId.CLAUDE_V3_5_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=8192,
        supports_performance_optimization=True,
        supports_prompt_caching=True,
    ),
    LanguageModelId.CLAUDE_V4_5_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V3_5_SONNET: LanguageModelInfo(
        context_window_size=200000, max_output_tokens=8192
    ),
    LanguageModelId.CLAUDE_V3_5_SONNET_V2: LanguageModelInfo(
        context_window_size=200000, max_output_tokens=8192, supports_prompt_caching=True
    ),
    LanguageModelId.CLAUDE_V3_7_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_5_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_1_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_5_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_6_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=64000,
        supports_thinking=True,
    ),
    # NOTE: add new models here
}


ModelIdT = TypeVar("ModelIdT")
ModelInfoT = TypeVar("ModelInfoT")
WrapperT = TypeVar("WrapperT")


class BaseBedrockWrapper:
    buffer_tokens: int = Field(default=128, ge=0)
    _tokenizer: Encoding = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tokenizer = get_encoding("cl100k_base")

    def _truncate_text(
        self, text: str, max_chars: int | None, max_tokens: int | None, text_type: str
    ) -> str:
        if not max_chars and not max_tokens:
            return text

        token_ids = self._tokenizer.encode(text, allowed_special="all")
        final_text = text
        truncated = False

        if max_tokens and len(token_ids) > max_tokens:
            effective_tokens = max_tokens - self.buffer_tokens
            truncated_token_ids = token_ids[:effective_tokens]
            final_text = self._tokenizer.decode(truncated_token_ids)
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
            connect_timeout=60,
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
            response = bedrock_client.list_inference_profiles(
                maxResults=1000, typeEquals="SYSTEM_DEFINED"
            )
            available_profiles = {
                profile["inferenceProfileId"]
                for profile in response.get("inferenceProfileSummaries", [])
            }
            return cross_region_id in available_profiles
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
        config = self._build_base_config(resolved_model_id, is_cross_region, **kwargs)
        if is_cross_region:
            config.update(
                {"max_tokens": final_max_tokens, "temperature": final_temperature}
            )
        else:
            config["model_kwargs"].update(
                {"max_tokens": final_max_tokens, "temperature": final_temperature}
            )
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
            budget = kwargs.get(
                "thinking_budget_tokens", self.DEFAULT_THINKING_BUDGET_TOKENS
            )
            think_config = {"thinking": {"type": "enabled", "budget_tokens": budget}}
            if is_cross_region:
                config.setdefault("additional_model_request_fields", {}).update(
                    think_config
                )
            else:
                config.setdefault("model_kwargs", {}).update(think_config)
            logger.debug("Applied thinking mode (budget_tokens=%d)", budget)

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


class BatchProcessor(BaseModel):
    max_concurrency: int = Field(default=5, ge=1)
    retry_multiplier: float = Field(default=30.0, ge=1.0)
    retry_max_wait: int = Field(default=120, ge=0)
    max_retries: int = Field(default=5, ge=1)
    batch_size: int = Field(default=10, ge=1)

    def execute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        batch_func: Callable[..., list[Any]],
        sequential_func: Callable[..., Any],
        task_name: str,
        run_config: dict[str, Any] | None = None,
        show_progress: bool = True,
    ) -> list[Any]:
        if not items_to_process:
            return []
        max_concurrency = (
            run_config.get("max_concurrency", self.max_concurrency)
            if run_config
            else self.max_concurrency
        )
        batch_size = (
            run_config.get("batch_size", self.batch_size)
            if run_config
            else self.batch_size
        )
        prepared_batch_func = self._create_batch_func(batch_func, max_concurrency)
        retrying_sequential_func = self._create_retry_decorator(task_name)(
            sequential_func
        )
        all_results = []
        num_items = len(items_to_process)
        num_chunks = math.ceil(num_items / batch_size)
        logger.info(
            "Starting processing for '%s': %d items in %d chunks (batch size: %d)",
            task_name,
            num_items,
            num_chunks,
            batch_size,
        )
        for i in tqdm(
            range(0, num_items, batch_size),
            desc=f"Processing: {task_name}",
            disable=not show_progress,
        ):
            chunk_items = items_to_process[i : i + batch_size]
            chunk_num = (i // batch_size) + 1
            logger.debug(
                "Processing chunk %d/%d (%d items)",
                chunk_num,
                num_chunks,
                len(chunk_items),
            )
            chunk_inputs = prepare_inputs_func(chunk_items)
            if not chunk_inputs:
                logger.warning(
                    "No valid inputs prepared for chunk %d, skipping", chunk_num
                )
                continue
            try:
                logger.debug("Attempting batch processing for chunk %d", chunk_num)
                chunk_results = prepared_batch_func(chunk_inputs)
                all_results.extend(chunk_results)
                logger.debug("Chunk %d processed successfully in batch mode", chunk_num)
            except Exception as e:
                logger.warning(
                    "Batch processing failed for chunk %d: %s. Falling back to sequential processing",
                    chunk_num,
                    e,
                )
                chunk_results = self._process_sequentially_with_fallback(
                    chunk_inputs,
                    retrying_sequential_func,
                    f"{task_name} (chunk {chunk_num})",
                    show_progress=show_progress,
                )
                all_results.extend(chunk_results)
        logger.info("Completed '%s': processed %d results", task_name, len(all_results))
        return all_results

    @staticmethod
    def _create_batch_func(
        batch_func: Callable[..., list[Any]], max_concurrency: int
    ) -> Callable:
        def _batch_func(inputs: list[dict[str, Any]]) -> list[Any]:
            return batch_func(
                inputs, config=RunnableConfig(max_concurrency=max_concurrency)
            )

        return _batch_func

    def _create_retry_decorator(self, operation_name: str) -> Callable:
        return tenacity.retry(
            wait=tenacity.wait_exponential(
                multiplier=self.retry_multiplier, max=self.retry_max_wait
            ),
            stop=tenacity.stop_after_attempt(self.max_retries),
            before_sleep=self._create_retry_log_callback(operation_name),
            reraise=True,
        )

    @staticmethod
    def _create_retry_log_callback(operation_name: str) -> Callable:
        def log_retry(retry_state):
            wait_time = retry_state.next_action.sleep if retry_state.next_action else 0
            logger.warning(
                "Retrying '%s' (attempt %d failed). Waiting %.1fs",
                operation_name,
                retry_state.attempt_number,
                wait_time,
            )

        return log_retry

    @staticmethod
    def _process_sequentially_with_fallback(
        inputs: list[dict[str, Any]],
        sequential_func: Callable[[dict[str, Any]], Any],
        task_name: str,
        show_progress: bool = True,
    ) -> list[Any]:
        logger.info("Processing %d items sequentially for '%s'", len(inputs), task_name)
        results = []
        progress_desc = f"Sequential Processing: '{task_name}'"
        successful_count = 0
        for single_input in tqdm(inputs, desc=progress_desc, disable=not show_progress):
            try:
                result = sequential_func(single_input)
                results.append(result)
                successful_count += 1
            except Exception as e:
                logger.error(
                    "Sequential processing failed for single item in '%s': %s",
                    task_name,
                    e,
                )
                continue
        logger.info(
            "Sequential processing completed for '%s': %d/%d items processed successfully",
            task_name,
            successful_count,
            len(inputs),
        )
        return results

    async def aexecute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        batch_func: Callable[..., Any],
        sequential_func: Callable[..., Any],
        task_name: str,
        run_config: dict[str, Any] | None = None,
        show_progress: bool = True,
    ) -> list[Any]:
        if not items_to_process:
            return []
        max_concurrency = (
            run_config.get("max_concurrency", self.max_concurrency)
            if run_config
            else self.max_concurrency
        )
        batch_size = (
            run_config.get("batch_size", self.batch_size)
            if run_config
            else self.batch_size
        )
        prepared_batch_func = self._create_async_batch_func(batch_func, max_concurrency)
        retrying_sequential_func = self._create_retry_decorator(task_name)(
            sequential_func
        )
        all_results = []
        num_items = len(items_to_process)
        num_chunks = math.ceil(num_items / batch_size)
        logger.info(
            "Starting async processing for '%s': %d items in %d chunks (batch size: %d)",
            task_name,
            num_items,
            num_chunks,
            batch_size,
        )
        chunk_iterator = async_tqdm(
            range(0, num_items, batch_size),
            desc=f"Processing: {task_name}",
            disable=not show_progress,
        )
        for i in chunk_iterator:
            chunk_items = items_to_process[i : i + batch_size]
            chunk_num = (i // batch_size) + 1
            logger.debug(
                "Processing chunk %d/%d (%d items)",
                chunk_num,
                num_chunks,
                len(chunk_items),
            )
            chunk_inputs = prepare_inputs_func(chunk_items)
            if not chunk_inputs:
                logger.warning(
                    "No valid inputs prepared for chunk %d, skipping", chunk_num
                )
                continue
            try:
                chunk_results = await prepared_batch_func(chunk_inputs)
                all_results.extend(chunk_results)
            except Exception as e:
                logger.warning(
                    "Async batch processing failed for chunk %d: %s. Falling back to concurrent sequential processing",
                    chunk_num,
                    e,
                )
                chunk_results = await self._aprocess_sequentially_with_fallback(
                    chunk_inputs,
                    retrying_sequential_func,
                    f"{task_name} (chunk {chunk_num})",
                    max_concurrency,
                    show_progress,
                )
                all_results.extend(chunk_results)
        logger.info("Completed '%s': processed %d results", task_name, len(all_results))
        return all_results

    @staticmethod
    def _create_async_batch_func(
        batch_func: Callable[..., Any], max_concurrency: int
    ) -> Callable:
        async def _batch_func(inputs: list[dict[str, Any]]) -> list[Any]:
            return await batch_func(
                inputs, config=RunnableConfig(max_concurrency=max_concurrency)
            )

        return _batch_func

    @staticmethod
    async def _aprocess_sequentially_with_fallback(
        inputs: list[dict[str, Any]],
        sequential_func: Callable[[dict[str, Any]], Any],
        task_name: str,
        max_concurrency: int,
        show_progress: bool = True,
    ) -> list[Any]:
        logger.info("Processing %d items concurrently for '%s'", len(inputs), task_name)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _process_one(single_input):
            async with semaphore:
                try:
                    return await sequential_func(single_input)
                except Exception as e:
                    logger.error(
                        "Concurrent sequential processing failed for item in '%s': %s",
                        task_name,
                        e,
                    )
                    return None

        tasks = [_process_one(single_input) for single_input in inputs]
        progress_desc = f"Concurrent Fallback: '{task_name}'"
        results = await async_tqdm.gather(
            *tasks, disable=not show_progress, desc=progress_desc
        )
        successful_results = [res for res in results if res is not None]
        logger.info(
            "Concurrent sequential processing completed for '%s': %d/%d items processed successfully",
            task_name,
            len(successful_results),
            len(inputs),
        )
        return successful_results


class HTMLTagOutputParser(BaseOutputParser):
    tag_names: str | list[str]

    def parse(self, text: str) -> str | dict[str, str]:
        if not text:
            return {} if isinstance(self.tag_names, list) else ""
        soup = BeautifulSoup(text, "html.parser")
        parsed: dict[str, str] = {}
        tag_list = (
            self.tag_names if isinstance(self.tag_names, list) else [self.tag_names]
        )
        for tag_name in tag_list:
            if tag := soup.find(tag_name):
                if hasattr(tag, "decode_contents"):
                    parsed[tag_name] = str(tag.decode_contents()).strip()
                else:
                    parsed[tag_name] = str(tag).strip()
        if isinstance(self.tag_names, list):
            return parsed
        return next(iter(parsed.values()), "")

    @property
    def _type(self) -> str:
        return "html_tag_output_parser"


class RetryableBase:
    @staticmethod
    def _retry(operation_name: str) -> Callable:
        return tenacity.retry(
            wait=tenacity.wait_exponential(
                multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_WAIT
            ),
            stop=tenacity.stop_after_attempt(MAX_RETRIES),
            before_sleep=lambda retry_state: logger.warning(
                "Retrying '%s' (attempt %d failed). Waiting %.1fs",
                operation_name,
                retry_state.attempt_number,
                retry_state.next_action.sleep if retry_state.next_action else 0,
            ),
            reraise=True,
        )


class RobustXMLOutputParser(XMLOutputParser):
    def parse(self, text: str) -> dict[str, Any]:
        text = self._unescape_xml_tags(text)
        original_sections = self._detect_xml_sections(text)

        try:
            result = super().parse(text)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in parsed result")
        except Exception as e:
            logger.debug(
                f"Standard XML parsing failed: {type(e).__name__}: {e}. Trying lxml recovery..."
            )

        try:
            cleaned_text = self._clean_xml_for_lxml(text)
            result = self._try_lxml_recover_parse(cleaned_text)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in lxml result")
        except Exception as e:
            logger.debug(
                f"LXML recovery parsing failed: {type(e).__name__}: {e}. Trying sanitization..."
            )

        try:
            sanitized_text = self._sanitize_xml_content(text)
            result = super().parse(sanitized_text)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in sanitized result")
        except Exception as e:
            logger.debug(
                f"Sanitized XML parsing failed: {type(e).__name__}: {e}. Trying truncated XML fix..."
            )

        try:
            fixed_text = self._fix_truncated_xml(text)
            result = super().parse(fixed_text)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in truncated fix result")
        except Exception as e:
            logger.debug(
                f"Truncated XML fix failed: {type(e).__name__}: {e}. Trying aggressive cleaning..."
            )

        try:
            aggressively_cleaned = self._aggressively_clean_xml(text)
            result = super().parse(aggressively_cleaned)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in aggressive result")
        except Exception as e:
            logger.debug(
                f"Aggressive cleaning parsing failed: {type(e).__name__}: {e}. Trying XML fallback..."
            )

        try:
            fallback_result = self._extract_xml_fallback(text)
            if fallback_result:
                return fallback_result
        except Exception as e:
            logger.debug(
                f"XML fallback extraction failed: {type(e).__name__}: {e}. Trying tags fallback..."
            )

        try:
            fallback_result = self._extract_tags_fallback(text)
            if fallback_result:
                return fallback_result
        except Exception as e:
            logger.debug(
                f"Tags fallback extraction failed: {type(e).__name__}: {e}. Trying list fallback..."
            )

        try:
            fallback_result = self._extract_list_fallback(text)
            if fallback_result:
                return fallback_result
        except Exception as e:
            logger.debug(
                f"List fallback extraction failed: {type(e).__name__}: {e}. All methods exhausted."
            )

        logger.error("All XML parsing attempts failed for content: '%s...'", text[:200])
        raise ValueError(
            f"Failed to parse XML after multiple attempts. Content preview: '{text[:200]}...'"
        )

    @staticmethod
    def _detect_xml_sections(text: str) -> set[str]:
        pattern = r"<([a-zA-Z0-9_]+)>.*?</\1>"
        matches = re.findall(pattern, text, re.DOTALL)
        return set(matches)

    @staticmethod
    def _sections_preserved(
        original_sections: set[str], parsed_result: dict[str, Any]
    ) -> bool:
        if not original_sections:
            return True

        parsed_sections = (
            set(parsed_result.keys()) if isinstance(parsed_result, dict) else set()
        )
        missing_sections = original_sections - parsed_sections

        if missing_sections:
            return False
        return True

    @staticmethod
    def _extract_xml_fallback(text: str) -> dict[str, Any] | None:
        result = {}

        try:
            section_pattern = r"<([a-zA-Z0-9_]+)>(.*?)</\1>"
            section_matches = re.findall(section_pattern, text, re.DOTALL)

            for section_name, section_content in section_matches:
                section_result = RobustXMLOutputParser._parse_xml_section(
                    section_content
                )
                if section_result is not None:
                    result[section_name] = section_result

            return result if result else None

        except Exception:
            return None

    @staticmethod
    def _parse_xml_section(
        content: str,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        content = content.strip()
        if not content:
            return None

        child_pattern = r"<([a-zA-Z0-9_]+)>(.*?)</\1>"
        child_matches = re.findall(child_pattern, content, re.DOTALL)

        if not child_matches:
            return {"#text": content}

        children_by_tag = defaultdict(list)
        for child_tag, child_content in child_matches:
            parsed_child = RobustXMLOutputParser._parse_xml_element(child_content)
            children_by_tag[child_tag].append(parsed_child)

        result = {}
        for tag, children in children_by_tag.items():
            result[tag] = children[0] if len(children) == 1 else children

        if len(children_by_tag) == 1:
            child_tag = list(children_by_tag.keys())[0]
            children = children_by_tag[child_tag]
            if len(children) > 1:
                return {child_tag: children}

        return result

    @staticmethod
    def _parse_xml_element(content: str) -> dict[str, Any] | str:
        content = content.strip()
        if not content:
            return ""

        nested_pattern = r"<([a-zA-Z0-9_]+)>(.*?)</\1>"
        nested_matches = re.findall(nested_pattern, content, re.DOTALL)

        if not nested_matches:
            return content

        result: dict[str, Any] = {}
        for nested_tag, nested_content in nested_matches:
            parsed_nested = RobustXMLOutputParser._parse_xml_element(nested_content)

            if nested_tag in result:
                if not isinstance(result[nested_tag], list):
                    result[nested_tag] = [result[nested_tag]]
                result[nested_tag].append(parsed_nested)
            else:
                result[nested_tag] = parsed_nested

        text_content = content
        for nested_tag, nested_content in nested_matches:
            full_nested = f"<{nested_tag}>{nested_content}</{nested_tag}>"
            text_content = text_content.replace(full_nested, "").strip()

        if text_content and result:
            result["#text"] = text_content
        elif text_content and not result:
            return text_content

        return result

    @staticmethod
    def _clean_xml_for_lxml(text: str) -> bytes:
        def is_valid_xml_char(char: str) -> bool:
            cp = ord(char)
            return (
                cp == 0x9
                or cp == 0xA
                or cp == 0xD
                or (0x20 <= cp <= 0xD7FF)
                or (0xE000 <= cp <= 0xFFFD)
                or (0x10000 <= cp <= 0x10FFFF)
            )

        cleaned = "".join(char for char in text if is_valid_xml_char(char))
        return cleaned.strip().encode("utf-8")

    @staticmethod
    def _try_lxml_recover_parse(xml_bytes: bytes) -> dict[str, Any]:
        parser = etree.XMLParser(recover=True, encoding="utf-8")
        tree = etree.fromstring(xml_bytes, parser=parser)

        if tree is None:
            raise ValueError("lxml parser recovered a null tree")

        def _convert_etree_to_dict(element: etree._Element) -> dict[str, Any]:
            result: dict[str, Any] = {}
            children = list(element)

            if children:
                child_dict = defaultdict(list)
                for child in children:
                    child_result = _convert_etree_to_dict(child)
                    for key, value in child_result.items():
                        child_dict[key].append(value)

                processed_children = {
                    key: val[0] if len(val) == 1 else val
                    for key, val in child_dict.items()
                }
                result[element.tag] = processed_children
            else:
                result[element.tag] = {}

            if element.attrib:
                if not isinstance(result[element.tag], dict):
                    result[element.tag] = {"#text": result[element.tag]}
                if isinstance(result[element.tag], dict):
                    result[element.tag].update(
                        {f"@{k}": v for k, v in element.attrib.items()}
                    )

            if element.text and element.text.strip():
                text = element.text.strip()
                if not result[element.tag]:
                    result[element.tag] = text
                elif isinstance(result[element.tag], dict):
                    if "#text" not in result[element.tag]:
                        result[element.tag]["#text"] = text

            if not result[element.tag]:
                result[element.tag] = {}

            return result

        return _convert_etree_to_dict(tree)

    @staticmethod
    def _sanitize_xml_content(xml_content: str) -> str:
        def escape_text_only(text: str) -> str:
            placeholders = {}
            entity_pattern = r"&(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);"
            counter = [0]

            def save_entity(match: re.Match) -> str:
                key = f"\x00ENTITY{counter[0]}\x00"
                placeholders[key] = match.group(0)
                counter[0] += 1
                return key

            text = re.sub(entity_pattern, save_entity, text)

            text = text.replace("&", "&amp;")
            text = text.replace("<", "&lt;")
            text = text.replace(">", "&gt;")

            for key, value in placeholders.items():
                text = text.replace(key, value)

            return text

        def escape_leaf_content(match: re.Match) -> str:
            tag_open = match.group(1)
            content = match.group(2)
            tag_close = match.group(3)

            if not re.search(r"<[a-zA-Z_]", content):
                escaped_content = escape_text_only(content)
                return f"{tag_open}{escaped_content}{tag_close}"
            return match.group(0)

        pattern = r"(<([a-zA-Z_][a-zA-Z0-9_]*)\s*[^>]*>)(.*?)(</\2>)"
        prev_content = ""
        while prev_content != xml_content:
            prev_content = xml_content
            xml_content = re.sub(
                pattern, escape_leaf_content, xml_content, flags=re.DOTALL
            )

        return xml_content

    @staticmethod
    def _aggressively_clean_xml(xml_content: str) -> str:
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", xml_content)

        def escape_text_between_tags(match: re.Match[str]) -> str:
            content = match.group(1)
            content = re.sub(
                r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)",
                "&amp;",
                content,
            )
            return f">{content}<"

        cleaned = re.sub(r">([^<]*)<", escape_text_between_tags, cleaned)
        return cleaned.strip()

    @staticmethod
    def _unescape_xml_tags(text: str) -> str:
        text = re.sub(
            r"&lt;([a-zA-Z_][a-zA-Z0-9_]*(?:\s+[^&]*?)?)&gt;",
            r"<\1>",
            text,
        )
        text = re.sub(
            r"&lt;/([a-zA-Z_][a-zA-Z0-9_]*)&gt;",
            r"</\1>",
            text,
        )
        return text

    @staticmethod
    def _fix_truncated_xml(text: str) -> str:
        opening_tags = re.findall(r"<([a-zA-Z_][a-zA-Z0-9_]*)\b[^/>]*>", text)
        closing_tags = re.findall(r"</([a-zA-Z_][a-zA-Z0-9_]*)>", text)

        tag_stack = []
        for tag in opening_tags:
            tag_stack.append(tag)
        for tag in closing_tags:
            if tag_stack and tag_stack[-1] == tag:
                tag_stack.pop()
            elif tag in tag_stack:
                idx = len(tag_stack) - 1 - tag_stack[::-1].index(tag)
                tag_stack.pop(idx)

        text = re.sub(r"<[a-zA-Z_][a-zA-Z0-9_]*\s+[^>]*$", "", text)

        for tag in reversed(tag_stack):
            text += f"</{tag}>"

        return text

    @staticmethod
    def _extract_tags_fallback(text: str) -> dict[str, Any] | None:
        pattern = re.compile(r"<([a-zA-Z0-9_]+)\s*.*?>(.*?)</\1>", re.DOTALL)
        matches = pattern.findall(text)

        if not matches:
            return None

        content_map = defaultdict(list)
        for tag, content in matches:
            stripped_content = content.strip()
            if stripped_content:
                content_map[tag].append(stripped_content)

        if not content_map:
            return None

        result = {
            key: val[0] if len(val) == 1 else val for key, val in content_map.items()
        }

        return result

    @staticmethod
    def _extract_list_fallback(text: str) -> dict[str, Any] | None:
        item_patterns = [
            r"^\s*[•\-\*]\s*(.+?)(?=\n\s*[•\-\*]|\Z)",
            r"^\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\Z)",
        ]

        for pattern in item_patterns:
            items = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            if items:
                stripped_items = [item.strip() for item in items if item.strip()]
                if stripped_items:
                    return {"items": stripped_items}

        return None


def arg_as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value = value.lower().strip()
        if value in ("yes", "true", "t", "y", "1"):
            return True
        if value in ("no", "false", "f", "n", "0"):
            return False

    raise argparse.ArgumentTypeError("Boolean value expected")


def arg_as_list(value: str | None) -> list[str] | None:
    if value is None:
        return None

    try:
        urls = json.loads(value)
        if isinstance(urls, str):
            return [urls.strip()]
        if isinstance(urls, list):
            return urls
    except json.JSONDecodeError:
        if value.strip().startswith("[") and value.strip().endswith("]"):
            try:
                urls = ast.literal_eval(value)
                if isinstance(urls, list):
                    return urls
            except (SyntaxError, ValueError):
                pass
        return [value.strip()]

    return None


def create_robust_xml_output_parser(
    factory: BedrockLanguageModelFactory,
    enable_output_fixing: bool,
    output_fixing_model_id: LanguageModelId,
) -> BaseOutputParser:
    base_parser = RobustXMLOutputParser()
    if not enable_output_fixing:
        return base_parser

    try:
        fixing_llm = factory.get_model(model_id=output_fixing_model_id)
        logger.info(
            f"Created OutputFixingParser with model: '{output_fixing_model_id.value}'"
        )
        return OutputFixingParser.from_llm(parser=base_parser, llm=fixing_llm)
    except Exception as e:
        logger.error(
            f"Failed to create OutputFixingParser with model {output_fixing_model_id.value}: {e}"
        )
        raise RuntimeError(f"Failed to create OutputFixingParser: {e}") from e


def extract_text_from_html(html_content: str) -> str:
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    for tag_name in ["head", "meta", "script", "style", "title"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    def parse_element(element) -> str:
        if isinstance(element, NavigableString):
            return element.strip()

        if not isinstance(element, Tag):
            return ""

        if element.name == "img":
            alt = element.get("alt", "")
            src = element.get("src", "")
            return f"[Image: alt={alt}, src={src}]"

        if element.name == "a":
            href = element.get("href", "")
            link_text = "".join(parse_element(child) for child in element.children)
            return f"{link_text} ({href})" if href else link_text

        if element.name in ["table", "thead", "tbody", "tr", "td", "th"]:
            content = "".join(parse_element(child) for child in element.children)
            return content

        if element.name in ["code", "pre"]:
            code_content = "".join(parse_element(child) for child in element.children)
            return f"`{code_content}`"

        if element.name == "math":
            math_content = "".join(parse_element(child) for child in element.children)
            return f"$$ {math_content} $$"

        return " ".join(parse_element(child) for child in element.children)

    extracted_text = parse_element(soup)

    replacements = {"\\AND": "", "\\n": " ", "\\times": "x", "footnotemark:": ""}
    for old, new in replacements.items():
        extracted_text = extracted_text.replace(old, new)

    extracted_text = re.sub(r"\s+", " ", extracted_text).strip()

    return extracted_text


def measure_execution_time(func: Callable) -> Callable:
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(
            "'%s' execution time: %.2fs (%.2fmin)",
            func.__name__,
            execution_time,
            execution_time / 60,
        )
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(
            "'%s' execution time: %.2fs (%.2fmin)",
            func.__name__,
            execution_time,
            execution_time / 60,
        )
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def plot_langchain_graph(app: Runnable, output_file_path: Path) -> None:
    try:
        app.get_graph().draw_mermaid_png(
            output_file_path=str(output_file_path),
            draw_method=MermaidDrawMethod.API,
        )
    except Exception as e:
        logger.error(f"Error plotting Langchain graph: {str(e)}")
