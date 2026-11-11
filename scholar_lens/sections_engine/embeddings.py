"""The sections engine's Bedrock embeddings (its code index).

``langchain_aws.BedrockEmbeddings`` re-raises the endpoint's botocore error
as it is, so the shared retry policy (``adapters/retry.py``) classifies
throttling, timeouts and validation errors like every other Bedrock call.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_aws import BedrockEmbeddings
from pydantic import Field

from scholar_lens.adapters.bedrock.chat_models import (
    BaseBedrockModelFactory,
    BaseBedrockWrapper,
)
from scholar_lens.domain.constants import EmbeddingModelId
from scholar_lens.domain.model_registry import _EMBEDDING_MODEL_INFO, EmbeddingModelInfo

logger = logging.getLogger(__name__)


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
