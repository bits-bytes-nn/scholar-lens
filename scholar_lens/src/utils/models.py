from pydantic import BaseModel, Field

from ..constants import EmbeddingModelId, LanguageModelId


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
    uses_adaptive_thinking: bool = Field(
        default=False,
        description=(
            "Whether the model uses the newer adaptive thinking API "
            "(thinking.type='adaptive' + output_config.effort) instead of the "
            "legacy thinking.type='enabled' + budget_tokens. Opus 4.8+ requires "
            "the adaptive form."
        ),
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
    LanguageModelId.CLAUDE_V3_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=4096,
    ),
    LanguageModelId.CLAUDE_V3_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=4096,
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
    LanguageModelId.CLAUDE_V4_6_SONNET: LanguageModelInfo(
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
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_8_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        uses_adaptive_thinking=True,
        supports_1m_context_window=True,
    ),
    # NOTE: add new models here
}
