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
        description="Context window in tokens as invoked (native window, or 1M "
        "when supports_1m_context_window enables the long-context beta)."
    )
    max_output_tokens: int = Field(
        description="Maximum number of tokens the model can generate in a single response."
    )
    input_price_per_mtok: float = Field(
        description="List price in USD per million input tokens (first-party "
        "rate; Bedrock pricing can differ, so costs derived from it are estimates)."
    )
    output_price_per_mtok: float = Field(
        description="List price in USD per million output tokens."
    )
    long_prompt_threshold: int | None = Field(
        default=None,
        description="Prompt size (input tokens) above which a call is billed at "
        "the long-prompt rates instead (tiered pricing); None = flat pricing.",
    )
    long_prompt_price_per_mtok: tuple[float, float] | None = Field(
        default=None,
        description="(input, output) USD per million tokens for long prompts.",
    )
    supports_prompt_caching: bool = Field(default=False)
    supports_thinking: bool = Field(default=False)
    uses_adaptive_thinking: bool = Field(
        default=False,
        description=(
            "Uses thinking.type='adaptive' + output_config.effort instead of "
            "budget_tokens, and rejects sampling parameters (temperature/top_k)."
        ),
    )
    thinking_always_on: bool = Field(
        default=False,
        description=(
            "Thinking cannot be disabled (thinking.type='disabled' is rejected); "
            "effort is the only cost/depth control."
        ),
    )
    supports_1m_context_window: bool = Field(
        default=False,
        description="Needs the context-1m beta flag to reach a 1M window.",
    )
    count_tokens_proxy: str | None = Field(
        default=None,
        description=(
            "Model id used for Bedrock CountTokens when this model is not itself "
            "supported by it. Its tokenizer can differ (Opus 4.7+ and the 5.x "
            "models emit up to ~1.35x as many tokens), so budgets computed from "
            "proxy counts carry a safety factor (see fit_text)."
        ),
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

_SONNET_4_6 = LanguageModelId.CLAUDE_V4_6_SONNET.value


def _legacy(
    ctx: int,
    out: int,
    price: tuple[float, float],
    *,
    supports_prompt_caching: bool = False,
    supports_thinking: bool = False,
    supports_1m_context_window: bool = False,
) -> LanguageModelInfo:
    """Pre-4.7 models: budget_tokens thinking and sampling parameters."""
    return LanguageModelInfo(
        context_window_size=ctx,
        max_output_tokens=out,
        input_price_per_mtok=price[0],
        output_price_per_mtok=price[1],
        supports_prompt_caching=supports_prompt_caching,
        supports_thinking=supports_thinking,
        supports_1m_context_window=supports_1m_context_window,
    )


def _adaptive(
    price: tuple[float, float], *, always_on: bool, out: int = 128_000
) -> LanguageModelInfo:
    """Claude 4.7+ generation: 1M context (via the beta flag, as live-validated
    on Bedrock), adaptive thinking."""
    return LanguageModelInfo(
        context_window_size=200_000,
        supports_1m_context_window=True,
        max_output_tokens=out,
        input_price_per_mtok=price[0],
        output_price_per_mtok=price[1],
        supports_prompt_caching=True,
        supports_thinking=True,
        uses_adaptive_thinking=True,
        thinking_always_on=always_on,
        count_tokens_proxy=_SONNET_4_6,
    )


_HAIKU_3 = (0.25, 1.25)
_SONNET_3 = (3.0, 15.0)
_OPUS_LEGACY = (15.0, 75.0)
_OPUS = (5.0, 25.0)

_LANGUAGE_MODEL_INFO: dict[LanguageModelId, LanguageModelInfo] = {
    LanguageModelId.CLAUDE_V3_HAIKU: _legacy(
        200_000, 4096, _HAIKU_3, supports_prompt_caching=True
    ),
    LanguageModelId.CLAUDE_V3_SONNET: _legacy(200_000, 4096, _SONNET_3),
    LanguageModelId.CLAUDE_V3_OPUS: _legacy(200_000, 4096, _OPUS_LEGACY),
    LanguageModelId.CLAUDE_V3_5_HAIKU: _legacy(
        200_000, 8192, (0.8, 4.0), supports_prompt_caching=True
    ),
    LanguageModelId.CLAUDE_V4_5_HAIKU: _legacy(
        200_000,
        64_000,
        (1.0, 5.0),
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    # Tiered: prompts over 100K input tokens bill at $0.50/$2.50 per MTok.
    LanguageModelId.CLAUDE_V5_5_HAIKU: _adaptive(
        (0.10, 0.50), always_on=True
    ).model_copy(
        update={
            "long_prompt_threshold": 100_000,
            "long_prompt_price_per_mtok": (0.50, 2.50),
        }
    ),
    LanguageModelId.CLAUDE_V3_5_SONNET: _legacy(200_000, 8192, _SONNET_3),
    LanguageModelId.CLAUDE_V3_5_SONNET_V2: _legacy(
        200_000, 8192, _SONNET_3, supports_prompt_caching=True
    ),
    LanguageModelId.CLAUDE_V3_7_SONNET: _legacy(
        200_000, 64_000, _SONNET_3, supports_prompt_caching=True, supports_thinking=True
    ),
    LanguageModelId.CLAUDE_V4_SONNET: _legacy(
        200_000,
        64_000,
        _SONNET_3,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_5_SONNET: _legacy(
        200_000,
        64_000,
        _SONNET_3,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    # The 4.6 models still accept budget_tokens and sampling params, so they
    # keep the legacy request shape.
    LanguageModelId.CLAUDE_V4_6_SONNET: _legacy(
        200_000,
        64_000,
        _SONNET_3,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V5_SONNET: _adaptive((2.0, 10.0), always_on=False),
    LanguageModelId.CLAUDE_V5_5_SONNET: _adaptive((2.0, 10.0), always_on=True),
    LanguageModelId.CLAUDE_V4_OPUS: _legacy(
        200_000,
        32_000,
        _OPUS_LEGACY,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_1_OPUS: _legacy(
        200_000,
        32_000,
        _OPUS_LEGACY,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_5_OPUS: _legacy(
        200_000, 64_000, _OPUS, supports_prompt_caching=True, supports_thinking=True
    ),
    LanguageModelId.CLAUDE_V4_6_OPUS: LanguageModelInfo(
        context_window_size=200_000,
        max_output_tokens=64_000,
        input_price_per_mtok=_OPUS[0],
        output_price_per_mtok=_OPUS[1],
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
        count_tokens_proxy=_SONNET_4_6,
    ),
    LanguageModelId.CLAUDE_V4_7_OPUS: _adaptive(_OPUS, always_on=False),
    LanguageModelId.CLAUDE_V4_8_OPUS: _adaptive(_OPUS, always_on=False),
    LanguageModelId.CLAUDE_V5_5_OPUS: _adaptive((4.0, 20.0), always_on=True),
    LanguageModelId.CLAUDE_V5_1_FABLE: _adaptive((10.0, 50.0), always_on=True),
    # NOTE: add new models here
}


def model_info_for_invoked_id(invoked_id: str) -> LanguageModelInfo | None:
    """Registry entry for an id as Bedrock or LangChain reports it: with or
    without an inference-profile prefix (``global.``/``us.``) or ``anthropic.``."""

    def bare(value: str) -> str:
        return value.rsplit("anthropic.", 1)[-1]

    target = bare(invoked_id)
    for model_id, info in _LANGUAGE_MODEL_INFO.items():
        if bare(model_id.value) == target:
            return info
    return None
