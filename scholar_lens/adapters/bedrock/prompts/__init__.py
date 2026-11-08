"""LangChain prompt templates of the Bedrock chains (extraction, survey and
guide calls). Importing the package registers every template with
:func:`~scholar_lens.adapters.bedrock.prompt_safety.prompt_fence_tags`."""

from scholar_lens.adapters.bedrock.prompts import (
    base,
    extraction,
    survey,
    tech_guide,
)

__all__ = ["base", "extraction", "survey", "tech_guide"]
