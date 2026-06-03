import re
import sys
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field, field_validator

sys.path.append(str(Path(__file__).parent.parent))

from scholar_lens.src.constants import EmbeddingModelId, LanguageModelId

# Effort level for adaptive-thinking models (e.g. Opus 4.8). Lower effort means
# shorter reasoning traces — faster and cheaper. Ignored by legacy thinking
# models, which use a fixed token budget instead.
ThinkingEffort = Literal["low", "medium", "high"]


class Github(BaseModel):
    enabled: bool = Field(default=False)
    repo_name: str | None = Field(default=None)
    base_branch: str = Field(default="main")
    branch_prefix: str = Field(default="paper-reviews")
    author_name: str = Field(default="Scholar Lens Bot")
    author_email: EmailStr | None = Field(default=None)
    cover_images: dict[str, str] = Field(
        default_factory=dict,
        description="Maps a category slug (lowercase, hyphenated) to a cover "
        "image filename under the blog's assets/images directory.",
    )
    default_cover_image: str = Field(default="default.jpg")

    @field_validator("cover_images", mode="before")
    @classmethod
    def _default_empty_mapping(cls, v: dict[str, str] | None) -> dict[str, str]:
        return v or {}

    def cover_image_for(self, category: str) -> str:
        """Resolve the cover image for a category, falling back to the default.

        The category is normalised to a slug (lowercase, runs of non-alphanumeric
        characters collapsed to a single hyphen) so that ``"Multimodal Learning"``
        and ``"multimodal-learning"`` resolve identically.
        """
        slug = re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-")
        return self.cover_images.get(slug, self.default_cover_image)


class Resources(BaseModel):
    project_name: str = Field(min_length=1)
    stage: Literal["dev", "prod"] = Field(default="dev")
    profile_name: str | None = Field(default=None)
    default_region_name: str = Field(default="ap-northeast-2")
    bedrock_region_name: str = Field(default="us-west-2")
    s3_bucket_name: str | None = Field(default=None)
    s3_prefix: str = Field(default="scholar-lens")
    vpc_id: str | None = Field(default=None)
    subnet_ids: list[str] | None = Field(default=None)
    email_address: EmailStr | None = Field(default=None)
    github: Github = Field(default_factory=Github)


class Paper(BaseModel):
    figure_analysis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V3_HAIKU
    )
    citation_extraction_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET
    )
    attributes_extraction_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU
    )
    table_of_contents_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET
    )
    output_fixing_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET
    )


class Code(BaseModel):
    code_analysis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU
    )
    code_summarization_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU
    )
    embed_model_id: EmbeddingModelId = Field(default=EmbeddingModelId.TITAN_EMBED_V2)
    chunk_size: int = Field(default=1024)
    chunk_overlap: int = Field(default=256)


class Citations(BaseModel):
    citation_summarization_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU
    )
    citation_analysis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU
    )


class Explanation(BaseModel):
    paper_analysis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET
    )
    paper_enrichment_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET
    )
    paper_finalization_model_id: LanguageModelId
    paper_reflection_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET
    )
    paper_synthesis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET
    )
    reflector_enable_thinking: bool = Field(default=False)
    synthesizer_enable_thinking: bool = Field(default=False)
    thinking_effort: ThinkingEffort = Field(default="medium")


class Summary(BaseModel):
    summary_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V4_8_OPUS)
    summarizer_enable_thinking: bool = Field(default=False)
    thinking_effort: ThinkingEffort = Field(default="medium")


class TechGuide(BaseModel):
    relevance_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU
    )
    synopsis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_6_SONNET
    )
    writing_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V4_8_OPUS)
    writer_enable_thinking: bool = Field(default=False)
    thinking_effort: ThinkingEffort = Field(default="medium")


class Config(BaseModel):
    resources: Resources = Field(
        default_factory=lambda: Resources(project_name="scholar-lens")
    )
    paper: Paper = Field(
        default_factory=lambda: Paper(
            figure_analysis_model_id=LanguageModelId.CLAUDE_V3_HAIKU,
            citation_extraction_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
            attributes_extraction_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
            table_of_contents_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
            output_fixing_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
        )
    )
    code: Code = Field(
        default_factory=lambda: Code(
            code_analysis_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
            code_summarization_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        )
    )
    citations: Citations = Field(
        default_factory=lambda: Citations(
            citation_summarization_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
            citation_analysis_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        )
    )
    explanation: Explanation = Field(
        default_factory=lambda: Explanation(
            paper_analysis_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
            paper_enrichment_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
            paper_finalization_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
            paper_reflection_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
            paper_synthesis_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
        )
    )
    summary: Summary = Field(default_factory=Summary)
    tech_guide: TechGuide = Field(default_factory=TechGuide)
    output_language: str = Field(default="Korean")

    @classmethod
    def from_yaml(cls, file_path: str) -> "Config":
        with open(file_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data if config_data else {})

    @classmethod
    def load(cls) -> "Config":
        load_dotenv()
        config_path = Path(__file__).parent / "config.yaml"
        if not config_path.exists():
            return cls()
        return cls.from_yaml(str(config_path))
