import sys
import yaml
from pathlib import Path
from typing import Literal


from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field, FilePath

sys.path.append(str(Path(__file__).parent.parent))

from src.constants import EmbeddingModelId, LanguageModelId


class Github(BaseModel):
    enabled: bool = Field(default=False)
    repository: str | None = Field(default=None)
    base_branch: str = Field(default="main")
    author_name: str = Field(default="Scholar Lens Bot")
    author_email: EmailStr | None = Field(default=None)


class Resources(BaseModel):
    project_name: str = Field(min_length=1)
    stage: Literal["dev", "prod"] = Field(default="dev")
    profile_name: str | None = Field(default=None)
    default_region_name: str = Field(default="ap-northeast-2")
    bedrock_region_name: str = Field(default="us-west-2")
    s3_bucket_name: str | None = Field(default=None)
    s3_prefix: str | None = Field(default=None)
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
        default=LanguageModelId.CLAUDE_V3_5_HAIKU
    )
    table_of_contents_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET
    )
    output_fixing_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_SONNET
    )


class Code(BaseModel):
    code_analysis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V3_HAIKU
    )
    code_summarization_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V3_HAIKU
    )
    embed_model_id: EmbeddingModelId | None = Field(default=None)
    chunk_size: int = Field(default=1024)
    chunk_overlap: int = Field(default=256)


class Citations(BaseModel):
    citation_summarization_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V3_HAIKU
    )
    citation_analysis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V3_5_HAIKU
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


class Config(BaseModel):
    resources: Resources = Field(
        default_factory=lambda: Resources(project_name="scholar-lens")
    )
    paper: Paper = Field(
        default_factory=lambda: Paper(
            figure_analysis_model_id=LanguageModelId.CLAUDE_V3_HAIKU,
            citation_extraction_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
            attributes_extraction_model_id=LanguageModelId.CLAUDE_V3_5_HAIKU,
            table_of_contents_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
            output_fixing_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
        )
    )
    code: Code = Field(
        default_factory=lambda: Code(
            code_analysis_model_id=LanguageModelId.CLAUDE_V3_HAIKU,
            code_summarization_model_id=LanguageModelId.CLAUDE_V3_HAIKU,
        )
    )
    citations: Citations = Field(
        default_factory=lambda: Citations(
            citation_summarization_model_id=LanguageModelId.CLAUDE_V3_HAIKU,
            citation_analysis_model_id=LanguageModelId.CLAUDE_V3_5_HAIKU,
        )
    )
    explanation: Explanation = Field(
        default_factory=lambda: Explanation(
            paper_analysis_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
            paper_enrichment_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
            paper_finalization_model_id=LanguageModelId.CLAUDE_V3_5_HAIKU,
            paper_reflection_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
            paper_synthesis_model_id=LanguageModelId.CLAUDE_V4_5_SONNET,
        )
    )

    @classmethod
    def from_yaml(cls, file_path: FilePath) -> "Config":
        with open(file_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data if config_data else {})

    @classmethod
    def load(cls) -> "Config":
        load_dotenv()
        config_path = Path(__file__).parent / "config.yaml"
        if not config_path.exists():
            return cls()
        return cls.from_yaml(config_path)
