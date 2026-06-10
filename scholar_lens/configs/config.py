import re
import sys
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

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
    # Primary (lead) category label written to each artifact's front matter, by
    # artifact type. These are the values a blog's category tabs filter on, so
    # they are configurable to match the target blog without code changes.
    # Defaults are generic; override per-blog (e.g. to align with a Jekyll site's
    # category pages).
    review_category: str = Field(default="Paper Reviews")
    summary_category: str = Field(default="Paper Summaries")
    tech_guide_category: str = Field(default="Tech Guides")
    # Whether figures uploaded to S3 are world-readable. Default False (private):
    # blog images are served from the GitHub Pages repo, so public S3 ACLs are
    # an unnecessary exposure (and are rejected by buckets with Block Public
    # Access). Set True only if you intentionally serve assets straight from S3.
    public_assets: bool = Field(default=False)

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
    chunk_size: int = Field(default=1024, gt=0, le=1_000_000)
    chunk_overlap: int = Field(default=256, ge=0)

    @model_validator(mode="after")
    def _check_overlap_lt_size(self) -> "Code":
        # overlap >= size makes RecursiveCharacterTextSplitter loop/misbehave.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})."
            )
        return self


class Citations(BaseModel):
    citation_summarization_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU
    )
    citation_analysis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V4_5_HAIKU
    )
    # When True, download each cited paper's full text (slow, arXiv-heavy). When
    # False (default), summarise from the abstract resolved via Crossref/Semantic
    # Scholar/arXiv metadata — far fewer calls and no arXiv rate-limit storms.
    prefer_full_text: bool = Field(default=False)


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
    # Hard total-token ceiling for one review run (None = no limit). Guards
    # against runaway cost on the per-paragraph synthesis loop. Must be positive
    # when set — a non-positive ceiling would trip the budget guard immediately.
    max_total_tokens: int | None = Field(default=None, gt=0)


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
    # Fact-check each drafted section against the sources to remove ungrounded
    # claims (hallucinated APIs/flags). Adds one LLM call per section.
    verify_grounding: bool = Field(default=True)
    # Deep-research planning: derive web-search queries from the seed docs so the
    # guide draws on complementary material, not just a single page. Requires a
    # configured search provider (Tavily or Brave); a no-op otherwise.
    auto_research: bool = Field(default=True)
    max_research_queries: int = Field(default=6, gt=0)
    # How many top search-result pages to fetch into the corpus so the gathered
    # material reaches the section writer (not just the outline planner).
    fetch_top_results: int = Field(default=4, ge=0)
    # Per-section evaluate-and-revise loop (the review pipeline's reflect gate):
    # sections scoring below the threshold are revised up to N times.
    min_quality_score: int = Field(default=75, ge=0, le=100)
    max_revision_attempts: int = Field(default=2, ge=0)
    # Hard total-token ceiling for one guide run (None = no limit). Guards the
    # per-section write + evaluate/revise + grounding loop against runaway cost.
    max_total_tokens: int | None = Field(default=None, gt=0)


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
