from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    model_validator,
)

from scholar_lens.src.constants import EmbeddingModelId, LanguageModelId
from scholar_lens.src.logger import is_running_in_aws

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Effort level for adaptive-thinking models (e.g. Opus 4.8, Opus 5.5). Lower
# effort means shorter reasoning traces — faster and cheaper. Ignored by legacy
# thinking models, which use a fixed token budget instead.
ThinkingEffort = Literal["low", "medium", "high", "xhigh", "max"]


class _Section(BaseModel):
    # Reject unknown keys so a YAML typo (e.g. ``max_total_token``) fails loudly
    # instead of silently falling back to the default.
    model_config = ConfigDict(extra="forbid")


class Github(_Section):
    enabled: bool = Field(default=False)
    repo_name: str | None = Field(default=None)
    base_branch: str = Field(default="main")
    branch_prefix: str = Field(default="paper-reviews")
    author_name: str = Field(default="Scholar Lens Bot")
    author_email: EmailStr | None = Field(default=None)
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


class Resources(_Section):
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


class Paper(_Section):
    figure_analysis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_HAIKU
    )
    citation_extraction_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_SONNET
    )
    attributes_extraction_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_HAIKU
    )


class Code(_Section):
    code_analysis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_HAIKU
    )
    code_summarization_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_HAIKU
    )
    embed_model_id: EmbeddingModelId = Field(default=EmbeddingModelId.TITAN_EMBED_V2)
    chunk_size: int = Field(default=1024, gt=0, le=1_000_000)
    chunk_overlap: int = Field(default=256, ge=0)
    # Bound indexing cost: the most relevant files and chunks of the repos.
    max_files: int = Field(default=150, gt=0)
    max_chunks: int = Field(default=400, gt=0)

    @model_validator(mode="after")
    def _check_overlap_lt_size(self) -> "Code":
        # overlap >= size makes RecursiveCharacterTextSplitter loop/misbehave.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})."
            )
        return self


class Citations(_Section):
    citation_summarization_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_HAIKU
    )
    # When True, download each cited paper's full text (slow, arXiv-heavy). When
    # False (default), summarise from the abstract resolved via Crossref, then
    # OpenAlex, then Semantic Scholar (only with an API key), then arXiv — far
    # fewer calls and no arXiv rate-limit storms.
    prefer_full_text: bool = Field(default=False)


class Explanation(_Section):
    # Plans the review from the whole paper and edits the assembled article.
    paper_planning_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_OPUS
    )
    # Writes each planned section.
    paper_synthesis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_OPUS
    )
    # Checks each section against its brief (0-100 + missing items) and gates
    # the revise loop. An Opus-tier judge keeps scores stable.
    paper_evaluation_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_OPUS
    )
    # Writes the subtitle, TL;DR and closing assessment from the edited review.
    paper_finalization_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_OPUS
    )
    evaluator_enable_thinking: bool = Field(default=False)
    synthesizer_enable_thinking: bool = Field(default=False)
    thinking_effort: ThinkingEffort = Field(default="medium")
    # Planning and whole-article editing decide coverage and consistency, so
    # they think harder than the per-section calls.
    planning_effort: ThinkingEffort = Field(default="high")
    # A section scoring below min_quality_score, or missing a must-cover item,
    # is revised; max_synthesis_attempts counts drafts per section.
    min_quality_score: int = Field(default=70, ge=0, le=100)
    max_synthesis_attempts: int = Field(default=3, ge=1, le=6)
    # Upper bound on planned sections; overflow is merged into the last one.
    max_sections: int = Field(default=20, ge=1, le=60)
    # Whole-article edit passes; a second pass runs only for reported coverage
    # gaps. 0 disables editing.
    max_edit_rounds: int = Field(default=2, ge=0, le=3)
    # Hard total-token ceiling for one review run (None = no limit). The default
    # leaves ~6x headroom over a LoRA-length paper (~1.4M tokens, ~$6 on Opus
    # 5.5) and stops a runaway run near $40.
    max_total_tokens: int | None = Field(default=8_000_000, gt=0)


class Summary(_Section):
    summary_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V5_5_OPUS)
    summarizer_enable_thinking: bool = Field(default=False)
    thinking_effort: ThinkingEffort = Field(default="medium")
    # Hard total-token ceiling for one summary run (None = no limit); a typical
    # summary uses ~0.13M tokens.
    max_total_tokens: int | None = Field(default=1_000_000, gt=0)


class TechGuide(_Section):
    relevance_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_HAIKU
    )
    synopsis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_SONNET
    )
    writing_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V5_5_OPUS)
    # Per-section evaluator (0-100 score + feedback) gating the revise loop.
    evaluation_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_OPUS
    )
    writer_enable_thinking: bool = Field(default=False)
    # Thinking for the evaluator chain, independent of the writer (parity with
    # the review pipeline's separate evaluator/synthesizer flags). None → inherit
    # writer_enable_thinking, preserving prior single-flag behaviour.
    evaluator_enable_thinking: bool | None = Field(default=None)
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
    # Per-section evaluate-and-revise loop (the review pipeline's evaluate gate):
    # sections scoring below the threshold are revised up to N times.
    min_quality_score: int = Field(default=75, ge=0, le=100)
    max_revision_attempts: int = Field(default=1, ge=0, le=5)
    # Soft limit on one guide run (None = no limit); default keeps a run near
    # $15 on Opus 5.5. Past it the remaining sections skip evaluate/revise/
    # grounding; past 1.5x the run aborts.
    max_total_tokens: int | None = Field(default=3_000_000, gt=0)


class Config(_Section):
    resources: Resources = Field(
        default_factory=lambda: Resources(project_name="scholar-lens")
    )
    paper: Paper = Field(default_factory=Paper)
    code: Code = Field(default_factory=Code)
    citations: Citations = Field(default_factory=Citations)
    explanation: Explanation = Field(default_factory=Explanation)
    summary: Summary = Field(default_factory=Summary)
    tech_guide: TechGuide = Field(default_factory=TechGuide)
    output_language: str = Field(default="Korean")

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "Config":
        with open(file_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data if config_data else {})

    @classmethod
    def load(cls) -> "Config":
        load_dotenv()
        if CONFIG_PATH.exists():
            return cls.from_yaml(CONFIG_PATH)
        if is_running_in_aws():
            # Defaults have no bucket and GitHub disabled: a job would "succeed"
            # while publishing nothing. Fail instead.
            raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
        return cls()
