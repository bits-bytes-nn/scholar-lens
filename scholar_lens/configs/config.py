import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    model_validator,
)

from scholar_lens.adapters.aws.environment import is_running_in_aws
from scholar_lens.domain.constants import EmbeddingModelId, LanguageModelId

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
CONFIG_PATH_ENV = "SCHOLAR_LENS_CONFIG"

# Effort level for adaptive-thinking models (e.g. Opus 4.8, Opus 5.5). Lower
# effort means shorter reasoning traces — faster and cheaper. Ignored by legacy
# thinking models, which use a fixed token budget instead.
ThinkingEffort = Literal["low", "medium", "high", "xhigh", "max"]


class _Section(BaseModel):
    # Reject unknown keys so a YAML typo (e.g. ``max_total_token``) fails loudly
    # instead of silently falling back to the default.
    model_config = ConfigDict(extra="forbid")
    # Keys a section no longer has but an older config.yaml may still set:
    # each is dropped with a warning naming why, instead of failing the load.
    retired_keys: ClassVar[dict[str, str]] = {}

    @model_validator(mode="before")
    @classmethod
    def _drop_retired_keys(cls, data: Any) -> Any:
        if not (isinstance(data, dict) and cls.retired_keys):
            return data
        kept = dict(data)
        for key, reason in cls.retired_keys.items():
            if key in kept:
                del kept[key]
                logger.warning(
                    "Config key '%s' of %s is ignored and can be deleted: %s",
                    key,
                    cls.__name__,
                    reason,
                )
        return kept


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
    survey_category: str = Field(default="Paper Surveys")
    retired_keys: ClassVar[dict[str, str]] = {
        "public_assets": (
            "uploads to S3 are always private; posts load their images from "
            "the blog repository."
        ),
    }

    @model_validator(mode="after")
    def _check_repo(self) -> "Github":
        # Fail at load, not after a run's paid stages reach the PR step.
        owner, _, name = (self.repo_name or "").partition("/")
        if self.enabled and not (owner and name and "/" not in name):
            raise ValueError("github.enabled needs repo_name as 'owner/name'.")
        if not self.branch_prefix.strip("/"):
            raise ValueError("github.branch_prefix must not be empty.")
        return self


class Resources(_Section):
    project_name: str = Field(min_length=1)
    stage: Literal["dev", "prod"] = Field(default="dev")
    profile_name: str | None = Field(default=None)
    default_region_name: str = Field(default="ap-northeast-2")
    bedrock_region_name: str = Field(default="us-west-2")
    s3_bucket_name: str | None = Field(default=None)
    # Keys are built as "<prefix>/…"; the job role's IAM scope is the same.
    s3_prefix: str = Field(
        default="scholar-lens", pattern=r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$"
    )
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
    # The writing engine: "sections" (plan, sections scored and revised,
    # edit, finalize) or "evidence" (reading agents record checked evidence
    # that the outline gives to sections; docs/redesign.md). The evidence
    # engine reads with reader_model_id, plans with the planning model,
    # writes with the synthesis model and checks with the evaluation model.
    engine: Literal["sections", "evidence"] = Field(default="sections")
    reader_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V5_5_SONNET)
    # Evidence engine: the review's prose characters (Korean).
    min_total_chars: int = Field(default=25_000, ge=5_000, le=200_000)
    max_total_chars: int = Field(default=55_000, ge=5_000, le=200_000)

    @model_validator(mode="after")
    def _review_length_range(self) -> "Explanation":
        if self.min_total_chars > self.max_total_chars:
            raise ValueError("explanation.min_total_chars exceeds max_total_chars")
        return self

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
    # Evidence engine: hard ceiling on one run's estimated cost (list prices,
    # USD; None = no limit). A review costs about $5-11.
    max_cost_usd: float | None = Field(default=40.0, gt=0)


class Summary(_Section):
    summary_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V5_5_OPUS)
    # "sections" (one summary call) or "evidence" (docs/redesign.md).
    engine: Literal["sections", "evidence"] = Field(default="sections")
    reader_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V5_5_SONNET)
    # Evidence engine: the summary's prose characters (Korean; five minutes).
    min_total_chars: int = Field(default=2_500, ge=1_000, le=20_000)
    max_total_chars: int = Field(default=4_500, ge=1_000, le=20_000)

    @model_validator(mode="after")
    def _summary_length_range(self) -> "Summary":
        if self.min_total_chars > self.max_total_chars:
            raise ValueError("summary.min_total_chars exceeds max_total_chars")
        return self

    summarizer_enable_thinking: bool = Field(default=False)
    thinking_effort: ThinkingEffort = Field(default="medium")
    # Hard total-token ceiling for one summary run (None = no limit); a typical
    # summary uses ~0.13M tokens.
    max_total_tokens: int | None = Field(default=1_000_000, gt=0)
    # Evidence engine: hard ceiling on one run's estimated cost (list prices,
    # USD; None = no limit). A summary costs about $3.
    max_cost_usd: float | None = Field(default=12.0, gt=0)


# The default demoted domain classes of ``tech_guide.demoted_domains``.
DEMOTED_DOMAINS: dict[str, tuple[str, ...]] = {
    "forum": (
        "stackoverflow.com",
        "*.stackexchange.com",
        "reddit.com",
        "news.ycombinator.com",
        "discuss.*",
        "forum.*",
        "forums.*",
        "community.*",
    ),
    "social": ("x.com", "twitter.com", "linkedin.com", "facebook.com", "youtube.com"),
    "aggregator": ("*.libhunt.com", "stackshare.io", "alternativeto.net"),
}


class TechGuide(_Section):
    # The guide is written by the evidence engine (docs/redesign.md): the
    # relevance model gates the sources, plans the searches and files the
    # guide; reading agents use the synopsis model, the outline and sections
    # the writing model and the checks the evaluation model.
    relevance_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_HAIKU
    )
    synopsis_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_SONNET
    )
    writing_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V5_5_OPUS)
    evaluation_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_OPUS
    )
    thinking_effort: ThinkingEffort = Field(default="medium")
    # Deep-research planning: derive web-search queries from the seed docs so the
    # guide draws on complementary material, not just a single page. Requires a
    # configured search provider (Tavily or Brave); a no-op otherwise.
    auto_research: bool = Field(default=True)
    max_research_queries: int = Field(default=6, gt=0)
    # How many top search-result pages to fetch into the corpus so the gathered
    # material reaches the section writer (not just the outline planner); taken
    # round-robin across the queries, so 12 reads the top two of each.
    fetch_top_results: int = Field(default=12, ge=0)
    # Search results are ranked by source authority, then publication year,
    # then search rank; at most this many of the fetched pages are
    # third-party (or a demoted class below), the rest go to the official
    # docs and repository.
    max_third_party_pages: int = Field(default=4, ge=0)
    # Of the fetched result pages, this many slots are kept for primary
    # non-official material (papers, the project's design docs and posts,
    # reference-quality pages) while such results remain, so official pages
    # cannot fill every slot and the guide keeps its design rationale. The
    # third-party cap above still bounds them; demoted classes never count.
    # 3 is a quarter of the default fetch budget, within the default cap.
    min_secondary_pages: int = Field(default=3, ge=0)
    # Domain classes (e.g. forum, social, aggregator) -> host patterns
    # ("example.com" with its subdomains, "*.example.com", "forum.*"): their
    # pages rank last and count as third-party (config-template.yaml lists
    # the same defaults; a test keeps the two equal).
    demoted_domains: dict[str, list[str]] = Field(
        default_factory=lambda: {k: list(v) for k, v in DEMOTED_DOMAINS.items()}
    )
    # Domains the search backends leave out (Tavily exclude_domains, Brave
    # -site:); "*.example.com" is sent as example.com.
    excluded_domains: list[str] = Field(default_factory=list)
    # The gathered corpus (queries, results, pages and their records) is
    # saved under a hash of the inputs and reused by a rerun with the same
    # inputs within this many days (0 = always gather again; the CLI's
    # --fresh-sources forces a fresh gather once).
    source_snapshot_ttl_days: float = Field(default=7.0, ge=0)
    # Upper bound on guide sections, told to the outline planner so it plans a
    # self-contained guide (usage, internals and integration) within it.
    # Fewer, deeper sections: each teaches its features once (the guide's
    # facts and code are checked; more sections, more places to slip).
    max_sections: int = Field(default=11, ge=3, le=30)
    # Hard ceiling on one run's estimated cost (list prices, USD; None = no
    # limit). A guide costs about $15-23.
    max_cost_usd: float | None = Field(default=60.0, gt=0)
    # Pages read (the docs and reference material first) and the guide's
    # prose characters (Korean).
    max_sources_read: int = Field(default=40, ge=1, le=120)
    min_total_chars: int = Field(default=25_000, ge=5_000, le=200_000)
    max_total_chars: int = Field(default=45_000, ge=5_000, le=200_000)

    @model_validator(mode="after")
    def _length_range(self) -> "TechGuide":
        if self.min_total_chars > self.max_total_chars:
            raise ValueError("tech_guide.min_total_chars exceeds max_total_chars")
        return self


class Survey(_Section):
    # Plan stage: the scope call (lineages, search queries, landmark titles)
    # and the selection call (which papers, lineages and roles).
    scope_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V5_5_SONNET)
    selection_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_OPUS
    )
    thinking_effort: ThinkingEffort = Field(default="medium")
    # Papers a survey covers; the selection call picks the count in between
    # from the topic's breadth and the pool.
    min_papers: int = Field(default=6, ge=2, le=50)
    max_papers: int = Field(default=25, ge=2, le=60)
    # Best-ranked candidates the selection call sees.
    candidate_pool: int = Field(default=60, ge=10, le=200)
    # Soft limit on one plan run (None = no limit): past it the repair round
    # is skipped; past 1.5x the run aborts. A plan uses ~50k tokens.
    max_total_tokens: int | None = Field(default=300_000, gt=0)

    # Write stage (`survey write`, the evidence engine; docs/redesign.md):
    # reading agents use card_model_id, the outline (and the finalization
    # call) outline_model_id, the sections writing_model_id and the checks
    # evaluation_model_id.
    card_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V5_5_SONNET)
    outline_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V5_5_OPUS)
    writing_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V5_5_OPUS)
    evaluation_model_id: LanguageModelId = Field(
        default=LanguageModelId.CLAUDE_V5_5_OPUS
    )
    # Post length in characters of prose (Korean; other languages x2.5):
    # each core and supporting paper adds its share to its lineage section,
    # the other sections their depth's size; the total is then scaled into
    # [min_total_chars, max_total_chars].
    chars_per_core_paper: int = Field(default=3_000, ge=500, le=20_000)
    chars_per_supporting_paper: int = Field(default=800, ge=100, le=5_000)
    min_total_chars: int = Field(default=20_000, ge=5_000, le=200_000)
    max_total_chars: int = Field(default=70_000, ge=5_000, le=200_000)
    # Hard ceiling on one write run's estimated cost (list prices, USD; None
    # = no limit). A survey costs about $15-18.
    write_max_cost_usd: float | None = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> "Survey":
        if self.min_total_chars > self.max_total_chars:
            raise ValueError(
                f"min_total_chars ({self.min_total_chars}) must not exceed "
                f"max_total_chars ({self.max_total_chars})."
            )
        if self.min_papers > self.max_papers:
            raise ValueError(
                f"min_papers ({self.min_papers}) must not exceed max_papers "
                f"({self.max_papers})."
            )
        if self.max_papers > self.candidate_pool:
            raise ValueError(
                f"max_papers ({self.max_papers}) must not exceed candidate_pool "
                f"({self.candidate_pool})."
            )
        return self


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
    survey: Survey = Field(default_factory=Survey)
    output_language: str = Field(default="Korean")

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "Config":
        with open(file_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data if config_data else {})

    @classmethod
    def load(cls) -> "Config":
        """``config.yaml`` next to this module, or the file named by
        ``SCHOLAR_LENS_CONFIG`` (an evaluation run with other settings)."""
        load_dotenv()
        if override := os.environ.get(CONFIG_PATH_ENV):
            return cls.from_yaml(override)
        if CONFIG_PATH.exists():
            return cls.from_yaml(CONFIG_PATH)
        if is_running_in_aws():
            # Defaults have no bucket and GitHub disabled: a job would "succeed"
            # while publishing nothing. Fail instead.
            raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
        return cls()
