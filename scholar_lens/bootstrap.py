"""The composition root: builds the run's adapters from :class:`Config` and
wires each command's use case (docs/engine.md, 5.1).

Only this module and the entrypoints read the configuration; the use cases
receive their ports as arguments.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

# Every prompt template registers its fences before a run reads untrusted
# text (domain.prompt_safety); the sections engine's templates included.
import scholar_lens.adapters.bedrock.prompts  # noqa: F401
import scholar_lens.sections_engine.prompts.extraction  # noqa: F401
import scholar_lens.sections_engine.prompts.review  # noqa: F401
import scholar_lens.sections_engine.prompts.summary  # noqa: F401
from scholar_lens.adapters.aws.batch import BatchJobs
from scholar_lens.adapters.aws.billing import CostExplorerBilling
from scholar_lens.adapters.aws.environment import is_running_in_aws
from scholar_lens.adapters.aws.run_ledger import S3RunLedger
from scholar_lens.adapters.aws.session import AwsSession, aws_session
from scholar_lens.adapters.aws.storage import S3Handler
from scholar_lens.adapters.bedrock.role_models import BedrockEvidenceModels
from scholar_lens.adapters.bedrock.survey_calls import SurveyPlanner, SurveyWriter
from scholar_lens.adapters.checkpoint_version import VersionedCheckpoints
from scholar_lens.adapters.documents import paper_reader
from scholar_lens.adapters.files import LocalOrS3Files
from scholar_lens.adapters.github.conventions import fetch_blog_posts, load_conventions
from scholar_lens.adapters.github.publisher import FigureDownloads, Publisher
from scholar_lens.adapters.packages.pypi import PyPIIndex, package_extras
from scholar_lens.adapters.scholarly.arxiv import ArxivHandler
from scholar_lens.adapters.scholarly.candidates import CandidateFinder, ReferencePages
from scholar_lens.adapters.scholarly.index import (
    DataCiteIndex,
    OpenAlexIndex,
    SemanticScholarIndex,
)
from scholar_lens.adapters.scholarly.paper_source import (
    PaperSource,
    resolve_paper_source,
)
from scholar_lens.adapters.scholarly.plan_index import PaperIndex
from scholar_lens.adapters.scholarly.survey_documents import (
    EvidenceTextStore,
    PaperDocuments,
)
from scholar_lens.adapters.web.guide_research import TechGuideGenerator
from scholar_lens.adapters.web.research import (
    BraveSearchProvider,
    FallbackSearchProvider,
    NullSearchProvider,
    TavilySearchProvider,
    WebResearcher,
    WebSearchProvider,
)
from scholar_lens.adapters.web.snapshot import SnapshotPolicy
from scholar_lens.application.checkpoints import FreshStore
from scholar_lens.application.ports import (
    GuideWriting,
    PaperInputs,
    PaperOrigin,
    PaperWriting,
    PlanIndex,
    ReferenceReader,
    RoleSpec,
    SurveyLimits,
    SurveyWriting,
    TextStore,
    UsageTracker,
    WebResearch,
)
from scholar_lens.application.run import PipelineOutcome, RunContext
from scholar_lens.application.use_cases.guide import run_guide
from scholar_lens.application.use_cases.paper import run_paper
from scholar_lens.application.use_cases.survey_plan import run_survey_plan
from scholar_lens.application.use_cases.survey_write import run_survey_write
from scholar_lens.configs import Config
from scholar_lens.domain.constants import EnvVars, LanguageModelId, S3Paths
from scholar_lens.domain.conventions import BlogPost, PostConventions
from scholar_lens.domain.jobs import JobRequest
from scholar_lens.domain.paper.models import Paper
from scholar_lens.domain.post_names import slugify
from scholar_lens.domain.specs.paper import SUMMARY_MAX_SECTIONS
from scholar_lens.sections_engine import writers as sections_writers

logger = logging.getLogger(__name__)

# Working directory for papers, figures and generated posts: the container's
# scratch space in AWS, the repository root locally.
ROOT_DIR: Path = (
    Path("/tmp") if is_running_in_aws() else Path(__file__).resolve().parents[1]
)


@dataclass(kw_only=True)
class AwsRunContext(RunContext):
    """The run context on AWS: the configuration, the boto sessions the
    adapters are built on and the run's bucket."""

    config: Config
    default_boto_session: AwsSession
    bedrock_boto_session: AwsSession
    s3_handler: S3Handler | None = None


# ---- the blog -----------------------------------------------------------------


def load_run_conventions(config: Config, *, fallback: bool = True) -> PostConventions:
    """The blog's base-branch conventions when GitHub is configured, else the
    bundled copy (``fallback`` False: an unreachable blog raises)."""
    github = config.resources.github
    return load_conventions(
        repo_name=github.repo_name,
        branch=github.base_branch,
        token=os.getenv(EnvVars.GITHUB_TOKEN.value),
        use_blog=github.enabled,
        fallback=fallback,
    )


def load_run_blog_posts(config: Config) -> list[BlogPost]:
    """The blog's posts, or none when GitHub is not configured. A failed
    listing raises; :meth:`RunContext.require_blog_posts` then goes on with
    none (the model proposes fresh short tags, and the blog's own CI is left
    to catch a description another post already uses)."""
    github = config.resources.github
    if not (github.enabled and github.repo_name):
        return []
    return fetch_blog_posts(
        github.repo_name, github.base_branch, os.getenv(EnvVars.GITHUB_TOKEN.value)
    )


class ConfiguredBlog:
    """The :class:`~scholar_lens.application.ports.BlogSource` of the
    configured blog repository."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def conventions(self, *, fallback: bool = True) -> PostConventions:
        return load_run_conventions(self.config, fallback=fallback)

    def posts(self) -> list[BlogPost]:
        return load_run_blog_posts(self.config)


def build_publisher(context: RunContext) -> Publisher:
    """The run's :class:`Publisher`."""
    assert isinstance(context, AwsRunContext)
    resources = context.config.resources
    return Publisher(
        resources.github,
        root_dir=context.root_dir,
        s3_handler=context.s3_handler,
        s3_bucket_name=resources.s3_bucket_name,
        s3_prefix=resources.s3_prefix,
        conventions=context.require_conventions(),
        site_posts=context.require_blog_posts(),
    )


def run_context(
    config: Config,
    *,
    default_boto_session: AwsSession,
    bedrock_boto_session: AwsSession,
    s3_handler: S3Handler | None = None,
    root_dir: Path = ROOT_DIR,
    publisher_factory: Any = None,
) -> AwsRunContext:
    """A run context on the given sessions (``publisher_factory``: the
    run's publisher, default :func:`build_publisher`)."""
    return AwsRunContext(
        config=config,
        blog=ConfiguredBlog(config),
        publisher_factory=publisher_factory or build_publisher,
        default_boto_session=default_boto_session,
        bedrock_boto_session=bedrock_boto_session,
        s3_handler=s3_handler,
        files=LocalOrS3Files(s3_handler),
        root_dir=root_dir,
    )


def build_context(config: Config) -> AwsRunContext:
    """The run context of ``config`` (boto sessions and the optional bucket).

    The AWS profile comes from the env in Batch (instance role /
    ``AWS_PROFILE``) and from config locally.
    """
    profile_name = (
        os.getenv(EnvVars.AWS_PROFILE_NAME.value)
        if is_running_in_aws()
        else config.resources.profile_name
    )
    default_session = aws_session(
        profile_name=profile_name, region_name=config.resources.default_region_name
    )
    bedrock_session = aws_session(
        profile_name=profile_name, region_name=config.resources.bedrock_region_name
    )
    s3_handler = (
        S3Handler(default_session, config.resources.s3_bucket_name)
        if config.resources.s3_bucket_name
        else None
    )
    return run_context(
        config,
        default_boto_session=default_session,
        bedrock_boto_session=bedrock_session,
        s3_handler=s3_handler,
    )


# ---- shared adapters ----------------------------------------------------------


def evidence_store(
    local_dir: Path,
    s3_handler: S3Handler | None,
    s3_prefix: str,
    area: S3Paths = S3Paths.SURVEYS,
    *,
    fresh: bool = False,
) -> TextStore:
    """The run's reading cache and checkpoint store
    (:class:`EvidenceTextStore`, checkpoints under the check code's version,
    :class:`VersionedCheckpoints`); a ``fresh`` run writes it without
    reading it."""
    # Checkpoints sit under the check code's version: a code change never
    # resumes from a stage the old code made.
    store: TextStore = VersionedCheckpoints(
        EvidenceTextStore(local_dir, s3_handler, s3_prefix, area)
    )
    return FreshStore(store) if fresh else store


def evidence_models(context: AwsRunContext) -> BedrockEvidenceModels:
    return BedrockEvidenceModels(
        context.bedrock_boto_session, context.config.resources.bedrock_region_name
    )


def _reader_tag(model_id: str) -> str:
    return slugify(LanguageModelId(model_id).value)


# ---- guide --------------------------------------------------------------------


def package_release(name: str) -> str | None:
    """``name``'s latest release on PyPI (None when unknown; raises
    :class:`~scholar_lens.domain.sources.IndexUnreachable` when PyPI cannot
    answer)."""
    found = package_extras(name)
    return (found[0] or None) if found else None


def build_search_provider() -> WebSearchProvider:
    """Tavily (LLM-oriented, returns cleaned content) > Brave > no web search."""
    if os.getenv(EnvVars.TAVILY_API_KEY.value):
        if os.getenv(EnvVars.BRAVE_API_KEY.value):
            logger.info("Using Tavily Search, falling back to Brave, for web research.")
            return FallbackSearchProvider(TavilySearchProvider(), BraveSearchProvider())
        logger.info("Using Tavily Search for web research.")
        return TavilySearchProvider()
    if os.getenv(EnvVars.BRAVE_API_KEY.value):
        logger.info("Using Brave Search for web research.")
        return BraveSearchProvider()
    logger.info("No search API key set; researching the supplied URLs only.")
    return NullSearchProvider()


class GuideToolkit:
    """The guide use case's :class:`~scholar_lens.application.ports.GuideTools`
    on Bedrock, the web and PyPI."""

    def __init__(self, context: AwsRunContext) -> None:
        self.context = context
        self.cfg = context.config.tech_guide
        self.packages = PyPIIndex()
        self.figures = FigureDownloads()

    @property
    def models(self) -> BedrockEvidenceModels:
        return evidence_models(self.context)

    @property
    def writing(self) -> GuideWriting:
        cfg, effort = self.cfg, self.cfg.thinking_effort
        reader = RoleSpec(cfg.synopsis_model_id, effort, True)
        return GuideWriting(
            roles={
                "read": RoleSpec(cfg.synopsis_model_id, effort, False),
                "outline": RoleSpec(cfg.writing_model_id, "high", True),
                "write": RoleSpec(cfg.writing_model_id, effort, True, reader),
                "evaluate": RoleSpec(cfg.evaluation_model_id, effort, True, reader),
            },
            max_cost_usd=cfg.max_cost_usd,
            reader_tag=_reader_tag(cfg.synopsis_model_id),
            max_sections=cfg.max_sections,
            max_sources=cfg.max_sources_read,
            min_total_chars=cfg.min_total_chars,
            max_total_chars=cfg.max_total_chars,
        )

    def researcher(self) -> WebResearcher:
        return WebResearcher(search_provider=build_search_provider())

    def research(
        self, researcher: WebResearch, tracker: UsageTracker, *, title_max_length: int
    ) -> TechGuideGenerator:
        cfg = self.cfg
        researcher = cast(WebResearcher, researcher)
        return TechGuideGenerator(
            relevance_model_id=LanguageModelId(cfg.relevance_model_id),
            boto_session=self.context.bedrock_boto_session,
            researcher=researcher,
            language=self.context.config.output_language,
            title_max_length=title_max_length,
            auto_research=cfg.auto_research,
            max_research_queries=cfg.max_research_queries,
            fetch_top_results=cfg.fetch_top_results,
            callbacks=[tracker],
            release_lookup=package_release,
            repo_tags=researcher.repo_tags,
            max_third_party_pages=cfg.max_third_party_pages,
            min_secondary_pages=cfg.min_secondary_pages,
            official_scope_link_share=cfg.official_scope_link_share,
            demoted_domains=cfg.demoted_domains,
            excluded_domains=cfg.excluded_domains,
        )

    def snapshots(self, *, fresh: bool) -> SnapshotPolicy:
        context = self.context
        return SnapshotPolicy(
            EvidenceTextStore(
                context.papers_dir / "guide-sources",
                context.s3_handler,
                context.config.resources.s3_prefix,
                S3Paths.GUIDES,
            ),
            ttl_days=self.cfg.source_snapshot_ttl_days,
            fresh=fresh,
        )

    def evidence_store(self, *, fresh: bool) -> TextStore:
        context = self.context
        return evidence_store(
            context.papers_dir / "guide-evidence",
            context.s3_handler,
            context.config.resources.s3_prefix,
            S3Paths.GUIDES,
            fresh=fresh,
        )


# ---- paper review and summary ---------------------------------------------------


class PaperToolkit:
    """The paper use case's :class:`~scholar_lens.application.ports.PaperTools`:
    the paper's sources and parsers, the evidence engine on Bedrock and the
    sections engine."""

    def __init__(self, context: AwsRunContext) -> None:
        self.context = context
        self.config = context.config
        self.figures = FigureDownloads()

    @property
    def models(self) -> BedrockEvidenceModels:
        return evidence_models(self.context)

    def engine(
        self, post: Literal["review", "summary"]
    ) -> Literal["evidence", "sections"]:
        cfg = self.config.explanation if post == "review" else self.config.summary
        return cfg.engine

    def inputs(self, kind: str) -> PaperInputs:
        """Only the sections review engine reads citations and searches the
        code index; summaries and evidence-engine reviews use the codebase
        summary alone."""
        sections_review = kind == "review" and self.engine("review") == "sections"
        return PaperInputs(citations=sections_review, code_index=sections_review)

    def resolve(self, source: str) -> PaperSource:
        return resolve_paper_source(source)

    async def read(
        self,
        paper_dir: Path,
        source: PaperOrigin,
        repo_urls: list[str],
        parse_pdf: bool,
        tracker: UsageTracker,
        *,
        inputs: PaperInputs,
    ) -> tuple[Paper, Any, Any]:
        session = self.context.bedrock_boto_session
        return await paper_reader.prepare_paper_data(
            self.context,
            paper_dir,
            cast(PaperSource, source),
            repo_urls,
            parse_pdf,
            tracker,
            inputs=inputs,
            index_factory=functools.partial(
                sections_writers.code_index, self.config.code, session
            ),
            citations_factory=functools.partial(
                sections_writers.citation_summarizer, self.config.citations, session
            ),
        )

    def writing(self, post: Literal["review", "summary"]) -> PaperWriting:
        if post == "review":
            cfg = self.config.explanation
            writer_id, outline_id, judge_id = (
                cfg.paper_synthesis_model_id,
                cfg.paper_planning_model_id,
                cfg.paper_evaluation_model_id,
            )
            reader_id, effort = cfg.reader_model_id, cfg.thinking_effort
            lengths = (cfg.min_total_chars, cfg.max_total_chars)
            max_sections, chars_per_reader = cfg.max_sections, cfg.chars_per_reader
            max_cost = cfg.max_cost_usd
            # The review plans at the configured planning effort.
            planning_effort: str = cfg.planning_effort
        else:
            scfg = self.config.summary
            writer_id = outline_id = judge_id = scfg.summary_model_id
            reader_id, effort = scfg.reader_model_id, scfg.thinking_effort
            lengths = (scfg.min_total_chars, scfg.max_total_chars)
            max_sections = SUMMARY_MAX_SECTIONS
            chars_per_reader = scfg.chars_per_reader
            max_cost = scfg.max_cost_usd
            # A summary's short outline at high effort.
            planning_effort = "high"
        reader = RoleSpec(reader_id, effort, True)
        return PaperWriting(
            roles={
                "read": RoleSpec(reader_id, effort, False),
                "outline": RoleSpec(outline_id, planning_effort, True),
                "write": RoleSpec(writer_id, effort, True, reader),
                "evaluate": RoleSpec(judge_id, effort, True, reader),
            },
            max_cost_usd=max_cost,
            reader_tag=_reader_tag(reader_id),
            max_sections=max_sections,
            chars_per_reader=chars_per_reader,
            min_total_chars=lengths[0],
            max_total_chars=lengths[1],
        )

    def evidence_store(self, *, fresh: bool) -> TextStore:
        context = self.context
        return evidence_store(
            context.papers_dir / "paper-evidence",
            context.s3_handler,
            self.config.resources.s3_prefix,
            S3Paths.PAPERS,
            fresh=fresh,
        )

    def sections(self) -> sections_writers.SectionsPaperWriter:
        return sections_writers.SectionsPaperWriter(
            explanation=self.config.explanation,
            summary=self.config.summary,
            language=self.config.output_language,
            boto_session=self.context.bedrock_boto_session,
        )


# ---- survey -------------------------------------------------------------------

# Images of a paper's arXiv HTML rendering offered to its reading agent.
MAX_PAPER_IMAGES = 40


class SurveyPlanToolkit:
    """The survey plan use case's
    :class:`~scholar_lens.application.ports.SurveyPlanTools`: the scope and
    selection calls on Bedrock, the candidate step on the scholarly indexes."""

    def __init__(self, context: AwsRunContext) -> None:
        self.context = context
        self.cfg = context.config.survey

    @property
    def limits(self) -> SurveyLimits:
        cfg = self.cfg
        return SurveyLimits(
            min_papers=cfg.min_papers,
            max_papers=cfg.max_papers,
            candidate_pool=cfg.candidate_pool,
        )

    def planner(self, tracker: UsageTracker) -> SurveyPlanner:
        cfg = self.cfg
        return SurveyPlanner(
            scope_model_id=LanguageModelId(cfg.scope_model_id),
            selection_model_id=LanguageModelId(cfg.selection_model_id),
            boto_session=self.context.bedrock_boto_session,
            thinking_effort=cfg.thinking_effort,
            max_total_tokens=cfg.max_total_tokens,
            callbacks=[tracker],
        )

    def finder(self) -> CandidateFinder:
        return CandidateFinder(
            openalex=OpenAlexIndex(),
            arxiv=ArxivHandler(),
            datacite=DataCiteIndex(),
            semantic_scholar=SemanticScholarIndex(),
            researcher=WebResearcher(),
        )


class SurveyWriteToolkit:
    """The survey write use case's
    :class:`~scholar_lens.application.ports.SurveyWriteTools`: the evidence
    engine and the finalization call on Bedrock, the papers' text from arXiv,
    OpenAlex and their PDFs."""

    def __init__(self, context: AwsRunContext) -> None:
        self.context = context
        self.cfg = context.config.survey
        self.figures = FigureDownloads()

    @property
    def models(self) -> BedrockEvidenceModels:
        return evidence_models(self.context)

    @property
    def writing(self) -> SurveyWriting:
        cfg, effort = self.cfg, self.cfg.thinking_effort
        # The writer falls back to the reading model when Bedrock's content
        # filter keeps stopping it (attack-heavy sections).
        reader = RoleSpec(cfg.card_model_id, effort, True)
        return SurveyWriting(
            roles={
                "read": RoleSpec(cfg.card_model_id, effort, False),
                "outline": RoleSpec(cfg.outline_model_id, "high", True),
                "write": RoleSpec(cfg.writing_model_id, effort, True, reader),
                "evaluate": RoleSpec(cfg.evaluation_model_id, effort, True, reader),
            },
            max_cost_usd=cfg.write_max_cost_usd,
            reader_tag=_reader_tag(cfg.card_model_id),
            chars_per_core=cfg.chars_per_core_paper,
            chars_per_supporting=cfg.chars_per_supporting_paper,
            min_total_chars=cfg.min_total_chars,
            max_total_chars=cfg.max_total_chars,
        )

    def index(self) -> PaperIndex:
        return PaperIndex(DataCiteIndex(), OpenAlexIndex())

    def finalizer(self, tracker: UsageTracker) -> SurveyWriter:
        return SurveyWriter(
            finalization_model_id=LanguageModelId(self.cfg.outline_model_id),
            boto_session=self.context.bedrock_boto_session,
            language=self.context.config.output_language,
            thinking_effort=self.cfg.thinking_effort,
            callbacks=[tracker],
        )

    def reader(self) -> ReferencePages:
        return ReferencePages(WebResearcher(max_images_per_page=MAX_PAPER_IMAGES))

    def documents(
        self, reader: ReferenceReader, index: PlanIndex, abstracts: dict[str, str]
    ) -> PaperDocuments:
        return PaperDocuments(
            cast(ReferencePages, reader).researcher,
            cast(PaperIndex, index).openalex,
            abstracts,
        )

    def evidence_store(self, *, fresh: bool) -> TextStore:
        context = self.context
        return evidence_store(
            context.papers_dir / "survey-evidence",
            context.s3_handler,
            context.config.resources.s3_prefix,
            fresh=fresh,
        )


# ---- remote jobs and costs -----------------------------------------------------

# The cost allocation tag every resource of the deployment carries.
COST_TAG_KEY = "ProjectName"


def job_queue(context: AwsRunContext) -> BatchJobs:
    """The deployment's Batch jobs and their result manifests."""
    return BatchJobs(
        context.default_boto_session,
        s3_handler=context.s3_handler,
        s3_prefix=context.config.resources.s3_prefix,
    )


def cost_sources(
    context: AwsRunContext, *, billed: bool
) -> tuple[S3RunLedger, CostExplorerBilling | None]:
    """The run ledger (result manifests) and, with ``billed``, what Cost
    Explorer billed for the project."""
    resources = context.config.resources
    session = context.default_boto_session
    assert resources.s3_bucket_name
    ledger = S3RunLedger(
        session.client("s3"),
        resources.s3_bucket_name,
        f"{resources.s3_prefix}/{S3Paths.RESULTS.value}".lstrip("/"),
    )
    billing = (
        CostExplorerBilling(
            # Cost Explorer's API endpoint is in us-east-1.
            session.client("ce", region_name="us-east-1"),
            COST_TAG_KEY,
            resources.project_name,
        )
        if billed
        else None
    )
    return ledger, billing


# ---- the commands -------------------------------------------------------------


async def run_use_case(
    context: AwsRunContext, request: JobRequest, tracker: Any
) -> PipelineOutcome:
    """Run ``request``'s use case on the adapters built from ``context``."""
    if request.kind == "guide":
        return await run_guide(context, request, tracker, GuideToolkit(context))
    if request.kind == "survey_plan":
        return await run_survey_plan(
            context, request, tracker, SurveyPlanToolkit(context)
        )
    if request.kind == "survey_write":
        return await run_survey_write(
            context, request, tracker, SurveyWriteToolkit(context)
        )
    return await run_paper(context, request, tracker, PaperToolkit(context))
