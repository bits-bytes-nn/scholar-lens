"""Ports: what the use cases need from the outside world."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from scholar_lens.domain.conventions import BlogPost, PostConventions
from scholar_lens.domain.costs import RunRecord
from scholar_lens.domain.evidence import SourceDocument
from scholar_lens.domain.guide.models import GuideMaterial, TechGuide
from scholar_lens.domain.guide.sdk_index import PackageSource
from scholar_lens.domain.jobs import JobStatusReport
from scholar_lens.domain.paper.models import Paper, ReviewText
from scholar_lens.domain.publishing import PublishRequest
from scholar_lens.domain.survey.plan import (
    Candidate,
    PlanPaper,
    SourceRecord,
    SurveyScope,
    SurveySelection,
    TimeRange,
)
from scholar_lens.domain.survey.references import CandidatePool, ReadReference
from scholar_lens.domain.usage import RunUsage, UsageEvent


class RunLedger(Protocol):
    """Finished runs, from their result manifests."""

    def runs(self, since: datetime, until: datetime) -> Iterable[RunRecord]: ...


class JobQueue(Protocol):
    """Remote jobs (AWS Batch) and their results."""

    def describe(self, job_id: str) -> JobStatusReport: ...

    def wait(self, job_id: str, *, timeout_seconds: float) -> JobStatusReport:
        """The report once the job is terminal or ``timeout_seconds`` pass
        (the last report either way)."""
        ...


class BilledCosts(Protocol):
    """What the provider billed for the project, by service."""

    def billed(self, since: datetime, until: datetime) -> dict[str, float]: ...


# ---- the evidence engine's ports -----------------------------------------------

ModelRole = Literal["read", "outline", "write", "evaluate"]
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ModelText(str):
    """A text answer that knows whether the model stopped at its output
    limit (``cut``): a cut answer is incomplete however it ends."""

    cut: bool

    def __new__(cls, text: str, *, cut: bool = False) -> ModelText:
        obj = super().__new__(cls, text)
        obj.cut = cut
        return obj


def was_cut(text: str) -> bool:
    return bool(getattr(text, "cut", False))


@dataclass(frozen=True)
class AgentLimits:
    turns: int
    total_tokens: int | None = None


@dataclass(frozen=True)
class AgentOutcome:
    text: str
    stop_reason: str


# What repeated calls share, before the call's own prompt: one part, or
# parts ordered from the most stable (shared by the most calls) to the least.
ModelContext = str | tuple[str, ...]
# Cache points the adapter places in a call's messages; parts past it join
# the last. One: the calls that repeat a context (a section's check rounds)
# repeat all of it, and a section's earlier evidence grows from one section
# to the next, so a second point would not be read back.
MAX_CONTEXT_PARTS = 1


class LanguageModel(Protocol):
    """Model calls by role (the adapter maps a role to a model and effort).
    ``context``: see :data:`ModelContext`; empty parts are left out, and a
    call without context writes no message to the cache."""

    async def structured(
        self,
        *,
        role: ModelRole,
        system: str,
        prompt: str,
        schema: type[SchemaT],
        context: ModelContext = "",
    ) -> SchemaT: ...

    async def text(
        self, *, role: ModelRole, system: str, prompt: str, context: ModelContext = ""
    ) -> str: ...

    async def run_agent(
        self,
        *,
        role: ModelRole,
        system: str,
        prompt: str,
        tools: list[Callable[..., str]],
        limits: AgentLimits,
    ) -> AgentOutcome: ...


class DocumentSource(Protocol):
    """A survey paper's text, as complete as can be had (full text, else PDF
    text, else the abstract); None when there is none."""

    async def document(self, paper: PlanPaper) -> SourceDocument | None: ...


class TextStore(Protocol):
    """A cache of text blobs by key (local files, S3)."""

    def get(self, key: str) -> str | None: ...

    def put(self, key: str, text: str) -> None: ...


class CheckpointStore(Protocol):
    """Saved stage outputs of the post engine, by a hash of each stage's
    inputs (docs/engine.md, 4.9). The adapter is a :class:`TextStore` (the
    reading cache's local directory and S3 prefix): the engine keeps its
    checkpoints under its ``checkpoints/`` area. ``get`` is None for a key
    never saved; a ``put`` that could not be kept anywhere raises."""

    def get(self, key: str) -> str | None: ...

    def put(self, key: str, text: str) -> None: ...


# ---- the run's ports ---------------------------------------------------------


class GithubSettings(Protocol):
    """The blog settings a use case reads (``resources.github``)."""

    @property
    def enabled(self) -> bool: ...

    @property
    def review_category(self) -> str: ...

    @property
    def summary_category(self) -> str: ...

    @property
    def tech_guide_category(self) -> str: ...

    @property
    def survey_category(self) -> str: ...


class ResourceSettings(Protocol):
    """Where a run's artifacts go (``resources``)."""

    @property
    def github(self) -> GithubSettings: ...

    @property
    def s3_prefix(self) -> str: ...

    @property
    def s3_bucket_name(self) -> str | None: ...


class RunSettings(Protocol):
    """The settings every use case reads; the composition root builds each
    command's adapters from the rest of the configuration."""

    @property
    def output_language(self) -> str: ...

    @property
    def resources(self) -> ResourceSettings: ...


class UsageTracker(Protocol):
    """The run's model-call accounting: every call recorded and priced."""

    @property
    def input_tokens(self) -> int: ...

    @property
    def output_tokens(self) -> int: ...

    @property
    def total_tokens(self) -> int: ...

    @property
    def call_count(self) -> int: ...

    @property
    def cost_usd(self) -> float: ...

    def record(self, event: UsageEvent) -> None: ...

    def run_usage(self) -> RunUsage: ...

    def estimated_cost_usd(self) -> float: ...


class ObjectStore(Protocol):
    """The run's bucket (S3): text and bytes by key."""

    @property
    def bucket_name(self) -> str: ...

    def get_text(self, key: str) -> str | None: ...

    def put_text(self, key: str, text: str) -> None: ...

    def get_bytes(self, key: str) -> bytes | None: ...


class FileLocationError(ValueError):
    """A location :class:`Files` does not serve (another bucket, a bad URI,
    a write it does not support)."""


class Files(Protocol):
    """Text and bytes at a location: a local path or ``s3://<bucket>/<key>``
    in the run's bucket. Reads give None for a missing file; text is UTF-8
    (``UnicodeDecodeError`` otherwise); a location outside what the adapter
    serves raises :class:`FileLocationError`. Writes create the parent
    folders."""

    def read_text(self, location: str) -> str | None: ...

    def read_bytes(self, location: str) -> bytes | None: ...

    def write_text(self, location: str, text: str) -> None: ...

    def write_bytes(self, location: str, data: bytes) -> None: ...

    def modified(self, location: str) -> datetime | None:
        """When the file was last written (None: missing or unknown)."""
        ...

    def make_dir(self, location: str) -> None:
        """A local folder (and its parents); nothing for a bucket prefix."""
        ...


class BlogSource(Protocol):
    """The blog's post conventions and published posts."""

    def conventions(self, *, fallback: bool = True) -> PostConventions:
        """The blog's current conventions; with ``fallback`` the bundled copy
        when the blog cannot be read, else the error propagates."""
        ...

    def posts(self) -> list[BlogPost]:
        """The blog's posts (none when no blog is configured); raises when
        they cannot be listed."""
        ...


class PostPublisher(Protocol):
    """Saves a post, uploads it and opens its blog pull request."""

    def find_published(
        self, slug: str, branch_id: str, category: str
    ) -> BlogPost | None: ...

    async def publish(self, request: PublishRequest) -> tuple[str | None, Path]: ...

    async def create_pull_request(
        self, request: PublishRequest, markdown_path: Path
    ) -> str: ...


@dataclass(frozen=True)
class RoleSpec:
    """The model a role of the evidence engine runs on: its id, effort and
    thinking switch, and the role it falls back to when the model keeps
    refusing (Bedrock's content filter)."""

    model_id: str
    effort: str
    thinking: bool
    fallback: RoleSpec | None = None


class EvidenceModels(Protocol):
    """Builds the evidence engine's :class:`LanguageModel` for a run: each
    role on its model, every call recorded on ``tracker`` and the run stopped
    once its estimated cost passes ``max_cost_usd``."""

    def language_model(
        self,
        roles: Mapping[ModelRole, RoleSpec],
        tracker: UsageTracker,
        max_cost_usd: float | None,
    ) -> LanguageModel: ...


class FigureCheck(Protocol):
    """Which figure URLs cannot be downloaded (and so cannot be published)."""

    async def unservable(self, urls: list[str]) -> list[str]: ...


class PackageIndex(Protocol):
    """A package index (PyPI): releases and their code, read and never run.
    ``release`` is the run's target release (the latest when None); a
    series ("0.6") is its newest final release.

    Both raise :class:`~scholar_lens.domain.sources.ReleaseUnavailable`
    when the index knows the package but not ``release`` (or, for the
    code, has no file of it that can be read)."""

    def extras(
        self, name: str, release: str | None = None
    ) -> tuple[str, frozenset[str]] | None:
        """(version, extras) of ``name`` at ``release``; None when unknown or
        unreachable."""
        ...

    def source(self, name: str, release: str | None = None) -> PackageSource | None:
        """The Python code of ``name`` at ``release``; None when it cannot be
        read."""
        ...


# ---- the guide's ports ---------------------------------------------------------


class WebResearch(Protocol):
    """Web pages and search results (the run's researcher)."""

    @property
    def supports_search(self) -> bool: ...

    def search_notes(self) -> list[str]: ...

    def close(self) -> None: ...


class GuideResearch(Protocol):
    """Gathers a guide's sources and files the finished guide."""

    @property
    def notes(self) -> list[str]: ...

    @property
    def products(self) -> tuple[str, ...]: ...

    async def gather(
        self,
        urls: list[str],
        *,
        reference_urls: list[str],
        discover_subpages: bool,
        search_queries: list[str] | None,
        snapshots: Any = None,
    ) -> GuideMaterial:
        """The guide's material; ``snapshots`` (:meth:`GuideTools.snapshots`)
        reuses a corpus gathered from the same inputs."""
        ...

    async def classify(
        self,
        guide: TechGuide,
        *,
        category_options: list[str],
        controlled_tags: list[str],
    ) -> dict[str, str]: ...


@dataclass(frozen=True)
class GuideWriting:
    """How the evidence engine writes a guide (from the configuration)."""

    roles: Mapping[ModelRole, RoleSpec]
    max_cost_usd: float | None
    reader_tag: str
    max_sections: int
    max_sources: int
    min_total_chars: int
    max_total_chars: int


class GuideTools(Protocol):
    """What the guide use case builds its run from."""

    @property
    def writing(self) -> GuideWriting: ...

    @property
    def models(self) -> EvidenceModels: ...

    @property
    def packages(self) -> PackageIndex: ...

    @property
    def figures(self) -> FigureCheck: ...

    def researcher(self) -> WebResearch:
        """A new researcher (the use case closes it)."""
        ...

    def research(
        self, researcher: WebResearch, tracker: UsageTracker, *, title_max_length: int
    ) -> GuideResearch:
        """The run's source gathering on ``researcher``."""
        ...

    def snapshots(self, *, fresh: bool) -> Any:
        """The saved corpora a run reuses (written but not read when
        ``fresh``)."""
        ...

    def evidence_store(self, *, fresh: bool) -> TextStore:
        """The reading cache and checkpoint store of the guide engine."""
        ...


# ---- the paper review and summary's ports ---------------------------------------


@dataclass(frozen=True)
class PaperInputs:
    """The paid inputs the selected writer reads beyond the paper, its
    attributes and the codebase summary (which every paper post uses)."""

    # Citations and their prerequisite summaries.
    citations: bool
    # The searchable code index (one model call per chunk, plus embeddings).
    code_index: bool


class PaperOrigin(Protocol):
    """Where a paper is read from (an arXiv id or a PDF URL)."""

    @property
    def source_id(self) -> str: ...

    @property
    def arxiv_html_id(self) -> str | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PaperWriting:
    """How the evidence engine writes a review or summary (from the
    configuration)."""

    roles: Mapping[ModelRole, RoleSpec]
    max_cost_usd: float | None
    reader_tag: str
    max_sections: int
    chars_per_reader: int
    min_total_chars: int
    max_total_chars: int


class SectionsWriter(Protocol):
    """The legacy sections engine's review and summary writers. ``material``
    is what :meth:`PaperTools.read` returned besides the paper (the
    citation summarizer and code index)."""

    async def review(
        self,
        paper: Paper,
        *,
        citation_summarizer: Any,
        code_retriever: Any,
        translation_guideline: list[dict[str, Any]] | None,
        tracker: UsageTracker,
    ) -> tuple[ReviewText, list[str]]: ...

    async def summary(
        self,
        paper: Paper,
        *,
        translation_guideline: list[dict[str, Any]] | None,
        tracker: UsageTracker,
    ) -> tuple[dict[str, str], list[str]]: ...


class PaperTools(Protocol):
    """What the paper review and summary use case builds its run from."""

    @property
    def models(self) -> EvidenceModels: ...

    @property
    def figures(self) -> FigureCheck: ...

    def engine(
        self, post: Literal["review", "summary"]
    ) -> Literal["evidence", "sections"]:
        """The engine that writes ``post``."""
        ...

    def inputs(self, kind: str) -> PaperInputs:
        """What the writer chosen for ``kind`` consumes."""
        ...

    def resolve(self, source: str) -> PaperOrigin:
        """The paper ``source`` names (an arXiv id or a PDF URL)."""
        ...

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
        """The parsed paper, and the sections engine's code index and
        citation summarizer (None unless ``inputs`` asks for them)."""
        ...

    def writing(self, post: Literal["review", "summary"]) -> PaperWriting: ...

    def evidence_store(self, *, fresh: bool) -> TextStore: ...

    def sections(self) -> SectionsWriter: ...


# ---- the survey's ports ----------------------------------------------------------


class SurveyPlanning(Protocol):
    """The plan stage's model calls: the scope, then the selection."""

    async def scope(
        self,
        topic: str,
        time_range: TimeRange,
        required: list[Candidate],
        references: Sequence[ReadReference] = (),
    ) -> SurveyScope: ...

    async def select(
        self,
        topic: str,
        scope: SurveyScope,
        pool: CandidatePool,
        shortlist_: list[Candidate],
        *,
        min_papers: int,
        max_papers: int,
        today: date,
    ) -> SurveySelection: ...


class CandidateSearch(Protocol):
    """The plan stage's candidate step on the scholarly indexes (blocking)."""

    def resolve_required(self, paper_ids: Sequence[str]) -> list[SourceRecord]: ...

    def read_references(self, urls: Sequence[str]) -> list[ReadReference]: ...

    def referenced_records(
        self, references: Sequence[ReadReference]
    ) -> list[SourceRecord]: ...

    def gather(
        self,
        scope: SurveyScope,
        known: list[SourceRecord],
        *,
        time_range: TimeRange,
        today: date,
    ) -> CandidatePool: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SurveyLimits:
    """How many papers a plan selects, from how many candidates."""

    min_papers: int
    max_papers: int
    candidate_pool: int


class SurveyPlanTools(Protocol):
    """What the survey plan use case builds its run from."""

    @property
    def limits(self) -> SurveyLimits: ...

    def planner(self, tracker: UsageTracker) -> SurveyPlanning: ...

    def finder(self) -> CandidateSearch:
        """The candidate step on a new researcher (the use case closes it)."""
        ...


class PlanIndex(Protocol):
    """Bibliographic records and abstracts of plan papers (blocking)."""

    def records(self, paper_ids: Sequence[str]) -> dict[str, SourceRecord]: ...


class SurveyDocuments(DocumentSource, Protocol):
    """A survey paper's text, and notes on papers read from less than their
    full text (known once they are read)."""

    def notes(self) -> list[str]: ...


class SurveyFinalizer(Protocol):
    """The write stage's finalization call: the finished post's topic, tags
    and search description."""

    @property
    def language(self) -> str: ...

    @property
    def notes(self) -> list[str]: ...

    async def finalize(
        self,
        title: str,
        body: str,
        *,
        category_options: Sequence[str],
        controlled_tags: Sequence[str],
    ) -> dict[str, str]: ...


class ReferenceReader(Protocol):
    """Reads the plan's reference articles (blocking), on a researcher the
    use case closes."""

    def read_references(self, urls: Sequence[str]) -> list[ReadReference]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SurveyWriting:
    """How the evidence engine writes a survey (from the configuration)."""

    roles: Mapping[ModelRole, RoleSpec]
    max_cost_usd: float | None
    reader_tag: str
    chars_per_core: int
    chars_per_supporting: int
    min_total_chars: int
    max_total_chars: int


class SurveyWriteTools(Protocol):
    """What the survey write use case builds its run from."""

    @property
    def writing(self) -> SurveyWriting: ...

    @property
    def models(self) -> EvidenceModels: ...

    @property
    def figures(self) -> FigureCheck: ...

    def index(self) -> PlanIndex: ...

    def finalizer(self, tracker: UsageTracker) -> SurveyFinalizer: ...

    def reader(self) -> ReferenceReader: ...

    def documents(
        self, reader: ReferenceReader, index: PlanIndex, abstracts: dict[str, str]
    ) -> SurveyDocuments: ...

    def evidence_store(self, *, fresh: bool) -> TextStore: ...
