"""Paper-survey plan stage: scope → candidates → selection → plan file.

``survey plan`` writes an editable plan (``papers/survey-<slug>/plan.json``
and its S3 copy); the write stage (``survey write --plan``,
``survey_writer``) reads it. The scope and selection calls are the only model
calls; the candidate step is deterministic API work (``scholarly_index``) and
every check on the models' answers is code (``survey_plan``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TypeVar

import boto3
import httpx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from .arxiv_handler import ArxivHandler, ArxivPaperError
from .aws_helpers import S3Handler
from .citation_metadata import QuotaExhaustedError, titles_match
from .constants import LanguageModelId, LocalPaths, S3Paths
from .jobs import JobRequest
from .logger import logger
from .metrics import TokenBudgetGuard, TokenUsageTracker
from .prompt_safety import neutralize_prompt_tags
from .prompts import BasePrompt, SurveyScopePrompt, SurveySelectionPrompt
from .publish_types import slugify
from .runtime import ROOT_DIR, PipelineOutcome, RunContext
from .scholarly_index import (
    DATACITE_MAX_IDS,
    DataCiteIndex,
    OpenAlexIndex,
    SemanticScholarIndex,
    arxiv_record,
    datacite_record,
    openalex_record,
)
from .survey_plan import (
    Candidate,
    SourceRecord,
    SurveyInputError,
    SurveyPlan,
    SurveyPlanError,
    SurveyScope,
    SurveySelection,
    TimeRange,
    add_citation_data,
    build_plan,
    check_selection,
    citation_edges,
    in_time_range,
    merge_records,
    parse_paper_id,
    parse_tagged_json,
    rank_candidates,
    semantic_scholar_id,
    shortlist,
    shortlist_digest,
    unversioned_arxiv_id,
)
from .utils import BedrockLanguageModelFactory, RetryableBase

M = TypeVar("M")

# Results kept per search query and provider, and per landmark title.
SEARCH_RESULTS = 25
LANDMARK_RESULTS = 5
# Past max_total_tokens the repair round is skipped; past this multiple the
# run aborts before the next call (the guide pipeline's two tiers).
HARD_BUDGET_MULTIPLE = 1.5
_UNDERCOUNT_NOTE = (
    "Citation counts are OpenAlex's, which undercount arXiv-era ML papers and "
    "list few of their references (so few citation edges); set "
    "SEMANTIC_SCHOLAR_API_KEY for fuller counts and edges."
)


class CandidateSearchError(RuntimeError):
    """Every topic search, or every arXiv metadata lookup, failed (outage or
    spent quota): nothing reliable to plan from. The run fails; try again."""


# ---- candidate step (deterministic, network) -------------------------------


@dataclass
class _Calls:
    """Runs index calls, counting requests and failures per provider. A
    transport or HTTP error, or a response that is not JSON, is a failed
    call the step can do without; a provider whose quota is spent, or that
    failed twice in a row (arXiv's 429s last minutes), is skipped for the
    rest of the run; any other error propagates."""

    made: Counter[str] = field(default_factory=Counter)
    failed: Counter[str] = field(default_factory=Counter)
    failing: Counter[str] = field(default_factory=Counter)
    skipped: set[str] = field(default_factory=set)
    # Calls that returned, over all providers.
    answered: int = 0

    def run(self, provider: str, call: Callable[[], list[M]]) -> list[M]:
        return self.attempt(provider, call) or []

    def attempt(self, provider: str, call: Callable[[], list[M]]) -> list[M] | None:
        """``call``'s result; None when it failed or the provider is skipped."""
        if provider in self.skipped:
            return None
        self.made[provider] += 1
        try:
            result = call()
        except QuotaExhaustedError as e:
            self._fail(provider, e, skip=True)
        except (httpx.HTTPError, json.JSONDecodeError, ArxivPaperError) as e:
            self._fail(provider, e, skip=self.failing[provider] + 1 >= _MAX_FAILURES)
        else:
            self.failing[provider] = 0
            self.answered += 1
            return result
        return None

    def required(self, provider: str, call: Callable[[], list[M]]) -> list[M]:
        """A call the run cannot do without: counted, never caught."""
        self.made[provider] += 1
        return call()

    def _fail(self, provider: str, error: Exception, *, skip: bool) -> None:
        self.failed[provider] += 1
        self.failing[provider] += 1
        logger.warning("%s call failed: %s", provider, error)
        if skip:
            self.skipped.add(provider)
            logger.warning("Skipping %s for the rest of this run.", provider)


# Consecutive failures after which a provider is skipped for the run.
_MAX_FAILURES = 2


@dataclass
class CandidatePool:
    """Candidates in rank order with the citation edges among them."""

    candidates: list[Candidate]
    edges: list[tuple[str, str]]
    sources: list[str]
    notes: list[str]


class CandidateFinder:
    """Gathers survey candidates from OpenAlex and arXiv search, with
    arXiv's metadata from DataCite (and Semantic Scholar's citation data with
    a key). Blocking: run it in a thread."""

    def __init__(
        self,
        *,
        openalex: OpenAlexIndex,
        arxiv: ArxivHandler,
        datacite: DataCiteIndex,
        semantic_scholar: SemanticScholarIndex,
    ) -> None:
        self.openalex = openalex
        self.arxiv = arxiv
        self.datacite = datacite
        self.semantic_scholar = semantic_scholar
        self.calls = _Calls()
        # arXiv ids OpenAlex gave that have no arXiv record (DataCite): the
        # id is wrong or its lookup failed. Their records are left out.
        self.unverified: set[str] = set()
        # Candidates with neither an arXiv id nor a DOI (an OpenAlex work
        # only): the write stage cannot resolve them, so they are left out.
        self.unresolvable = 0

    def resolve_required(self, paper_ids: Sequence[str]) -> list[SourceRecord]:
        """Records of the papers the user requires: arXiv's metadata (from
        DataCite) for an arXiv id, the OpenAlex record for a DOI, plus the
        OpenAlex records of the same paper. ``SurveyInputError`` when the
        index does not know the id; an unreachable index raises its error."""
        records: list[SourceRecord] = []
        for paper_id in dict.fromkeys(parse_paper_id(p) for p in paper_ids):
            if unversioned_arxiv_id(paper_id):
                items = self.calls.required(
                    "DataCite", lambda i=paper_id: self.datacite.arxiv_papers([i])
                )
                found = [
                    r for item in items if (r := datacite_record(item, "required"))
                ]
                if not found:
                    raise SurveyInputError(f"No arXiv paper {paper_id!r}.")
                found += self._openalex_records(
                    lambda i=paper_id: self.openalex.by_doi(f"10.48550/arxiv.{i}"),
                    "required",
                )
            else:
                works = self.calls.required(
                    "OpenAlex", lambda i=paper_id: self.openalex.by_doi(i)
                )
                found = [r for w in works if (r := openalex_record(w, "required"))]
                if not found:
                    raise SurveyInputError(f"No paper found for DOI {paper_id!r}.")
            title = found[0].title
            found += [
                r
                for r in self._openalex_records(
                    lambda t=title: self.openalex.search_title(t), "required"
                )
                if titles_match(title, r.title)
            ]
            records += found
        return records

    def gather(
        self,
        scope: SurveyScope,
        required: list[SourceRecord],
        *,
        time_range: TimeRange,
        today: date,
    ) -> CandidatePool:
        records = list(required)
        answered = self.calls.answered
        for query in scope.queries:
            label = f"query:{query}"
            records += self._openalex_records(
                lambda q=query: self.openalex.search(
                    q,
                    per_page=SEARCH_RESULTS,
                    since=time_range.since,
                    until=time_range.until,
                ),
                label,
            )
            records += self._arxiv_records(
                lambda q=query: self.arxiv.search(_arxiv_query(q), SEARCH_RESULTS),
                label,
            )
        for title in scope.landmarks:
            records += self._landmark(title)
        if self.calls.answered == answered:
            raise CandidateSearchError(
                "Every paper search failed (OpenAlex and arXiv); try again later."
            )
        records = self._verify_with_arxiv(records)
        candidates = self._pool(records, time_range)
        if not candidates:
            raise SurveyInputError(
                f"No papers found for the topic {scope.topic!r} in the time range."
            )
        enriched = self._add_semantic_scholar(candidates)
        edges = citation_edges(candidates)
        ranked = rank_candidates(candidates, edges, today=today)
        return CandidatePool(
            candidates=ranked,
            edges=edges,
            sources=self._sources(enriched),
            notes=self._notes(enriched),
        )

    def _openalex_records(
        self, call: Callable[[], list[dict[str, Any]]], found_by: str
    ) -> list[SourceRecord]:
        works = self.calls.run("OpenAlex", call)
        return [r for w in works if (r := openalex_record(w, found_by))]

    def _arxiv_records(
        self, call: Callable[[], list[Any]], found_by: str | None
    ) -> list[SourceRecord]:
        return [arxiv_record(m, found_by) for m in self.calls.run("arXiv", call)]

    def _landmark(self, title: str) -> list[SourceRecord]:
        """Records whose title is the landmark title: OpenAlex first; arXiv
        when OpenAlex has no matching record with an arXiv id."""
        label = f"landmark:{title}"
        found = [
            r
            for r in self._openalex_records(
                lambda: self.openalex.search_title(title, per_page=LANDMARK_RESULTS),
                label,
            )
            if titles_match(title, r.title)
        ]
        if not any(r.arxiv_id for r in found):
            found += [
                r
                for r in self._arxiv_records(
                    lambda: self.arxiv.search(
                        f'ti:"{_quote_free(title)}"', LANDMARK_RESULTS
                    ),
                    label,
                )
                if titles_match(title, r.title)
            ]
        return found

    def _verify_with_arxiv(self, records: list[SourceRecord]) -> list[SourceRecord]:
        """``records`` plus arXiv's own records (DataCite) of the arXiv ids
        only OpenAlex returned: arXiv's title, authors, abstract and
        first-version date are the paper's (OpenAlex files some arXiv ids,
        PPO's among them, under another work's title). A record whose arXiv
        id gets no arXiv record (a wrong id, or its lookup failed) is left
        out rather than trusted; a required one only loses the arXiv id.
        ``CandidateSearchError`` when every lookup failed."""
        have = {r.arxiv_id for r in records if r.provider == "arxiv"}
        missing = sorted(
            {r.arxiv_id for r in records if r.arxiv_id and r.arxiv_id not in have}
        )
        found: list[SourceRecord] = []
        answered = False
        for start in range(0, len(missing), DATACITE_MAX_IDS):
            chunk = missing[start : start + DATACITE_MAX_IDS]
            items = self.calls.attempt(
                "DataCite", lambda c=chunk: self.datacite.arxiv_papers(c)
            )
            answered |= items is not None
            found += [r for item in items or [] if (r := datacite_record(item, None))]
        if missing and not answered:
            raise CandidateSearchError(
                "Every arXiv metadata lookup (DataCite) failed; try again later."
            )
        verified = have | {r.arxiv_id for r in found}
        unverified = set(missing) - verified
        self.unverified |= unverified
        kept = []
        for record in records:
            if record.arxiv_id not in unverified:
                kept.append(record)
            elif "required" in record.found_by:
                kept.append(replace(record, arxiv_id=None))
        return kept + found

    def _pool(
        self, records: list[SourceRecord], time_range: TimeRange
    ) -> list[Candidate]:
        candidates = merge_records(records)
        for candidate in candidates:
            candidate.required = "required" in candidate.found_by
        resolvable = [c for c in candidates if c.arxiv_id or c.doi]
        self.unresolvable = len(candidates) - len(resolvable)
        return [
            c
            for c in resolvable
            if in_time_range(c, time_range.since, time_range.until)
        ]

    def _add_semantic_scholar(self, candidates: list[Candidate]) -> int:
        """Semantic Scholar counts and references (with a key); the number of
        candidates it knew."""
        if not self.semantic_scholar.enabled:
            return 0
        targets = [(c, i) for c in candidates if (i := semantic_scholar_id(c))]
        found = self.calls.run(
            "Semantic Scholar",
            lambda: self.semantic_scholar.papers([i for _, i in targets]),
        )
        for (candidate, _), data in zip(targets, found, strict=False):
            add_citation_data(candidate, data)
        return sum(1 for data in found if data is not None)

    def _sources(self, enriched: int) -> list[str]:
        sources = [
            f"{provider}: {n} requests ({self.calls.failed[provider]} failed)"
            for provider, n in sorted(self.calls.made.items())
        ]
        if enriched:
            sources.append(f"Semantic Scholar citation data for {enriched} candidates")
        return sources

    def _notes(self, enriched: int) -> list[str]:
        notes = []
        if not enriched:
            notes.append(_UNDERCOUNT_NOTE)
        if self.unverified:
            notes.append(
                f"{len(self.unverified)} arXiv id(s) from OpenAlex have no arXiv "
                "record at DataCite (a wrong id, or a failed lookup); those "
                "records were left out: " + ", ".join(sorted(self.unverified))
            )
        if self.unresolvable:
            notes.append(
                f"{self.unresolvable} candidate(s) with neither an arXiv id nor a "
                "DOI were left out (the write stage cannot resolve them)."
            )
        for provider in sorted(self.calls.failed):
            if failed := self.calls.failed[provider]:
                notes.append(
                    f"{failed} of {self.calls.made[provider]} {provider} requests "
                    "failed; the pool may miss papers."
                )
        return notes


def _quote_free(text: str) -> str:
    return " ".join(text.replace('"', " ").split())


def _arxiv_query(query: str) -> str:
    """arXiv API query matching every word of ``query`` anywhere."""
    return " AND ".join(f"all:{word}" for word in _quote_free(query).split())


# ---- model calls ---------------------------------------------------------------


class SurveyCalls(RetryableBase, TokenBudgetGuard):
    """Model calls answering with JSON in a tag, parsed and checked in code
    with one repair round, within the run's two-tier token budget (shared
    by the plan and write stages)."""

    async def _ask(
        self,
        chain: Runnable,
        payload: dict[str, Any],
        parse: Callable[[str], tuple[M, list[str]]],
        *,
        tag: str,
        accept_last: bool = False,
    ) -> M:
        """Call, parse and check; once more with the problems listed (within
        the soft token limit); then fail with ``SurveyPlanError``. With
        ``accept_last`` the last answer that parses is returned despite its
        problems (the caller removes what they concern)."""
        payload = {**payload, "repair_note": ""}
        problems: list[str] = []
        for attempt in range(2):
            raw = await self._invoke(chain, payload)
            try:
                result, problems = parse(raw)
            except SurveyPlanError as e:
                problems = [str(e)[:1000]]
            else:
                last = attempt == 1 or self._over_soft_budget()
                if not problems or (accept_last and last):
                    return result
            logger.warning(
                "Survey %s unusable (attempt %d): %s", tag, attempt + 1, problems
            )
            if self._over_soft_budget():
                break
            payload["repair_note"] = (
                "\nYour previous answer could not be used:\n"
                + "\n".join(f"- {p}" for p in problems)
                + f"\nReturn the complete answer again as valid JSON inside <{tag}> "
                "tags, fixing every problem."
            )
        raise SurveyPlanError(
            f"The survey {tag} call gave no usable answer: {problems}"
        )

    @RetryableBase._retry("survey_call")
    async def _invoke(self, chain: Runnable, payload: dict[str, Any]) -> str:
        if self._token_tracker is not None and self.max_total_tokens:
            self._token_tracker.check_budget(
                int(self.max_total_tokens * HARD_BUDGET_MULTIPLE)
            )
        raw: str = await chain.ainvoke(payload)
        return raw

    def _over_soft_budget(self) -> bool:
        return bool(
            self._token_tracker is not None
            and self.max_total_tokens
            and self._token_tracker.total_tokens >= self.max_total_tokens
        )


class SurveyPlanner(SurveyCalls):
    """The scope and selection calls, each parsed and checked in code with
    one repair round."""

    def __init__(
        self,
        *,
        scope_model_id: LanguageModelId,
        selection_model_id: LanguageModelId,
        boto_session: boto3.Session,
        thinking_effort: str,
        max_total_tokens: int | None,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.callbacks = callbacks or []
        self._init_token_budget(self.callbacks, max_total_tokens)
        factory = BedrockLanguageModelFactory(boto_session=boto_session)

        def chain(
            prompt: type[BasePrompt], model_id: LanguageModelId, *, thinking: bool
        ) -> Runnable:
            model = factory.get_model(
                model_id,
                temperature=0.0,
                callbacks=self.callbacks or None,
                enable_thinking=thinking,
                thinking_effort=thinking_effort,
            )
            return prompt.get_prompt() | model | StrOutputParser()

        self.scope_chain = chain(SurveyScopePrompt, scope_model_id, thinking=False)
        # Choosing and grouping the papers is the stage's main judgment.
        self.selection_chain = chain(
            SurveySelectionPrompt, selection_model_id, thinking=True
        )

    async def scope(
        self, topic: str, time_range: TimeRange, required: list[Candidate]
    ) -> SurveyScope:
        payload = {
            "topic": neutralize_prompt_tags(topic),
            "time_range": _range_text(time_range),
            "required_papers": neutralize_prompt_tags(
                "\n".join(f"- {c.title} ({c.year or 'year unknown'})" for c in required)
                or "- none"
            ),
        }

        def parse(raw: str) -> tuple[SurveyScope, list[str]]:
            return parse_tagged_json(raw, "scope", SurveyScope), []

        return await self._ask(self.scope_chain, payload, parse, tag="scope")

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
    ) -> SurveySelection:
        payload = {
            "topic": neutralize_prompt_tags(topic),
            "scope": scope.topic,
            "lineages": "\n".join(
                f"- {lineage.id}: {lineage.name} — {lineage.description}"
                for lineage in scope.lineages
            ),
            "candidates": neutralize_prompt_tags(
                shortlist_digest(shortlist_, pool.edges, today)
            ),
            "min_papers": min(min_papers, len(shortlist_)),
            "max_papers": max_papers,
        }

        def parse(raw: str) -> tuple[SurveySelection, list[str]]:
            selection = parse_tagged_json(raw, "selection", SurveySelection)
            problems = check_selection(
                selection, shortlist_, min_papers=min_papers, max_papers=max_papers
            )
            return selection, problems

        return await self._ask(self.selection_chain, payload, parse, tag="selection")


def _range_text(time_range: TimeRange) -> str:
    if time_range.since is None and time_range.until is None:
        return "none"
    return f"{time_range.since or 'open'} to {time_range.until or 'open'}"


# ---- the stage -----------------------------------------------------------------


async def plan_survey(
    request: JobRequest,
    *,
    planner: SurveyPlanner,
    finder: CandidateFinder,
    min_papers: int,
    max_papers: int,
    candidate_pool: int,
    now: datetime,
) -> SurveyPlan:
    """Scope the topic, gather and rank candidates, select the papers and
    build the plan."""
    topic = (request.topic or "").strip()
    user_range = TimeRange(since=request.since_year, until=request.until_year)
    required_records = await asyncio.to_thread(
        finder.resolve_required, request.paper_ids
    )
    required = merge_records(required_records)
    scope = await planner.scope(topic, user_range, required)
    pool = await asyncio.to_thread(
        finder.gather, scope, required_records, time_range=user_range, today=now.date()
    )
    picks = shortlist(pool.candidates, candidate_pool)
    logger.info(
        "Survey pool: %d candidates, %d citation edges; shortlist of %d.",
        len(pool.candidates),
        len(pool.edges),
        len(picks),
    )
    selection = await planner.select(
        topic,
        scope,
        pool,
        picks,
        min_papers=min_papers,
        max_papers=max_papers,
        today=now.date(),
    )
    return build_plan(
        topic=topic,
        scope=scope,
        selection=selection,
        pool=picks,
        edges=pool.edges,
        time_range=user_range,
        sources=pool.sources,
        notes=pool.notes,
        created_at=now,
    )


def save_plan(plan: SurveyPlan, path: Path) -> None:
    """Write ``plan`` to ``path``; a different plan already there (perhaps
    edited by hand) is kept next to it as ``plan.<UTC timestamp>.json``."""
    text = plan.to_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        path.rename(path.with_name(f"{path.stem}.{stamp}{path.suffix}"))
    path.write_text(text, encoding="utf-8")


def upload_plan(plan: SurveyPlan, s3_handler: S3Handler, key: str) -> None:
    """Upload ``plan`` to ``key``; a different plan already there (perhaps
    edited and staged by hand) is kept next to it as
    ``plan-<sha256 prefix>.json``, the name ``survey write --remote`` gives an
    uploaded plan, so the same content never takes two keys."""
    text = plan.to_json()
    old = s3_handler.get_text(key)
    if old is not None and old != text:
        digest = hashlib.sha256(old.encode()).hexdigest()[:16]
        s3_handler.put_text(key.rsplit("/", 1)[0] + f"/plan-{digest}.json", old)
    s3_handler.put_text(key, text)


def survey_slug(topic: str) -> str:
    """The plan's folder name; the user's topic, so a rerun of the same
    request lands in the same place. A topic with non-ASCII letters (Korean)
    gets a hash suffix: the slug drops those letters, so "LLM 선호 최적화" and
    "LLM 추론 최적화" would otherwise share one folder, branch and post."""
    slug = f"survey-{slugify(topic, max_length=80)}"
    if any(not ch.isascii() and ch.isalpha() for ch in topic):
        digest = hashlib.sha1(topic.strip().encode("utf-8")).hexdigest()[:8]
        if not slug.endswith(digest):  # slugify's own suffix (all non-ASCII)
            slug = f"{slug}-{digest}"
    return slug


async def run_survey_plan_pipeline(
    context: RunContext, request: JobRequest, tracker: TokenUsageTracker
) -> PipelineOutcome:
    cfg = context.config.survey
    planner = SurveyPlanner(
        scope_model_id=LanguageModelId(cfg.scope_model_id),
        selection_model_id=LanguageModelId(cfg.selection_model_id),
        boto_session=context.bedrock_boto_session,
        thinking_effort=cfg.thinking_effort,
        max_total_tokens=cfg.max_total_tokens,
        callbacks=[tracker],
    )
    finder = CandidateFinder(
        openalex=OpenAlexIndex(),
        arxiv=ArxivHandler(),
        datacite=DataCiteIndex(),
        semantic_scholar=SemanticScholarIndex(),
    )
    plan = await plan_survey(
        request,
        planner=planner,
        finder=finder,
        min_papers=cfg.min_papers,
        max_papers=cfg.max_papers,
        candidate_pool=cfg.candidate_pool,
        now=datetime.now(UTC),
    )
    slug = survey_slug(plan.topic)
    path = ROOT_DIR / LocalPaths.PAPERS_DIR.value / slug / "plan.json"
    await asyncio.to_thread(save_plan, plan, path)
    s3_uri = None
    if context.s3_handler is not None:
        resources = context.config.resources
        key = f"{resources.s3_prefix}/{S3Paths.SURVEYS.value}/{slug}/plan.json".lstrip(
            "/"
        )
        await asyncio.to_thread(upload_plan, plan, context.s3_handler, key)
        s3_uri = f"s3://{resources.s3_bucket_name}/{key}"
    logger.info(
        "Survey plan: %d papers in %d lineages -> %s",
        len(plan.papers),
        len(plan.lineages),
        path,
    )
    return PipelineOutcome(
        title=plan.scope,
        plan_path=path,
        s3_uri=s3_uri,
        warnings=list(plan.notes),
    )
