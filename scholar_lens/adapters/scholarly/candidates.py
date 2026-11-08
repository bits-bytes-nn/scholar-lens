"""The survey plan stage's candidate step (deterministic API work): the
required papers resolved, the user's references read, candidates gathered
from OpenAlex and arXiv search with arXiv's metadata from DataCite (and
Semantic Scholar's citation data with a key)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, TypeVar

import httpx

from scholar_lens.adapters.scholarly.arxiv import ArxivHandler, ArxivPaperError
from scholar_lens.adapters.scholarly.index import (
    DATACITE_MAX_IDS,
    OPENALEX_MAX_DOIS,
    DataCiteIndex,
    OpenAlexIndex,
    SemanticScholarIndex,
    arxiv_record,
    datacite_record,
    openalex_record,
)
from scholar_lens.adapters.scholarly.metadata import ProviderUnavailableError
from scholar_lens.adapters.web.research import WebResearcher
from scholar_lens.domain.research import (
    ResearchCorpus,
    SourceFetchError,
    page_key,
)
from scholar_lens.domain.survey.plan import (
    Candidate,
    SourceRecord,
    SurveyInputError,
    SurveyScope,
    TimeRange,
    add_citation_data,
    citation_edges,
    in_time_range,
    merge_records,
    paper_id_of,
    parse_paper_id,
    rank_candidates,
    reference_kind,
    semantic_scholar_id,
    unversioned_arxiv_id,
)
from scholar_lens.domain.survey.references import CandidatePool, ReadReference
from scholar_lens.domain.titles import titles_match

logger = logging.getLogger(__name__)


M = TypeVar("M")


# Results kept per search query and provider, and per landmark title.
SEARCH_RESULTS = 25


LANDMARK_RESULTS = 5


_UNDERCOUNT_NOTE = (
    "Citation counts are OpenAlex's, which undercount arXiv-era ML papers and "
    "list few of their references (so few citation edges); set "
    "SEMANTIC_SCHOLAR_API_KEY for fuller counts and edges."
)


class CandidateSearchError(RuntimeError):
    """Every topic search, or every arXiv metadata lookup, failed (outage or
    spent quota): nothing reliable to plan from. The run fails; try again."""


def read_references(
    urls: Sequence[str], researcher: WebResearcher
) -> list[ReadReference]:
    """Each reference read once, without sub-pages (an arXiv link as the
    paper's HTML full text); one spelling per page. A reference that cannot
    be read is kept without its page: the material is optional. Blocking."""
    read: list[ReadReference] = []
    seen: set[str] = set()
    for url in urls:
        # One per paper (an arXiv link and its arXiv DOI) or page.
        if (key := paper_id_of(url) or page_key(url)) in seen:
            continue
        seen.add(key)
        try:
            corpus = researcher.research([url], discover_subpages=False)
        except SourceFetchError as e:
            logger.warning("Could not read the reference %s: %s", url, e)
            corpus = ResearchCorpus()
        page = corpus.pages[0] if corpus.pages else None
        read.append(ReadReference(url, reference_kind(url), page))
    return read


@dataclass
class _Calls:
    """Runs index calls, counting requests and failures per provider. A
    transport or HTTP error, or a response that is not JSON, is a failed
    call the step can do without; a provider whose quota is spent or key is
    rejected, or that
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
        except ProviderUnavailableError as e:
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
        researcher: WebResearcher | None = None,
    ) -> None:
        self.openalex = openalex
        self.arxiv = arxiv
        self.datacite = datacite
        self.semantic_scholar = semantic_scholar
        self.researcher = researcher or WebResearcher()
        self.calls = _Calls()
        # Papers the user's references are or cite, and how many of them an
        # index knew.
        self.referenced_ids = 0
        self.referenced_found = 0
        # arXiv ids OpenAlex gave that have no arXiv record (DataCite): the
        # id is wrong or its lookup failed. Their records are left out.
        self.unverified: set[str] = set()
        # Candidates with neither an arXiv id nor a DOI (an OpenAlex work
        # only): the write stage cannot resolve them, so they are left out.
        self.unresolvable = 0

    def close(self) -> None:
        """Release the researcher's connections."""
        self.researcher.close()

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

    def read_references(self, urls: Sequence[str]) -> list[ReadReference]:
        return read_references(urls, self.researcher)

    def referenced_records(
        self, references: Sequence[ReadReference]
    ) -> list[SourceRecord]:
        """Records of the papers the references are or cite: arXiv's metadata
        (DataCite) for arXiv ids, OpenAlex's for other DOIs, each labelled
        ``reference:<url>`` for every reference that names it. They are
        candidates, not required. A paper no index knows, or whose lookup
        failed, is left out."""
        labels: dict[str, set[str]] = {}
        for reference in references:
            for paper_id in reference.cited_ids():
                labels.setdefault(paper_id, set()).add(f"reference:{reference.url}")
        arxiv_ids = [i for i in labels if unversioned_arxiv_id(i)]
        dois = [i for i in labels if not unversioned_arxiv_id(i)]
        records: list[SourceRecord] = []
        for start in range(0, len(arxiv_ids), DATACITE_MAX_IDS):
            chunk = arxiv_ids[start : start + DATACITE_MAX_IDS]
            items = self.calls.run(
                "DataCite", lambda c=chunk: self.datacite.arxiv_papers(c)
            )
            records += [r for item in items if (r := datacite_record(item, None))]
        for start in range(0, len(dois), OPENALEX_MAX_DOIS):
            chunk = dois[start : start + OPENALEX_MAX_DOIS]
            records += self._openalex_records(
                lambda c=chunk: self.openalex.by_dois(c), "reference"
            )
        found: list[SourceRecord] = []
        for record in records:
            key = next((k for k in (record.arxiv_id, record.doi) if k in labels), None)
            if key is not None:
                found.append(replace(record, found_by=set(labels[key])))
        self.referenced_ids = len(labels)
        self.referenced_found = len(
            {k for r in found for k in (r.arxiv_id, r.doi) if k in labels}
        )
        return found

    def gather(
        self,
        scope: SurveyScope,
        known: list[SourceRecord],
        *,
        time_range: TimeRange,
        today: date,
    ) -> CandidatePool:
        """The ranked pool: ``known`` records (the required and referenced
        papers) plus the topic searches' and landmark titles' results."""
        records = list(known)
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
        if self.referenced_ids:
            sources.append(
                f"References: {self.referenced_ids} papers linked, "
                f"{self.referenced_found} found in the indexes"
            )
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


class ReferencePages:
    """The :class:`~scholar_lens.application.ports.ReferenceReader` on a
    researcher."""

    def __init__(self, researcher: WebResearcher) -> None:
        self.researcher = researcher

    def read_references(self, urls: Sequence[str]) -> list[ReadReference]:
        return read_references(urls, self.researcher)

    def close(self) -> None:
        self.researcher.close()
