"""Paper-index adapters for the survey plan: OpenAlex works, arXiv search,
arXiv's DOI records at DataCite and Semantic Scholar citation data, turned
into :class:`~scholar_lens.src.survey_plan.SourceRecord` values.

HTTP pacing and the single 429 retry come from
``citation_metadata._get_json`` (the one retry layer for these APIs); arXiv
calls go through :class:`~scholar_lens.src.arxiv_handler.ArxivHandler` and
its process-wide limiter. Failures raise (``httpx.HTTPError``,
``QuotaExhaustedError``, ``ArxivPaperError``): the candidate step decides
which ones it can do without.
"""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Any

from . import citation_metadata
from .arxiv_handler import ArxivMetadata
from .constants import EnvVars
from .rate_limiter import RateLimiter
from .survey_plan import (
    CitationData,
    SourceRecord,
    normalize_doi,
    unversioned_arxiv_id,
)

_OPENALEX_URL = "https://api.openalex.org/works"
_SEMANTIC_SCHOLAR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
_DATACITE_URL = "https://api.datacite.org/dois"
# DataCite's public API allows 3,000 requests per 5 minutes per IP.
_DATACITE_LIMITER = RateLimiter(rate=5.0, per=1.0, name="datacite")
# arXiv DOIs looked up per DataCite query (one OR query, one page).
DATACITE_MAX_IDS = 50
_TIMEOUT = 30.0
_OPENALEX_FIELDS = (
    "id,doi,title,authorships,publication_date,primary_location,locations,"
    "cited_by_count,abstract_inverted_index,referenced_works"
)
# Semantic Scholar's batch endpoint takes up to 500 ids per request.
SEMANTIC_SCHOLAR_MAX_IDS = 500
_SEMANTIC_SCHOLAR_FIELDS = "citationCount,venue,externalIds,references.externalIds"


def _openalex_short_id(value: str | None) -> str | None:
    """``W123`` from ``https://openalex.org/W123``."""
    match = re.search(r"(W\d+)$", value or "")
    return match.group(1) if match else None


def _parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _source_name(location: dict[str, Any] | None) -> str | None:
    return ((location or {}).get("source") or {}).get("display_name")


def openalex_record(work: dict[str, Any], found_by: str) -> SourceRecord | None:
    """A work as a record; None without a title or id. The arXiv id comes
    from an arXiv DOI or an arXiv location; the venue is the first location
    that is not arXiv."""
    title = " ".join((work.get("title") or "").split())
    openalex_id = _openalex_short_id(work.get("id"))
    if not title or not openalex_id:
        return None
    locations = [work.get("primary_location") or {}, *(work.get("locations") or [])]
    arxiv_id = unversioned_arxiv_id(work.get("doi")) or next(
        (
            found
            for loc in locations
            if (found := unversioned_arxiv_id(loc.get("landing_page_url")))
        ),
        None,
    )
    doi = normalize_doi(work.get("doi"))
    venue = next(
        (
            name
            for loc in locations
            if (name := _source_name(loc)) and "arxiv" not in name.lower()
        ),
        None,
    )
    authors = [
        name
        for a in work.get("authorships") or []
        if (name := (a.get("author") or {}).get("display_name"))
        and re.search(r"[^\W\d_]", name)
    ]
    return SourceRecord(
        provider="openalex",
        title=title,
        authors=authors,
        published=_parse_date(work.get("publication_date")),
        venue=venue,
        arxiv_id=arxiv_id,
        doi=None if doi and unversioned_arxiv_id(doi) else doi,
        openalex_id=openalex_id,
        citation_count=work.get("cited_by_count"),
        # OpenAlex's abstracts of arXiv records are unreliable (one paper's
        # DOI carried another's abstract); arXiv supplies those.
        abstract=(
            None
            if arxiv_id
            else citation_metadata._inverted_index_text(
                work.get("abstract_inverted_index")
            )
        ),
        references=[
            w
            for ref in work.get("referenced_works") or []
            if (w := _openalex_short_id(ref))
        ],
        found_by={found_by},
    )


def arxiv_record(metadata: ArxivMetadata, found_by: str | None) -> SourceRecord:
    doi = normalize_doi(metadata.doi)
    return SourceRecord(
        provider="arxiv",
        title=" ".join(metadata.title.split()),
        authors=list(metadata.authors),
        published=metadata.published.date(),
        arxiv_id=unversioned_arxiv_id(metadata.arxiv_id),
        doi=None if doi and unversioned_arxiv_id(doi) else doi,
        abstract=metadata.abstract,
        found_by={found_by} if found_by else set(),
    )


def _datacite_author(creator: dict[str, Any]) -> str:
    given, family = creator.get("givenName"), creator.get("familyName")
    if given and family:
        return f"{given} {family}"
    return str(creator.get("name") or "").strip()


def datacite_record(item: dict[str, Any], found_by: str | None) -> SourceRecord | None:
    """arXiv's own metadata of a paper, from its DOI record at DataCite
    (arXiv registers ``10.48550/arXiv.<id>`` for every paper): title,
    authors, abstract and the first version's submission date."""
    attributes = item.get("attributes") or {}
    arxiv_id = unversioned_arxiv_id(attributes.get("doi") or item.get("id"))
    titles = [t.get("title") for t in attributes.get("titles") or [] if t.get("title")]
    if not arxiv_id or not titles:
        return None
    submitted = sorted(
        str(d.get("date"))[:10]
        for d in attributes.get("dates") or []
        if d.get("dateType") == "Submitted" and d.get("date")
    )
    abstract = next(
        (
            d.get("description")
            for d in attributes.get("descriptions") or []
            if d.get("descriptionType") == "Abstract" and d.get("description")
        ),
        None,
    )
    return SourceRecord(
        provider="arxiv",
        title=" ".join(str(titles[0]).split()),
        authors=[
            a for c in attributes.get("creators") or [] if (a := _datacite_author(c))
        ],
        published=_parse_date(submitted[0]) if submitted else None,
        arxiv_id=arxiv_id,
        abstract=" ".join(abstract.split()) if abstract else None,
        found_by={found_by} if found_by else set(),
    )


class DataCiteIndex:
    """arXiv DOI records at DataCite: arXiv's metadata without spending the
    arXiv API's one-request-per-3-seconds budget."""

    def arxiv_papers(self, arxiv_ids: list[str]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for start in range(0, len(arxiv_ids), DATACITE_MAX_IDS):
            chunk = arxiv_ids[start : start + DATACITE_MAX_IDS]
            query = " OR ".join(f'doi:"10.48550/arxiv.{i}"' for i in chunk)
            data = citation_metadata._get_json(
                _DATACITE_URL,
                params={"query": query, "page[size]": str(len(chunk))},
                limiter=_DATACITE_LIMITER,
                timeout=_TIMEOUT,
            )
            found = data.get("data") if isinstance(data, dict) else None
            items += found if isinstance(found, list) else []
        return items


class OpenAlexIndex:
    """OpenAlex works API: searches, lookups and one citation hop. The
    optional API key is sent as a Bearer token, never in the URL."""

    def __init__(
        self, *, api_key: str | None = None, mailto: str | None = None
    ) -> None:
        self.api_key = api_key or os.environ.get(EnvVars.OPENALEX_API_KEY.value)
        self.mailto = mailto or os.environ.get(EnvVars.CROSSREF_MAILTO.value)

    def _works(self, params: dict[str, str]) -> list[dict[str, Any]]:
        params = {**params, "select": _OPENALEX_FIELDS}
        if self.mailto:
            params["mailto"] = self.mailto
        data = citation_metadata._get_json(
            _OPENALEX_URL,
            params=params,
            limiter=citation_metadata._OPENALEX_LIMITER,
            timeout=_TIMEOUT,
            headers=(
                {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
            ),
        )
        results = data.get("results") if isinstance(data, dict) else None
        return results if isinstance(results, list) else []

    def search(
        self,
        query: str,
        *,
        per_page: int,
        since: int | None = None,
        until: int | None = None,
    ) -> list[dict[str, Any]]:
        """Works for a free-text query, most relevant first."""
        params = {"search": _search_text(query), "per-page": str(per_page)}
        if since or until:
            params["filter"] = f"publication_year:{since or ''}-{until or ''}"
        return self._works(params)

    def search_title(self, title: str, *, per_page: int = 5) -> list[dict[str, Any]]:
        return self._works(
            {"filter": f"title.search:{_search_text(title)}", "per-page": str(per_page)}
        )

    def by_doi(self, doi: str) -> list[dict[str, Any]]:
        return self._works({"filter": f"doi:{doi}"})


def _search_text(value: str) -> str:
    # Commas and colons are filter syntax in OpenAlex parameters.
    return " ".join(re.sub(r"[,:|]", " ", value).split())


def _semantic_scholar_keys(external_ids: dict[str, Any] | None) -> set[str]:
    ids = external_ids or {}
    keys = set()
    if arxiv_id := unversioned_arxiv_id(ids.get("ArXiv")):
        keys.add(f"arxiv:{arxiv_id}")
    if (doi := normalize_doi(ids.get("DOI"))) and not unversioned_arxiv_id(doi):
        keys.add(f"doi:{doi}")
    return keys


class SemanticScholarIndex:
    """Semantic Scholar's batch endpoint; used only with an API key (the
    keyless pool is shared and mostly answers 429)."""

    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get(EnvVars.SEMANTIC_SCHOLAR_API_KEY.value)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def papers(self, ids: list[str]) -> list[CitationData | None]:
        """Citation data for ``ARXIV:<id>`` / ``DOI:<doi>`` ids, aligned with
        ``ids`` (None for a paper Semantic Scholar does not know)."""
        found: list[CitationData | None] = []
        for start in range(0, len(ids), SEMANTIC_SCHOLAR_MAX_IDS):
            chunk = ids[start : start + SEMANTIC_SCHOLAR_MAX_IDS]
            data = citation_metadata._get_json(
                _SEMANTIC_SCHOLAR_BATCH_URL,
                params={"fields": _SEMANTIC_SCHOLAR_FIELDS},
                json_body={"ids": chunk},
                limiter=citation_metadata._SEMANTIC_SCHOLAR_LIMITER,
                timeout=_TIMEOUT,
                headers={"x-api-key": self.api_key} if self.api_key else None,
            )
            rows = data if isinstance(data, list) else []
            rows += [None] * (len(chunk) - len(rows))
            found += [_semantic_scholar_paper(row) for row in rows[: len(chunk)]]
        return found


def _semantic_scholar_paper(row: Any) -> CitationData | None:
    if not isinstance(row, dict):
        return None
    count = row.get("citationCount")
    return CitationData(
        citation_count=count if isinstance(count, int) else None,
        venue=(row.get("venue") or "").strip() or None,
        reference_keys=set().union(
            *(
                _semantic_scholar_keys(ref.get("externalIds"))
                for ref in row.get("references") or []
                if isinstance(ref, dict)
            )
        ),
    )
