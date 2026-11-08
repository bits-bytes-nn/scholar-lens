"""The gathered corpus of a guide, saved and reused (docs/engine.md 3.1).

A guide's sources vary with every web search, so the same request can be
written from a different corpus each run. The gather step saves what it
collected (the planned queries, the search results, every page with its
text and source record, the products and the source set) under a hash of
its inputs, and a rerun with the same inputs within the TTL reads it back
instead of gathering again. The store is the evidence text store (local
directory plus S3).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from scholar_lens.adapters.retry import is_access_error
from scholar_lens.domain.research import PageContent, ResearchCorpus, SearchResult
from scholar_lens.domain.sources import (
    DomainClasses,
    OfficialDocs,
    SourceSet,
    TagLookup,
    TargetVersion,
)

logger = logging.getLogger(__name__)

# Bumped when the saved layout or what a record means changes: an older
# snapshot is then gathered again instead of misread.
SNAPSHOT_FORMAT = 1
_PAGE_FIELDS = {f.name for f in dataclasses.fields(PageContent)}
_RESULT_FIELDS = {f.name for f in dataclasses.fields(SearchResult)}


class TextStore(Protocol):
    def get(self, key: str) -> str | None: ...

    def put(self, key: str, text: str) -> None: ...


@dataclass(frozen=True)
class SnapshotPolicy:
    """Where gathered corpora are kept and when one is reused: within
    ``ttl_days`` of being gathered (0 never reuses) unless ``fresh``."""

    store: TextStore
    ttl_days: float = 7.0
    fresh: bool = False
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


@dataclass
class CorpusSnapshot:
    """What a gather step collected, as saved."""

    topic: str
    corpus: ResearchCorpus
    references: list[str]
    queries: list[str]
    products: tuple[str, ...]
    sources: SourceSet
    # The gather step's run notes and the searches' failure notes.
    notes: list[str]
    reference_ids: list[str] = field(default_factory=list)
    reference_searches: int = 0
    reference_urls: list[str] = field(default_factory=list)
    gathered_at: str = ""


def snapshot_key(inputs: Mapping[str, Any]) -> str:
    """The store key of a corpus gathered from ``inputs`` (the requested
    URLs, references, explicit queries and every setting that changes what
    is gathered): a hash of their canonical JSON."""
    canonical = json.dumps(
        {"format": SNAPSHOT_FORMAT, **inputs}, sort_keys=True, default=list
    )
    return f"sources/{hashlib.sha256(canonical.encode()).hexdigest()[:24]}.json"


def _sources_json(sources: SourceSet) -> dict[str, Any]:
    official, target = sources.official, sources.target
    return {
        "hosts": sorted(official.hosts),
        "versions": sorted(official.versions, key=lambda v: v or ""),
        "spaces": sorted(list(s) for s in official.spaces),
        "release": target.release,
        "pinned": target.pinned,
        "requested": sorted(target.requested, key=lambda v: v or ""),
        "repositories": sorted(sources.repositories),
        "requested_refs": sorted(list(r) for r in sources.requested_refs),
    }


def _sources_from_json(
    data: Mapping[str, Any], demoted: DomainClasses, repo_tags: TagLookup | None
) -> SourceSet:
    official = OfficialDocs(
        hosts=frozenset(data["hosts"]),
        versions=frozenset(data["versions"]),
        spaces=frozenset(tuple(s) for s in data["spaces"]),
    )
    return SourceSet(
        official,
        TargetVersion(data["release"], data["pinned"], frozenset(data["requested"])),
        repositories=frozenset(data["repositories"]),
        requested_refs=frozenset(tuple(r) for r in data["requested_refs"]),
        demoted=demoted,
        repo_tags=repo_tags,
    )


def dump_snapshot(snapshot: CorpusSnapshot) -> str:
    return json.dumps(
        {
            "format": SNAPSHOT_FORMAT,
            "gathered_at": snapshot.gathered_at,
            "topic": snapshot.topic,
            "references": snapshot.references,
            "queries": snapshot.queries,
            "products": list(snapshot.products),
            "notes": snapshot.notes,
            "reference_ids": snapshot.reference_ids,
            "reference_searches": snapshot.reference_searches,
            "reference_urls": snapshot.reference_urls,
            "sources": _sources_json(snapshot.sources),
            "pages": [dataclasses.asdict(p) for p in snapshot.corpus.pages],
            "search_results": [
                dataclasses.asdict(r) for r in snapshot.corpus.search_results
            ],
        },
        ensure_ascii=False,
    )


def load_snapshot(
    text: str, *, demoted: DomainClasses, repo_tags: TagLookup | None
) -> CorpusSnapshot | None:
    """The saved snapshot, or None when it is of another format or does
    not parse (then the corpus is gathered again)."""
    try:
        data = json.loads(text)
        if data.get("format") != SNAPSHOT_FORMAT:
            return None
        pages = [
            PageContent(**{k: v for k, v in p.items() if k in _PAGE_FIELDS})
            for p in data["pages"]
        ]
        results = [
            SearchResult(**{k: v for k, v in r.items() if k in _RESULT_FIELDS})
            for r in data["search_results"]
        ]
        return CorpusSnapshot(
            topic=data["topic"],
            corpus=ResearchCorpus(pages=pages, search_results=results),
            references=list(data["references"]),
            queries=list(data["queries"]),
            products=tuple(data["products"]),
            sources=_sources_from_json(data["sources"], demoted, repo_tags),
            notes=list(data["notes"]),
            reference_ids=list(data.get("reference_ids", [])),
            reference_searches=int(data.get("reference_searches", 0)),
            reference_urls=list(data.get("reference_urls", [])),
            gathered_at=str(data.get("gathered_at", "")),
        )
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("Ignoring an unreadable corpus snapshot: %s", e)
        return None


def read_snapshot(
    policy: SnapshotPolicy,
    key: str,
    *,
    demoted: DomainClasses,
    repo_tags: TagLookup | None,
) -> CorpusSnapshot | None:
    """The snapshot under ``key`` when the policy reuses it: not ``fresh``,
    gathered within the TTL. A store that cannot be read is a miss, except a
    refused permission, which is raised."""
    if policy.fresh or policy.ttl_days <= 0:
        return None
    try:
        text = policy.store.get(key)
    except Exception as e:  # noqa: BLE001 - a cache miss, not the run's failure
        if is_access_error(e):
            raise
        logger.warning("Could not read the corpus snapshot %s: %s", key, e)
        return None
    if text is None:
        return None
    snapshot = load_snapshot(text, demoted=demoted, repo_tags=repo_tags)
    if snapshot is None:
        return None
    try:
        gathered = datetime.fromisoformat(snapshot.gathered_at)
    except ValueError:
        return None
    if policy.now() - gathered > timedelta(days=policy.ttl_days):
        logger.info("Corpus snapshot %s is older than the TTL; gathering again.", key)
        return None
    return snapshot


def save_snapshot(policy: SnapshotPolicy, key: str, snapshot: CorpusSnapshot) -> None:
    snapshot.gathered_at = policy.now().isoformat()
    policy.store.put(key, dump_snapshot(snapshot))
    logger.info("Saved the corpus snapshot %s.", key)
