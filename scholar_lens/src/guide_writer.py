"""The evidence engine for guides (``tech_guide.engine: evidence``), on the
legacy acquisition: :meth:`TechGuideGenerator.gather` collects the docs, the
user's reference material, the planned searches and the official reference
pages; each page becomes a source document with its authority label; the
engine reads, plans and writes; the result is a :class:`TechGuide` the guide
pipeline classifies and publishes as before."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from typing import Any

from ..domain.evidence import Authority, SourceDocument
from ..domain.post_names import slugify
from .constants import LanguageModelId, LocalPaths, S3Paths
from .guide_authority import official_reference, unverified_identifiers
from .logger import logger
from .metrics import TokenUsageTracker
from .prompts.tech_guide import (
    OFFICIAL_LABEL,
    OLDER_DOCS_LABEL,
    REFERENCE_LABEL,
)
from .runtime import ROOT_DIR, RunContext
from .tech_guide import TechGuide, TechGuideGenerator
from .web_research import OfficialDocs, PageContent, ResearchCorpus, source_label

_AUTHORITY: dict[str, Authority] = {
    OFFICIAL_LABEL: "official",
    OLDER_DOCS_LABEL: "older_docs",
    REFERENCE_LABEL: "reference",
}
_PREFIX: dict[str, str] = {
    "official": "D",
    "reference": "R",
    "older_docs": "O",
    "third_party": "W",
}
# A page shorter than this has nothing to record (a redirect stub, a menu).
_MIN_PAGE_CHARS = 300


def guide_documents(
    pages: Sequence[PageContent], official: OfficialDocs
) -> list[SourceDocument]:
    """Each readable page as a source document, numbered per authority (D1
    official, R1 reference material, W1 third-party, O1 older docs)."""
    counts: dict[str, int] = {}
    documents = []
    for page in pages:
        if len(page.text.strip()) < _MIN_PAGE_CHARS:
            continue
        authority = _AUTHORITY.get(source_label(page, official), "third_party")
        counts[authority] = counts.get(authority, 0) + 1
        documents.append(
            SourceDocument.from_markdown(
                f"{_PREFIX[authority]}{counts[authority]}",
                page.title or page.url,
                "full_text",
                page.text,
                figures=[(u, c) for u, c in page.image_captions.items() if c],
                url=page.final_url or page.url,
                authority=authority,
            )
        )
    return documents


def _source_urls(
    cited: Sequence[SourceDocument], requested: Sequence[str]
) -> list[str]:
    """The pages the body cites (official first), then the requested pages
    the body did not cite: the user's inputs are always listed."""
    order = {"official": 0, "reference": 1, "third_party": 2, "older_docs": 3}
    urls = [
        d.url for d in sorted(cited, key=lambda d: order[d.authority or "third_party"])
    ]
    return list(dict.fromkeys([*urls, *requested]))


async def write_guide_with_evidence(
    context: RunContext,
    generator: TechGuideGenerator,
    *,
    urls: list[str],
    reference_urls: list[str],
    discover_subpages: bool,
    search_queries: list[str] | None,
    translation_guideline: list[dict[str, Any]] | None,
    tracker: TokenUsageTracker,
    title_max_length: int,
) -> TechGuide:
    from ..adapters.bedrock.strands_models import StrandsLanguageModel
    from ..application.guide_engine import EvidenceGuideEngine, GuideSettings
    from .evidence_models import role_model, unservable
    from .survey_documents import EvidenceTextStore

    cfg = context.config.tech_guide
    material = await generator.gather(
        urls,
        reference_urls=reference_urls,
        discover_subpages=discover_subpages,
        search_queries=search_queries,
    )
    corpus: ResearchCorpus = material.corpus
    documents = guide_documents(corpus.pages, generator.official)
    logger.info(
        "Guide sources: %s",
        ", ".join(f"{d.paper_id}={d.url}" for d in documents),
    )
    effort = cfg.thinking_effort
    model = StrandsLanguageModel(
        context.bedrock_boto_session,
        {
            "read": role_model(context, cfg.synopsis_model_id, effort, False),
            "outline": role_model(context, cfg.writing_model_id, "high", True),
            "write": dataclasses.replace(
                role_model(context, cfg.writing_model_id, effort, True),
                fallback=role_model(context, cfg.synopsis_model_id, effort, True),
            ),
            "evaluate": role_model(context, cfg.evaluation_model_id, effort, True),
        },
        tracker.record,
    )
    official_text = official_reference(generator.official, corpus)
    products = generator.products

    def identifier_check(markdown: str) -> list[str]:
        return unverified_identifiers(markdown, official_text, products)

    language = context.config.output_language
    engine = EvidenceGuideEngine(
        model,
        GuideSettings(
            language=language,
            title_max_length=title_max_length,
            reader_tag=slugify(LanguageModelId(cfg.synopsis_model_id).value),
            max_sections=cfg.max_sections,
            max_sources=cfg.max_sources_read,
            min_total_chars=cfg.min_total_chars,
            max_total_chars=cfg.max_total_chars,
        ),
        store=EvidenceTextStore(
            ROOT_DIR / LocalPaths.PAPERS_DIR.value / "guide-evidence",
            context.s3_handler,
            context.config.resources.s3_prefix,
            S3Paths.GUIDES,
        ),
        identifier_check=identifier_check,
    )
    result = await engine.write(
        material.topic,
        documents,
        guideline=(
            json.dumps(translation_guideline, ensure_ascii=False)
            if translation_guideline
            else ""
        ),
    )
    notes = [*generator.notes, *result.notes]
    if unverified := identifier_check(result.body):
        notes.append(
            "Flags/metrics the official pages fetched for this guide never "
            "mention; verify them against the current reference: "
            + ", ".join(unverified)
        )
    figures = [
        e.figure_url
        for e in result.ledger
        if e.kind == "figure" and e.figure_url and e.figure_url in result.body
    ]
    # guide_document drops every image not in image_urls.
    if broken := await unservable(figures):
        notes.append(
            "Figures that cannot be downloaded were left out: " + ", ".join(broken)
        )
    source_urls = _source_urls(result.cited, [*urls, *material.references])
    titles = {d.url: d.title for d in documents} | {
        p.url: p.title for p in corpus.pages if p.title
    }
    return TechGuide(
        topic=result.title,
        body=result.body,
        source_urls=source_urls,
        source_titles={u: titles[u] for u in source_urls if titles.get(u)},
        notes=notes,
        image_urls=[u for u in figures if u not in set(broken)],
    )
