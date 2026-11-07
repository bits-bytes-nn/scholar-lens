"""The evidence engine for guides: :meth:`TechGuideGenerator.gather`
collects the docs, the user's reference material, the planned searches and
the official reference pages; each page becomes a source document with its
authority label; the engine reads, plans and writes; the result is a
:class:`TechGuide` the guide pipeline classifies and publishes."""

from __future__ import annotations

import asyncio
import dataclasses
import html
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from ..adapters.packages.pypi import package_extras, package_source
from ..domain.evidence import Authority, SourceDocument
from ..domain.guide.evidence_guide import (
    SOURCE_ID_PREFIX,
    install_requirements,
    python_blocks,
    unknown_extras,
    unverified_imports,
)
from ..domain.guide.sdk_index import SdkIndex, api_problems
from ..domain.post_names import slugify
from .constants import LanguageModelId, S3Paths
from .guide_authority import official_reference, unverified_identifiers
from .logger import logger
from .metrics import TokenUsageTracker
from .prompts.tech_guide import (
    OFFICIAL_LABEL,
    OLDER_DOCS_LABEL,
    REFERENCE_LABEL,
)
from .runtime import RunContext
from .tech_guide import TechGuide, TechGuideGenerator
from .web_research import (
    OfficialDocs,
    PageContent,
    ResearchCorpus,
    page_key,
    source_label,
)

_AUTHORITY: dict[str, Authority] = {
    OFFICIAL_LABEL: "official",
    OLDER_DOCS_LABEL: "older_docs",
    REFERENCE_LABEL: "reference",
}
_PREFIX: dict[str, str] = {
    "official": f"{SOURCE_ID_PREFIX}D",
    "reference": f"{SOURCE_ID_PREFIX}R",
    "older_docs": f"{SOURCE_ID_PREFIX}O",
    "third_party": f"{SOURCE_ID_PREFIX}W",
}
# A page shorter than this has nothing to record (a redirect stub, a menu).
_MIN_PAGE_CHARS = 300
# Images that never explain anything: logos, icons, badges, avatars.
_DECORATIVE_WORDS = frozenset(
    {
        "logo",
        "icon",
        "favicon",
        "badge",
        "avatar",
        "emoji",
        "sprite",
        "banner",
        "button",
    }
)
_BADGE_HOSTS = ("shields.io", "badgen.net")


def _decorative(url: str) -> bool:
    """Whether an image URL names a decoration: a word of its path (split at
    non-letters, plurals folded) such as ``logo`` or ``icons``, or a badge
    service host. "silicon" and "lexicon" are not icons."""
    parts = urlsplit(url)
    if parts.hostname and parts.hostname.endswith(_BADGE_HOSTS):
        return True
    words = re.split(r"[^a-z]+", parts.path.lower())
    return any(
        w in _DECORATIVE_WORDS or w.rstrip("s") in _DECORATIVE_WORDS for w in words
    )


_GITHUB_REPO = re.compile(
    r"^https?://(?:github\.com|raw\.githubusercontent\.com)/([\w.-]+)/([\w.-]+)", re.I
)


def _repository(url: str) -> str:
    m = _GITHUB_REPO.match(url)
    return f"{m.group(1)}/{m.group(2)}".lower().removesuffix(".git") if m else ""


def official_repositories(
    pages: Sequence[PageContent], official: OfficialDocs
) -> set[str]:
    """The GitHub repositories the official docs pages link to most (their
    "edit on GitHub" and source links): the project's own repositories,
    whose pages are as official as the docs."""
    counts: dict[str, int] = {}
    for page in pages:
        if _AUTHORITY.get(source_label(page, official)) != "official":
            continue
        for repo in {r for link in page.links if (r := _repository(link))}:
            counts[repo] = counts.get(repo, 0) + 1
    if not counts:
        return set()
    top = max(counts.values())
    # Linked from at least half as many official pages as the most linked
    # one, and from more than one when several official pages were read.
    official_pages = sum(
        _AUTHORITY.get(source_label(p, official)) == "official" for p in pages
    )
    floor = max(2 if official_pages > 2 else 1, (top + 1) // 2)
    return {repo for repo, n in counts.items() if n >= floor}


# Packages looked up per guide (the docs' install commands name a few).
_MAX_PACKAGES = 6


# Packages whose code a guide is checked against (the product's own).
_MAX_SDK_PACKAGES = 2


def product_sdk(
    provided: Mapping[str, tuple[str, frozenset[str]]], products: Sequence[str]
) -> SdkIndex | None:
    """The index of the product's own packages among those the official
    pages install (``strands-agents`` for "strands"), from their PyPI source;
    None when none matches or none can be read."""
    keys = [re.sub(r"[^a-z0-9]", "", p.lower()) for p in products if p]
    names = [
        n
        for n in provided
        if any(re.sub(r"[^a-z0-9]", "", n.lower()).startswith(k) for k in keys if k)
    ][:_MAX_SDK_PACKAGES]
    files: dict[str, str] = {}
    versions: list[str] = []
    for name in names:
        if (found := package_source(name)) is not None:
            versions.append(f"{name} {found[0]}")
            files.update(found[1])
    if not files:
        return None
    return SdkIndex.from_sources(files, ", ".join(versions))


def docs_packages(official_text: str) -> dict[str, tuple[str, frozenset[str]]]:
    """name -> (latest version, extras) on PyPI of the packages the official
    pages install; unknown or unreachable ones are left out."""
    names = list(dict.fromkeys(n for n, _ in install_requirements(official_text)))
    found: dict[str, tuple[str, frozenset[str]]] = {}
    for name in names[:_MAX_PACKAGES]:
        if (info := package_extras(name)) is not None:
            found[name] = info
    return found


def guide_documents(
    pages: Sequence[PageContent], official: OfficialDocs
) -> list[SourceDocument]:
    """Each readable page as a source document, numbered per authority
    (src-D1 official, src-R1 reference material, src-W1 third-party, src-O1
    older docs)."""
    counts: dict[str, int] = {}
    documents = []
    repos = official_repositories(pages, official)
    for page in pages:
        if len(page.text.strip()) < _MIN_PAGE_CHARS:
            continue
        authority = _AUTHORITY.get(source_label(page, official), "third_party")
        if (
            authority == "third_party"
            and _repository(page.final_url or page.url) in repos
        ):
            # The project's own repository (its design docs, source, README).
            authority = "official"
        counts[authority] = counts.get(authority, 0) + 1
        documents.append(
            SourceDocument.from_markdown(
                f"{_PREFIX[authority]}{counts[authority]}",
                html.unescape(page.title or page.url),
                "full_text",
                page.text,
                figures=[
                    (u, page.image_captions.get(u, ""))
                    for u in page.image_urls
                    if not _decorative(u)
                ],
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
    # One entry per page however it was spelled (a trailing slash, .html).
    seen: dict[str, str] = {}
    for url in [*urls, *requested]:
        seen.setdefault(page_key(url), url)
    return list(seen.values())


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
    from .evidence_models import budgeted, role_model, unservable
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
            "evaluate": dataclasses.replace(
                role_model(context, cfg.evaluation_model_id, effort, True),
                fallback=role_model(context, cfg.synopsis_model_id, effort, True),
            ),
        },
        budgeted(tracker, cfg.max_cost_usd),
    )
    official_text = official_reference(generator.official, corpus)
    products = generator.products

    provided = await asyncio.to_thread(docs_packages, official_text)
    package_facts = "\n".join(
        f"- {name} {version}: extras {', '.join(sorted(extras)) or 'none'}"
        for name, (version, extras) in provided.items()
    )

    # The product's own code (read from PyPI, never run): what it defines
    # decides import paths, keyword arguments and defaults over the pages.
    sdk = await asyncio.to_thread(product_sdk, provided, products)
    if sdk is not None:
        logger.info(
            "Guide code checked against %s %s (%d modules).",
            "/".join(sorted(sdk.packages)),
            sdk.version,
            len(sdk.modules),
        )
    sdk_params = {p for c in sdk.callables.values() for p in c.params} if sdk else set()

    def identifier_check(markdown: str) -> list[str]:
        # Flags and arguments, product import paths and install extras that
        # neither the official pages, the package's code nor the package
        # index shows (they fail as pasted).
        blocks = python_blocks(markdown)
        return [
            *(
                i
                for i in unverified_identifiers(markdown, official_text, products)
                if i not in sdk_params
            ),
            *(
                api_problems(blocks, sdk)
                if sdk is not None
                else unverified_imports(markdown, official_text, products)
            ),
            *unknown_extras(markdown, {n: e for n, (_, e) in provided.items()}),
        ]

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
            context.papers_dir / "guide-evidence",
            context.s3_handler,
            context.config.resources.s3_prefix,
            S3Paths.GUIDES,
        ),
        identifier_check=identifier_check,
        products=products,
        package_facts=package_facts,
        sdk=sdk,
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
    # A requested page under the address it was served from (after
    # redirects), as the cited documents have it.
    served = {p.url: p.final_url or p.url for p in corpus.pages}
    source_urls = _source_urls(
        result.cited, [served.get(u, u) for u in [*urls, *material.references]]
    )
    titles = {d.url: d.title for d in documents} | {
        p.url: p.title for p in corpus.pages if p.title
    }
    return TechGuide(
        topic=result.title,
        body=result.body,
        source_urls=source_urls,
        source_titles={
            u: html.unescape(titles[u]) for u in source_urls if titles.get(u)
        },
        notes=notes,
        image_urls=[u for u in figures if u not in set(broken)],
    )
