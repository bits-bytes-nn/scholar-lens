"""``scholar-lens guide``: gather the sources, write the guide with the
evidence engine, classify and publish it.

:meth:`GuideResearch.gather` collects the docs, the user's reference
material, the planned searches and the official reference pages, each with
its source record; each page becomes a source document carrying that
record; the engine reads, plans and writes; the result is a
:class:`TechGuide` the use case classifies and publishes."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, get_args
from urllib.parse import urlparse

from scholar_lens.application.guide_engine import EvidenceGuideEngine, GuideSettings
from scholar_lens.application.ports import (
    GuideResearch,
    GuideTools,
    PackageIndex,
    UsageTracker,
)
from scholar_lens.application.run import (
    PipelineOutcome,
    RunContext,
    check_post_type,
    find_published_post,
    load_translation_guideline,
    publish_artifact,
)
from scholar_lens.domain.evidence import (
    Authority,
    SourceAuthority,
    SourceDocument,
    authority_rank,
)
from scholar_lens.domain.guide.authority import (
    official_reference,
    unverified_identifiers,
)
from scholar_lens.domain.guide.evidence_guide import (
    SOURCE_ID_PREFIX,
    install_requirements,
    python_blocks,
    unknown_extras,
    unverified_imports,
)
from scholar_lens.domain.guide.models import TechGuide
from scholar_lens.domain.guide.sdk_index import PackageSource, SdkIndex, api_problems
from scholar_lens.domain.jobs import JobRequest
from scholar_lens.domain.markdown.lint import (
    drop_empty_headings,
    drop_unknown_images,
    figure_markers_to_images,
    fill_image_alts,
)
from scholar_lens.domain.post_format import build_post_meta, cut_title, format_guide
from scholar_lens.domain.post_names import slugify, title_slug
from scholar_lens.domain.publishing import PublishRequest, build_pr_body
from scholar_lens.domain.research import PageContent, ResearchCorpus, page_key
from scholar_lens.domain.sources import (
    ReleaseUnavailable,
    engine_authority,
    normalize_url,
)

logger = logging.getLogger(__name__)

_PREFIX: dict[Authority, str] = {
    "official": f"{SOURCE_ID_PREFIX}D",
    "reference": f"{SOURCE_ID_PREFIX}R",
    "older_docs": f"{SOURCE_ID_PREFIX}O",
    "third_party": f"{SOURCE_ID_PREFIX}W",
}
# A page shorter than this has nothing to record (a redirect stub, a menu).
_MIN_PAGE_CHARS = 300
# Packages looked up per guide (the docs' install commands name a few).
_MAX_PACKAGES = 6
# Packages whose code a guide is checked against (the product's own).
_MAX_SDK_PACKAGES = 2


async def run_guide(
    context: RunContext,
    request: JobRequest,
    tracker: UsageTracker,
    tools: GuideTools,
) -> PipelineOutcome:
    """Generate and publish a guide; raises ``NotTechnicalContentError`` when
    the URLs are not technical documentation."""
    sources = [*request.urls, *request.reference_urls]
    translation_guideline = await load_translation_guideline(context)
    researcher = tools.researcher()
    spec = context.require_conventions()
    github = context.config.resources.github
    # Both fail before any model call.
    check_post_type(
        spec, github.tech_guide_category, "tech_guide_category", artifact="guide"
    )
    if request.search_queries and not researcher.supports_search:
        raise ValueError(
            "--search-query needs a web search provider: set TAVILY_API_KEY or "
            "BRAVE_API_KEY (in Batch, the SSM parameters tavily-api-key or "
            "brave-api-key)."
        )
    generator = tools.research(
        researcher, tracker, title_max_length=spec.seo_title_max_length
    )
    try:
        guide = await write_guide_with_evidence(
            context,
            tools,
            generator,
            urls=request.urls,
            reference_urls=request.reference_urls,
            discover_subpages=request.discover_subpages,
            search_queries=request.search_queries or None,
            translation_guideline=translation_guideline,
            tracker=tracker,
            title_max_length=spec.seo_title_max_length,
            fresh_sources=request.fresh_gathering,
        )
        taxonomy = await generator.classify(
            guide,
            category_options=list(spec.topics),
            controlled_tags=list(spec.tags.controlled),
        )
    except Exception as e:
        # Keep the generator's notes (e.g. unreadable references) on a failed run.
        e.run_notes = [  # type: ignore[attr-defined]
            *generator.notes,
            *researcher.search_notes(),
        ]
        raise
    finally:
        researcher.close()
    guide.topic = guide_title(
        guide.topic, taxonomy.get("seo_title") or "", spec.seo_title_max_length
    )
    # Keyed by the sources, not the model's title: a rerun replaces the post
    # an earlier run published, whatever title it gets now.
    branch_id = guide_branch_id(sources)
    published, lookup_notes = await find_published_post(
        context,
        slug=title_slug(guide.topic, "guide"),
        branch_id=branch_id,
        category=github.tech_guide_category,
    )
    meta = build_post_meta(
        title=guide.topic,
        category_type=github.tech_guide_category,
        proposed_topic=taxonomy.get("category", ""),
        specific_tags=_split_list(taxonomy.get("keywords", "")),
        controlled_tags=_split_list(taxonomy.get("controlled_tags", "")),
        proposed_seo_title=taxonomy.get("seo_title"),
        spec=spec,
        known_tags=context.require_blog_tags(),
        published=published,
    )
    slug = slugify(guide.topic, max_length=80)
    work_dir = context.papers_dir / f"guide-{slug}"
    context.files.make_dir(str(work_dir))
    document = guide_document(
        format_guide(github, guide, meta=meta),
        guide,
        korean=context.config.output_language == "Korean",
    )
    publish_request = _build_publish_request(
        guide, work_dir, document, branch_id=branch_id
    )
    publish_request.published_path = published.path if published else None
    publish_request.blocking += guide.blocking
    publish_request.review_notes += [
        *lookup_notes,
        *guide.notes,
        *researcher.search_notes(),
    ]
    return await publish_artifact(context, publish_request, create_pr=request.create_pr)


async def write_guide_with_evidence(
    context: RunContext,
    tools: GuideTools,
    generator: GuideResearch,
    *,
    urls: list[str],
    reference_urls: list[str],
    discover_subpages: bool,
    search_queries: list[str] | None,
    translation_guideline: list[dict[str, Any]] | None,
    tracker: UsageTracker,
    title_max_length: int,
    fresh_sources: bool = False,
) -> TechGuide:
    """The evidence engine on the gathered corpus: reading agents, an
    evidence-assigning outline, verified sections."""
    writing = tools.writing
    # The gathered corpus is saved and reused for the same inputs within
    # its TTL, so reruns and evaluations write from the same sources.
    material = await generator.gather(
        urls,
        reference_urls=reference_urls,
        discover_subpages=discover_subpages,
        search_queries=search_queries,
        snapshots=tools.snapshots(fresh=fresh_sources),
    )
    corpus: ResearchCorpus = material.corpus
    documents = guide_documents(corpus.pages)
    logger.info(
        "Guide sources: %s",
        ", ".join(f"{d.paper_id}={d.url}" for d in documents),
    )
    model = tools.models.language_model(writing.roles, tracker, writing.max_cost_usd)
    official_text = official_reference(corpus)
    products = generator.products

    # The index is read at the run's target release for the product's own
    # distributions (the others at their latest); each answer is read once.
    packages = ReadOnce(tools.packages)
    release = material.release
    index_notes: list[str] = []
    latest = await asyncio.to_thread(docs_packages, official_text, packages)
    provided = latest
    if release:
        own = await asyncio.to_thread(product_distributions, latest, products, packages)
        provided = await asyncio.to_thread(
            release_packages, latest, own, release, packages, index_notes
        )
    package_facts = "\n".join(
        f"- {name} {version}: extras {', '.join(sorted(extras)) or 'none'}"
        for name, (version, extras) in provided.items()
    )

    # The product's own code (read from PyPI, never run): what it defines
    # decides import paths, keyword arguments and defaults over the pages.
    sdk = await asyncio.to_thread(
        product_sdk, latest, products, packages, release, index_notes
    )
    if sdk is not None:
        logger.info(
            "Guide code checked against %s %s (%d modules%s).",
            "/".join(sorted(sdk.packages)),
            sdk.version,
            len(sdk.modules),
            ", partial" if sdk.partial else "",
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

    store = tools.evidence_store(fresh=context.fresh)
    engine = EvidenceGuideEngine(
        model,
        GuideSettings(
            language=context.config.output_language,
            title_max_length=title_max_length,
            reader_tag=writing.reader_tag,
            max_sections=writing.max_sections,
            max_sources=writing.max_sources,
            min_total_chars=writing.min_total_chars,
            max_total_chars=writing.max_total_chars,
        ),
        store=store,
        checkpoints=store,
        roles=writing.roles,
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
    notes = [*generator.notes, *dict.fromkeys(index_notes), *result.notes]
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
    if broken := await tools.figures.unservable(figures):
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
        blocking=list(result.blocking),
        image_urls=[u for u in figures if u not in set(broken)],
    )


def _normalized(name: str) -> str:
    """A distribution or import name as PEP 503 compares them."""
    return re.sub(r"[-_.]+", "-", name).lower()


class ReadOnce:
    """A :class:`PackageIndex` whose answer for a (name, release) is read
    once: the product's distributions are found from their latest code,
    which the code check then reuses."""

    def __init__(self, index: PackageIndex) -> None:
        self._index = index
        self._extras: dict[
            tuple[str, str | None], tuple[str, frozenset[str]] | None | Exception
        ] = {}
        self._sources: dict[
            tuple[str, str | None], PackageSource | None | Exception
        ] = {}

    def extras(
        self, name: str, release: str | None = None
    ) -> tuple[str, frozenset[str]] | None:
        if (name, release) not in self._extras:
            try:
                self._extras[name, release] = self._index.extras(name, release)
            except ReleaseUnavailable as e:
                self._extras[name, release] = e
        if isinstance(found := self._extras[name, release], Exception):
            raise found
        return found

    def source(self, name: str, release: str | None = None) -> PackageSource | None:
        if (name, release) not in self._sources:
            try:
                self._sources[name, release] = self._index.source(name, release)
            except ReleaseUnavailable as e:
                self._sources[name, release] = e
        if isinstance(found := self._sources[name, release], Exception):
            raise found
        return found


def _release_note(name: str, release: str, error: Exception, check: str) -> str:
    return (
        f"The package index has no usable {name} {release} ({error}): the "
        f"{check} is skipped rather than run against another release."
    )


def product_distributions(
    provided: Mapping[str, tuple[str, frozenset[str]]],
    products: Sequence[str],
    packages: PackageIndex,
) -> list[str]:
    """The product's own distributions among those the official pages
    install: those whose name is a product name (PEP 503 normalised), else
    (up to the code check's cap) those whose latest code installs a
    top-level package of that name."""
    wanted = {_normalized(p) for p in products if p.strip()}
    if not wanted:
        return []
    if exact := [n for n in provided if _normalized(n) in wanted]:
        return exact
    found: list[str] = []
    for name in provided:
        if len(found) >= _MAX_SDK_PACKAGES:
            break
        if (code := packages.source(name)) is None:
            continue
        own = SdkIndex.from_sources(code.files, top_levels=code.top_levels)
        if {_normalized(p) for p in own.packages} & wanted:
            found.append(name)
    return found


def release_packages(
    provided: Mapping[str, tuple[str, frozenset[str]]],
    own: Sequence[str],
    release: str,
    packages: PackageIndex,
    notes: list[str],
) -> dict[str, tuple[str, frozenset[str]]]:
    """``provided`` with the product's distributions (``own``) at the
    target ``release``. One the index has no such release of is left out,
    so its extras are not checked, with a run note."""
    out = dict(provided)
    for name in own:
        try:
            found = packages.extras(name, release)
        except ReleaseUnavailable as e:
            notes.append(_release_note(name, release, e, "install extras check"))
            out.pop(name, None)
            continue
        if found is None:
            # Unknown or unreachable at the target: its latest release's
            # extras are another release's, so it is not checked.
            notes.append(
                f"Package {name} {release} could not be read from the package "
                "index; its install extras were not checked."
            )
            out.pop(name, None)
            continue
        out[name] = found
    return out


def product_sdk(
    provided: Mapping[str, tuple[str, frozenset[str]]],
    products: Sequence[str],
    packages: PackageIndex,
    release: str | None = None,
    notes: list[str] | None = None,
) -> SdkIndex | None:
    """The index of the product's own distributions
    (:func:`product_distributions`), from their code at the target
    ``release`` on the package index (their latest when None). None when
    none matches or none can be read; a distribution the index has no
    usable file of at ``release`` is left out with a run note in ``notes``
    (its code is not checked against another release)."""
    packages = packages if isinstance(packages, ReadOnce) else ReadOnce(packages)
    read: list[tuple[str, PackageSource]] = []
    for name in product_distributions(provided, products, packages):
        if len(read) >= _MAX_SDK_PACKAGES:
            break
        try:
            found = packages.source(name, release)
        except ReleaseUnavailable as e:
            if notes is not None and release:
                notes.append(_release_note(name, release, e, "code check"))
            continue
        if found is None:
            if notes is not None and release:
                notes.append(
                    f"Package {name} {release} could not be read from the "
                    "package index; its code was not checked."
                )
            continue
        read.append((name, found))
    files = {path: text for _, found in read for path, text in found.files.items()}
    if not files:
        return None
    return SdkIndex.from_sources(
        files,
        ", ".join(f"{name} {found.version}" for name, found in read),
        partial=any(not found.complete for _, found in read),
        top_levels={t for _, found in read for t in found.top_levels},
    )


def docs_packages(
    official_text: str, packages: PackageIndex
) -> dict[str, tuple[str, frozenset[str]]]:
    """name -> (latest version, extras) on the package index of the packages
    the official pages install; unknown or unreachable ones are left out."""
    names = list(dict.fromkeys(n for n, _ in install_requirements(official_text)))
    found: dict[str, tuple[str, frozenset[str]]] = {}
    for name in names[:_MAX_PACKAGES]:
        if (info := packages.extras(name)) is not None:
            found[name] = info
    return found


def guide_documents(pages: Sequence[PageContent]) -> list[SourceDocument]:
    """Each readable page as a source document carrying its source record
    (decided at gather time, never here), numbered per engine authority
    (src-D1 official, src-R1 reference material, src-W1 third-party, src-O1
    older docs)."""
    counts: dict[str, int] = {}
    documents = []
    for page in pages:
        if len(page.text.strip()) < _MIN_PAGE_CHARS:
            continue
        authority = engine_authority(page.authority)
        counts[authority] = counts.get(authority, 0) + 1
        documents.append(
            SourceDocument.from_markdown(
                f"{_PREFIX[authority]}{counts[authority]}",
                html.unescape(page.title or page.url),
                "full_text",
                page.text,
                # The page's content images: its chrome is left out when it
                # is read, and what is decorative is the reader's call.
                figures=[(u, page.image_captions.get(u, "")) for u in page.image_urls],
                url=page.final_url or page.url,
                authority=authority,
                source_authority=_source_authority(page.authority),
                version=page.version,
                published=page.published,
                language=page.language,
            )
        )
    return documents


def _source_authority(value: str) -> SourceAuthority | Literal[""]:
    return value if value in get_args(SourceAuthority) else ""  # type: ignore[return-value]


def _source_urls(
    cited: Sequence[SourceDocument], requested: Sequence[str]
) -> list[str]:
    """The pages the body cites (official first), then the requested pages
    the body did not cite: the user's inputs are always listed."""
    urls = [d.url for d in sorted(cited, key=lambda d: authority_rank(d.authority))]
    # One entry per page however it was spelled (a trailing slash, .html).
    seen: dict[str, str] = {}
    for url in [*urls, *requested]:
        seen.setdefault(page_key(url), url)
    return list(seen.values())


def guide_title(title: str, seo_title: str, limit: int) -> str:
    """``title`` when it fits the blog's ``seo_title`` limit, else the
    taxonomy's ``seo_title`` (when it fits) or the title cut at a word
    boundary: a guide title over the limit is a list of sections, not a name,
    and the part before its colon ("vLLM") names the tool, not the guide."""
    if len(title) <= limit:
        return title
    seo_title = " ".join(seo_title.split())
    if seo_title and len(seo_title) <= limit:
        return seo_title
    return cut_title(title, limit)


def guide_document(document: str, guide: TechGuide, *, korean: bool) -> str:
    """Copied ``[Image: …]`` markers become images, images the writer was
    never offered (a logo, a tracking pixel, an invented URL) are dropped, and
    every image gets alt text (Korean in a ``korean`` post): the paper
    pipeline's order."""
    document = figure_markers_to_images(document)
    document = drop_unknown_images(document, set(guide.image_urls))
    # A section whose only content was a dropped image heads nothing.
    document = drop_empty_headings(document)
    return fill_image_alts(document, {}, guide.topic, korean=korean)


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def guide_branch_id(urls: list[str]) -> str:
    """A guide's branch is keyed by its source URLs (documentation and
    reference material), not the model-chosen title, so rerunning the same
    request reuses the branch and pull request."""
    key = "\n".join(sorted({_source_key(url) for url in urls}))
    return f"guide-{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def _source_key(url: str) -> str:
    """:func:`normalize_url` plus the query, which can select the page
    (``view?id=intro``); the fragment never does."""
    query = urlparse(url).query
    return normalize_url(url) + (f"?{query}" if query else "")


def _build_publish_request(
    guide: TechGuide, work_dir: Path, document: str, *, branch_id: str
) -> PublishRequest:
    pr_body = build_pr_body(
        kind="technical guide",
        title=guide.topic,
        fields={
            "Sources": f"{len(guide.source_urls)} page(s)",
            "Characters": f"{len(document):,}",
        },
    )
    return PublishRequest(
        title=guide.topic,
        markdown=document,
        work_dir=work_dir,
        branch_id=branch_id,
        pr_title=f"Tech Guide: {guide.topic}",
        pr_body=pr_body,
        commit_message=f"feat: add technical guide for '{guide.topic}'",
        kind="guide",
    )
