"""``scholar-lens review`` and ``summarize``: resolve → read → write →
publish.

The writer is the evidence engine (``explanation.engine: evidence``,
``summary.engine: evidence``) or the legacy sections engine, behind the same
ports: the parsed paper (text with figure markers) and its official code
summary become source documents; the engine reads, plans and writes; the
result takes the shapes the formatter publishes (``ReviewText``, the
summary's result dict)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from scholar_lens.application.paper_engine import EvidencePaperEngine, PaperSettings
from scholar_lens.application.ports import PaperTools, UsageTracker
from scholar_lens.application.run import (
    PipelineOutcome,
    RunContext,
    check_post_type,
    find_published_post,
    load_translation_guideline,
    publish_artifact,
)
from scholar_lens.domain.conventions import BlogPost
from scholar_lens.domain.evidence import SourceDocument
from scholar_lens.domain.jobs import JobRequest
from scholar_lens.domain.markdown.front_matter import split_front_matter
from scholar_lens.domain.markdown.lint import (
    FIGURE_MARKER,
    drop_unknown_images,
    figure_markers_to_images,
    fill_image_alts,
)
from scholar_lens.domain.paper.evidence_paper import assemble_review, assemble_summary
from scholar_lens.domain.paper.models import Paper, ReviewText
from scholar_lens.domain.post_format import (
    format_review,
    format_summary,
    paper_post_meta,
)
from scholar_lens.domain.post_names import title_slug
from scholar_lens.domain.publishing import PublishRequest, build_pr_body
from scholar_lens.domain.tags import unsupported_tags
from scholar_lens.domain.validation import unsourced_table_numbers

logger = logging.getLogger(__name__)


async def run_paper(
    context: RunContext,
    request: JobRequest,
    tracker: UsageTracker,
    tools: PaperTools,
) -> PipelineOutcome:
    """Generate and publish a review (``kind="review"``) or summary."""
    assert request.source is not None  # guaranteed by JobRequest validation
    source = tools.resolve(request.source)
    code_retriever: Any = None
    try:
        # Checked first: a config or S3 failure must not follow the paid stages.
        github = context.config.resources.github
        if request.kind == "summarize":
            check_post_type(
                context.require_conventions(),
                github.summary_category,
                "summary_category",
                artifact="summarize",
            )
        else:
            check_post_type(
                context.require_conventions(),
                github.review_category,
                "review_category",
                artifact="review",
            )
        translation_guideline = await load_translation_guideline(context)
        paper_dir = context.papers_dir / source.source_id
        # Arbitrary PDF-URL sources have no arXiv HTML rendering.
        parse_pdf = request.parse_pdf or source.arxiv_html_id is None
        paper, code_retriever, citation_summarizer = await tools.read(
            paper_dir,
            source,
            request.repo_urls,
            parse_pdf,
            tracker,
            inputs=tools.inputs(request.kind),
        )
        branch_id = paper_branch_id(paper, request.kind)
        github = context.config.resources.github
        published, lookup_notes = await find_published_post(
            context,
            slug=title_slug(paper.title, request.kind),
            branch_id=branch_id,
            category=(
                github.summary_category
                if request.kind == "summarize"
                else github.review_category
            ),
        )
        if request.kind == "summarize":
            document, degraded, blocking, verified = await _generate_summary(
                context, tools, paper, translation_guideline, tracker, published
            )
        else:
            document, degraded, blocking, verified = await _generate_review(
                context,
                tools,
                paper,
                citation_summarizer,
                code_retriever,
                translation_guideline,
                tracker,
                published,
            )
        degraded += (
            lookup_notes
            + _text_only_notes(paper)
            + _unanalysed_figure_notes(paper)
            + _code_notes(paper)
            + _number_notes(paper, document)
        )
        # Keep only images that are real figures of this paper.
        document = figure_markers_to_images(document)
        document = drop_unknown_images(document, _figure_sources(paper))
        document = fill_image_alts(
            document,
            _figure_captions(paper),
            paper.title,
            korean=context.config.output_language == "Korean",
        )
        publish_request = _build_publish_request(
            paper,
            paper_dir,
            document,
            request.kind,
            branch_id=branch_id,
            known_tags=set(context.require_blog_tags())
            | set(context.require_conventions().tags.controlled),
            published=published,
        )
        publish_request.blocking += blocking
        publish_request.review_notes += degraded
        publish_request.link_sources = paper.link_sources(verified)
        outcome = await publish_artifact(
            context, publish_request, create_pr=request.create_pr
        )
        return outcome
    finally:
        if code_retriever is not None:
            await code_retriever.delete_index()
        source.close()


async def _generate_review(
    context: RunContext,
    tools: PaperTools,
    paper: Paper,
    citation_summarizer: Any,
    code_retriever: Any,
    translation_guideline: list[dict[str, Any]] | None,
    tracker: UsageTracker,
    published: BlogPost | None,
) -> tuple[str, list[str], list[str], tuple[str, ...]]:
    """The review post, its notes, its blocking problems and the links its
    writer verified beyond the paper's sources."""
    github = context.config.resources.github
    if tools.engine("review") == "evidence":
        written, notes, blocking = await write_paper_with_evidence(
            context,
            tools,
            paper,
            post="review",
            translation_guideline=translation_guideline,
            tracker=tracker,
        )
        assert isinstance(written, ReviewText)
        meta = paper_post_meta(
            paper,
            github.review_category,
            context.require_conventions(),
            known_tags=context.require_blog_tags(),
            published=published,
        )
        document = format_review(github, paper, written, meta=meta)
        return document, notes, blocking, written.link_sources
    # The sections engine's prerequisite summaries (_prepare_paper_data builds
    # the summarizer for that engine only).
    if citation_summarizer is None:
        raise RuntimeError("The sections review needs the citation summarizer.")
    review, notes = await tools.sections().review(
        paper,
        citation_summarizer=citation_summarizer,
        code_retriever=code_retriever,
        translation_guideline=translation_guideline,
        tracker=tracker,
    )
    meta = paper_post_meta(
        paper,
        github.review_category,
        context.require_conventions(),
        known_tags=context.require_blog_tags(),
        published=published,
    )
    document = format_review(github, paper, review, meta=meta)
    return document, notes, [], review.link_sources


async def _generate_summary(
    context: RunContext,
    tools: PaperTools,
    paper: Paper,
    translation_guideline: list[dict[str, Any]] | None,
    tracker: UsageTracker,
    published: BlogPost | None,
) -> tuple[str, list[str], list[str], tuple[str, ...]]:
    if tools.engine("summary") == "evidence":
        written, notes, blocking = await write_paper_with_evidence(
            context,
            tools,
            paper,
            post="summary",
            translation_guideline=translation_guideline,
            tracker=tracker,
        )
        assert isinstance(written, dict)
        github = context.config.resources.github
        meta = paper_post_meta(
            paper,
            github.summary_category,
            context.require_conventions(),
            known_tags=context.require_blog_tags(),
            published=published,
        )
        return format_summary(github, paper, written, meta=meta), notes, blocking, ()
    result, notes = await tools.sections().summary(
        paper, translation_guideline=translation_guideline, tracker=tracker
    )
    github = context.config.resources.github
    meta = paper_post_meta(
        paper,
        github.summary_category,
        context.require_conventions(),
        known_tags=context.require_blog_tags(),
        published=published,
    )
    return format_summary(github, paper, result, meta=meta), notes, [], ()


def paper_documents(paper: Paper) -> tuple[SourceDocument, list[SourceDocument]]:
    """The paper as a source document (its figures from the text's
    ``[Image: …]`` markers) and, when the pipeline has one, its official
    code summary as reference material."""
    figures: dict[str, str] = {}
    for m in FIGURE_MARKER.finditer(paper.content.text):
        figures.setdefault(m.group("src"), m.group("caption") or m.group("alt") or "")
    document = SourceDocument.from_markdown(
        paper.arxiv_id,
        paper.title,
        "pdf_text" if paper.is_pdf_parsed else "full_text",
        paper.content.text,
        figures=list(figures.items()),
        url=f"https://arxiv.org/abs/{paper.arxiv_id}",
        authority="official",
    )
    extra = []
    if paper.codebase_summary:
        extra.append(
            SourceDocument.from_markdown(
                f"{paper.arxiv_id}-code",
                f"{paper.title}: official code",
                "full_text",
                paper.codebase_summary,
                authority="reference",
            )
        )
    return document, extra


async def write_paper_with_evidence(
    context: RunContext,
    tools: PaperTools,
    paper: Paper,
    *,
    post: Literal["review", "summary"],
    translation_guideline: list[dict[str, Any]] | None,
    tracker: UsageTracker,
) -> tuple[ReviewText | dict[str, str], list[str], list[str]]:
    writing = tools.writing(post)
    model = tools.models.language_model(writing.roles, tracker, writing.max_cost_usd)
    document, extra = paper_documents(paper)
    store = tools.evidence_store(fresh=context.fresh)
    engine = EvidencePaperEngine(
        model,
        PaperSettings(
            language=context.config.output_language,
            post=post,
            reader_tag=writing.reader_tag,
            max_sections=writing.max_sections,
            chars_per_reader=writing.chars_per_reader,
            min_total_chars=writing.min_total_chars,
            max_total_chars=writing.max_total_chars,
        ),
        store=store,
        checkpoints=store,
        roles=writing.roles,
    )
    result = await engine.write(
        paper.title,
        document,
        extra=extra,
        abstract=paper.abstract or "",
        guideline=(
            json.dumps(translation_guideline, ensure_ascii=False)
            if translation_guideline
            else ""
        ),
    )
    notes = list(result.notes)
    remote = [
        e.figure_url
        for e in result.ledger
        if e.kind == "figure"
        and e.figure_url
        and e.figure_url.startswith("http")
        and any(e.figure_url in md for _, md in result.sections)
    ]
    if broken := await tools.figures.unservable(remote):
        notes.append(
            "Figures that cannot be downloaded were left out: " + ", ".join(broken)
        )
        sections = [
            (kind, _drop_images(md, set(broken))) for kind, md in result.sections
        ]
    else:
        sections = result.sections
    if post == "review":
        key, body, closing = assemble_review(sections)
        return (
            ReviewText(
                body=body,
                key_takeaways=key,
                closing=closing,
                subtitle=result.subtitle,
                description=result.description,
                notes=tuple(notes),
            ),
            notes,
            list(result.blocking),
        )
    return (
        {
            "summary": assemble_summary(sections),
            "subtitle": result.subtitle,
            "description": result.description,
            "urls": "",
        },
        notes,
        list(result.blocking),
    )


def _drop_images(markdown: str, urls: set[str]) -> str:
    return re.sub(
        r"!\[[^\]]*\]\(([^)\s]+)[^)]*\)\n?",
        lambda m: "" if m.group(1) in urls else m.group(0),
        markdown,
    )


def _figure_sources(paper: Paper) -> set[str]:
    """Every form a figure may take in the post: as extracted and its basename."""
    sources: set[str] = set()
    for fig in paper.figures:
        sources.add(str(fig.path))
        sources.add(Path(str(fig.path)).name)
    return sources


def _figure_captions(paper: Paper) -> dict[str, str]:
    captions: dict[str, str] = {}
    for fig in paper.figures:
        caption = fig.caption if fig.caption != "Refer to caption" else ""
        caption = caption or fig.analysis or ""  # arXiv's placeholder is no caption
        captions[str(fig.path)] = caption
        captions[Path(str(fig.path)).name] = caption
    return captions


def _text_only_notes(paper: Paper) -> list[str]:
    """The paper's source downgrades (HTML rendering to PDF) and a note when
    PDF layout parsing failed and the post was written from the PDF's text
    layer alone."""
    notes = list(paper.content.notes)
    if paper.content.text_only:
        notes.append("PDF parsed as text only; figures and tables unavailable.")
    return notes


def _unanalysed_figure_notes(paper: Paper) -> list[str]:
    """A note when figure analysis gave up on some figures (they were written
    about from their captions only)."""
    count = sum(fig.analysis is None for fig in paper.figures)
    if not count:
        return []
    return [f"{count} figure(s) were not analysed (captions only): check them."]


def _code_notes(paper: Paper) -> list[str]:
    """A note when the user gave a repository but code analysis produced
    nothing (it degrades to "no codebase" instead of failing the run)."""
    if paper.repo_urls and not paper.codebase_summary:
        return [
            "Code analysis of the given repository failed or found no Python "
            "files; the post was written without it."
        ]
    return []


def _number_notes(paper: Paper, document: str) -> list[str]:
    """Table numbers the paper never states (an invented or mistyped value
    is the defect; a derived one is fine but worth a look)."""
    missing = unsourced_table_numbers(document, paper.content.text)
    if not missing:
        return []
    return [
        "Table values not found in the paper (check each is copied or derived "
        f"correctly): {', '.join(missing)}"
    ]


def _tag_notes(paper: Paper, document: str) -> list[str]:
    """Tags whose phrase or acronym the paper itself never uses: a likely
    mis-tag worth a reviewer's look (checked against the paper, not the
    translated post)."""
    front, _, _ = split_front_matter(document)
    tags = [str(t) for t in (front or {}).get("tags") or []]
    return [
        f"Tag '{tag}' (or its acronym) does not appear in the paper; check it fits."
        for tag in unsupported_tags(tags, paper.content.text)
    ]


def _new_tags(document: str, known_tags: set[str]) -> list[str]:
    front, _, _ = split_front_matter(document)
    tags = (front or {}).get("tags") or []
    return [str(t) for t in tags if str(t) not in known_tags]


def _build_publish_request(
    paper: Paper,
    paper_dir: Path,
    document: str,
    kind: str,
    *,
    branch_id: str,
    known_tags: set[str],
    published: BlogPost | None,
) -> PublishRequest:
    label = "summary" if kind == "summarize" else "review"
    pr_body = build_pr_body(
        kind=f"paper {label}",
        title=paper.title,
        fields={
            "Paper ID": paper.arxiv_id,
            "PDF URL": str(paper.pdf_url),
            "PDF Parsing": "enabled" if paper.is_pdf_parsed else "disabled",
            "Characters": f"{len(document):,}",
            "Repositories": ", ".join(str(r) for r in paper.repo_urls),
            # Tags not used by any post yet: confirm they are worth creating.
            "New tags": ", ".join(_new_tags(document, known_tags)),
        },
    )
    return PublishRequest(
        title=paper.title,
        markdown=document,
        work_dir=paper_dir,
        branch_id=branch_id,
        pr_title=f"Paper {label.capitalize()}: {paper.title}",
        pr_body=pr_body,
        commit_message=f"feat: add paper {label} for '{paper.title}'",
        kind="summarize" if kind == "summarize" else "review",
        review_notes=_tag_notes(paper, document),
        published_path=published.path if published else None,
    )


def paper_branch_id(paper: Paper, kind: str) -> str:
    """Version-free id: a new arXiv version (or title) updates the same
    branch, pull request and post."""
    label = "summary" if kind == "summarize" else "review"
    return f"{label}-{re.sub(r'v\d+$', '', paper.arxiv_id)}"
