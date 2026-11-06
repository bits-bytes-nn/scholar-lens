"""The evidence engine for paper reviews and summaries
(``explanation.engine: evidence``, ``summary.engine: evidence``), on the legacy
acquisition: the parsed paper (text with figure markers) and its official
code summary become source documents; the engine reads, plans and writes;
the result takes the shapes the formatter publishes (``ReviewText``, the
summary's result dict)."""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Literal

from ..domain.evidence import SourceDocument
from ..domain.markdown.lint import FIGURE_MARKER
from ..domain.paper.evidence_paper import assemble_review, assemble_summary
from ..domain.post_names import slugify
from .constants import LanguageModelId, LocalPaths, S3Paths
from .explainer import Paper, ReviewText
from .metrics import TokenUsageTracker
from .runtime import ROOT_DIR, RunContext


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
    paper: Paper,
    *,
    post: Literal["review", "summary"],
    translation_guideline: list[dict[str, Any]] | None,
    tracker: TokenUsageTracker,
) -> tuple[ReviewText | dict[str, str], list[str]]:
    from ..adapters.bedrock.strands_models import StrandsLanguageModel
    from ..application.paper_engine import EvidencePaperEngine, PaperSettings
    from .evidence_models import role_model, unservable
    from .survey_documents import EvidenceTextStore

    if post == "review":
        cfg = context.config.explanation
        writer_id, outline_id, judge_id = (
            cfg.paper_synthesis_model_id,
            cfg.paper_planning_model_id,
            cfg.paper_evaluation_model_id,
        )
        reader_id = cfg.reader_model_id
        lengths = (cfg.min_total_chars, cfg.max_total_chars)
        max_sections, chars_per_reader = cfg.max_sections, 15_000
    else:
        scfg = context.config.summary
        writer_id = outline_id = judge_id = scfg.summary_model_id
        reader_id = scfg.reader_model_id
        lengths = (scfg.min_total_chars, scfg.max_total_chars)
        max_sections, chars_per_reader = 6, 40_000
    effort = (
        context.config.explanation.thinking_effort
        if post == "review"
        else context.config.summary.thinking_effort
    )
    model = StrandsLanguageModel(
        context.bedrock_boto_session,
        {
            "read": role_model(context, reader_id, effort, False),
            "outline": role_model(context, outline_id, "high", True),
            "write": dataclasses.replace(
                role_model(context, writer_id, effort, True),
                fallback=role_model(context, reader_id, effort, True),
            ),
            "evaluate": role_model(context, judge_id, effort, True),
        },
        tracker.record,
    )
    document, extra = paper_documents(paper)
    engine = EvidencePaperEngine(
        model,
        PaperSettings(
            language=context.config.output_language,
            post=post,
            reader_tag=slugify(LanguageModelId(reader_id).value),
            max_sections=max_sections,
            chars_per_reader=chars_per_reader,
            min_total_chars=lengths[0],
            max_total_chars=lengths[1],
        ),
        store=EvidenceTextStore(
            ROOT_DIR / LocalPaths.PAPERS_DIR.value / "paper-evidence",
            context.s3_handler,
            context.config.resources.s3_prefix,
            S3Paths.RESULTS,
        ),
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
    if broken := await unservable(remote):
        notes.append("Figures that cannot be downloaded were left out: " + ", ".join(broken))
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
        )
    return (
        {
            "summary": assemble_summary(sections),
            "subtitle": result.subtitle,
            "description": result.description,
            "urls": "",
        },
        notes,
    )


def _drop_images(markdown: str, urls: set[str]) -> str:
    return re.sub(
        r"!\[[^\]]*\]\(([^)\s]+)[^)]*\)\n?",
        lambda m: "" if m.group(1) in urls else m.group(0),
        markdown,
    )
