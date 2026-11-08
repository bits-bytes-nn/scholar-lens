"""The sections engine behind the paper use case's ports: its review and
summary writers (:class:`~scholar_lens.application.ports.SectionsWriter`)
and the code index and citation summarizer the paper reader builds for it.

The settings are the configuration's ``explanation``, ``summary``, ``code``
and ``citations`` sections, read as they are (this package goes with them,
docs/engine.md E9).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import boto3

from scholar_lens.application.ports import UsageTracker
from scholar_lens.domain.constants import EmbeddingModelId, LanguageModelId
from scholar_lens.domain.paper.models import Paper, ReviewText
from scholar_lens.legacy.citation_summarizer import CitationSummarizer
from scholar_lens.legacy.code_index import CodeIndex
from scholar_lens.legacy.explainer import PaperReviewer
from scholar_lens.legacy.summarizer import PaperSummarizer

logger = logging.getLogger(__name__)


class SectionsPaperWriter:
    """The review (:class:`PaperReviewer`) and summary
    (:class:`PaperSummarizer`) of the sections engine."""

    def __init__(
        self,
        *,
        explanation: Any,
        summary: Any,
        language: str,
        boto_session: boto3.Session,
    ) -> None:
        self.explanation = explanation
        self.summary_settings = summary
        self.language = language
        self.boto_session = boto_session

    async def review(
        self,
        paper: Paper,
        *,
        citation_summarizer: Any,
        code_retriever: Any,
        translation_guideline: list[dict[str, Any]] | None,
        tracker: UsageTracker,
    ) -> tuple[ReviewText, list[str]]:
        cfg = self.explanation
        reviewer = PaperReviewer(
            paper,
            planning_model_id=LanguageModelId(cfg.paper_planning_model_id),
            synthesis_model_id=LanguageModelId(cfg.paper_synthesis_model_id),
            evaluation_model_id=LanguageModelId(cfg.paper_evaluation_model_id),
            finalization_model_id=LanguageModelId(cfg.paper_finalization_model_id),
            boto_session=self.boto_session,
            citation_summarizer=citation_summarizer,
            code_retriever=code_retriever,
            translation_guideline=translation_guideline,
            language=self.language,
            min_quality_score=cfg.min_quality_score,
            max_synthesis_attempts=cfg.max_synthesis_attempts,
            max_sections=cfg.max_sections,
            max_edit_rounds=cfg.max_edit_rounds,
            thinking_effort=cfg.thinking_effort,
            planning_effort=cfg.planning_effort,
            evaluator_enable_thinking=cfg.evaluator_enable_thinking,
            synthesizer_enable_thinking=cfg.synthesizer_enable_thinking,
            callbacks=[tracker],
            max_total_tokens=cfg.max_total_tokens,
        )
        review = await reviewer.run()
        logger.info("Generated the review body, TL;DR and closing assessment.")
        return review, [*review.notes, *_citation_notes(citation_summarizer)]

    async def summary(
        self,
        paper: Paper,
        *,
        translation_guideline: list[dict[str, Any]] | None,
        tracker: UsageTracker,
    ) -> tuple[dict[str, str], list[str]]:
        cfg = self.summary_settings
        summarizer = PaperSummarizer(
            LanguageModelId(cfg.summary_model_id),
            self.boto_session,
            language=self.language,
            translation_guideline=translation_guideline,
            enable_thinking=cfg.summarizer_enable_thinking,
            thinking_effort=cfg.thinking_effort,
            max_total_tokens=cfg.max_total_tokens,
            callbacks=[tracker],
        )
        result = await summarizer.summarize(paper)
        logger.info("Generated the paper summary.")
        return result, list(summarizer.notes)


def _citation_notes(citation_summarizer: CitationSummarizer) -> list[str]:
    """A note when some prerequisite references had no abstract or text to
    summarize (the sections citing them were written without the summary)."""
    notes = []
    for idents, what in (
        (citation_summarizer.unsourced_references(), "had no abstract or text to"),
        (list(dict.fromkeys(citation_summarizer.failed)), "failed to"),
    ):
        if idents:
            shown = ", ".join(f"'{ident[:80]}'" for ident in idents[:5])
            more = f" and {len(idents) - 5} more" if len(idents) > 5 else ""
            notes.append(
                f"{len(idents)} cited reference(s) {what} summarize: {shown}{more}."
            )
    return notes


def code_index(
    code: Any, boto_session: boto3.Session, paper_dir: Path, callbacks: list[Any]
) -> CodeIndex:
    """The searchable code index of the paper's repositories (``code``: the
    configuration's ``code`` section)."""
    return CodeIndex(
        LanguageModelId(code.code_analysis_model_id),
        LanguageModelId(code.code_summarization_model_id),
        EmbeddingModelId(code.embed_model_id),
        paper_dir=paper_dir,
        boto_session=boto_session,
        chunk_size=code.chunk_size,
        chunk_overlap=code.chunk_overlap,
        max_files=code.max_files,
        max_chunks=code.max_chunks,
        callbacks=callbacks,
    )


def citation_summarizer(
    citations: Any, boto_session: boto3.Session, paper_dir: Path, callbacks: list[Any]
) -> CitationSummarizer:
    """The prerequisite summaries of the paper's citations (``citations``:
    the configuration's ``citations`` section)."""
    return CitationSummarizer(
        LanguageModelId(citations.citation_summarization_model_id),
        boto_session,
        paper_dir=paper_dir,
        callbacks=callbacks,
        prefer_full_text=citations.prefer_full_text,
    )
