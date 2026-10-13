"""Paper review workflow: plan → enrich → write + check → edit → finalize.

1. **Plan** — one call reads the whole paper and returns a :class:`ReviewPlan`:
   ordered sections with must-cover items, difficulty for a general DS/MLE
   reader, prerequisites, concept ownership, figures and a glossary. Code then
   normalizes it (:func:`normalize_plan`).
2. **Enrich** — per section, citation summaries for the planned prerequisites
   and code snippets where the plan asks for them; no extra LLM routing call.
3. **Write + check** — each section is written from its brief, the article so
   far and the glossary, then checked against the brief; drafts below the bar
   (or with missing items) are revised from the best draft.
4. **Edit** — one pass over the whole article returns targeted edits for
   terminology, cross-section repetition, transitions and coverage gaps,
   applied exact-match in code; a second round runs only when the first
   reported gaps or had edits skipped.
5. **Finalize** — subtitle, TL;DR and the closing assessment.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import boto3
import nltk
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.runnables import Runnable
from nltk.tokenize.punkt import PunktTokenizer
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from .citation_summarizer import CitationSummarizer
from .code_retriever import CodeRetriever
from .constants import LanguageModelId
from .content_extractor import Attributes, Citation
from .formatting import same_as_title
from .logger import is_running_in_aws, logger
from .markdown_lint import writing_issues
from .metrics import TokenBudgetGuard
from .parser import Content, Figure
from .prompt_safety import neutralize_prompt_tags
from .prompts import (
    BasePrompt,
    PaperEditPrompt,
    PaperFinalizationPrompt,
    PaperPlanningPrompt,
    PaperSectionCheckPrompt,
    PaperSectionPrompt,
)
from .review_plan import (
    RANGE_PADDING,
    PlanError,
    ReviewPlan,
    SourceText,
    apply_edits,
    approx_words,
    article_ledger,
    check_plan_shape,
    concepts_owned_elsewhere,
    draft_rank,
    ensure_section_heading,
    figure_catalog,
    marker_src,
    normalize_plan,
    number_sentences,
    parse_edits,
    parse_lines,
    parse_plan,
    previous_context,
    publish_blockers,
    rejected_revision_note,
    render_brief,
    render_closing_points,
    render_coverage,
    render_glossary,
    render_outline,
    section_source,
    tag_body,
)
from .utils import (
    BedrockLanguageModelFactory,
    HTMLTagOutputParser,
    RetryableBase,
    is_affirmative,
    measure_execution_time,
    parse_quality_score,
)

ROOT_DIR: Path = (
    Path("/tmp")
    if is_running_in_aws()
    else Path(__file__).resolve().parent.parent.parent
)

nltk.data.path.append(str(ROOT_DIR))

# Language-neutral sentinel for "no code repository was provided"; the review
# and summary prompts key off this exact string.
NO_CODEBASE = "(no code repository provided)"
_TAG_EXPLANATION = "explanation"


def _ensure_nltk_data() -> None:
    """Use the installed ``punkt_tab`` (e.g. baked into the image via
    ``NLTK_DATA``); download it only when it is missing."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
        return
    except LookupError:
        logger.info("NLTK resource 'punkt_tab' not found. Downloading...")
    if not nltk.download("punkt_tab", download_dir=str(ROOT_DIR), quiet=True):
        raise RuntimeError(
            "NLTK resource 'punkt_tab' is missing and could not be downloaded; "
            "install it (python -m nltk.downloader punkt_tab) or set NLTK_DATA."
        )


class Paper(BaseModel):
    model_config = ConfigDict(frozen=True)
    arxiv_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    published: datetime
    pdf_url: HttpUrl
    content: Content
    attributes: Attributes
    citations: list[Citation] = Field(default_factory=list)
    updated: datetime | None = None
    abstract: str | None = None
    journal_ref: str | None = None
    primary_category: str | None = None
    categories: list[str] = Field(default_factory=list)
    doi: str | None = None
    figures: list[Figure] = Field(default_factory=list)
    repo_urls: list[HttpUrl] = Field(default_factory=list)
    codebase_summary: str | None = None
    is_pdf_parsed: bool = Field(default=False)

    @field_validator("repo_urls", mode="before")
    @classmethod
    def convert_str_urls(cls, v: Any) -> list[HttpUrl]:
        if v is None:
            return []
        return [HttpUrl(url) for url in v]


class ReviewText(NamedTuple):
    """The generated review: body sections, TL;DR, closing assessment, and the
    title translated into the output language (the post's subtitle)."""

    body: str
    key_takeaways: str
    closing: str
    subtitle: str = ""
    # Degraded-output notes (sections skipped or below the bar, unusable
    # finalization) for the PR checklist and the job result.
    notes: tuple[str, ...] = ()


@dataclass
class SectionCheck:
    score: int
    missing: list[str] = field(default_factory=list)
    feedback: str = ""
    # Blocking defects (the checker's repetition, wrong derived numbers and
    # unsupported claims, plus deterministic rule slips): a section never
    # passes with any.
    must_fix: list[str] = field(default_factory=list)
    # The subset of must-fix the blog validator would still reject after the
    # publisher's own fixes: a draft with any can never be published.
    blockers: list[str] = field(default_factory=list)

    def passes(self, min_score: int) -> bool:
        return self.score >= min_score and not self.missing and not self.must_fix

    def better_than(self, other: SectionCheck | None, min_score: int) -> bool:
        if other is None:
            return True
        return self._rank(min_score) > other._rank(min_score)

    def _rank(self, min_score: int) -> tuple[bool, bool, int, int]:
        return draft_rank(
            passes=self.passes(min_score),
            publishable=not self.blockers,
            blocking=len(self.missing) + len(self.must_fix),
            score=self.score,
        )


@dataclass
class Enrichment:
    citation_summaries: list[str] = field(default_factory=list)
    code: list[dict[str, Any]] = field(default_factory=list)


# Output ceiling for the planner and editor (whole-paper inputs, high effort).
_WHOLE_PAPER_MAX_TOKENS = 128_000


class PaperReviewer(RetryableBase, TokenBudgetGuard):
    """Generates a :class:`ReviewText` for one :class:`Paper`."""

    MAX_CONTINUATIONS = 4
    MAX_EDITS = 40
    MAX_REFERENCES = 300

    def __init__(
        self,
        paper: Paper,
        *,
        planning_model_id: LanguageModelId,
        synthesis_model_id: LanguageModelId,
        evaluation_model_id: LanguageModelId,
        finalization_model_id: LanguageModelId,
        boto_session: boto3.Session,
        citation_summarizer: CitationSummarizer,
        code_retriever: CodeRetriever | None = None,
        translation_guideline: list[dict[str, Any]] | None = None,
        language: str = "Korean",
        min_quality_score: int = 70,
        max_synthesis_attempts: int = 3,
        max_sections: int = 20,
        max_edit_rounds: int = 2,
        thinking_effort: str = "medium",
        planning_effort: str = "high",
        evaluator_enable_thinking: bool = False,
        synthesizer_enable_thinking: bool = False,
        max_code_results: int = 8,
        callbacks: list[Any] | None = None,
        max_total_tokens: int | None = None,
    ) -> None:
        _ensure_nltk_data()
        self.paper = paper
        self.language = language
        self.callbacks = callbacks or []
        self._init_token_budget(self.callbacks, max_total_tokens)
        self.citation_summarizer = citation_summarizer
        self.code_retriever = code_retriever
        self.translation_guideline = translation_guideline or []
        self.min_quality_score = min_quality_score
        self.max_synthesis_attempts = max(1, max_synthesis_attempts)
        self.max_sections = max_sections
        self.max_edit_rounds = max_edit_rounds
        self.max_code_results = max_code_results
        self.planning_model_id = planning_model_id
        self.notes: list[str] = []
        # Defects a below-bar section still carries, quoted for the editor,
        # and the coverage it still lacks (nothing to quote).
        self.open_defects: list[str] = []
        self.open_gaps: list[str] = []
        self.llm_factory = BedrockLanguageModelFactory(boto_session=boto_session)

        self.planner = self._chain(
            PaperPlanningPrompt,
            planning_model_id,
            StrOutputParser(),
            enable_thinking=True,
            thinking_effort=planning_effort,
            supports_1m_context_window=True,
            # Whole-paper calls with long thinking: allow the model's full
            # output budget (capped to its maximum by the factory).
            max_tokens=_WHOLE_PAPER_MAX_TOKENS,
        )
        self.writer = self._chain(
            PaperSectionPrompt,
            synthesis_model_id,
            StrOutputParser(),
            enable_thinking=synthesizer_enable_thinking,
            thinking_effort=thinking_effort,
            supports_1m_context_window=True,
        )
        self.checker = self._chain(
            PaperSectionCheckPrompt,
            evaluation_model_id,
            HTMLTagOutputParser(tag_names=PaperSectionCheckPrompt.output_variables),
            enable_thinking=evaluator_enable_thinking,
            thinking_effort=thinking_effort,
        )
        # Raw text: edits are parsed verbatim (LaTeX backslashes, `&`).
        self.editor = self._chain(
            PaperEditPrompt,
            planning_model_id,
            StrOutputParser(),
            enable_thinking=True,
            thinking_effort=planning_effort,
            supports_1m_context_window=True,
            # Whole-paper calls with long thinking: allow the model's full
            # output budget (capped to its maximum by the factory).
            max_tokens=_WHOLE_PAPER_MAX_TOKENS,
        )
        self.finalizer = self._chain(
            PaperFinalizationPrompt,
            finalization_model_id,
            HTMLTagOutputParser(tag_names=PaperFinalizationPrompt.output_variables),
            supports_1m_context_window=True,
        )

    def _chain(
        self,
        prompt_cls: type[BasePrompt],
        model_id: LanguageModelId,
        output_parser: BaseOutputParser,
        **kwargs: Any,
    ) -> Runnable:
        llm = self.llm_factory.get_model(
            model_id, temperature=0.0, callbacks=self.callbacks or None, **kwargs
        )
        return prompt_cls.get_prompt() | llm | output_parser

    @RetryableBase._retry("paper_review_llm_call")
    async def _call(self, chain: Runnable, payload: dict[str, Any]) -> Any:
        self._enforce_token_budget()
        return await chain.ainvoke(payload)

    # ---- inputs -----------------------------------------------------------

    def _source(self) -> SourceText:
        """The paper text split into anchorable units (sentences; headings,
        tables, code and display math kept whole). Prompt-fence tags are
        defanged once here: every downstream fence quotes this text."""
        text = neutralize_prompt_tags(self.paper.content.text)
        tokenizer = PunktTokenizer("english")
        return SourceText.from_text(
            text, lambda chunk: list(tokenizer.span_tokenize(chunk))
        )

    def _references(self) -> tuple[str, dict[str, str]]:
        """Reference list shown to the planner, and ``R<n>`` → identifier
        (arXiv id when known, else the title) for the citation summarizer."""
        lines, ids = [], {}
        for i, citation in enumerate(self.paper.citations[: self.MAX_REFERENCES], 1):
            ref = f"R{i}"
            ids[ref] = citation.arxiv_id or citation.title
            year = f" ({citation.year})" if citation.year else ""
            arxiv = f" [arXiv:{citation.arxiv_id}]" if citation.arxiv_id else ""
            lines.append(f"{ref}: {citation.title} — {citation.authors}{year}{arxiv}")
        return neutralize_prompt_tags("\n".join(lines) or "- none"), ids

    def _guideline(self) -> str:
        return str(self.translation_guideline or "")

    # ---- stages -----------------------------------------------------------

    async def _plan(
        self, source: SourceText, figures: list[tuple[str, str]]
    ) -> ReviewPlan:
        references, _ = self._references()
        numbered = self.llm_factory.fit_text(
            self.planning_model_id, number_sentences(source), label="review plan"
        )
        payload = {
            "title": self.paper.title,
            "numbered_sentences": numbered,
            "figures": "\n".join(f"{fid}: {marker}" for fid, marker in figures)
            or "- none",
            "references": references,
            "codebase_summary": neutralize_prompt_tags(
                self.paper.codebase_summary or NO_CODEBASE
            ),
            "translation_guideline": self._guideline(),
            "repair_note": "",
            "language": self.language,
        }
        figure_ids = {fid: marker_src(marker) for fid, marker in figures}
        last_error: PlanError | None = None
        for attempt in range(2):
            raw = await self._call(self.planner, payload)
            try:
                plan = parse_plan(raw)
                check_plan_shape(plan, sentence_count=len(source))
            except PlanError as e:
                last_error = e
                logger.warning("Review plan unusable (attempt %d): %s", attempt + 1, e)
                payload["repair_note"] = (
                    "Your previous plan could not be used: "
                    f"{str(e)[:500]}. Return the complete plan again as valid "
                    "JSON inside <plan> tags, following the schema exactly."
                )
                continue
            plan = normalize_plan(
                plan,
                sentence_count=len(source),
                figure_ids=figure_ids,
                max_sections=self.max_sections,
                closing_ranges=source.closing_ranges(),
            )
            logger.info(
                "Review plan: %d sections (%s).",
                len(plan.sections),
                ", ".join(s.difficulty for s in plan.sections),
            )
            return plan
        raise PlanError(f"The planner did not return a usable plan: {last_error}")

    async def _enrich(
        self, plan: ReviewPlan, index: int, source: str, reference_ids: dict[str, str]
    ) -> Enrichment:
        section = plan.sections[index]
        enrichment = Enrichment()
        identifiers = [
            reference_ids[p.reference.strip()]
            for p in section.prerequisites
            if p.reference and p.reference.strip() in reference_ids
        ]
        if identifiers:
            try:
                summaries = await self.citation_summarizer.summarize(
                    list(dict.fromkeys(identifiers)), source
                )
                enrichment.citation_summaries = [
                    s for s in summaries if CitationSummarizer.FAILURE_STRING not in s
                ]
            except Exception as e:  # noqa: BLE001 - supplementary input
                logger.warning(
                    "Citation summaries failed for '%s': %s", section.title, e
                )
        if section.needs_code and self.code_retriever is not None:
            try:
                enrichment.code = await self.code_retriever.search_similar_code(
                    source, k=self.max_code_results
                )
            except Exception as e:  # noqa: BLE001 - supplementary input
                logger.warning("Code search failed for '%s': %s", section.title, e)
        return enrichment

    async def _write(
        self,
        base: dict[str, Any],
        title: str,
        *,
        draft_to_revise: str = "",
        feedback: str = "",
    ) -> tuple[str, list[str]]:
        """One draft of a section, following has_more continuations; also the
        must-cover items the writer reports the source cannot support."""
        written, missing_source = "", []
        for _ in range(self.MAX_CONTINUATIONS):
            raw = await self._call(
                self.writer,
                {
                    **base,
                    "current_explanation": written,
                    "draft_to_revise": draft_to_revise,
                    "improvement_feedback": feedback,
                },
            )
            chunk, has_more = _split_writer_output(raw)
            missing_source += parse_lines(tag_body(raw, "missing_source"))
            if written:
                chunk = _strip_repeated_heading(chunk, title)
            if chunk:
                written = f"{written}\n\n{chunk}".strip()
            if not has_more:
                break
        return written, list(dict.fromkeys(missing_source))

    async def _check(
        self,
        base: dict[str, Any],
        draft: str,
        target_words: int,
        difficulty: str,
        unsupported: list[str],
    ) -> SectionCheck:
        result = await self._call(
            self.checker,
            {
                "section_brief": base["section_brief"],
                "source_text": base["source_text"],
                "owned_elsewhere": base["owned_elsewhere"],
                "already_in_article": base["already_in_article"],
                "glossary": base["glossary"],
                "length_report": (
                    f"Draft: about {approx_words(draft)} English-word equivalents. "
                    f"Target: about {target_words} for a {difficulty} section."
                ),
                "citation_summaries": base["citation_summaries"],
                "code": base["code"],
                "draft": draft,
                "unsupported_items": "\n".join(f"- {i}" for i in unsupported)
                or "- none",
                "language": self.language,
            },
        )
        return SectionCheck(
            score=parse_quality_score(result.get("quality_score")),
            missing=parse_lines(result.get("missing_items")),
            feedback=(result.get("improvement_feedback") or "").strip(),
            must_fix=parse_lines(result.get("must_fix")),
        )

    async def _section(
        self,
        plan: ReviewPlan,
        index: int,
        source: str,
        enrichment: Enrichment,
        previous: list[str],
        figure_markers: dict[str, str],
    ) -> str:
        section = plan.sections[index]
        base = {
            "plan_outline": render_outline(plan),
            "section_brief": render_brief(plan, index, figure_markers),
            "source_text": source or "(no direct source text; see the brief)",
            "citation_summaries": "\n\n".join(enrichment.citation_summaries)
            or "- none",
            "code": render_code(enrichment.code),
            "glossary": render_glossary(plan),
            "owned_elsewhere": concepts_owned_elsewhere(plan, index),
            "already_in_article": article_ledger(previous),
            "translation_guideline": self._guideline(),
            "previous_sections": previous_context(previous),
            "language": self.language,
        }
        best_draft, best_check = "", None
        draft_to_revise, feedback = "", ""
        previous_check: SectionCheck | None = None
        for attempt in range(1, self.max_synthesis_attempts + 1):
            draft, missing_source = await self._write(
                base, section.title, draft_to_revise=draft_to_revise, feedback=feedback
            )
            if not draft:
                logger.warning(
                    "Section '%s': empty draft (attempt %d).", section.title, attempt
                )
                empty = "The previous draft was empty. Write the full section."
                # Revising the kept draft: its checker feedback still applies.
                feedback = "\n\n".join(
                    part for part in (_revision_feedback(best_check), empty) if part
                )
                continue
            check = await self._check(
                base, draft, section.target_words, section.difficulty, missing_source
            )
            # Deterministic rule slips are the writer's to fix (one owner: this
            # loop); the checker itself weighs the writer's unsupported items.
            check.blockers = publish_blockers(draft)
            check.must_fix += writing_issues(draft) + check.blockers
            logger.info(
                "Section %d '%s' attempt %d: score %d, %d missing, %d must-fix.",
                index + 1,
                section.title,
                attempt,
                check.score,
                len(check.missing),
                len(check.must_fix),
            )
            for problem in [*check.missing, *check.must_fix]:
                logger.info("  - %s", problem[:200])
            # A passing draft always wins, even over a higher-scoring one that
            # still has blocking defects.
            if check.passes(self.min_quality_score):
                best_draft, best_check = draft, check
                break
            worse = best_check is not None and not check.better_than(
                best_check, self.min_quality_score
            )
            if not worse:
                best_draft, best_check = draft, check
            if _stalled(check, previous_check, self.min_quality_score):
                # Over the bar with full coverage, and another revision did not
                # reduce the blocking items: more attempts only trade one
                # subjective defect for another.
                break
            previous_check = check
            draft_to_revise = best_draft
            feedback = _revision_feedback(best_check)
            if worse:
                # Revising the same draft with the same feedback would repeat
                # the failed attempt; say what went wrong in it.
                note = rejected_revision_note(
                    check.score, len(check.missing) + len(check.must_fix)
                )
                rejected = _revision_feedback(check)
                feedback += f"\n\n{note}" + (
                    f"\n{rejected}" if rejected and rejected != feedback else ""
                )
        if best_check is not None and not best_check.passes(self.min_quality_score):
            open_items = [*best_check.missing, *best_check.must_fix]
            self.open_defects += [
                f"[{section.title}] {item}" for item in best_check.must_fix
            ]
            self.open_gaps += [
                f"[{section.title}] {item}" for item in best_check.missing
            ]
            self.notes.append(
                f"Review section '{section.title}' was kept with "
                f"{len(open_items)} open must-fix/missing item(s) (score "
                f"{best_check.score}); the editor was asked to fix them: check it. "
                "Open: " + "; ".join(item[:120] for item in open_items[:3])
            )
        return ensure_section_heading(best_draft, section.title) if best_draft else ""

    async def _edit(self, plan: ReviewPlan, article: str, paper_text: str) -> str:
        notes: list[str] = []
        if self.open_defects:
            # The section loop ran out of attempts on these: they are still in
            # the article, and the edit pass is the last chance to fix them.
            notes.append(
                "These defects a section checker found are still in the article; "
                "fix each with an edit (the quoted text is from the article):\n"
                + "\n".join(f"- {d[:400]}" for d in self.open_defects[:20])
            )
        if self.open_gaps:
            notes.append(
                "A section checker found these points missing from the article; "
                "add each only where the paper supports it, otherwise leave it out:\n"
                + "\n".join(f"- {d[:400]}" for d in self.open_gaps[:20])
            )
        for round_number in range(1, self.max_edit_rounds + 1):
            round_note = "\n\n".join(notes)
            reserve = len(article) // 2 + 40_000
            raw = await self._call(
                self.editor,
                {
                    "article": article,
                    "paper_text": self.llm_factory.fit_text(
                        self.planning_model_id,
                        paper_text,
                        reserve_tokens=reserve,
                        label="review edit",
                    ),
                    "coverage_plan": render_coverage(plan),
                    "glossary": render_glossary(plan),
                    "round_note": round_note,
                    "max_edits": str(self.MAX_EDITS),
                    "language": self.language,
                },
            )
            edits = parse_edits(raw)
            article, applied, skipped = apply_edits(
                article, edits, max_edits=self.MAX_EDITS
            )
            missing = parse_lines(tag_body(raw, "missing_coverage"))
            logger.info(
                "Edit round %d: %d of %d edit(s) applied, %d gap(s) reported.",
                round_number,
                applied,
                len(edits),
                len(missing),
            )
            if skipped:
                logger.warning(
                    "Edit round %d skipped %d edit(s): %s",
                    round_number,
                    len(skipped),
                    "; ".join(skipped[:5]),
                )
            if not missing and not skipped:
                break
            notes = []
            if missing:
                notes.append(
                    "A previous editing pass reported these plan items as still "
                    "missing from the article. Add each where it belongs:\n"
                    + "\n".join(f"- {m}" for m in missing)
                )
            if skipped:
                notes.append(
                    "These edits from the previous pass could not be applied; "
                    "redo the ones still needed with a find copied exactly once "
                    "from the current article:\n"
                    + "\n".join(f"- {r}" for r in skipped[:20])
                )
        return article

    def _closing_points(self, plan: ReviewPlan) -> str:
        """The planner's closing points plus the paper's own conclusion text,
        which no body section covers."""
        source = self._source()
        conclusion = "\n\n".join(
            source.slice(a, b) for a, b in source.closing_ranges()
        ).strip()
        points = render_closing_points(plan)
        if not conclusion:
            return points
        return f"{points}\n\nThe paper's own conclusion:\n{conclusion[:12000]}"

    async def _finalize(self, plan: ReviewPlan, article: str) -> dict[str, str]:
        payload = {
            "explanation": article,
            "title": self.paper.title,
            "glossary": render_glossary(plan),
            "closing_points": self._closing_points(plan),
            "repair_note": "",
            "language": self.language,
        }
        # Each part is kept from the first attempt where it is well formed, so
        # a repair never replaces a valid part with a worse one.
        final: dict[str, str] = {}
        problems: dict[str, str] = {}
        for attempt in range(2):
            result = await self._call(self.finalizer, payload)
            answer = {k: (result.get(k) or "").strip() for k in _FINAL_PARTS}
            answer["key_takeaways"] = _strip_tldr_heading(answer["key_takeaways"])
            found = _finalization_problems(answer, self.paper.title)
            for part in _FINAL_PARTS:
                # Replace a kept part only with a well-formed one, or an empty
                # kept part with anything.
                if part not in final or (
                    part in problems and (part not in found or not final[part])
                ):
                    final[part] = answer[part]
                    if part in found:
                        problems[part] = found[part]
                    else:
                        problems.pop(part, None)
            if not problems:
                break
            logger.warning(
                "Finalization attempt %d unusable: %s",
                attempt + 1,
                "; ".join(problems.values()),
            )
            payload["repair_note"] = (
                "Your previous answer could not be used: "
                + "; ".join(problems.values())
                + ". Return all three parts again in the exact format."
            )
        else:
            self.notes.append(
                "Review finalization still malformed after a repair attempt: "
                + "; ".join(problems.values())
            )
        return final

    @measure_execution_time
    async def run(self) -> ReviewText:
        paper = self._source()
        figures = figure_catalog(paper.text)
        figure_markers = {marker_src(m): m for _, m in figures}
        plan = await self._plan(paper, figures)
        _, reference_ids = self._references()

        written: list[str] = []
        for index, section in enumerate(plan.sections):
            source = section_source(
                section,
                paper,
                figure_markers,
                excluded=plan.excluded_ranges,
                pad=RANGE_PADDING,
            )
            enrichment = await self._enrich(plan, index, source, reference_ids)
            text = await self._section(
                plan, index, source, enrichment, written, figure_markers
            )
            if text:
                written.append(text)
            else:
                logger.warning("Section '%s' produced no text; skipped.", section.title)
                self.notes.append(
                    f"Review section '{section.title}' produced no text and was "
                    "skipped: its must-cover items are missing."
                )
        if not written:
            raise RuntimeError("The review produced no sections.")

        article = "\n\n".join(written)
        if self.max_edit_rounds:
            article = await self._edit(plan, article, paper.text)
        final = await self._finalize(plan, article)
        return ReviewText(
            body=article,
            key_takeaways=final["key_takeaways"],
            closing=final["closing"],
            subtitle=final["subtitle"],
            notes=tuple(self.notes),
        )


def _split_writer_output(raw: str) -> tuple[str, bool]:
    """``(section_markdown, has_more)`` from the writer's tagged output; a
    missing or unbalanced wrapper keeps the content without the tags."""
    has_more = re.search(r"<has_more>(.*?)</has_more>", raw or "", re.DOTALL)
    body = re.search(
        rf"<{_TAG_EXPLANATION}>(.*?)</{_TAG_EXPLANATION}>", raw or "", re.DOTALL
    )
    if body:
        text = body.group(1)
    else:
        text = re.sub(
            r"<(has_more|missing_source)>(?:.*?</\1>|.*\Z)",
            "",
            raw or "",
            flags=re.DOTALL,
        )  # an unclosed tag runs to the end: drop its value too
        text = re.sub(r"</?(?:explanation|has_more|missing_source)>", "", text)
    # Entities would leak into the post and break the editor's exact matching.
    text = html.unescape(text)
    return text.strip(), bool(has_more and is_affirmative(has_more.group(1).strip()))


def _strip_repeated_heading(chunk: str, title: str) -> str:
    """A continuation chunk must not restart the section with its heading."""
    match = re.match(r"\s*##\s+([^\n]+)\n+", chunk)
    if match and match.group(1).strip() == title.strip():
        return chunk[match.end() :]
    return chunk


def render_code(code: list[dict[str, Any]]) -> str:
    """Retrieved snippets as fenced blocks labelled with their source file."""
    blocks = []
    for item in code:
        content = str(item.get("content", "")).strip()
        if content:
            source = item.get("source") or "repository"
            blocks.append(f"# source: {source}\n```python\n{content}\n```")
    return "\n\n".join(blocks) or "- none"


_TLDR_HEADING = re.compile(r"\A\s*##\s*TL;?DR\s*\n+", re.IGNORECASE)


def _strip_tldr_heading(text: str) -> str:
    return _TLDR_HEADING.sub("", text)


_FINAL_PARTS = ("subtitle", "key_takeaways", "closing")


def _finalization_problems(final: dict[str, str], title: str) -> dict[str, str]:
    """Structural problems of the finalizer's answer, by part."""
    problems: dict[str, str] = {}
    if not final["subtitle"]:
        problems["subtitle"] = "subtitle is empty"
    elif same_as_title(final["subtitle"], title):
        problems["subtitle"] = (
            "subtitle only repeats the title; write a short descriptive subtitle"
        )
    questions = re.findall(r"^###\s+\S", final["key_takeaways"], re.MULTILINE)
    if len(questions) != 4:
        problems["key_takeaways"] = (
            f"key_takeaways must have exactly four ### questions (found {len(questions)})"
        )
    if not final["closing"]:
        problems["closing"] = "closing is empty"
    elif not re.match(r"##\s+\S", final["closing"].lstrip()):
        problems["closing"] = "closing must start with its own '## ' heading"
    # Errors the validator would reject the whole post for.
    for part in ("key_takeaways", "closing"):
        if (
            part not in problems
            and final[part]
            and (blockers := publish_blockers(final[part]))
        ):
            problems[part] = "; ".join(blockers[:3])
    return problems


def _stalled(
    check: SectionCheck, previous: SectionCheck | None, min_score: int
) -> bool:
    """A revision that cleared the score bar and covers everything but did not
    reduce the must-fix items: stop revising. Never while the draft still has
    an error the validator would reject (those are fixable, not subjective)."""
    return (
        previous is not None
        and not check.blockers
        and check.score >= min_score
        and not check.missing
        and len(check.must_fix) >= len(previous.must_fix)
    )


def _revision_feedback(check: SectionCheck | None) -> str:
    if check is None:
        return ""
    parts = [check.feedback] if check.feedback else []
    if check.must_fix:
        parts.append(
            "Blocking defects (fix every one):\n"
            + "\n".join(f"- {m}" for m in check.must_fix)
        )
    if check.missing:
        parts.append(
            "Missing must-cover items (add each with the depth it needs):\n"
            + "\n".join(f"- {m}" for m in check.missing)
        )
    return "\n\n".join(parts)
