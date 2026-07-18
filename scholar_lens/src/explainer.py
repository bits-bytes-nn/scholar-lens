import re
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Literal, TypedDict

import boto3
import nltk
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from .citation_summarizer import CitationSummarizer
from .code_retriever import CodeRetriever
from .constants import LanguageModelId
from .content_extractor import Attributes, Citation
from .logger import is_running_in_aws, logger
from .metrics import TokenBudgetGuard
from .parser import Content, Figure
from .prompts import (
    BasePrompt,
    PaperAnalysisPrompt,
    PaperEnrichmentPrompt,
    PaperEvaluationPrompt,
    PaperFinalizationPrompt,
    PaperSynthesisPrompt,
)
from .utils import (
    BedrockLanguageModelFactory,
    HTMLTagOutputParser,
    RetryableBase,
    create_robust_xml_output_parser,
    is_affirmative,
    measure_execution_time,
    parse_quality_score,
)
from .web_research import neutralize_prompt_tags

ROOT_DIR: Path = (
    Path("/tmp")
    if is_running_in_aws()
    else Path(__file__).resolve().parent.parent.parent
)

nltk.data.path.append(str(ROOT_DIR))


def _ensure_nltk_data() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("NLTK resource 'punkt_tab' not found. Downloading...")
        nltk.download("punkt_tab", download_dir=ROOT_DIR)


class ExplainerConfig:
    MAX_ITERS: int = 20
    MAX_SYNTHESIS_ATTEMPTS: int = 3
    MIN_QUALITY_SCORE: int = 70
    RECURSION_LIMIT: int = 200
    SEARCH_RESULTS_LIMIT: int = 10
    # Hard cap on the per-section has_more continuation loop, so a model that
    # keeps emitting has_more=y cannot loop (and bill) forever.
    MAX_CONTINUATIONS: int = 8


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
    table_of_contents: dict[str, Any] = Field(default_factory=dict)
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


class ExplainerState(TypedDict):
    paper: Paper
    paragraphs: list[str]
    structure: dict[str, Any]
    # Analyzer section metadata aligned 1:1 with `paragraphs` (same length/order).
    aligned_sections: list[dict[str, Any]]
    explanations: list[str]
    key_takeaways: str
    current_index: int
    citation_summaries: list[str] | None
    code: list[dict[str, float | str]] | None
    synthesis_attempts: int
    quality_score: int
    accumulated_feedback: list[str]
    # Best (highest-scoring) draft seen for the current section across retries.
    # A retried synthesis can score lower than an earlier attempt; we keep the
    # best rather than blindly using the last (mirrors the tech-guide evaluator).
    best_explanation: str
    best_quality_score: int


class ExplainerGraph(RetryableBase, TokenBudgetGuard):
    def __init__(
        self,
        paper: Paper,
        paper_analysis_model_id: LanguageModelId,
        paper_enrichment_model_id: LanguageModelId,
        paper_finalization_model_id: LanguageModelId,
        paper_evaluation_model_id: LanguageModelId,
        paper_synthesis_model_id: LanguageModelId,
        output_fixing_model_id: LanguageModelId,
        boto_session: boto3.Session,
        citation_summarizer: CitationSummarizer,
        code_retriever: CodeRetriever | None = None,
        translation_guideline: list[list[dict[str, str]]] | None = None,
        max_results: int = ExplainerConfig.SEARCH_RESULTS_LIMIT,
        max_iters: int = ExplainerConfig.MAX_ITERS,
        max_synthesis_attempts: int = ExplainerConfig.MAX_SYNTHESIS_ATTEMPTS,
        min_quality_score: int = ExplainerConfig.MIN_QUALITY_SCORE,
        enable_output_fixing: bool = False,
        evaluator_enable_thinking: bool = False,
        synthesizer_enable_thinking: bool = False,
        thinking_effort: str = "medium",
        language: str = "Korean",
        callbacks: list[Any] | None = None,
        max_total_tokens: int | None = None,
        max_continuations: int = ExplainerConfig.MAX_CONTINUATIONS,
    ) -> None:
        _ensure_nltk_data()

        self.paper = paper
        self.language = language
        self.callbacks = callbacks or []
        self.max_continuations = max_continuations
        # Locate a token tracker among the callbacks for budget enforcement.
        self._init_token_budget(self.callbacks, max_total_tokens)
        self.citation_summarizer = citation_summarizer
        self.code_retriever = code_retriever
        self.translation_guideline = translation_guideline or []
        self.max_results = max_results
        self.max_iters = max_iters
        self.max_synthesis_attempts = max_synthesis_attempts
        self.min_quality_score = min_quality_score

        self.llm_factory = BedrockLanguageModelFactory(boto_session=boto_session)
        self._initialize_chains(
            paper_analysis_model_id,
            paper_enrichment_model_id,
            paper_finalization_model_id,
            paper_evaluation_model_id,
            paper_synthesis_model_id,
            output_fixing_model_id,
            enable_output_fixing,
            evaluator_enable_thinking=evaluator_enable_thinking,
            synthesizer_enable_thinking=synthesizer_enable_thinking,
            thinking_effort=thinking_effort,
        )
        self.workflow = self._create_workflow()

    def _initialize_chains(
        self,
        analysis_id: LanguageModelId,
        enrichment_id: LanguageModelId,
        finalization_id: LanguageModelId,
        evaluation_id: LanguageModelId,
        synthesis_id: LanguageModelId,
        output_fixing_model_id: LanguageModelId,
        enable_output_fixing: bool,
        *,
        evaluator_enable_thinking: bool = False,
        synthesizer_enable_thinking: bool = False,
        thinking_effort: str = "medium",
    ) -> None:
        robust_xml_output_parser = create_robust_xml_output_parser(
            self.llm_factory,
            enable_output_fixing=enable_output_fixing,
            output_fixing_model_id=output_fixing_model_id,
        )
        # The analyzer is the one node that legitimately receives the whole
        # paper, so it gets the 1M window and its input is fitted to the model's
        # budget at call time (see analyze_paper).
        self.analysis_model_id = analysis_id
        self.analyzer = self._create_chain(
            PaperAnalysisPrompt,
            analysis_id,
            robust_xml_output_parser,
            supports_1m_context_window=True,
        )
        self.enricher = self._create_chain(
            PaperEnrichmentPrompt,
            enrichment_id,
            HTMLTagOutputParser(tag_names=PaperEnrichmentPrompt.output_variables),
        )
        self.finalizer = self._create_chain(
            PaperFinalizationPrompt,
            finalization_id,
            HTMLTagOutputParser(tag_names=PaperFinalizationPrompt.output_variables),
        )
        self.evaluator = self._create_chain(
            PaperEvaluationPrompt,
            evaluation_id,
            HTMLTagOutputParser(tag_names=PaperEvaluationPrompt.output_variables),
            enable_thinking=evaluator_enable_thinking,
            thinking_effort=thinking_effort,
        )
        self.synthesizer = self._create_chain(
            PaperSynthesisPrompt,
            synthesis_id,
            StrOutputParser(),
            enable_thinking=synthesizer_enable_thinking,
            thinking_effort=thinking_effort,
            supports_1m_context_window=True,
        )

    def _create_chain(
        self,
        prompt_cls: type[BasePrompt],
        model_id: LanguageModelId,
        output_parser: BaseOutputParser,
        **kwargs: Any,
    ) -> Runnable:
        kwargs.setdefault("callbacks", self.callbacks or None)
        llm = self.llm_factory.get_model(model_id, temperature=0.0, **kwargs)
        return prompt_cls.get_prompt() | llm | output_parser

    def _create_workflow(self) -> CompiledStateGraph:
        def check_continue_node(state: ExplainerState) -> dict[str, Any]:
            # The evaluate loop has exited for this section. Commit the best-scoring
            # draft seen across retries (the last synthesis may have scored lower
            # than an earlier attempt), then reset the per-section best trackers.
            best = state.get("best_explanation", "")
            current_index = state["current_index"]
            explanations = state["explanations"]
            if best and 0 <= current_index < len(explanations):
                if explanations[current_index] != best:
                    explanations = explanations.copy()
                    explanations[current_index] = best
                    logger.info(
                        "Section %d: kept best draft (score %d) over the last "
                        "synthesis.",
                        current_index,
                        state.get("best_quality_score", 0),
                    )
                    return {
                        "explanations": explanations,
                        "best_explanation": "",
                        "best_quality_score": -1,
                    }
            return {"best_explanation": "", "best_quality_score": -1}

        def decide_next_step(
            state: ExplainerState,
        ) -> Literal["go_finalize", "go_update_index"]:
            if state["current_index"] >= len(state["paragraphs"]) - 1:
                logger.info("Processing complete - reached final paragraph")
                return "go_finalize"
            if state["current_index"] >= self.max_iters - 1:
                logger.warning("Processing stopped - maximum iterations reached")
                return "go_finalize"
            return "go_update_index"

        def should_retry_synthesis(
            state: ExplainerState,
        ) -> Literal["go_continue", "go_retry_synthesis"]:
            if state["synthesis_attempts"] >= self.max_synthesis_attempts:
                logger.warning(
                    "Maximum synthesis attempts reached (Quality score: %d) - continuing",
                    state["quality_score"],
                )
                return "go_continue"
            if state["quality_score"] >= self.min_quality_score:
                logger.info(
                    "Quality score %d meets threshold - continuing",
                    state["quality_score"],
                )
                return "go_continue"
            logger.info(
                "Quality score %d below threshold - retrying synthesis",
                state["quality_score"],
            )
            return "go_retry_synthesis"

        def update_index_node(state: ExplainerState) -> dict[str, Any]:
            return {
                "current_index": state["current_index"] + 1,
                "synthesis_attempts": 0,
                "quality_score": 100,
                "accumulated_feedback": [],
                "best_explanation": "",
                "best_quality_score": -1,
            }

        workflow = StateGraph(ExplainerState)

        workflow.add_node("analyze", self.analyze_paper)
        workflow.add_node("enrich", self.enrich_paper)
        workflow.add_node("finalize", self.finalize_paper)
        workflow.add_node("evaluate", self.evaluate_paper)
        workflow.add_node("synthesize", self.synthesize_paper)
        workflow.add_node("check_continue", check_continue_node)
        workflow.add_node("update_index", update_index_node)

        workflow.set_entry_point("analyze")

        workflow.add_edge("analyze", "enrich")
        workflow.add_edge("enrich", "synthesize")
        workflow.add_edge("synthesize", "evaluate")

        workflow.add_conditional_edges(
            "evaluate",
            should_retry_synthesis,
            {
                "go_retry_synthesis": "synthesize",
                "go_continue": "check_continue",
            },
        )

        workflow.add_conditional_edges(
            "check_continue",
            decide_next_step,
            {
                "go_update_index": "update_index",
                "go_finalize": "finalize",
            },
        )

        workflow.add_edge("update_index", "enrich")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    @RetryableBase._retry("paper_analysis")
    def analyze_paper(self, state: ExplainerState) -> dict[str, Any]:
        self._enforce_token_budget()
        # Defang prompt-fence tags in the untrusted paper body once, at the single
        # point it enters the pipeline: the derived sentences/paragraphs flow into
        # every downstream fence (numbered_sentences, current_content, text, …), so
        # a literal "</paper>"/"</current_content>" in the source can't break out.
        text = neutralize_prompt_tags(state["paper"].content.text)
        sentences = nltk.sent_tokenize(text)

        numbered_sentences_str = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
        # Fit to the analyzer model's context window (exact, via CountTokens) so a
        # very long paper drops trailing sentences instead of failing the call;
        # the returned indices only reference sentences that were shown.
        numbered_sentences_str = self.llm_factory.fit_text(
            self.analysis_model_id, numbered_sentences_str, label="paper structure"
        )

        structure = self.analyzer.invoke({"numbered_sentences": numbered_sentences_str})

        paragraphs, aligned_sections = self._extract_paragraphs_by_indices(
            sentences, structure
        )

        logger.info(
            "Extracted %d paragraphs based on sentence indices.",
            len(paragraphs),
        )
        logger.debug("Structure: '%s'", pformat(structure))

        return {
            "paragraphs": paragraphs,
            "structure": structure,
            # Section metadata aligned 1:1 with `paragraphs` (same length, same
            # order). Built alongside the paragraphs so reordered/duplicated/
            # out-of-bounds analyzer indices can't misalign the two lists.
            "aligned_sections": aligned_sections,
            "explanations": [],
            "synthesis_attempts": 0,
            "quality_score": 100,
            "accumulated_feedback": [],
            "best_explanation": "",
            "best_quality_score": -1,
        }

    _EMPTY_SECTION: dict[str, Any] = {
        "section": [{"section_title": "", "key_points": []}]
    }

    @staticmethod
    def _extract_paragraphs_by_indices(
        sentences: list[str], structure: dict[str, Any]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Slice the paper into paragraphs and return per-paragraph section meta.

        Returns ``(paragraphs, aligned_sections)`` where the two lists are the
        same length and ``aligned_sections[i]`` is the analyzer section object
        that owns ``paragraphs[i]``. We pair each valid start index with its
        section object BEFORE sorting/deduping, so reordered, duplicated, or
        out-of-bounds analyzer indices can never misalign the metadata from the
        text (the previous positional ``paper_structure[i - offset]`` lookup broke
        whenever the raw sections weren't already ascending, unique, and in-range).
        """
        pairs: list[tuple[int, dict[str, Any]]] = []
        for section in structure.get("paper_structure", []):
            try:
                start_num_str = section["section"][0]["starting_sentence_number"]
                start_index = int(start_num_str)
                if 0 <= start_index < len(sentences):
                    pairs.append((start_index, section))
                else:
                    logger.warning(
                        "LLM returned an out-of-bounds sentence index: %d. Skipping.",
                        start_index,
                    )
            except (ValueError, KeyError, IndexError) as e:
                logger.warning(
                    "Could not parse starting_sentence_number from section: '%s'. Error: %s",
                    section,
                    e,
                )

        if not pairs:
            logger.warning(
                "No valid section start indices found. Treating entire text as one section."
            )
            return [" ".join(sentences)], [ExplainerGraph._EMPTY_SECTION]

        # Sort by start index and drop duplicate starts (keep the first section
        # that claimed each index), keeping the section object paired throughout.
        pairs.sort(key=lambda p: p[0])
        deduped: list[tuple[int, dict[str, Any]]] = []
        seen_starts: set[int] = set()
        for start_index, section in pairs:
            if start_index not in seen_starts:
                seen_starts.add(start_index)
                deduped.append((start_index, section))

        # Prepend a synthetic index-0 section (empty metadata) when the first real
        # section doesn't start at sentence 0, so the opening text is still covered.
        if deduped[0][0] != 0:
            deduped.insert(0, (0, ExplainerGraph._EMPTY_SECTION))
            logger.debug("Prepended sentence index 0 to ensure full coverage.")

        paragraphs: list[str] = []
        aligned_sections: list[dict[str, Any]] = []
        for i, (start_idx, section) in enumerate(deduped):
            end_idx = deduped[i + 1][0] if i + 1 < len(deduped) else len(sentences)
            paragraphs.append(" ".join(sentences[start_idx:end_idx]))
            aligned_sections.append(section)

        return paragraphs, aligned_sections

    @staticmethod
    def _get_section_analysis(
        state: ExplainerState, current_index: int
    ) -> dict[str, Any]:
        """Return the section metadata aligned with paragraph ``current_index``.

        ``aligned_sections`` is built 1:1 with ``paragraphs`` in
        :meth:`_extract_paragraphs_by_indices`, so this is a direct positional
        lookup with no offset arithmetic.
        """
        aligned_sections = state.get("aligned_sections", [])
        if 0 <= current_index < len(aligned_sections):
            return aligned_sections[current_index]
        logger.info(
            "No matching section analysis for paragraph index %d; using an empty "
            "section title.",
            current_index,
        )
        # Empty title rather than a fabricated "Introduction" (which both
        # mislabels mid-paper sections and leaks an English word into a
        # non-English review). The synthesizer derives the heading itself.
        return ExplainerGraph._EMPTY_SECTION

    @RetryableBase._retry("paper_enrichment")
    async def enrich_paper(self, state: ExplainerState) -> dict[str, Any]:
        self._enforce_token_budget()
        current_index = state["current_index"]
        current_paragraph = state["paragraphs"][current_index]
        current_analysis = self._get_section_analysis(state, current_index)
        citations_text = "\n".join(
            str(citation) for citation in state["paper"].citations
        )

        # current_paragraph is already defanged (neutralized in analyze_paper);
        # citations and the codebase summary are also untrusted, so defang them too.
        result = await self.enricher.ainvoke(
            {
                "text": current_paragraph,
                "analysis": current_analysis,
                "citations": neutralize_prompt_tags(citations_text),
                "codebase_summary": neutralize_prompt_tags(
                    state["paper"].codebase_summary or "No codebase summary available"
                ),
            }
        )

        reference_identifiers = self._parse_reference_identifiers(
            result.get("reference_identifiers", "")
        )
        logger.debug("Reference identifiers: '%s'", reference_identifiers)
        should_search_code = is_affirmative(result.get("should_search_code"))
        logger.debug("Should search code: '%s'", should_search_code)

        citation_summaries = None
        if reference_identifiers:
            try:
                citation_summaries = await self.citation_summarizer.summarize(
                    reference_identifiers, current_paragraph
                )
            except Exception as e:
                # Degrade gracefully (mirrors the code-search path below): a
                # citation failure — e.g. an arXiv 429 escaping the summarizer —
                # must not crash the whole review.
                logger.warning("Failed to summarize citations: '%s'", str(e))

        code = None
        if should_search_code and self.code_retriever:
            try:
                code = await self.code_retriever.search_similar_code(
                    current_paragraph, k=self.max_results
                )
            except Exception as e:
                logger.warning("Failed to search code: '%s'", str(e))

        if citation_summaries:
            logger.debug("Citation summaries: '%s'", pformat(citation_summaries))
        if code:
            logger.info("Found %d code snippets", len(code))
            logger.debug("Code: '%s'", pformat(code))

        return {
            "citation_summaries": citation_summaries,
            "code": code,
        }

    @RetryableBase._retry("paper_finalization")
    def finalize_paper(self, state: ExplainerState) -> dict:
        self._enforce_token_budget()
        result = self.finalizer.invoke(
            {
                "explanation": "\n".join(state["explanations"]),
                "language": self.language,
            }
        )
        key_takeaways_str = result.get("key_takeaways", "").strip()
        return {"key_takeaways": key_takeaways_str}

    @staticmethod
    def _parse_reference_identifiers(reference_identifiers: str) -> list[str]:
        return [
            line.strip() for line in reference_identifiers.splitlines() if line.strip()
        ]

    @RetryableBase._retry("paper_evaluation")
    def evaluate_paper(self, state: ExplainerState) -> dict[str, Any]:
        self._enforce_token_budget()
        current_index = state["current_index"]
        current_explanation = state["explanations"][current_index]
        current_paragraph = state["paragraphs"][current_index]
        section_data = self._get_section_analysis(state, current_index)
        current_analysis = section_data.get("section", [])

        accumulated_feedback = state["accumulated_feedback"].copy()

        if not current_explanation.strip():
            feedback = (
                "The generated explanation was empty or could not be processed. "
                "Produce a more concise explanation for this section."
            )
            accumulated_feedback.append(feedback)
            logger.warning(
                "Empty current_paragraph detected. Adding length reduction feedback."
            )

            return {
                "accumulated_feedback": accumulated_feedback,
                "quality_score": 0,
                "synthesis_attempts": state["synthesis_attempts"] + 1,
            }

        result = self.evaluator.invoke(
            {
                "current_content": current_paragraph,
                "current_explanation": current_explanation,
                "table_of_contents": str(state["paper"].table_of_contents or {}),
                "citation_summaries": str(state.get("citation_summaries") or []),
                "code": str(state.get("code") or []),
                "analysis": current_analysis,
                "translation_guideline": str(self.translation_guideline),
                "language": self.language,
            }
        )

        if feedback := result.get("improvement_feedback"):
            accumulated_feedback.append(feedback)
            logger.debug("Improvement feedback: '%s'", feedback)

        quality_score = parse_quality_score(result.get("quality_score"))

        # Keep the best-scoring draft across retries: a lower-scoring rewrite must
        # not silently replace a better earlier attempt. On a strict improvement
        # (or the first attempt) adopt the current draft as the best.
        update: dict[str, Any] = {
            "accumulated_feedback": accumulated_feedback,
            "quality_score": quality_score,
            "synthesis_attempts": state["synthesis_attempts"] + 1,
        }
        if quality_score > state.get("best_quality_score", -1):
            update["best_explanation"] = current_explanation
            update["best_quality_score"] = quality_score
        return update

    def synthesize_paper(self, state: ExplainerState) -> dict[str, Any]:
        current_index = state["current_index"]
        current_paragraph = state["paragraphs"][current_index]
        explanations = state["explanations"].copy()

        while len(explanations) <= current_index:
            explanations.append("")

        previous_explanation = "\n".join(explanations[:current_index])
        section_data = self._get_section_analysis(state, current_index)
        analysis = section_data.get("section", [])
        final_explanation_for_paragraph = ""

        logger.info(
            "Synthesizing paper... (index: %d, attempt: %d)",
            current_index,
            state["synthesis_attempts"] + 1,
        )

        for continuation in range(self.max_continuations):
            self._enforce_token_budget()
            improvement_feedback = "\n".join(state["accumulated_feedback"])

            result = self._synthesize_paper(
                current_paragraph=current_paragraph,
                previous_explanation=previous_explanation,
                current_explanation=final_explanation_for_paragraph,
                table_of_contents=state["paper"].table_of_contents or {},
                citation_summaries=state.get("citation_summaries"),
                code=state.get("code"),
                analysis=analysis,
                improvement_feedback=improvement_feedback,
            )

            logger.debug("Synthesized chunk:\n%s", result["explanation"])

            if explanation_chunk := result.get("explanation", "").strip():
                final_explanation_for_paragraph += explanation_chunk + "\n"

            if not is_affirmative(result.get("has_more")):
                break
            if continuation == self.max_continuations - 1:
                logger.warning(
                    "Reached max continuations (%d) for section %d; stopping.",
                    self.max_continuations,
                    current_index,
                )

        explanations[current_index] = final_explanation_for_paragraph.strip()

        # Do NOT reset citation_summaries/code here. This node is followed by
        # `evaluate` (which scores the draft against those same references) and,
        # on a low score, a retry that re-enters this node — both need the
        # enrichment context. The next section's `enrich` unconditionally
        # overwrites both keys, so a reset here is redundant as well as harmful.
        return {"explanations": explanations}

    @RetryableBase._retry("paper_synthesis")
    def _synthesize_paper(
        self,
        *,
        current_paragraph: str,
        previous_explanation: str,
        current_explanation: str,
        table_of_contents: dict[str, Any],
        citation_summaries: list[str] | None,
        code: list[dict[str, float | str]] | None,
        analysis: dict[str, Any],
        improvement_feedback: str,
    ) -> dict[str, str]:
        result = self.synthesizer.invoke(
            {
                "current_content": current_paragraph,
                "previous_explanation": previous_explanation,
                "current_explanation": current_explanation,
                "table_of_contents": str(table_of_contents),
                "citation_summaries": str(citation_summaries or []),
                "code": str(code or []),
                "analysis": analysis,
                "translation_guideline": str(self.translation_guideline),
                "improvement_feedback": improvement_feedback,
                "language": self.language,
            }
        )
        has_more_match = re.search(r"<has_more>(.*?)</has_more>", result, re.DOTALL)
        explanation_match = re.search(
            r"<explanation>(.*?)</explanation>", result, re.DOTALL
        )
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            # The model omitted the wrapper tag — don't drop the whole chunk; use
            # the raw output minus any has_more tag so content isn't silently lost.
            explanation = re.sub(
                r"<has_more>.*?</has_more>", "", result, flags=re.DOTALL
            ).strip()

        return {
            "explanation": explanation,
            "has_more": has_more_match.group(1).strip() if has_more_match else "n",
        }

    @measure_execution_time
    async def run(self) -> tuple[str, str]:
        initial_state: ExplainerState = {
            "paper": self.paper,
            "paragraphs": [],
            "structure": {},
            "aligned_sections": [],
            "explanations": [],
            "key_takeaways": "",
            "current_index": 0,
            "citation_summaries": None,
            "code": None,
            "synthesis_attempts": 0,
            "quality_score": 100,
            "accumulated_feedback": [],
            "best_explanation": "",
            "best_quality_score": -1,
        }
        final_state = await self.workflow.ainvoke(
            initial_state,
            RunnableConfig(recursion_limit=ExplainerConfig.RECURSION_LIMIT),
        )

        return "\n".join(final_state["explanations"]), final_state["key_takeaways"]
