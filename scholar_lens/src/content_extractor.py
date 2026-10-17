import ast
import re
from collections.abc import Sequence
from typing import Any

import boto3
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field, field_validator

from .constants import LanguageModelId, arxiv_id_from_reference
from .logger import logger
from .prompt_safety import neutralize_prompt_tags
from .prompts import (
    AttributesExtractionPrompt,
    CitationExtractionPrompt,
)
from .utils import (
    BedrockLanguageModelFactory,
    HTMLTagOutputParser,
    RetryableBase,
    is_affirmative,
    is_placeholder,
)


class Attributes(BaseModel):
    affiliation: str = Field(min_length=1)
    category: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    # Title/authors parsed from the PDF text — used to fill metadata for sources
    # (e.g. arbitrary PDF URLs) that don't carry it. "N/A"/empty means unknown.
    title: str | None = Field(default=None)
    authors: list[str] = Field(default_factory=list)
    # Blog taxonomy picked from the closed controlled-tag list, and a shorter
    # title for search snippets when the real one exceeds ~70 characters.
    controlled_tags: list[str] = Field(default_factory=list)
    seo_title: str | None = Field(default=None)


class Citation(BaseModel):
    authors: str = Field(min_length=1)
    year: int | None = Field(default=None)
    title: str = Field(min_length=1)
    arxiv_id: str | None = Field(default=None)

    @field_validator("arxiv_id", mode="before")
    @classmethod
    def validate_arxiv_id(cls, v: Any) -> str | None:
        # An unusable id ("N/A", a DOI) must not drop the whole citation.
        return arxiv_id_from_reference(str(v)) if v is not None else None

    @field_validator("year", mode="before")
    @classmethod
    def validate_year(cls, v: Any) -> int | None:
        # Models write "2017a", "2020, July" or "n.d."; keep the 4-digit year.
        if v is None or isinstance(v, int):
            return v
        match = re.search(r"(19|20)\d{2}", str(v))
        return int(match.group()) if match else None

    def __repr__(self) -> str:
        return f"Citation(title='{self.title}', arxiv_id='{self.arxiv_id}')"


class ContentExtractor(RetryableBase):
    # Hard cap on the paginated citation-extraction loop, so a model that keeps
    # emitting has_more=y (or novel near-duplicate citations) cannot loop forever.
    MAX_CITATION_PAGES: int = 20
    # Attributes-prompt taxonomy overrides (empty: the prompt's defaults apply).
    _taxonomy: dict[str, str] = {}

    def __init__(
        self,
        citation_extraction_model_id: LanguageModelId,
        attributes_extraction_model_id: LanguageModelId,
        boto_session: boto3.Session,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
        category_options: list[str] | None = None,
        controlled_tags: list[str] | None = None,
        existing_keywords: list[str] | None = None,
    ) -> None:
        # Taxonomy overrides for the attributes prompt; None keeps its defaults.
        self._taxonomy: dict[str, str] = {}
        if category_options:
            self._taxonomy["category_options"] = ", ".join(category_options)
        if controlled_tags:
            self._taxonomy["controlled_tags"] = ", ".join(controlled_tags)
        self.boto_session = boto_session
        # Wire the run's TokenUsageTracker (passed as a callback) into every model
        # so the full-text citation/attributes extraction — among the largest
        # single inputs in a run — counts toward the token budget and cost metric,
        # not just the generation stage.
        self._callbacks = list(callbacks) if callbacks else None
        # Reuse candidates for keywords: the tags the blog's posts already use.
        self.existing_keywords: list[str] = list(existing_keywords or [])
        self.llm_factory = BedrockLanguageModelFactory(boto_session=self.boto_session)
        self._initialize_chains(
            citation_extraction_model_id, attributes_extraction_model_id
        )

    def _initialize_chains(
        self,
        citation_extraction_model_id: LanguageModelId,
        attributes_extraction_model_id: LanguageModelId,
    ) -> None:
        # Both extractors are fed the full paper text, which for long papers
        # (e.g. 500k+ chars) exceeds the 200k default context window. Enable the
        # 1M context window on each so long papers don't fail with
        # "prompt is too long" (no-op on models that don't support it). The text
        # itself is fitted to each model's exact budget at call time via
        # llm_factory.fit_text (Bedrock CountTokens).
        citation_llm = self.llm_factory.get_model(
            citation_extraction_model_id,
            temperature=0.0,
            supports_1m_context_window=True,
            callbacks=self._callbacks,
        )
        attributes_llm = self.llm_factory.get_model(
            attributes_extraction_model_id,
            temperature=0.0,
            supports_1m_context_window=True,
            callbacks=self._callbacks,
        )
        self._citation_model_id = citation_extraction_model_id
        self._attributes_model_id = attributes_extraction_model_id
        self.citation_extractor = (
            CitationExtractionPrompt.get_prompt()
            | citation_llm
            | HTMLTagOutputParser(tag_names=CitationExtractionPrompt.output_variables)
        )
        self.attributes_extractor = (
            AttributesExtractionPrompt.get_prompt()
            | attributes_llm
            | HTMLTagOutputParser(tag_names=AttributesExtractionPrompt.output_variables)
        )

    @classmethod
    async def create(
        cls,
        citation_extraction_model_id: LanguageModelId,
        attributes_extraction_model_id: LanguageModelId,
        boto_session: boto3.Session,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
        category_options: list[str] | None = None,
        controlled_tags: list[str] | None = None,
        existing_keywords: list[str] | None = None,
    ) -> "ContentExtractor":
        extractor = cls(
            citation_extraction_model_id=citation_extraction_model_id,
            attributes_extraction_model_id=attributes_extraction_model_id,
            boto_session=boto_session,
            callbacks=callbacks,
            category_options=category_options,
            controlled_tags=controlled_tags,
            existing_keywords=existing_keywords,
        )
        return extractor

    async def extract_citations(self, html_content: str) -> list[Citation]:
        citations = []
        existing_citations_str = ""
        seen_citations = set()
        # Fit once: every pagination call sends the same paper text, and each fit
        # is a CountTokens binary search.
        text = neutralize_prompt_tags(
            self.llm_factory.fit_text(
                self._citation_model_id, html_content, label="citation text"
            )
        )

        for page in range(self.MAX_CITATION_PAGES):
            result = await self._extract_citations(text, existing_citations_str)
            raw_citations_str = result.get("citations", "")
            if not raw_citations_str.strip():
                break

            newly_parsed_citations = [
                self._parse_citation(line)
                for line in raw_citations_str.split("\n")
                if line.strip()
            ]

            unique_new_citations = []
            for cit in newly_parsed_citations:
                if cit and (key := self._dedup_key(cit)) not in seen_citations:
                    unique_new_citations.append(cit)
                    seen_citations.add(key)

            if not unique_new_citations:
                break

            citations.extend(unique_new_citations)
            if not is_affirmative(result.get("has_more")):
                break

            existing_citations_str = "\n".join(repr(c) for c in citations)
            if page == self.MAX_CITATION_PAGES - 1:
                logger.warning(
                    "Reached max citation pages (%d); stopping extraction.",
                    self.MAX_CITATION_PAGES,
                )

        logger.info("Extracted %d unique citations", len(citations))
        return citations

    @RetryableBase._retry("citation_extraction")
    async def _extract_citations(
        self, text: str, existing_citations: str
    ) -> dict[str, str]:
        """``text`` is already fitted and defanged by :meth:`extract_citations`."""
        return await self.citation_extractor.ainvoke(
            {"text": text, "existing_citations": existing_citations}
        )

    @staticmethod
    def _dedup_key(cit: Citation) -> str:
        """Stable identity key for deduping parsed citations across pages.

        Prefer the arXiv id (a strong unique identifier). Otherwise fall back to
        a normalized (title, year, first-author-surname) — normalizing lets us
        still collapse the model's formatting variants of the SAME reference
        ("Smith et al." vs "Smith, J., et al.") across pagination calls, while
        keeping genuinely distinct same-title works (different authors/year)
        apart.
        """
        if cit.arxiv_id:
            return f"arxiv:{cit.arxiv_id.lower()}"

        def _norm(value: str) -> str:
            # \w keeps non-Latin titles distinct instead of collapsing to "".
            return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())

        first_author = re.split(r"[,;&]| and ", cit.authors, maxsplit=1)[0]
        # Reduce "Smith, J." / "John Smith" to a single surname-ish token so
        # formatting variants collapse but different authors don't.
        surname = _norm(first_author).split(" ")[-1] if first_author.strip() else ""
        return f"{_norm(cit.title)}|{cit.year or ''}|{surname}"

    @staticmethod
    def _parse_citation(citation_str: str) -> Citation | None:
        try:
            cit_tuple = ast.literal_eval(citation_str)
            if not isinstance(cit_tuple, tuple) or len(cit_tuple) != 4:
                raise ValueError("Input is not a tuple of length 4")

            return Citation(
                authors=str(cit_tuple[0]).strip(),
                year=cit_tuple[1] or None,
                title=str(cit_tuple[2]).strip(),
                arxiv_id=str(cit_tuple[3]).strip() if cit_tuple[3] else None,
            )
        except (ValueError, SyntaxError, IndexError, TypeError) as e:
            logger.warning("Failed to parse citation: '%s' - %s", citation_str, e)
            return None

    @RetryableBase._retry("attributes_extraction")
    async def extract_attributes(self, html_content: str) -> Attributes:
        result = await self.attributes_extractor.ainvoke(
            {
                "text": neutralize_prompt_tags(
                    self.llm_factory.fit_text(
                        self._attributes_model_id, html_content, label="attributes text"
                    )
                ),
                "existing_keywords": ", ".join(self.existing_keywords) or "(none)",
                **self._taxonomy,
            }
        )
        keywords_str = result.get("keywords", "")
        extracted_keywords = [
            kw.strip() for kw in keywords_str.split(",") if kw.strip()
        ]

        def _clean_na(value: str) -> str:
            return "" if is_placeholder(value) else value.strip()

        title = _clean_na(result.get("title", "")) or None
        authors = [
            a.strip()
            for a in _clean_na(result.get("authors", "")).split(",")
            if a.strip()
        ]

        return Attributes(
            affiliation=result.get("affiliation", "N/A"),
            category=result.get("category", "N/A"),
            keywords=extracted_keywords,
            title=title,
            authors=authors,
            controlled_tags=[
                tag.strip()
                for tag in result.get("controlled_tags", "").split(",")
                if tag.strip()
            ],
            seo_title=_clean_na(result.get("seo_title", "")) or None,
        )
