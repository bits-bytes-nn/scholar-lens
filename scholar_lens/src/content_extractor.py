import ast
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import boto3
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field, field_validator

from .aws_helpers import S3Handler
from .constants import LanguageModelId, LocalPaths, S3Paths
from .logger import logger
from .prompts import (
    AttributesExtractionPrompt,
    CitationExtractionPrompt,
    TableOfContentsPrompt,
)
from .utils import (
    BedrockLanguageModelFactory,
    HTMLTagOutputParser,
    RetryableBase,
    create_robust_xml_output_parser,
    is_affirmative,
    is_placeholder,
)
from .web_research import neutralize_prompt_tags


class Attributes(BaseModel):
    affiliation: str = Field(min_length=1)
    category: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    # Title/authors parsed from the PDF text — used to fill metadata for sources
    # (e.g. arbitrary PDF URLs) that don't carry it. "N/A"/empty means unknown.
    title: str | None = Field(default=None)
    authors: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    authors: str = Field(min_length=1)
    year: int | None = Field(default=None)
    title: str = Field(min_length=1)
    arxiv_id: str | None = Field(default=None)

    @field_validator("arxiv_id")
    @classmethod
    def validate_arxiv_id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned_v = v.replace("arXiv:", "").strip()
        if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", cleaned_v):
            raise ValueError(f"Invalid arXiv ID format: {v}")
        return cleaned_v

    def __repr__(self) -> str:
        return f"Citation(title='{self.title}', arxiv_id='{self.arxiv_id}')"


class ContentExtractor(RetryableBase):
    # Hard cap on the paginated citation-extraction loop, so a model that keeps
    # emitting has_more=y (or novel near-duplicate citations) cannot loop forever.
    MAX_CITATION_PAGES: int = 20

    def __init__(
        self,
        citation_extraction_model_id: LanguageModelId,
        attributes_extraction_model_id: LanguageModelId,
        table_of_contents_model_id: LanguageModelId,
        output_fixing_model_id: LanguageModelId,
        boto_session: boto3.Session,
        root_dir: Path | None = None,
        bucket_name: str | None = None,
        s3_prefix: str | None = None,
        enable_output_fixing: bool = False,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> None:
        self.boto_session = boto_session
        # Wire the run's TokenUsageTracker (passed as a callback) into every model
        # so the full-text citation/attributes/TOC extraction — among the largest
        # single inputs in a run — counts toward the token budget and cost metric,
        # not just the generation stage.
        self._callbacks = list(callbacks) if callbacks else None
        self.s3_handler = S3Handler(boto_session, bucket_name) if bucket_name else None
        self.s3_keywords_prefix = (
            f"{s3_prefix}/{S3Paths.KEYWORDS.value}"
            if s3_prefix
            else S3Paths.KEYWORDS.value
        )
        self.keywords_path = (
            (root_dir or Path.cwd())
            / LocalPaths.ASSETS_DIR.value
            / LocalPaths.KEYWORDS_FILE.value
        )
        self.existing_keywords: set[str] = set()
        self.llm_factory = BedrockLanguageModelFactory(boto_session=self.boto_session)
        self._initialize_chains(
            citation_extraction_model_id,
            attributes_extraction_model_id,
            table_of_contents_model_id,
            output_fixing_model_id,
            enable_output_fixing,
        )

    def _initialize_chains(
        self,
        citation_extraction_model_id: LanguageModelId,
        attributes_extraction_model_id: LanguageModelId,
        table_of_contents_model_id: LanguageModelId,
        output_fixing_model_id: LanguageModelId,
        enable_output_fixing: bool,
    ) -> None:
        # All three extractors are fed the full paper text, which for long papers
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
        table_of_contents_llm = self.llm_factory.get_model(
            table_of_contents_model_id,
            temperature=0.0,
            supports_1m_context_window=True,
            callbacks=self._callbacks,
        )
        self._citation_model_id = citation_extraction_model_id
        self._attributes_model_id = attributes_extraction_model_id
        self._toc_model_id = table_of_contents_model_id
        robust_xml_output_parser = create_robust_xml_output_parser(
            self.llm_factory,
            enable_output_fixing=enable_output_fixing,
            output_fixing_model_id=output_fixing_model_id,
            callbacks=self._callbacks,
        )

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
        self.table_of_contents_extractor = (
            TableOfContentsPrompt.get_prompt()
            | table_of_contents_llm
            | robust_xml_output_parser
        )

    @classmethod
    async def create(
        cls,
        citation_extraction_model_id: LanguageModelId,
        attributes_extraction_model_id: LanguageModelId,
        table_of_contents_model_id: LanguageModelId,
        output_fixing_model_id: LanguageModelId,
        boto_session: boto3.Session,
        root_dir: Path | None = None,
        bucket_name: str | None = None,
        s3_prefix: str | None = None,
        enable_output_fixing: bool = False,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> "ContentExtractor":
        extractor = cls(
            citation_extraction_model_id=citation_extraction_model_id,
            attributes_extraction_model_id=attributes_extraction_model_id,
            table_of_contents_model_id=table_of_contents_model_id,
            output_fixing_model_id=output_fixing_model_id,
            boto_session=boto_session,
            root_dir=root_dir,
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            enable_output_fixing=enable_output_fixing,
            callbacks=callbacks,
        )
        await extractor._initialize_keywords()
        return extractor

    async def _initialize_keywords(self) -> None:
        s3_key = f"{self.s3_keywords_prefix}/{self.keywords_path.name}"
        if self.s3_handler and self.s3_handler.exists(s3_key):
            await self.s3_handler.download_file_async(s3_key, self.keywords_path)

        if self.keywords_path.exists():
            self.existing_keywords = set(
                line.strip()
                for line in self.keywords_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

    async def extract_citations(self, html_content: str) -> list[Citation]:
        citations = []
        existing_citations_str = ""
        seen_citations = set()

        for page in range(self.MAX_CITATION_PAGES):
            result = await self._extract_citations(html_content, existing_citations_str)
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
        self, html_content: str, existing_citations: str
    ) -> dict[str, str]:
        return await self.citation_extractor.ainvoke(
            {
                # Defang injection: the untrusted paper body is fenced in the
                # prompt, so a literal "</paper_content>…" in the source must not
                # break out (parity with the summarizer/explainer paths).
                "text": neutralize_prompt_tags(
                    self.llm_factory.fit_text(
                        self._citation_model_id, html_content, label="citation text"
                    )
                ),
                "existing_citations": existing_citations,
            }
        )

    @staticmethod
    def _dedup_key(cit: Citation) -> str:
        """Stable identity key for deduping parsed citations across pages.

        Prefer the arXiv id (a strong unique identifier). Otherwise fall back to
        a normalized (title, year, first-author-surname) — normalizing lets us
        still collapse the model's formatting variants of the SAME reference
        ("Smith et al." vs "Smith, J., et al.") across pagination calls, while
        keeping genuinely distinct same-title works (different authors/year)
        apart, which a title-only key silently dropped.
        """
        if cit.arxiv_id:
            return f"arxiv:{cit.arxiv_id.lower()}"

        def _norm(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

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
                year=int(cit_tuple[1]) if cit_tuple[1] else None,
                title=str(cit_tuple[2]).strip(),
                arxiv_id=str(cit_tuple[3]).strip() if cit_tuple[3] else None,
            )
        except (ValueError, SyntaxError, IndexError) as e:
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
                "existing_keywords": ", ".join(sorted(list(self.existing_keywords))),
            }
        )
        keywords_str = result.get("keywords", "")
        extracted_keywords = [
            kw.strip() for kw in keywords_str.split(",") if kw.strip()
        ]
        await self._update_keywords(extracted_keywords)

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
        )

    async def _update_keywords(self, new_keywords: list[str]) -> None:
        updated = False
        for keyword in new_keywords:
            if keyword and keyword not in self.existing_keywords:
                self.existing_keywords.add(keyword)
                updated = True

        if not updated:
            return

        self.keywords_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_keywords = "\n".join(sorted(list(self.existing_keywords)))
        self.keywords_path.write_text(sorted_keywords, encoding="utf-8")

        if self.s3_handler:
            await self.s3_handler.upload_file_async(
                self.keywords_path, self.s3_keywords_prefix
            )

    @RetryableBase._retry("table_of_contents_extraction")
    async def extract_table_of_contents(self, html_content: str) -> dict[str, Any]:
        return await self.table_of_contents_extractor.ainvoke(
            {
                "paper_content": neutralize_prompt_tags(
                    self.llm_factory.fit_text(
                        self._toc_model_id, html_content, label="TOC text"
                    )
                )
            }
        )
