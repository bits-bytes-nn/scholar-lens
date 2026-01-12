import ast
import re
from pathlib import Path
from typing import Any

import boto3
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
)


class Attributes(BaseModel):
    affiliation: str = Field(min_length=1)
    category: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)


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
    ) -> None:
        self.boto_session = boto_session
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
        citation_llm = self.llm_factory.get_model(
            citation_extraction_model_id, temperature=0.0
        )
        attributes_llm = self.llm_factory.get_model(
            attributes_extraction_model_id, temperature=0.0
        )
        table_of_contents_llm = self.llm_factory.get_model(
            table_of_contents_model_id, temperature=0.0, supports_1m_context_window=True
        )
        robust_xml_output_parser = create_robust_xml_output_parser(
            self.llm_factory,
            enable_output_fixing=enable_output_fixing,
            output_fixing_model_id=output_fixing_model_id,
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

        while True:
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
                if cit and repr(cit) not in seen_citations:
                    unique_new_citations.append(cit)
                    seen_citations.add(repr(cit))

            if not unique_new_citations:
                break

            citations.extend(unique_new_citations)
            if result.get("has_more", "n").strip().lower() != "y":
                break

            existing_citations_str = "\n".join(repr(c) for c in citations)

        logger.info("Extracted %d unique citations", len(citations))
        return citations

    @RetryableBase._retry("citation_extraction")
    async def _extract_citations(
        self, html_content: str, existing_citations: str
    ) -> dict[str, str]:
        return await self.citation_extractor.ainvoke(
            {"text": html_content, "existing_citations": existing_citations}
        )

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
                "text": html_content,
                "existing_keywords": ", ".join(sorted(list(self.existing_keywords))),
            }
        )
        keywords_str = result.get("keywords", "")
        extracted_keywords = [
            kw.strip() for kw in keywords_str.split(",") if kw.strip()
        ]
        await self._update_keywords(extracted_keywords)

        return Attributes(
            affiliation=result.get("affiliation", "N/A"),
            category=result.get("category", "N/A"),
            keywords=extracted_keywords,
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
            {"paper_content": html_content}
        )
