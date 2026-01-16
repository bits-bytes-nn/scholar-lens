import asyncio
import re
from pathlib import Path

import boto3
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader

from .arxiv_handler import ArxivHandler, ArxivNotFoundError
from .constants import AppConstants, LanguageModelId, LocalPaths
from .logger import logger
from .parser import Content, ContentParseError, HTMLParser
from .prompts import CitationAnalysisPrompt, CitationSummaryPrompt
from .utils import BedrockLanguageModelFactory, HTMLTagOutputParser, RetryableBase


DEFAULT_MAX_CONCURRENCY: int = 10
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_SEARCH_RESULTS_LIMIT: int = 10
DEFAULT_TIMEOUT: int = 60


class CitationSummarizer(RetryableBase):
    FAILURE_STRING: str = "Could not generate summary."
    MAX_CHARS_PER_CONTENT: int = 100000

    def __init__(
        self,
        citation_summarizing_model_id: LanguageModelId,
        citation_analysis_model_id: LanguageModelId,
        boto_session: boto3.Session,
        paper_dir: Path | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        self.boto_session = boto_session
        self.llm_factory = BedrockLanguageModelFactory(boto_session=self.boto_session)
        self.references_dir = (
            paper_dir or Path.cwd()
        ) / LocalPaths.REFERENCES_DIR.value
        self.semaphore = asyncio.Semaphore(max_concurrency)

        analysis_llm = self.llm_factory.get_model(
            citation_analysis_model_id, temperature=0.0
        )
        summarizing_llm = (
            analysis_llm
            if citation_summarizing_model_id == citation_analysis_model_id
            else self.llm_factory.get_model(
                citation_summarizing_model_id, temperature=0.0
            )
        )

        self.citation_analyzer = (
            CitationAnalysisPrompt.get_prompt()
            | analysis_llm
            | HTMLTagOutputParser(tag_names=CitationAnalysisPrompt.output_variables)
        )
        self.citation_summarizer = (
            CitationSummaryPrompt.get_prompt() | summarizing_llm | StrOutputParser()
        )
        self.arxiv_handler = ArxivHandler(max_retries=max_retries)
        self.html_parser = HTMLParser(timeout=timeout)

    async def summarize(
        self, reference_identifiers: list[str], original_content: str
    ) -> list[str]:
        async def process_with_semaphore(identifier: str) -> str | None:
            async with self.semaphore:
                return await self._process_identifier(identifier, original_content)

        tasks = [
            process_with_semaphore(identifier) for identifier in reference_identifiers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        summaries: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(
                    "Failed to process an identifier after retries: '%s'", result
                )
            elif result is not None and isinstance(result, str):
                if self.FAILURE_STRING not in result:
                    summaries.append(result)
                else:
                    logger.warning("Excluding failed summary: '%s'", result)

        logger.info(
            "Generated %d summaries from %d identifiers",
            len(summaries),
            len(reference_identifiers),
        )
        return summaries

    async def _process_identifier(
        self, identifier: str, original_content: str
    ) -> str | None:
        is_arxiv, arxiv_id = await self._classify_identifier(identifier)
        if is_arxiv and arxiv_id:
            return await self._process_arxiv_item(arxiv_id, original_content)
        return await self._process_title_item(identifier, original_content)

    async def _classify_identifier(self, identifier: str) -> tuple[bool, str | None]:
        if identifier.lower().startswith("arxiv:") or re.match(
            r"^\d{4}\.\d{5}(v\d+)?$", identifier
        ):
            return True, identifier.replace("arXiv:", "")

        arxiv_id = await asyncio.to_thread(
            self.arxiv_handler.search_by_title,
            identifier,
            max_results=DEFAULT_SEARCH_RESULTS_LIMIT,
        )
        if arxiv_id:
            logger.info("Found 'arxiv_id' '%s' for title: '%s'", arxiv_id, identifier)
            return True, arxiv_id.replace("arXiv:", "")

        logger.warning("No 'arxiv_id' found for title: '%s'", identifier)
        return False, None

    @RetryableBase._retry("arxiv_item_processing")
    async def _process_arxiv_item(
        self, arxiv_id: str, original_content: str
    ) -> str | None:
        try:
            content = await self._extract_paper_content(arxiv_id)
            summary = await self.citation_summarizer.ainvoke(
                {
                    "reference_content": content.text[: self.MAX_CHARS_PER_CONTENT],
                    "original_content": original_content[: self.MAX_CHARS_PER_CONTENT],
                }
            )
            title, authors = await asyncio.to_thread(
                self.arxiv_handler.get_title_and_authors, arxiv_id
            )
            return f"Title: [{title}]({AppConstants.External.ARXIV_PDF.value}/{arxiv_id})\nAuthors: {authors}\n\n{summary}"
        except (ArxivNotFoundError, ContentParseError) as e:
            logger.error("Non-retryable error for arxiv_id '%s': %s", arxiv_id, e)
            return f"[{arxiv_id}]\nCould not generate summary."
        except Exception as e:
            logger.error("Error during processing arxiv_id '%s': %s", arxiv_id, e)
            raise

    async def _extract_paper_content(self, arxiv_id: str) -> Content:
        try:
            if content := await self._parse_html_content(arxiv_id):
                return content
        except ContentParseError as e:
            logger.info(
                "HTML parsing failed for '%s', falling back to PDF: %s", arxiv_id, e
            )

        return await self._parse_pdf_content(arxiv_id)

    async def _parse_html_content(self, arxiv_id: str) -> Content | None:
        async with self.html_parser:
            result = await self.html_parser.parse(arxiv_id, extract_text=True)
            if result.content and result.content.text:
                return result.content
        return None

    async def _parse_pdf_content(self, arxiv_id: str) -> Content:
        try:
            paper_path = await asyncio.to_thread(
                self.arxiv_handler.download_paper, arxiv_id, self.references_dir
            )

            def read_pdf():
                reader = PdfReader(paper_path)
                return " ".join(
                    page.extract_text().strip()
                    for page in reader.pages
                    if page.extract_text()
                )

            text = await asyncio.to_thread(read_pdf)
            logger.info(
                "PDF parsing succeeded for '%s', extracted %d characters",
                arxiv_id,
                len(text),
            )
            return Content(text=text)
        except Exception as e:
            raise ContentParseError(f"PDF parsing failed for {arxiv_id}") from e

    @RetryableBase._retry("title_item_processing")
    async def _process_title_item(self, title: str, original_content: str) -> str:
        analysis = await self.citation_analyzer.ainvoke(
            {"reference_title": title, "original_content": original_content}
        )
        analysis = analysis.get("analysis", "")
        if not analysis or not analysis.strip():
            return f"Title: [{title}]\n{self.FAILURE_STRING}"
        return f"Title: [{title}]\n{analysis}"
