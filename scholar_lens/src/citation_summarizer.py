import asyncio
from pathlib import Path
from typing import Any

import boto3
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader

from .arxiv_handler import ArxivHandler, ArxivNotFoundError
from .citation_metadata import ChainedMetadataResolver, ReferenceMetadata, titles_match
from .constants import AppConstants, LanguageModelId, LocalPaths, normalize_arxiv_id
from .logger import logger
from .parser import Content, ContentParseError, HTMLParser
from .prompts import CitationAnalysisPrompt, CitationSummaryPrompt
from .utils import BedrockLanguageModelFactory, HTMLTagOutputParser, RetryableBase

DEFAULT_MAX_CONCURRENCY: int = 5
DEFAULT_SEARCH_RESULTS_LIMIT: int = 10
DEFAULT_TIMEOUT: int = 60


class CitationSummarizer(RetryableBase):
    FAILURE_STRING: str = "Could not generate summary."

    def __init__(
        self,
        citation_summarizing_model_id: LanguageModelId,
        citation_analysis_model_id: LanguageModelId,
        boto_session: boto3.Session,
        paper_dir: Path | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        metadata_resolver: ChainedMetadataResolver | None = None,
        callbacks: list[Any] | None = None,
        prefer_full_text: bool = False,
    ) -> None:
        self.boto_session = boto_session
        self.llm_factory = BedrockLanguageModelFactory(boto_session=self.boto_session)
        self.references_dir = (
            paper_dir or Path.cwd()
        ) / LocalPaths.REFERENCES_DIR.value
        self.semaphore = asyncio.Semaphore(max_concurrency)
        # Resolve title/abstract from lenient APIs first; only touch arXiv full
        # text when explicitly asked (prefer_full_text) or as a last resort.
        self.metadata_resolver = metadata_resolver or ChainedMetadataResolver()
        self.prefer_full_text = prefer_full_text
        self._callbacks = callbacks or None
        # Cache summaries by identifier so the same reference cited from multiple
        # paragraphs is summarised only once (the enrich loop calls per-paragraph).
        # The in-flight map gives single-flight semantics: concurrent callers for
        # the same identifier await one shared task instead of each doing the work.
        self._summary_cache: dict[str, str | None] = {}
        self._inflight: dict[str, asyncio.Future[str | None]] = {}
        self._cache_lock = asyncio.Lock()

        self.citation_summarizing_model_id = citation_summarizing_model_id
        analysis_llm = self.llm_factory.get_model(
            citation_analysis_model_id, temperature=0.0, callbacks=self._callbacks
        )
        summarizing_llm = (
            analysis_llm
            if citation_summarizing_model_id == citation_analysis_model_id
            else self.llm_factory.get_model(
                citation_summarizing_model_id,
                temperature=0.0,
                callbacks=self._callbacks,
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
        self.arxiv_handler = ArxivHandler()
        # Timeout for per-call HTMLParser instances. A parser is NOT shared across
        # tasks: summarize() resolves up to `max_concurrency` identifiers at once,
        # and the parser's __aexit__ closes its httpx client — a shared instance
        # would close the client out from under a concurrent task's in-flight GET.
        self._parser_timeout = timeout

    def _fit_summary_inputs(
        self, reference_content: str, original_content: str
    ) -> dict[str, str]:
        """Fit the (reference, original) pair to the summarizing model's window
        with Bedrock CountTokens. Both texts share one prompt, so first hold back
        room for the prompt template + the model's response, then split the
        REMAINING budget between the two (each gets half)."""
        model_id = self.citation_summarizing_model_id
        window = self.llm_factory.effective_context_window(model_id)
        # Headroom for prompt template + output, then halve what's left per input.
        usable = max(window - self.llm_factory.DEFAULT_CONTEXT_RESERVE_TOKENS, 1000)
        # fit_text computes budget = window - reserve_tokens, so to give each input
        # `usable // 2` tokens we reserve everything above that.
        per_input_reserve = window - usable // 2
        return {
            "reference_content": self.llm_factory.fit_text(
                model_id,
                reference_content,
                reserve_tokens=per_input_reserve,
                label="citation reference",
            ),
            "original_content": self.llm_factory.fit_text(
                model_id,
                original_content,
                reserve_tokens=per_input_reserve,
                label="citation original",
            ),
        }

    async def summarize(
        self, reference_identifiers: list[str], original_content: str
    ) -> list[str]:
        async def process_with_semaphore(identifier: str) -> str | None:
            # Serve from cache first; otherwise register an in-flight task so that
            # concurrent callers for the same identifier (the enrich loop fires one
            # summarize() per paragraph) share a single resolution instead of each
            # re-doing the expensive arXiv/metadata work.
            async with self._cache_lock:
                if identifier in self._summary_cache:
                    return self._summary_cache[identifier]
                existing = self._inflight.get(identifier)
                if existing is None:
                    existing = asyncio.ensure_future(
                        self._resolve_once(identifier, original_content)
                    )
                    self._inflight[identifier] = existing
                    owner = True
                else:
                    owner = False
            try:
                return await existing
            finally:
                if owner:
                    async with self._cache_lock:
                        self._inflight.pop(identifier, None)

        # De-duplicate identifiers within this call too.
        unique_identifiers = list(dict.fromkeys(reference_identifiers))
        tasks = [
            process_with_semaphore(identifier) for identifier in unique_identifiers
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
            len(unique_identifiers),
        )
        return summaries

    async def _resolve_once(self, identifier: str, original_content: str) -> str | None:
        """Do the bounded work for one identifier and memoise the result.

        Reached at most once per identifier across the whole run (callers funnel
        through the in-flight map in ``summarize``).
        """
        async with self.semaphore:
            result = await self._process_identifier(identifier, original_content)
        async with self._cache_lock:
            self._summary_cache[identifier] = result
        return result

    async def _process_identifier(
        self, identifier: str, original_content: str
    ) -> str | None:
        # Explicit arXiv id -> go straight to arXiv (we already have the id).
        if arxiv_id := normalize_arxiv_id(identifier):
            return await self._process_arxiv_item(arxiv_id, original_content)

        # Otherwise resolve metadata/abstract from lenient providers FIRST; this
        # avoids the arXiv API entirely for most references and works from the
        # abstract instead of downloading full text.
        metadata = await asyncio.to_thread(self.metadata_resolver.resolve, identifier)
        if metadata is not None and (
            metadata.has_abstract or not self.prefer_full_text
        ):
            summary = await self._summarize_from_metadata(
                metadata, original_content, identifier
            )
            if summary is not None:
                return summary

        # Last resort: arXiv title search (rate-limited) + the legacy path.
        found_arxiv_id = await asyncio.to_thread(
            self.arxiv_handler.search_by_title,
            identifier,
            DEFAULT_SEARCH_RESULTS_LIMIT,
        )
        if found_arxiv_id:
            logger.info(
                "Found 'arxiv_id' '%s' for title: '%s'", found_arxiv_id, identifier
            )
            return await self._process_arxiv_item(
                found_arxiv_id.replace("arXiv:", ""), original_content
            )
        logger.warning("No metadata or arXiv match for: '%s'", identifier)
        return await self._process_title_item(identifier, original_content)

    @staticmethod
    def _title_matches(queried: str | None, resolved: str | None) -> bool:
        return titles_match(queried, resolved)

    @RetryableBase._retry("citation_metadata_summary")
    async def _summarize_from_metadata(
        self,
        metadata: ReferenceMetadata,
        original_content: str,
        queried_identifier: str | None = None,
    ) -> str | None:
        """Summarise a reference from its abstract (no full-text download).

        The resolver returns the top fuzzy match from Crossref/Semantic Scholar,
        which is sometimes the WRONG paper — embedding its URL produces a
        mis-attributed citation link. Only attach the link when the resolved
        title closely matches the queried title; otherwise keep the (queried)
        title as plain text so we never point at a wrong paper.
        """
        if not self._title_matches(queried_identifier, metadata.title):
            # A different paper's abstract would be summarised under this
            # citation; let the caller fall through to the arXiv/title paths.
            logger.info(
                "Resolved title '%s' does not match '%s'; skipping its abstract.",
                metadata.title,
                queried_identifier,
            )
            return None
        reference_text = metadata.abstract or metadata.title
        if not reference_text.strip():
            return None
        summary = await self.citation_summarizer.ainvoke(
            self._fit_summary_inputs(reference_text, original_content)
        )
        if not summary or self.FAILURE_STRING in summary:
            return None
        link = f"[{metadata.title}]({metadata.url})" if metadata.url else metadata.title
        author_line = f"Authors: {metadata.author_str}\n" if metadata.authors else ""
        return f"Title: {link}\n{author_line}\n{summary}"

    @RetryableBase._retry("arxiv_item_processing")
    async def _process_arxiv_item(
        self, arxiv_id: str, original_content: str
    ) -> str | None:
        try:
            # One arXiv metadata fetch gives title, authors AND abstract — no
            # separate get_title_and_authors call (was a 2nd-3rd API hit).
            metadata = await asyncio.to_thread(
                self.arxiv_handler.fetch_metadata, arxiv_id
            )
            title = metadata.title
            authors = ", ".join(metadata.authors)

            if self.prefer_full_text:
                content = await self._extract_paper_content(arxiv_id)
                reference_text = content.text
            else:
                # Abstract-first: skip the full-text HTML/PDF download entirely.
                reference_text = metadata.abstract or ""
                if not reference_text.strip():
                    content = await self._extract_paper_content(arxiv_id)
                    reference_text = content.text

            summary = await self.citation_summarizer.ainvoke(
                self._fit_summary_inputs(reference_text, original_content)
            )
            url = f"{AppConstants.External.ARXIV_PDF.value}/{arxiv_id}"
            return f"Title: [{title}]({url})\nAuthors: {authors}\n\n{summary}"
        except (ArxivNotFoundError, ContentParseError) as e:
            # Nothing usable: return no summary rather than a placeholder that
            # would reach the writer prompts as if it were content.
            logger.error("Non-retryable error for arxiv_id '%s': %s", arxiv_id, e)
            return None
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
        # Fresh parser per call: concurrent tasks must not share one httpx client
        # (its __aexit__ close would abort another task's in-flight request).
        async with HTMLParser(timeout=self._parser_timeout) as parser:
            result = await parser.parse(arxiv_id, extract_text=True)
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
    async def _process_title_item(
        self, title: str, original_content: str
    ) -> str | None:
        analysis = await self.citation_analyzer.ainvoke(
            {"reference_title": title, "original_content": original_content}
        )
        analysis = analysis.get("analysis", "")
        if not analysis or not analysis.strip() or self.FAILURE_STRING in analysis:
            return None
        return f"Title: [{title}]\n{analysis}"
