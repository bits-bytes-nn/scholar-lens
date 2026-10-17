import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import fitz
from langchain_core.output_parsers import StrOutputParser

from .arxiv_handler import ArxivHandler, ArxivPaperError
from .citation_metadata import ChainedMetadataResolver, arxiv_id_from_doi
from .constants import AppConstants, LanguageModelId, LocalPaths, normalize_arxiv_id
from .logger import logger
from .parser import Content, ContentParseError, HTMLParser
from .prompt_safety import neutralize_prompt_tags
from .prompts import CitationSummaryPrompt
from .utils import BedrockLanguageModelFactory, RetryableBase

DEFAULT_MAX_CONCURRENCY: int = 5
DEFAULT_SEARCH_RESULTS_LIMIT: int = 10
DEFAULT_TIMEOUT: int = 60


@dataclass(frozen=True)
class ReferenceSource:
    """What a citation summary is written from: a header naming the work
    (linked when its URL is trusted) and its abstract or full text."""

    header: str
    text: str


class CitationSummarizer(RetryableBase):
    FAILURE_STRING: str = "Could not generate summary."

    def __init__(
        self,
        citation_summarizing_model_id: LanguageModelId,
        boto_session: boto3.Session | None = None,
        paper_dir: Path | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        metadata_resolver: ChainedMetadataResolver | None = None,
        callbacks: list[Any] | None = None,
        prefer_full_text: bool = False,
    ) -> None:
        self.boto_session = boto_session or boto3.Session()
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
        # A reference's source is fetched once per run (single-flight: concurrent
        # callers await one task); its summary is written per citing section,
        # since it says why the reference matters THERE.
        self._source_cache: dict[str, ReferenceSource | None] = {}
        self._inflight: dict[str, asyncio.Future[ReferenceSource | None]] = {}
        self._summary_cache: dict[tuple[str, str], str | None] = {}
        self._cache_lock = asyncio.Lock()

        self.citation_summarizing_model_id = citation_summarizing_model_id
        summarizing_llm = self.llm_factory.get_model(
            citation_summarizing_model_id, temperature=0.0, callbacks=self._callbacks
        )
        self.citation_summarizer = (
            CitationSummaryPrompt.get_prompt() | summarizing_llm | StrOutputParser()
        )
        self.arxiv_handler = ArxivHandler()
        # Timeout for per-call HTMLParser instances. A parser is NOT shared across
        # tasks: its __aexit__ closes its httpx client, which would abort a
        # concurrent task's in-flight GET.
        self._parser_timeout = timeout

    def _fit_summary_inputs(
        self, reference_content: str, original_content: str
    ) -> dict[str, str]:
        """Fit the (reference, original) pair to the summarizing model's window
        with Bedrock CountTokens. Both texts share one prompt, so first hold back
        room for the prompt template + the model's response, then split the
        REMAINING budget between the two (each gets half). Both are untrusted
        (reference text from external APIs, the paper's own text), so their
        prompt-fence tags are defanged first."""
        model_id = self.citation_summarizing_model_id
        window = self.llm_factory.effective_context_window(model_id)
        usable = max(window - self.llm_factory.DEFAULT_CONTEXT_RESERVE_TOKENS, 1000)
        # fit_text computes budget = window - reserve_tokens, so to give each input
        # `usable // 2` tokens we reserve everything above that.
        per_input_reserve = window - usable // 2
        return {
            "reference_content": self.llm_factory.fit_text(
                model_id,
                neutralize_prompt_tags(reference_content),
                reserve_tokens=per_input_reserve,
                label="citation reference",
            ),
            "original_content": self.llm_factory.fit_text(
                model_id,
                neutralize_prompt_tags(original_content),
                reserve_tokens=per_input_reserve,
                label="citation original",
            ),
        }

    async def summarize(
        self, reference_identifiers: list[str], original_content: str
    ) -> list[str]:
        """Summaries of the references, each written for ``original_content``;
        a reference with no abstract or text is left out (and counted)."""
        section_key = hashlib.sha1(original_content.encode("utf-8")).hexdigest()[:12]

        async def one(identifier: str) -> str | None:
            source = await self._source(identifier)
            if source is None:
                return None
            key = (identifier, section_key)
            if key not in self._summary_cache:
                async with self.semaphore:
                    self._summary_cache[key] = await self._summary(
                        source, original_content
                    )
            return self._summary_cache[key]

        unique_identifiers = list(dict.fromkeys(reference_identifiers))
        results = await asyncio.gather(
            *(one(identifier) for identifier in unique_identifiers),
            return_exceptions=True,
        )
        summaries: list[str] = []
        unsourced = 0
        for identifier, result in zip(unique_identifiers, results, strict=True):
            if isinstance(result, BaseException):
                logger.error("Citation summary failed for '%s': %s", identifier, result)
            elif result is None:
                unsourced += self._source_cache.get(identifier) is None
            else:
                summaries.append(result)
        logger.info(
            "Generated %d summaries from %d identifiers (%d without an abstract or "
            "text).",
            len(summaries),
            len(unique_identifiers),
            unsourced,
        )
        return summaries

    def unsourced_references(self) -> list[str]:
        """References looked up this run that no source (metadata abstract or
        arXiv text) was found for; their summaries were skipped."""
        return [ident for ident, source in self._source_cache.items() if source is None]

    async def _source(self, identifier: str) -> ReferenceSource | None:
        async with self._cache_lock:
            if identifier in self._source_cache:
                return self._source_cache[identifier]
            task = self._inflight.get(identifier)
            owner = task is None
            if task is None:
                task = asyncio.ensure_future(self._fetch_source(identifier))
                self._inflight[identifier] = task
        try:
            source = await task
        finally:
            if owner:
                async with self._cache_lock:
                    self._inflight.pop(identifier, None)
        async with self._cache_lock:
            self._source_cache[identifier] = source
        return source

    async def _fetch_source(self, identifier: str) -> ReferenceSource | None:
        async with self.semaphore:
            return await self._resolve_source(identifier)

    async def _resolve_source(self, identifier: str) -> ReferenceSource | None:
        # Explicit arXiv id -> go straight to arXiv (we already have the id).
        if arxiv_id := normalize_arxiv_id(identifier):
            return await self._arxiv_source(arxiv_id)
        # Otherwise resolve the abstract from lenient providers first, which
        # avoids the arXiv API for most references. The resolver only returns
        # a record whose title matches the citation.
        metadata = await asyncio.to_thread(self.metadata_resolver.resolve, identifier)
        abstract: ReferenceSource | None = None
        if metadata is not None and metadata.has_abstract:
            link = (
                f"[{metadata.title}]({metadata.url})"
                if metadata.url
                else metadata.title
            )
            authors = f"Authors: {metadata.author_str}\n" if metadata.authors else ""
            # Title, URL and authors come from external APIs: defang them too.
            abstract = ReferenceSource(
                header=neutralize_prompt_tags(f"Title: {link}\n{authors}"),
                text=metadata.abstract or "",
            )
            if not self.prefer_full_text:
                return abstract
        # A resolved arXiv DOI names the paper; otherwise search arXiv by title.
        found_arxiv_id = arxiv_id_from_doi(
            metadata.url if metadata else None
        ) or await asyncio.to_thread(
            self.arxiv_handler.search_by_title,
            identifier,
            DEFAULT_SEARCH_RESULTS_LIMIT,
        )
        if found_arxiv_id:
            logger.info(
                "Found 'arxiv_id' '%s' for title: '%s'", found_arxiv_id, identifier
            )
            full = await self._arxiv_source(found_arxiv_id.replace("arXiv:", ""))
            if full is not None:
                return full
        if abstract is None:  # full text preferred but unavailable: abstract
            logger.warning("No abstract or text found for reference '%s'", identifier)
        return abstract

    async def _arxiv_source(self, arxiv_id: str) -> ReferenceSource | None:
        try:
            # One arXiv metadata fetch gives title, authors and abstract.
            metadata = await asyncio.to_thread(
                self.arxiv_handler.fetch_metadata, arxiv_id
            )
        except ArxivPaperError as e:
            # Not found, or arXiv failing after its retries. A title lookup
            # falls back to the metadata abstract it already has; an explicit
            # arXiv id has no other source (the metadata providers match by
            # title), so that reference is skipped.
            logger.error("No usable text for arxiv_id '%s': %s", arxiv_id, e)
            return None
        abstract = (metadata.abstract or "").strip()
        text = "" if self.prefer_full_text else abstract
        if not text:
            try:
                text = (await self._extract_paper_content(arxiv_id)).text
            except ContentParseError as e:
                # Full text failed: the abstract (if any) still grounds a summary.
                logger.warning("No full text for arxiv_id '%s': %s", arxiv_id, e)
                text = abstract
        if not text.strip():
            logger.error("No usable text for arxiv_id '%s'", arxiv_id)
            return None
        url = f"{AppConstants.External.ARXIV_PDF.value}/{arxiv_id}"
        return ReferenceSource(
            header=neutralize_prompt_tags(
                f"Title: [{metadata.title}]({url})\n"
                f"Authors: {', '.join(metadata.authors)}\n"
            ),
            text=text,
        )

    async def _summary(
        self, source: ReferenceSource, original_content: str
    ) -> str | None:
        if not source.text.strip():
            return None
        summary = await self._invoke_summary(
            self._fit_summary_inputs(source.text, original_content)
        )
        if not summary or self.FAILURE_STRING in summary:
            logger.warning("Excluding failed summary for '%s'", source.header.strip())
            return None
        return f"{source.header}\n{summary}"

    @RetryableBase._retry("citation_summary")
    async def _invoke_summary(self, inputs: dict[str, str]) -> str:
        """The model call alone is retried, so a throttle never refetches the
        reference from arXiv."""
        return await self.citation_summarizer.ainvoke(inputs)

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
        # Fresh parser per call: concurrent tasks must not share one httpx client.
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

            def read_pdf() -> str:
                with fitz.open(str(paper_path)) as doc:
                    return " ".join(page.get_text().strip() for page in doc)

            text = await asyncio.to_thread(read_pdf)
            logger.info(
                "PDF parsing succeeded for '%s', extracted %d characters",
                arxiv_id,
                len(text),
            )
            return Content(text=text)
        except Exception as e:
            raise ContentParseError(f"PDF parsing failed for {arxiv_id}") from e
