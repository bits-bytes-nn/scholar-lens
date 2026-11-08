"""The evidence engine's paper texts and cache, on the legacy fetchers.

A paper is read as completely as it can be had: its arXiv HTML rendering
(sections, LaTeX, figures), else its PDF (arXiv, or the open-access copy
OpenAlex lists for its DOI) as text page by page, else its abstract. This
module moves to the adapters once ``web_research`` and ``paper_source`` do.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

import pymupdf

from scholar_lens.adapters.aws.storage import S3Handler, S3OperationError
from scholar_lens.adapters.retry import is_access_error
from scholar_lens.adapters.scholarly.index import OpenAlexIndex
from scholar_lens.adapters.scholarly.paper_source import PaperSourceError, PdfUrlSource
from scholar_lens.adapters.web.research import WebResearcher
from scholar_lens.domain.constants import S3Paths
from scholar_lens.domain.evidence import SourceDocument
from scholar_lens.domain.research import ResearchCorpus, SourceFetchError
from scholar_lens.domain.survey.plan import PlanPaper, unversioned_arxiv_id

logger = logging.getLogger(__name__)

# Pages of a PDF read as text (appendices beyond are rarely needed).
MAX_PDF_PAGES = 40


class PaperDocuments:
    def __init__(
        self,
        researcher: WebResearcher,
        openalex: OpenAlexIndex,
        abstracts: dict[str, str],
        *,
        pdf_source: Callable[[str], PdfUrlSource] = PdfUrlSource,
    ) -> None:
        self.researcher = researcher
        self.openalex = openalex
        self.abstracts = abstracts
        self._pdf_source = pdf_source
        # Source downgrades, by kind: paper id -> why (see :meth:`notes`).
        self._html_to_pdf: dict[str, str] = {}
        self._abstract_only: list[str] = []
        self._cut: dict[str, int] = {}

    def notes(self) -> list[str]:
        """Run notes for every paper read from less than its full text: from
        the PDF because the arXiv HTML rendering could not be used (with the
        reason), from its abstract alone, or with its PDF text cut."""
        notes: list[str] = []
        if self._html_to_pdf:
            notes.append(
                "Read from the PDF (the arXiv HTML rendering could not be used): "
                + ", ".join(f"{pid} ({why})" for pid, why in self._html_to_pdf.items())
            )
        if self._abstract_only:
            notes.append(
                "Read from the abstract only (no HTML or PDF text could be had): "
                + ", ".join(self._abstract_only)
            )
        if self._cut:
            notes.append(
                f"PDF text cut at {MAX_PDF_PAGES} pages: "
                + ", ".join(f"{pid} ({n} pages)" for pid, n in self._cut.items())
            )
        return notes

    async def document(self, paper: PlanPaper) -> SourceDocument | None:
        arxiv_id = unversioned_arxiv_id(paper.id)
        if arxiv_id and (
            doc := await asyncio.to_thread(self._arxiv_html, paper, arxiv_id)
        ):
            return doc
        pdf_url = (
            f"https://arxiv.org/pdf/{arxiv_id}"
            if arxiv_id
            else await asyncio.to_thread(self._open_access_pdf, paper.id)
        )
        if pdf_url and (doc := await asyncio.to_thread(self._pdf, paper, pdf_url)):
            return doc
        # The PDF did not do either: the HTML note no longer describes it.
        self._html_to_pdf.pop(paper.id, None)
        if abstract := self.abstracts.get(paper.id):
            self._abstract_only.append(paper.id)
            return SourceDocument.from_markdown(
                paper.id, paper.title, "abstract", f"## Abstract\n\n{abstract}"
            )
        return None

    def _arxiv_html(self, paper: PlanPaper, arxiv_id: str) -> SourceDocument | None:
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        try:
            corpus = self.researcher.research(
                [f"https://arxiv.org/abs/{arxiv_id}"], discover_subpages=False
            )
        except SourceFetchError as e:
            logger.warning("Could not read arXiv %s: %s", arxiv_id, e)
            self._html_to_pdf[paper.id] = _reason(e)
            corpus = ResearchCorpus()
        page = corpus.pages[0] if corpus.pages else None
        if page is None or "/html/" not in (page.final_url or page.url):
            if paper.id not in self._html_to_pdf:
                fallbacks = getattr(self.researcher, "arxiv_fallbacks", {})
                self._html_to_pdf[paper.id] = (
                    fallbacks.get(html_url, "unreadable")
                    if isinstance(fallbacks, dict)
                    else "unreadable"
                )
            return None
        return SourceDocument.from_markdown(
            paper.id,
            paper.title,
            "full_text",
            page.text,
            figures=list(page.image_captions.items()),
        )

    def _open_access_pdf(self, doi: str) -> str | None:
        try:
            works = self.openalex.by_doi(doi)
        except (
            Exception
        ) as e:  # noqa: BLE001 - an index outage falls back to the abstract
            logger.warning("OpenAlex lookup of %s failed: %s", doi, e)
            return None
        for work in works:
            for loc in [
                work.get("best_oa_location") or {},
                *(work.get("locations") or []),
            ]:
                if url := (loc or {}).get("pdf_url"):
                    return str(url)
        return None

    def _pdf(self, paper: PlanPaper, url: str) -> SourceDocument | None:
        source = None
        try:
            # A malformed or non-public URL (an index's pdf_url) is one
            # paper's missing PDF, not the survey's failure.
            source = self._pdf_source(url)
            with tempfile.TemporaryDirectory() as tmp:
                path = source.download_pdf(Path(tmp))
                with pymupdf.open(path) as pdf:
                    total = pdf.page_count
                    pages = [
                        f"## Page {n}\n\n{page.get_text().strip()}"
                        for n, page in enumerate(pdf, 1)
                        if n <= MAX_PDF_PAGES
                    ]
        except (PaperSourceError, RuntimeError, ValueError) as e:
            logger.warning("Could not read the PDF of %s (%s): %s", paper.id, url, e)
            return None
        finally:
            if source is not None:
                source.close()
        text = "\n\n".join(pages)
        if len(text) < 2_000:  # a scanned or empty PDF
            return None
        if total > MAX_PDF_PAGES:
            self._cut[paper.id] = total
        return SourceDocument.from_markdown(paper.id, paper.title, "pdf_text", text)


def _reason(error: BaseException) -> str:
    """A fetch failure's cause, short: "HTTP 429", else its message."""
    cause = error.__cause__
    response = getattr(cause, "response", None)
    if (status := getattr(response, "status_code", None)) is not None:
        return f"HTTP {status}"
    return str(cause or error)[:160]


class EvidenceTextStore:
    """Evidence on disk (``local_dir``) and, with a bucket, in S3
    (``<prefix>/<area>/``: ``surveys``, ``guides`` or ``papers``). It also
    holds the engine's stage checkpoints (its ``checkpoints/`` keys): a
    Batch attempt's scratch disk is gone with its container, so a retried
    attempt reads them back from S3."""

    def __init__(
        self,
        local_dir: Path,
        s3_handler: S3Handler | None,
        s3_prefix: str,
        area: S3Paths = S3Paths.SURVEYS,
    ) -> None:
        self.local_dir = local_dir
        self.s3_handler = s3_handler
        self.s3_prefix = f"{s3_prefix}/{area.value}".lstrip("/")

    def get(self, key: str) -> str | None:
        path = self.local_dir / key
        if path.is_file():
            return path.read_text(encoding="utf-8")
        if self.s3_handler is not None:
            return self.s3_handler.get_text(f"{self.s3_prefix}/{key}")
        return None

    def put(self, key: str, text: str) -> None:
        path = self.local_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if self.s3_handler is not None:
            try:
                self.s3_handler.put_text(f"{self.s3_prefix}/{key}", text)
            except S3OperationError as e:
                # A cache write is not the run's work: the evidence is in hand
                # (and on disk); a refused permission is still raised.
                if is_access_error(e):
                    raise
                logger.warning("Could not cache evidence %s in S3: %s", key, e)
