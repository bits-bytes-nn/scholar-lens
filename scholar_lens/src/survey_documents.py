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
from pathlib import Path

import pymupdf

from ..adapters.aws.storage import S3Handler, S3OperationError
from ..adapters.retry import is_access_error
from ..adapters.scholarly.index import OpenAlexIndex
from ..domain.evidence import SourceDocument
from ..domain.survey.plan import PlanPaper, unversioned_arxiv_id
from .constants import S3Paths
from .paper_source import PaperSourceError, PdfUrlSource
from .web_research import ResearchCorpus, SourceFetchError, WebResearcher

logger = logging.getLogger(__name__)

# Pages of a PDF read as text (appendices beyond are rarely needed).
MAX_PDF_PAGES = 40


class PaperDocuments:
    def __init__(
        self,
        researcher: WebResearcher,
        openalex: OpenAlexIndex,
        abstracts: dict[str, str],
    ) -> None:
        self.researcher = researcher
        self.openalex = openalex
        self.abstracts = abstracts

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
        if abstract := self.abstracts.get(paper.id):
            return SourceDocument.from_markdown(
                paper.id, paper.title, "abstract", f"## Abstract\n\n{abstract}"
            )
        return None

    def _arxiv_html(self, paper: PlanPaper, arxiv_id: str) -> SourceDocument | None:
        try:
            corpus = self.researcher.research(
                [f"https://arxiv.org/abs/{arxiv_id}"], discover_subpages=False
            )
        except SourceFetchError as e:
            logger.warning("Could not read arXiv %s: %s", arxiv_id, e)
            corpus = ResearchCorpus()
        page = corpus.pages[0] if corpus.pages else None
        if page is None or "/html/" not in (page.final_url or page.url):
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
            source = PdfUrlSource(url)
            with tempfile.TemporaryDirectory() as tmp:
                path = source.download_pdf(Path(tmp))
                with pymupdf.open(path) as pdf:
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
        return SourceDocument.from_markdown(paper.id, paper.title, "pdf_text", text)


class EvidenceTextStore:
    """Evidence on disk (``local_dir``) and, with a bucket, in S3
    (``<prefix>/<area>/``: ``surveys`` or ``guides``)."""

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
