import re
import requests
from datetime import datetime
from pathlib import Path

import arxiv
from pydantic import BaseModel, Field, HttpUrl, ValidationInfo, field_validator

from .constants import AppConstants
from .logger import logger

DEFAULT_TIMEOUT: int = 10


class ArxivPaperError(Exception):
    pass


class ArxivDownloadError(ArxivPaperError):
    pass


class ArxivNotFoundError(ArxivPaperError):
    pass


class ArxivMetadata(BaseModel):
    arxiv_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    published: datetime
    updated: datetime
    abstract: str | None = None
    journal_ref: str | None = None
    primary_category: str | None = None
    categories: list[str] = Field(default_factory=list)
    doi: str | None = None
    pdf_url: HttpUrl | None = None

    def __repr__(self) -> str:
        return f"ArxivMetadata(arxiv_id='{self.arxiv_id}', title='{self.title}')"

    @field_validator("abstract", mode="before")
    @classmethod
    def clean_abstract(cls, v: str | None) -> str | None:
        if v:
            return " ".join(v.strip().split())
        return v

    @field_validator("doi", mode="before")
    @classmethod
    def generate_and_validate_doi(
        cls, v: str | None, info: ValidationInfo
    ) -> str | None:
        if v:
            return v

        if arxiv_id := info.data.get("arxiv_id"):
            clean_arxiv_id = arxiv_id.split("v")[0]
            standard_doi = f"10.48550/arXiv.{clean_arxiv_id}"
            try:
                response = requests.get(
                    f"{AppConstants.External.DOI_ORG.value}/{standard_doi}",
                    timeout=DEFAULT_TIMEOUT,
                    allow_redirects=True,
                )
                if response.status_code == 200:
                    logger.info(
                        "Successfully validated and generated DOI: '%s'", standard_doi
                    )
                    return standard_doi
                else:
                    logger.warning(
                        "Generated DOI for '%s' failed validation with status %s",
                        arxiv_id,
                        response.status_code,
                    )
            except requests.RequestException as e:
                logger.error("DOI validation request for '%s' failed: %s", arxiv_id, e)

        return None

    @field_validator("updated", mode="before")
    @classmethod
    def set_updated_if_missing(
        cls, v: datetime | None, info: ValidationInfo
    ) -> datetime:
        if v is not None:
            return v
        published = info.data.get("published")
        if published is not None:
            return published
        raise ValueError("Both 'updated' and 'published' are None")


class ArxivHandler:
    DEFAULT_DELAY_SECONDS: int = 3
    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_PAGE_SIZE: int = 100

    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES):
        self.client = arxiv.Client(
            page_size=self.DEFAULT_PAGE_SIZE,
            delay_seconds=self.DEFAULT_DELAY_SECONDS,
            num_retries=max_retries,
        )

    def download_paper(self, arxiv_id: str, papers_dir: Path) -> Path:
        try:
            paper = self._fetch_single_paper(arxiv_id)
            safe_id = paper.get_short_id().replace(".", "_")
            paper_dir = papers_dir / safe_id
            paper_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{safe_id}.pdf"
            pdf_path = Path(
                paper.download_pdf(dirpath=str(paper_dir), filename=filename)
            )
            logger.info("Successfully downloaded PDF to: '%s'", pdf_path)
            return pdf_path
        except ArxivNotFoundError:
            raise
        except Exception as e:
            raise ArxivDownloadError(
                f"Failed to download paper '{arxiv_id}': {str(e)}"
            ) from e

    def fetch_metadata(self, arxiv_id: str) -> ArxivMetadata:
        try:
            paper = self._fetch_single_paper(arxiv_id)
            metadata = self._paper_to_metadata(paper)
            logger.info("Successfully fetched metadata for arXiv ID: '%s'", arxiv_id)
            return metadata
        except (ArxivNotFoundError, ArxivPaperError):
            raise
        except Exception as e:
            logger.error(
                "Failed to process metadata for arXiv ID: '%s': %s", arxiv_id, str(e)
            )
            raise ArxivPaperError(f"Could not process metadata for '{arxiv_id}'") from e

    def _fetch_single_paper(self, arxiv_id: str) -> arxiv.Result:
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            return next(self.client.results(search))
        except StopIteration as e:
            raise ArxivNotFoundError(
                f"Paper not found for arXiv ID: '{arxiv_id}'"
            ) from e
        except Exception as e:
            raise ArxivPaperError(
                f"An unexpected error occurred while fetching '{arxiv_id}': {e}"
            ) from e

    @staticmethod
    def _paper_to_metadata(paper: arxiv.Result) -> ArxivMetadata:
        author_names = [author.name.strip() for author in paper.authors if author.name]

        return ArxivMetadata(
            arxiv_id=paper.get_short_id(),
            title=paper.title,
            authors=author_names,
            published=paper.published,
            updated=paper.updated,
            abstract=paper.summary,
            journal_ref=paper.journal_ref,
            primary_category=paper.primary_category,
            categories=paper.categories or [],
            doi=paper.doi,
            pdf_url=HttpUrl(paper.pdf_url) if paper.pdf_url else None,
        )

    def get_title_and_authors(self, arxiv_id: str) -> tuple[str, str]:
        paper = self._fetch_single_paper(arxiv_id)
        author_names = ", ".join([author.name for author in paper.authors])
        return paper.title, author_names

    def search_by_title(self, title: str, max_results: int = 10) -> str | None:
        normalized_query_title = self._normalize_title(title)
        queries_to_try = [f'ti:"{title}"']
        if "-" in title:
            queries_to_try.append(f'ti:"{title.replace("-", " ")}"')

        for query in queries_to_try:
            try:
                search = arxiv.Search(query=query, max_results=max_results)
                for paper in self.client.results(search):
                    if self._normalize_title(paper.title) == normalized_query_title:
                        return paper.get_short_id()
            except Exception as e:
                logger.warning("Search query '%s' failed: %s", query, e)
                continue

        logger.warning("No exact title match found for: '%s'", title)
        return None

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"[\s.-]+", "", title).lower()
