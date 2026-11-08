"""The paper a review or summary is written from: its parsed text, figures,
extracted attributes and citations, and the generated review text."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from scholar_lens.domain.ids import arxiv_id_from_reference, normalize_arxiv_id
from scholar_lens.domain.text import is_placeholder


class Content(BaseModel):
    model_config = ConfigDict(frozen=False)
    text: str = Field(default="")
    # The layout parser failed and only the PDF's text layer was read, so the
    # paper's figures and tables are unavailable.
    text_only: bool = Field(default=False)
    # Source downgrades (the HTML rendering failed and the PDF was read …),
    # each a run note for the PR's review notes.
    notes: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        return f"Content(text='{self.text[:50]}...')"

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else ""


class Figure(BaseModel):
    model_config = ConfigDict(frozen=True)
    figure_id: str
    path: str
    caption: str | None = Field(default=None)
    analysis: str | None = Field(default=None)

    def __str__(self) -> str:
        return (
            f"Figure(figure_id='{self.figure_id}', path='{self.path}', "
            f"caption='{self.caption}', analysis='{self.analysis}')"
        )

    @field_validator("caption", "analysis", mode="before")
    @classmethod
    def validate_text_fields(cls, v: str | None) -> str | None:
        return v.strip() if v is not None else v


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

    def linkable_arxiv(self) -> set[str]:
        """The unversioned arXiv ids a post about the paper may link: the
        paper itself and its citations' ids (checked against arXiv when they
        were extracted). A writer links other ids from memory."""
        return {
            re.sub(r"v\d+$", "", arxiv_id)
            for arxiv_id in (
                normalize_arxiv_id(self.arxiv_id),
                *(c.arxiv_id for c in self.citations),
            )
            if arxiv_id
        }

    @field_validator("repo_urls", mode="before")
    @classmethod
    def convert_str_urls(cls, v: Any) -> list[HttpUrl]:
        if v is None:
            return []
        return [HttpUrl(url) for url in v]


class ReviewText(NamedTuple):
    """The generated review: body sections, TL;DR, closing assessment, and the
    title translated into the output language (the post's subtitle)."""

    body: str
    key_takeaways: str
    closing: str
    subtitle: str = ""
    # The post's search snippet (front-matter ``description``); empty lets the
    # blog derive one.
    description: str = ""
    # Degraded-output notes (sections skipped or below the bar, unusable
    # finalization) for the PR checklist and the job result.
    notes: tuple[str, ...] = ()


def backfill_metadata(
    meta_fields: dict[str, Any], attributes: Attributes, *, title_is_fallback: bool
) -> dict[str, Any]:
    """Prefer the title/authors parsed from the paper text when the source had
    none (a bare PDF URL yields a URL-derived title and placeholder authors)."""
    title = str(meta_fields.get("title", "")).strip()
    if not is_placeholder(attributes.title) and (title_is_fallback or not title):
        meta_fields["title"] = attributes.title
    authors = [a for a in (meta_fields.get("authors") or []) if not is_placeholder(a)]
    if not authors and attributes.authors:
        meta_fields["authors"] = attributes.authors
    return meta_fields
