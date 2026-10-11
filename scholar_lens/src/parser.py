import asyncio
import base64
import io
import json
import os
import re
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import Any, Self
from urllib.parse import urljoin

import boto3
import fitz
import httpx
from bs4 import BeautifulSoup, Tag
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import AppConstants, EnvVars, LanguageModelId, LocalPaths
from .logger import logger
from .metrics import TokenBudgetExceeded
from .pinned_transport import PinnedAsyncHTTPTransport
from .prompts import FigureAnalysisPrompt
from .url_guard import UnsafeUrlError, assert_url_is_public
from .utils import (
    BedrockLanguageModelFactory,
    RetryableBase,
    extract_text_from_html,
)
from .utils.helpers import preserve_listing_spaces
from .utils.images import is_svg, svg_to_png
from .utils.retry import is_transient_error

DEFAULT_RESIZE_QUALITY: int = 85
DEFAULT_TIMEOUT: int = 60
MAX_IMAGE_SIZE_BYTES: int = 5242880
# Resize fallback loop tuning: once JPEG quality bottoms out, shrink dimensions
# by RESIZE_SCALE_FACTOR per pass and give up below MIN_RESIZE_DIMENSION px.
MIN_RESIZE_QUALITY: int = 10
RESIZE_QUALITY_STEP: int = 10
RESIZE_SCALE_FACTOR: float = 0.8
MIN_RESIZE_DIMENSION: int = 100
# A parse yielding less text than this is treated as failed (e.g. an arXiv
# "conversion failed" page or an empty document-parse response).
MIN_PARSED_TEXT_CHARS: int = 500


class ParserError(Exception):
    pass


class ContentParseError(ParserError):
    pass


class FigureParseError(ParserError):
    pass


class Content(BaseModel):
    model_config = ConfigDict(frozen=False)
    text: str = Field(default="")

    def __str__(self) -> str:
        return f"Content(text='{self.text[:50]}...')"

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else ""


def _is_budget_breach(error: BaseException | None) -> bool:
    """A token-budget stop anywhere in the cause chain: the run must end."""
    while error is not None:
        if isinstance(error, TokenBudgetExceeded):
            return True
        error = error.__cause__ or error.__context__
    return False


class Figure(BaseModel, RetryableBase):
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

    @classmethod
    async def from_llm(
        cls,
        figure_analyser: Runnable,
        figure_id: str,
        path: str,
        caption: str | None = None,
    ) -> "Figure":
        try:
            image_data = await cls._get_image_data(path)
        except FigureParseError as e:
            logger.warning("Failed to get image data for figure '%s': %s", figure_id, e)
            return cls(figure_id=figure_id, path=path, caption=caption)
        try:
            analysis = await cls._analyse(figure_analyser, caption, image_data)
        except Exception as e:
            if _is_budget_breach(e):
                raise
            # Retries exhausted, or the model refused the image: the figure (and
            # its caption) still belongs in the post, unanalysed (the pipeline
            # notes unanalysed figures on the PR).
            kind = "transient" if is_transient_error(e) else "model"
            logger.warning(
                "Figure '%s' left unanalysed (%s error): %s", figure_id, kind, e
            )
            return cls(figure_id=figure_id, path=path, caption=caption)
        return cls(figure_id=figure_id, path=path, caption=caption, analysis=analysis)

    @staticmethod
    @RetryableBase._retry("figure_analysing")
    async def _analyse(
        figure_analyser: Runnable, caption: str | None, image_data: str
    ) -> str:
        return await figure_analyser.ainvoke(
            {"caption": caption, "image_data": image_data}
        )

    @staticmethod
    async def _get_image_data(path: str) -> str:
        image_bytes: bytes

        if path.startswith(("http://", "https://")):
            # Figure URLs can originate from untrusted paper/page content; run
            # them through the SSRF guard before fetching server-side.
            try:
                assert_url_is_public(path)
            except UnsafeUrlError as e:
                raise FigureParseError(f"Unsafe image URL '{path}': {e}") from e
            try:
                # PinnedAsyncHTTPTransport pins the connection to the validated
                # IP (no DNS-rebinding window after the assert_url_is_public check
                # above, and any redirect hop is re-validated).
                async with httpx.AsyncClient(
                    timeout=DEFAULT_TIMEOUT, transport=PinnedAsyncHTTPTransport()
                ) as client:
                    response = await client.get(path)
                    response.raise_for_status()
                    image_bytes = response.content
            except UnsafeUrlError as e:
                raise FigureParseError(f"Unsafe image URL '{path}': {e}") from e
            except httpx.HTTPError as e:
                raise FigureParseError(f"HTTP error for image '{path}': {e}") from e
        else:
            try:
                with open(path, "rb") as f:
                    image_bytes = f.read()
            except OSError as e:
                raise FigureParseError(f"Failed to read file '{path}': {e}") from e

        # Vector figures are rasterized first (the vision model takes bitmaps).
        if is_svg(image_bytes):
            try:
                # Off the event loop: the render waits on a child process.
                image_bytes = await asyncio.to_thread(svg_to_png, image_bytes)
            except (ValueError, UnicodeDecodeError) as e:
                raise FigureParseError(f"Could not render SVG '{path}': {e}") from e
        # Normalise to JPEG so the data URI's declared "image/jpeg" media type is
        # always accurate (figures may be PNG/WebP/etc.); then resize if needed.
        image_bytes = Figure._to_jpeg(image_bytes)
        processed_bytes = Figure._resize_if_needed(image_bytes)
        return base64.b64encode(processed_bytes).decode("utf-8")

    @staticmethod
    def _to_jpeg(image_bytes: bytes) -> bytes:
        """Re-encode any supported image to JPEG; pass bytes through unchanged if
        decoding fails (the downstream model still sniffs the real bytes)."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as opened:
                rgb = (
                    opened.convert("RGB")
                    if opened.mode in ("RGBA", "P", "LA", "CMYK")
                    else opened
                )
                buffer = io.BytesIO()
                rgb.save(buffer, format="JPEG", quality=DEFAULT_RESIZE_QUALITY)
                return buffer.getvalue()
        except Exception as e:  # noqa: BLE001 - keep original bytes on any decode error
            logger.warning("Could not re-encode image to JPEG: %s", e)
            return image_bytes

    @staticmethod
    def _resize_if_needed(image_bytes: bytes, max_iterations: int = 20) -> bytes:
        if len(image_bytes) <= MAX_IMAGE_SIZE_BYTES:
            return image_bytes

        logger.info(
            "Image size (%.2f MB) exceeds limit. Resizing...",
            len(image_bytes) / (1024 * 1024),
        )
        try:
            img: Image.Image = Image.open(io.BytesIO(image_bytes))

            if img.mode in ("RGBA", "P", "CMYK"):
                img = img.convert("RGB")

            quality = DEFAULT_RESIZE_QUALITY
            iteration = 0

            while iteration < max_iterations:
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality)

                if len(buffer.getvalue()) <= MAX_IMAGE_SIZE_BYTES:
                    resized_bytes = buffer.getvalue()
                    logger.info(
                        "Image successfully resized to %.2f MB after %d iterations.",
                        len(resized_bytes) / (1024 * 1024),
                        iteration + 1,
                    )
                    return resized_bytes

                if quality <= MIN_RESIZE_QUALITY:
                    new_width = int(img.width * RESIZE_SCALE_FACTOR)
                    new_height = int(img.height * RESIZE_SCALE_FACTOR)
                    if (
                        new_width < MIN_RESIZE_DIMENSION
                        or new_height < MIN_RESIZE_DIMENSION
                    ):
                        raise FigureParseError(
                            "Cannot resize image below 5MB limit even at minimum size"
                        )
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    quality = DEFAULT_RESIZE_QUALITY
                else:
                    quality -= RESIZE_QUALITY_STEP

                iteration += 1

            raise FigureParseError(
                f"Failed to resize image below 5MB limit after {max_iterations} iterations"
            )

        except FigureParseError:
            raise
        except Exception as e:
            logger.error(
                "Failed to resize image due to unexpected error: %s",
                e,
                exc_info=True,
            )
            raise FigureParseError(f"Image resizing failed: {e}") from e

    @field_validator("caption", "analysis", mode="before")
    @classmethod
    def validate_text_fields(cls, v: str | None) -> str | None:
        return v.strip() if v is not None else v


class HTMLParseResult(BaseModel):
    content: Content
    figures: list[Figure] = Field(default_factory=list)


class Region(BaseModel):
    model_config = ConfigDict(frozen=True)
    page: int = Field(gt=0)
    coordinates: Sequence[dict[str, float]]

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(
        cls, v: Sequence[dict[str, float]]
    ) -> Sequence[dict[str, float]]:
        if len(v) < 4:
            raise ValueError("Coordinates must contain at least 4 points")
        return v


class BaseParser:
    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self._async_client: httpx.AsyncClient | None = None

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=False
            )
        return self._async_client

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()


class RichParser(BaseParser):
    def __init__(
        self,
        figure_analysis_model_id: LanguageModelId,
        boto_session: boto3.Session,
        timeout: int = DEFAULT_TIMEOUT,
        callbacks: list[Any] | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self.figure_analyser = self._initialize_chain(
            figure_analysis_model_id, boto_session, callbacks
        )

    @staticmethod
    def _initialize_chain(
        model_id: LanguageModelId,
        boto_session: boto3.Session,
        callbacks: list[Any] | None = None,
    ) -> Runnable:
        llm_factory = BedrockLanguageModelFactory(boto_session=boto_session)
        figure_analyser_llm = llm_factory.get_model(
            model_id, temperature=0.0, callbacks=callbacks
        )

        return (
            FigureAnalysisPrompt.get_prompt() | figure_analyser_llm | StrOutputParser()
        )

    async def parse(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("RichParser is an abstract class")


class HTMLParser(BaseParser):
    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        super().__init__(timeout)
        self.url: str | None = None

    async def parse(self, arxiv_id: str, extract_text: bool = True) -> HTMLParseResult:
        html_content, _ = await self._fetch_html_with_fallback(arxiv_id)
        soup = BeautifulSoup(preserve_listing_spaces(html_content), "html.parser")
        content = self._extract_content(soup, extract_text)
        return HTMLParseResult(content=content)

    @staticmethod
    def _extract_content(soup: BeautifulSoup, extract_text: bool) -> Content:
        for selector in [".ltx_page_main", "body"]:
            if content_tag := soup.select_one(selector):
                content_str = str(content_tag)
                text = (
                    extract_text_from_html(content_str) if extract_text else content_str
                )
                return Content(text=text)
        return Content()

    async def _fetch_html_with_fallback(self, arxiv_id: str) -> tuple[str, str]:
        urls_to_try = [
            f"{AppConstants.External.ARXIV_HTML.value}/{arxiv_id}",
            f"{AppConstants.External.AR5IV_LABS_HTML.value}/{arxiv_id}",
        ]
        for url in urls_to_try:
            try:
                html = await self._fetch_html_from_url(url)
                return html, url
            except ContentParseError:
                logger.warning("Failed to fetch from '%s', trying next URL", url)
                continue
        raise ContentParseError(f"Failed to fetch HTML for arXiv ID: '{arxiv_id}'")

    async def _fetch_html_from_url(self, url: str) -> str:
        try:
            response = await self.async_client.get(url)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as e:
            raise ContentParseError(f"Failed to fetch HTML from '{url}': {e}") from e


class HTMLRichParser(RichParser, HTMLParser):
    async def parse(self, arxiv_id: str, extract_text: bool = True) -> HTMLParseResult:
        try:
            html_content, base_url = await self._fetch_html_with_fallback(arxiv_id)
            self.url = base_url
            soup = BeautifulSoup(preserve_listing_spaces(html_content), "html.parser")

            figures = await self._extract_figures(soup)
            content = self._extract_content(soup, extract_text)
            if extract_text and len(content.text) < MIN_PARSED_TEXT_CHARS:
                raise ContentParseError(
                    f"HTML rendering has only {len(content.text)} characters of text."
                )

            logger.info("Extracted %d characters from HTML", len(content.text))
            logger.info("Extracted %d figures from HTML", len(figures))

            return HTMLParseResult(content=content, figures=figures)

        except ContentParseError as e:
            raise ParserError(f"Failed to parse HTML for arXiv ID: '{arxiv_id}'") from e

    def _construct_figure_url(self, img_src: str) -> str:
        if not self.url:
            raise ParserError(
                "Base URL for the paper is not set. Cannot construct absolute figure URL."
            )

        # Resolve against the page URL per RFC 3986 — never concatenate. arXiv
        # serves '.../html/<id>' (no trailing slash) with version-prefixed
        # relative srcs ('<id>v1/Media/x.png'), so appending yields a doubled id
        # segment that 404s; ar5iv uses root-relative srcs ('/html/<id>/...').
        # urljoin handles both, and a 404 here silently strips every figure from
        # the model's context (see attach_figure_context).
        return urljoin(self.url, img_src)

    async def _extract_figures(self, soup: BeautifulSoup) -> list[Figure]:
        # Every image the text extraction turns into a marker: bitmaps and SVG
        # <object> panels alike, including the second panel of a figure and
        # figures nested inside another figure.
        root = soup.select_one(".ltx_page_main") or soup
        figure_tasks: list[Coroutine[Any, Any, Figure]] = []
        seen_srcs: set[str] = set()
        for media in root.select(_FIGURE_MEDIA):
            src = media.get("data" if media.name == "object" else "src")
            if not isinstance(src, str) or not src or src.startswith("data:"):
                continue
            if src in seen_srcs:
                continue
            seen_srcs.add(src)
            figure_tasks.append(
                Figure.from_llm(
                    self.figure_analyser,
                    str(len(figure_tasks)),
                    self._construct_figure_url(src),
                    _figure_caption(media),
                )
            )

        all_figures = await asyncio.gather(*figure_tasks, return_exceptions=True)

        unique_figures: list[Figure] = []
        seen_paths = set()
        for figure in all_figures:
            if isinstance(figure, Exception):
                logger.warning("Figure extraction failed: %s", figure)
                continue
            if isinstance(figure, Figure) and figure.path not in seen_paths:
                unique_figures.append(figure)
                seen_paths.add(figure.path)

        if len(all_figures) != len(unique_figures):
            logger.info(
                "Deduplicated %d figures down to %d unique figures based on path.",
                len(all_figures),
                len(unique_figures),
            )

        return unique_figures


_FIGURE_MEDIA = 'img.ltx_graphics, .ltx_figure img, object[type="image/svg+xml"]'


def _own_caption(figure: Tag, media: Tag) -> Tag | None:
    """The caption of ``figure`` that belongs to ``media``: the figure's own
    caption, or, when one LaTeXML figure holds several captioned figures
    (``\\includegraphics; \\caption; \\includegraphics; \\caption``), the
    first of its captions after the image."""
    captions = figure.find_all("figcaption", recursive=False)
    if len(captions) <= 1:
        return captions[0] if captions else None
    following = media.find_next("figcaption")
    return following if following in captions else captions[-1]


def _figure_caption(media: Tag) -> str:
    """Caption text for an image: its nearest captioned figure, with the
    enclosing figure's caption first when that one is only a sub-caption
    ("(a) ..."); else the image's alt text."""
    parts: list[str] = []
    for figure in media.find_parents("figure"):
        caption = _own_caption(figure, media)
        if caption is None:
            continue
        text = _caption_text(caption)
        parts.insert(0, text)
        if not _is_subfigure(figure, text):
            break
    if any(parts):
        return " ".join(p for p in parts if p)
    alt = str(media.get("alt") or "").strip()
    return alt or _PLACEHOLDER_CAPTION


# LaTeXML ids sub-figures "<figure id>.sfN"; their captions read "(a) ...".
_SUBFIGURE_ID = re.compile(r"\.sf\d+$")
_SUBCAPTION = re.compile(r"^\(?[a-z]\)\s")


def _is_subfigure(figure: Tag, caption: str) -> bool:
    return (
        "ltx_figure_panel" in (figure.get("class") or [])
        or bool(_SUBFIGURE_ID.search(str(figure.get("id") or "")))
        or bool(_SUBCAPTION.match(caption))
    )


def _caption_text(caption: Tag) -> str:
    return " ".join(extract_text_from_html(str(caption)).split())


class PDFParser(RichParser):
    FIGURE_CATEGORIES = frozenset({"chart", "figure"})

    def __init__(
        self,
        figure_analysis_model_id: LanguageModelId,
        boto_session: boto3.Session,
        timeout: int = DEFAULT_TIMEOUT,
        api_key: str | None = None,
        callbacks: list[Any] | None = None,
    ):
        super().__init__(figure_analysis_model_id, boto_session, timeout, callbacks)
        self.api_key = api_key or os.environ.get(EnvVars.UPSTAGE_API_KEY.value)
        if not self.api_key:
            raise ValueError(
                f"{EnvVars.UPSTAGE_API_KEY.value} must be provided or set in environment"
            )

    async def parse(
        self,
        pdf_path: Path,
        figures_dir: Path | None = None,
        use_cache: bool = True,
        extract_text: bool = True,
    ) -> tuple[list[Figure], Content]:
        try:
            response = await self._get_or_parse_document(pdf_path, use_cache)
            elements = response.get("elements", [])

            figures_dir = figures_dir or pdf_path.parent / LocalPaths.FIGURES_DIR.value
            figures_dir.mkdir(parents=True, exist_ok=True)

            figures = await self._extract_figures(elements, pdf_path, figures_dir)
            content_html = _bind_figure_sources(
                response.get("content", {}).get("html", "").strip(), figures
            )
            text = extract_text_from_html(content_html)
            if len(text) < MIN_PARSED_TEXT_CHARS:
                raise ContentParseError(
                    f"Document parse returned only {len(text)} characters of text."
                )
            content = Content(text=text if extract_text else content_html)
            logger.info("Successfully extracted %d figures from PDF", len(figures))
            return figures, content
        except ParserError as e:
            logger.warning("Failed to parse PDF document '%s': %s", pdf_path, e)
            logger.info("Attempting fallback to PyMuPDF text extraction")
            return await self._parse_with_fallback(pdf_path, figures_dir, extract_text)

    async def _get_or_parse_document(
        self, pdf_path: Path, use_cache: bool
    ) -> dict[str, Any]:
        parsed_path = pdf_path.parent / LocalPaths.PARSED_FILE.value
        if use_cache and parsed_path.exists():
            cached = self._load_cached_response(parsed_path)
            if _has_content(cached):
                return cached
            logger.warning("Ignoring empty cached parse at '%s'.", parsed_path)

        response = await self._request_document_parse(pdf_path)
        # Never cache an empty response: it would pin the failure for re-runs.
        if _has_content(response):
            self._cache_response(parsed_path, response)
        return response

    @staticmethod
    def _cache_response(path: Path, response: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.warning("Failed to cache response: %s", e)

    @staticmethod
    def _load_cached_response(path: Path) -> dict[str, Any]:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ContentParseError(f"Failed to load cached response: {e}") from e

    async def _request_document_parse(self, pdf_path: Path) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with open(pdf_path, "rb") as f:
                files = {"document": (pdf_path.name, f.read(), "application/pdf")}
                response = await self.async_client.post(
                    AppConstants.External.UPSTAGE_DOCUMENT_PARSE.value,
                    headers=headers,
                    files=files,
                )
                response.raise_for_status()
                return response.json()
        except OSError as e:
            raise ContentParseError(f"Cannot read PDF file '{pdf_path}': {e}") from e
        except httpx.HTTPError as e:
            raise ContentParseError(f"Document parsing API request failed: {e}") from e
        except ValueError as e:
            # response.json() raises JSONDecodeError (a ValueError) on a non-JSON
            # / truncated API response — surface as a parse error, not a crash.
            raise ContentParseError(
                f"Document parsing API returned invalid JSON: {e}"
            ) from e

    async def _extract_figures(
        self, elements: list[dict[str, Any]], pdf_path: Path, figures_dir: Path
    ) -> list[Figure]:
        figure_data = []
        for element in elements:
            category = element.get("category", "").lower()
            if category in self.FIGURE_CATEGORIES and element.get("coordinates"):
                soup = BeautifulSoup(
                    element.get("content", {}).get("html", ""), "html.parser"
                )
                img = soup.find("img")
                caption = ""
                if isinstance(img, Tag) and (alt_text := img.get("alt")):
                    caption = str(alt_text).strip()

                figure_data.append(
                    {
                        "page": element["page"],
                        "coordinates": element["coordinates"],
                        "caption": caption,
                        "figure_id": element.get("id", ""),
                    }
                )

        if not figure_data:
            return []

        regions = [Region(**fd) for fd in figure_data]
        paths = extract_figures_from_pdf(pdf_path, figures_dir, regions)

        # ``paths`` is aligned with ``regions`` (None where a crop failed), so a
        # skipped region can never shift captions onto the wrong image.
        tasks = [
            Figure.from_llm(
                figure_analyser=self.figure_analyser,
                figure_id=str(fd["figure_id"]) or str(i),
                path=str(path),
                caption=fd["caption"] or None,
            )
            for i, (fd, path) in enumerate(zip(figure_data, paths, strict=True))
            if path is not None
        ]

        all_figures = await asyncio.gather(*tasks, return_exceptions=True)

        unique_figures = []
        seen_paths = set()
        for figure in all_figures:
            if isinstance(figure, Exception):
                logger.warning("Figure extraction failed: %s", figure)
                continue
            if not isinstance(figure, Figure):
                continue
            if figure.path not in seen_paths:
                unique_figures.append(figure)
                seen_paths.add(figure.path)

        if len(all_figures) != len(unique_figures):
            logger.info(
                "Deduplicated %d figures down to %d unique figures based on path.",
                len(all_figures),
                len(unique_figures),
            )

        return unique_figures

    async def _parse_with_fallback(
        self,
        pdf_path: Path,
        figures_dir: Path | None = None,
        extract_text: bool = True,
    ) -> tuple[list[Figure], Content]:
        # Fallback when the primary (Upstage) parser fails, e.g. a 413 on a large
        # PDF. Use PyMuPDF (fitz): it reads the PDF's text layer directly, is
        # lightweight (no layout/OCR vision models, so it never OOMs the way
        # Unstructured "hi_res" did), and is already a dependency. For a
        # review/summary the text is what matters; figures are best-effort and
        # captured on the primary path when it succeeds, so the fallback is
        # text-only.
        try:
            text = await asyncio.to_thread(self._extract_text_with_fitz, pdf_path)
        except Exception as e:
            logger.error("PyMuPDF fallback parser failed: %s", e)
            raise ParserError(f"Both Upstage and PyMuPDF parsers failed: {e}") from e

        if not text.strip():
            raise ParserError(
                f"PyMuPDF extracted no text from '{pdf_path}' (the PDF may be "
                "scanned/image-only with no text layer)."
            )

        logger.info("PyMuPDF fallback parser extracted %d characters", len(text))
        return [], Content(text=text if extract_text else "")

    @staticmethod
    def _extract_text_with_fitz(pdf_path: Path) -> str:
        with fitz.open(str(pdf_path)) as doc:
            return "\n\n".join(page.get_text() for page in doc)


def extract_figures_from_pdf(
    pdf_path: Path,
    figures_dir: Path,
    regions: list[Region],
    zoom: int = 2,
    dpi: int = 300,
    figure_name_template: str = "{idx}.png",
) -> list[Path | None]:
    """Crop each region to an image; the result is aligned with ``regions``
    (``None`` for a region that could not be cropped)."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {e}") from e

    with doc:
        return [
            _process_region(
                doc, region, idx, figures_dir, zoom, dpi, figure_name_template
            )
            for idx, region in enumerate(regions)
        ]


def _is_valid_region(region: Region, doc: fitz.Document) -> bool:
    page_index = region.page - 1
    return 0 <= page_index < len(doc)


def _process_region(
    doc: fitz.Document,
    region: Region,
    idx: int,
    figures_dir: Path,
    zoom: int,
    dpi: int,
    name_template: str,
) -> Path | None:
    if not _is_valid_region(region, doc):
        logger.warning(
            "Invalid page number %d for region %d. Skipping.", region.page, idx
        )
        return None

    try:
        page = doc[region.page - 1]
        coords = region.coordinates

        if not (
            isinstance(coords, list)
            and len(coords) >= 4
            and all("x" in p and "y" in p for p in coords)
        ):
            logger.warning("Invalid coordinate format for region %d. Skipping.", idx)
            return None

        rect_coords = [
            coords[0]["x"] * page.rect.width,
            coords[0]["y"] * page.rect.height,
            coords[2]["x"] * page.rect.width,
            coords[2]["y"] * page.rect.height,
        ]

        clip_rect = fitz.Rect(*rect_coords)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, dpi=dpi)

        figure_path = figures_dir / name_template.format(idx=idx)
        pix.save(figure_path)
        return figure_path
    except Exception as e:
        logger.warning("Error processing region %d: %s", idx, e)
        return None


def _has_content(response: dict[str, Any]) -> bool:
    html = response.get("content", {}).get("html", "")
    return bool(response.get("elements")) and bool(str(html).strip())


def _bind_figure_sources(html: str, figures: list[Figure]) -> str:
    """Point each document-parse ``<figure id=N>`` image at its cropped file.

    The parse output has no image sources; tying markers to figures by element
    id (not by order) keeps every caption on the image it describes.
    """
    if not html or not figures:
        return html
    by_id = {fig.figure_id: Path(fig.path).name for fig in figures}
    soup = BeautifulSoup(html, "html.parser")
    for figure_tag in soup.find_all("figure"):
        if not isinstance(figure_tag, Tag):
            continue
        name = by_id.get(str(figure_tag.get("id", "")))
        img = figure_tag.find("img")
        if name and isinstance(img, Tag):
            img["src"] = name
    return str(soup)


_IMAGE_MARKER = re.compile(r"\[Image:\s*alt=(.*?),\s*src=(.*?)\]", re.DOTALL)


def _marker_text(value: str) -> str:
    """One line, no brackets or bare quotes: a figure marker must stay one
    parseable token through sentence splitting and section slicing."""
    flat = " ".join(value.split()).replace("[", "(").replace("]", ")")
    return flat.replace('"', "'")


_PLACEHOLDER_CAPTION = "Refer to caption"  # arXiv's alt/caption placeholder


def _real_text(value: str | None) -> str:
    text = (value or "").strip()
    return "" if text == _PLACEHOLDER_CAPTION else text


def _figure_description(figure: Figure) -> str:
    caption = _real_text(figure.caption)
    analysis = (figure.analysis or "").strip()
    if caption and analysis:
        return f"{caption} (Figure analysis, approximate: {analysis})"
    return caption or analysis


def attach_figure_context(
    text: str, figures: list[Figure], *, is_pdf_parsed: bool
) -> str:
    """Rewrite ``[Image: ...]`` markers into figure references the writers can
    cite: the image path (file name for PDF-parsed papers, whose figures are
    published with the post) plus a description: the paper's own caption
    first, then the vision analysis labeled as approximate (writers must not
    take its readings as the paper's numbers). Markers matching no extracted
    figure are removed; no marker is ever paired with a figure by position."""
    figure_map = {Path(fig.path).name: fig for fig in figures}

    def repl(match: re.Match) -> str:
        alt_value = _real_text(" ".join(match.group(1).split()).strip("[]"))
        src_value = match.group(2).strip()
        matched = figure_map.get(Path(src_value).name) if src_value else None
        if matched is None:
            if src_value:
                logger.warning("No extracted figure for image src '%s'", src_value)
            return ""
        path = Path(matched.path).name if is_pdf_parsed else matched.path
        alt_text = _marker_text(alt_value or _real_text(matched.caption))
        description = _marker_text(_figure_description(matched) or alt_value)
        return f'[Image: alt="{alt_text}", src="{path}", caption="{description}"]'

    return _IMAGE_MARKER.sub(repl, text)
