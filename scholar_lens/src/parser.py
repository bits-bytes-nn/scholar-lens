import asyncio
import base64
import io
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Coroutine

import boto3
import fitz
import httpx
from bs4 import BeautifulSoup, Tag
from langchain.schema.runnable import Runnable
from langchain.schema.output_parser import StrOutputParser
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator
from unstructured.documents.elements import Image as UnstructuredImage
from unstructured.partition.pdf import partition_pdf

from .constants import AppConstants, EnvVars, LanguageModelId, LocalPaths
from .logger import logger
from .prompts import FigureAnalysisPrompt
from .utils import (
    BedrockLanguageModelFactory,
    RetryableBase,
    extract_text_from_html,
)

DEFAULT_RESIZE_QUALITY: int = 85
DEFAULT_TIMEOUT: int = 60
MAX_IMAGE_SIZE_BYTES: int = 5242880
MIN_FIGURE_AREA: float = 10000.0


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
    @RetryableBase._retry("figure_analysing")
    async def from_llm(
        cls,
        figure_analyser: Runnable,
        figure_id: str,
        path: str,
        caption: str | None = None,
    ) -> "Figure":
        analysis = None
        try:
            image_data = await cls._get_image_data(path)
            analysis = await figure_analyser.ainvoke(
                {"caption": caption, "image_data": image_data}
            )
        except FigureParseError as e:
            logger.warning("Failed to get image data for figure '%s': %s", figure_id, e)
        except Exception as e:
            raise RuntimeError(
                f"LLM failed to analyze figure '{figure_id}': {e}"
            ) from e

        return cls(figure_id=figure_id, path=path, caption=caption, analysis=analysis)

    @staticmethod
    async def _get_image_data(path: str) -> str:
        image_bytes: bytes

        if path.startswith(("http://", "https://")):
            try:
                async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                    response = await client.get(path)
                    response.raise_for_status()
                    image_bytes = response.content
            except httpx.HTTPError as e:
                raise FigureParseError(f"HTTP error for image '{path}': {e}") from e
        else:
            try:
                with open(path, "rb") as f:
                    image_bytes = f.read()
            except IOError as e:
                raise FigureParseError(f"Failed to read file '{path}': {e}") from e

        processed_bytes = Figure._resize_if_needed(image_bytes)
        return base64.b64encode(processed_bytes).decode("utf-8")

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

                if quality <= 10:
                    new_width = int(img.width * 0.8)
                    new_height = int(img.height * 0.8)
                    if new_width < 100 or new_height < 100:
                        raise FigureParseError(
                            "Cannot resize image below 5MB limit even at minimum size"
                        )
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    quality = DEFAULT_RESIZE_QUALITY
                else:
                    quality -= 10

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

    async def __aenter__(self) -> "BaseParser":
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
    ) -> None:
        super().__init__(timeout=timeout)
        self.figure_analyser = self._initialize_chain(
            figure_analysis_model_id, boto_session
        )

    @staticmethod
    def _initialize_chain(
        model_id: LanguageModelId,
        boto_session: boto3.Session,
    ) -> Runnable:
        llm_factory = BedrockLanguageModelFactory(boto_session=boto_session)
        figure_analyser_llm = llm_factory.get_model(model_id, temperature=0.0)

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
        soup = BeautifulSoup(html_content, "html.parser")
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

            if response.status_code in (301, 302, 303, 307, 308):
                logger.warning("Redirect detected for URL '%s'", url)
                raise ContentParseError("Redirect detected")

            return response.text
        except httpx.HTTPError as e:
            raise ContentParseError(f"Failed to fetch HTML from '{url}': {e}") from e


class HTMLRichParser(RichParser, HTMLParser):
    async def parse(self, arxiv_id: str, extract_text: bool = True) -> HTMLParseResult:
        try:
            html_content, base_url = await self._fetch_html_with_fallback(arxiv_id)
            self.url = base_url
            soup = BeautifulSoup(html_content, "html.parser")

            is_ar5iv = "ar5iv" in self.url
            figures = await self._extract_figures(soup, is_ar5iv)
            content = self._extract_content(soup, extract_text)

            logger.info("Extracted %d characters from HTML", len(content.text))
            logger.info("Extracted %d figures from HTML", len(figures))

            return HTMLParseResult(content=content, figures=figures)

        except ContentParseError as e:
            raise ParserError(f"Failed to parse HTML for arXiv ID: '{arxiv_id}'") from e

    def _construct_figure_url(self, img_src: str, is_ar5iv: bool) -> str:
        if not self.url:
            raise ParserError(
                "Base URL for the paper is not set. Cannot construct absolute figure URL."
            )

        if is_ar5iv:
            base = AppConstants.External.AR5IV_LABS_HTML.value.replace("/html", "")
            return f"{base}/{img_src}"

        base_url = self.url.rstrip("/")
        return f"{base_url}/{img_src}"

    async def _extract_figures(
        self, soup: BeautifulSoup, is_ar5iv: bool
    ) -> list[Figure]:
        figure_tasks: list[Coroutine[Any, Any, Figure]] = []
        seen_srcs = set()

        for i, figure_tag in enumerate(soup.select(".ltx_figure")):
            img = figure_tag.select_one("img")
            caption_tag = figure_tag.select_one("figcaption")
            if isinstance(img, Tag) and img.get("src") and isinstance(caption_tag, Tag):
                src = img.get("src")
                if isinstance(src, str):
                    path = self._construct_figure_url(src, is_ar5iv)
                    caption = caption_tag.get_text(strip=True)
                    figure_tasks.append(
                        Figure.from_llm(self.figure_analyser, str(i), path, caption)
                    )
                    seen_srcs.add(src)

        offset = len(figure_tasks)
        for i, img in enumerate(
            soup.select(".ltx_td > img.ltx_graphics"), start=offset
        ):
            if isinstance(img, Tag) and (src := img.get("src")):
                if isinstance(src, str) and src not in seen_srcs:
                    path = self._construct_figure_url(src, is_ar5iv)
                    alt_text_raw = img.get("alt", "Refer to caption")
                    alt_text = str(alt_text_raw) if alt_text_raw else "Refer to caption"
                    caption = alt_text.strip()
                    figure_tasks.append(
                        Figure.from_llm(self.figure_analyser, str(i), path, caption)
                    )
                    seen_srcs.add(src)

        offset = len(figure_tasks)
        for i, img in enumerate(soup.select("img.ltx_graphics"), start=offset):
            if isinstance(img, Tag) and (src := img.get("src")):
                if isinstance(src, str) and src not in seen_srcs:
                    path = self._construct_figure_url(src, is_ar5iv)
                    alt_text_raw = img.get("alt", "Refer to caption")
                    alt_text = str(alt_text_raw) if alt_text_raw else "Refer to caption"
                    caption = alt_text.strip()
                    figure_tasks.append(
                        Figure.from_llm(self.figure_analyser, str(i), path, caption)
                    )
                    seen_srcs.add(src)

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


class PDFParser(RichParser):
    VALID_CATEGORIES = frozenset(
        {
            "caption",
            "chart",
            "equation",
            "figure",
            "footer",
            "footnote",
            "header",
            "heading1",
            "index",
            "list",
            "paragraph",
            "table",
        }
    )
    FIGURE_CATEGORIES = frozenset({"chart", "figure"})

    def __init__(
        self,
        figure_analysis_model_id: LanguageModelId,
        boto_session: boto3.Session,
        timeout: int = DEFAULT_TIMEOUT,
        api_key: str | None = None,
    ):
        super().__init__(figure_analysis_model_id, boto_session, timeout)
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
            content_html = response.get("content", {}).get("html", "").strip()

            content = Content(
                text=(
                    extract_text_from_html(content_html)
                    if extract_text
                    else content_html
                )
            )
            logger.info("Successfully extracted %d figures from PDF", len(figures))
            return figures, content
        except ParserError as e:
            logger.warning("Failed to parse PDF document '%s': %s", pdf_path, e)
            logger.info("Attempting fallback to Unstructured parser")
            return await self._parse_with_fallback(pdf_path, figures_dir, extract_text)

    async def _get_or_parse_document(
        self, pdf_path: Path, use_cache: bool
    ) -> dict[str, Any]:
        parsed_path = pdf_path.parent / LocalPaths.PARSED_FILE.value
        if use_cache and parsed_path.exists():
            return self._load_cached_response(parsed_path)

        response = await self._request_document_parse(pdf_path)
        self._cache_response(parsed_path, response)
        return response

    @staticmethod
    def _cache_response(path: Path, response: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.warning("Failed to cache response: %s", e)

    @staticmethod
    def _load_cached_response(path: Path) -> dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
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
        except IOError as e:
            raise ContentParseError(f"Cannot read PDF file '{pdf_path}': {e}") from e
        except httpx.HTTPError as e:
            raise ContentParseError(f"Document parsing API request failed: {e}") from e

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

        tasks = [
            Figure.from_llm(
                figure_analyser=self.figure_analyser,
                figure_id=str(fd["figure_id"]) or str(i),
                path=str(path),
                caption=fd["caption"] or None,
            )
            for i, (fd, path) in enumerate(zip(figure_data, paths))
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
        try:
            figures_dir = figures_dir or pdf_path.parent / LocalPaths.FIGURES_DIR.value
            figures_dir.mkdir(parents=True, exist_ok=True)

            elements = partition_pdf(
                filename=str(pdf_path),
                extract_images_in_pdf=True,
                extract_image_block_output_dir=str(figures_dir),
                strategy="hi_res",
                infer_table_structure=True,
            )

            content_parts = []
            valid_figure_paths = []
            if extract_text:
                for element in elements:
                    if isinstance(element, UnstructuredImage):
                        img_path_str = getattr(element.metadata, "image_path", None)
                        if img_path_str:
                            img_path = Path(img_path_str)
                            try:
                                with Image.open(img_path) as img:
                                    width, height = img.size

                                if width * height >= MIN_FIGURE_AREA:
                                    content_parts.append("[Image: alt=, src=]")
                                    valid_figure_paths.append(img_path)
                                else:
                                    logger.debug(
                                        "Skipping small image from Unstructured: '%s' (area: %d)",
                                        img_path.name,
                                        width * height,
                                    )
                            except Exception as e:
                                logger.warning(
                                    "Could not process image for size check '%s': %s",
                                    img_path,
                                    e,
                                )

                    elif hasattr(element, "text") and element.text:
                        content_parts.append(element.text)

            content = Content(text="\n\n".join(content_parts) if extract_text else "")

            figure_tasks = []
            for img_path in valid_figure_paths:
                figure_tasks.append(
                    Figure.from_llm(
                        figure_analyser=self.figure_analyser,
                        figure_id=img_path.stem,
                        path=str(img_path),
                        caption=None,
                    )
                )

            all_figures = await asyncio.gather(*figure_tasks, return_exceptions=True)
            figures = []
            for f in all_figures:
                if isinstance(f, Exception):
                    logger.warning("Figure extraction failed: %s", f)
                    continue
                if isinstance(f, Figure):
                    figures.append(f)

            logger.info(
                "Unstructured parser extracted %d figures and %d characters",
                len(figures),
                len(content.text),
            )
            return figures, content

        except Exception as e:
            logger.error("Unstructured parser also failed: %s", e)
            raise ParserError(
                f"Both Upstage and Unstructured parsers failed: {e}"
            ) from e


def extract_figures_from_pdf(
    pdf_path: Path,
    figures_dir: Path,
    regions: list[Region],
    zoom: int = 2,
    dpi: int = 300,
    figure_name_template: str = "{idx}.png",
) -> list[Path]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {e}") from e

    with doc:
        return [
            path
            for path in (
                _process_region(
                    doc, region, idx, figures_dir, zoom, dpi, figure_name_template
                )
                for idx, region in enumerate(regions)
            )
            if path is not None
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
