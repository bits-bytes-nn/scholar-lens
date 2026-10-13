"""Tests for the pure / offline pieces of the PDF figure parser.

These tests deliberately avoid the Bedrock and Upstage code paths. Only the
pydantic models and the local ``fitz``-based figure extraction are exercised;
no AWS or network calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest
import tenacity

from scholar_lens.src.parser import (
    Content,
    Figure,
    Region,
    _is_valid_region,
    extract_figures_from_pdf,
)
from scholar_lens.src.utils.retry import MAX_RETRIES

# A simple rectangle covering most of the page (normalised corner coords).
SQUARE_COORDS: list[dict[str, float]] = [
    {"x": 0.1, "y": 0.1},
    {"x": 0.5, "y": 0.1},
    {"x": 0.5, "y": 0.5},
    {"x": 0.1, "y": 0.5},
]


@pytest.fixture
def one_page_pdf(tmp_path: Path) -> Path:
    """Build a real single-page PDF so ``get_pixmap`` always succeeds."""
    pdf_path = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Hello figure")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class TestContentModel:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("  padded  ", "padded"),
            ("\n\ttabbed\n", "tabbed"),
            ("clean", "clean"),
        ],
    )
    def test_validate_text_strips_whitespace(self, raw: str, expected: str) -> None:
        assert Content(text=raw).text == expected

    def test_non_string_coerced_to_empty(self) -> None:
        assert Content(text=None).text == ""  # type: ignore[arg-type]

    def test_str_truncates_to_50_chars(self) -> None:
        content = Content(text="x" * 200)
        rendered = str(content)
        assert rendered == f"Content(text='{'x' * 50}...')"


class TestFigureModel:
    def test_validate_text_fields_strip_caption_and_analysis(self) -> None:
        fig = Figure(
            figure_id="f1",
            path="/tmp/f.png",
            caption="  a caption  ",
            analysis="\n analysis \t",
        )
        assert fig.caption == "a caption"
        assert fig.analysis == "analysis"

    def test_none_text_fields_preserved(self) -> None:
        fig = Figure(figure_id="f1", path="/tmp/f.png")
        assert fig.caption is None
        assert fig.analysis is None

    def test_resize_if_needed_returns_small_bytes_unchanged(self) -> None:
        small = b"not really an image but well under 5MB"
        assert Figure._resize_if_needed(small) is small


class TestRegionModel:
    def test_valid_region_accepted(self) -> None:
        region = Region(page=1, coordinates=SQUARE_COORDS)
        assert region.page == 1
        assert len(region.coordinates) == 4

    def test_fewer_than_four_coordinates_rejected(self) -> None:
        with pytest.raises(ValueError):
            Region(page=1, coordinates=SQUARE_COORDS[:3])

    @pytest.mark.parametrize("page", [0, -1])
    def test_non_positive_page_rejected(self, page: int) -> None:
        with pytest.raises(ValueError):
            Region(page=page, coordinates=SQUARE_COORDS)


class TestIsValidRegion:
    def test_in_range_page_is_valid(self, one_page_pdf: Path) -> None:
        doc = fitz.open(str(one_page_pdf))
        with doc:
            region = Region(page=1, coordinates=SQUARE_COORDS)
            assert _is_valid_region(region, doc) is True

    def test_page_beyond_document_is_invalid(self, one_page_pdf: Path) -> None:
        doc = fitz.open(str(one_page_pdf))
        with doc:
            region = Region(page=2, coordinates=SQUARE_COORDS)
            assert _is_valid_region(region, doc) is False


class TestExtractFiguresFromPdf:
    def test_produces_existing_png(self, one_page_pdf: Path, tmp_path: Path) -> None:
        figures_dir = tmp_path / "figures"
        region = Region(page=1, coordinates=SQUARE_COORDS)

        paths = extract_figures_from_pdf(one_page_pdf, figures_dir, [region])

        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".png"

    def test_out_of_range_region_keeps_alignment(
        self, one_page_pdf: Path, tmp_path: Path
    ) -> None:
        # Regression: a skipped region used to shorten the list, so zipping it
        # with the captions shifted every later caption onto the wrong image.
        figures_dir = tmp_path / "figures"
        bad = Region(page=99, coordinates=SQUARE_COORDS)
        good = Region(page=1, coordinates=SQUARE_COORDS)

        paths = extract_figures_from_pdf(one_page_pdf, figures_dir, [bad, good])

        assert paths[0] is None
        assert paths[1] is not None and paths[1].name == "1.png"

    def test_missing_pdf_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_figures_from_pdf(
                tmp_path / "ghost.pdf",
                tmp_path / "figures",
                [Region(page=1, coordinates=SQUARE_COORDS)],
            )


class TestFitzTextFallback:
    """The PyMuPDF (fitz) fallback used when the primary PDF parser fails."""

    def test_extracts_text_layer(self, one_page_pdf: Path) -> None:
        from scholar_lens.src.parser import PDFParser

        text = PDFParser._extract_text_with_fitz(one_page_pdf)
        assert "Hello figure" in text

    async def test_fallback_returns_text_no_figures(self, one_page_pdf: Path) -> None:
        from scholar_lens.src.parser import PDFParser

        parser = PDFParser.__new__(PDFParser)  # no Bedrock/Upstage init
        figures, content = await parser._parse_with_fallback(one_page_pdf)
        assert figures == []  # fallback is text-only
        assert content.text_only
        assert "Hello figure" in content.text

    async def test_fallback_raises_on_empty_text(self, tmp_path: Path) -> None:
        # A PDF with no text layer (blank page) must raise, not return "" — an
        # empty body would otherwise produce an empty summary downstream.
        from scholar_lens.src.parser import ParserError, PDFParser

        blank = tmp_path / "blank.pdf"
        doc = fitz.open()
        doc.new_page(width=612, height=792)
        doc.save(str(blank))
        doc.close()

        parser = PDFParser.__new__(PDFParser)
        with pytest.raises(ParserError, match="no text"):
            await parser._parse_with_fallback(blank)


class TestConstructFigureUrl:
    """arXiv figure srcs are *relative*; resolving them by concatenation
    produced a doubled id segment that 404'd, which silently stripped every
    figure from the model's context and left hallucinated image links in posts.
    """

    @staticmethod
    def _parser(base_url: str):
        from scholar_lens.src.parser import HTMLRichParser

        parser = HTMLRichParser.__new__(HTMLRichParser)
        parser.url = base_url
        return parser

    def test_arxiv_version_prefixed_relative_src(self) -> None:
        # arxiv.org/html/<id> serves srcs like '<id>v1/Media/x.png'; the base has
        # no trailing slash, so RFC 3986 drops the last segment.
        parser = self._parser("https://arxiv.org/html/2502.17416")
        assert (
            parser._construct_figure_url("2502.17416v1/Media/looping.png")
            == "https://arxiv.org/html/2502.17416v1/Media/looping.png"
        )

    def test_arxiv_versioned_base(self) -> None:
        parser = self._parser("https://arxiv.org/html/2502.17416v1")
        assert (
            parser._construct_figure_url("2502.17416v1/Media/looping.png")
            == "https://arxiv.org/html/2502.17416v1/Media/looping.png"
        )

    def test_absolute_src_is_left_alone(self) -> None:
        parser = self._parser("https://arxiv.org/html/2502.17416")
        absolute = "https://cdn.example.com/fig.png"
        assert parser._construct_figure_url(absolute) == absolute

    def test_missing_base_url_raises(self) -> None:
        from scholar_lens.src.parser import ParserError

        parser = self._parser("")
        with pytest.raises(ParserError, match="Base URL"):
            parser._construct_figure_url("x.png")


class TestAttachFigureContext:
    """Markers are tied to figures by source name only, never by position."""

    @staticmethod
    def _figure(path: str, *, caption=None, analysis=None):  # type: ignore[no-untyped-def]
        from scholar_lens.src.parser import Figure

        return Figure(figure_id="1", path=path, caption=caption, analysis=analysis)

    def test_quoted_alt_with_comma_or_brackets_is_kept_whole(self) -> None:
        from scholar_lens.src.parser import attach_figure_context

        fig = self._figure("/w/figures/3.png", caption="c")
        out = attach_figure_context(
            '[Image: alt="x, src=evil.png", src="3.png"] '
            '[Image: alt="[brackets]", src="3.png"]',
            [fig],
            is_pdf_parsed=True,
        )
        assert 'alt="x, src=evil.png", src="3.png"' in out
        assert 'alt="(brackets)", src="3.png"' in out

    def test_caption_used_when_analysis_missing(self) -> None:
        from scholar_lens.src.parser import attach_figure_context

        fig = self._figure("/w/figures/3.png", caption="Figure 1: router")
        out = attach_figure_context(
            '[Image: alt="", src="3.png"]', [fig], is_pdf_parsed=True
        )
        assert (
            out
            == '[Image: alt="Figure 1: router", src="3.png", caption="Figure 1: router"]'
        )

    def test_paper_caption_comes_first_and_analysis_is_marked_approximate(
        self,
    ) -> None:
        # The vision analysis misreads plots (e.g. a colormap's direction); the
        # writer must see the paper's caption as the authority.
        from scholar_lens.src.parser import attach_figure_context

        fig = self._figure(
            "/w/figures/3.png",
            caption="Figure 1: Latency overhead.",
            analysis="A heatmap; darker is slower.",
        )
        out = attach_figure_context(
            '[Image: alt="", src="3.png"]', [fig], is_pdf_parsed=True
        )
        assert out.endswith(
            'caption="Figure 1: Latency overhead. (Figure analysis, approximate: '
            'A heatmap; darker is slower.)"]'
        )

    def test_placeholder_caption_yields_to_the_analysis(self) -> None:
        from scholar_lens.src.parser import attach_figure_context

        fig = self._figure(
            "/w/figures/3.png", caption="Refer to caption", analysis="A bar chart."
        )
        out = attach_figure_context(
            '[Image: alt="", src="3.png"]', [fig], is_pdf_parsed=True
        )
        assert out.endswith('caption="A bar chart."]')

    def test_placeholder_alt_is_not_a_marker_alt(self) -> None:
        from scholar_lens.src.parser import attach_figure_context

        fig = self._figure("/w/figures/3.png", caption="Refer to caption")
        out = attach_figure_context(
            '[Image: alt="Refer to caption", src="3.png"]', [fig], is_pdf_parsed=True
        )
        assert out.startswith('[Image: alt="", src="3.png"')

    def test_markers_without_source_are_not_filled_by_order(self) -> None:
        # Regression: empty-alt markers used to be filled with figures in order,
        # attaching captions to the wrong images.
        from scholar_lens.src.parser import attach_figure_context

        fig = self._figure("/w/figures/7.png", analysis="A bar chart.")
        out = attach_figure_context(
            'a [Image: alt="", src=""] b [Image: alt="x", src="7.png"]',
            [fig],
            is_pdf_parsed=True,
        )
        assert out.startswith("a  b ")
        assert 'src="7.png", caption="A bar chart."' in out

    def test_sourceless_markers_are_counted_and_captions_kept(self) -> None:
        # A table cell holding image markers without a src: no image is
        # invented, the caption text stays, and one warning gives the count.
        from unittest.mock import patch

        from scholar_lens.src.parser import attach_figure_context

        text = '| [Image: alt="", src=""] Table 2: panels | [Image: alt="", src=""] |'
        with patch("scholar_lens.src.parser.logger") as log:
            out = attach_figure_context(text, [], is_pdf_parsed=True)
        assert "[Image:" not in out
        assert "Table 2: panels" in out
        log.warning.assert_called_once()
        assert log.warning.call_args.args[1] == 2

    def test_html_figures_keep_absolute_url(self) -> None:
        from scholar_lens.src.parser import attach_figure_context

        url = "https://arxiv.org/html/2502.17416v1/x1.png"
        out = attach_figure_context(
            '[Image: alt="Fig", src="x1.png"]',
            [self._figure(url, analysis="Loss curve.")],
            is_pdf_parsed=False,
        )
        assert f'src="{url}"' in out


class TestBindFigureSources:
    def test_figures_bound_by_element_id(self) -> None:
        from scholar_lens.src.parser import Figure, _bind_figure_sources
        from scholar_lens.src.utils.helpers import extract_text_from_html

        html = (
            "<figure id='3'><img alt='' data-coord='x'/></figure>"
            "<figure id='11'><img alt='MoE layer' data-coord='y'/></figure>"
        )
        figures = [Figure(figure_id="11", path="/w/figures/1.png")]
        text = extract_text_from_html(_bind_figure_sources(html, figures))
        assert '[Image: alt="MoE layer", src="1.png"]' in text
        assert '[Image: alt="", src=""]' in text  # unmatched figure stays unbound


class TestEmptyDocumentParse:
    def test_empty_response_is_not_content(self) -> None:
        from scholar_lens.src.parser import _has_content

        assert not _has_content({"elements": [], "content": {"html": ""}})
        assert not _has_content({"elements": [{"id": 1}], "content": {"html": " "}})
        assert _has_content({"elements": [{"id": 1}], "content": {"html": "<p>x</p>"}})

    async def test_empty_parse_falls_back_to_pymupdf(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from scholar_lens.src.parser import Content, PDFParser

        parser = PDFParser.__new__(PDFParser)

        async def empty(*_a, **_k):  # type: ignore[no-untyped-def]
            return {"elements": [{"id": 1}], "content": {"html": "<p>tiny</p>"}}

        async def fallback(*_a, **_k):  # type: ignore[no-untyped-def]
            return [], Content(text="fitz text")

        async def no_figures(*_a, **_k):  # type: ignore[no-untyped-def]
            return []

        monkeypatch.setattr(parser, "_get_or_parse_document", empty)
        monkeypatch.setattr(parser, "_extract_figures", no_figures)
        monkeypatch.setattr(parser, "_parse_with_fallback", fallback)
        figures, content = await parser.parse(tmp_path / "p.pdf", tmp_path / "figs")
        assert content.text == "fitz text"

    async def test_empty_response_not_cached(self, tmp_path: Path) -> None:
        from scholar_lens.src.constants import LocalPaths
        from scholar_lens.src.parser import PDFParser

        parser = PDFParser.__new__(PDFParser)

        async def request(_path):  # type: ignore[no-untyped-def]
            return {"elements": [], "content": {"html": ""}}

        parser._request_document_parse = request  # type: ignore[method-assign]
        pdf = tmp_path / "p.pdf"
        await parser._get_or_parse_document(pdf, use_cache=True)
        assert not (tmp_path / LocalPaths.PARSED_FILE.value).exists()


class TestParseCache:
    GOOD = {"elements": [{"id": 1}], "content": {"html": "<p>" + "x " * 200 + "</p>"}}

    @staticmethod
    def _parser(calls: list[str]):  # type: ignore[no-untyped-def]
        from scholar_lens.src.parser import PDFParser

        parser = PDFParser.__new__(PDFParser)

        async def request(_path):  # type: ignore[no-untyped-def]
            calls.append("parse")
            return TestParseCache.GOOD

        parser._request_document_parse = request  # type: ignore[method-assign]
        return parser

    async def test_cache_from_another_pdf_is_reparsed(self, tmp_path: Path) -> None:
        calls: list[str] = []
        parser = self._parser(calls)
        pdf = tmp_path / "p.pdf"
        pdf.write_bytes(b"%PDF-1.7 v1")
        await parser._get_or_parse_document(pdf, use_cache=True)
        await parser._get_or_parse_document(pdf, use_cache=True)
        assert calls == ["parse"]  # same bytes: cached
        pdf.write_bytes(b"%PDF-1.7 v2")  # a newer arXiv version, same folder
        await parser._get_or_parse_document(pdf, use_cache=True)
        assert calls == ["parse", "parse"]

    async def test_cache_without_a_hash_is_kept(self, tmp_path: Path) -> None:
        from scholar_lens.src.constants import LocalPaths

        calls: list[str] = []
        parser = self._parser(calls)
        pdf = tmp_path / "p.pdf"
        pdf.write_bytes(b"%PDF-1.7")
        (tmp_path / LocalPaths.PARSED_FILE.value).write_text(json.dumps(self.GOOD))
        await parser._get_or_parse_document(pdf, use_cache=True)
        assert calls == []


class TestFigureAnalysisFailure:
    @staticmethod
    def _analyser(error: Exception):  # type: ignore[no-untyped-def]
        class Analyser:
            async def ainvoke(self, _inputs):  # type: ignore[no-untyped-def]
                raise error

        return Analyser()

    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # The backoff policy is bound when decorating; swap in no wait.
        monkeypatch.setattr(Figure._analyse.retry, "wait", tenacity.wait_none())

    @staticmethod
    def _image(monkeypatch) -> None:  # type: ignore[no-untyped-def]
        async def data(_path: str) -> str:
            return "aGk="

        monkeypatch.setattr(Figure, "_get_image_data", staticmethod(data))

    def test_exhausted_throttling_keeps_the_figure_unanalysed(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import asyncio

        from botocore.exceptions import ClientError

        self._image(monkeypatch)
        throttled = ClientError({"Error": {"Code": "ThrottlingException"}}, "Converse")
        calls = 0
        error_then_ok = self._analyser(throttled)

        class Counting:
            async def ainvoke(self, inputs):  # type: ignore[no-untyped-def]
                nonlocal calls
                calls += 1
                return await error_then_ok.ainvoke(inputs)

        figure = asyncio.run(
            Figure.from_llm(Counting(), "1", "/p/1.png", "Figure 1: X.")
        )
        assert figure.caption == "Figure 1: X." and figure.analysis is None
        assert calls == MAX_RETRIES  # retried before giving up

    def test_one_throttle_is_retried_not_skipped(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import asyncio

        from botocore.exceptions import ClientError

        self._image(monkeypatch)
        calls = 0

        class Flaky:
            async def ainvoke(self, _inputs):  # type: ignore[no-untyped-def]
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise ClientError(
                        {"Error": {"Code": "ThrottlingException"}}, "Converse"
                    )
                return "A bar chart."

        figure = asyncio.run(Figure.from_llm(Flaky(), "1", "/p/1.png"))
        assert calls == 2 and figure.analysis == "A bar chart."

    def test_model_error_keeps_the_figure_unanalysed(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import asyncio

        from botocore.exceptions import ClientError

        self._image(monkeypatch)
        denied = ClientError(
            {
                "Error": {"Code": "AccessDeniedException"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "Converse",
        )
        figure = asyncio.run(
            Figure.from_llm(self._analyser(denied), "1", "/p/1.png", "Figure 1: X.")
        )
        assert figure.caption == "Figure 1: X." and figure.analysis is None

    def test_token_budget_breach_still_stops_the_run(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import asyncio

        from scholar_lens.src.metrics import TokenBudgetExceeded

        self._image(monkeypatch)
        with pytest.raises(TokenBudgetExceeded):
            asyncio.run(
                Figure.from_llm(
                    self._analyser(TokenBudgetExceeded("over budget")), "1", "/p/1.png"
                )
            )


class TestUpstageHtmlEscaping:
    """Regression: document-parse HTML is not escaped, so a literal "<"
    swallowed the text after it (math, chat-template tokens, prose)."""

    @pytest.mark.parametrize(
        ("html", "kept"),
        [
            (
                "<p>$$p(y_t \\mid y_{<t})\\right)$$ holds.</p>",
                "y_{<t})\\right)$$ holds.",
            ),
            ("<p><start_of_turn>user hi<end_of_turn></p>", "<start_of_turn>user hi"),
            ("<p>Score: <score> out of 5.</p>", "<score> out of 5."),
            (
                "<p>in the format of <problem, original response></p>",
                "<problem, original response>",
            ),
            ("<p>We require x <y and the loss.</p>", "x <y and the loss."),
            # "<" followed by a tag name that is not a tag ("p", "th").
            (
                "<p id='1'>We require $0<p<1$ and the loss stays bounded; q>0 holds.</p>",
                "$0<p<1$ and the loss stays bounded; q>0 holds.",
            ),
            ("<p>if n<th then x</p><p>Next</p>", "n<th then x"),
            ("<p>$|x|<p$ and $0<p\\leq 1$.</p>", "$|x|<p$ and $0<p\\leq 1$."),
        ],
    )
    def test_stray_angle_brackets_survive(self, html: str, kept: str) -> None:
        from scholar_lens.src.parser import escape_stray_angle_brackets
        from scholar_lens.src.utils import extract_text_from_html

        assert kept in extract_text_from_html(escape_stray_angle_brackets(html))

    def test_document_parse_tags_are_kept(self) -> None:
        from scholar_lens.src.parser import escape_stray_angle_brackets

        html = "<table><tr><td colspan='2'>a</td></tr></table><br><figure id='3'>"
        assert escape_stray_angle_brackets(html) == html


class TestPdfFigureCaption:
    @staticmethod
    def _element(category: str, html: str) -> dict:
        return {"category": category, "content": {"html": html}}

    def test_caption_comes_from_the_following_caption_element(self) -> None:
        from scholar_lens.src.parser import _pdf_figure_caption

        elements = [
            self._element(
                "figure", "<figure><img alt='h B=0 Pretrained Weights'></figure>"
            ),
            self._element("caption", "<p>Figure 1: Our reparametrization.</p>"),
        ]
        assert _pdf_figure_caption(elements, 0) == "Figure 1: Our reparametrization."

    def test_ocr_alt_text_is_not_a_caption(self) -> None:
        from scholar_lens.src.parser import _pdf_figure_caption

        elements = [
            self._element("figure", "<figure><img alt='h B=0 Pretrained'></figure>"),
            self._element("paragraph", "<p>We then train the model.</p>"),
        ]
        assert _pdf_figure_caption(elements, 0) == ""

    def test_multi_panel_figure_shares_the_caption_after_its_panels(self) -> None:
        from scholar_lens.src.parser import _pdf_figure_caption

        elements = [
            self._element("figure", "<figure><img alt='(a)'></figure>"),
            self._element("figure", "<figure><img alt='(b)'></figure>"),
            self._element("figure", "<figure><img alt='(c)'></figure>"),
            self._element("caption", "<p>Figure 2: Three panels.</p>"),
        ]
        assert _pdf_figure_caption(elements, 0) == "Figure 2: Three panels."
        assert _pdf_figure_caption(elements, 1) == "Figure 2: Three panels."

    def test_a_figure_does_not_take_the_caption_of_a_later_figure_run(
        self,
    ) -> None:
        # Regression: every adjacent figure was skipped, so a logo at the end
        # of one page (or a long run of figures) inherited the next caption.
        from scholar_lens.src.parser import _pdf_figure_caption

        def figure(page: int) -> dict:
            return {**self._element("figure", "<figure></figure>"), "page": page}

        caption = {
            **self._element("caption", "<p>Figure 3: Results.</p>"),
            "page": 2,
        }
        elements = [figure(1), figure(2), figure(2), caption]
        assert _pdf_figure_caption(elements, 0) == ""
        assert _pdf_figure_caption(elements, 1) == "Figure 3: Results."
        run = [figure(2) for _ in range(5)] + [caption]
        assert _pdf_figure_caption(run, 0) == ""
        assert _pdf_figure_caption(run, 1) == "Figure 3: Results."


class TestFigureMarkerMatching:
    def test_same_file_name_in_two_folders_never_swaps(self) -> None:
        # Regression: markers matched figures by basename, so figs/a/curve.png
        # and figs/b/curve.png both became the second figure.
        from scholar_lens.src.parser import attach_figure_context

        base = "https://arxiv.org/html/2401.00001v1/"
        figures = [
            Figure(figure_id="0", path=base + "figs/a/curve.png", caption="First."),
            Figure(figure_id="1", path=base + "figs/b/curve.png", caption="Second."),
        ]
        text = '[Image: alt="", src="figs/a/curve.png"] and [Image: alt="", src="figs/b/curve.png"]'
        out = attach_figure_context(text, figures, is_pdf_parsed=False)
        assert out.index("First.") < out.index("Second.")
        assert out.count("figs/a/curve.png") == 1


class TestArxivHtmlOnly:
    async def test_conversion_stub_is_a_parse_error(self, monkeypatch) -> None:
        # arXiv answers 200 with "See pages 1-last of main.pdf" when its
        # conversion failed; the pipeline then reads the PDF.
        from scholar_lens.src.parser import ContentParseError, HTMLParser

        urls: list[str] = []

        async def fake_fetch(self, url: str) -> str:  # type: ignore[no-untyped-def]
            urls.append(url)
            return "<div class='ltx_page_main'>See pages 1-last of main.pdf</div>"

        monkeypatch.setattr(HTMLParser, "_fetch_html_from_url", fake_fetch)
        with pytest.raises(ContentParseError, match="almost no text"):
            await HTMLParser()._fetch_html("1412.6980")
        # ar5iv serves placeholder images for figures it lost: never asked.
        assert urls == ["https://arxiv.org/html/1412.6980"]
