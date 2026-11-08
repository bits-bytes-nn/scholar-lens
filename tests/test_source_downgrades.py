"""Source downgrades become run notes, and arXiv 429/5xx are retried through
the shared limiter before a fallback (synthetic responses, no network)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pymupdf
import pytest
import requests
import responses

from scholar_lens.adapters.http.rate_limit import ARXIV_RETRIES, RateLimiter
from scholar_lens.domain.survey.plan import PlanPaper
from scholar_lens.src.paper_source import PaperFetchError, PdfUrlSource
from scholar_lens.src.survey_documents import MAX_PDF_PAGES, PaperDocuments
from scholar_lens.src.web_research import (
    PageContent,
    ResearchCorpus,
    SourceFetchError,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class CountingLimiter(RateLimiter):
    def __init__(self, clock: Clock) -> None:
        super().__init__(1.0, 3.0, name="arxiv", monotonic=clock, sleep=clock.sleep)
        self.acquired = 0
        self.holds: list[float] = []

    def acquire(self) -> None:
        self.acquired += 1
        super().acquire()

    def penalize(self, retry_after_seconds: float) -> None:
        self.holds.append(retry_after_seconds)
        super().penalize(retry_after_seconds)


@pytest.fixture
def limiter() -> CountingLimiter:
    return CountingLimiter(Clock())


# ---- arXiv PDFs go through the limiter ----------------------------------------


@responses.activate
def test_an_arxiv_pdf_waits_for_the_limiter_and_retries_a_server_error(
    tmp_path: Path, minimal_pdf_bytes: bytes, limiter: CountingLimiter
) -> None:
    url = "https://arxiv.org/pdf/2401.00001"
    responses.add(responses.GET, url, status=503)
    responses.add(
        responses.GET,
        url,
        body=minimal_pdf_bytes,
        headers={"Content-Type": "application/pdf"},
    )
    source = PdfUrlSource(url, session=requests.Session(), rate_limiter=limiter)
    assert source.download_pdf(tmp_path).is_file()
    assert limiter.acquired == 2
    assert limiter.holds == [10.0]  # a 5xx without Retry-After


@responses.activate
def test_an_arxiv_pdf_rate_limit_fails_after_the_retries(
    tmp_path: Path, limiter: CountingLimiter
) -> None:
    url = "https://export.arxiv.org/pdf/2401.00001"
    for _ in range(ARXIV_RETRIES + 1):
        responses.add(responses.GET, url, status=429, headers={"Retry-After": "5"})
    source = PdfUrlSource(url, session=requests.Session(), rate_limiter=limiter)
    with pytest.raises(PaperFetchError, match="HTTP 429"):
        source.download_pdf(tmp_path)
    assert limiter.acquired == ARXIV_RETRIES + 1
    assert limiter.holds == [5.0] * (ARXIV_RETRIES + 1)


@responses.activate
def test_another_hosts_pdf_is_not_paced_or_retried(
    tmp_path: Path, limiter: CountingLimiter
) -> None:
    url = "https://papers.example.org/p.pdf"
    responses.add(responses.GET, url, status=503)
    source = PdfUrlSource(url, session=requests.Session(), rate_limiter=limiter)
    with pytest.raises(PaperFetchError):
        source.download_pdf(tmp_path)
    assert (limiter.acquired, len(responses.calls)) == (0, 1)


# ---- arXiv HTML is retried before the PDF fallback --------------------------------


async def test_arxiv_html_retries_a_rate_limit_then_reads_the_page(
    limiter: CountingLimiter,
) -> None:
    from scholar_lens.src.parser import HTMLParser

    answers = iter([httpx.Response(429), httpx.Response(200, text="<p>paper</p>")])
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: next(answers))
    )
    parser = HTMLParser(http_client=client, rate_limiter=limiter)
    assert await parser._fetch_html_from_url("https://arxiv.org/html/1") == (
        "<p>paper</p>"
    )
    assert limiter.acquired == 2 and limiter.holds == [30.0]


async def test_arxiv_html_failure_names_its_status(limiter: CountingLimiter) -> None:
    from scholar_lens.src.parser import ContentParseError, HTMLParser, failure_reason

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(502))
    )
    parser = HTMLParser(http_client=client, rate_limiter=limiter)
    with pytest.raises(ContentParseError) as caught:
        await parser._fetch_html_from_url("https://arxiv.org/html/1")
    assert limiter.acquired == ARXIV_RETRIES + 1
    assert failure_reason(caught.value).endswith("HTTP 502")


# ---- survey papers ---------------------------------------------------------------


def _pdf(path: Path, pages: int) -> Path:
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {n} " + "text of the paper " * 20)
    doc.save(path)
    doc.close()
    return path


class _Researcher:
    """arXiv answers 429 for every HTML rendering."""

    arxiv_fallbacks: dict[str, str] = {}

    def research(self, urls: list[str], **_kwargs: object) -> ResearchCorpus:
        error = httpx.HTTPStatusError(
            "429",
            request=httpx.Request("GET", urls[0]),
            response=httpx.Response(429),
        )
        raise SourceFetchError("Could not reach any of the requested pages") from error


async def test_every_survey_downgrade_is_a_run_note(tmp_path: Path) -> None:
    long_pdf = _pdf(tmp_path / "long.pdf", MAX_PDF_PAGES + 5)

    def pdf_source(url: str) -> SimpleNamespace:
        if "2401.00001" not in url:
            raise PaperFetchError(f"'{url}' is unavailable right now (HTTP 503).")
        return SimpleNamespace(download_pdf=lambda _dir: long_pdf, close=lambda: None)

    openalex = SimpleNamespace(by_doi=lambda doi: [])
    documents = PaperDocuments(
        _Researcher(),  # type: ignore[arg-type]
        openalex,  # type: ignore[arg-type]
        {"2401.00002": "An abstract.", "10.1/x": "Another abstract."},
        pdf_source=pdf_source,  # type: ignore[arg-type]
    )
    papers = [
        PlanPaper.model_construct(id=pid, title=pid, role=role)
        for pid, role in (
            ("2401.00001", "core"),
            ("2401.00002", "supporting"),
            ("10.1/x", "supporting"),
        )
    ]
    kinds = [(await documents.document(p)).kind for p in papers]  # type: ignore[union-attr]
    assert kinds == ["pdf_text", "abstract", "abstract"]
    assert documents.notes() == [
        "Read from the PDF (the arXiv HTML rendering could not be used): "
        "2401.00001 (HTTP 429)",
        "Read from the abstract only (no HTML or PDF text could be had): "
        "2401.00002, 10.1/x",
        f"PDF text cut at {MAX_PDF_PAGES} pages: 2401.00001 "
        f"({MAX_PDF_PAGES + 5} pages)",
    ]


def test_a_paper_without_an_html_rendering_names_why(tmp_path: Path) -> None:
    class NoRendering:
        arxiv_fallbacks = {
            "https://arxiv.org/html/2401.00003": "no HTML rendering (HTTP 404)"
        }

        def research(self, urls: list[str], **_kwargs: object) -> ResearchCorpus:
            return ResearchCorpus(
                pages=[PageContent(url=urls[0], title="", text="abstract")]
            )

    documents = PaperDocuments(NoRendering(), None, {})  # type: ignore[arg-type]
    paper = PlanPaper.model_construct(id="2401.00003", title="T")
    assert documents._arxiv_html(paper, "2401.00003") is None
    assert "2401.00003 (no HTML rendering (HTTP 404))" in documents.notes()[0]


# ---- guides ----------------------------------------------------------------------


def test_a_guides_arxiv_paper_read_from_its_abstract_is_noted() -> None:
    from tests.test_tech_guide import _make_generator

    gen = _make_generator()
    gen.researcher.arxiv_fallbacks = {
        "https://arxiv.org/html/2309.06180": "no HTML rendering (HTTP 404)"
    }
    corpus = ResearchCorpus(
        pages=[
            PageContent(
                url="https://arxiv.org/pdf/2309.06180",
                title="",
                text="abstract",
                final_url="https://arxiv.org/abs/2309.06180",
            ),
            PageContent(url="https://arxiv.org/html/2401.1", title="", text="full"),
        ]
    )
    gen._note_abstract_pages(corpus)
    assert gen.notes == [
        "arXiv papers read from their abstract only (no usable HTML rendering): "
        "https://arxiv.org/abs/2309.06180 (no HTML rendering (HTTP 404))"
    ]
