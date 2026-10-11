"""Tests for the paper-source abstraction (arXiv decoupling)."""

from __future__ import annotations

import pytest
import requests
import responses

from scholar_lens.src.paper_source import (
    ArxivSource,
    NotAPdfError,
    PaperSource,
    PaperSourceError,
    PdfUrlSource,
    extract_arxiv_id_from_url,
    is_arxiv_id,
    resolve_paper_source,
)

PDF_URL = "https://example.com/papers/great-paper.pdf"
HTML_URL = "https://example.com/papers/landing"


class TestArxivIdDetection:
    @pytest.mark.parametrize(
        "value",
        [
            "2401.06066",
            "2401.06066v1",
            "2401.06066v12",
            "2312.56789",
            "arXiv:2401.06066",
            "hep-th/9901001",
            "math.GT/0309136v2",
        ],
    )
    def test_ids_recognised(self, value: str) -> None:
        assert is_arxiv_id(value)

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://arxiv.org/abs/hep-th/9901001", "hep-th/9901001"),
            ("https://arxiv.org:443/abs/2401.06066", "2401.06066"),
            ("https://arxiv.org/pdf/2401.06066v2.pdf", "2401.06066v2"),
            ("https://export.arxiv.org/abs/2401.06066", "2401.06066"),
        ],
    )
    def test_ids_extracted_from_arxiv_urls(self, url: str, expected: str) -> None:
        assert extract_arxiv_id_from_url(url) == expected

    def test_legacy_id_source_id_is_path_safe(self) -> None:
        source = resolve_paper_source("hep-th/9901001")
        assert source.source_id == "hep-th_9901001"
        assert source.arxiv_html_id == "hep-th/9901001"

    @pytest.mark.parametrize(
        "value",
        [
            "https://arxiv.org/abs/2401.06066",
            "https://example.com/foo.pdf",
            "not-an-id",
            "",
        ],
    )
    def test_non_modern_ids_rejected(self, value: str) -> None:
        assert not is_arxiv_id(value)

    def test_whitespace_is_stripped(self) -> None:
        assert is_arxiv_id("  2401.06066  ")


class TestExtractArxivIdFromUrl:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://arxiv.org/abs/2401.06066", "2401.06066"),
            ("https://arxiv.org/pdf/2401.06066v2.pdf", "2401.06066v2"),
            ("https://arxiv.org/html/2401.06066", "2401.06066"),
            ("http://www.arxiv.org/abs/2312.56789v3", "2312.56789v3"),
            ("https://ar5iv.org/abs/2106.09685", "2106.09685"),
            ("https://ar5iv.labs.arxiv.org/html/2106.09685", "2106.09685"),
            ("https://www.alphaxiv.org/abs/2106.09685v2", "2106.09685v2"),
            ("https://www.alphaxiv.org/overview/2106.09685", "2106.09685"),
        ],
    )
    def test_extracts_from_arxiv_urls(self, url: str, expected: str) -> None:
        assert extract_arxiv_id_from_url(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/2401.06066",
            "https://openreview.net/pdf?id=abc",
            "https://arxiv.org/list/cs.AI/recent",  # no id in path
        ],
    )
    def test_returns_none_for_non_arxiv(self, url: str) -> None:
        assert extract_arxiv_id_from_url(url) is None

    def test_arxiv_org_subdomain_not_spoofable(self) -> None:
        # A host merely containing "arxiv.org" as a substring must not match.
        assert (
            extract_arxiv_id_from_url("https://arxiv.org.evil.com/abs/2401.06066")
            is None
        )


class TestResolvePaperSource:
    def test_bare_arxiv_id_to_arxiv_source(self) -> None:
        src = resolve_paper_source("2401.06066")
        assert isinstance(src, ArxivSource)
        assert isinstance(src, PaperSource)

    def test_arxiv_url_to_arxiv_source(self) -> None:
        src = resolve_paper_source("https://arxiv.org/pdf/2401.06066")
        assert isinstance(src, ArxivSource)
        assert src.arxiv_html_id == "2401.06066"

    @pytest.mark.parametrize(
        "typed", ["arxiv.org/abs/2106.09685", "www.alphaxiv.org/abs/2106.09685"]
    )
    def test_url_without_scheme(self, typed: str) -> None:
        src = resolve_paper_source(typed)
        assert isinstance(src, ArxivSource) and src.arxiv_html_id == "2106.09685"

    def test_pdf_url_without_scheme(self) -> None:
        src = resolve_paper_source("openreview.net/pdf?id=nZeVKeeFYf9")
        assert isinstance(src, PdfUrlSource)
        assert str(src.pdf_url).startswith("https://openreview.net/pdf")

    def test_generic_url_to_pdf_url_source(self) -> None:
        src = resolve_paper_source(PDF_URL)
        assert isinstance(src, PdfUrlSource)
        assert src.arxiv_html_id is None

    @pytest.mark.parametrize("bad", ["not a url or id", "ftp://x/y", "42", ""])
    def test_unresolvable_raises(self, bad: str) -> None:
        with pytest.raises(PaperSourceError):
            resolve_paper_source(bad)


class TestPaperSourceContract:
    def test_abc_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            PaperSource()  # type: ignore[abstract]


class TestArxivSource:
    def test_source_id_replaces_dots(self) -> None:
        assert ArxivSource("2401.06066v1").source_id == "2401_06066v1"

    def test_arxiv_html_id_preserves_dots(self) -> None:
        assert ArxivSource("2401.06066").arxiv_html_id == "2401.06066"


class TestPdfUrlSourceSlug:
    def test_source_id_uses_stem_plus_hash(self) -> None:
        src = PdfUrlSource(PDF_URL, session=requests.Session())
        assert src.source_id.startswith("great-paper-")
        assert len(src.source_id.rsplit("-", 1)[1]) == 8  # sha1[:8]

    def test_distinct_urls_distinct_ids(self) -> None:
        a = PdfUrlSource("https://a.com/x/p.pdf")
        b = PdfUrlSource("https://b.com/y/p.pdf")
        assert a.source_id != b.source_id  # same stem, different hash

    def test_openreview_id_names_the_paper(self) -> None:
        src = PdfUrlSource("https://openreview.net/pdf?id=nZeVKeeFYf9")
        assert src.source_id.startswith("nzevkeefyf9-")
        assert src._fallback_title() == "Nzevkeefyf9"

    def test_no_stem_falls_back_to_paper_hash(self) -> None:
        src = PdfUrlSource("https://example.com/")
        assert src.source_id.startswith("paper-")

    def test_fallback_title_humanises_stem(self) -> None:
        src = PdfUrlSource("https://x.com/v1/attention-is-all-you-need.pdf")
        assert src._fallback_title() == "Attention Is All You Need"

    def test_fetch_metadata_uses_fallback_title(self) -> None:
        src = PdfUrlSource("https://x.com/deep-residual-learning.pdf")
        md = src.fetch_metadata()
        assert md.title == "Deep Residual Learning"
        assert str(md.pdf_url).startswith("https://x.com/")


class TestPdfUrlSourceDownload:
    @responses.activate
    def test_rejects_html_content_type(self, tmp_papers_dir) -> None:
        # A .pdf URL that actually serves HTML must be rejected, not trusted.
        responses.add(
            responses.HEAD,
            PDF_URL,
            headers={"Content-Type": "text/html; charset=utf-8"},
            status=200,
        )
        src = PdfUrlSource(PDF_URL, session=requests.Session())
        with pytest.raises(NotAPdfError):
            src.download_pdf(tmp_papers_dir)

    @responses.activate
    def test_failed_head_with_html_error_page_falls_back_to_magic_probe(
        self, tmp_papers_dir
    ) -> None:
        # Regression: HEAD 405 + an HTML error page used to reject a URL whose
        # GET serves a real PDF.
        responses.add(
            responses.HEAD,
            PDF_URL,
            headers={"Content-Type": "text/html"},
            status=405,
        )
        responses.add(
            responses.GET,
            PDF_URL,
            body=b"%PDF-1.4 rest",
            headers={"Content-Type": "application/pdf"},
            status=200,
        )
        src = PdfUrlSource(PDF_URL, session=requests.Session())
        src._assert_is_pdf()  # no NotAPdfError

    @responses.activate
    def test_redirect_to_internal_host_is_blocked(self, tmp_papers_dir) -> None:
        # A 302 to an internal target (SSRF via redirect) must be rejected: the
        # download path re-validates every hop against the SSRF guard.
        responses.add(
            responses.HEAD,
            PDF_URL,
            status=302,
            headers={"Location": "http://169.254.169.254/latest/meta-data/"},
        )
        src = PdfUrlSource(PDF_URL, session=requests.Session())
        with pytest.raises(PaperSourceError):
            src.download_pdf(tmp_papers_dir)

    @responses.activate
    def test_accepts_pdf_content_type_and_downloads(
        self, tmp_papers_dir, minimal_pdf_bytes
    ) -> None:
        responses.add(
            responses.HEAD,
            PDF_URL,
            headers={"Content-Type": "application/pdf"},
            status=200,
        )
        responses.add(
            responses.GET,
            PDF_URL,
            body=minimal_pdf_bytes,
            content_type="application/pdf",
            status=200,
        )
        src = PdfUrlSource(PDF_URL, session=requests.Session())
        path = src.download_pdf(tmp_papers_dir)
        assert path.exists()
        assert path.read_bytes().startswith(b"%PDF-")

    @responses.activate
    def test_magic_byte_probe_when_content_type_missing(
        self, tmp_papers_dir, minimal_pdf_bytes
    ) -> None:
        # No content-type on HEAD -> falls back to ranged magic-byte GET probe.
        responses.add(responses.HEAD, PDF_URL, status=200)
        responses.add(responses.GET, PDF_URL, body=minimal_pdf_bytes, status=200)
        src = PdfUrlSource(PDF_URL, session=requests.Session())
        path = src.download_pdf(tmp_papers_dir)
        assert path.exists()

    @responses.activate
    def test_rejects_non_pdf_body_without_content_type(self, tmp_papers_dir) -> None:
        responses.add(responses.HEAD, PDF_URL, status=200)
        responses.add(responses.GET, PDF_URL, body=b"<html>nope</html>", status=200)
        src = PdfUrlSource(PDF_URL, session=requests.Session())
        with pytest.raises(NotAPdfError):
            src.download_pdf(tmp_papers_dir)

    @responses.activate
    def test_download_truncated_non_pdf_is_rejected(self, tmp_papers_dir) -> None:
        # HEAD claims pdf but the body is not a real PDF -> post-download guard.
        responses.add(
            responses.HEAD, PDF_URL, headers={"Content-Type": "application/pdf"}
        )
        responses.add(responses.GET, PDF_URL, body=b"junk-bytes", status=200)
        src = PdfUrlSource(PDF_URL, session=requests.Session())
        with pytest.raises(NotAPdfError):
            src.download_pdf(tmp_papers_dir)

    @responses.activate
    def test_declared_content_length_over_cap_rejected(self, tmp_papers_dir) -> None:
        # A Content-Length above the 100 MB cap is rejected up front (no download).
        from scholar_lens.src.paper_source import _MAX_PDF_BYTES

        responses.add(
            responses.HEAD, PDF_URL, headers={"Content-Type": "application/pdf"}
        )
        responses.add(
            responses.GET,
            PDF_URL,
            body=b"%PDF-1.4 short",
            headers={"Content-Length": str(_MAX_PDF_BYTES + 1)},
            status=200,
        )
        src = PdfUrlSource(PDF_URL, session=requests.Session())
        with pytest.raises(PaperSourceError, match="exceeds"):
            src.download_pdf(tmp_papers_dir)

    @responses.activate
    def test_body_streamed_past_cap_rejected_and_partial_cleaned_up(
        self, tmp_papers_dir, monkeypatch
    ) -> None:
        # A body that streams past the cap (no/lying Content-Length) is aborted
        # mid-download AND the partial file is unlinked (no half-PDF left behind).
        import scholar_lens.src.paper_source as ps

        monkeypatch.setattr(ps, "_MAX_PDF_BYTES", 10)  # tiny cap for the test
        responses.add(
            responses.HEAD, PDF_URL, headers={"Content-Type": "application/pdf"}
        )
        responses.add(responses.GET, PDF_URL, body=b"%PDF-" + b"x" * 1000, status=200)
        src = PdfUrlSource(PDF_URL, session=requests.Session())
        with pytest.raises(PaperSourceError, match="limit"):
            src.download_pdf(tmp_papers_dir)
        pdf_path = tmp_papers_dir / src.source_id / f"{src.source_id}.pdf"
        assert not pdf_path.exists()  # partial file cleaned up

    @responses.activate
    def test_too_many_redirects_rejected(self, tmp_papers_dir) -> None:
        # A redirect chain longer than _MAX_REDIRECTS is rejected (not followed
        # forever) — a DoS / loop guard. All hops stay on public hosts.
        from scholar_lens.src.paper_source import _MAX_REDIRECTS

        responses.add(
            responses.HEAD, PDF_URL, headers={"Content-Type": "application/pdf"}
        )
        # Chain: PDF_URL -> /r0 -> /r1 -> ... , always one more hop than allowed.
        for i in range(_MAX_REDIRECTS + 2):
            here = PDF_URL if i == 0 else f"https://example.com/r{i - 1}"
            responses.add(
                responses.GET,
                here,
                status=302,
                headers={"Location": f"https://example.com/r{i}"},
            )
        src = PdfUrlSource(PDF_URL, session=requests.Session())
        with pytest.raises(PaperSourceError, match="[Tt]oo many redirects"):
            src.download_pdf(tmp_papers_dir)

    @responses.activate
    def test_public_redirect_is_followed(
        self, tmp_papers_dir, minimal_pdf_bytes
    ) -> None:
        # A 302 to another PUBLIC location is followed and the PDF downloads (the
        # happy path the per-hop re-validation must not break). Uses a resolvable
        # host so the SSRF guard's real DNS lookup succeeds.
        final = "https://example.com/cdn/paper.pdf"
        responses.add(
            responses.HEAD, PDF_URL, headers={"Content-Type": "application/pdf"}
        )
        responses.add(responses.GET, PDF_URL, status=302, headers={"Location": final})
        responses.add(
            responses.GET,
            final,
            body=minimal_pdf_bytes,
            content_type="application/pdf",
            status=200,
        )
        src = PdfUrlSource(PDF_URL, session=requests.Session())
        path = src.download_pdf(tmp_papers_dir)
        assert path.exists()
        assert path.read_bytes().startswith(b"%PDF-")

    @responses.activate
    def test_redirect_with_empty_location_rejected(self, tmp_papers_dir) -> None:
        responses.add(
            responses.HEAD, PDF_URL, headers={"Content-Type": "application/pdf"}
        )
        responses.add(responses.GET, PDF_URL, status=302, headers={"Location": ""})
        src = PdfUrlSource(PDF_URL, session=requests.Session())
        with pytest.raises(PaperSourceError):
            src.download_pdf(tmp_papers_dir)


class TestPdfUrlSourceSsrfPinning:
    def test_default_session_mounts_pinned_adapter(self) -> None:
        # The arbitrary-PDF-URL path is the primary SSRF vector, so a source built
        # without an injected session must pin every hop to an SSRF-validated IP
        # (closing the DNS-rebinding TOCTOU window a plain requests.Session leaves).
        from scholar_lens.src.pinned_transport import PinnedRequestsAdapter

        src = PdfUrlSource(PDF_URL)
        for scheme in ("https://", "http://"):
            assert isinstance(src._session.adapters[scheme], PinnedRequestsAdapter)

    def test_injected_session_is_left_untouched(self) -> None:
        # An explicitly injected session (e.g. the `responses` transport stub used
        # in the hermetic download tests) must NOT be re-mounted.
        from scholar_lens.src.pinned_transport import PinnedRequestsAdapter

        session = requests.Session()
        src = PdfUrlSource(PDF_URL, session=session)
        assert src._session is session
        assert not isinstance(src._session.adapters["https://"], PinnedRequestsAdapter)


class TestArxivIdRules:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1412.6980", "1412.6980"),  # 4-digit sequence before 2015
            ("2401.0606", None),  # 4 digits after 2015-01 is not an arXiv id
            ("1412.69801", None),  # 5 digits before 2015-01 neither
            ("arXiv:2106.09685v2", "2106.09685v2"),
        ],
    )
    def test_sequence_length_follows_the_date(self, value, expected) -> None:  # type: ignore[no-untyped-def]
        from scholar_lens.src.constants import normalize_arxiv_id

        assert normalize_arxiv_id(value) == expected

    def test_reference_urls_are_reduced_to_ids(self) -> None:
        from scholar_lens.src.content_extractor import Citation

        cite = Citation(
            authors="Hu", title="LoRA", arxiv_id="https://arxiv.org/pdf/2106.09685.pdf"
        )
        assert cite.arxiv_id == "2106.09685"


@pytest.mark.parametrize(
    "url", ["https://xn--arxiv-.org/abs/2106.09685", "https://[::1/x.pdf"]
)
def test_malformed_url_is_a_source_error(url: str) -> None:
    with pytest.raises(PaperSourceError, match="not a valid URL"):
        resolve_paper_source(url)
