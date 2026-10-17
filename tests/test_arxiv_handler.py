"""Tests for the pure pieces of the arXiv handler (pydantic model + helpers).

The real arXiv API is never contacted. The only outbound call in scope is the
DOI-validation ``requests.get`` to doi.org inside ``ArxivMetadata`` validation,
which is mocked with ``responses`` (or proven not to fire at all).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from scholar_lens.src import arxiv_handler
from scholar_lens.src.arxiv_handler import ArxivHandler, ArxivMetadata


class _NoWaitLimiter:
    def acquire(self) -> None:
        pass

    def penalize(self, seconds: float) -> None:
        pass


@pytest.fixture(autouse=True)
def _no_arxiv_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The process-wide limiter allows one request per 3 s; tests never wait."""
    monkeypatch.setattr(arxiv_handler, "_ARXIV_RATE_LIMITER", _NoWaitLimiter())


PUBLISHED = datetime(2024, 1, 1, 12, 0, 0)
UPDATED = datetime(2024, 1, 2, 12, 0, 0)


def _base_kwargs(**overrides: object) -> dict[str, object]:
    # The default ``arxiv_id`` is a NON-arXiv slug so the ``mode="before"`` DOI
    # validator never matches the arXiv-id pattern and never makes a network
    # call. Tests that exercise the DOI lookup override ``arxiv_id`` with a real
    # id AND activate ``responses`` to mock doi.org.
    kwargs: dict[str, object] = {
        "arxiv_id": "test-paper-001",
        "title": "A Title",
        "authors": ["Alice"],
        "published": PUBLISHED,
        "updated": UPDATED,
        "doi": None,
    }
    kwargs.update(overrides)
    return kwargs


class TestCleanAbstract:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("  hello   world  ", "hello world"),
            ("line one\n\nline two", "line one line two"),
            ("a\t\tb\n  c", "a b c"),
            ("single", "single"),
        ],
    )
    def test_whitespace_collapsed(self, raw: str, expected: str) -> None:
        meta = ArxivMetadata(**_base_kwargs(abstract=raw))
        assert meta.abstract == expected

    def test_none_abstract_preserved(self) -> None:
        meta = ArxivMetadata(**_base_kwargs(abstract=None))
        assert meta.abstract is None


class TestSetUpdatedIfMissing:
    def test_explicit_updated_is_kept(self) -> None:
        meta = ArxivMetadata(**_base_kwargs())
        assert meta.updated == UPDATED

    def test_missing_updated_falls_back_to_published(self) -> None:
        meta = ArxivMetadata(**_base_kwargs(updated=None))
        assert meta.updated == PUBLISHED

    def test_both_none_raises(self) -> None:
        with pytest.raises(ValueError):
            ArxivMetadata(**_base_kwargs(published=None, updated=None))


class TestGenerateAndValidateDoi:
    # The arXiv DOI is deterministic (10.48550/arXiv.<id>); it is synthesised
    # offline with no doi.org round-trip. None of these touch the network.
    def test_real_id_generates_doi(self) -> None:
        meta = ArxivMetadata(**_base_kwargs(arxiv_id="2401.06066"))
        assert meta.doi == "10.48550/arXiv.2401.06066"

    def test_versioned_id_strips_version_for_doi(self) -> None:
        meta = ArxivMetadata(**_base_kwargs(arxiv_id="2401.06066v3"))
        assert meta.doi == "10.48550/arXiv.2401.06066"

    def test_slug_style_id_generates_no_doi(self) -> None:
        meta = ArxivMetadata(**_base_kwargs(arxiv_id="foo-bar-123"))
        assert meta.doi is None

    def test_explicit_doi_is_preserved(self) -> None:
        meta = ArxivMetadata(
            **_base_kwargs(arxiv_id="foo-bar-123", doi="10.1000/explicit")
        )
        assert meta.doi == "10.1000/explicit"


class _FakePaper:
    def __init__(self, title: str, short_id: str) -> None:
        self.title = title
        self._short_id = short_id

    def get_short_id(self) -> str:
        return self._short_id


class TestSearchByTitleSimilarity:
    def _handler(self, returned: list[_FakePaper]):
        from unittest.mock import MagicMock

        handler = ArxivHandler.__new__(ArxivHandler)
        handler.client = MagicMock()
        handler.client.results = MagicMock(return_value=iter(returned))
        return handler

    def test_exact_match_accepted(self) -> None:
        handler = self._handler([_FakePaper("Attention Is All You Need", "1706.03762")])
        assert handler.search_by_title("Attention Is All You Need") == "1706.03762"

    def test_exact_title_beats_a_survey_listed_first(self) -> None:
        title = "Training Compute-Optimal Large Language Models"
        handler = self._handler(
            [
                _FakePaper(f"{title}: A Survey", "2501.00001"),
                _FakePaper(title, "2203.15556"),
            ]
        )
        assert handler.search_by_title(title) == "2203.15556"

    def test_closest_passing_title_without_an_exact_one(self) -> None:
        title = "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
        handler = self._handler(
            [
                _FakePaper(f"{title} Revisited", "2501.00002"),
                # British spelling: not the same string, the closest match.
                _FakePaper(title.replace("Modeling", "Modelling"), "2312.00752"),
            ]
        )
        assert handler.search_by_title(title) == "2312.00752"

    def test_no_match_returns_none(self) -> None:
        handler = self._handler(
            [_FakePaper("Something Entirely Different", "9999.99999")]
        )
        assert handler.search_by_title("Attention Is All You Need") is None

    def test_sibling_version_rejected(self) -> None:
        handler = self._handler([_FakePaper("Qwen2.5 Technical Report", "2412.15115")])
        assert handler.search_by_title("Qwen2 Technical Report") is None

    def test_hyphen_retry_accepts_the_spaced_title(self) -> None:
        from unittest.mock import MagicMock

        handler = ArxivHandler.__new__(ArxivHandler)
        handler.client = MagicMock()
        handler.client.results = MagicMock(
            side_effect=[
                iter([]),
                iter([_FakePaper("Few Shot Learners in Vision", "2001.00001")]),
            ]
        )
        assert handler.search_by_title("Few-Shot Learners in Vision") == "2001.00001"
        queries = [c.args[0].query for c in handler.client.results.call_args_list]
        assert queries == [
            'ti:"Few-Shot Learners in Vision"',
            'ti:"Few Shot Learners in Vision"',
        ]


class TestDownloadFolder:
    def test_unversioned_id_downloads_into_its_own_folder(
        self, monkeypatch, tmp_path
    ) -> None:
        # Regression: "2106.09685" downloaded into "2106_09685v2/" while the
        # rest of the paper lived in "2106_09685/".
        from types import SimpleNamespace

        from scholar_lens.src.paper_source import ArxivSource

        handler = ArxivHandler()
        paper = SimpleNamespace(
            get_short_id=lambda: "2106.09685v2", pdf_url="https://arxiv.org/pdf/x"
        )
        monkeypatch.setattr(handler, "_fetch_single_paper", lambda _id: paper)
        monkeypatch.setattr(
            ArxivHandler, "_download", staticmethod(lambda url, path: path.touch())
        )
        path = ArxivSource("2106.09685", handler=handler).download_pdf(tmp_path)
        assert path == tmp_path / "2106_09685" / "2106_09685.pdf"
