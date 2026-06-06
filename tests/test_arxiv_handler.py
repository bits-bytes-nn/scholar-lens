"""Tests for the pure pieces of the arXiv handler (pydantic model + helpers).

The real arXiv API is never contacted. The only outbound call in scope is the
DOI-validation ``requests.get`` to doi.org inside ``ArxivMetadata`` validation,
which is mocked with ``responses`` (or proven not to fire at all).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from scholar_lens.src.arxiv_handler import ArxivHandler, ArxivMetadata

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


class TestNormalizeTitle:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Attention Is All You Need", "attentionisallyouneed"),
            ("Self-Attention", "selfattention"),
            ("A.B.C", "abc"),
            ("  Mixed  -  Title.", "mixedtitle"),
        ],
    )
    def test_normalization_removes_whitespace_dots_dashes(
        self, raw: str, expected: str
    ) -> None:
        assert ArxivHandler._normalize_title(raw) == expected


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

    def test_no_match_returns_none(self) -> None:
        handler = self._handler(
            [_FakePaper("Something Entirely Different", "9999.99999")]
        )
        assert handler.search_by_title("Attention Is All You Need") is None

    def test_low_similarity_match_rejected(self, monkeypatch) -> None:
        # Force the normalized titles to collide while the raw titles differ a
        # lot, so the similarity gate (not the exact gate) does the rejecting.
        monkeypatch.setattr(
            ArxivHandler, "_normalize_title", staticmethod(lambda t: "x")
        )
        handler = self._handler(
            [_FakePaper("A completely unrelated paper title here", "0000.00000")]
        )
        assert handler.search_by_title("Attention Is All You Need") is None
