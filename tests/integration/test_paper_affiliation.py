"""The organisation of a paper whose text names none the model could read."""

from __future__ import annotations

from typing import Any

import pytest

from scholar_lens.adapters.documents.paper_reader import with_affiliation
from scholar_lens.adapters.scholarly.index import OpenAlexIndex
from scholar_lens.domain.paper.models import Attributes


def _work(*institutions: list[str]) -> dict[str, Any]:
    return {
        "authorships": [
            {"institutions": [{"display_name": name} for name in names]}
            for names in institutions
        ]
    }


@pytest.mark.parametrize(
    "works,expected",
    [
        ([_work(["UC Berkeley"], ["Stanford"], ["UC Berkeley"])], "UC Berkeley"),
        ([_work(["Microsoft"], ["Meta"])], "Microsoft"),  # a tie: the first listed
        ([_work([], [])], None),  # the index records none
        ([], None),
    ],
)
def test_the_affiliation_is_the_most_common_institution(
    monkeypatch: pytest.MonkeyPatch, works: list[dict[str, Any]], expected: str | None
) -> None:
    index = OpenAlexIndex()
    monkeypatch.setattr(index, "by_doi", lambda doi: works)
    assert index.affiliation("10.48550/arXiv.1") == expected


def test_a_failed_lookup_is_no_affiliation(monkeypatch: pytest.MonkeyPatch) -> None:
    index = OpenAlexIndex()

    def fail(doi: str) -> list[dict[str, Any]]:
        raise RuntimeError("down")

    monkeypatch.setattr(index, "by_doi", fail)
    assert index.affiliation("10.48550/arXiv.1") is None


@pytest.mark.parametrize(
    "affiliation,looked_up,expected",
    [
        ("N/A", "Brown University", "Brown University"),
        ("N/A", None, "N/A"),
        ("Microsoft", "Other", "Microsoft"),  # the paper's own wins
    ],
)
async def test_the_index_fills_only_a_missing_affiliation(
    affiliation: str, looked_up: str | None, expected: str
) -> None:
    calls: list[str] = []

    def lookup(doi: str) -> str | None:
        calls.append(doi)
        return looked_up

    attributes = Attributes(affiliation=affiliation, category="Language-Models")
    result = await with_affiliation(attributes, "10.48550/arXiv.1", lookup)
    assert result.affiliation == expected
    assert bool(calls) is (affiliation == "N/A")
