"""The organisation of a paper whose text names none the model could read."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from scholar_lens.adapters.documents.paper_reader import with_affiliation
from scholar_lens.adapters.scholarly.index import OpenAlexIndex
from scholar_lens.domain.paper.models import Attributes

DOI = "10.48550/arXiv.1"


TITLE = "ReWOO: Decoupling Reasoning from Observations"


def _work(
    *institutions: list[str], doi: str = DOI, title: str = TITLE
) -> dict[str, Any]:
    return {
        "doi": f"https://doi.org/{doi}",
        "title": title,
        "authorships": [
            {"institutions": [{"display_name": name} for name in names]}
            for names in institutions
        ],
    }


@pytest.mark.parametrize(
    "works,expected",
    [
        ([_work(["UC Berkeley"], ["Stanford"], ["UC Berkeley"])], "UC Berkeley"),
        ([_work(["Microsoft"], ["Meta"])], "Microsoft"),  # a tie: the first listed
        ([_work(["Meta (United States)"])], "Meta"),  # OpenAlex's country dropped
        ([_work(["Other"], doi="10.1/other")], None),  # another work's record
        ([_work(["Other"], title="Proximal Policy Optimization")], None),  # misattached
        ([_work([], [])], None),  # the index records none
        ([], None),
    ],
)
def test_the_affiliation_is_the_most_common_institution(
    monkeypatch: pytest.MonkeyPatch, works: list[dict[str, Any]], expected: str | None
) -> None:
    index = OpenAlexIndex()
    monkeypatch.setattr(index, "by_doi", lambda doi: works)
    assert index.affiliation(DOI, TITLE) == expected


def test_a_timeout_is_asked_again_once_then_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = OpenAlexIndex()
    calls: list[str] = []

    def timeout(doi: str) -> list[dict[str, Any]]:
        calls.append(doi)
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(index, "by_doi", timeout)
    with pytest.raises(httpx.ReadTimeout):
        index.affiliation(DOI, TITLE)
    assert len(calls) == 2
    assert index.affiliation("10.1/a|10.1/b", TITLE) is None  # filter syntax: not asked


@pytest.mark.parametrize(
    "affiliation,looked_up,expected,note",
    [
        ("N/A", "Brown University", "Brown University", "taken from OpenAlex"),
        ("N/A", None, "N/A", None),
        ("N/A", RuntimeError("down"), "N/A", "lookup failed"),
        ("Microsoft", "Other", "Microsoft", None),  # the paper's own wins
    ],
)
async def test_the_index_fills_only_a_missing_affiliation(
    affiliation: str, looked_up: Any, expected: str, note: str | None
) -> None:
    def lookup(doi: str, title: str) -> str | None:
        if isinstance(looked_up, Exception):
            raise looked_up
        return looked_up

    notes: list[str] = []
    attributes = Attributes(affiliation=affiliation, category="Language-Models")
    result = await with_affiliation(attributes, DOI, TITLE, lookup, notes)
    assert result.affiliation == expected
    assert (note is None and not notes) or (note is not None and note in notes[0])
