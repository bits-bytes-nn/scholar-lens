"""Venue names: hosts, institutes and series are not venues; the major
venues get their short names (defects of the 2026-11-03 security survey)."""

from __future__ import annotations

import pytest

from scholar_lens.adapters.scholarly.index import openalex_venue
from scholar_lens.domain.venues import venue_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Zenodo (CERN European Organization for Nuclear Research)", None),
        ("CISPA Helmholtz Center for Information Security", None),
        ("Lecture notes in computer science", None),
        ("arXiv.org", None),
        ("neural information processing systems", "NeurIPS"),
        (
            "Proceedings of the conference - Association for Computational "
            "Linguistics. Meeting",
            "ACL",
        ),
        (
            "Findings of the Association for Computational Linguistics: EMNLP 2024",
            "Findings of ACL",
        ),
        ("Trans. Mach. Learn. Res.", "TMLR"),
        ("Network and Distributed System Security Symposium", "NDSS"),
        (
            "journal of artificial intelligence research",
            "Journal Of Artificial Intelligence Research",
        ),
        ("Nature Machine Intelligence", "Nature Machine Intelligence"),
        ("", None),
        (None, None),
    ],
)
def test_venue_name(raw: str | None, expected: str | None) -> None:
    assert venue_name(raw) == expected


def test_openalex_venue_skips_repositories_and_book_series() -> None:
    locations = [
        {"source": {"display_name": "Zenodo", "type": "repository"}},
        {
            "source": {
                "display_name": "Lecture notes in computer science",
                "type": "book series",
            }
        },
        {
            "source": {
                "display_name": "neural information processing systems",
                "type": "conference",
            }
        },
    ]
    assert openalex_venue(locations) == "NeurIPS"
    assert openalex_venue(locations[:2]) is None


def test_reference_list_shows_venues_by_their_names() -> None:
    from scholar_lens.domain.survey.plan import PlanPaper
    from scholar_lens.domain.survey.post import reference_list

    def paper(pid: str, venue: str | None) -> PlanPaper:
        return PlanPaper(
            id=pid,
            title="T",
            authors="A et al.",
            year=2024,
            url=f"https://arxiv.org/abs/{pid}",
            lineage="l",
            role="core",
            reason="r",
            venue=venue,
        )

    out = reference_list(
        [
            paper("2401.00001", "Zenodo (CERN)"),
            paper("2401.00002", "arXiv.org"),
            paper("2401.00003", "neural information processing systems"),
        ]
    )
    lines = out.splitlines()
    assert lines[-3].endswith("[T](https://arxiv.org/abs/2401.00001).")
    assert lines[-2].endswith(" arXiv.") and lines[-1].endswith(" NeurIPS.")
