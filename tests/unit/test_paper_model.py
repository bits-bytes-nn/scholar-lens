"""The paper model's own rules (built through its constructor, so its
validators run)."""

from __future__ import annotations

from datetime import datetime

import pytest

from scholar_lens.domain.paper.models import Attributes, Content, Paper


def _paper(**kwargs: object) -> Paper:
    return Paper(
        arxiv_id="2401.06066",
        title="A Paper",
        published=datetime(2024, 1, 11),
        pdf_url="https://arxiv.org/pdf/2401.06066",
        content=Content(text="The body."),
        attributes=Attributes(affiliation="Lab", category="ML", keywords=["x"]),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("given", [["https://github.com/acme/tool"], None])
def test_repository_urls_are_kept_and_are_link_sources(given: list[str] | None) -> None:
    paper = _paper(repo_urls=given)
    assert [str(u).rstrip("/") for u in paper.repo_urls] == (given or [])
    if given:
        assert paper.link_sources().allows("https://github.com/acme/tool")
