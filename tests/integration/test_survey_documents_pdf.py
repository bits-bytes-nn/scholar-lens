"""One paper's unusable PDF URL leaves that paper to its abstract; it never
fails the survey."""

from __future__ import annotations

import pytest

from scholar_lens.adapters.scholarly.survey_documents import PaperDocuments
from scholar_lens.domain.survey.plan import PlanPaper


@pytest.mark.parametrize("url", ["ftp://example.org/p.pdf", "http://10.0.0.5/p.pdf"])
def test_an_unusable_pdf_url_is_one_missing_pdf(url: str) -> None:
    documents = PaperDocuments(None, None, {})  # type: ignore[arg-type]
    paper = PlanPaper.model_construct(id="10.1/x", title="X")
    assert documents._pdf(paper, url) is None
