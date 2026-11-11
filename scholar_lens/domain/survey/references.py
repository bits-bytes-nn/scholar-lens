"""The user's reference material of a survey: what each reference is, the
papers it cites, and the excerpts the prompts read; and the candidate pool
the plan stage gathers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from scholar_lens.domain.research import PageContent, ResearchCorpus
from scholar_lens.domain.survey.plan import (
    Candidate,
    PlanReference,
    ReferenceKind,
    cited_paper_ids,
    paper_id_of,
)


@dataclass
class ReadReference:
    """A reference the user passed and its page (None when it could not be
    read)."""

    url: str
    kind: ReferenceKind
    page: PageContent | None

    def cited_ids(self) -> list[str]:
        """Plan ids of the paper the reference is (a paper link) and of the
        papers its content links to or names by arXiv id."""
        own = paper_id_of(self.url)
        cited = (
            cited_paper_ids(self.page.content_links, self.page.text, own=own)
            if self.page
            else []
        )
        return [own, *cited] if own else cited

    def plan_reference(self) -> PlanReference:
        return PlanReference(
            url=self.url, title=self.page.title if self.page else "", kind=self.kind
        )


def reference_excerpts(pages: Sequence[PageContent], query: str, max_chars: int) -> str:
    """Each reference page's parts most relevant to ``query``, an equal share
    of ``max_chars`` each, labelled "(reference material)" and defanged
    (:meth:`ResearchCorpus.relevant_text`); "none" without pages."""
    if not pages:
        return "none"
    pages = [replace(p, user_reference=True, authority="reference") for p in pages]
    return ResearchCorpus(pages=pages).relevant_text(
        query,
        max_chars=max_chars,
        labelled=True,
        prefer=[p.url for p in pages],
    )


def unread_note(references: Sequence[ReadReference]) -> list[str]:
    """The plan note on references that could not be read, if any."""
    unread = [r.url for r in references if r.page is None]
    if not unread:
        return []
    return [
        "Reference material that could not be read (a paper link among it is "
        "still a candidate): " + ", ".join(unread)
    ]


@dataclass
class CandidatePool:
    """Candidates in rank order with the citation edges among them."""

    candidates: list[Candidate]
    edges: list[tuple[str, str]]
    sources: list[str]
    notes: list[str]
