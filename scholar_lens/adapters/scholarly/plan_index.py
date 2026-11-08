"""Bibliographic records and abstracts of a survey plan's papers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from scholar_lens.adapters.scholarly.index import (
    DataCiteIndex,
    OpenAlexIndex,
    datacite_record,
    openalex_record,
)
from scholar_lens.domain.survey.plan import SourceRecord, unversioned_arxiv_id


class PaperIndex:
    """Bibliographic records and abstracts of plan papers: arXiv's own
    metadata (DataCite) for arXiv ids, OpenAlex's for DOIs. A DOI paper that
    is also on arXiv takes arXiv's abstract (OpenAlex's abstracts of arXiv
    records are unreliable) and keeps its arXiv id. Blocking."""

    def __init__(self, datacite: DataCiteIndex, openalex: OpenAlexIndex) -> None:
        self.datacite = datacite
        self.openalex = openalex

    def records(self, paper_ids: Sequence[str]) -> dict[str, SourceRecord]:
        found: dict[str, SourceRecord] = {}
        arxiv_of = {i: i for i in paper_ids if unversioned_arxiv_id(i)}
        for doi in (i for i in paper_ids if i not in arxiv_of):
            works = self.openalex.by_doi(doi)
            if records := [r for w in works if (r := openalex_record(w, "plan"))]:
                found[doi] = records[0]
                if records[0].arxiv_id:
                    arxiv_of[doi] = records[0].arxiv_id
        if not arxiv_of:
            return found
        arxiv = {
            record.arxiv_id: record
            for item in self.datacite.arxiv_papers(sorted(set(arxiv_of.values())))
            if (record := datacite_record(item, None)) and record.arxiv_id
        }
        for paper_id, arxiv_id in arxiv_of.items():
            if arxiv_id not in arxiv:
                continue
            if paper_id in found:
                found[paper_id] = replace(
                    found[paper_id], abstract=arxiv[arxiv_id].abstract
                )
            else:
                found[paper_id] = arxiv[arxiv_id]
        return found
