"""ContentExtractor citation parsing, dedup keys and attribute parsing."""

from __future__ import annotations

import re
from typing import Any

import pytest

from scholar_lens.src.constants import LanguageModelId


class TestCitationFieldCoercion:
    """One malformed field must not drop the whole citation."""

    @pytest.mark.parametrize(
        ("arxiv_id", "expected"),
        [("N/A", None), ("arXiv:hep-th/9802150", "hep-th/9802150"), (None, None)],
    )
    def test_arxiv_id_coerced(self, arxiv_id: str | None, expected: str | None) -> None:
        from scholar_lens.src.content_extractor import Citation

        cit = Citation(authors="A", title="T", arxiv_id=arxiv_id)
        assert cit.arxiv_id == expected

    @pytest.mark.parametrize(
        ("year", "expected"), [("2017a", 2017), ("2020, July", 2020), ("n.d.", None)]
    )
    def test_year_coerced(self, year: str, expected: int | None) -> None:
        from scholar_lens.src.content_extractor import Citation

        assert Citation(authors="A", title="T", year=year).year == expected

    def test_parse_citation_tuple_with_letter_year(self) -> None:
        from scholar_lens.src.content_extractor import ContentExtractor

        cit = ContentExtractor._parse_citation("('Hu et al.', '2021a', 'LoRA', 'N/A')")
        assert cit is not None and cit.year == 2021 and cit.arxiv_id is None

    def test_non_latin_titles_keep_distinct_keys(self) -> None:
        from scholar_lens.src.content_extractor import Citation, ContentExtractor

        a = Citation(authors="김", title="대규모 언어 모델", year=2024)
        b = Citation(authors="김", title="강화 학습 개요", year=2024)
        assert ContentExtractor._dedup_key(a) != ContentExtractor._dedup_key(b)


class TestCitationDedupKey:
    def test_distinct_same_title_works_kept_apart(self) -> None:
        # Regression (data-2): two different references sharing a title (different
        # authors/year, no arXiv id) must NOT collapse to one dedup key.
        from scholar_lens.src.content_extractor import Citation, ContentExtractor

        a = Citation(authors="Smith", year=2020, title="Deep Learning", arxiv_id=None)
        b = Citation(authors="Jones", year=2023, title="Deep Learning", arxiv_id=None)
        assert ContentExtractor._dedup_key(a) != ContentExtractor._dedup_key(b)

    def test_author_formatting_variant_collapses(self) -> None:
        from scholar_lens.src.content_extractor import Citation, ContentExtractor

        a = Citation(authors="Smith", year=2020, title="Deep Learning", arxiv_id=None)
        c = Citation(
            authors="Smith, J., et al.",
            year=2020,
            title="Deep Learning",
            arxiv_id=None,
        )
        assert ContentExtractor._dedup_key(a) == ContentExtractor._dedup_key(c)

    def test_arxiv_id_dominates_identity(self) -> None:
        from scholar_lens.src.content_extractor import Citation, ContentExtractor

        d = Citation(authors="X", year=None, title="T", arxiv_id="2401.06066")
        e = Citation(authors="Y", year=2021, title="Other", arxiv_id="2401.06066")
        assert ContentExtractor._dedup_key(d) == ContentExtractor._dedup_key(e)


class TestContentExtractorSurface:
    async def test_attributes_are_parsed_from_the_extractor_output(self) -> None:
        from scholar_lens.src.content_extractor import ContentExtractor

        extractor = ContentExtractor.__new__(ContentExtractor)
        extractor.existing_keywords = set()
        extractor._attributes_model_id = LanguageModelId.CLAUDE_V5_5_HAIKU
        extractor.llm_factory = type(
            "F", (), {"fit_text": staticmethod(lambda m, t, **k: t)}
        )()

        seen: dict[str, Any] = {}

        class _Chain:
            async def ainvoke(self, inputs: dict[str, Any]) -> dict[str, str]:
                seen.update(inputs)
                return {
                    "keywords": "LoRA, PEFT",
                    "category": "Language Models",
                    "affiliation": "Microsoft",
                    "controlled_tags": "Fine-Tuning , Efficient-Training",
                    "seo_title": "",
                }

        extractor.attributes_extractor = _Chain()
        extractor._taxonomy = {"controlled_tags": "Fine-Tuning, Efficient-Training"}
        attrs = await extractor.extract_attributes("paper text")
        assert attrs.keywords == ["LoRA", "PEFT"]
        assert attrs.controlled_tags == ["Fine-Tuning", "Efficient-Training"]
        assert attrs.seo_title is None
        assert seen["controlled_tags"] == "Fine-Tuning, Efficient-Training"
        assert "category_options" not in seen  # prompt default applies


def _record(arxiv_id: str, title: str, authors: list[str], year: int) -> Any:
    from datetime import datetime

    from scholar_lens.src.arxiv_handler import ArxivMetadata

    published = datetime(year, 6, 1)
    return ArxivMetadata(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        published=published,
        updated=published,
    )


# What arXiv holds under the ids the blog's reviews had to fix by hand.
ARXIV_RECORDS = {
    "1706.03762": _record(
        "1706.03762v7",
        "Attention Is All You Need",
        ["Ashish Vaswani", "Noam Shazeer"],
        2017,
    ),
    "1912.03356": _record(
        "1912.03356v1",
        "Vehicular Networking for Connected Autonomous Driving",
        ["Someone Else"],
        2019,
    ),
    "2403.05530": _record(
        "2403.05530v5",
        "Gemini 1.5: Unlocking multimodal understanding across millions of "
        "tokens of context",
        ["Gemini Team", "Petko Georgiev"],
        2024,
    ),
    "2112.00861": _record(
        "2112.00861v3",
        "A General Language Assistant as a Laboratory for Alignment",
        ["Amanda Askell", "Yuntao Bai"],
        2021,
    ),
}


class FakeArxiv:
    """``fetch_metadata_batch`` over fixed records (absent ids are unknown to
    arXiv), recording each batch; ``error`` makes every call fail."""

    def __init__(self, records: dict[str, Any], error: Exception | None = None):
        self.records = records
        self.error = error
        self.batches: list[list[str]] = []

    def fetch_metadata_batch(self, arxiv_ids: list[str]) -> dict[str, Any]:
        self.batches.append(list(arxiv_ids))
        if self.error is not None:
            raise self.error
        asked = {re.sub(r"v\d+$", "", i) for i in arxiv_ids}
        return {i: r for i, r in self.records.items() if i in asked}


def _cite(authors: str, year: int | None, title: str, arxiv_id: str | None) -> Any:
    from scholar_lens.src.content_extractor import Citation

    return Citation(authors=authors, year=year, title=title, arxiv_id=arxiv_id)


class TestVerifyArxivIds:
    """Every extracted arXiv id is checked against arXiv's record of it."""

    def _verify(self, citations: list[Any], arxiv: FakeArxiv) -> list[Any]:
        from scholar_lens.src.content_extractor import verify_arxiv_ids

        return verify_arxiv_ids(citations, arxiv)  # type: ignore[arg-type]

    def test_an_id_naming_the_cited_work_is_kept(self) -> None:
        cite = _cite(
            "Vaswani, A., Shazeer, N.",
            2017,
            "Attention is all you need",
            "arXiv:1706.03762v5",
        )
        (kept,) = self._verify([cite], FakeArxiv(ARXIV_RECORDS))
        assert kept.arxiv_id == "1706.03762v5"

    def test_an_id_of_another_work_is_dropped_and_the_citation_kept(self) -> None:
        jax = _cite(
            "Bradbury, J.",
            2018,
            "JAX: composable transformations of Python+NumPy programs",
            "1912.03356",
        )
        (kept,) = self._verify([jax], FakeArxiv(ARXIV_RECORDS))
        assert kept.arxiv_id is None
        assert kept.title.startswith("JAX") and kept.authors == "Bradbury, J."

    def test_a_matching_title_by_other_authors_is_dropped(self) -> None:
        # Saunders et al. cited under Askell et al.'s id with Askell's title.
        cite = _cite(
            "Saunders, W.",
            2022,
            "A General Language Assistant as a Laboratory for Alignment",
            "2112.00861",
        )
        (kept,) = self._verify([cite], FakeArxiv(ARXIV_RECORDS))
        assert kept.arxiv_id is None

    def test_a_year_before_the_preprint_is_dropped(self) -> None:
        cite = _cite("Vaswani", 2014, "Attention Is All You Need", "1706.03762")
        (kept,) = self._verify([cite], FakeArxiv(ARXIV_RECORDS))
        assert kept.arxiv_id is None

    def test_a_venue_year_after_the_preprint_is_kept(self) -> None:
        cite = _cite("Vaswani", 2018, "Attention Is All You Need", "1706.03762")
        (kept,) = self._verify([cite], FakeArxiv(ARXIV_RECORDS))
        assert kept.arxiv_id == "1706.03762"

    def test_an_id_arxiv_does_not_know_is_dropped(self) -> None:
        cite = _cite("Smith", 2024, "Some Paper", "2405.99999")
        (kept,) = self._verify([cite], FakeArxiv(ARXIV_RECORDS))
        assert kept.arxiv_id is None

    @pytest.mark.parametrize("made_up", ["2024.01234", "2021.01000"])
    def test_a_malformed_id_is_dropped_before_any_lookup(self, made_up: str) -> None:
        arxiv = FakeArxiv(ARXIV_RECORDS)
        cite = _cite("Smith", 2024, "Some Benchmark", made_up)
        assert cite.arxiv_id is None
        assert self._verify([cite], arxiv) == [cite]
        assert arxiv.batches == []

    def test_one_id_reused_for_several_works_stays_only_on_its_own(self) -> None:
        gemini = _cite(
            "Gemini Team",
            2024,
            "Gemini 1.5: Unlocking multimodal understanding across millions of "
            "tokens of context",
            "2403.05530",
        )
        benchmarks = [
            _cite(
                "Hendrycks, D.",
                2021,
                "Measuring Massive Multitask Language " "Understanding",
                "2403.05530",
            ),
            _cite(
                "Rein, D.",
                2023,
                "GPQA: A Graduate-Level Google-Proof Q&A " "Benchmark",
                "2403.05530",
            ),
        ]
        arxiv = FakeArxiv(ARXIV_RECORDS)
        verified = self._verify([benchmarks[0], gemini, benchmarks[1]], arxiv)
        assert [c.arxiv_id for c in verified] == [None, "2403.05530", None]
        assert len(arxiv.batches) == 1  # one batched lookup for the page

    def test_unreachable_arxiv_drops_every_id_and_keeps_the_citations(self) -> None:
        from scholar_lens.src.arxiv_handler import ArxivPaperError

        cites = [
            _cite("Vaswani", 2017, "Attention Is All You Need", "1706.03762"),
            _cite("Hu", 2021, "LoRA", None),
        ]
        arxiv = FakeArxiv(ARXIV_RECORDS, error=ArxivPaperError("HTTP 503"))
        verified = self._verify(cites, arxiv)
        assert [c.arxiv_id for c in verified] == [None, None]
        assert [c.title for c in verified] == [c.title for c in cites]

    async def test_extraction_verifies_before_dedup(self) -> None:
        # Deduplication keys on the id: a wrong id shared by two references
        # must be dropped before it merges them into one.
        from scholar_lens.src.content_extractor import ContentExtractor

        page = "\n".join(
            [
                "('Gemini Team', 2024, 'Gemini 1.5: Unlocking multimodal "
                "understanding across millions of tokens of context', "
                "'arXiv:2403.05530')",
                "('Hendrycks, D.', 2021, 'Measuring Massive Multitask Language "
                "Understanding', 'arXiv:2403.05530')",
            ]
        )

        class _Chain:
            async def ainvoke(self, _: dict[str, Any]) -> dict[str, str]:
                return {"citations": page, "has_more": "n"}

        extractor = ContentExtractor.__new__(ContentExtractor)
        extractor._citation_model_id = LanguageModelId.CLAUDE_V5_5_HAIKU
        extractor.llm_factory = type(
            "F", (), {"fit_text": staticmethod(lambda m, t, **k: t)}
        )()
        extractor.citation_extractor = _Chain()
        extractor.arxiv_handler = FakeArxiv(ARXIV_RECORDS)  # type: ignore[assignment]
        citations = await extractor.extract_citations("paper text")
        assert [(c.title[:9], c.arxiv_id) for c in citations] == [
            ("Gemini 1.", "2403.05530"),
            ("Measuring", None),
        ]
