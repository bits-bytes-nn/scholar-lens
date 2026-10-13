"""ContentExtractor citation parsing, dedup keys and attribute parsing."""

from __future__ import annotations

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
