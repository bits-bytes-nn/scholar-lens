"""Tests for the citation-summarization redesign (no real network or Bedrock).

Covers the three new/changed pieces:
  * ``RateLimiter`` — token-bucket pacing, Retry-After penalty, arg validation.
  * ``citation_metadata`` — provider parsing, abstract-preferring chain, the
    never-raise contract, and JATS stripping.
  * ``CitationSummarizer`` — abstract-first routing, per-run cache/dedup, and
    failure exclusion (chains and resolvers are faked, so Bedrock is untouched).
"""

from __future__ import annotations

from typing import Any

import boto3
import httpx
import pytest

from scholar_lens.src.arxiv_handler import ArxivMetadata
from scholar_lens.src.citation_metadata import (
    ChainedMetadataResolver,
    CrossrefProvider,
    MetadataProvider,
    ReferenceMetadata,
    SemanticScholarProvider,
    _strip_jats,
)
from scholar_lens.src.citation_summarizer import CitationSummarizer
from scholar_lens.src.constants import LanguageModelId
from scholar_lens.src.rate_limiter import RateLimiter

# --------------------------------------------------------------------------- #
# RateLimiter
# --------------------------------------------------------------------------- #


class _FakeClock:
    """Deterministic monotonic clock + sleep that advances it (no real waiting)."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _limiter_with_clock(clock: _FakeClock, **kwargs: Any) -> RateLimiter:
    limiter = RateLimiter(**kwargs)
    limiter._monotonic = clock.monotonic  # type: ignore[assignment]
    limiter._last_refill = clock.monotonic()
    return limiter


def test_rate_limiter_rejects_nonpositive_args() -> None:
    with pytest.raises(ValueError):
        RateLimiter(rate=0, per=1.0)
    with pytest.raises(ValueError):
        RateLimiter(rate=1.0, per=0)


def test_rate_limiter_allows_burst_then_paces(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr("scholar_lens.src.rate_limiter.time.sleep", clock.sleep)
    # 1 token per 2s, capacity 1 → first acquire is free, second must wait ~2s.
    limiter = _limiter_with_clock(clock, rate=1.0, per=2.0, name="t")

    start = clock.now
    limiter.acquire()  # immediate (full bucket)
    assert clock.now == start
    limiter.acquire()  # must pace
    assert clock.now - start == pytest.approx(2.0, abs=1e-6)


def test_rate_limiter_penalize_blocks_until_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr("scholar_lens.src.rate_limiter.time.sleep", clock.sleep)
    limiter = _limiter_with_clock(clock, rate=10.0, per=1.0, name="t")

    start = clock.now
    limiter.penalize(5.0)
    limiter.acquire()  # blocked by the penalty even though tokens were plentiful
    assert clock.now - start >= 5.0


def test_rate_limiter_penalize_ignores_nonpositive() -> None:
    limiter = RateLimiter(rate=1.0, per=1.0)
    limiter.penalize(0)
    limiter.penalize(-3)
    assert limiter._blocked_until == 0.0


def test_rate_limiter_no_unbounded_burst_after_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: a long penalty must never be credited as accrued tokens beyond
    # the bucket capacity when it clears. Drive the refill clock across a 30s
    # penalty and assert tokens are capped at capacity (== rate), not 30*rate.
    clock = _FakeClock()
    monkeypatch.setattr("scholar_lens.src.rate_limiter.time.sleep", clock.sleep)
    limiter = _limiter_with_clock(clock, rate=5.0, per=1.0, name="t")

    limiter.penalize(30.0)
    # Simulate the acquire() poll loop ticking through the penalty window.
    for _ in range(35):
        with limiter._lock:
            limiter._time_until_available(clock.monotonic())
        clock.sleep(1.0)
    # Penalty long cleared; tokens must not exceed capacity.
    assert limiter._tokens <= limiter._capacity
    assert limiter._capacity == 5.0


# --------------------------------------------------------------------------- #
# citation_metadata
# --------------------------------------------------------------------------- #


def test_strip_jats_removes_tags_and_collapses_whitespace() -> None:
    assert (
        _strip_jats("<jats:p>Hello   <jats:i>world</jats:i></jats:p>") == "Hello world"
    )
    assert _strip_jats(None) is None
    assert _strip_jats("   ") is None


def test_strip_jats_preserves_inequality_prose() -> None:
    # Regression (data-1): a greedy <[^>]+> strip treated "x < y and a >" as a tag
    # and deleted the clause. A real parser keeps inequality signs in prose.
    text = "<jats:p>We prove that for all x < y and a > b, the bound holds.</jats:p>"
    out = _strip_jats(text)
    assert out == "We prove that for all x < y and a > b, the bound holds."


def test_strip_jats_decodes_entities() -> None:
    # Escaped markup must be decoded to real characters, not leaked as entities.
    assert _strip_jats("<jats:p>R&amp;D rose &gt; 5%</jats:p>") == "R&D rose > 5%"


def test_reference_metadata_helpers() -> None:
    md = ReferenceMetadata(title="T", authors=["A B", "C D"], abstract="x")
    assert md.author_str == "A B, C D"
    assert md.has_abstract is True
    assert ReferenceMetadata(title="T", authors=[], abstract="  ").has_abstract is False


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    """Patch ``httpx.get`` in citation_metadata to return a fixed JSON payload."""

    def fake_get(
        url: str, *, params: Any = None, timeout: Any = None
    ) -> httpx.Response:
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr("scholar_lens.src.citation_metadata.httpx.get", fake_get)


def test_crossref_provider_parses_and_strips_abstract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_httpx(
        monkeypatch,
        {
            "message": {
                "items": [
                    {
                        "title": ["Attention Is All You Need"],
                        "author": [
                            {"given": "Ashish", "family": "Vaswani"},
                            {"given": "Noam", "family": "Shazeer"},
                        ],
                        "abstract": "<jats:p>We propose the Transformer.</jats:p>",
                        "URL": "https://doi.org/x",
                        "type": "journal-article",
                    }
                ]
            }
        },
    )
    md = CrossrefProvider().lookup("attention")
    assert md is not None
    assert md.title == "Attention Is All You Need"
    assert md.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert md.abstract == "We propose the Transformer."
    assert md.url == "https://doi.org/x"  # primary type → URL trusted


def test_crossref_drops_url_for_book_chapter_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A book chapter that merely cites the paper must NOT contribute its URL
    # (mis-attribution guard); title/abstract are still usable.
    _patch_httpx(
        monkeypatch,
        {
            "message": {
                "items": [
                    {
                        "title": ["Attention Is All You Need"],
                        "author": [{"given": "Some", "family": "Editor"}],
                        "abstract": "<jats:p>A chapter discussing transformers.</jats:p>",
                        "URL": "https://doi.org/10.1007/978-3-031-84300-6_13",
                        "type": "book-chapter",
                    }
                ]
            }
        },
    )
    md = CrossrefProvider().lookup("attention is all you need")
    assert md is not None
    assert md.title == "Attention Is All You Need"
    assert md.url is None  # book-chapter URL dropped


def test_crossref_provider_returns_none_on_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_httpx(monkeypatch, {"message": {"items": []}})
    assert CrossrefProvider().lookup("nothing") is None


def test_crossref_provider_never_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(url: str, *, params: Any = None, timeout: Any = None) -> httpx.Response:
        raise httpx.ConnectError("down")

    monkeypatch.setattr("scholar_lens.src.citation_metadata.httpx.get", boom)
    assert CrossrefProvider().lookup("x") is None


def test_semantic_scholar_provider_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_httpx(
        monkeypatch,
        {
            "data": [
                {
                    "title": "BERT",
                    "abstract": "We introduce BERT.",
                    "authors": [{"name": "Jacob Devlin"}, {"name": ""}],
                    "url": "https://s2.org/bert",
                }
            ]
        },
    )
    md = SemanticScholarProvider().lookup("bert")
    assert md is not None
    assert md.authors == ["Jacob Devlin"]
    assert md.abstract == "We introduce BERT."


class _StubProvider(MetadataProvider):
    def __init__(
        self, result: ReferenceMetadata | None, *, raises: bool = False
    ) -> None:
        self._result = result
        self._raises = raises
        self.calls = 0

    def lookup(self, title: str) -> ReferenceMetadata | None:
        self.calls += 1
        if self._raises:
            raise RuntimeError("provider exploded")
        return self._result


_TITLE = "Attention Is All You Need"


def test_chained_resolver_prefers_result_with_abstract() -> None:
    no_abstract = ReferenceMetadata(title=_TITLE, authors=["x"], abstract=None)
    with_abstract = ReferenceMetadata(title=_TITLE, authors=["y"], abstract="real")
    first = _StubProvider(no_abstract)
    second = _StubProvider(with_abstract)
    resolver = ChainedMetadataResolver([first, second])

    result = resolver.resolve(_TITLE)
    assert result is with_abstract
    assert first.calls == 1 and second.calls == 1


def test_chained_resolver_stops_at_complete_match() -> None:
    complete = ReferenceMetadata(
        title=_TITLE, authors=["x"], abstract="abs", url="https://u"
    )
    later = _StubProvider(complete)
    resolver = ChainedMetadataResolver([_StubProvider(complete), later])
    assert resolver.resolve(_TITLE) is complete
    assert later.calls == 0  # no second (rate-limited) provider call


def test_chained_resolver_discards_other_papers() -> None:
    # Regression: the top fuzzy hit for "Attention Is All You Need" can be a
    # different paper; it must never be returned (and summarised) as this one.
    wrong = ReferenceMetadata(
        title="Attention Is All You Need In Speech Separation",
        authors=["x"],
        abstract="abs",
        url="https://wrong",
    )
    assert ChainedMetadataResolver([_StubProvider(wrong)]).resolve(_TITLE) is None


def test_chained_resolver_falls_back_to_first_hit_without_abstract() -> None:
    only = ReferenceMetadata(title=_TITLE, authors=["x"], abstract=None)
    resolver = ChainedMetadataResolver([_StubProvider(only), _StubProvider(None)])
    assert resolver.resolve(_TITLE) is only


def test_chained_resolver_swallows_provider_exceptions() -> None:
    good = ReferenceMetadata(title=_TITLE, authors=["x"], abstract="z")
    resolver = ChainedMetadataResolver(
        [_StubProvider(None, raises=True), _StubProvider(good)]
    )
    assert resolver.resolve(_TITLE) is good


# --------------------------------------------------------------------------- #
# CitationSummarizer
# --------------------------------------------------------------------------- #


class _FakeChain:
    """Stand-in for a LangChain runnable: records calls, returns canned text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def ainvoke(self, _inputs: dict[str, Any]) -> str:
        self.calls += 1
        return self.text


class _FakeResolver:
    def __init__(self, result: ReferenceMetadata | None) -> None:
        self.result = result
        self.calls = 0

    def resolve(self, title: str) -> ReferenceMetadata | None:
        self.calls += 1
        return self.result


@pytest.fixture
def summarizer(monkeypatch: pytest.MonkeyPatch) -> CitationSummarizer:
    from scholar_lens.src.utils import factories

    # Inference-profile lookup would call Bedrock; use the base model id.
    monkeypatch.setattr(
        factories,
        "_resolve_inference_profile",
        lambda session, model_id, region: model_id.value,
    )
    session = boto3.Session(region_name="us-east-1")
    summ = CitationSummarizer(
        citation_summarizing_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        citation_analysis_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        boto_session=session,
    )
    # fit_text would call Bedrock CountTokens over the network; pass text through.
    summ.llm_factory.fit_text = (  # type: ignore[method-assign]
        lambda model_id, text, **kw: text
    )
    return summ


@pytest.mark.asyncio
async def test_summarize_resolves_from_metadata_abstract(
    summarizer: CitationSummarizer,
) -> None:
    md = ReferenceMetadata(
        title="Some Paper", authors=["A B"], abstract="An abstract.", url="https://x"
    )
    summarizer.metadata_resolver = _FakeResolver(md)  # type: ignore[assignment]
    summarizer.citation_summarizer = _FakeChain("A concise summary.")  # type: ignore[assignment]

    # Query the matching title so the resolved URL is trusted and linked.
    out = await summarizer.summarize(["Some Paper"], "original context")
    assert len(out) == 1
    assert "[Some Paper](https://x)" in out[0]  # link attached on a confident match
    assert "A concise summary." in out[0]
    assert "Authors: A B" in out[0]


@pytest.mark.asyncio
async def test_summarize_skips_abstract_of_a_different_paper(
    summarizer: CitationSummarizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Resolver returned a DIFFERENT paper: its abstract must not be summarised
    # under this citation; the summariser falls through to the title path.
    md = ReferenceMetadata(
        title="A Totally Different Paper",
        authors=["X"],
        abstract="abs",
        url="https://wrong",
    )
    summarizer.metadata_resolver = _FakeResolver(md)  # type: ignore[assignment]
    abstract_chain = _FakeChain("summary of the wrong paper")
    summarizer.citation_summarizer = abstract_chain  # type: ignore[assignment]
    monkeypatch.setattr(summarizer.arxiv_handler, "search_by_title", lambda *a: None)

    class _Analyzer:
        async def ainvoke(self, _inputs: dict[str, Any]) -> dict[str, str]:
            return {"analysis": "RoBERTa background."}

    summarizer.citation_analyzer = _Analyzer()  # type: ignore[assignment]

    out = await summarizer.summarize(["RoBERTa Pretraining Approach"], "ctx")
    assert abstract_chain.calls == 0
    assert out == ["Title: [RoBERTa Pretraining Approach]\nRoBERTa background."]


@pytest.mark.parametrize(
    ("queried", "resolved", "expected"),
    [
        (
            "LoRA: Low-Rank Adaptation of LLMs",
            "LoRA: Low-Rank Adaptation of LLMs",
            True,
        ),
        ("attention is all you need", "Attention Is All You Need.", True),
        ("RoBERTa Pretraining", "Unrelated Journal Article on Plants", False),
        # Regressions: near-miss titles of different papers.
        (
            "LoRA: Low-Rank Adaptation of Large Language Models",
            "LoRA+: Efficient Low Rank Adaptation of Large Models",
            False,
        ),
        (
            "Attention Is All You Need",
            "Attention Is All You Need In Speech Separation",
            False,
        ),
        (
            "Deep Residual Learning for Image Recognition",
            "Deep Residual Learning: A Survey",
            False,
        ),
        (None, "x", False),
        ("x", None, False),
    ],
)
def test_title_matches_gate(
    queried: str | None, resolved: str | None, expected: bool
) -> None:
    assert CitationSummarizer._title_matches(queried, resolved) is expected


@pytest.mark.asyncio
async def test_summarize_caches_and_dedups_identifiers(
    summarizer: CitationSummarizer,
) -> None:
    md = ReferenceMetadata(title="dup", authors=["A"], abstract="abs", url=None)
    resolver = _FakeResolver(md)
    chain = _FakeChain("summary")
    summarizer.metadata_resolver = resolver  # type: ignore[assignment]
    summarizer.citation_summarizer = chain  # type: ignore[assignment]

    # Same identifier three times across two calls → resolved/summarised once.
    await summarizer.summarize(["dup", "dup"], "ctx")
    await summarizer.summarize(["dup"], "ctx")
    assert resolver.calls == 1
    assert chain.calls == 1


@pytest.mark.asyncio
async def test_summarize_single_flight_across_concurrent_calls(
    summarizer: CitationSummarizer,
) -> None:
    # Two concurrent summarize() calls (as the per-paragraph enrich loop does)
    # for the SAME identifier must share one resolution — not double the work.
    import asyncio as _asyncio

    md = ReferenceMetadata(title="dup", authors=["A"], abstract="abs", url=None)

    class _SlowResolver:
        def __init__(self) -> None:
            self.calls = 0
            self._gate = _asyncio.Event()

        def resolve(self, title: str) -> ReferenceMetadata | None:
            # Runs in a worker thread (asyncio.to_thread); block briefly so the
            # second concurrent caller has to coalesce onto the in-flight task.
            self.calls += 1
            import time as _time

            _time.sleep(0.05)
            return md

    resolver = _SlowResolver()
    chain = _FakeChain("summary")
    summarizer.metadata_resolver = resolver  # type: ignore[assignment]
    summarizer.citation_summarizer = chain  # type: ignore[assignment]

    out_a, out_b = await _asyncio.gather(
        summarizer.summarize(["dup"], "ctx"),
        summarizer.summarize(["dup"], "ctx"),
    )
    assert resolver.calls == 1
    assert chain.calls == 1
    assert out_a == out_b and len(out_a) == 1


@pytest.mark.asyncio
async def test_summarize_excludes_failure_summaries(
    summarizer: CitationSummarizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Drive the explicit-arXiv path (fully stubbable, no network) and have the
    # summary chain emit the sentinel failure string — it must be dropped.
    meta = ArxivMetadata(
        arxiv_id="2301.00002",
        title="P",
        authors=["A"],
        published="2023-01-01T00:00:00Z",  # type: ignore[arg-type]
        updated="2023-01-01T00:00:00Z",  # type: ignore[arg-type]
        abstract="abs",
    )
    monkeypatch.setattr(summarizer.arxiv_handler, "fetch_metadata", lambda _id: meta)
    summarizer.citation_summarizer = _FakeChain(  # type: ignore[assignment]
        CitationSummarizer.FAILURE_STRING
    )
    out = await summarizer.summarize(["arXiv:2301.00002"], "ctx")
    assert out == []


@pytest.mark.asyncio
async def test_arxiv_id_routes_to_arxiv_and_uses_abstract(
    summarizer: CitationSummarizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = ArxivMetadata(
        arxiv_id="2301.00001",
        title="Arxiv Paper",
        authors=["Author One"],
        published="2023-01-01T00:00:00Z",  # type: ignore[arg-type]
        updated="2023-01-01T00:00:00Z",  # type: ignore[arg-type]
        abstract="The arxiv abstract.",
    )
    monkeypatch.setattr(summarizer.arxiv_handler, "fetch_metadata", lambda _id: meta)

    chain = _FakeChain("arxiv summary")
    summarizer.citation_summarizer = chain  # type: ignore[assignment]
    # Should NOT consult the metadata resolver for an explicit arXiv id.
    resolver = _FakeResolver(None)
    summarizer.metadata_resolver = resolver  # type: ignore[assignment]

    out = await summarizer.summarize(["arXiv:2301.00001"], "ctx")
    assert len(out) == 1
    assert "Arxiv Paper" in out[0]
    assert "arxiv summary" in out[0]
    assert resolver.calls == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("arXiv:2301.00001", "2301.00001"),
        ("2301.00001v2", "2301.00001v2"),
        ("hep-th/9901001", "hep-th/9901001"),
        ("arXiv:math.GT/0309136v2", "math.GT/0309136v2"),
        ("Attention Is All You Need", None),
        ("N/A", None),
    ],
)
def test_normalize_arxiv_id(value: str, expected: str | None) -> None:
    from scholar_lens.src.constants import normalize_arxiv_id

    assert normalize_arxiv_id(value) == expected


@pytest.mark.asyncio
async def test_arxiv_not_found_yields_no_summary(
    summarizer: CitationSummarizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: a placeholder "Could not generate summary." used to reach the
    # writer prompts as if it were a citation summary.
    from scholar_lens.src.arxiv_handler import ArxivNotFoundError

    def missing(_id: str) -> None:
        raise ArxivNotFoundError("gone")

    monkeypatch.setattr(summarizer.arxiv_handler, "fetch_metadata", missing)
    assert await summarizer.summarize(["2301.99999"], "ctx") == []


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
    async def test_attributes_do_not_write_keyword_vocabulary(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:  # type: ignore[no-untyped-def]
        # New keywords are no longer appended to assets/keywords.txt or S3; tag
        # normalisation is deterministic downstream.
        from scholar_lens.src.content_extractor import ContentExtractor

        extractor = ContentExtractor.__new__(ContentExtractor)
        extractor.existing_keywords = set()
        extractor.keywords_path = tmp_path / "keywords.txt"
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
        assert not extractor.keywords_path.exists()


class TestTitleGateSiblings:
    @pytest.mark.parametrize(
        "queried,resolved",
        [
            (
                "LoRA: Low-Rank Adaptation of Large Language Models",
                "LoRA+: Efficient Low Rank Adaptation of Large Models",
            ),
            ("GPT-4 Technical Report", "GPT-4o Technical Report"),
            ("Qwen2 Technical Report", "Qwen2.5 Technical Report"),
            (
                "Gemma 2: Improving Open Language Models",
                "Gemma 3: Improving Open Language Models",
            ),
        ],
    )
    def test_sibling_versions_do_not_match(self, queried: str, resolved: str) -> None:
        from scholar_lens.src.citation_metadata import titles_match

        assert titles_match(queried, resolved) is False

    def test_same_title_with_punctuation_matches(self) -> None:
        from scholar_lens.src.citation_metadata import titles_match

        assert titles_match("Attention is All you Need", "Attention Is All You Need.")
