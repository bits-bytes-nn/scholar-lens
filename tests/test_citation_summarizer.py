"""CitationSummarizer: abstract-first routing, per-run caching and failure
exclusion (chains, resolvers and arXiv are faked; Bedrock is untouched)."""

from __future__ import annotations

from typing import Any

import boto3
import pytest

from scholar_lens.src.arxiv_handler import ArxivMetadata
from scholar_lens.src.citation_metadata import ReferenceMetadata
from scholar_lens.src.citation_summarizer import CitationSummarizer
from scholar_lens.src.constants import LanguageModelId


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


@pytest.mark.asyncio
async def test_reference_without_an_abstract_is_left_out(
    summarizer: CitationSummarizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: a reference whose lookups were throttled (or that has no
    # abstract) used to be "summarised" from its title alone.
    record = ReferenceMetadata(
        title="Rank Analysis of Incomplete Block Designs",
        authors=["R. A. Bradley"],
        abstract=None,
        url="https://doi.org/10.2307/2334029",
    )
    summarizer.metadata_resolver = _FakeResolver(record)  # type: ignore[assignment]
    chain = _FakeChain("invented from the title")
    summarizer.citation_summarizer = chain  # type: ignore[assignment]
    monkeypatch.setattr(summarizer.arxiv_handler, "search_by_title", lambda *a: None)

    out = await summarizer.summarize([record.title], "section text")
    assert out == [] and chain.calls == 0


@pytest.mark.asyncio
async def test_prefer_full_text_falls_back_to_the_abstract(
    summarizer: CitationSummarizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: with prefer_full_text a reference that is not on arXiv but
    # has an abstract was dropped instead of summarised from the abstract.
    md = ReferenceMetadata(
        title="A Journal Paper", authors=["A B"], abstract="Its abstract.", url=None
    )
    summarizer.prefer_full_text = True
    summarizer.metadata_resolver = _FakeResolver(md)  # type: ignore[assignment]
    summarizer.citation_summarizer = _FakeChain("From the abstract.")  # type: ignore[assignment]
    monkeypatch.setattr(summarizer.arxiv_handler, "search_by_title", lambda *a: None)

    out = await summarizer.summarize([md.title], "section text")
    assert len(out) == 1 and "From the abstract." in out[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("identifier", ["2106.09685", "LoRA: Low-Rank Adaptation"])
async def test_prefer_full_text_keeps_the_arxiv_abstract_when_text_fails(
    summarizer: CitationSummarizer, monkeypatch: pytest.MonkeyPatch, identifier: str
) -> None:
    # Regression: with prefer_full_text an arXiv reference whose full text
    # could not be parsed was dropped although its abstract was fetched.
    from scholar_lens.src import arxiv_handler as ah
    from scholar_lens.src.parser import ContentParseError

    md = ah.ArxivMetadata.model_construct(
        title="LoRA", authors=["E Hu"], abstract="LoRA abstract."
    )
    summarizer.prefer_full_text = True
    summarizer.metadata_resolver = _FakeResolver(None)  # type: ignore[assignment]
    chain = _FakeChain("From the abstract.")
    summarizer.citation_summarizer = chain  # type: ignore[assignment]
    monkeypatch.setattr(summarizer.arxiv_handler, "fetch_metadata", lambda i: md)
    monkeypatch.setattr(
        summarizer.arxiv_handler, "search_by_title", lambda *a: "2106.09685"
    )

    async def no_text(arxiv_id: str) -> None:
        raise ContentParseError("no text")

    monkeypatch.setattr(summarizer, "_extract_paper_content", no_text)
    out = await summarizer.summarize([identifier], "section text")
    assert len(out) == 1 and "From the abstract." in out[0] and chain.calls == 1


@pytest.mark.asyncio
async def test_summary_is_written_per_citing_section(
    summarizer: CitationSummarizer,
) -> None:
    md = ReferenceMetadata(title="ref", authors=["A"], abstract="abs", url=None)
    resolver = _FakeResolver(md)
    chain = _FakeChain("summary")
    summarizer.metadata_resolver = resolver  # type: ignore[assignment]
    summarizer.citation_summarizer = chain  # type: ignore[assignment]

    await summarizer.summarize(["ref"], "method section")
    await summarizer.summarize(["ref"], "results section")
    assert resolver.calls == 1  # the abstract is fetched once
    assert chain.calls == 2  # but each section gets its own summary


@pytest.mark.asyncio
async def test_a_throttled_model_call_does_not_refetch_the_reference(
    summarizer: CitationSummarizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from botocore.exceptions import ClientError
    from tenacity import wait_none

    meta = ArxivMetadata(
        arxiv_id="2301.00003",
        title="P",
        authors=["A"],
        published="2023-01-01T00:00:00Z",  # type: ignore[arg-type]
        updated="2023-01-01T00:00:00Z",  # type: ignore[arg-type]
        abstract="abs",
    )
    fetches = []
    monkeypatch.setattr(
        summarizer.arxiv_handler,
        "fetch_metadata",
        lambda _id: fetches.append(_id) or meta,
    )

    class _Flaky:
        calls = 0

        async def ainvoke(self, _inputs: dict[str, Any]) -> str:
            _Flaky.calls += 1
            if _Flaky.calls == 1:
                raise ClientError(
                    {"Error": {"Code": "ThrottlingException", "Message": "slow"}},
                    "Converse",
                )
            return "summary"

    summarizer.citation_summarizer = _Flaky()  # type: ignore[assignment]
    monkeypatch.setattr(
        CitationSummarizer._invoke_summary.retry,  # type: ignore[attr-defined]
        "wait",
        wait_none(),
    )
    out = await summarizer.summarize(["2301.00003"], "ctx")
    assert len(out) == 1 and _Flaky.calls == 2 and fetches == ["2301.00003"]


@pytest.mark.asyncio
async def test_prefer_full_text_keeps_the_resolved_abstract_when_arxiv_fails(
    summarizer: CitationSummarizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: an arXiv metadata failure after its retries threw away the
    # abstract the metadata resolver had already returned.
    from scholar_lens.src.arxiv_handler import ArxivPaperError

    md = ReferenceMetadata(
        title="A Paper", authors=["A B"], abstract="Its abstract.", url=None
    )
    summarizer.prefer_full_text = True
    summarizer.metadata_resolver = _FakeResolver(md)  # type: ignore[assignment]
    summarizer.citation_summarizer = _FakeChain("From the abstract.")  # type: ignore[assignment]
    monkeypatch.setattr(
        summarizer.arxiv_handler, "search_by_title", lambda *a: "2301.00001"
    )

    def failing(_id: str) -> None:
        raise ArxivPaperError("arXiv unavailable")

    monkeypatch.setattr(summarizer.arxiv_handler, "fetch_metadata", failing)
    out = await summarizer.summarize([md.title], "section text")
    assert len(out) == 1 and "From the abstract." in out[0]
