"""Reference metadata providers, the resolver chain and the title gate
(no real network)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from scholar_lens.src import citation_metadata
from scholar_lens.src.citation_metadata import (
    ChainedMetadataResolver,
    CrossrefProvider,
    MetadataProvider,
    ReferenceMetadata,
    SemanticScholarProvider,
    _strip_jats,
    titles_match,
)


class _NoWaitLimiter:
    def __init__(self) -> None:
        self.penalties: list[float] = []

    def acquire(self) -> None:
        pass

    def penalize(self, seconds: float) -> None:
        self.penalties.append(seconds)


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(citation_metadata, "_CROSSREF_LIMITER", _NoWaitLimiter())
    monkeypatch.setattr(
        citation_metadata, "_SEMANTIC_SCHOLAR_LIMITER", _NoWaitLimiter()
    )


def test_429_waits_for_retry_after_then_retries_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: one attempt per provider, Retry-After ignored, so a burst of
    # citations turned into silent misses.
    statuses = [429, 200]

    def fake_get(url: str, **_: Any) -> httpx.Response:
        status = statuses.pop(0)
        body = {"message": {"items": [{"title": ["T"], "abstract": "<p>A.</p>"}]}}
        return httpx.Response(
            status,
            json=body,
            headers={"Retry-After": "7"},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr("scholar_lens.src.citation_metadata.httpx.get", fake_get)
    record = CrossrefProvider().lookup("T")
    assert record is not None and record.abstract == "A."
    assert citation_metadata._CROSSREF_LIMITER.penalties == [7.0]  # type: ignore[attr-defined]


def test_semantic_scholar_sends_its_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_get(url: str, *, headers: Any = None, **_: Any) -> httpx.Response:
        seen["headers"] = headers
        return httpx.Response(200, json={"data": []}, request=httpx.Request("GET", url))

    monkeypatch.setattr("scholar_lens.src.citation_metadata.httpx.get", fake_get)
    assert SemanticScholarProvider(api_key="k").lookup("x") is None
    assert seen["headers"] == {"x-api-key": "k"}


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
        url: str, *, params: Any = None, timeout: Any = None, **_: Any
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
    def boom(url: str, **_: Any) -> httpx.Response:
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
def test_titles_match_gate(
    queried: str | None, resolved: str | None, expected: bool
) -> None:
    assert titles_match(queried, resolved) is expected


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
        assert titles_match(queried, resolved) is False

    def test_same_title_with_punctuation_matches(self) -> None:
        assert titles_match("Attention is All you Need", "Attention Is All You Need.")
