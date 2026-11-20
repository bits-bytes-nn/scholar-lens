"""AgentCore Web Search as a guide's search backend."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    SSOTokenLoadError,
)

from scholar_lens.adapters.agentcore.web_search import (
    MAX_QUERY_CHARS,
    AgentCoreSearchProvider,
)
from scholar_lens.adapters.web.research import FallbackSearchProvider, WebSearchProvider
from scholar_lens.domain.research import SearchResult


class _Client:
    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def invoke_web_search(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class _Session:
    def __init__(self, client: _Client) -> None:
        self._client = client
        self.region: str | None = None

    def client(self, name: str, *, region_name: str, **_: Any) -> _Client:
        assert name == "bedrock-agentcore"
        self.region = region_name
        return self._client


def _provider(answer: Any) -> tuple[AgentCoreSearchProvider, _Client, _Session]:
    client = _Client(answer)
    session = _Session(client)
    provider = AgentCoreSearchProvider(session, "ap-northeast-1")  # type: ignore[arg-type]
    return provider, client, session


def _error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "m"}}, "InvokeWebSearch")


def test_results_become_search_results_in_the_configured_region() -> None:
    answer = {
        "results": [
            {
                "url": "https://docs.vllm.ai/en/latest/design/arch_overview/",
                "title": "Architecture Overview",
                "text": "The engine core runs the scheduler.",
                "publishedAt": dt.datetime(2026, 10, 29, tzinfo=dt.UTC),
            },
            {"url": "https://example.org/x", "text": "no title, no date"},
            {"title": "no url"},
        ]
    }
    provider, client, session = _provider(answer)
    results = provider.search("vLLM scheduler", count=6, exclude=["pinterest.com"])
    assert session.region == "ap-northeast-1"
    assert results == [
        SearchResult(
            title="Architecture Overview",
            url="https://docs.vllm.ai/en/latest/design/arch_overview/",
            description="The engine core runs the scheduler.",
            published="2026-10-29",
        ),
        SearchResult(
            title="", url="https://example.org/x", description="no title, no date"
        ),
    ]
    (call,) = client.calls
    assert call["webSearchId"] == "web-search.v1"
    assert call["maxResults"] == 6
    assert call["filters"] == {"domainFilter": {"exclude": ["pinterest.com"]}}


def test_domains_restrict_and_the_query_fits_the_api() -> None:
    provider, client, _ = _provider({"results": []})
    provider.search("word " * 80, count=99, domains=["docs.ray.io"])
    (call,) = client.calls
    assert len(call["searchQuery"]) <= MAX_QUERY_CHARS
    assert not call["searchQuery"].endswith(" ")
    assert call["maxResults"] == 25
    assert call["filters"] == {"domainFilter": {"include": ["docs.ray.io"]}}


@pytest.mark.parametrize(
    "error",
    [
        _error("ThrottlingException"),
        _error("InternalServerException"),
        EndpointConnectionError(endpoint_url="https://x"),
    ],
)
def test_a_transient_failure_counts_and_leaves_the_backend_on(error: Exception) -> None:
    provider, _, _ = _provider(error)
    assert provider.search("q") == []
    assert provider.failed_searches == 1
    assert not provider.exhausted


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (_error("AccessDeniedException"), "AccessDeniedException"),
        (_error("ResourceNotFoundException"), "ResourceNotFoundException"),
        (_error("UnknownOperationException"), "UnknownOperationException"),
        (NoCredentialsError(), "no AWS credentials"),
        (SSOTokenLoadError(error_msg="expired"), "no AWS credentials"),
    ],
)
def test_a_denied_call_stops_the_backend(error: Exception, reason: str) -> None:
    provider, client, _ = _provider(error)
    assert provider.search("q") == []
    assert provider.exhausted and provider.stop_reason == reason
    # Stopped: later queries make no call and count as failed.
    assert provider.search("r") == []
    assert len(client.calls) == 1 and provider.failed_searches == 2


def test_one_long_word_is_cut_to_the_limit_and_both_domain_lists_are_sent() -> None:
    provider, client, _ = _provider({"results": []})
    provider.search("x" * 500, domains=["a.example"], exclude=["b.example"])
    (call,) = client.calls
    assert call["searchQuery"] == "x" * MAX_QUERY_CHARS
    assert call["filters"] == {
        "domainFilter": {"include": ["a.example"], "exclude": ["b.example"]}
    }


class _Fixed(WebSearchProvider):
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str, **_: Any) -> list[SearchResult]:
        self.queries.append(query)
        return self.results


def test_a_denied_agentcore_hands_every_query_to_the_next_backend() -> None:
    agentcore, _, _ = _provider(_error("AccessDeniedException"))
    hit = SearchResult(title="t", url="https://b.example/")
    nxt = _Fixed([hit])
    chain = FallbackSearchProvider(agentcore, nxt)
    assert chain.search("one") == [hit]
    assert chain.search("two") == [hit]
    assert nxt.queries == ["one", "two"]
