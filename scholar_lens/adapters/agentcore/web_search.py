"""Web search on Amazon Bedrock AgentCore Web Search (``InvokeWebSearch``).

The API is called with the run's AWS credentials (IAM only, no key) in a
region that offers it, which need not be the run's region. Each result
carries a text extract of about a thousand characters, not the page, so the
researcher fetches the page itself as it does for Brave. AWS's acceptable-use
terms ask that output built on a result cite it (guides list their sources)
and forbid storing results in bulk or building an index from them: a guide
keeps the few dozen results of its own queries, with the corpus snapshot that
lets a rerun with the same inputs write from the same sources.
"""

from __future__ import annotations

import logging
from collections.abc import Collection

from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from scholar_lens.adapters.aws.session import AwsSession
from scholar_lens.adapters.retry import CREDENTIAL_ERRORS
from scholar_lens.adapters.web.research import WebSearchProvider
from scholar_lens.domain.research import SearchResult
from scholar_lens.domain.sources import iso_date

logger = logging.getLogger(__name__)

# The AWS-managed search tool.
WEB_SEARCH_ID = "web-search.v1"
# InvokeWebSearch's limits: the query's length and the results per call.
MAX_QUERY_CHARS = 200
MAX_RESULTS = 25
# Answers no later query gets past: the caller may not search, the tool or
# the operation is not in the region, or the request is not signed right.
_STOPPING_CODES = frozenset(
    {
        "AccessDeniedException",
        "ResourceNotFoundException",
        "UnknownOperationException",
        "UnrecognizedClientException",
        "MissingAuthenticationTokenException",
        "ExpiredTokenException",
        "InvalidSignatureException",
    }
)


def _query(text: str) -> str:
    """``text`` within the API's length, cut at a word."""
    text = " ".join(text.split())
    if len(text) <= MAX_QUERY_CHARS:
        return text
    cut = text[:MAX_QUERY_CHARS]
    return cut.rsplit(" ", 1)[0] if " " in cut else cut


class AgentCoreSearchProvider(WebSearchProvider):
    """:class:`~scholar_lens.adapters.web.research.WebSearchProvider` on
    AgentCore Web Search."""

    def __init__(self, session: AwsSession, region_name: str) -> None:
        """No request is made until :meth:`search` is called.

        Args:
            session: The AWS session whose credentials sign the calls.
            region_name: A region that offers AgentCore Web Search.
        """
        self.region_name = region_name
        self.failed_searches = 0
        self.exhausted = False
        self.stop_reason = ""
        # One retry for a throttled or failed call (a search is a read);
        # the fallback backend takes a query that fails again.
        self._client = session.client(
            "bedrock-agentcore",
            region_name=region_name,
            config=BotoConfig(
                retries={"mode": "standard", "max_attempts": 2},
                connect_timeout=10,
                read_timeout=30,
            ),
        )

    def search(
        self,
        query: str,
        *,
        count: int = 5,
        domains: Collection[str] = (),
        exclude: Collection[str] = (),
    ) -> list[SearchResult]:
        if self.exhausted:
            # Stopped for the run (last in the chain, or alone): no call.
            self.failed_searches += 1
            return []
        domain_filter: dict[str, list[str]] = {}
        if domains:
            domain_filter["include"] = list(domains)
        if exclude:
            domain_filter["exclude"] = list(exclude)
        request: dict[str, object] = {
            "webSearchId": WEB_SEARCH_ID,
            "searchQuery": _query(query),
            "maxResults": max(1, min(count, MAX_RESULTS)),
        }
        if domain_filter:
            request["filters"] = {"domainFilter": domain_filter}
        try:
            response = self._client.invoke_web_search(**request)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            logger.warning("AgentCore web search failed for '%s': %s", query, code)
            self.failed_searches += 1
            if code in _STOPPING_CODES:
                self.exhausted = True
                self.stop_reason = code
            return []
        except CREDENTIAL_ERRORS as e:
            logger.warning("AgentCore web search has no credentials: %s", e)
            self.failed_searches += 1
            self.exhausted = True
            self.stop_reason = "no AWS credentials"
            return []
        except BotoCoreError as e:
            logger.warning("AgentCore web search failed for '%s': %s", query, e)
            self.failed_searches += 1
            return []
        results = response.get("results")
        if not isinstance(results, list):
            return []
        return [
            SearchResult(
                title=str(r.get("title") or ""),
                url=str(r["url"]),
                description=str(r.get("text") or ""),
                published=iso_date(str(r.get("publishedAt") or "")),
            )
            for r in results
            if isinstance(r, dict) and r.get("url")
        ]
