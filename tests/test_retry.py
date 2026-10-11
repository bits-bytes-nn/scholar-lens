"""Tests for the RetryableBase retry predicate (which errors are retried)."""

from __future__ import annotations

import pytest

from scholar_lens.src.utils.retry import _is_retryable


class _CustomValueError(ValueError):
    pass


class TestIsRetryable:
    def test_transient_errors_are_retried(self) -> None:
        assert _is_retryable(ConnectionError("reset")) is True
        assert _is_retryable(TimeoutError("slow")) is True

    def test_unknown_errors_are_not_retried(self) -> None:
        # Allow-list design: an unrecognised error would fail the same way again.
        assert _is_retryable(RuntimeError("profile lookup failed")) is False

    def test_wrapped_errors_are_classified_by_their_cause(self) -> None:
        from botocore.exceptions import ClientError

        def wrapped(code: str) -> RuntimeError:
            try:
                raise ClientError({"Error": {"Code": code}}, "Converse")
            except ClientError as e:
                try:
                    raise RuntimeError("LLM failed to analyze figure") from e
                except RuntimeError as outer:
                    return outer

        assert _is_retryable(wrapped("ThrottlingException")) is True
        assert _is_retryable(wrapped("ValidationException")) is False

    def test_http_status_errors(self) -> None:
        import httpx

        request = httpx.Request("GET", "https://example.com")

        def status_error(code: int) -> httpx.HTTPStatusError:
            return httpx.HTTPStatusError(
                "x", request=request, response=httpx.Response(code, request=request)
            )

        assert _is_retryable(status_error(503)) is True
        assert _is_retryable(status_error(429)) is True
        assert _is_retryable(status_error(404)) is False

    def test_arxiv_errors_are_not_retried_again(self) -> None:
        from scholar_lens.src.arxiv_handler import ArxivPaperError

        try:
            raise TimeoutError("slow")
        except TimeoutError as e:
            try:
                raise ArxivPaperError("arXiv fetch failed after retries") from e
            except ArxivPaperError as outer:
                assert _is_retryable(outer) is False

    def test_deterministic_errors_are_not_retried(self) -> None:
        assert _is_retryable(ValueError("bad")) is False
        assert _is_retryable(TypeError("bad")) is False
        assert _is_retryable(KeyError("missing")) is False

    def test_subclass_of_non_retryable_is_not_retried(self) -> None:
        # MRO-based match: a ValueError subclass is still terminal.
        assert _is_retryable(_CustomValueError("bad")) is False

    def test_token_budget_is_terminal(self) -> None:
        from scholar_lens.src.metrics import TokenBudgetExceeded

        assert _is_retryable(TokenBudgetExceeded("over")) is False

    def test_domain_rejections_are_terminal(self) -> None:
        from scholar_lens.src.paper_source import NotAPdfError
        from scholar_lens.src.url_guard import UnsafeUrlError

        assert _is_retryable(NotAPdfError("nope")) is False
        assert _is_retryable(UnsafeUrlError("internal")) is False


class TestRetryPredicate:
    """Only transient failures are retried; deterministic ones fail fast."""

    @staticmethod
    def _client_error(code: str, status: int = 400):  # type: ignore[no-untyped-def]
        from botocore.exceptions import ClientError

        return ClientError(
            {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
            "Converse",
        )

    @pytest.mark.parametrize(
        ("code", "status"),
        [
            ("ThrottlingException", 429),
            ("ServiceUnavailableException", 503),
            ("ModelNotReadyException", 429),
            ("InternalServerException", 500),
        ],
    )
    def test_transient_aws_errors_retried(self, code: str, status: int) -> None:
        from scholar_lens.src.utils.retry import _is_retryable

        assert _is_retryable(self._client_error(code, status))

    @pytest.mark.parametrize(
        "code",
        ["AccessDeniedException", "ValidationException", "ResourceNotFoundException"],
    )
    def test_deterministic_aws_errors_not_retried(self, code: str) -> None:
        from scholar_lens.src.utils.retry import _is_retryable

        assert not _is_retryable(self._client_error(code))

    def test_read_timeout_retried(self) -> None:
        from botocore.exceptions import ReadTimeoutError

        from scholar_lens.src.utils.retry import _is_retryable

        assert _is_retryable(ReadTimeoutError(endpoint_url="https://bedrock"))
