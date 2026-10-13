"""Tests for the RetryableBase retry predicate (which errors are retried)."""

from __future__ import annotations

import pytest

from scholar_lens.src.utils.retry import is_transient_error


class _CustomValueError(ValueError):
    pass


class TestIsRetryable:
    def test_transient_errors_are_retried(self) -> None:
        assert is_transient_error(ConnectionError("reset")) is True
        assert is_transient_error(TimeoutError("slow")) is True

    def test_unknown_errors_are_not_retried(self) -> None:
        # Allow-list design: an unrecognised error would fail the same way again.
        assert is_transient_error(RuntimeError("profile lookup failed")) is False

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

        assert is_transient_error(wrapped("ThrottlingException")) is True
        assert is_transient_error(wrapped("ValidationException")) is False

    def test_http_status_errors(self) -> None:
        import httpx

        request = httpx.Request("GET", "https://example.com")

        def status_error(code: int) -> httpx.HTTPStatusError:
            return httpx.HTTPStatusError(
                "x", request=request, response=httpx.Response(code, request=request)
            )

        assert is_transient_error(status_error(503)) is True
        assert is_transient_error(status_error(429)) is True
        assert is_transient_error(status_error(404)) is False

    def test_arxiv_errors_are_not_retried_again(self) -> None:
        from scholar_lens.src.arxiv_handler import ArxivPaperError

        try:
            raise TimeoutError("slow")
        except TimeoutError as e:
            try:
                raise ArxivPaperError("arXiv fetch failed after retries") from e
            except ArxivPaperError as outer:
                assert is_transient_error(outer) is False

    def test_deterministic_errors_are_not_retried(self) -> None:
        assert is_transient_error(ValueError("bad")) is False
        assert is_transient_error(TypeError("bad")) is False
        assert is_transient_error(KeyError("missing")) is False

    def test_subclass_of_non_retryable_is_not_retried(self) -> None:
        # MRO-based match: a ValueError subclass is still terminal.
        assert is_transient_error(_CustomValueError("bad")) is False

    def test_token_budget_is_terminal(self) -> None:
        from scholar_lens.src.metrics import TokenBudgetExceeded

        assert is_transient_error(TokenBudgetExceeded("over")) is False

    def test_domain_rejections_are_terminal(self) -> None:
        from scholar_lens.src.paper_source import NotAPdfError
        from scholar_lens.src.url_guard import UnsafeUrlError

        assert is_transient_error(NotAPdfError("nope")) is False
        assert is_transient_error(UnsafeUrlError("internal")) is False


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
        from scholar_lens.src.utils.retry import is_transient_error

        assert is_transient_error(self._client_error(code, status))

    @pytest.mark.parametrize(
        "code",
        ["AccessDeniedException", "ValidationException", "ResourceNotFoundException"],
    )
    def test_deterministic_aws_errors_not_retried(self, code: str) -> None:
        from scholar_lens.src.utils.retry import is_transient_error

        assert not is_transient_error(self._client_error(code))

    def test_read_timeout_retried(self) -> None:
        from botocore.exceptions import ReadTimeoutError

        from scholar_lens.src.utils.retry import is_transient_error

        assert is_transient_error(ReadTimeoutError(endpoint_url="https://bedrock"))


class TestRetryBudget:
    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The backoff is capped by RETRY_MAX_WAIT, read when decorating.
        monkeypatch.setattr("scholar_lens.src.utils.retry.RETRY_MAX_WAIT", 0)

    @staticmethod
    def _call_failing_with(error: BaseException) -> int:
        from scholar_lens.src.utils.retry import RetryableBase

        calls = 0

        @RetryableBase._retry("test")
        def call() -> None:
            nonlocal calls
            calls += 1
            raise error

        with pytest.raises(type(error)):
            call()
        return calls

    def test_read_timeout_is_retried_once(self) -> None:
        from botocore.exceptions import ReadTimeoutError

        error = ReadTimeoutError(endpoint_url="https://bedrock")
        assert self._call_failing_with(error) == 2

    def test_other_transient_errors_use_the_full_budget(self) -> None:
        from scholar_lens.src.utils.retry import MAX_RETRIES

        assert self._call_failing_with(ConnectionError("reset")) == MAX_RETRIES
