"""Tests for the RetryableBase retry predicate (which errors are retried)."""

from __future__ import annotations

from scholar_lens.src.utils.retry import _is_retryable


class _CustomValueError(ValueError):
    pass


class TestIsRetryable:
    def test_transient_errors_are_retried(self) -> None:
        # Unforeseen / transient errors default to retryable (deny-list design).
        assert _is_retryable(ConnectionError("reset")) is True
        assert _is_retryable(TimeoutError("slow")) is True
        assert _is_retryable(RuntimeError("bedrock throttled")) is True

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
