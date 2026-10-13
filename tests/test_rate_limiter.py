"""Token-bucket pacing and Retry-After penalties (no real waiting)."""

from __future__ import annotations

from typing import Any

import pytest

from scholar_lens.src.rate_limiter import RateLimiter


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
