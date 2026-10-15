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


@pytest.mark.parametrize(
    ("header", "expected"),
    [("3600", 60.0), ("12", 12.0), ("inf", 0.0), ("nan", 0.0), ("soon", 0.0)],
)
def test_arxiv_retry_after_is_capped(header: str, expected: float) -> None:
    from types import SimpleNamespace

    from scholar_lens.src.rate_limiter import (
        MAX_RETRY_AFTER_SECONDS,
        arxiv_retry_after_seconds,
    )

    assert MAX_RETRY_AFTER_SECONDS == 60.0
    error = Exception("429")
    error.response = SimpleNamespace(headers={"Retry-After": header})  # type: ignore[attr-defined]
    assert arxiv_retry_after_seconds(error) == expected


def _fake_clock_arxiv_limiter(
    monkeypatch: pytest.MonkeyPatch, module: Any, name: str
) -> tuple[_FakeClock, RateLimiter]:
    clock = _FakeClock()
    monkeypatch.setattr("scholar_lens.src.rate_limiter.time.sleep", clock.sleep)
    limiter = _limiter_with_clock(clock, rate=1.0, per=3.0, name="arxiv")
    monkeypatch.setattr(module, name, limiter)
    return clock, limiter


async def test_arxiv_html_fetch_is_paced_and_penalty_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from scholar_lens.src import parser
    from scholar_lens.src.parser import ContentParseError, HTMLParser

    clock, limiter = _fake_clock_arxiv_limiter(
        monkeypatch, parser, "ARXIV_RATE_LIMITER"
    )
    statuses = iter([200, 429])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), headers={"Retry-After": "86400"})

    html = HTMLParser()
    html._async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    start = clock.now
    await html._fetch_html_from_url("https://arxiv.org/html/1")
    with pytest.raises(ContentParseError):
        await html._fetch_html_from_url("https://arxiv.org/html/2")
    assert clock.now - start == pytest.approx(3.0)  # the second request waited
    # A day-long Retry-After holds arXiv back for the cap, not a day.
    assert limiter._blocked_until - clock.now == pytest.approx(60.0)


def test_arxiv_pdf_download_is_paced_and_penalty_capped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import requests
    import responses

    from scholar_lens.src import arxiv_handler
    from scholar_lens.src.arxiv_handler import ArxivHandler

    clock, limiter = _fake_clock_arxiv_limiter(
        monkeypatch, arxiv_handler, "_ARXIV_RATE_LIMITER"
    )
    with responses.RequestsMock() as mock:
        mock.add(
            responses.GET,
            "https://arxiv.org/pdf/1",
            status=429,
            headers={"Retry-After": "86400"},
        )
        mock.add(
            responses.GET,
            "https://arxiv.org/pdf/2",
            status=429,
            headers={"Retry-After": "86400"},
        )
        start = clock.now
        for name in ("1", "2"):
            with pytest.raises(requests.HTTPError):
                ArxivHandler._download(
                    f"https://arxiv.org/pdf/{name}", tmp_path / f"{name}.pdf"
                )
    # The second download waited out the capped penalty, then set a new one.
    assert clock.now - start == pytest.approx(60.0)
    assert limiter._blocked_until - clock.now == pytest.approx(60.0)


def test_a_429_in_the_message_alone_is_not_a_rate_limit() -> None:
    from types import SimpleNamespace

    from scholar_lens.src.rate_limiter import arxiv_retry_after_seconds

    not_found = Exception("404 for https://arxiv.org/abs/2405.04291")
    not_found.response = SimpleNamespace(headers={}, status_code=404)  # type: ignore[attr-defined]
    assert arxiv_retry_after_seconds(not_found) == 0.0
    limited = Exception("rate limited")
    limited.response = SimpleNamespace(headers={}, status_code=429)  # type: ignore[attr-defined]
    assert arxiv_retry_after_seconds(limited) == 30.0
