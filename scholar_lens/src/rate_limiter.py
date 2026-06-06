"""A thread-safe token-bucket rate limiter for polite external API access.

arXiv (and similar) APIs throttle aggressively; firing many concurrent requests
(as the citation stage does) triggers HTTP 429/503 storms that previously stalled
whole jobs for hours. This limiter paces requests to a configured rate and lets a
server-provided ``Retry-After`` push the next allowed time further out.

It is deliberately simple and synchronous (``acquire`` blocks), because the arXiv
client runs on worker threads via ``asyncio.to_thread`` — so a plain ``threading``
primitive is the right tool and composes with concurrent callers.
"""

from __future__ import annotations

import threading
import time

from .logger import logger


class RateLimiter:
    """Token-bucket limiter: at most ``rate`` requests per ``per`` seconds."""

    def __init__(
        self, rate: float = 1.0, per: float = 3.0, *, name: str = "api"
    ) -> None:
        if rate <= 0 or per <= 0:
            raise ValueError("rate and per must be positive")
        self.rate = rate
        self.per = per
        self.name = name
        self._capacity = rate
        self._tokens = rate
        self._lock = threading.Lock()
        self._monotonic = time.monotonic
        self._last_refill = self._monotonic()
        # Absolute monotonic time before which no request may proceed (set by a
        # server Retry-After). 0 means "no penalty in effect".
        self._blocked_until = 0.0

    def acquire(self) -> None:
        """Block until a token is available (and any Retry-After penalty clears)."""
        while True:
            with self._lock:
                now = self._monotonic()
                wait = self._time_until_available(now)
                if wait <= 0:
                    self._tokens -= 1
                    return
            # Sleep outside the lock so other threads can refill/observe.
            time.sleep(min(wait, self.per))

    def penalize(self, retry_after_seconds: float) -> None:
        """Honour a server ``Retry-After``: block all callers until it elapses."""
        if retry_after_seconds <= 0:
            return
        with self._lock:
            target = self._monotonic() + retry_after_seconds
            self._blocked_until = max(self._blocked_until, target)
            self._tokens = 0.0
        logger.warning(
            "Rate limiter '%s' penalized for %.1fs (server Retry-After).",
            self.name,
            retry_after_seconds,
        )

    def _time_until_available(self, now: float) -> float:
        """Seconds to wait before a token is available. Refills under the lock."""
        if now < self._blocked_until:
            # A penalty is in force. Advance the refill clock so the penalty
            # window is NOT later credited as accrued tokens (which would let a
            # burst through the instant the penalty clears).
            self._last_refill = now
            return self._blocked_until - now
        # Refill proportionally to elapsed time.
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(
            self._capacity, self._tokens + elapsed * (self.rate / self.per)
        )
        if self._tokens >= 1:
            return 0.0
        # Time for one more token to accrue.
        return (1 - self._tokens) * (self.per / self.rate)
