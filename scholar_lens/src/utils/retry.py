"""The project's single retry policy for model and external-API calls.

botocore clients are configured with one attempt (see ``factories``), so this
decorator is the only layer that retries (arXiv API calls are retried inside
``ArxivHandler`` and are denied here). Only known-transient failures are
retried — throttling, 5xx, timeouts, dropped connections — found anywhere in an
exception's cause chain; everything else fails fast, since it would fail the
same way again (bad request, missing permission, budget breach, invalid or
truncated output, programming errors).
"""

from collections.abc import Callable

import httpx
import requests
import tenacity
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from ..logger import logger

MAX_RETRIES: int = 5
# A read timeout means a long generation ran past the client timeout; each
# retry repeats that whole (billable, untracked) call, so it gets one retry.
READ_TIMEOUT_MAX_ATTEMPTS: int = 2
RETRY_MAX_WAIT: int = 120
RETRY_MULTIPLIER: int = 30

# AWS error codes that mean "try again later".
_RETRYABLE_AWS_CODES: frozenset[str] = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "ServiceUnavailable",
        "InternalServerException",
        "InternalFailure",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "RequestTimeout",
        "RequestTimeoutException",
        "SlowDown",
    }
)

_TRANSIENT_BOTO_ERRORS = (
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

# Deterministic failures matched by class name (so this leaf module needs no
# imports from the domain modules): a retry would fail the same way.
_NON_RETRYABLE_NAMES: frozenset[str] = frozenset(
    {
        "TokenBudgetExceeded",
        "ValidationError",
        "ValueError",
        "TypeError",
        "KeyError",
        "OutputParserException",
        "NotAPdfError",
        "NotTechnicalContentError",
        "NoPythonFilesError",
        "UnsafeUrlError",
        "ArxivNotFoundError",
        # ArxivHandler already retried the arXiv call; retrying again here
        # would multiply the attempts against a rate-limited API.
        "ArxivPaperError",
        "PaperSourceError",
        "OutputTruncatedError",
    }
)

_TRANSIENT_HTTP_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    requests.Timeout,
    requests.ConnectionError,
    TimeoutError,
    ConnectionError,
)
_MAX_CAUSE_DEPTH = 6


def _is_transient_status(status: int) -> bool:
    return status == 429 or status >= 500


def _classify(exc: BaseException) -> bool | None:
    """True/False for an exception this policy recognises, None otherwise."""
    names = {klass.__name__ for klass in type(exc).__mro__}
    if not names.isdisjoint(_NON_RETRYABLE_NAMES):
        return False
    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        return error.get("Code", "") in _RETRYABLE_AWS_CODES or _is_transient_status(
            status
        )
    if isinstance(exc, _TRANSIENT_BOTO_ERRORS):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return _is_transient_status(exc.response.status_code)
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return _is_transient_status(exc.response.status_code)
    if isinstance(exc, _TRANSIENT_HTTP_ERRORS):
        return True
    return None


def _cause_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while (
        current is not None and current not in chain and len(chain) < _MAX_CAUSE_DEPTH
    ):
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _is_retryable(exc: BaseException) -> bool:
    """Retry only when the cause chain holds a known-transient error and no
    known-deterministic one (a wrapper such as ``RuntimeError(...) from
    ClientError`` is classified by what it wraps)."""
    verdicts = [_classify(link) for link in _cause_chain(exc)]
    if False in verdicts:
        return False
    return True in verdicts


def is_transient_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a known-transient failure (throttling, 5xx, timeout,
    dropped connection) under this policy — for callers that poll or degrade
    instead of retrying."""
    return _is_retryable(exc)


def _should_stop(retry_state: tenacity.RetryCallState) -> bool:
    if retry_state.attempt_number >= MAX_RETRIES:
        return True
    outcome = retry_state.outcome
    error = outcome.exception() if outcome is not None and outcome.failed else None
    return (
        error is not None
        and retry_state.attempt_number >= READ_TIMEOUT_MAX_ATTEMPTS
        and any(isinstance(link, ReadTimeoutError) for link in _cause_chain(error))
    )


class RetryableBase:
    @staticmethod
    def _retry(operation_name: str) -> Callable:
        return tenacity.retry(
            # Exponential backoff with jitter so concurrent retries against a
            # rate-limited API don't stampede.
            wait=tenacity.wait_exponential_jitter(
                initial=RETRY_MULTIPLIER, max=RETRY_MAX_WAIT
            ),
            stop=_should_stop,
            retry=tenacity.retry_if_exception(_is_retryable),
            before_sleep=lambda retry_state: logger.warning(
                "Retrying '%s' (attempt %d failed: %s). Waiting %.1fs",
                operation_name,
                retry_state.attempt_number,
                (
                    retry_state.outcome.exception()
                    if retry_state.outcome and retry_state.outcome.failed
                    else "unknown error"
                ),
                retry_state.next_action.sleep if retry_state.next_action else 0,
            ),
            reraise=True,
        )
