from collections.abc import Callable

import tenacity

from ..logger import logger

MAX_RETRIES: int = 5
RETRY_MAX_WAIT: int = 120
RETRY_MULTIPLIER: int = 30

# Exceptions that are terminal and must NOT be retried — retrying them only
# burns backoff time before the same failure reraises. Matched by class name to
# avoid importing those modules here (keeps this leaf util dependency-free).
# TokenBudgetExceeded is a hard cost guardrail: a single raise must abort now.
_NON_RETRYABLE_NAMES: frozenset[str] = frozenset({"TokenBudgetExceeded"})


def _is_retryable(exc: BaseException) -> bool:
    return type(exc).__name__ not in _NON_RETRYABLE_NAMES


class RetryableBase:
    @staticmethod
    def _retry(operation_name: str) -> Callable:
        return tenacity.retry(
            # Exponential backoff with jitter to avoid a thundering herd of
            # concurrent retries hammering rate-limited APIs (e.g. arXiv 429).
            wait=tenacity.wait_exponential_jitter(
                initial=RETRY_MULTIPLIER, max=RETRY_MAX_WAIT
            ),
            stop=tenacity.stop_after_attempt(MAX_RETRIES),
            # Never retry terminal errors (e.g. a token-budget breach).
            retry=tenacity.retry_if_exception(_is_retryable),
            before_sleep=lambda retry_state: logger.warning(
                "Retrying '%s' (attempt %d failed). Waiting %.1fs",
                operation_name,
                retry_state.attempt_number,
                retry_state.next_action.sleep if retry_state.next_action else 0,
            ),
            reraise=True,
        )
