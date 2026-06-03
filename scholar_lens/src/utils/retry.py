from collections.abc import Callable

import tenacity

from ..logger import logger

MAX_RETRIES: int = 5
RETRY_MAX_WAIT: int = 120
RETRY_MULTIPLIER: int = 30


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
            before_sleep=lambda retry_state: logger.warning(
                "Retrying '%s' (attempt %d failed). Waiting %.1fs",
                operation_name,
                retry_state.attempt_number,
                retry_state.next_action.sleep if retry_state.next_action else 0,
            ),
            reraise=True,
        )
