from collections.abc import Callable

import tenacity

from ..logger import logger

MAX_RETRIES: int = 5
RETRY_MAX_WAIT: int = 120
RETRY_MULTIPLIER: int = 30

# Exceptions that are terminal and must NOT be retried — retrying them only
# burns backoff time (up to ~10 min across MAX_RETRIES × RETRY_MAX_WAIT) before
# the same failure reraises. Matched by class name to avoid importing those
# modules here (keeps this leaf util dependency-free). We keep a deny-list rather
# than an allow-list so a genuinely transient error we didn't foresee still gets
# retried; the entries below are the deterministic failures seen in practice:
#   - TokenBudgetExceeded: a hard cost guardrail — a single raise must abort now.
#   - Validation/parse/type errors: the same input will fail identically on retry
#     (pydantic ValidationError, our output-parser errors, ValueError/TypeError/
#     KeyError, and the domain "not a PDF / not technical / no python files"
#     rejections that are decisions, not glitches).
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
    }
)


def _is_retryable(exc: BaseException) -> bool:
    # Walk the MRO by name so subclasses of a non-retryable type are also caught
    # (e.g. a custom ValueError subclass) without importing the classes here.
    names = {klass.__name__ for klass in type(exc).__mro__}
    return names.isdisjoint(_NON_RETRYABLE_NAMES)


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
