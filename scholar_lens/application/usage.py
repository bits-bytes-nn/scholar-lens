"""The stage a model call belongs to, for cost tracking.

Code that runs a pipeline step wraps it in :func:`usage_stage`; every model
call made inside (also from tasks and worker threads started there, since
``asyncio`` and ``asyncio.to_thread`` copy the context) is recorded under that
stage. Nested stages replace the outer one for their duration.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from ..domain.usage import UNSTAGED

_STAGE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "usage_stage", default=UNSTAGED
)

F = TypeVar("F", bound=Callable[..., Any])


def current_stage() -> str:
    return _STAGE.get()


@contextmanager
def usage_stage(name: str) -> Iterator[None]:
    token = _STAGE.set(name)
    try:
        yield
    finally:
        _STAGE.reset(token)


def staged(name: str) -> Callable[[F], F]:
    """Decorate a function or coroutine function to run under stage ``name``."""

    def decorate(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def run_async(*args: Any, **kwargs: Any) -> Any:
                with usage_stage(name):
                    return await fn(*args, **kwargs)

            return run_async  # type: ignore[return-value]

        @functools.wraps(fn)
        def run(*args: Any, **kwargs: Any) -> Any:
            with usage_stage(name):
                return fn(*args, **kwargs)

        return run  # type: ignore[return-value]

    return decorate
