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
from typing import Any, NamedTuple, TypeVar

from ..domain.usage import UNSTAGED

_STAGE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "usage_stage", default=UNSTAGED
)

F = TypeVar("F", bound=Callable[..., Any])


def current_stage() -> str:
    return _STAGE.get()


class Allocation(NamedTuple):
    """Whose run a model call is billed to: the deployment and the kind."""

    project: str
    stage: str
    kind: str


_ALLOCATION: contextvars.ContextVar[Allocation | None] = contextvars.ContextVar(
    "cost_allocation", default=None
)


def current_allocation() -> Allocation | None:
    return _ALLOCATION.get()


@contextmanager
def cost_allocation(project: str, stage: str, kind: str) -> Iterator[None]:
    """Run model calls made inside under the kind's cost allocation (its
    application inference profiles)."""
    token = _ALLOCATION.set(Allocation(project, stage, kind))
    try:
        yield
    finally:
        _ALLOCATION.reset(token)


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
