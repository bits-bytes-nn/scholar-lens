import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ..logger import logger
from .retry import RetryableBase


class BatchProcessor(BaseModel):
    """Runs a LangChain chain over many items in chunks.

    Each chunk goes through ``abatch(..., return_exceptions=True)``; only the
    items that failed are retried one by one (with the shared retry policy), so
    successful results are never re-billed. Results stay aligned with the
    inputs: an item that still fails yields ``None``.
    """

    max_concurrency: int = Field(default=5, ge=1)
    batch_size: int = Field(default=10, ge=1)
    # INFO progress line every N chunks (long runs otherwise look stuck).
    progress_every: int = Field(default=5, ge=1)

    async def aexecute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        batch_func: Callable[..., Awaitable[list[Any]]],
        sequential_func: Callable[[dict[str, Any]], Awaitable[Any]],
        task_name: str,
    ) -> list[Any]:
        if not items_to_process:
            return []
        retrying = RetryableBase._retry(task_name)(sequential_func)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def retry_one(single_input: dict[str, Any]) -> Any:
            async with semaphore:
                try:
                    return await retrying(single_input)
                except Exception as e:
                    logger.error("'%s' failed for one item: %s", task_name, e)
                    return None

        results: list[Any] = []
        total_chunks = -(-len(items_to_process) // self.batch_size)
        for number, start in enumerate(
            range(0, len(items_to_process), self.batch_size), start=1
        ):
            chunk_inputs = prepare_inputs_func(
                items_to_process[start : start + self.batch_size]
            )
            if not chunk_inputs:
                logger.warning(
                    "No valid inputs for '%s' chunk at %d; skipping.", task_name, start
                )
                continue
            chunk_results = list(
                await batch_func(
                    chunk_inputs,
                    config=RunnableConfig(max_concurrency=self.max_concurrency),
                    return_exceptions=True,
                )
            )
            failed = [
                i for i, r in enumerate(chunk_results) if isinstance(r, Exception)
            ]
            if failed:
                logger.warning(
                    "'%s': retrying %d of %d items individually.",
                    task_name,
                    len(failed),
                    len(chunk_inputs),
                )
                retried = await asyncio.gather(
                    *(retry_one(chunk_inputs[i]) for i in failed)
                )
                for i, value in zip(failed, retried, strict=True):
                    chunk_results[i] = value
            results.extend(chunk_results)
            if number % self.progress_every == 0 and number < total_chunks:
                logger.info(
                    "'%s': %d/%d items done.",
                    task_name,
                    len(results),
                    len(items_to_process),
                )
        logger.info("Completed '%s': %d results", task_name, len(results))
        return results
