import asyncio
import math
from collections.abc import Callable
from typing import Any

import tenacity
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from ..logger import logger


class BatchProcessor(BaseModel):
    max_concurrency: int = Field(default=5, ge=1)
    retry_multiplier: float = Field(default=30.0, ge=1.0)
    retry_max_wait: int = Field(default=120, ge=0)
    max_retries: int = Field(default=5, ge=1)
    batch_size: int = Field(default=10, ge=1)

    def execute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        batch_func: Callable[..., list[Any]],
        sequential_func: Callable[..., Any],
        task_name: str,
        run_config: dict[str, Any] | None = None,
        show_progress: bool = True,
    ) -> list[Any]:
        if not items_to_process:
            return []
        max_concurrency = (
            run_config.get("max_concurrency", self.max_concurrency)
            if run_config
            else self.max_concurrency
        )
        batch_size = (
            run_config.get("batch_size", self.batch_size)
            if run_config
            else self.batch_size
        )
        prepared_batch_func = self._create_batch_func(batch_func, max_concurrency)
        retrying_sequential_func = self._create_retry_decorator(task_name)(
            sequential_func
        )
        all_results = []
        num_items = len(items_to_process)
        num_chunks = math.ceil(num_items / batch_size)
        logger.info(
            "Starting processing for '%s': %d items in %d chunks (batch size: %d)",
            task_name,
            num_items,
            num_chunks,
            batch_size,
        )
        for i in tqdm(
            range(0, num_items, batch_size),
            desc=f"Processing: {task_name}",
            disable=not show_progress,
        ):
            chunk_items = items_to_process[i : i + batch_size]
            chunk_num = (i // batch_size) + 1
            logger.debug(
                "Processing chunk %d/%d (%d items)",
                chunk_num,
                num_chunks,
                len(chunk_items),
            )
            chunk_inputs = prepare_inputs_func(chunk_items)
            if not chunk_inputs:
                logger.warning(
                    "No valid inputs prepared for chunk %d, skipping", chunk_num
                )
                continue
            try:
                logger.debug("Attempting batch processing for chunk %d", chunk_num)
                chunk_results = prepared_batch_func(chunk_inputs)
                all_results.extend(chunk_results)
                logger.debug("Chunk %d processed successfully in batch mode", chunk_num)
            except Exception as e:
                logger.warning(
                    "Batch processing failed for chunk %d: %s. Falling back to sequential processing",
                    chunk_num,
                    e,
                )
                chunk_results = self._process_sequentially_with_fallback(
                    chunk_inputs,
                    retrying_sequential_func,
                    f"{task_name} (chunk {chunk_num})",
                    show_progress=show_progress,
                )
                all_results.extend(chunk_results)
        logger.info("Completed '%s': processed %d results", task_name, len(all_results))
        return all_results

    @staticmethod
    def _create_batch_func(
        batch_func: Callable[..., list[Any]], max_concurrency: int
    ) -> Callable:
        def _batch_func(inputs: list[dict[str, Any]]) -> list[Any]:
            return batch_func(
                inputs, config=RunnableConfig(max_concurrency=max_concurrency)
            )

        return _batch_func

    def _create_retry_decorator(self, operation_name: str) -> Callable:
        return tenacity.retry(
            wait=tenacity.wait_exponential(
                multiplier=self.retry_multiplier, max=self.retry_max_wait
            ),
            stop=tenacity.stop_after_attempt(self.max_retries),
            before_sleep=self._create_retry_log_callback(operation_name),
            reraise=True,
        )

    @staticmethod
    def _create_retry_log_callback(operation_name: str) -> Callable:
        def log_retry(retry_state):
            wait_time = retry_state.next_action.sleep if retry_state.next_action else 0
            logger.warning(
                "Retrying '%s' (attempt %d failed). Waiting %.1fs",
                operation_name,
                retry_state.attempt_number,
                wait_time,
            )

        return log_retry

    @staticmethod
    def _process_sequentially_with_fallback(
        inputs: list[dict[str, Any]],
        sequential_func: Callable[[dict[str, Any]], Any],
        task_name: str,
        show_progress: bool = True,
    ) -> list[Any]:
        logger.info("Processing %d items sequentially for '%s'", len(inputs), task_name)
        results = []
        progress_desc = f"Sequential Processing: '{task_name}'"
        successful_count = 0
        for single_input in tqdm(inputs, desc=progress_desc, disable=not show_progress):
            try:
                result = sequential_func(single_input)
                results.append(result)
                successful_count += 1
            except Exception as e:
                logger.error(
                    "Sequential processing failed for single item in '%s': %s",
                    task_name,
                    e,
                )
                # Preserve positional alignment: append a None placeholder rather
                # than skipping, so len(results) == len(inputs) and callers that
                # zip results back to inputs don't shift onto the wrong item.
                results.append(None)
        logger.info(
            "Sequential processing completed for '%s': %d/%d items processed successfully",
            task_name,
            successful_count,
            len(inputs),
        )
        return results

    async def aexecute_with_fallback(
        self,
        items_to_process: list[Any],
        prepare_inputs_func: Callable[[list[Any]], list[dict[str, Any]]],
        batch_func: Callable[..., Any],
        sequential_func: Callable[..., Any],
        task_name: str,
        run_config: dict[str, Any] | None = None,
        show_progress: bool = True,
    ) -> list[Any]:
        if not items_to_process:
            return []
        max_concurrency = (
            run_config.get("max_concurrency", self.max_concurrency)
            if run_config
            else self.max_concurrency
        )
        batch_size = (
            run_config.get("batch_size", self.batch_size)
            if run_config
            else self.batch_size
        )
        prepared_batch_func = self._create_async_batch_func(batch_func, max_concurrency)
        retrying_sequential_func = self._create_retry_decorator(task_name)(
            sequential_func
        )
        all_results = []
        num_items = len(items_to_process)
        num_chunks = math.ceil(num_items / batch_size)
        logger.info(
            "Starting async processing for '%s': %d items in %d chunks (batch size: %d)",
            task_name,
            num_items,
            num_chunks,
            batch_size,
        )
        chunk_iterator = async_tqdm(
            range(0, num_items, batch_size),
            desc=f"Processing: {task_name}",
            disable=not show_progress,
        )
        for i in chunk_iterator:
            chunk_items = items_to_process[i : i + batch_size]
            chunk_num = (i // batch_size) + 1
            logger.debug(
                "Processing chunk %d/%d (%d items)",
                chunk_num,
                num_chunks,
                len(chunk_items),
            )
            chunk_inputs = prepare_inputs_func(chunk_items)
            if not chunk_inputs:
                logger.warning(
                    "No valid inputs prepared for chunk %d, skipping", chunk_num
                )
                continue
            try:
                chunk_results = await prepared_batch_func(chunk_inputs)
                all_results.extend(chunk_results)
            except Exception as e:
                logger.warning(
                    "Async batch processing failed for chunk %d: %s. Falling back to concurrent sequential processing",
                    chunk_num,
                    e,
                )
                chunk_results = await self._aprocess_sequentially_with_fallback(
                    chunk_inputs,
                    retrying_sequential_func,
                    f"{task_name} (chunk {chunk_num})",
                    max_concurrency,
                    show_progress,
                )
                all_results.extend(chunk_results)
        logger.info("Completed '%s': processed %d results", task_name, len(all_results))
        return all_results

    @staticmethod
    def _create_async_batch_func(
        batch_func: Callable[..., Any], max_concurrency: int
    ) -> Callable:
        async def _batch_func(inputs: list[dict[str, Any]]) -> list[Any]:
            return await batch_func(
                inputs, config=RunnableConfig(max_concurrency=max_concurrency)
            )

        return _batch_func

    @staticmethod
    async def _aprocess_sequentially_with_fallback(
        inputs: list[dict[str, Any]],
        sequential_func: Callable[[dict[str, Any]], Any],
        task_name: str,
        max_concurrency: int,
        show_progress: bool = True,
    ) -> list[Any]:
        logger.info("Processing %d items concurrently for '%s'", len(inputs), task_name)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _process_one(single_input):
            async with semaphore:
                try:
                    return await sequential_func(single_input)
                except Exception as e:
                    logger.error(
                        "Concurrent sequential processing failed for item in '%s': %s",
                        task_name,
                        e,
                    )
                    return None

        tasks = [_process_one(single_input) for single_input in inputs]
        progress_desc = f"Concurrent Fallback: '{task_name}'"
        results = await async_tqdm.gather(
            *tasks, disable=not show_progress, desc=progress_desc
        )
        # Keep failed items as None placeholders rather than dropping them: the
        # batch path returns exactly one result per input, and callers (e.g.
        # code_retriever._augment_documents) zip the results back against their
        # inputs positionally. Filtering out failures here would shorten the list
        # and silently shift every downstream item onto the wrong input.
        results = list(results)
        successful_count = sum(1 for res in results if res is not None)
        logger.info(
            "Concurrent sequential processing completed for '%s': %d/%d items processed successfully",
            task_name,
            successful_count,
            len(inputs),
        )
        return results
