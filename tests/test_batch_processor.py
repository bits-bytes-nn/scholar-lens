"""Tests for utils.batch.BatchProcessor (pure logic, no AWS).

Uses simple in-memory callables. ``max_retries=1`` is used throughout so the
tenacity retry decorator makes a single attempt and never sleeps on backoff,
keeping every test fast (<2s). Progress bars are disabled via
``show_progress=False``.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from scholar_lens.src.utils.batch import BatchProcessor


@pytest.fixture(autouse=True)
def _silence_logging() -> None:
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _prepare(items: list[Any]) -> list[dict[str, Any]]:
    return [{"x": item} for item in items]


def _batch_ok(inputs: list[dict[str, Any]], config: Any = None) -> list[Any]:
    return [d["x"] * 10 for d in inputs]


def _batch_fail(inputs: list[dict[str, Any]], config: Any = None) -> list[Any]:
    raise RuntimeError("batch boom")


def _seq_ok(single_input: dict[str, Any]) -> Any:
    return single_input["x"] * 10


def _seq_fail(single_input: dict[str, Any]) -> Any:
    raise RuntimeError("seq boom")


async def _abatch_ok(inputs: list[dict[str, Any]], config: Any = None) -> list[Any]:
    return [d["x"] * 10 for d in inputs]


async def _abatch_fail(inputs: list[dict[str, Any]], config: Any = None) -> list[Any]:
    raise RuntimeError("async batch boom")


async def _aseq_ok(single_input: dict[str, Any]) -> Any:
    return single_input["x"] * 10


async def _aseq_fail(single_input: dict[str, Any]) -> Any:
    raise RuntimeError("async seq boom")


@pytest.fixture
def processor() -> BatchProcessor:
    # max_retries=1 -> single attempt, no exponential backoff sleep.
    return BatchProcessor(batch_size=3, max_concurrency=2, max_retries=1)


class TestExecuteWithFallbackSync:
    def test_all_items_succeed_in_order(self, processor: BatchProcessor) -> None:
        result = processor.execute_with_fallback(
            [1, 2, 3], _prepare, _batch_ok, _seq_ok, "task", show_progress=False
        )
        assert result == [10, 20, 30]

    def test_chunking_processes_all_items(self, processor: BatchProcessor) -> None:
        # 7 items / batch_size 3 -> 3 chunks; all processed in order.
        result = processor.execute_with_fallback(
            list(range(1, 8)),
            _prepare,
            _batch_ok,
            _seq_ok,
            "task",
            show_progress=False,
        )
        assert result == [10, 20, 30, 40, 50, 60, 70]

    def test_empty_items_returns_empty(self, processor: BatchProcessor) -> None:
        result = processor.execute_with_fallback(
            [], _prepare, _batch_ok, _seq_ok, "task", show_progress=False
        )
        assert result == []

    def test_empty_prepared_inputs_skips_chunk(self, processor: BatchProcessor) -> None:
        result = processor.execute_with_fallback(
            [1, 2, 3],
            lambda items: [],
            _batch_ok,
            _seq_ok,
            "task",
            show_progress=False,
        )
        assert result == []

    def test_batch_failure_falls_back_to_sequential(
        self, processor: BatchProcessor
    ) -> None:
        # Batch raises -> per-item sequential retry func succeeds for all.
        result = processor.execute_with_fallback(
            [1, 2, 3], _prepare, _batch_fail, _seq_ok, "task", show_progress=False
        )
        assert result == [10, 20, 30]

    def test_sequential_failure_skips_item(self, processor: BatchProcessor) -> None:
        # Batch raises, then each sequential item also raises. The fallback
        # logs and skips failed items rather than propagating -> empty result.
        result = processor.execute_with_fallback(
            [1, 2, 3], _prepare, _batch_fail, _seq_fail, "task", show_progress=False
        )
        assert result == []

    def test_run_config_overrides_batch_size(self, processor: BatchProcessor) -> None:
        # run_config batch_size of 2 still processes every item.
        result = processor.execute_with_fallback(
            [1, 2, 3, 4, 5],
            _prepare,
            _batch_ok,
            _seq_ok,
            "task",
            run_config={"batch_size": 2, "max_concurrency": 1},
            show_progress=False,
        )
        assert result == [10, 20, 30, 40, 50]


class TestAExecuteWithFallbackAsync:
    async def test_all_items_succeed_in_order(self, processor: BatchProcessor) -> None:
        result = await processor.aexecute_with_fallback(
            [1, 2, 3, 4, 5],
            _prepare,
            _abatch_ok,
            _aseq_ok,
            "task",
            show_progress=False,
        )
        assert result == [10, 20, 30, 40, 50]

    async def test_empty_items_returns_empty(self, processor: BatchProcessor) -> None:
        result = await processor.aexecute_with_fallback(
            [], _prepare, _abatch_ok, _aseq_ok, "task", show_progress=False
        )
        assert result == []

    async def test_batch_failure_falls_back_to_concurrent_sequential(
        self, processor: BatchProcessor
    ) -> None:
        result = await processor.aexecute_with_fallback(
            [1, 2, 3],
            _prepare,
            _abatch_fail,
            _aseq_ok,
            "task",
            show_progress=False,
        )
        # Concurrent fallback gathers results; order is preserved by gather.
        assert result == [10, 20, 30]

    async def test_sequential_failure_skips_item(
        self, processor: BatchProcessor
    ) -> None:
        # Failed items return None and are filtered out -> empty result.
        result = await processor.aexecute_with_fallback(
            [1, 2, 3],
            _prepare,
            _abatch_fail,
            _aseq_fail,
            "task",
            show_progress=False,
        )
        assert result == []


class TestBatchProcessorValidation:
    def test_default_field_values(self) -> None:
        bp = BatchProcessor()
        assert bp.max_concurrency == 5
        assert bp.batch_size == 10
        assert bp.max_retries == 5

    def test_invalid_batch_size_rejected(self) -> None:
        with pytest.raises(Exception):
            BatchProcessor(batch_size=0)

    def test_invalid_max_concurrency_rejected(self) -> None:
        with pytest.raises(Exception):
            BatchProcessor(max_concurrency=0)
