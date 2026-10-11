"""Tests for utils.batch.BatchProcessor (pure logic, no AWS)."""

from __future__ import annotations

from typing import Any

import pytest

from scholar_lens.src.utils import retry as retry_module
from scholar_lens.src.utils.batch import BatchProcessor


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry_module, "RETRY_MULTIPLIER", 0)
    monkeypatch.setattr(retry_module, "RETRY_MAX_WAIT", 0)


def _prepare(items: list[Any]) -> list[dict[str, Any]]:
    return [{"x": item} for item in items]


class FakeChain:
    """abatch/ainvoke stand-in that fails chosen items and counts calls."""

    def __init__(self, fail_in_batch: set[int], fail_always: set[int] | None = None):
        self.fail_in_batch = fail_in_batch
        self.fail_always = fail_always or set()
        self.batch_calls = 0
        self.single_calls: list[int] = []

    async def abatch(
        self, inputs: list[dict[str, Any]], config: Any = None, **kwargs: Any
    ) -> list[Any]:
        assert kwargs.get("return_exceptions") is True
        self.batch_calls += 1
        return [
            RuntimeError("throttled") if d["x"] in self.fail_in_batch else d["x"] * 10
            for d in inputs
        ]

    async def ainvoke(self, payload: dict[str, Any]) -> Any:
        self.single_calls.append(payload["x"])
        if payload["x"] in self.fail_always:
            raise ValueError("bad input")  # non-retryable: no backoff loop
        return payload["x"] * 10


@pytest.fixture
def processor() -> BatchProcessor:
    return BatchProcessor(batch_size=3, max_concurrency=2)


async def _run(processor: BatchProcessor, items: list[int], chain: FakeChain) -> list:
    return await processor.aexecute_with_fallback(
        items, _prepare, chain.abatch, chain.ainvoke, "task"
    )


class TestAExecuteWithFallback:
    async def test_all_succeed_in_order_across_chunks(
        self, processor: BatchProcessor
    ) -> None:
        chain = FakeChain(fail_in_batch=set())
        assert await _run(processor, [1, 2, 3, 4, 5], chain) == [10, 20, 30, 40, 50]
        assert chain.batch_calls == 2 and chain.single_calls == []

    async def test_empty_items_returns_empty(self, processor: BatchProcessor) -> None:
        assert await _run(processor, [], FakeChain(set())) == []

    async def test_only_failed_items_are_retried(
        self, processor: BatchProcessor
    ) -> None:
        chain = FakeChain(fail_in_batch={2})
        assert await _run(processor, [1, 2, 3], chain) == [10, 20, 30]
        # Items 1 and 3 succeeded in the batch and must not be billed again.
        assert chain.single_calls == [2]

    async def test_persistent_failure_keeps_alignment(
        self, processor: BatchProcessor
    ) -> None:
        chain = FakeChain(fail_in_batch={2}, fail_always={2})
        assert await _run(processor, [1, 2, 3], chain) == [10, None, 30]


class TestBatchProcessorValidation:
    def test_defaults(self) -> None:
        bp = BatchProcessor()
        assert bp.max_concurrency == 5 and bp.batch_size == 10

    @pytest.mark.parametrize("field", ["batch_size", "max_concurrency"])
    def test_non_positive_rejected(self, field: str) -> None:
        with pytest.raises(ValueError):
            BatchProcessor(**{field: 0})
