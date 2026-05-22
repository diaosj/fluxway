"""
Unit tests for SPMDWorker.stop() drain wakeup mechanism.

Verifies that worker.stop() correctly wakes coroutines blocked on:
- _drain_cond.wait() (explicit notify_all)
- _execution_queue.get() (sentinel value)
- asyncio.Event.wait() (task cancellation via CancelledError)

Uses lightweight stubs (no Ray actors) matching the pattern in test_drain_barrier.py.
"""

import asyncio
import types

import pytest

from fluxway.spmd.worker.worker import SPMDWorker as _ActorCls


def _make_worker_stub():
    raw_cls = _ActorCls.__ray_actor_class__
    stub = types.SimpleNamespace(
        _running=True,
        _driver_handle=None,
        _drain_target_count=5,
        _drain_cond=asyncio.Condition(),
        _execution_queue=asyncio.PriorityQueue(),
        _run_task=None,
        _heartbeat_task=None,
        _background_tasks=[],
        spmd_id="test_spmd",
        rank=0,
    )
    stub.stop = types.MethodType(raw_cls.stop, stub)
    return stub


class TestStopWakesDrainCondition:
    """stop() must wake coroutines blocked on _drain_cond.wait()."""

    @pytest.mark.asyncio
    async def test_drain_cond_woken_by_stop(self):
        stub = _make_worker_stub()
        assert stub._drain_target_count == 5

        woke_up = asyncio.Event()

        async def blocked_on_drain():
            async with stub._drain_cond:
                while stub._drain_target_count is not None:
                    await stub._drain_cond.wait()
            woke_up.set()

        task = asyncio.create_task(blocked_on_drain())
        await asyncio.sleep(0.02)
        assert not woke_up.is_set()

        await stub.stop()

        await asyncio.sleep(0.02)
        assert woke_up.is_set()
        assert stub._drain_target_count is None

        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_stop_clears_drain_target(self):
        stub = _make_worker_stub()
        stub._drain_target_count = 42

        await stub.stop()

        assert stub._drain_target_count is None

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self):
        stub = _make_worker_stub()
        assert stub._running is True

        await stub.stop()

        assert stub._running is False


class TestStopSentinel:
    """stop() must put a sentinel into _execution_queue to unblock .get()."""

    @pytest.mark.asyncio
    async def test_sentinel_unblocks_queue_get(self):
        stub = _make_worker_stub()
        received = asyncio.Event()
        sentinel_item = None

        async def blocked_on_queue():
            nonlocal sentinel_item
            sentinel_item = await stub._execution_queue.get()
            received.set()

        task = asyncio.create_task(blocked_on_queue())
        await asyncio.sleep(0.02)
        assert not received.is_set()

        await stub.stop()

        await asyncio.sleep(0.02)
        assert received.is_set()
        assert sentinel_item is not None
        # Sentinel is (999, 0, "__stop__")
        assert sentinel_item[2] == "__stop__"

        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class TestStopCancelsRunTask:
    """stop() must cancel _run_task, which interrupts any await point."""

    @pytest.mark.asyncio
    async def test_run_task_cancelled(self):
        stub = _make_worker_stub()

        was_cancelled = False

        async def fake_run():
            nonlocal was_cancelled
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                was_cancelled = True
                raise

        stub._run_task = asyncio.create_task(fake_run())
        await asyncio.sleep(0.02)

        await stub.stop()

        assert was_cancelled
        assert stub._run_task.done()

    @pytest.mark.asyncio
    async def test_stop_idempotent_with_no_tasks(self):
        stub = _make_worker_stub()
        stub._run_task = None
        stub._heartbeat_task = None

        await stub.stop()
        assert stub._running is False
