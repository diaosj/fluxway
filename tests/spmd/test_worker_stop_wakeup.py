"""
Unit tests for SPMDWorker.stop() drain wakeup mechanism.

Verifies that worker.stop() correctly wakes coroutines blocked on:
- _drain_cond.wait() (explicit notify_all)
- _execution_queue.get() (sentinel value)
- asyncio.Event.wait() (task cancellation via CancelledError)

Uses lightweight stubs (no Ray actors) matching the pattern in test_drain_barrier.py.
"""

import asyncio
import threading
import types
import typing
from unittest.mock import AsyncMock, Mock

import pytest

from fluxway.spmd.worker.worker import LocalInputPrepareRuntime, SPMDWorker as _ActorCls


def _make_worker_stub():
    raw_cls = _ActorCls.__ray_actor_class__  # type: ignore[reportAttributeAccessIssue]
    stub = types.SimpleNamespace(
        _running=True,
        _driver_handle=None,
        _drain_target_count=typing.cast(int | None, 5),
        _drain_cond=asyncio.Condition(),
        _execution_queue=asyncio.PriorityQueue(),
        _execution_queues={},
        _run_task=None,
        _lane_run_tasks={},
        _heartbeat_task=None,
        _background_tasks=set(),
        _outbound_tasks=set(),
        _send_tasks=set(),
        _plan_tasks=set(),
        spmd_id="test_spmd",
        rank=0,
        config=types.SimpleNamespace(operator_metas={}),
    )
    stub._execution_queues = {"__default__": stub._execution_queue}
    stub._input_prepare_runtime = LocalInputPrepareRuntime(
        background_tasks=stub._background_tasks,
        on_task_done=lambda _task: None,
        schedule_key=lambda op_name, step: (0, 0, step, op_name),
    )
    stub._delivery_runtime = types.SimpleNamespace(cancel_pending=AsyncMock())
    stub._worker_process_call_runtime = types.SimpleNamespace(
        cancel_pending=AsyncMock(),
        close=lambda: None,
    )
    stub._local_step_plan_executor = types.SimpleNamespace(
        user_callable_executor=types.SimpleNamespace(shutdown=AsyncMock())
    )
    stub._actor_owned_operator_names = lambda: set()
    stub._shutdown_local_operator_instances = types.MethodType(
        raw_cls._shutdown_local_operator_instances, stub
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
                while stub._drain_target_count is not None:  # type: ignore[reportUnnecessaryComparison]
                    await stub._drain_cond.wait()
            woke_up.set()

        task = asyncio.create_task(blocked_on_drain())
        await asyncio.sleep(0.02)
        assert not woke_up.is_set()

        await stub.stop()

        await asyncio.sleep(0.02)
        assert woke_up.is_set()
        assert stub._drain_target_count is None  # type: ignore[reportUnnecessaryComparison]

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

        assert stub._drain_target_count is None  # type: ignore[reportUnnecessaryComparison]

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self):
        stub = _make_worker_stub()
        assert stub._running is True

        await stub.stop()

        assert stub._running is False  # type: ignore[reportUnnecessaryComparison]

    @pytest.mark.asyncio
    async def test_stop_shuts_down_user_callables(self):
        stub = _make_worker_stub()

        await stub.stop()

        shutdown = stub._local_step_plan_executor.user_callable_executor.shutdown
        shutdown.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_stop_closes_local_runtimes_when_user_shutdown_fails(self):
        stub = _make_worker_stub()
        shutdown = stub._local_step_plan_executor.user_callable_executor.shutdown
        shutdown.side_effect = RuntimeError("metrics drain failed")
        stub._input_prepare_runtime.close = Mock()
        stub._worker_process_call_runtime.close = Mock()

        with pytest.raises(RuntimeError, match="metrics drain failed"):
            await stub.stop()

        stub._input_prepare_runtime.close.assert_called_once_with()
        stub._worker_process_call_runtime.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_stop_runs_user_shutdown_when_cancel_pending_fails(self):
        stub = _make_worker_stub()
        stub._input_prepare_runtime.cancel_pending = AsyncMock(
            side_effect=RuntimeError("cancel failed")
        )
        shutdown = stub._local_step_plan_executor.user_callable_executor.shutdown
        stub._input_prepare_runtime.close = Mock()
        stub._worker_process_call_runtime.close = Mock()

        with pytest.raises(RuntimeError, match="cancel failed"):
            await stub.stop()

        shutdown.assert_awaited_once_with()
        stub._input_prepare_runtime.close.assert_called_once_with()
        stub._worker_process_call_runtime.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_stop_shuts_down_worker_process_operator_once(self, monkeypatch):
        stub = _make_worker_stub()
        shutdown_count = 0

        class Operator:
            async def on_shutdown(self) -> None:
                nonlocal shutdown_count
                await asyncio.sleep(0)
                shutdown_count += 1

        operator = Operator()
        operator_meta = types.SimpleNamespace(
            owner_class_module=Operator.__module__,
            owner_class_qualname=Operator.__qualname__,
        )
        stub.config = types.SimpleNamespace(
            operator_metas={"step": operator_meta, "flush": operator_meta}
        )
        stub._actor_owned_operator_names = lambda: set()
        monkeypatch.setattr(
            "fluxway.spmd.worker.worker._resolve_owner_class",
            lambda *_: Operator,
        )
        monkeypatch.setattr(
            "fluxway.spmd.worker.worker.get_instance_storage",
            lambda: ({id(Operator): operator}, threading.Lock()),
        )

        await stub.stop()

        assert shutdown_count == 1

    @pytest.mark.asyncio
    async def test_stop_attempts_all_worker_process_operators_on_failure(
        self, monkeypatch
    ):
        stub = _make_worker_stub()
        calls = []

        class FailingOperator:
            def on_shutdown(self) -> None:
                calls.append("failing")
                raise RuntimeError("local shutdown failed")

        class HealthyOperator:
            def on_shutdown(self) -> None:
                calls.append("healthy")

        classes = {
            "FailingOperator": FailingOperator,
            "HealthyOperator": HealthyOperator,
        }
        instances = {
            id(FailingOperator): FailingOperator(),
            id(HealthyOperator): HealthyOperator(),
        }
        stub.config = types.SimpleNamespace(
            operator_metas={
                name: types.SimpleNamespace(
                    owner_class_module="test",
                    owner_class_qualname=name,
                )
                for name in classes
            }
        )
        stub._actor_owned_operator_names = lambda: set()
        monkeypatch.setattr(
            "fluxway.spmd.worker.worker._resolve_owner_class",
            lambda _, qualname: classes[qualname],
        )
        monkeypatch.setattr(
            "fluxway.spmd.worker.worker.get_instance_storage",
            lambda: (instances, threading.Lock()),
        )

        with pytest.raises(RuntimeError, match="local shutdown failed"):
            await stub.stop()

        assert calls == ["failing", "healthy"]


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
        assert sentinel_item == (999, 0, 0, "__stop__")

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
