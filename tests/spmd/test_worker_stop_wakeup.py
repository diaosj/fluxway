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
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from fluxway.common.allocation import AllocationRegistry
from fluxway.common.coherence import (
    CoherenceDirectory,
    CoherenceInstance,
    CoherenceSubject,
    ExecutionFenceState,
)
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.operator_instances import OperatorInstanceContainer
from fluxway.spmd.worker.coherence_sequencer import CoherenceSequencer
from fluxway.spmd.worker.lifecycle import WorkerExecutionLifecycle
from fluxway.spmd.worker.source_scope import SourceExecutionScope
from fluxway.spmd.worker.worker import LocalInputPrepareRuntime, SPMDWorker as _ActorCls


async def _async_admit(directory: CoherenceDirectory, instance) -> None:
    directory.admit(instance)


def _make_worker_stub():
    raw_cls = _ActorCls.__ray_actor_class__  # type: ignore[reportAttributeAccessIssue]
    directory = CoherenceDirectory()
    stub = types.SimpleNamespace(
        _execution_lifecycle=WorkerExecutionLifecycle(),
        _driver_handle=None,
        _drain_step_count=5,
        _drain_cond=asyncio.Condition(),
        _execution_queue=asyncio.PriorityQueue(),
        _execution_queues={},
        _run_task=None,
        _lane_run_tasks={},
        _heartbeat_task=None,
        _background_tasks=set(),
        _outbound_tasks=set(),
        _send_tasks={},
        _plan_tasks=set(),
        _plan_task_steps={},
        spmd_id="test_spmd",
        rank=0,
        config=types.SimpleNamespace(operator_specs={}),
        _operator_specs={},
        _op_manager=object(),
        _fence_state=ExecutionFenceState.OPEN,
        _coherence_input_tensors={},
        _coherence_output_intents={},
        _coherence_directory=directory,
        _coherence_sequencer=CoherenceSequencer(
            lambda _instances, payload: payload(),
            lambda instance: _async_admit(directory, instance),
        ),
        _allocation_registry=AllocationRegistry(),
        _checkpoint_coordinator=None,
        _checkpoint_repository_root=None,
        _source_scope=SourceExecutionScope(
            acquire=AsyncMock(),
            spawn=asyncio.create_task,
            completed=lambda _operator: 0,
            depth=lambda _operator: 1,
        ),
    )
    stub._execution_lifecycle.topology_ready()
    stub._execution_lifecycle.activate()
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
    stub._operator_instance = types.MethodType(raw_cls._operator_instance, stub)
    stub._operator_instances = types.MethodType(raw_cls._operator_instances, stub)
    stub._shutdown_local_operator_instances = types.MethodType(
        raw_cls._shutdown_local_operator_instances, stub
    )
    stub._shutdown_checkpoint_coordinator = types.MethodType(
        raw_cls._shutdown_checkpoint_coordinator, stub
    )
    stub.stop = types.MethodType(raw_cls.stop, stub)
    stub._shutdown = types.MethodType(raw_cls._shutdown, stub)
    return stub


class TestStopWakesDrainCondition:
    """stop() must wake coroutines blocked on _drain_cond.wait()."""

    @pytest.mark.asyncio
    async def test_drain_cond_woken_by_stop(self):
        stub = _make_worker_stub()
        assert stub._drain_step_count == 5

        woke_up = asyncio.Event()

        async def blocked_on_drain():
            async with stub._drain_cond:
                while stub._drain_step_count is not None:
                    await stub._drain_cond.wait()
            woke_up.set()

        task = asyncio.create_task(blocked_on_drain())
        await asyncio.sleep(0.02)
        assert not woke_up.is_set()

        await stub.stop()

        await asyncio.sleep(0.02)
        assert woke_up.is_set()
        assert stub._drain_step_count is None

        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_stop_clears_drain_target(self):
        stub = _make_worker_stub()
        stub._drain_step_count = 42

        await stub.stop()

        assert stub._drain_step_count is None

    @pytest.mark.asyncio
    async def test_stop_after_prior_coherence_fence_does_not_double_cleanup(self):
        stub = _make_worker_stub()
        instance = CoherenceInstance(CoherenceSubject(1, 0, 101), 0)
        tensor = TransportTensor(np.arange(2, dtype=np.int64))
        tensor._bind_coherence_instances((instance,))
        tensor._bind_consumer_lease_owner(stub._coherence_sequencer)
        stub._coherence_sequencer.issue_intent(instance)
        reservation_ready = asyncio.Event()

        async def retire(_instances, _payload):
            tensor._consumer_lease_reservation_ref = (
                stub._coherence_sequencer.reserve_consumer_lease((instance,))
            )
            reservation_ready.set()
            await asyncio.Event().wait()

        stub._coherence_sequencer._retire_payload = retire
        delivery = asyncio.create_task(
            stub._coherence_sequencer.commit_payload((instance,), object())
        )
        stub._coherence_sequencer.close_intent(instance)
        await reservation_ready.wait()
        stub._coherence_input_tensors = {("op", 0): {tensor}}

        user_operator_failure = RuntimeError("user operator failed")
        stub._coherence_sequencer.fence(user_operator_failure)
        await stub.stop()
        assert tensor._runtime_state.name == "RELEASED"
        assert stub._coherence_sequencer._fenced_error is user_operator_failure
        with pytest.raises(asyncio.CancelledError):
            await delivery

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self):
        stub = _make_worker_stub()
        assert stub._execution_lifecycle.active is True

        await stub.stop()

        assert stub._execution_lifecycle.active is False  # type: ignore[reportUnnecessaryComparison]

    @pytest.mark.asyncio
    async def test_stop_shuts_down_user_callables(self):
        stub = _make_worker_stub()

        await stub.stop()

        shutdown = stub._local_step_plan_executor.user_callable_executor.shutdown
        shutdown.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_stop_cancels_owned_send_tasks(self):
        stub = _make_worker_stub()
        send_task = asyncio.create_task(asyncio.Event().wait())
        stub._send_tasks[send_task] = ("op", 0)

        await stub.stop()

        assert send_task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_reports_send_task_cancellation_cleanup_failure(self):
        stub = _make_worker_stub()

        async def fail_during_cancellation():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                raise RuntimeError("send cancellation cleanup failed") from error

        send_task = asyncio.create_task(fail_during_cancellation())
        stub._send_tasks[send_task] = ("op", 0)

        with pytest.raises(RuntimeError, match="send cancellation cleanup failed"):
            await stub.stop()

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
    async def test_stop_shuts_down_worker_process_operator_once(self):
        stub = _make_worker_stub()
        shutdown_count = 0

        class Operator:
            async def on_shutdown(self) -> None:
                nonlocal shutdown_count
                await asyncio.sleep(0)
                shutdown_count += 1

        operator = Operator()
        operator_meta = types.SimpleNamespace(owner=Operator)
        stub.config = types.SimpleNamespace(
            operator_specs={"step": operator_meta, "flush": operator_meta}
        )
        stub._operator_specs = stub.config.operator_specs
        stub._instance_container = OperatorInstanceContainer()
        stub._instance_container.get_or_create(Operator, lambda: operator)

        await stub.stop()

        assert shutdown_count == 1

    @pytest.mark.asyncio
    async def test_stop_attempts_all_worker_process_operators_on_failure(
        self,
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

        stub.config = types.SimpleNamespace(
            operator_specs={
                "failing": types.SimpleNamespace(owner=FailingOperator),
                "healthy": types.SimpleNamespace(owner=HealthyOperator),
            }
        )
        stub._operator_specs = stub.config.operator_specs
        stub._instance_container = OperatorInstanceContainer()
        stub._instance_container.get_or_create(FailingOperator, FailingOperator)
        stub._instance_container.get_or_create(HealthyOperator, HealthyOperator)

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
        assert stub._execution_lifecycle.active is False
