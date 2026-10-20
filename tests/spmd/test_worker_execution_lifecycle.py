import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from fluxway.common.types import (
    ControllerEvent,
    ControllerState,
    DrainBoundary,
    DriverEvent,
    DriverState,
    WorkerHeartbeat,
    WorkerStatus,
)
from fluxway.spmd.worker.checkpoint_activity import CheckpointActivity
from fluxway.spmd.worker.lifecycle import WorkerExecutionLifecycle, WorkerPhase
from fluxway.spmd.worker.source_scope import SourceExecutionScope
from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.control_plane.errors import ExecutionFailedError
from tests.control_plane.test_execution_lifecycle import harness
from tests.spmd.test_driver_failure_channel import _driver


def fresh_worker():
    worker = object.__new__(SPMDWorker.__ray_metadata__.modified_class)
    worker.spmd_id = "proof"
    worker.rank = 0
    worker._execution_lifecycle = WorkerExecutionLifecycle()
    worker._expected_predecessor_count = None
    worker._expected_successor_count = None
    worker._predecessor_handlers = {}
    worker._successor_handlers = {}
    worker._operator_send_context_cache = {}
    worker._mailbox_ready_steps = {}
    worker._completed_step_count = {"process": 0}
    worker._scheduled_step_count = {"process": 0}
    worker._failure = None
    worker.config = SimpleNamespace(auto_start_source_operators=False)
    worker._profiler_runtime = SimpleNamespace(
        wrap_data_plane_handlers=lambda _, value: value
    )
    worker._source_scope = SourceExecutionScope(
        acquire=worker._acquire_source_work,
        spawn=worker._spawn_background_task,
        completed=lambda name: worker._completed_step_count[name],
        depth=lambda _: 1,
    )
    return worker


@pytest.mark.parametrize("order", ["counts-first", "handlers-first", "isolated"])
@pytest.mark.asyncio
async def test_topology_prepares_restore_without_authorizing_execution(order):
    worker = fresh_worker()
    if order == "counts-first":
        worker.set_expected_topology(1, 1)
    if order != "isolated":
        worker.register_predecessor_handlers("pred", {})
        worker.register_successor_handlers("succ", {})
    if order == "handlers-first":
        worker.set_expected_topology(1, 1)
    if order == "isolated":
        worker.set_expected_topology(0, 0)
    assert worker._execution_lifecycle.phase is WorkerPhase.READY
    assert not worker.is_running()
    worker._restore_checkpoint_runtime_state(
        {"drain_boundary": DrainBoundary(committed_cut=17).to_dict()}
    )
    worker.activate_execution()
    worker.activate_execution()
    assert worker.is_running()
    assert worker._scheduled_step_count == {"process": 17}
    with pytest.raises(RuntimeError, match="fresh Worker"):
        worker._restore_checkpoint_runtime_state({})
    await worker._execution_lifecycle.close(lambda: asyncio.sleep(0))
    with pytest.raises(RuntimeError, match="activation requires READY"):
        worker.activate_execution()


@pytest.mark.asyncio
async def test_pending_restore_preserves_heartbeat_progress_and_excludes_activation():
    worker = fresh_worker()
    worker.set_expected_topology(0, 0)
    entered, release = asyncio.Event(), asyncio.Event()

    async def restore(*_):
        entered.set()
        await release.wait()

    worker._drain_outbound_tasks = AsyncMock()
    worker._get_checkpoint_coordinator = lambda _: SimpleNamespace(restore=restore)
    worker._driver_handle = MagicMock()
    worker._checkpoint_activity = CheckpointActivity()
    pending = asyncio.create_task(
        worker.checkpoint_restore(
            "repository",
            "checkpoint",
            None,
            None,
            None,
            {"drain_boundary": DrainBoundary(committed_cut=17).to_dict()},
            {},
        )
    )
    await asyncio.wait_for(entered.wait(), 1)
    try:
        with pytest.raises(RuntimeError, match="activation requires READY"):
            worker.activate_execution()
        await asyncio.wait_for(worker._send_heartbeat(), 1)
        heartbeat = (
            worker._driver_handle.receive_worker_heartbeat.remote.call_args.args[1]
        )
        assert heartbeat.status is WorkerStatus.RESTORING
        assert not pending.done()
    finally:
        release.set()
        await pending
    worker.activate_execution()
    assert worker.is_running()


def test_ready_heartbeat_does_not_authorize_driver_start():
    driver, _ = _driver()
    driver.state_machine.reset()
    driver.receive_worker_heartbeat(
        0,
        WorkerHeartbeat(
            rank=0,
            timestamp=time.time(),
            status=WorkerStatus.READY,
        ),
    )
    assert driver._topology_ready
    assert driver.state_machine.state is DriverState.READY


@pytest.mark.asyncio
async def test_driver_activation_reserves_phase_without_locking_heartbeat():
    driver, _ = _driver()
    driver.state_machine.reset()
    driver.state_machine.require_transition(DriverEvent.INIT_COMPLETE)
    entered, release = asyncio.Event(), asyncio.Event()

    async def activate():
        entered.set()
        await release.wait()

    driver.executor = SimpleNamespace(activate_execution=activate)
    driver._notify_driver_components = AsyncMock()
    task = asyncio.create_task(driver.activate_execution())
    await asyncio.wait_for(entered.wait(), 1)
    try:
        assert driver.state_machine.state is DriverState.ACTIVATING
        assert not driver.state_machine.can_transition(
            DriverState.ACTIVATING, DriverEvent.RESTORE_REQUESTED
        )
        driver.receive_worker_heartbeat(
            0,
            WorkerHeartbeat(
                rank=0,
                timestamp=time.time(),
                status=WorkerStatus.READY,
            ),
        )
        assert driver._topology_ready
        assert not task.done()
    finally:
        release.set()
        await task
    assert driver.state_machine.state is DriverState.RUNNING


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event,outcome",
    [
        (ControllerEvent.ALL_DRIVERS_COMPLETED, ControllerState.COMPLETED),
        (ControllerEvent.CANCEL, ControllerState.CANCELLED),
    ],
)
async def test_terminal_cleanup_survives_waiter_cancel_and_is_shared(event, outcome):
    controller, _, _, _ = harness(1)
    controller.state_machine._state = ControllerState.RUNNING
    started, release = asyncio.Event(), asyncio.Event()

    async def finalize(**_):
        started.set()
        await release.wait()

    controller._finalize_runtime = AsyncMock(side_effect=finalize)
    await controller._execution_lifecycle.transition(event)
    await asyncio.wait_for(started.wait(), 1)
    waiter = asyncio.create_task(controller.stop())
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    assert not controller._execution_lifecycle._terminal_cleanup.cancelled()
    release.set()
    await asyncio.gather(controller.stop(), controller.stop())
    assert await controller.state_machine.wait_until_terminal(1) is outcome
    controller._finalize_runtime.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_failure_is_observable_to_every_close_waiter():
    controller, _, _, _ = harness(1)
    controller.state_machine._state = ControllerState.RUNNING
    controller._finalize_runtime = AsyncMock(side_effect=RuntimeError("reclaim failed"))
    results = await asyncio.gather(
        controller.stop(), controller.stop(), return_exceptions=True
    )
    assert all(
        isinstance(result, Exception) and "reclaim failed" in str(result)
        for result in results
    )
    assert controller.state_machine.state is ControllerState.FAILED
    controller._finalize_runtime.assert_awaited_once()


@pytest.mark.asyncio
async def test_failure_settles_pending_startup_before_reclamation():
    controller, _, _, _ = harness(1)
    controller.state_machine._state = ControllerState.PENDING
    controller.initialize = AsyncMock()
    controller._ready_event.clear()
    startup = asyncio.create_task(controller.start())
    await asyncio.sleep(0)
    await controller._execution_lifecycle.fail(
        RuntimeError("startup failed"), "STARTUP"
    )
    assert (
        await controller.state_machine.wait_until_terminal(1) is ControllerState.FAILED
    )
    (result,) = await asyncio.gather(startup, return_exceptions=True)
    assert isinstance(result, ExecutionFailedError)
    assert "startup failed" in str(result)
    controller._finalize_runtime.assert_awaited_once()


@pytest.mark.asyncio
async def test_driver_stop_excludes_late_activation_and_shares_shutdown():
    driver, _ = _driver()
    driver.state_machine.reset()
    driver.state_machine.require_transition(DriverEvent.INIT_COMPLETE)
    entered, release = asyncio.Event(), asyncio.Event()

    async def activate():
        entered.set()
        await release.wait()

    driver.executor = SimpleNamespace(
        activate_execution=activate,
        stop_workers=AsyncMock(return_value=[]),
    )
    driver.heartbeat_manager.stop = AsyncMock()
    driver._wait_for_driver_component_notifications = AsyncMock()
    driver._notify_driver_components = AsyncMock()
    activation = asyncio.create_task(driver.activate_execution())
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.gather(driver.stop(), driver.stop())
    release.set()
    with pytest.raises(RuntimeError):
        await activation
    assert driver.state_machine.state is DriverState.STOPPED
    driver.executor.stop_workers.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_close_retains_one_release_operation_after_waiter_cancel():
    lifecycle = WorkerExecutionLifecycle()
    lifecycle.topology_ready()
    lifecycle.activate()
    entered, release = asyncio.Event(), asyncio.Event()

    async def release_resources():
        entered.set()
        await release.wait()

    close = AsyncMock(side_effect=release_resources)
    waiter = asyncio.create_task(lifecycle.close(close))
    await asyncio.wait_for(entered.wait(), 1)
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    with pytest.raises(RuntimeError, match="activation requires READY"):
        lifecycle.activate()
    release.set()
    await asyncio.gather(lifecycle.close(close), lifecycle.close(close))
    assert lifecycle.phase is WorkerPhase.CLOSED
    close.assert_awaited_once()


@pytest.mark.asyncio
async def test_terminal_notification_failure_cannot_publish_success():
    controller, _, _, _ = harness(1)
    controller.state_machine._state = ControllerState.RUNNING

    async def notify(state):
        if state is ControllerState.COMPLETED:
            raise RuntimeError("terminal observer failed")

    controller._notify_control_plane_components = notify
    await controller._execution_lifecycle.transition(
        ControllerEvent.ALL_DRIVERS_COMPLETED
    )
    assert (
        await controller.state_machine.wait_until_terminal(1) is ControllerState.FAILED
    )
    with pytest.raises(ExecutionFailedError, match="terminal observer failed"):
        await controller.stop()


@pytest.mark.asyncio
async def test_closed_execution_cannot_start_again():
    controller, _, _, _ = harness(1)
    await controller.stop()
    with pytest.raises(RuntimeError, match="startup requires a live execution"):
        await controller._execution_lifecycle.start(None)
