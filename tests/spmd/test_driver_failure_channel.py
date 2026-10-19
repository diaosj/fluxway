from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from fluxway import OperatorSpec
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types import (
    DriverEvent,
    DriverStatus,
    DriverState,
    ExecutionFailure,
    WorkerHeartbeat,
    WorkerStatus,
)
from fluxway.spmd.driver.driver import SPMDDriver
from fluxway.spmd.driver.state_machine.driver_fsm import DriverStateMachine
from tests.compiled_graph_fixtures import single_spmd_config


def _failure(
    *,
    rank: int | None = 0,
    phase: str = "LOCAL_STEP_PLAN",
    exception_type: str = "RuntimeError",
    message: str = "worker failed",
) -> ExecutionFailure:
    return ExecutionFailure(
        spmd_id="dense",
        rank=rank,
        op_name="train" if rank is not None else None,
        step=3 if rank is not None else None,
        phase=phase,
        exception_type=exception_type,
        message=message,
        traceback="traceback",
    )


def _running_fsm() -> tuple[DriverStateMachine, MagicMock]:
    driver = MagicMock()
    driver.spmd_id = "dense"
    driver._submit_driver_effect.side_effect = (
        lambda operation, args=(), **_: operation(*args)
    )
    fsm = DriverStateMachine(driver)
    assert fsm.trigger(DriverEvent.INIT_COMPLETE)
    assert fsm.trigger(DriverEvent.START)
    assert fsm.trigger(DriverEvent.ACTIVATION_COMPLETE)
    assert fsm.state is DriverState.RUNNING
    return fsm, driver


def _driver() -> tuple[Any, MagicMock]:
    controller = MagicMock()
    config = single_spmd_config(
        name="driver-failure-channel",
        spmd_id="dense",
        mesh=DeviceMesh(("worker",), (1,)),
        operators={"train": OperatorSpec(name="train")},
    )
    driver = SPMDDriver.__ray_actor_class__(  # type: ignore[reportAttributeAccessIssue]
        spmd_id="dense",
        config=config,
        placement_group=object(),
        worker_bundle_indices=[0],
        controller_handle=controller,
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a"],
    )
    assert driver.state_machine.trigger(DriverEvent.INIT_COMPLETE)
    assert driver.state_machine.trigger(DriverEvent.START)
    assert driver.state_machine.trigger(DriverEvent.ACTIVATION_COMPLETE)
    driver._submit_driver_effect = lambda operation, args=(), **_: operation(*args)
    return driver, controller


def test_worker_error_preserves_structured_failure_as_the_only_error_fact() -> None:
    fsm, driver = _running_fsm()
    failure = _failure()

    assert fsm.trigger(DriverEvent.WORKER_ERROR, failure=failure)

    assert fsm.state is DriverState.ERROR
    assert fsm.failure is failure
    assert fsm.get_worker_failures() == {0: failure}
    assert not hasattr(fsm, "get_error_message")
    driver._report_error.assert_called_once_with(failure)


def test_rejected_worker_failure_does_not_mutate_fsm_error_facts() -> None:
    fsm, _ = _running_fsm()

    with pytest.raises(ValueError, match="requires a worker rank"):
        fsm.trigger(DriverEvent.WORKER_ERROR, failure=_failure(rank=None))

    assert fsm.state is DriverState.RUNNING
    assert fsm.failure is None
    assert fsm.get_worker_failures() == {}

    failure = _failure()
    assert fsm.trigger(DriverEvent.WORKER_ERROR, failure=failure)
    assert fsm.failure is failure


@pytest.mark.parametrize(
    "initial_state",
    (DriverState.INITIALIZING, DriverState.READY),
)
def test_worker_failure_fails_driver_before_running(
    initial_state: DriverState,
) -> None:
    driver = MagicMock()
    driver.spmd_id = "dense"
    driver._submit_driver_effect.side_effect = (
        lambda operation, args=(), **_: operation(*args)
    )
    fsm = DriverStateMachine(driver)
    if initial_state is DriverState.READY:
        assert fsm.trigger(DriverEvent.INIT_COMPLETE)
    assert fsm.state is initial_state
    failure = _failure()

    assert fsm.trigger(DriverEvent.WORKER_ERROR, failure=failure)

    assert fsm.state is DriverState.ERROR
    assert fsm.failure is failure
    driver._report_error.assert_called_once_with(failure)


def test_driver_receive_worker_failure_reports_the_same_structured_fact() -> None:
    driver, controller = _driver()
    failure = _failure()

    driver.receive_worker_failure(0, failure)

    assert driver.state_machine.state is DriverState.ERROR
    assert driver.state_machine.failure is failure
    assert driver.state_machine.get_worker_failures() == {0: failure}
    controller.report_spmd_error.remote.assert_called_once_with("dense", failure)


@pytest.mark.asyncio
async def test_worker_failure_notifies_registered_driver_components() -> None:
    driver, _ = _driver()
    component = MagicMock()
    component.component_id = "dense:train"
    component.on_state = AsyncMock()
    component.checkpoint_state.return_value = {}
    driver.driver_components.register(component)

    driver.receive_worker_failure(0, _failure())
    await driver._wait_for_driver_component_notifications()

    component.on_state.assert_awaited_once_with(DriverState.ERROR)


@pytest.mark.asyncio
async def test_driver_retains_error_component_task_until_lifecycle_join() -> None:
    driver, _ = _driver()
    release = asyncio.Event()
    component = MagicMock()
    component.component_id = "dense:train"

    async def on_state(state):
        assert state is DriverState.ERROR
        await release.wait()

    component.on_state = on_state
    component.checkpoint_state.return_value = {}
    driver.driver_components.register(component)

    driver.receive_worker_failure(0, _failure())
    await asyncio.sleep(0)
    assert len(driver._driver_component_notification_tasks) == 1
    release.set()
    await driver._wait_for_driver_component_notifications()
    assert driver._driver_component_notification_tasks == set()


def test_worker_failure_propagates_controller_report_submission_error() -> None:
    driver, controller = _driver()
    failure = _failure()
    controller.report_spmd_error.remote.side_effect = RuntimeError(
        "controller unavailable"
    )

    with pytest.raises(RuntimeError, match="controller unavailable"):
        driver.receive_worker_failure(0, failure)

    assert driver.state_machine.state is DriverState.ERROR
    assert driver.state_machine.failure is failure


def test_driver_status_exposes_structured_failures() -> None:
    driver, _ = _driver()
    failure = _failure()
    driver.receive_worker_failure(0, failure)

    status = driver.get_status()

    assert status["failure"] is failure
    assert status["worker_failures"] == {0: failure}
    assert status["worker_failures"][0] is failure
    assert "error_workers" not in status


def test_worker_heartbeat_failure_uses_the_structured_failure_channel() -> None:
    driver, controller = _driver()
    failure = _failure()
    heartbeat = WorkerHeartbeat(
        rank=0,
        timestamp=1.0,
        status=WorkerStatus.ERROR,
        failure=failure,
    )

    driver.receive_worker_heartbeat(0, heartbeat)

    assert driver.state_machine.state is DriverState.ERROR
    assert driver.state_machine.failure is failure
    assert driver._worker_heartbeats == {0: heartbeat}
    assert not driver._workers_ready
    controller.report_spmd_error.remote.assert_called_once_with("dense", failure)


def test_driver_heartbeat_exposes_failure_only_in_error_state() -> None:
    driver, controller = _driver()
    driver._worker_heartbeat_times[0] = time.time()

    driver.heartbeat_manager._send_heartbeat()

    healthy = controller.receive_driver_heartbeat.remote.call_args.args[0]
    assert healthy.status is DriverStatus.HEALTHY
    assert healthy.failure is None

    failure = _failure()
    driver.receive_worker_failure(0, failure)
    driver.heartbeat_manager._send_heartbeat()

    failed = controller.receive_driver_heartbeat.remote.call_args.args[0]
    assert failed.status is DriverStatus.ERROR
    assert failed.failure is failure


def test_running_worker_heartbeat_timeout_fails_immediately() -> None:
    driver, controller = _driver()
    driver._worker_heartbeat_times[0] = (
        time.time() - driver.heartbeat_manager._heartbeat_config.heartbeat_timeout - 1.0
    )

    driver.heartbeat_manager._send_heartbeat()

    failure = driver.state_machine.failure
    assert failure is not None
    assert failure.spmd_id == "dense"
    assert failure.rank == 0
    assert failure.phase == "WORKER_HEARTBEAT"
    assert failure.exception_type == "WorkerHeartbeatTimeout"
    assert driver.state_machine.state is DriverState.ERROR
    controller.report_spmd_error.remote.assert_called_once_with("dense", failure)
    heartbeat = controller.receive_driver_heartbeat.remote.call_args.args[0]
    assert heartbeat.status is DriverStatus.ERROR
    assert heartbeat.failure is failure


def test_snapshotting_worker_heartbeat_timeout_does_not_fail_driver() -> None:
    driver, controller = _driver()
    assert driver.state_machine.trigger(DriverEvent.DRAIN_REQUESTED)
    assert driver.state_machine.trigger(DriverEvent.DRAIN_COMPLETE)
    assert driver.state_machine.trigger(DriverEvent.SNAPSHOT_REQUESTED)
    assert driver.state_machine.state is DriverState.SNAPSHOTTING
    driver._worker_heartbeat_times[0] = (
        time.time() - driver.heartbeat_manager._heartbeat_config.heartbeat_timeout - 1.0
    )

    driver.heartbeat_manager._send_heartbeat()

    assert driver.state_machine.failure is None
    assert driver.state_machine.state is DriverState.SNAPSHOTTING
    controller.report_spmd_error.remote.assert_not_called()


@pytest.mark.parametrize(
    "live_state",
    (DriverState.READY, DriverState.BLOCKED),
)
def test_worker_heartbeat_timeout_fails_other_live_states(
    live_state: DriverState,
) -> None:
    driver, controller = _driver()
    if live_state is DriverState.READY:
        driver.state_machine.reset()
        assert driver.state_machine.trigger(DriverEvent.INIT_COMPLETE)
    else:
        assert driver.state_machine.trigger(DriverEvent.BLOCK_REQUESTED)
    assert driver.state_machine.state is live_state
    driver._worker_heartbeat_times[0] = (
        time.time() - driver.heartbeat_manager._heartbeat_config.heartbeat_timeout - 1.0
    )

    driver.heartbeat_manager._send_heartbeat()

    failure = driver.state_machine.failure
    assert failure is not None
    assert failure.phase == "WORKER_HEARTBEAT"
    assert failure.exception_type == "WorkerHeartbeatTimeout"
    assert driver.state_machine.state is DriverState.ERROR
    controller.report_spmd_error.remote.assert_called_once_with("dense", failure)


def test_initializing_worker_heartbeat_age_is_governed_by_init_timeout() -> None:
    driver, controller = _driver()
    driver.state_machine.reset()
    driver._worker_heartbeat_times[0] = (
        time.time() - driver.heartbeat_manager._heartbeat_config.heartbeat_timeout - 1.0
    )

    driver.heartbeat_manager._send_heartbeat()

    assert driver.state_machine.state is DriverState.INITIALIZING
    assert driver.state_machine.failure is None
    controller.report_spmd_error.remote.assert_not_called()
    heartbeat = controller.receive_driver_heartbeat.remote.call_args.args[0]
    assert heartbeat.status is DriverStatus.WARNING
    assert heartbeat.failure is None


@pytest.mark.asyncio
async def test_driver_initialization_exception_preserves_exception_context(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "ray.get_runtime_context", lambda: MagicMock(current_actor=None)
    )
    driver, controller = _driver()
    driver.state_machine.reset()
    driver.executor.start_workers = AsyncMock(side_effect=RuntimeError("startup boom"))

    with pytest.raises(RuntimeError, match="startup boom"):
        await driver.initialize()

    failure = driver.state_machine.failure
    assert failure is not None
    assert failure.spmd_id == "dense"
    assert failure.rank is None
    assert failure.phase == "DRIVER_INITIALIZATION"
    assert failure.exception_type == "RuntimeError"
    assert failure.message == "startup boom"
    assert failure.traceback is not None
    assert "RuntimeError: startup boom" in failure.traceback
    controller.report_spmd_error.remote.assert_called_once_with("dense", failure)


def test_driver_initialization_timeout_is_a_named_structured_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, controller = _driver()
    driver.state_machine.reset()
    driver.heartbeat_manager._init_start_time = (
        time.time() - driver.heartbeat_manager._init_timeout - 1.0
    )
    driver.heartbeat_manager._heartbeat_running = True
    driver.heartbeat_manager._send_heartbeat = MagicMock()
    monkeypatch.setattr(
        "fluxway.spmd.driver.heartbeat.time.sleep",
        lambda _interval: setattr(
            driver.heartbeat_manager, "_heartbeat_running", False
        ),
    )

    driver.heartbeat_manager._heartbeat_loop()

    failure = driver.state_machine.failure
    assert failure is not None
    assert failure.phase == "DRIVER_INITIALIZATION"
    assert failure.exception_type == "WorkerInitializationTimeout"
    assert "0/1 workers reported" in failure.message
    assert failure.traceback is None
    controller.report_spmd_error.remote.assert_called_once_with("dense", failure)


def test_worker_heartbeat_reports_readiness_without_starting_execution() -> None:
    driver, _ = _driver()
    driver.state_machine._state = DriverState.READY
    driver.receive_worker_heartbeat(
        0, WorkerHeartbeat(rank=0, timestamp=1.0, status=WorkerStatus.RUNNING)
    )
    assert driver._workers_ready
    assert driver._topology_ready
    assert driver.state_machine.state is DriverState.READY
