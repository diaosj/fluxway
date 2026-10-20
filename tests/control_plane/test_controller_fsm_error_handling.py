"""Fail-fast Controller state-machine tests.

Collective SPMD recovery requires checkpointed gang reconstruction. Until that
protocol exists, every driver error must terminate the whole job.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from fluxway.common.types import ControllerEvent, ControllerState, ExecutionFailure
from fluxway.control_plane.state_machine.controller_fsm import ControllerStateMachine
from fluxway.control_plane.lifecycle import ExecutionLifecycle


def _make_controller() -> MagicMock:
    controller = MagicMock()
    controller._profiler_runtime = SimpleNamespace(enabled=False)
    controller._notify_control_plane_components = AsyncMock()
    controller._execution_lifecycle = ExecutionLifecycle(controller)
    controller._spmd_manager = None
    controller.state_machine = ControllerStateMachine(controller)
    controller._reclaim_all_spmds = AsyncMock()
    controller._release_stopped_spmd_handles = MagicMock()
    controller.stop_checkpointer = AsyncMock()

    async def finalize_runtime(*, export_profiler_trace: bool) -> None:
        assert export_profiler_trace is False
        await controller.stop_checkpointer()
        try:
            await controller._reclaim_all_spmds()
        finally:
            controller._release_stopped_spmd_handles()

    controller._finalize_runtime = AsyncMock(side_effect=finalize_runtime)
    return controller


def _failure(spmd_id: str = "collective") -> ExecutionFailure:
    return ExecutionFailure(
        spmd_id=spmd_id,
        rank=None,
        op_name=None,
        step=None,
        phase="DRIVER",
        exception_type="RuntimeError",
        message="rank lost",
        traceback=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_state",
    (
        ControllerState.PENDING,
        ControllerState.INITIALIZING,
        ControllerState.TOPOLOGY_REGISTERING,
        ControllerState.RUNNING,
    ),
)
async def test_driver_error_fails_job_without_partial_restart(
    initial_state: ControllerState,
    caplog: pytest.LogCaptureFixture,
) -> None:
    controller = _make_controller()
    fsm = controller.state_machine
    fsm._state = initial_state
    failure = _failure()

    transitioned = await fsm.trigger_event(
        ControllerEvent.DRIVER_ERROR,
        spmd_id="collective",
        failure=failure,
    )

    assert transitioned is True
    assert await fsm.wait_until_terminal(2) == ControllerState.FAILED
    assert fsm.get_failures() == {"collective": failure}
    assert fsm.get_failures()["collective"] is failure
    assert not any(
        "Invalid state transition" in record.getMessage() for record in caplog.records
    )
    controller._reclaim_all_spmds.assert_awaited_once_with()
    controller._release_stopped_spmd_handles.assert_called_once_with()


@pytest.mark.asyncio
async def test_failure_reclaim_exception_still_reaches_failed() -> None:
    controller = _make_controller()
    controller._reclaim_all_spmds.side_effect = RuntimeError("reclaim failed")
    fsm = controller.state_machine
    fsm._state = ControllerState.RUNNING

    await fsm.trigger_event(
        ControllerEvent.DRIVER_ERROR,
        spmd_id="collective",
        failure=_failure(),
    )

    assert await fsm.wait_until_terminal(2) == ControllerState.FAILED
    controller._reclaim_all_spmds.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_reset_clears_terminal_error_state() -> None:
    controller = _make_controller()
    fsm = controller.state_machine
    fsm._state = ControllerState.RUNNING
    await fsm.trigger_event(
        ControllerEvent.DRIVER_ERROR,
        spmd_id="collective",
        failure=_failure(),
    )

    await fsm.wait_until_terminal(2)
    await fsm.reset()

    assert fsm.state == ControllerState.INITIALIZING
    assert fsm.get_failures() == {}


@pytest.mark.asyncio
async def test_driver_error_rejects_mismatched_failure_identity() -> None:
    controller = _make_controller()
    fsm = controller.state_machine
    fsm._state = ControllerState.RUNNING

    with pytest.raises(ValueError, match="failure.spmd_id"):
        await fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR,
            spmd_id="collective",
            failure=_failure("other"),
        )

    assert fsm.state == ControllerState.RUNNING
    assert fsm.get_failures() == {}
    controller._reclaim_all_spmds.assert_not_awaited()


def test_controller_has_no_partial_spmd_restart_api() -> None:
    from fluxway.control_plane.controller import ControllerActor

    assert not hasattr(ControllerActor, "restart_spmd")


@pytest.mark.asyncio
async def test_failure_transition_does_not_wait_for_resource_reclaim():
    controller = _make_controller()
    fsm = controller.state_machine
    fsm._state = ControllerState.RUNNING
    entered, release = asyncio.Event(), asyncio.Event()

    async def reclaim():
        entered.set()
        await release.wait()

    controller._reclaim_all_spmds.side_effect = reclaim
    await asyncio.wait_for(
        fsm.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id="collective", failure=_failure()
        ),
        2,
    )
    assert fsm.state is ControllerState.FAILING
    await entered.wait()
    assert not controller._execution_lifecycle._terminal_cleanup.done()
    release.set()
    assert await fsm.wait_until_terminal(2) is ControllerState.FAILED
    controller._reclaim_all_spmds.assert_awaited_once()


@pytest.mark.asyncio
async def test_error_during_completion_changes_outcome_without_duplicate_cleanup():
    controller = _make_controller()
    fsm = controller.state_machine
    fsm._state = ControllerState.RUNNING
    entered, release = asyncio.Event(), asyncio.Event()

    async def finalize(*, export_profiler_trace):
        assert export_profiler_trace
        entered.set()
        await release.wait()

    controller._finalize_runtime.side_effect = finalize
    await fsm.trigger_event(ControllerEvent.ALL_DRIVERS_COMPLETED)
    await entered.wait()
    cleanup = controller._execution_lifecycle._terminal_cleanup
    await fsm.trigger_event(
        ControllerEvent.DRIVER_ERROR, spmd_id="collective", failure=_failure()
    )
    assert fsm.state is ControllerState.FAILING
    assert controller._execution_lifecycle._terminal_cleanup is cleanup
    release.set()
    assert await fsm.wait_until_terminal(2) is ControllerState.FAILED
    controller._finalize_runtime.assert_awaited_once()


@pytest.mark.asyncio
async def test_terminal_notification_failure_does_not_skip_resource_cleanup():
    controller = _make_controller()
    fsm = controller.state_machine
    fsm._state = ControllerState.RUNNING

    async def notify(state):
        if state is ControllerState.FAILING:
            raise RuntimeError("component notification failed")

    controller._notify_control_plane_components.side_effect = notify
    await fsm.trigger_event(
        ControllerEvent.DRIVER_ERROR, spmd_id="collective", failure=_failure()
    )
    assert await fsm.wait_until_terminal(2) is ControllerState.FAILED
    controller._finalize_runtime.assert_awaited_once()
    assert fsm.get_failures()["controller"].message == "component notification failed"
