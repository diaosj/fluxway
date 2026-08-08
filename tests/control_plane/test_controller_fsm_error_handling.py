"""Fail-fast Controller state-machine tests.

Collective SPMD recovery requires checkpointed gang reconstruction. Until that
protocol exists, every driver error must terminate the whole job.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from fluxway.common.types import ControllerEvent, ControllerState, ExecutionFailure
from fluxway.control_plane.state_machine.controller_fsm import ControllerStateMachine


def _make_controller() -> MagicMock:
    controller = MagicMock()
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
    fsm = ControllerStateMachine(controller)
    fsm._state = initial_state
    failure = _failure()

    transitioned = await fsm.trigger_event(
        ControllerEvent.DRIVER_ERROR,
        spmd_id="collective",
        failure=failure,
    )

    assert transitioned is True
    assert fsm.state == ControllerState.FAILED
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
    fsm = ControllerStateMachine(controller)
    fsm._state = ControllerState.RUNNING

    await fsm.trigger_event(
        ControllerEvent.DRIVER_ERROR,
        spmd_id="collective",
        failure=_failure(),
    )

    assert fsm.state == ControllerState.FAILED
    controller._reclaim_all_spmds.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_reset_clears_terminal_error_state() -> None:
    controller = _make_controller()
    fsm = ControllerStateMachine(controller)
    fsm._state = ControllerState.RUNNING
    await fsm.trigger_event(
        ControllerEvent.DRIVER_ERROR,
        spmd_id="collective",
        failure=_failure(),
    )

    await fsm.reset()

    assert fsm.state == ControllerState.INITIALIZING
    assert fsm.get_failures() == {}


@pytest.mark.asyncio
async def test_driver_error_rejects_mismatched_failure_identity() -> None:
    controller = _make_controller()
    fsm = ControllerStateMachine(controller)
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
