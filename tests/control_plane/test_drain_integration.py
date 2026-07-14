import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from fluxway.common.types import ControllerEvent
from fluxway.common.types import ControllerState
from fluxway.common.types.drain import DrainStrategy
from fluxway.control_plane.controller import ControllerActor


class MockDrainStrategy(DrainStrategy):
    def get_initial_drain_target(self, baseline: int = 0):
        return baseline + 2

    def get_next_drain_target(self, current_completed_step: int):
        return current_completed_step + 2


@pytest.fixture
def mock_spmd_manager():
    mgr = MagicMock()
    mgr.get_all_spmd_ids.return_value = ["op1", "op2"]

    mock_driver1 = AsyncMock()
    mock_driver2 = AsyncMock()

    def get_driver(spmd_id):
        if spmd_id == "op1":
            return mock_driver1
        return mock_driver2

    mgr.get_driver_handle = get_driver
    mgr.mock_driver1 = mock_driver1
    mgr.mock_driver2 = mock_driver2
    return mgr


@pytest.fixture
def controller(mock_spmd_manager):
    # Setup partial graph stub
    graph_stub = MagicMock()
    graph_stub.nodes.keys.return_value = ["op1", "op2"]

    ctrl = ControllerActor.__ray_actor_class__(  # type: ignore[attr-defined]
        graph=graph_stub,
        placement_recipe=MagicMock(),
    )
    ctrl._spmd_manager = mock_spmd_manager
    ctrl._graph = graph_stub

    # inject mock strategy and checkpointer
    ctrl._drain_strategy = MockDrainStrategy()
    ctrl._checkpointer = AsyncMock()

    return ctrl


@pytest.mark.asyncio
async def test_controller_orchestrates_drain(controller, mock_spmd_manager):
    controller._last_requested_drain_target = 0
    controller._drained_drivers = set()

    # Initial drain issued
    await controller._issue_initial_drain()
    mock_spmd_manager.mock_driver1.request_drain.remote.assert_called_with(2)
    mock_spmd_manager.mock_driver2.request_drain.remote.assert_called_with(2)
    assert controller._last_requested_drain_target == 2

    # Driver 1 reports drained
    await controller.report_driver_drained("op1")
    controller._checkpointer.save.assert_not_called()  # op2 not yet drained

    # Driver 2 reports drained, completing the set and triggering checkpoint.
    await controller.report_driver_drained("op2")

    # We call this manually simulating the FSM dispatching it
    await controller._on_all_drivers_drained()
    await controller._checkpoint_task

    controller._checkpointer.save.assert_called_with(2)
    mock_spmd_manager.mock_driver1.resume.remote.assert_called()
    mock_spmd_manager.mock_driver2.resume.remote.assert_called()

    # Verifies next target issued
    mock_spmd_manager.mock_driver1.request_drain.remote.assert_called_with(4)


@pytest.mark.asyncio
async def test_stop_checkpointer_skips_self_await(controller):
    controller._checkpoint_task = asyncio.current_task()

    await controller.stop_checkpointer()

    assert controller._checkpoint_task is None


@pytest.mark.asyncio
async def test_all_drivers_completed_transitions_through_completing(controller):
    cleanup_gate = asyncio.Event()
    controller._state_machine._state = ControllerState.RUNNING
    controller._cleanup_all_spmds = AsyncMock(side_effect=cleanup_gate.wait)
    controller.stop_checkpointer = AsyncMock()
    controller._release_stopped_spmd_handles = MagicMock()

    transition_task = asyncio.create_task(
        controller._state_machine.trigger_event(ControllerEvent.ALL_DRIVERS_COMPLETED)
    )
    await asyncio.sleep(0)
    assert controller._state_machine.state == ControllerState.COMPLETING
    assert not transition_task.done()

    cleanup_gate.set()

    assert await transition_task is True
    assert controller._state_machine.state == ControllerState.COMPLETED
    controller._release_stopped_spmd_handles.assert_called_once_with()


@pytest.mark.asyncio
async def test_wait_for_completion_observes_completed_after_cleanup(controller):
    cleanup_gate = asyncio.Event()
    controller._state_machine._state = ControllerState.RUNNING
    controller._cleanup_all_spmds = AsyncMock(side_effect=cleanup_gate.wait)
    controller.stop_checkpointer = AsyncMock()
    controller._release_stopped_spmd_handles = MagicMock()

    transition_task = asyncio.create_task(
        controller._state_machine.trigger_event(ControllerEvent.ALL_DRIVERS_COMPLETED)
    )
    wait_task = asyncio.create_task(controller.wait_for_completion(timeout=1.0))
    await asyncio.sleep(0)

    assert controller._state_machine.state == ControllerState.COMPLETING
    assert not wait_task.done()

    cleanup_gate.set()

    assert await transition_task is True
    assert await wait_task is True
    controller._release_stopped_spmd_handles.assert_called_once_with()


@pytest.mark.asyncio
async def test_shutdown_hook_failure_marks_controller_failed(controller):
    controller._state_machine._state = ControllerState.RUNNING
    controller._cleanup_all_spmds = AsyncMock(
        side_effect=RuntimeError("metrics drain failed")
    )
    controller.stop_checkpointer = AsyncMock()
    controller._release_stopped_spmd_handles = MagicMock()

    assert await controller._state_machine.trigger_event(
        ControllerEvent.ALL_DRIVERS_COMPLETED
    )

    assert controller._state_machine.state == ControllerState.FAILED
    assert await controller.wait_for_completion(timeout=0.1) is False


@pytest.mark.asyncio
async def test_driver_handle_release_failure_marks_controller_failed(controller):
    controller._state_machine._state = ControllerState.RUNNING
    controller._cleanup_all_spmds = AsyncMock()
    controller.stop_checkpointer = AsyncMock()
    controller._release_stopped_spmd_handles = MagicMock(
        side_effect=RuntimeError("handle release failed")
    )

    assert await controller._state_machine.trigger_event(
        ControllerEvent.ALL_DRIVERS_COMPLETED
    )

    assert controller._state_machine.state == ControllerState.FAILED
    assert await controller.wait_for_completion(timeout=0.1) is False


@pytest.mark.asyncio
async def test_unexpected_terminal_callback_failure_marks_controller_failed(controller):
    controller._state_machine._state = ControllerState.RUNNING
    controller._state_machine._on_enter_completing = AsyncMock(
        side_effect=RuntimeError("unexpected terminal failure")
    )

    assert await controller._state_machine.trigger_event(
        ControllerEvent.ALL_DRIVERS_COMPLETED
    )

    assert controller._state_machine.state == ControllerState.FAILED


@pytest.mark.asyncio
async def test_eager_stop_propagates_shutdown_failure_and_releases_handles(controller):
    controller.block_all_spmds = AsyncMock()
    controller.stop_checkpointer = AsyncMock()
    controller._cleanup_all_spmds = AsyncMock(
        side_effect=RuntimeError("metrics drain failed")
    )
    controller._release_stopped_spmd_handles = MagicMock()

    with pytest.raises(RuntimeError, match="metrics drain failed"):
        await controller.stop(save_checkpoint=False)

    controller._release_stopped_spmd_handles.assert_called_once_with()
    controller.stop_checkpointer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_eager_stop_aggregates_cleanup_and_handle_release_failures(controller):
    controller.block_all_spmds = AsyncMock()
    controller.stop_checkpointer = AsyncMock()
    controller._cleanup_all_spmds = AsyncMock(
        side_effect=RuntimeError("metrics drain failed")
    )
    controller._release_stopped_spmd_handles = MagicMock(
        side_effect=RuntimeError("handle release failed")
    )

    with pytest.raises(ExceptionGroup, match="terminal cleanup failed") as raised:
        await controller.stop(save_checkpoint=False)

    assert {str(error) for error in raised.value.exceptions} == {
        "metrics drain failed",
        "handle release failed",
    }
    controller.stop_checkpointer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_eager_stop_finalizes_runtime_when_blocking_fails(controller):
    controller.block_all_spmds = AsyncMock(side_effect=RuntimeError("block failed"))
    controller._finalize_runtime = AsyncMock()

    with pytest.raises(RuntimeError, match="block failed"):
        await controller.stop(save_checkpoint=False)

    controller._finalize_runtime.assert_awaited_once_with(export_profiler_trace=False)


@pytest.mark.asyncio
async def test_block_all_spmds_reports_submission_and_rpc_failures(
    controller, mock_spmd_manager
):
    mock_spmd_manager.mock_driver1.block_all_ops.remote = MagicMock(
        side_effect=RuntimeError("submit failed")
    )
    mock_spmd_manager.mock_driver2.block_all_ops.remote = AsyncMock(
        side_effect=RuntimeError("rpc failed")
    )

    with pytest.raises(ExceptionGroup, match="block SPMDs failed") as raised:
        await controller.block_all_spmds()

    assert {str(error) for error in raised.value.exceptions} == {
        "submit failed",
        "rpc failed",
    }


@pytest.mark.asyncio
async def test_cleanup_all_spmds_preserves_driver_handles_until_terminal_release(
    controller,
):
    controller._heartbeat_manager = MagicMock()
    controller._heartbeat_manager.stop_monitoring_async = AsyncMock()
    controller._spmd_manager.stop_spmd = AsyncMock()
    controller._scheduler = MagicMock()
    controller._scheduler.pg_manager.get_all_spmd_ids.return_value = []

    await controller._cleanup_all_spmds()

    controller._spmd_manager.stop_spmd.assert_any_call(
        "op1", save_checkpoint=False, release_driver_handle=False
    )
    controller._spmd_manager.stop_spmd.assert_any_call(
        "op2", save_checkpoint=False, release_driver_handle=False
    )
    controller._heartbeat_manager.unregister_spmd.assert_any_call("op1")
    controller._heartbeat_manager.unregister_spmd.assert_any_call("op2")


@pytest.mark.asyncio
async def test_cleanup_all_spmds_propagates_failures_after_releasing_resources(
    controller,
):
    controller._heartbeat_manager = MagicMock()
    controller._heartbeat_manager.stop_monitoring_async = AsyncMock()
    controller._spmd_manager.stop_spmd = AsyncMock(
        side_effect=(RuntimeError("metrics drain failed"), None)
    )
    controller._scheduler = MagicMock()
    controller._scheduler.pg_manager.has_job_pg = True

    with pytest.raises(ExceptionGroup, match="SPMD cleanup failed") as raised:
        await controller._cleanup_all_spmds()

    assert str(raised.value.exceptions[0]) == "metrics drain failed"
    assert controller._spmd_manager.stop_spmd.await_count == 2
    controller._scheduler.release_resources.assert_called_once_with()


@pytest.mark.asyncio
async def test_cleanup_all_spmds_continues_after_monitoring_failures(controller):
    controller._heartbeat_manager = MagicMock()
    controller._heartbeat_manager.stop_monitoring_async = AsyncMock(
        side_effect=RuntimeError("monitor stop failed")
    )
    controller._heartbeat_manager.unregister_spmd.side_effect = (
        RuntimeError("unregister failed"),
        None,
    )
    controller._spmd_manager.stop_spmd = AsyncMock()
    controller._scheduler = MagicMock()
    controller._scheduler.pg_manager.has_job_pg = True

    with pytest.raises(ExceptionGroup, match="SPMD cleanup failed") as raised:
        await controller._cleanup_all_spmds()

    assert {str(error) for error in raised.value.exceptions} == {
        "monitor stop failed",
        "unregister failed",
    }
    assert controller._spmd_manager.stop_spmd.await_count == 2
    controller._scheduler.release_resources.assert_called_once_with()


@pytest.mark.asyncio
async def test_driver_error_transitions_through_failing(controller):
    cleanup_gate = asyncio.Event()
    controller._state_machine._state = ControllerState.RUNNING
    controller._cleanup_all_spmds = AsyncMock(side_effect=cleanup_gate.wait)
    controller.stop_checkpointer = AsyncMock()
    controller._release_stopped_spmd_handles = MagicMock()

    transition_task = asyncio.create_task(
        controller._state_machine.trigger_event(
            ControllerEvent.DRIVER_ERROR,
            spmd_id="op1",
            error="rank lost",
        )
    )
    await asyncio.sleep(0)
    assert controller._state_machine.state == ControllerState.FAILING
    assert not transition_task.done()

    cleanup_gate.set()

    assert await transition_task is True
    assert controller._state_machine.state == ControllerState.FAILED
    controller._release_stopped_spmd_handles.assert_called_once_with()
    controller.stop_checkpointer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_restore_snapshot_waits_for_all_drivers_ready(
    controller, mock_spmd_manager
):
    ready_gate = asyncio.Event()
    controller._heartbeat_manager = MagicMock()

    async def mock_wait(*args, **kwargs):
        await ready_gate.wait()
        return True

    controller._heartbeat_manager.wait_until_all_drivers_ready = AsyncMock(
        side_effect=mock_wait
    )
    controller._state_manager = MagicMock()
    controller._async_step_manager = MagicMock()
    controller._check_and_enforce_step_limits = AsyncMock()
    controller.block_all_spmds = AsyncMock()
    controller.unblock_all_spmds = AsyncMock()

    spmd_state = {"completed_step_count": {}, "static_args": {}}
    manifest = {
        "op1": {
            "spmd_state": spmd_state,
            "rank_storage": {"0": {"ckpt_dir": "/tmp/ckpt"}},
        },
        "op2": {
            "spmd_state": spmd_state,
            "rank_storage": {"0": {"ckpt_dir": "/tmp/ckpt"}},
        },
    }

    restore_task = asyncio.create_task(
        controller.restore_snapshot("restore_10_0", 10, manifest)
    )
    await asyncio.sleep(0)

    controller._heartbeat_manager.wait_until_all_drivers_ready.assert_awaited_once_with(
        timeout=600
    )
    controller.block_all_spmds.assert_not_called()
    mock_spmd_manager.mock_driver1.restore_from_disk.remote.assert_not_called()
    mock_spmd_manager.mock_driver2.restore_from_disk.remote.assert_not_called()

    ready_gate.set()

    await restore_task

    controller.block_all_spmds.assert_awaited_once_with()
    mock_spmd_manager.mock_driver1.restore_from_disk.remote.assert_called_once_with(
        spmd_state, {0: {"ckpt_dir": "/tmp/ckpt"}}
    )
    mock_spmd_manager.mock_driver2.restore_from_disk.remote.assert_called_once_with(
        spmd_state, {0: {"ckpt_dir": "/tmp/ckpt"}}
    )
