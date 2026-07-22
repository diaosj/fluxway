import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from fluxway.common.types import ControllerEvent
from fluxway.common.types import ControllerState, ExecutionFailure
from fluxway.common.types.drain import DrainStrategy
from fluxway.common.decorators import operator, spmd
from fluxway.control_plane import ExecutionFailedError
from fluxway.control_plane.controller import ControllerActor
from fluxway.control_plane.control_ledger import ControlLedger
from fluxway.control_plane.ledger_client import ControlRecord
from fluxway.control_plane.graph.builder import Builder
from fluxway.resources import ResourceProfile


@spmd(
    resource_profile=ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("data",),
    mesh_axis_shapes=(1,),
)
class _DrainSPMD:
    @operator(name="run")
    def run(self):
        return None


def _drain_graph():
    builder = Builder()
    builder.add_node("op1", _DrainSPMD)
    builder.add_node("op2", _DrainSPMD)
    builder.add_func_edge("op1", "run", "op2", "run", tensor_map={})
    return builder.build("drain-integration")


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
    graph = _drain_graph()

    ctrl = ControllerActor.__ray_actor_class__(  # type: ignore[attr-defined]
        graph=graph,
        placement_recipe=MagicMock(),
    )
    ctrl._spmd_manager = mock_spmd_manager
    ctrl._graph = graph

    # inject mock strategy and checkpointer
    ctrl._drain_strategy = MockDrainStrategy()
    ctrl._checkpointer = AsyncMock()

    return ctrl


def test_whole_job_terminal_rejects_divergent_source_clocks() -> None:
    bindings = {
        "source_a:read": ("source_a", "read"),
        "source_b:read": ("source_b", "read"),
    }

    with pytest.raises(RuntimeError, match="aligned source terminal counts"):
        ControllerActor.__ray_metadata__.modified_class._terminal_target(
            bindings,
            {"source_a:read": 3, "source_b:read": 5},
        )


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

    await controller._checkpoint_task

    controller._checkpointer.save.assert_called_with(2)
    mock_spmd_manager.mock_driver1.resume.remote.assert_called()
    mock_spmd_manager.mock_driver2.resume.remote.assert_called()

    # Verifies next target issued
    mock_spmd_manager.mock_driver1.request_drain.remote.assert_called_with(4)


@pytest.mark.asyncio
async def test_terminal_control_stream_reuses_driver_drain(
    controller, mock_spmd_manager
):
    ledger = ControlLedger(4)
    await ledger.append(
        tuple(
            ControlRecord(
                stream_id="op1:read",
                position=position,
                payload={"batch": position},
            )
            for position in range(3)
        )
    )
    controller._control_ledger = ledger
    controller._source_control_stream_bindings = lambda: {"op1:read": ("op1", "read")}
    controller._state_machine._state = ControllerState.RUNNING

    await controller.close_control_stream("op1:read")

    assert controller._terminal_drain_target == 3
    assert controller._last_requested_drain_target == 3
    mock_spmd_manager.mock_driver1.request_terminal_drain.remote.assert_called_once_with(
        3
    )
    mock_spmd_manager.mock_driver2.request_terminal_drain.remote.assert_called_once_with(
        3
    )


@pytest.mark.asyncio
async def test_terminal_control_stream_defers_driver_rpc_until_running(
    controller, mock_spmd_manager
):
    ledger = ControlLedger(4)
    await ledger.append(
        tuple(
            ControlRecord(
                stream_id="op1:read",
                position=position,
                payload={"batch": position},
            )
            for position in range(3)
        )
    )
    controller._control_ledger = ledger
    controller._source_control_stream_bindings = lambda: {"op1:read": ("op1", "read")}
    controller._state_machine._state = ControllerState.READY

    await controller.close_control_stream("op1:read")

    assert controller._terminal_drain_target == 3
    mock_spmd_manager.mock_driver1.request_terminal_drain.remote.assert_not_called()
    mock_spmd_manager.mock_driver2.request_terminal_drain.remote.assert_not_called()

    controller._state_machine._state = ControllerState.RUNNING
    await controller._schedule_terminal_drain_if_complete()

    mock_spmd_manager.mock_driver1.request_terminal_drain.remote.assert_called_once_with(
        3
    )
    mock_spmd_manager.mock_driver2.request_terminal_drain.remote.assert_called_once_with(
        3
    )


@pytest.mark.asyncio
async def test_start_dispatches_latched_terminal_without_periodic_drain(controller):
    events = []
    controller.initialize = AsyncMock()
    controller._ready_event.set()
    controller._state_machine._state = ControllerState.READY
    controller._control_ledger = MagicMock()

    async def schedule_terminal():
        assert controller._state_machine.state is ControllerState.RUNNING
        controller._terminal_drain_target = 3
        events.append("terminal")

    async def start_sources():
        events.append("sources")

    controller._schedule_terminal_drain_if_complete = AsyncMock(
        side_effect=schedule_terminal
    )
    controller._issue_initial_drain = AsyncMock()
    controller.start_source_operators = AsyncMock(side_effect=start_sources)

    assert await controller.start() is None

    assert events == ["terminal", "sources"]
    controller._issue_initial_drain.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_without_checkpoint_skips_terminal_ledger(controller):
    controller.initialize = AsyncMock()
    controller._ready_event.set()
    controller._state_machine._state = ControllerState.READY
    controller._control_ledger = None
    controller._issue_initial_drain = AsyncMock()
    controller.start_source_operators = AsyncMock()
    controller._schedule_terminal_drain_if_complete = AsyncMock()

    assert await controller.start() is None

    controller._schedule_terminal_drain_if_complete.assert_not_awaited()
    controller._issue_initial_drain.assert_awaited_once_with()
    controller.start_source_operators.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_zero_terminal_target_completes_without_starting_sources(controller):
    controller.initialize = AsyncMock()
    controller._ready_event.set()
    controller._state_machine._state = ControllerState.READY
    controller._control_ledger = MagicMock()
    controller._issue_initial_drain = AsyncMock()
    controller.start_source_operators = AsyncMock()

    async def complete_terminal():
        controller._terminal_drain_target = 0
        controller._state_machine._state = ControllerState.COMPLETED

    controller._schedule_terminal_drain_if_complete = AsyncMock(
        side_effect=complete_terminal
    )

    assert await controller.start() is None

    controller._issue_initial_drain.assert_not_awaited()
    controller.start_source_operators.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_checkpointer_skips_self_await(controller):
    controller._checkpoint_task = asyncio.current_task()

    await controller.stop_checkpointer()

    assert controller._checkpoint_task is None


@pytest.mark.asyncio
async def test_checkpoint_failure_enters_failed_without_resuming(controller):
    controller._state_machine._state = ControllerState.RUNNING
    controller._last_requested_drain_target = 2
    controller._checkpointer.save.side_effect = RuntimeError("storage unavailable")
    controller._finalize_runtime = AsyncMock()
    controller._resume_and_schedule_next_drain = AsyncMock()

    await controller._on_all_drivers_drained()
    await controller._checkpoint_task

    assert controller._state_machine.state is ControllerState.FAILED
    controller._resume_and_schedule_next_drain.assert_not_awaited()
    failure = controller._state_machine.get_failures()["controller"]
    assert failure.phase == "CHECKPOINT"
    assert failure.step == 2


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
    wait_task = asyncio.create_task(controller.wait_for_completion())
    await asyncio.sleep(0)

    assert controller._state_machine.state == ControllerState.COMPLETING
    assert not wait_task.done()

    cleanup_gate.set()

    assert await transition_task is True
    assert await asyncio.wait_for(wait_task, timeout=0.2) is True
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
    with pytest.raises(ExecutionFailedError) as raised:
        await controller.wait_for_completion(timeout=0.1)
    failure = raised.value.failures["controller"]
    assert failure.phase == "CONTROLLER_CLEANUP"
    assert failure.exception_type == "RuntimeError"
    assert failure.message == "metrics drain failed"


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
    with pytest.raises(ExecutionFailedError) as raised:
        await controller.wait_for_completion(timeout=0.1)
    assert raised.value.failures["controller"].message == "handle release failed"


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
    failure = controller._state_machine.get_failures()["controller"]
    assert failure.phase == "CONTROLLER_CLEANUP"
    assert failure.message == "unexpected terminal failure"


@pytest.mark.asyncio
async def test_wait_for_completion_raises_builtin_timeout(controller):
    controller._state_machine._state = ControllerState.RUNNING

    with pytest.raises(TimeoutError):
        await controller.wait_for_completion(timeout=0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ("report_spmd_error", "_on_driver_error"))
async def test_controller_error_entrypoints_forward_failure(controller, method_name):
    failure = ExecutionFailure(
        spmd_id="op1",
        rank=None,
        op_name=None,
        step=None,
        phase="DRIVER",
        exception_type="RuntimeError",
        message="rank lost",
        traceback=None,
    )
    controller._state_machine.trigger_event = AsyncMock()

    await getattr(controller, method_name)("op1", failure)

    controller._state_machine.trigger_event.assert_awaited_once_with(
        ControllerEvent.DRIVER_ERROR,
        spmd_id="op1",
        failure=failure,
    )


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
async def test_cleanup_all_spmds_stops_spmds_concurrently(controller):
    both_started = asyncio.Event()
    started: set[str] = set()
    cleanup_order: list[tuple[str, str]] = []

    async def stop_spmd(spmd_id, **kwargs):
        started.add(spmd_id)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.1)
        cleanup_order.append(("stop", spmd_id))

    def unregister_spmd(spmd_id):
        assert ("stop", spmd_id) in cleanup_order
        cleanup_order.append(("unregister", spmd_id))

    def release_resources():
        assert {event for event in cleanup_order if event[0] == "unregister"} == {
            ("unregister", "op1"),
            ("unregister", "op2"),
        }
        cleanup_order.append(("release", "job"))

    controller._heartbeat_manager = MagicMock()
    controller._heartbeat_manager.stop_monitoring_async = AsyncMock()
    controller._heartbeat_manager.unregister_spmd.side_effect = unregister_spmd
    controller._spmd_manager.stop_spmd = AsyncMock(side_effect=stop_spmd)
    controller._scheduler = MagicMock()
    controller._scheduler.pg_manager.has_job_pg = True
    controller._scheduler.release_resources.side_effect = release_resources

    await controller._cleanup_all_spmds()

    assert started == {"op1", "op2"}
    assert controller._spmd_manager.stop_spmd.await_count == 2
    controller._heartbeat_manager.unregister_spmd.assert_any_call("op1")
    controller._heartbeat_manager.unregister_spmd.assert_any_call("op2")
    assert cleanup_order[-1] == ("release", "job")


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

    failure = ExecutionFailure(
        spmd_id="op1",
        rank=None,
        op_name=None,
        step=None,
        phase="DRIVER",
        exception_type="RuntimeError",
        message="rank lost",
        traceback=None,
    )
    transition_task = asyncio.create_task(
        controller._state_machine.trigger_event(
            ControllerEvent.DRIVER_ERROR,
            spmd_id="op1",
            failure=failure,
        )
    )
    await asyncio.sleep(0)
    assert controller._state_machine.state == ControllerState.FAILING
    assert not transition_task.done()

    cleanup_gate.set()

    assert await transition_task is True
    assert controller._state_machine.state == ControllerState.FAILED
    assert controller._state_machine.get_failures() == {"op1": failure}
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

    spmd_state = {
        "completed_step_count": {},
        "static_args": {},
        "driver_components": {},
    }
    controller._control_ledger = ControlLedger(1)
    manifest = {
        "op1": {
            "spmd_state": spmd_state,
            "rank_storage": {"0": {"ckpt_dir": "/tmp/ckpt"}},
        },
        "op2": {
            "spmd_state": spmd_state,
            "rank_storage": {"0": {"ckpt_dir": "/tmp/ckpt"}},
        },
        "__fluxway_control_plane__": {
            "control_ledger": controller._control_ledger.checkpoint_state(cuts={}),
            "components": {},
            "last_checkpointed_step": 0,
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
        spmd_state, {0: {"ckpt_dir": "/tmp/ckpt"}}, 10
    )
    mock_spmd_manager.mock_driver2.restore_from_disk.remote.assert_called_once_with(
        spmd_state, {0: {"ckpt_dir": "/tmp/ckpt"}}, 10
    )


@pytest.mark.asyncio
async def test_terminal_checkpoint_restore_uses_existing_completion_path(
    controller, mock_spmd_manager
):
    terminal_ledger = ControlLedger(4)
    records = tuple(
        ControlRecord(
            stream_id="op1:read",
            position=position,
            payload={"batch": position},
        )
        for position in range(3)
    )
    await terminal_ledger.append(records)
    for position in range(3):
        await terminal_ledger.receive("op1:read", position)
    await terminal_ledger.seal("op1:read")
    terminal_state = terminal_ledger.checkpoint_state(cuts={"op1:read": 3})

    controller._control_ledger = ControlLedger(4)
    controller._source_control_stream_bindings = lambda: {"op1:read": ("op1", "read")}
    controller._heartbeat_manager = MagicMock()
    controller._heartbeat_manager.wait_until_all_drivers_ready = AsyncMock(
        return_value=True
    )
    controller._state_manager = MagicMock()
    controller._async_step_manager = MagicMock()
    controller._flow_control_manager = MagicMock()
    controller._flow_control_manager.observe_op_completed.return_value = {}
    controller._apply_flow_control_windows = AsyncMock()
    controller.block_all_spmds = AsyncMock()
    controller.unblock_all_spmds = AsyncMock()
    controller._resume_and_schedule_next_drain = AsyncMock()
    spmd_state = {
        "completed_step_count": {"read": 3},
        "static_args": {},
        "driver_components": {},
    }
    manifest = {
        spmd_id: {
            "spmd_state": spmd_state,
            "rank_storage": {"0": {"ckpt_dir": "/tmp/ckpt"}},
        }
        for spmd_id in ("op1", "op2")
    }
    manifest["__fluxway_control_plane__"] = {
        "control_ledger": terminal_state,
        "components": {},
        "last_checkpointed_step": 3,
    }

    await controller.restore_snapshot("restore_terminal", 3, manifest)

    controller.unblock_all_spmds.assert_not_awaited()
    controller._resume_and_schedule_next_drain.assert_awaited_once_with()
