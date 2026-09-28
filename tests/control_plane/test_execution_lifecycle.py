import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fluxway.common.types import (
    ControllerState,
    DrainBoundary,
    DrainEvent,
    DrainEventKind,
    DriverEvent,
    DriverState,
)
from fluxway.common.types.drain import DrainAdmission, GraphModeDrainStrategy
from fluxway.control_plane.controller import ControllerActor
from fluxway.control_plane.lifecycle import ExecutionLifecycle
from fluxway.control_plane.state_machine import ControllerStateMachine
from fluxway.control_plane.spmd_manager.manager import SPMDHandle, SPMDManager
from fluxway.spmd.driver.driver import SPMDDriver
from fluxway.spmd.driver.snapshot.orchestrator import SnapshotOrchestrator
from fluxway.spmd.driver.state_machine import DriverStateMachine
from fluxway.spmd.driver.synchronizer import SPMDSynchronizer

Controller = ControllerActor.__ray_metadata__.modified_class
Driver = SPMDDriver.__ray_metadata__.modified_class


def harness(count=2):
    controller = object.__new__(Controller)
    controller._initialized = False
    controller._ready_event = asyncio.Event()
    controller._drain_boundary = DrainBoundary()
    controller._drain_boundary_lock = asyncio.Lock()
    controller._admission = None
    controller._checkpoint_task = None
    controller._drained_drivers = set()
    controller._source_control = None
    controller._profiler_runtime = SimpleNamespace(enabled=False)
    controller._checkpointer = None
    controller._heartbeat_manager = None
    controller._async_step_manager = SimpleNamespace(
        _config=SimpleNamespace(max_steps=None),
        is_all_completed=lambda: False,
    )
    controller._drain_strategy = GraphModeDrainStrategy(10)
    controller._notify_control_plane_components = AsyncMock()
    machine = ControllerStateMachine(controller)
    machine._state = ControllerState.READY
    controller._finalize_runtime = AsyncMock()
    controller._state_machine = machine
    controller._execution_lifecycle = ExecutionLifecycle(controller)
    calls = []
    receivers = {}
    handles = {}
    for index in range(count):
        name = f"driver{index}"
        sync = SPMDSynchronizer(name, 1)
        sync.register_expected_ops(("read",))
        local = SimpleNamespace(spmd_id=name)
        fsm = DriverStateMachine(local)
        fsm.require_transition(DriverEvent.INIT_COMPLETE)
        local.state_machine = fsm
        local.spmd_synchronizer = sync
        local._notify_driver_components = AsyncMock()
        local._profiler_runtime = SimpleNamespace(enabled=False)
        local._report_flow_control_state = lambda: None

        async def install(admission, name=name):
            calls.append((name, "admit", admission.target))

        async def source(name=name):
            calls.append((name, "source"))

        local.executor = SimpleNamespace(
            start_source_operators=source, activate_execution=AsyncMock()
        )
        receiver = SnapshotOrchestrator(
            name,
            SimpleNamespace(set_drain_boundary=install),
            sync,
            None,
            fsm,
            SimpleNamespace(enabled=False),
            None,
            AsyncMock(),
        )
        receivers[name] = receiver

        async def activate(local=local, name=name):
            calls.append((name, "activate"))
            await Driver.activate_execution(local)

        async def start_source(local=local):
            await Driver.start_source_operators(local)

        handles[name] = SimpleNamespace(
            activate_execution=SimpleNamespace(remote=activate),
            start_source_operators=SimpleNamespace(remote=start_source),
            request_drain=SimpleNamespace(remote=receiver.request_drain),
            resume=SimpleNamespace(remote=receiver.resume),
        )
    controller._graph = SimpleNamespace(
        requires_source_control_runtime=True,
        nodes=dict.fromkeys(handles),
    )
    controller._spmd_manager = SPMDManager()
    for spmd_id, driver in handles.items():
        handle = SPMDHandle(spmd_id)
        handle.set_driver_handle(driver)
        controller._spmd_manager.register_spmd(spmd_id, handle)

    async def initialize():
        controller._initialized = True
        controller._ready_event.set()

    async def layout():
        calls.append(("layout",))

    controller.initialize = initialize
    controller.activate_source_layout_from_descriptor = layout
    return controller, handles, receivers, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", [0, 3, None])
@pytest.mark.parametrize("phase", ["ready", "layout", "drivers", "sources"])
async def test_startup_waits_for_all_receivers(cut, phase):
    controller, handles, receivers, calls = harness()
    entered, release = asyncio.Event(), asyncio.Event()
    if phase == "ready":
        if cut is not None:
            await controller._reduce_drain_event(
                DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, cut)
            )
    else:
        if phase == "layout":
            original = controller.activate_source_layout_from_descriptor
        elif phase == "drivers":
            original = handles["driver1"].activate_execution.remote
        else:
            original = handles["driver1"].start_source_operators.remote

        async def pause():
            entered.set()
            await release.wait()
            await original()

        if phase == "layout":
            controller.activate_source_layout_from_descriptor = pause
        elif phase == "drivers":
            handles["driver1"].activate_execution.remote = pause
        else:
            handles["driver1"].start_source_operators.remote = pause
    startup = asyncio.create_task(controller.start())
    if phase != "ready":
        await asyncio.wait_for(entered.wait(), 2)
        assert controller.state_machine.state is ControllerState.ACTIVATING
        if cut is not None:
            await controller._reduce_drain_event(
                DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, cut)
            )
        assert not [call for call in calls if len(call) > 1 and call[1] == "admit"]
        release.set()
    await asyncio.wait_for(startup, 2)
    target = 10 if cut is None else cut
    assert [call for call in calls if len(call) > 1 and call[1] == "admit"] == [
        ("driver0", "admit", target),
        ("driver1", "admit", target),
    ]
    if cut == 0 and phase != "sources":
        assert not any("source" in call for call in calls)
    expected_state = DriverState.DRAINED if cut == 0 else DriverState.DRAINING
    assert all(r.state_machine.state is expected_state for r in receivers.values())
    assert controller.state_machine.state is ControllerState.RUNNING


@pytest.mark.asyncio
async def test_failed_activation_never_admits():
    controller, handles, _, calls = harness()

    async def fail():
        raise RuntimeError("activation failed")

    handles["driver1"].activate_execution.remote = fail
    with pytest.raises(RuntimeError, match="activation failed"):
        await controller.start()
    assert (
        await controller.state_machine.wait_until_terminal(2) is ControllerState.FAILED
    )
    assert not any("admit" in call for call in calls)


@pytest.mark.asyncio
async def test_ready_receiver_rejection_does_not_poison_replay():
    _, _, receivers, calls = harness(1)
    receiver = receivers["driver0"]
    admission = DrainAdmission(1, 3)
    with pytest.raises(RuntimeError, match="drain requires"):
        await receiver.request_drain(admission)
    assert receiver._admission is None
    receiver.state_machine.require_transition(DriverEvent.START)
    receiver.state_machine.require_transition(DriverEvent.ACTIVATION_COMPLETE)
    await receiver.request_drain(admission)
    assert calls == [("driver0", "admit", 3)]


@pytest.mark.asyncio
async def test_receiver_duplicate_awaits_failed_application():
    _, _, receivers, _ = harness(1)
    receiver = receivers["driver0"]
    receiver.state_machine.require_transition(DriverEvent.START)
    receiver.state_machine.require_transition(DriverEvent.ACTIVATION_COMPLETE)
    entered, release = asyncio.Event(), asyncio.Event()

    async def fail(admission):
        entered.set()
        await release.wait()
        raise RuntimeError("install failed")

    receiver.executor.set_drain_boundary = fail
    admission = DrainAdmission(1, 3)
    first = asyncio.create_task(receiver.request_drain(admission))
    await entered.wait()
    second = asyncio.create_task(receiver.request_drain(admission))
    await asyncio.sleep(0)
    assert not second.done()
    release.set()
    outcomes = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(outcome, RuntimeError) for outcome in outcomes)
    with pytest.raises(RuntimeError, match="install failed"):
        await receiver.request_drain(admission)


@pytest.mark.asyncio
async def test_concurrent_delivery_waits_for_one_application():
    controller, _, _, calls = harness(1)
    await controller.start()
    await asyncio.gather(
        *(controller._execution_lifecycle.deliver_current() for _ in range(10))
    )
    assert [call for call in calls if "admit" in call] == [("driver0", "admit", 10)]


@pytest.mark.asyncio
async def test_cancelled_admission_waiter_does_not_lose_late_failure():
    controller, handles, _, _ = harness(1)
    await controller.start()
    entered, release = asyncio.Event(), asyncio.Event()

    def fail_later(_admission):
        async def operation():
            entered.set()
            await release.wait()
            raise RuntimeError("late admission failure")

        return asyncio.create_task(operation())

    handles["driver0"].request_drain.remote = fail_later
    request = asyncio.create_task(
        controller._reduce_drain_event(
            DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 2)
        )
    )
    await entered.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    release.set()
    delivery = controller._execution_lifecycle._delivery
    assert delivery is not None
    with pytest.raises(RuntimeError, match="late admission failure"):
        await delivery[1]
    await asyncio.sleep(0)
    assert (
        await controller.state_machine.wait_until_terminal(2) is ControllerState.FAILED
    )
    assert not controller._execution_lifecycle._operations


@pytest.mark.asyncio
async def test_new_revision_completes_while_old_admission_waits():
    controller, handles, _, _ = harness(1)
    await controller.start()
    entered, release = asyncio.Event(), asyncio.Event()
    delivered = []

    async def receive(admission):
        delivered.append(admission.target)
        if admission.target == 7:
            entered.set()
            await release.wait()

    handles["driver0"].request_drain.remote = receive
    old = asyncio.create_task(
        controller._reduce_drain_event(
            DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 7)
        )
    )
    await entered.wait()
    await asyncio.wait_for(
        controller._reduce_drain_event(
            DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 3)
        ),
        2,
    )
    assert delivered == [7, 3]
    assert not old.done()
    release.set()
    await old
    await asyncio.sleep(0)
    assert not controller._execution_lifecycle._operations


@pytest.mark.asyncio
async def test_preflight_failure_enters_failure_lifecycle():
    controller, _, _, _ = harness(1)
    await controller.start()
    controller.spmd_manager.release_all_driver_handles()
    with pytest.raises(RuntimeError, match="Driver handle not set"):
        await controller._reduce_drain_event(
            DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 2)
        )
    assert (
        await controller.state_machine.wait_until_terminal(2) is ControllerState.FAILED
    )
    assert controller.state_machine.get_failures()["controller"].phase == "ADMISSION"


@pytest.mark.asyncio
async def test_admission_failure_reclaims_slow_peer_without_self_await():
    controller, handles, _, _ = harness(2)
    await controller.start()
    waiting = asyncio.Event()

    async def slow(_admission):
        waiting.set()
        await asyncio.Event().wait()

    async def fail(_admission):
        await waiting.wait()
        raise RuntimeError("peer failed")

    handles["driver0"].request_drain.remote = slow
    handles["driver1"].request_drain.remote = fail
    with pytest.raises(RuntimeError, match="peer failed"):
        await controller._reduce_drain_event(
            DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 2)
        )
    assert (
        await controller.state_machine.wait_until_terminal(2) is ControllerState.FAILED
    )
    assert not controller.spmd_manager._pending_driver_calls
    assert not controller._execution_lifecycle._operations
    controller._finalize_runtime.assert_awaited_once()


@pytest.mark.asyncio
async def test_reductions_before_delivery_task_starts_do_not_lose_terminal():
    controller, _, _, calls = harness(1)
    await controller._execution_lifecycle.fanout("activate_execution")
    controller.state_machine._state = ControllerState.RUNNING
    await asyncio.gather(
        controller._reduce_drain_event(DrainEvent(DrainEventKind.PERIODIC_REQUEST, 10)),
        controller._reduce_drain_event(
            DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 2)
        ),
    )
    assert ("driver0", "admit", 2) in calls


@pytest.mark.asyncio
async def test_resume_defers_terminal_until_every_driver_resumed():
    controller, handles, receivers, calls = harness()
    await controller.start()
    for receiver in receivers.values():
        receiver.synchronizer.report_worker_op_completed(0, "read", 10)
        await receiver._drained_component_notification
    await controller._reduce_drain_event(DrainEvent(DrainEventKind.DRAIN_COMPLETE, 10))
    await controller._reduce_drain_event(DrainEvent(DrainEventKind.SNAPSHOT_BEGIN, 10))
    await controller._reduce_drain_event(
        DrainEvent(DrainEventKind.CHECKPOINT_COMMITTED, 10)
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original = handles["driver1"].resume.remote

    async def pause():
        entered.set()
        await release.wait()
        await original()

    handles["driver1"].resume.remote = pause
    continuation = asyncio.create_task(controller._resume_and_schedule_next_drain())
    await asyncio.wait_for(entered.wait(), 2)
    assert controller.state_machine.state is ControllerState.RESUMING
    await controller._reduce_drain_event(
        DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 12)
    )
    assert not any(len(call) == 3 and call[2] == 12 for call in calls)
    release.set()
    await asyncio.wait_for(continuation, 2)
    assert [call for call in calls if len(call) == 3 and call[2] == 12] == [
        ("driver0", "admit", 12),
        ("driver1", "admit", 12),
    ]


@pytest.mark.asyncio
async def test_failed_resume_does_not_open_next_gate():
    controller, handles, _, calls = harness(1)
    controller.state_machine._state = ControllerState.RUNNING
    controller._drain_boundary = DrainBoundary.from_committed_cut(2)

    async def fail():
        raise RuntimeError("resume failed")

    handles["driver0"].resume.remote = fail
    with pytest.raises(RuntimeError, match="resume failed"):
        await controller._resume_and_schedule_next_drain()
    assert (
        await controller.state_machine.wait_until_terminal(2) is ControllerState.FAILED
    )
    assert not any("admit" in call for call in calls)


@pytest.mark.asyncio
async def test_topology_readiness_cannot_manufacture_committed_progress():
    controller, _, _, _ = harness()
    controller._checkpointer = SimpleNamespace(interval_steps=10)
    controller._async_step_manager.get_all_spmd_steps = lambda: {
        "reader": SimpleNamespace(completed_step_count={"read": 25})
    }
    controller._drain_boundary = DrainBoundary.from_committed_cut(0)
    controller.state_machine._state = ControllerState.TOPOLOGY_REGISTERING
    await controller._on_all_topology_ready()
    assert controller._drain_boundary.committed_cut == 0
    assert controller.state_machine.state is ControllerState.READY


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [False, True])
async def test_drained_save_commit_resume_or_complete(terminal):
    controller, _, receivers, calls = harness(1)
    controller._checkpointer = SimpleNamespace(save=AsyncMock())
    if terminal:
        controller._async_step_manager._config.max_steps = 10
    await controller.start()
    receiver = receivers["driver0"]
    receiver.synchronizer.report_worker_op_completed(0, "read", 10)
    await receiver._drained_component_notification
    await controller.report_driver_drained("driver0", controller._admission)
    await asyncio.wait_for(controller._checkpoint_task, 2)
    assert controller._drain_boundary.committed_cut == 10
    controller._checkpointer.save.assert_awaited_once()
    if terminal:
        assert (
            await controller.state_machine.wait_until_terminal(2)
            is ControllerState.COMPLETED
        )
        assert receiver.state_machine.state is DriverState.DRAINED
        assert [call for call in calls if "admit" in call] == [("driver0", "admit", 10)]
    else:
        assert controller.state_machine.state is ControllerState.RUNNING
        assert receiver.state_machine.state is DriverState.DRAINING
        assert [call for call in calls if "admit" in call] == [
            ("driver0", "admit", 10),
            ("driver0", "admit", 20),
        ]


@pytest.mark.asyncio
async def test_failed_save_is_visible_and_never_resumes():
    controller, _, receivers, calls = harness(1)
    controller._checkpointer = SimpleNamespace(
        save=AsyncMock(side_effect=RuntimeError("save failed"))
    )
    await controller.start()
    receiver = receivers["driver0"]
    receiver.synchronizer.report_worker_op_completed(0, "read", 10)
    await receiver._drained_component_notification
    await controller.report_driver_drained("driver0", controller._admission)
    with pytest.raises(RuntimeError, match="save failed"):
        await asyncio.wait_for(controller._checkpoint_task, 2)
    assert (
        await controller.state_machine.wait_until_terminal(2) is ControllerState.FAILED
    )
    assert controller._drain_boundary.committed_cut is None
    assert [call for call in calls if "admit" in call] == [("driver0", "admit", 10)]


@pytest.mark.asyncio
async def test_restore_precedes_layout_and_admits_from_committed_cut():
    controller, _, _, calls = harness(1)

    async def restore(reference):
        assert reference == "latest"
        assert controller.state_machine.state is ControllerState.ACTIVATING
        assert not calls
        calls.append(("restore",))
        controller._drain_boundary = DrainBoundary.from_committed_cut(17)
        return 17

    controller._checkpointer = SimpleNamespace(restore=restore)
    assert await controller.start("latest") == 17
    assert calls[:2] == [("restore",), ("layout",)]
    assert calls[-1] == ("driver0", "admit", 27)


@pytest.mark.asyncio
async def test_restore_rejects_running_graph_before_io():
    controller, _, _, _ = harness(1)
    controller._checkpointer = SimpleNamespace(restore=AsyncMock())
    controller.state_machine._state = ControllerState.RUNNING
    with pytest.raises(RuntimeError, match="pre-execution"):
        await controller.restore_checkpoint()
    controller._checkpointer.restore.assert_not_awaited()


@pytest.mark.asyncio
async def test_admission_rpc_failure_enters_failed_lifecycle():
    controller, handles, _, calls = harness(1)
    await controller.start()
    failed = AsyncMock(side_effect=RuntimeError("admission delivery failed"))
    handles["driver0"].request_drain.remote = failed
    with pytest.raises(RuntimeError, match="admission delivery failed"):
        await controller._reduce_drain_event(
            DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 2)
        )
    assert (
        await controller.state_machine.wait_until_terminal(2) is ControllerState.FAILED
    )
    await controller._execution_lifecycle.deliver_current()
    failed.assert_awaited_once()
    assert [call for call in calls if "admit" in call] == [("driver0", "admit", 10)]


@pytest.mark.asyncio
async def test_checkpoint_requires_configured_owner():
    controller, _, _, _ = harness(1)
    controller.state_machine._state = ControllerState.RUNNING
    with pytest.raises(RuntimeError, match="configured checkpointer"):
        await controller._execution_lifecycle.checkpoint(3)
    assert (
        await controller.state_machine.wait_until_terminal(2) is ControllerState.FAILED
    )


@pytest.mark.asyncio
async def test_terminal_shortening_progresses_while_old_driver_rpc_waits():
    controller, handles, receivers, calls = harness(1)
    await controller._execution_lifecycle.fanout("activate_execution")
    controller.state_machine._state = ControllerState.RUNNING
    entered, release = asyncio.Event(), asyncio.Event()
    original = handles["driver0"].request_drain.remote

    async def delayed_reply(admission):
        await original(admission)
        if admission.target == 10:
            entered.set()
            await release.wait()

    handles["driver0"].request_drain.remote = delayed_reply
    periodic = asyncio.create_task(
        controller._reduce_drain_event(DrainEvent(DrainEventKind.PERIODIC_REQUEST, 10))
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.wait_for(
            controller._reduce_drain_event(
                DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 2)
            ),
            2,
        )
        assert receivers["driver0"].synchronizer.drain_step_count == 2
        assert not controller._drain_boundary_lock.locked()
        assert not periodic.done()
    finally:
        release.set()
        await asyncio.wait_for(periodic, 2)
    assert [call for call in calls if "admit" in call] == [
        ("driver0", "admit", 10),
        ("driver0", "admit", 2),
    ]
