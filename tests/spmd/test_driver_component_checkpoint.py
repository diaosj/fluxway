import asyncio
from types import SimpleNamespace

import pytest

from fluxway.common.mesh import DeviceMesh
from fluxway.common.types import (
    CheckpointCommitPermit,
    CheckpointCut,
    DrainBoundary,
    DriverEvent,
    DriverState,
    SpmdCheckpoint,
    TopologySnapshot,
)
from fluxway.profiler.record import PAYLOAD_START_COLUMN, TraceArena
from fluxway.profiler.recorder import TraceRecorder
from fluxway.profiler.runtime import ProfilerRuntimeFactory
from fluxway.spmd.driver.snapshot.orchestrator import SnapshotOrchestrator
from fluxway.spmd.driver.state_machine import DriverStateMachine
from fluxway.spmd.driver.driver import SPMDDriver


class _RemoteMethod:
    def __init__(self, events, name):
        self._events = events
        self._name = name
        self.calls = []

    def remote(self, *args):
        self._events.append(self._name)
        self.calls.append(args)


class _Controller:
    def __init__(self, events):
        self.report_driver_drained = _RemoteMethod(events, "report_drained")
        self.receive_snapshot_metadata = _RemoteMethod(events, "snapshot_metadata")
        self.report_snapshot_failed = _RemoteMethod(events, "snapshot_failed")


class _Registry:
    def __init__(self, events):
        self._events = events
        self.restored = None
        self.last_cut = None

    def checkpoint_state(self, cut):
        self._events.append(("component_checkpoint", cut))
        self.last_cut = cut
        return {"reader:read": {"cursor": {"checkpoint_id": 3}}}

    def restore_state(self, state, cut):
        self._events.append(("component_restore", cut))
        self.restored = state
        self.last_cut = cut


class _Executor:
    def __init__(self, events):
        self._events = events
        self.fail_save = False
        mesh = SimpleNamespace(axis_names=("data",), shape=(1,))
        view = SimpleNamespace(mesh=mesh, world_size=1, operator_ids=())
        self.config = SimpleNamespace(require_graph_view=lambda: view)

    def get_worker_handles(self):
        return {0: object()}

    async def set_drain_boundary(self, target, terminal_cut=None):
        self._events.append(("worker_drain", target, terminal_cut))

    def clear_drain_boundary(self):
        self._events.append("worker_drain_clear")

    async def checkpoint_prepare_save(
        self, root, checkpoint_id, attempt_id, permit, cut
    ):
        self._events.append("worker_prepare")

    async def checkpoint_save(self):
        self._events.append("worker_save")
        if self.fail_save:
            raise RuntimeError("save failed")
        return [
            (
                0,
                {
                    "drain_boundary": DrainBoundary(committed_cut=5).to_dict(),
                },
                {},
            )
        ]

    async def checkpoint_finish_save(self):
        self._events.append("worker_finalize")

    async def checkpoint_commit_save(self, gathered):
        self._events.append("worker_commit")
        assert gathered == {}
        return {0: {}}

    async def checkpoint_rollback_save(self):
        self._events.append("worker_rollback")

    async def checkpoint_publish_save(self):
        self._events.append("worker_published")

    async def checkpoint_restore(self, *args):
        self._events.append("worker_restore")


class _Synchronizer:
    def __init__(self, events):
        self._events = events

    def drain_to_step_count(self, step_count, on_drained):
        self._events.append(("synchronizer_drain", step_count))
        on_drained()

    def clear_drain(self):
        self._events.append("synchronizer_clear")

    def load_spmd_state(self, state):
        self._events.append("synchronizer_restore")


class _PublicationOrchestrator:
    def __init__(self):
        self.calls = []

    async def publish_save(self, checkpoint_id, permit):
        self.calls.append(("publish_save", checkpoint_id, permit))

    async def rollback_save(self, checkpoint_id, permit):
        self.calls.append(("rollback_save", checkpoint_id, permit))


def _orchestrator(events, *, state=DriverState.DRAINED, profiler_runtime=None):
    controller = _Controller(events)
    registry = _Registry(events)
    driver = SimpleNamespace(
        spmd_id="reader",
        _submit_driver_effect=lambda operation, args=(), **_: operation(*args),
        _schedule_driver_component_state=lambda state: events.append(
            ("component_state", state)
        ),
        _report_error=lambda failure: events.append(("driver_error", failure)),
    )
    state_machine = DriverStateMachine(driver)
    state_machine._state = state

    async def notify(component_state):
        events.append(("component_state", component_state))

    if profiler_runtime is None:
        profiler_runtime = SimpleNamespace(enabled=False)

    return (
        SnapshotOrchestrator(
            spmd_id="reader",
            executor=_Executor(events),
            synchronizer=_Synchronizer(events),
            controller_handle=controller,
            state_machine=state_machine,
            profiler_runtime=profiler_runtime,
            component_registry=registry,
            notify_component_state=notify,
        ),
        controller,
        registry,
    )


def test_snapshot_contains_driver_component_state():
    events = []
    orchestrator, controller, _ = _orchestrator(events)
    cut = CheckpointCut.for_step("tx", 5)

    asyncio.run(
        orchestrator.prepare_snapshot(
            "snapshot",
            "attempt",
            CheckpointCommitPermit("attempt", 1, b"permit"),
            cut,
            "/tmp/repository",
        )
    )

    preliminary = controller.receive_snapshot_metadata.calls[0][2]
    assert isinstance(preliminary, SpmdCheckpoint)
    assert preliminary.driver_components == {}
    resolved_cut = CheckpointCut(
        cut_id=cut.cut_id,
        position=cut.position,
        stream_cuts={"reader:read": 5},
    )
    snapshot = orchestrator.capture_driver_components("snapshot", resolved_cut)
    assert isinstance(snapshot, SpmdCheckpoint)
    assert snapshot.driver_components == {
        "reader:read": {"cursor": {"checkpoint_id": 3}}
    }
    assert events.index("worker_finalize") < events.index("worker_commit")
    assert ("component_checkpoint", resolved_cut) in events


def test_restore_components_precedes_component_restore():
    events = []
    orchestrator, _, registry = _orchestrator(events, state=DriverState.READY)
    topology = TopologySnapshot(1, ("data",), (1,))
    snapshot = SpmdCheckpoint(
        topology,
        {
            "drain_boundary": DrainBoundary(committed_cut=5).to_dict(),
        },
        {},
        {"reader:read": {"cursor": {"checkpoint_id": 3}}},
    )
    restore_cut = CheckpointCut.for_step("restore", 5)

    asyncio.run(
        orchestrator._restore_from_manifest_unprofiled(
            "snapshot",
            restore_cut,
            snapshot,
            "/tmp/repository",
        )
    )

    assert registry.restored == {"reader:read": {"cursor": {"checkpoint_id": 3}}}
    assert registry.last_cut == restore_cut
    assert events.index(("component_restore", restore_cut)) < events.index(
        "worker_restore"
    )


def test_save_failure_rolls_back_component_transaction():
    events = []
    orchestrator, _, _ = _orchestrator(events)
    orchestrator.executor.fail_save = True

    asyncio.run(
        orchestrator.prepare_snapshot(
            "snapshot",
            "attempt",
            CheckpointCommitPermit("attempt", 1, b"permit"),
            CheckpointCut.for_step("tx", 5),
            "/tmp/repository",
        )
    )

    assert events.index("worker_rollback") < events.index("snapshot_failed")


def test_concurrent_snapshot_failure_coalesces_rollback():
    async def scenario():
        events = []
        orchestrator, _, _ = _orchestrator(events)
        orchestrator.executor.fail_save = True
        rollback_started = asyncio.Event()
        release_rollback = asyncio.Event()
        rollback_calls = 0

        async def rollback():
            nonlocal rollback_calls
            rollback_calls += 1
            rollback_started.set()
            await release_rollback.wait()

        orchestrator.executor.checkpoint_rollback_save = rollback
        persist_task = asyncio.create_task(
            orchestrator.prepare_snapshot(
                "snapshot",
                "attempt",
                CheckpointCommitPermit("attempt", 1, b"permit"),
                CheckpointCut.for_step("cut", 5),
                "/tmp/repository",
            )
        )
        await rollback_started.wait()
        fail_task = asyncio.create_task(orchestrator.fail_snapshot("snapshot"))
        await asyncio.sleep(0)
        assert rollback_calls == 1
        release_rollback.set()
        await asyncio.gather(persist_task, fail_task)
        assert rollback_calls == 1

    asyncio.run(scenario())


def test_unassembled_snapshot_failure_uses_prepublication_cleanup_once():
    async def scenario():
        events = []
        orchestrator, _, _ = _orchestrator(events)
        rollback_calls = 0

        async def rollback():
            nonlocal rollback_calls
            rollback_calls += 1

        orchestrator.executor.checkpoint_rollback_save = rollback
        await orchestrator._prepare_snapshot_unprofiled(
            "snapshot",
            "attempt",
            CheckpointCommitPermit("attempt", 1, b"permit"),
            CheckpointCut.for_step("cut", 5),
            "/tmp/repository",
        )

        await orchestrator.fail_snapshot("snapshot")
        await orchestrator.fail_snapshot("snapshot")

        assert rollback_calls == 1
        assert "worker_rollback" not in events

    asyncio.run(scenario())


def test_driver_uses_snapshot_publication_protocol_names():
    raw_driver = SPMDDriver.__ray_actor_class__
    orchestrator = _PublicationOrchestrator()
    driver = SimpleNamespace(snapshot_orchestrator=orchestrator)
    permit = CheckpointCommitPermit("attempt", 1, b"permit")

    asyncio.run(raw_driver.publish_snapshot(driver, "snapshot", permit))
    asyncio.run(raw_driver.rollback_snapshot(driver, "snapshot", permit))

    assert orchestrator.calls == [
        ("publish_save", "snapshot", permit),
        ("rollback_save", "snapshot", permit),
    ]


def test_profiled_driver_snapshot_records_canonical_checkpoint_id():
    events = []
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=64 * 1024),
        run_id="test-run",
        process_name="driver/reader",
    )
    profiler_runtime = ProfilerRuntimeFactory(recorder)
    orchestrator, _, _ = _orchestrator(events, profiler_runtime=profiler_runtime)
    cut = CheckpointCut.for_position("cut-5", 5)

    asyncio.run(
        orchestrator.prepare_snapshot(
            "ckpt-snap-1",
            "attempt",
            CheckpointCommitPermit("attempt", 1, b"permit"),
            cut,
            "/tmp/repository",
        )
    )

    rows = recorder.arena.snapshot()
    names = [recorder.metadata.name_for(int(row[4])) for row in rows]
    assert names == ["driver.snapshot"]

    span = rows[0]
    payload_fields = recorder.metadata.payload_fields_for(int(span[4]))
    assert payload_fields == ("checkpoint_id", "target_step", "worker_count")
    assert recorder.metadata.name_for(int(span[PAYLOAD_START_COLUMN])) == "ckpt-snap-1"
    assert int(span[PAYLOAD_START_COLUMN + 1]) == 5
    assert int(span[PAYLOAD_START_COLUMN + 2]) == 1


def test_prepare_snapshot_requires_position():
    events = []
    orchestrator, _, _ = _orchestrator(events)
    cut = CheckpointCut(cut_id="cut-none", position=None)

    with pytest.raises(ValueError, match="checkpoint cut requires a valid position"):
        asyncio.run(
            orchestrator.prepare_snapshot(
                "ckpt-snap-none",
                "attempt",
                CheckpointCommitPermit("attempt", 1, b"permit"),
                cut,
                "/tmp/repository",
            )
        )


@pytest.mark.parametrize(
    ("source_workers", "target_workers"),
    [(2, 2), (2, 4)],
)
def test_profiled_driver_restore_same_and_resized_topology(
    source_workers, target_workers
):
    checkpoint_id = f"ckpt-restore-{source_workers}-to-{target_workers}"
    events = []
    restore_calls = []

    async def restore_dispatch(*args):
        restore_calls.append(args)

    driver = SimpleNamespace(
        spmd_id="reader",
        _submit_driver_effect=lambda op, args=(), **_: op(*args),
        _schedule_driver_component_state=lambda s: events.append(
            ("component_state", s)
        ),
        _report_error=lambda f: events.append(("driver_error", f)),
    )
    machine = DriverStateMachine(driver)
    machine.require_transition(DriverEvent.INIT_COMPLETE)

    recorder = TraceRecorder(
        TraceArena(buffer_bytes=64 * 1024),
        run_id="test-run",
        process_name="driver/reader",
    )
    profiler_runtime = ProfilerRuntimeFactory(recorder)

    topology = TopologySnapshot(
        world_size=source_workers,
        mesh_axis_names=("dp",),
        mesh_axis_shapes=(source_workers,),
    )
    view = SimpleNamespace(
        world_size=target_workers,
        mesh=DeviceMesh(("dp",), (target_workers,)),
    )
    executor = SimpleNamespace(
        get_worker_handles=lambda: dict.fromkeys(range(target_workers)),
        checkpoint_restore=restore_dispatch,
        config=SimpleNamespace(require_graph_view=lambda: view),
    )
    registry = _Registry(events)
    sync = _Synchronizer(events)

    async def notify(s):
        events.append(("component_state", s))

    orchestrator = SnapshotOrchestrator(
        spmd_id="reader",
        executor=executor,
        synchronizer=sync,
        controller_handle=None,
        state_machine=machine,
        profiler_runtime=profiler_runtime,
        component_registry=registry,
        notify_component_state=notify,
    )

    cut = CheckpointCut.for_position("cut-111", 111)
    snapshot = SpmdCheckpoint(source_topology=topology, runtime_state={}, components={})

    asyncio.run(
        orchestrator.restore_from_manifest(
            checkpoint_id, cut, snapshot, "/tmp/repository"
        )
    )

    assert len(restore_calls) == 1
    assert restore_calls[0][1] == checkpoint_id
    assert machine.state is DriverState.READY

    rows = recorder.arena.snapshot()
    names = [recorder.metadata.name_for(int(row[4])) for row in rows]
    assert names == ["driver.restore"]

    span = rows[0]
    payload_fields = recorder.metadata.payload_fields_for(int(span[4]))
    assert payload_fields == ("checkpoint_id", "worker_count")
    assert recorder.metadata.name_for(int(span[PAYLOAD_START_COLUMN])) == checkpoint_id
    assert int(span[PAYLOAD_START_COLUMN + 1]) == target_workers
