import asyncio
from types import SimpleNamespace

from fluxway.common.types import (
    CheckpointCaptureMode,
    CheckpointCommitPermit,
    CheckpointCut,
    DriverEvent,
    DriverState,
    SpmdCheckpoint,
    TopologySnapshot,
)
from fluxway.spmd.driver.snapshot.orchestrator import SnapshotOrchestrator


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

    def checkpoint_state(self):
        self._events.append("component_checkpoint")
        return {"reader:read": {"cursor": {"checkpoint_id": 3}}}

    def restore_state(self, state):
        self._events.append("component_restore")
        self.restored = state


class _Executor:
    def __init__(self, events):
        self._events = events
        self.fail_capture = False
        mesh = SimpleNamespace(axis_names=("data",), shape=(1,))
        view = SimpleNamespace(mesh=mesh, world_size=1)
        self.config = SimpleNamespace(require_graph_view=lambda: view)

    def get_worker_handles(self):
        return {0: object()}

    async def set_drain_step_count(self, target):
        self._events.append(("worker_drain", target))

    def clear_drain_step_count(self):
        self._events.append("worker_drain_clear")

    async def checkpoint_prepare(
        self, root, checkpoint_id, attempt_id, permit, cut, capture_mode
    ):
        self._events.append("worker_prepare")

    async def checkpoint_capture(self):
        self._events.append("worker_capture")
        if self.fail_capture:
            raise RuntimeError("save failed")
        return [(0, {"completed_step_count": {"read": 5}}, {})]

    async def checkpoint_finalize(self):
        self._events.append("worker_finalize")

    async def checkpoint_commit(self, gathered):
        self._events.append("worker_commit")
        assert gathered == {}
        return {0: {}}

    async def checkpoint_abort(self):
        self._events.append("worker_abort")

    async def checkpoint_mark_published(self):
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


def _orchestrator(events, *, state=DriverState.DRAINED):
    state_box = {"value": state}
    controller = _Controller(events)
    registry = _Registry(events)

    def trigger(event):
        events.append(("driver_event", event))
        if event is DriverEvent.DRAIN_REQUESTED:
            state_box["value"] = DriverState.DRAINING
        elif event is DriverEvent.DRAIN_COMPLETE:
            state_box["value"] = DriverState.DRAINED

    async def notify(component_state):
        events.append(("component_state", component_state))

    return (
        SnapshotOrchestrator(
            spmd_id="reader",
            executor=_Executor(events),
            synchronizer=_Synchronizer(events),
            controller_handle=controller,
            get_driver_state=lambda: state_box["value"],
            trigger_state_event=trigger,
            profiler_runtime=SimpleNamespace(enabled=False),
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
        orchestrator.request_snapshot(
            "snapshot",
            "attempt",
            CheckpointCommitPermit("attempt", 1, b"permit"),
            cut,
            "/tmp/repository",
            CheckpointCaptureMode.FENCED,
        )
    )

    snapshot = controller.receive_snapshot_metadata.calls[0][2]
    assert isinstance(snapshot, SpmdCheckpoint)
    assert snapshot.runtime_state["driver_components"] == {
        "reader:read": {"cursor": {"checkpoint_id": 3}}
    }
    assert events.index("worker_finalize") < events.index("worker_commit")


def test_restore_components_precedes_backend_restore():
    events = []
    orchestrator, _, registry = _orchestrator(events)
    topology = TopologySnapshot(1, ("data",), (1,), "fluxway-device-mesh-v1")
    snapshot = SpmdCheckpoint(
        topology,
        {
            "completed_step_count": {"read": 5},
            "driver_components": {"reader:read": {"cursor": {"checkpoint_id": 3}}},
        },
        {},
    )

    asyncio.run(
        orchestrator._restore_from_manifest_unprofiled(
            "snapshot",
            CheckpointCut.for_step("restore", 5),
            snapshot,
            "/tmp/repository",
        )
    )

    assert registry.restored == {"reader:read": {"cursor": {"checkpoint_id": 3}}}
    assert events.index("component_restore") < events.index("worker_restore")


def test_capture_failure_aborts_backend_transaction():
    events = []
    orchestrator, _, _ = _orchestrator(events)
    orchestrator.executor.fail_capture = True

    asyncio.run(
        orchestrator.request_snapshot(
            "snapshot",
            "attempt",
            CheckpointCommitPermit("attempt", 1, b"permit"),
            CheckpointCut.for_step("tx", 5),
            "/tmp/repository",
            CheckpointCaptureMode.FENCED,
        )
    )

    assert events.index("worker_abort") < events.index("snapshot_failed")
