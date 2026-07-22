from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from fluxway.common.types import DriverEvent, DriverState
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
        return {"reader:read": {"cursor": {"snapshot_id": 3}}}

    def restore_state(self, state):
        self._events.append("component_restore")
        self.restored = state


class _Executor:
    def __init__(self, events):
        self._events = events
        self.fail_prepare = False
        self.fail_save = False

    def get_worker_handles(self):
        return [object()]

    async def set_drain_target_count(self, target):
        self._events.append(("worker_drain", target))

    def clear_drain_target_count(self):
        self._events.append("worker_drain_clear")

    async def prepare_checkpoint_artifacts(self, ckpt_dir, step):
        self._events.append("worker_prepare")
        if self.fail_prepare:
            raise RuntimeError("prepare failed")

    async def save_state_to_disk(self, ckpt_dir, step):
        self._events.append("worker_save")
        if self.fail_save:
            raise RuntimeError("save failed")
        return [(0, {"completed_step_count": {"read": step}}, {"rank": 0}, {})]

    async def commit_checkpoint_artifacts(self, ckpt_dir, step, metadata):
        self._events.append("worker_commit")

    async def abort_checkpoint_artifacts(self, ckpt_dir, step):
        self._events.append("worker_abort")

    async def load_state_from_disk(self, spmd_state, rank_storage, step):
        self._events.append("worker_restore")


class _Synchronizer:
    def __init__(self, events, on_drain=None):
        self._events = events
        self._on_drain = on_drain

    def drain_to_step_count(self, target_count, on_drained):
        self._events.append(("synchronizer_drain", target_count))
        if self._on_drain is not None:
            self._on_drain()
        on_drained()

    def clear_drain(self):
        self._events.append("synchronizer_clear")

    def load_spmd_state(self, state):
        self._events.append("synchronizer_restore")


def _orchestrator(events, *, state=DriverState.RUNNING):
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

    orchestrator = SnapshotOrchestrator(
        spmd_id="reader",
        executor=_Executor(events),
        synchronizer=_Synchronizer(events),
        checkpoint_backend=SimpleNamespace(_force_write=False),
        controller_handle=controller,
        get_driver_state=lambda: state_box["value"],
        trigger_state_event=trigger,
        profiler_runtime=SimpleNamespace(enabled=False),
        component_registry=registry,
        notify_component_state=notify,
    )
    return orchestrator, controller, registry


def test_component_is_quiesced_before_driver_reports_drained() -> None:
    events = []
    orchestrator, _, _ = _orchestrator(events)

    asyncio.run(orchestrator._request_drain_unprofiled(5))

    assert events.index(("component_state", DriverState.DRAINING)) < events.index(
        "report_drained"
    )


def test_snapshot_envelope_contains_driver_component_state(tmp_path) -> None:
    events = []
    orchestrator, controller, _ = _orchestrator(events, state=DriverState.SNAPSHOTTING)
    orchestrator._pending_snapshot_id = "snapshot"
    orchestrator._pending_target_step_count = 5
    orchestrator._pending_ckpt_dir = str(tmp_path)

    asyncio.run(orchestrator._persist_worker_states())

    rank_metadata = controller.receive_snapshot_metadata.calls[0][2]
    spmd_state = rank_metadata[0][0]
    assert spmd_state["driver_components"] == {
        "reader:read": {"cursor": {"snapshot_id": 3}}
    }


def test_driver_components_restore_before_worker_state() -> None:
    events = []
    orchestrator, _, registry = _orchestrator(events)
    spmd_state = {
        "completed_step_count": {"read": 5},
        "driver_components": {"reader:read": {"cursor": {"snapshot_id": 3}}},
    }

    asyncio.run(orchestrator._restore_from_disk_unprofiled(spmd_state, {0: {}}, 5))

    assert registry.restored == spmd_state["driver_components"]
    assert events.index(("component_state", DriverState.DRAINING)) < events.index(
        "component_restore"
    )
    assert events.index("component_restore") < events.index("worker_restore")
    assert events.index("worker_restore") < events.index(
        ("component_state", DriverState.RUNNING)
    )


def test_driver_component_restore_rejects_missing_state() -> None:
    events = []
    orchestrator, _, _ = _orchestrator(events)

    with pytest.raises(ValueError, match="driver_components"):
        asyncio.run(
            orchestrator._restore_from_disk_unprofiled(
                {"completed_step_count": {"read": 5}},
                {0: {}},
                5,
            )
        )


@pytest.mark.parametrize("phase", ["prepare", "save"])
def test_driver_aborts_artifacts_when_worker_persistence_fails(tmp_path, phase) -> None:
    events = []
    orchestrator, _, _ = _orchestrator(events, state=DriverState.SNAPSHOTTING)
    orchestrator._pending_snapshot_id = "snapshot"
    orchestrator._pending_target_step_count = 5
    orchestrator._pending_ckpt_dir = str(tmp_path)
    setattr(orchestrator.executor, f"fail_{phase}", True)

    asyncio.run(orchestrator._persist_worker_states())

    assert events.index("worker_abort") < events.index("snapshot_failed")
