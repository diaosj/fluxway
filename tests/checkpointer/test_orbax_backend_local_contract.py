import asyncio
from dataclasses import replace

import numpy as np
import pytest

from fluxway.common.types.checkpoint import (
    CheckpointRuntime,
    CheckpointRuntimeRole,
    CheckpointCommitPermit,
    SaveSession,
    CommitSaveSession,
    RestoreSession,
    CheckpointCut,
    CheckpointObjectRecord,
    CheckpointObjectRef,
    TopologyMismatchError,
    TopologySnapshot,
)
from fluxway.control_plane.checkpointer import (
    OrbaxCheckpointComponent,
    WorkerLocalCheckpointComponent,
)
from fluxway.control_plane.checkpointer import orbax_component as orbax_module


@pytest.mark.asyncio
async def test_worker_local_records_require_exact_topology(tmp_path):
    async def save(module, context):
        return {"local": np.array([context.rank])}

    async def restore(module, state, context):
        return None

    source = TopologySnapshot(1, (), ())
    changed = TopologySnapshot(2, ("data",), (2,))
    save_session = SaveSession(
        instance=None,
        attempt_id="tx",
        checkpoint_id="snapshot",
        commit_permit=CheckpointCommitPermit("tx", 1, b"permit"),
        cut=CheckpointCut.for_step("cut", 1),
        spmd_id="train",
        rank=0,
        source_topology=source,
        component_id="local",
    )
    restore_context = RestoreSession(
        instance=None,
        checkpoint_id="snapshot",
        cut=save_session.cut,
        spmd_id="train",
        rank=0,
        saved_topology=source,
        restore_topology=changed,
        component_id="local",
        component_checkpoint=None,
    )
    component = WorkerLocalCheckpointComponent("local", save, restore)
    await component.open(
        CheckpointRuntime(CheckpointRuntimeRole.WORKER, str(tmp_path), "train", 0)
    )
    await component.prepare_save(save_session)
    prepared = await component.save(save_session)
    await component.finish_save(save_session)
    publication = await component.commit_save(
        CommitSaveSession(
            instance=save_session.instance,
            attempt_id=save_session.attempt_id,
            checkpoint_id=save_session.checkpoint_id,
            commit_permit=save_session.commit_permit,
            cut=save_session.cut,
            spmd_id=save_session.spmd_id,
            rank=save_session.rank,
            source_topology=save_session.source_topology,
            component_id=save_session.component_id,
            worker_results={0: prepared},
        )
    )
    restore_context = replace(restore_context, component_checkpoint=publication)
    with pytest.raises(TopologyMismatchError):
        await component.restore(restore_context)
    await component.close()


@pytest.mark.asyncio
async def test_orbax_rollback_is_safe_when_cleanup_races(tmp_path, monkeypatch):
    async def save(module, context):
        return {"weights": np.array([1.0])}

    async def restore(module, state, context):
        return None

    async def layout(module, context):
        return {"weights": np.array([1.0])}

    topology = TopologySnapshot(1, (), ())
    session = SaveSession(
        instance=None,
        attempt_id="tx",
        checkpoint_id="snapshot",
        commit_permit=CheckpointCommitPermit("tx", 1, b"permit"),
        cut=CheckpointCut.for_step("cut", 1),
        spmd_id="train",
        rank=0,
        source_topology=topology,
        component_id="orbax",
    )
    component = OrbaxCheckpointComponent("orbax", save, restore, layout)
    await component.open(
        CheckpointRuntime(CheckpointRuntimeRole.WORKER, str(tmp_path), "train", 0)
    )
    staging = component._staging_root(session.attempt_id) / "shared"
    staging.mkdir(parents=True)
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    waiting = 0

    async def synchronized_to_thread(function, path):
        nonlocal waiting
        waiting += 1
        if waiting == 2:
            cleanup_started.set()
        await release_cleanup.wait()
        return function(path)

    monkeypatch.setattr(orbax_module.asyncio, "to_thread", synchronized_to_thread)
    first = asyncio.create_task(component.rollback_save(session))
    second = asyncio.create_task(component.rollback_save(session))
    await cleanup_started.wait()
    release_cleanup.set()
    await asyncio.gather(first, second)
    await component.close()


@pytest.mark.asyncio
async def test_orbax_rollback_removes_regular_completion_marker(tmp_path):
    async def save(module, context):
        return {"weights": np.array([1.0])}

    async def restore(module, state, context):
        return None

    async def layout(module, context):
        return {"weights": np.array([1.0])}

    session = SaveSession(
        instance=None,
        attempt_id="tx",
        checkpoint_id="snapshot",
        commit_permit=CheckpointCommitPermit("tx", 1, b"permit"),
        cut=CheckpointCut.for_step("cut", 1),
        spmd_id="train",
        rank=0,
        source_topology=TopologySnapshot(1, (), ()),
        component_id="orbax",
    )
    component = OrbaxCheckpointComponent("orbax", save, restore, layout)
    await component.open(
        CheckpointRuntime(CheckpointRuntimeRole.WORKER, str(tmp_path), "train", 0)
    )
    marker = component._staging_root(session.attempt_id) / ".shared-save-complete"
    marker.parent.mkdir(parents=True)
    marker.write_text("snapshot", encoding="utf-8")

    await component.rollback_save(session)

    assert not marker.exists()
    await component.close()


@pytest.mark.asyncio
async def test_orbax_restore_rejects_noncollective_topology(tmp_path, monkeypatch):
    async def save(module, context):
        return {"weights": np.array([1.0])}

    async def restore(module, state, context):
        return None

    async def layout(module, context):
        return {"weights": np.array([1.0])}

    monkeypatch.setattr(orbax_module, "_jax_process_count", lambda: 1)
    component = OrbaxCheckpointComponent("orbax", save, restore, layout)
    context = RestoreSession(
        instance=None,
        checkpoint_id="snapshot",
        cut=CheckpointCut.for_step("cut", 1),
        spmd_id="train",
        rank=0,
        saved_topology=TopologySnapshot(2, ("data",), (2,)),
        restore_topology=TopologySnapshot(2, ("data",), (2,)),
        component_id="orbax",
        component_checkpoint=None,
    )
    record = CheckpointObjectRecord(
        checkpoint_objects=(
            CheckpointObjectRef(
                component_id="orbax",
                checkpoint_object_id="state",
                version_id="snapshot",
                storage_locator=b"file:///tmp/missing",
            ),
        )
    )
    with pytest.raises(RuntimeError, match="one initialized JAX process"):
        await component._read_global(record, {}, context)
