import asyncio
from dataclasses import replace
from pathlib import Path

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
    TopologySnapshot,
)
from fluxway.control_plane.checkpointer.orbax import (
    OrbaxCheckpointComponent,
    _barrier_sync_key,
)


@pytest.mark.asyncio
async def test_orbax_backend_commits_and_restores_rank_local_state(tmp_path):
    restored: list[dict[str, object]] = []

    async def save(module, context):
        return {"optimizer": np.array([3], dtype=np.int64)}

    async def restore(module, state, context):
        restored.append(state)

    async def layout(module, context):
        return {"optimizer": np.array([0], dtype=np.int64)}

    topology = TopologySnapshot(1, (), (), "v1")
    context = SaveSession(
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
    backend = OrbaxCheckpointComponent("orbax", save, restore, layout)
    await backend.open(
        CheckpointRuntime(CheckpointRuntimeRole.WORKER, str(tmp_path), "train", 0)
    )
    await backend.start_save(context)
    prepared = await backend.save(context)
    await backend.finish_save(context)
    publication = await backend.commit_save(
        CommitSaveSession(
            instance=context.instance,
            attempt_id=context.attempt_id,
            checkpoint_id=context.checkpoint_id,
            commit_permit=context.commit_permit,
            cut=context.cut,
            spmd_id=context.spmd_id,
            rank=context.rank,
            source_topology=context.source_topology,
            component_id=context.component_id,
            worker_results={0: prepared},
        )
    )
    await backend.restore(
        RestoreSession(
            instance=None,
            checkpoint_id=context.checkpoint_id,
            cut=context.cut,
            spmd_id=context.spmd_id,
            rank=context.rank,
            saved_topology=topology,
            restore_topology=topology,
            component_id=context.component_id,
            component_checkpoint=publication,
        )
    )
    await backend.close()

    np.testing.assert_array_equal(restored[0]["optimizer"], np.array([3]))


@pytest.mark.asyncio
async def test_orbax_global_state_uses_one_collective_shared_checkpoint_object(
    tmp_path, monkeypatch
):
    import jax
    import orbax.checkpoint as ocp

    saved_paths = []

    class Handler:
        def close(self):
            return None

    class Checkpointer:
        def __init__(self, handler, multiprocessing_options):
            self.handler = handler

        def save(self, path, force, args):
            destination = Path(str(path))
            destination.mkdir(parents=True, exist_ok=True)
            saved_paths.append(destination)

        def close(self):
            return None

    monkeypatch.setattr(ocp, "Checkpointer", Checkpointer)
    monkeypatch.setattr(
        OrbaxCheckpointComponent, "_get_handler", lambda self: Handler()
    )

    async def save(module, context):
        return {"weights": jax.ShapeDtypeStruct((8, 4), np.float32)}

    async def restore(module, state, context):
        return None

    async def layout(module, context):
        return {"weights": jax.ShapeDtypeStruct((8, 4), np.float32)}

    topology = TopologySnapshot(2, ("data",), (2,), "v1")
    permit = CheckpointCommitPermit("attempt", 1, b"permit")
    contexts = tuple(
        SaveSession(
            instance=None,
            attempt_id="attempt",
            checkpoint_id="snapshot",
            commit_permit=permit,
            cut=CheckpointCut.for_step("cut", 1),
            spmd_id="train",
            rank=rank,
            source_topology=topology,
            component_id="dense",
        )
        for rank in range(2)
    )
    backends = tuple(
        OrbaxCheckpointComponent("dense", save, restore, layout) for _ in range(2)
    )
    for rank, backend in enumerate(backends):
        await backend.open(
            CheckpointRuntime(
                CheckpointRuntimeRole.WORKER, str(tmp_path), "train", rank
            )
        )
    await backends[0].start_save(contexts[0])
    prepared = await asyncio.gather(
        *(backend.save(contexts[rank]) for rank, backend in enumerate(backends))
    )
    for rank, backend in enumerate(backends):
        await backend.finish_save(contexts[rank])
    context = contexts[0]
    publication = await backends[0].commit_save(
        CommitSaveSession(
            instance=context.instance,
            attempt_id=context.attempt_id,
            checkpoint_id=context.checkpoint_id,
            commit_permit=context.commit_permit,
            cut=context.cut,
            spmd_id=context.spmd_id,
            rank=context.rank,
            source_topology=context.source_topology,
            component_id=context.component_id,
            worker_results=dict(enumerate(prepared)),
        )
    )

    assert len(saved_paths) == topology.world_size
    assert len(set(saved_paths)) == 1
    assert len(publication.shared_records) == 1
    assert publication.partitions == ()
    shared_checkpoint_objects = publication.shared_records[0].checkpoint_objects
    assert len(shared_checkpoint_objects) == 1
    assert shared_checkpoint_objects[0].partition_key is None
    for backend in backends:
        await backend.close()


def test_orbax_barrier_identity_scopes_component_snapshot_and_operation():
    topology = TopologySnapshot(2, ("data",), (2,), "v1")
    permit = CheckpointCommitPermit("attempt", 1, b"permit")
    context = SaveSession(
        instance=None,
        attempt_id="attempt",
        checkpoint_id="snapshot",
        commit_permit=permit,
        cut=CheckpointCut.for_step("cut", 1),
        spmd_id="train",
        rank=0,
        source_topology=topology,
        component_id="dense-a",
    )

    save_key = _barrier_sync_key(context, "save")
    assert save_key == _barrier_sync_key(replace(context, rank=1), "save")
    assert save_key != _barrier_sync_key(
        replace(context, component_id="dense-b"), "save"
    )
    assert save_key != _barrier_sync_key(
        replace(context, checkpoint_id="other-snapshot"), "save"
    )
    assert save_key != _barrier_sync_key(context, "restore")
