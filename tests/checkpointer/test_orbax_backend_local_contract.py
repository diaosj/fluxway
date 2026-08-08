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
    TopologyMismatchError,
    TopologySnapshot,
)
from fluxway.control_plane.checkpointer.orbax import OrbaxCheckpointComponent


@pytest.mark.asyncio
async def test_orbax_local_records_require_exact_topology(tmp_path):
    async def save(module, context):
        return {"local": np.array([context.rank])}

    async def restore(module, state, context):
        return None

    async def layout(module, context):
        return {"local": np.array([0])}

    source = TopologySnapshot(1, (), (), "v1")
    changed = TopologySnapshot(2, ("data",), (2,), "v1")
    save_session = SaveSession(
        instance=None,
        attempt_id="tx",
        checkpoint_id="snapshot",
        commit_permit=CheckpointCommitPermit("tx", 1, b"permit"),
        cut=CheckpointCut.for_step("cut", 1),
        spmd_id="train",
        rank=0,
        source_topology=source,
        component_id="orbax",
    )
    restore_context = RestoreSession(
        instance=None,
        checkpoint_id="snapshot",
        cut=save_session.cut,
        spmd_id="train",
        rank=0,
        saved_topology=source,
        restore_topology=changed,
        component_id="orbax",
        component_checkpoint=None,
    )
    backend = OrbaxCheckpointComponent("orbax", save, restore, layout)
    await backend.open(
        CheckpointRuntime(CheckpointRuntimeRole.WORKER, str(tmp_path), "train", 0)
    )
    await backend.start_save(save_session)
    prepared = await backend.save(save_session)
    await backend.finish_save(save_session)
    publication = await backend.commit_save(
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
        await backend.restore(restore_context)
    await backend.close()
