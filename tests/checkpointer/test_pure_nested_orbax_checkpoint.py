import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from fluxway.common.types.checkpoint import (
    CheckpointCommitPermit,
    CheckpointCut,
    CheckpointRuntime,
    CheckpointRuntimeRole,
    CommitSaveSession,
    RestoreSession,
    SaveSession,
    TopologySnapshot,
)
from fluxway.control_plane.checkpointer.orbax_component import OrbaxCheckpointComponent


@pytest.mark.asyncio
async def test_pure_nested_orbax_checkpoint_end_to_end(tmp_path):
    saved_state = {
        "model_state": {
            "encoder": {
                "dense": {
                    "kernel": jnp.array(
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32
                    ),
                    "bias": jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32),
                }
            }
        }
    }
    restored_container = []

    async def save_fn(owner, session):
        return saved_state

    async def restore_fn(owner, state, session):
        restored_container.append(state)

    async def layout_fn(owner, session):
        return {
            "model_state": {
                "encoder": {
                    "dense": {
                        "kernel": jax.ShapeDtypeStruct((2, 3), jnp.float32),
                        "bias": jax.ShapeDtypeStruct((3,), jnp.float32),
                    }
                }
            }
        }

    component = OrbaxCheckpointComponent("dense", save_fn, restore_fn, layout_fn)
    runtime = CheckpointRuntime(CheckpointRuntimeRole.WORKER, str(tmp_path), "train", 0)
    await component.open(runtime)

    topology = TopologySnapshot(1, (), ())
    permit = CheckpointCommitPermit("tx_attempt", 1, b"permit")
    save_session = SaveSession(
        instance=None,
        attempt_id="tx_attempt",
        checkpoint_id="snap_1",
        commit_permit=permit,
        cut=CheckpointCut.for_step("cut", 1),
        spmd_id="train",
        rank=0,
        source_topology=topology,
        component_id="dense",
    )

    await component.prepare_save(save_session)
    prepared = await component.save(save_session)
    await component.finish_save(save_session)

    commit_session = CommitSaveSession(
        instance=None,
        attempt_id=save_session.attempt_id,
        checkpoint_id=save_session.checkpoint_id,
        commit_permit=save_session.commit_permit,
        cut=save_session.cut,
        spmd_id=save_session.spmd_id,
        rank=0,
        source_topology=topology,
        component_id="dense",
        worker_results={0: prepared},
    )
    publication = await component.commit_save(commit_session)

    # 1. Publication validation: single collective shared record, zero partitions
    assert len(publication.shared_records) == 1
    assert publication.partitions == ()
    shared_locator = publication.shared_records[0].checkpoint_objects[0].storage_locator
    shared_path = component._file_path(shared_locator)
    assert shared_path.is_dir()

    # 2. Downstream eval verification: standard Orbax checkpointer can directly read model_state without layout
    ckptr = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    eval_restored = ckptr.restore(shared_path)
    ckptr.close()

    assert "model_state" in eval_restored
    assert "encoder" in eval_restored["model_state"]
    assert "dense" in eval_restored["model_state"]["encoder"]
    np.testing.assert_allclose(
        eval_restored["model_state"]["encoder"]["dense"]["kernel"],
        saved_state["model_state"]["encoder"]["dense"]["kernel"],
    )
    np.testing.assert_allclose(
        eval_restored["model_state"]["encoder"]["dense"]["bias"],
        saved_state["model_state"]["encoder"]["dense"]["bias"],
    )

    # 3. Component restore verification: restores exact nested structure
    restore_session = RestoreSession(
        instance=None,
        checkpoint_id=save_session.checkpoint_id,
        cut=save_session.cut,
        spmd_id="train",
        rank=0,
        saved_topology=topology,
        restore_topology=topology,
        component_id="dense",
        component_checkpoint=publication,
    )
    await component.restore(restore_session)
    await component.close()

    assert len(restored_container) == 1
    comp_restored = restored_container[0]
    np.testing.assert_allclose(
        comp_restored["model_state"]["encoder"]["dense"]["kernel"],
        saved_state["model_state"]["encoder"]["dense"]["kernel"],
    )
