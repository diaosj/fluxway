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
from fluxway.control_plane.checkpointer.safetensors_component import (
    SafeTensorsCheckpointComponent,
    _flatten_worker_state,
    _unflatten_worker_state,
    load_worker_state,
    save_worker_state,
)


def test_safetensors_format_round_trips_nested_tensor_state(tmp_path):
    state = {
        "weights": np.array([1.0, 2.0], dtype=np.float32),
        "config": {"step": 3, "enabled": True},
    }
    tensors = tmp_path / "state.safetensors"
    metadata = tmp_path / "state.meta.json"

    save_worker_state(state, str(tensors), str(metadata))
    restored = load_worker_state(str(tensors), str(metadata))

    np.testing.assert_array_equal(restored["weights"], state["weights"])
    assert restored["config"] == state["config"]


def test_safetensors_flatten_round_trips_small_pure_state():
    state = {"value": np.array([3]), "nested": {"step": 7}}
    tensors, metadata = _flatten_worker_state(state)
    restored = _unflatten_worker_state(tensors, metadata)
    np.testing.assert_array_equal(restored["value"], state["value"])
    assert restored["nested"] == {"step": 7}


@pytest.mark.asyncio
async def test_safetensors_component_uses_transaction_context(tmp_path):
    restored: list[dict[str, object]] = []

    async def save(module, context):
        assert context.cut.logical_step == 5
        return {"value": np.array([5], dtype=np.int64)}

    async def restore(module, state, context):
        restored.append(state)

    topology = TopologySnapshot(1, (), (), "v1")
    save_session = SaveSession(
        instance=None,
        attempt_id="tx",
        checkpoint_id="snapshot",
        commit_permit=CheckpointCommitPermit("tx", 1, b"permit"),
        cut=CheckpointCut.for_step("tx", 5),
        spmd_id="train",
        rank=0,
        source_topology=topology,
        component_id="state",
    )
    component = SafeTensorsCheckpointComponent("state", save, restore)
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
    await component.restore(
        RestoreSession(
            instance=None,
            checkpoint_id=save_session.checkpoint_id,
            cut=save_session.cut,
            spmd_id=save_session.spmd_id,
            rank=save_session.rank,
            saved_topology=topology,
            restore_topology=topology,
            component_id=save_session.component_id,
            component_checkpoint=publication,
        )
    )
    await component.close()

    np.testing.assert_array_equal(restored[0]["value"], np.array([5]))
