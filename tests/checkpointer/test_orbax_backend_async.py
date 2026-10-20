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
    CheckpointObjectRecord,
    TopologySnapshot,
)
from fluxway.control_plane.checkpointer.orbax_component import (
    OrbaxCheckpointComponent,
    _barrier_sync_key,
)
import fluxway.control_plane.checkpointer.orbax_component as orbax_component


@pytest.mark.asyncio
async def test_orbax_component_commits_and_restores_nested_pytree_state(tmp_path):
    import jax
    import jax.numpy as jnp

    restored: list[dict[str, object]] = []

    saved_state = {
        "model_state": {
            "linear": {
                "w": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
                "b": jnp.array([0.5, -0.5], dtype=jnp.float32),
            }
        }
    }

    async def save(module, context):
        return saved_state

    async def restore(module, state, context):
        restored.append(state)

    async def layout(module, context):
        return {
            "model_state": {
                "linear": {
                    "w": jax.ShapeDtypeStruct((2, 2), jnp.float32),
                    "b": jax.ShapeDtypeStruct((2,), jnp.float32),
                }
            }
        }

    topology = TopologySnapshot(1, (), ())
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
    component = OrbaxCheckpointComponent("orbax", save, restore, layout)
    await component.open(
        CheckpointRuntime(CheckpointRuntimeRole.WORKER, str(tmp_path), "train", 0)
    )
    await component.prepare_save(context)
    prepared = await component.save(context)
    await component.finish_save(context)
    publication = await component.commit_save(
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
    assert len(publication.shared_records) == 1
    assert publication.partitions == ()

    await component.restore(
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
    await component.close()

    assert len(restored) == 1
    np.testing.assert_allclose(
        restored[0]["model_state"]["linear"]["w"],
        saved_state["model_state"]["linear"]["w"],
    )
    np.testing.assert_allclose(
        restored[0]["model_state"]["linear"]["b"],
        saved_state["model_state"]["linear"]["b"],
    )


@pytest.mark.asyncio
async def test_orbax_global_state_uses_one_collective_shared_checkpoint_object(
    tmp_path, monkeypatch
):
    import jax
    import orbax.checkpoint as ocp

    monkeypatch.setattr(orbax_component, "_jax_process_count", lambda: 2)

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

    topology = TopologySnapshot(2, ("data",), (2,))
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
    components = tuple(
        OrbaxCheckpointComponent("dense", save, restore, layout) for _ in range(2)
    )
    for rank, component in enumerate(components):
        await component.open(
            CheckpointRuntime(
                CheckpointRuntimeRole.WORKER, str(tmp_path), "train", rank
            )
        )
    await components[0].prepare_save(contexts[0])
    prepared = await asyncio.gather(
        *(component.save(contexts[rank]) for rank, component in enumerate(components))
    )
    for rank, component in enumerate(components):
        await component.finish_save(contexts[rank])
    context = contexts[0]
    with pytest.raises(ValueError, match="every rank record"):
        await components[0].commit_save(
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
                worker_results={
                    0: prepared[0],
                    1: replace(prepared[1], record=CheckpointObjectRecord()),
                },
            )
        )
    publication = await components[0].commit_save(
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

    assert len(saved_paths) == 2
    assert len(set(saved_paths)) == 1
    assert len(publication.shared_records) == 1
    assert publication.partitions == ()
    shared_checkpoint_objects = publication.shared_records[0].checkpoint_objects
    assert len(shared_checkpoint_objects) == 1
    assert shared_checkpoint_objects[0].partition_key is None
    for component in components:
        await component.close()


@pytest.mark.asyncio
async def test_orbax_model_output_contains_model_state_only(tmp_path, monkeypatch):
    import orbax.checkpoint as ocp

    saved: list[tuple[Path, object]] = []

    class Handler:
        def close(self):
            return None

    class Checkpointer:
        def __init__(self, handler, multiprocessing_options):
            self.handler = handler

        def save(self, path, force, args):
            destination = Path(str(path))
            assert not destination.exists()
            assert force is False
            destination.mkdir(parents=True, exist_ok=False)
            saved.append((destination, args))

        def close(self):
            return None

    monkeypatch.setattr(ocp, "Checkpointer", Checkpointer)
    monkeypatch.setattr(ocp.args, "StandardSave", lambda value: ("save", value))
    monkeypatch.setattr(
        OrbaxCheckpointComponent, "_get_handler", lambda self: Handler()
    )

    model_state = {"weights": [1, 2]}
    optimizer_state = {"momentum": [3, 4]}

    async def save(module, context):
        return {
            "step": 9,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
        }

    async def restore(module, state, context):
        return None

    async def layout(module, context):
        return {"model_state": model_state}

    topology = TopologySnapshot(1, (), ())
    session = SaveSession(
        instance=None,
        attempt_id="attempt",
        checkpoint_id="snapshot",
        commit_permit=CheckpointCommitPermit("attempt", 1, b"permit"),
        cut=CheckpointCut.for_step("cut", 9),
        spmd_id="train",
        rank=0,
        source_topology=topology,
        component_id="model",
        repository_root=str(tmp_path),
        staging_output_path=str(tmp_path / "steps" / ".snapshot.attempt" / "model"),
        published_output_path=str(tmp_path / "steps" / "9" / "model"),
    )
    component = OrbaxCheckpointComponent(
        "model", save, restore, layout, output_path="model"
    )
    await component.open(
        CheckpointRuntime(CheckpointRuntimeRole.WORKER, str(tmp_path), "train", 0)
    )
    await component.save(session)
    assert len(saved) == 2
    by_name = {path.name: args for path, args in saved}
    assert by_name["model"] == ("save", model_state)
    assert by_name["shared"] == (
        "save",
        {"step": 9, "model_state": model_state, "optimizer_state": optimizer_state},
    )
    await component.close()


def test_orbax_barrier_identity_scopes_component_snapshot_and_operation():
    topology = TopologySnapshot(2, ("data",), (2,))
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
