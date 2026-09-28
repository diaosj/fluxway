import pytest
import jax.numpy as jnp

from fluxway.common.types.checkpoint import (
    CheckpointCommitPermit,
    CheckpointCut,
    TopologySnapshot,
)

from fluxway.control_plane.checkpointer import (
    OrbaxCheckpointComponent,
    WorkerLocalCheckpointComponent,
)
from fluxway.control_plane.checkpointer.component import CheckpointComponentRegistry
from fluxway.spmd.worker.checkpoint import WorkerCheckpointCoordinator


class _Owner:
    def __init__(self) -> None:
        self.training = {"model_state": {"weight": jnp.array([1.0, 2.0])}}
        self.local = {"reader_offset": 7}
        self.restored = {}


def test_worker_can_register_orbax_and_worker_local_components() -> None:
    orbax = OrbaxCheckpointComponent(
        "training",
        lambda owner, session: owner,
        lambda owner, state, session: None,
        lambda owner, session: owner,
    )
    local = WorkerLocalCheckpointComponent(
        "worker-local",
        lambda owner, session: owner,
        lambda owner, state, session: None,
    )
    registry = CheckpointComponentRegistry((orbax.registration(), local.registration()))
    assert tuple(registration.descriptor.component_id for registration in registry) == (
        "training",
        "worker-local",
    )
    assert orbax.descriptor.restore.topology.value == "global_reshard"
    assert local.descriptor.restore.topology.value == "exact_topology"


@pytest.mark.asyncio
async def test_worker_checkpoint_coordinator_round_trips_both_components(
    tmp_path,
) -> None:
    owner = _Owner()
    orbax = OrbaxCheckpointComponent(
        "training",
        lambda value, session: value.training,
        lambda value, state, session: value.restored.__setitem__("training", state),
        lambda value, session: {"model_state": {"weight": jnp.zeros((2,))}},
    )
    local = WorkerLocalCheckpointComponent(
        "worker-local",
        lambda value, session: value.local,
        lambda value, state, session: value.restored.__setitem__("local", state),
    )
    topology = TopologySnapshot(1, (), (), "v1")
    coordinator = WorkerCheckpointCoordinator(
        (orbax.registration(), local.registration()),
        owner,
        str(tmp_path),
        "train",
        0,
        topology,
    )
    permit = CheckpointCommitPermit("attempt", 1, b"permit")
    cut = CheckpointCut.for_step("cut", 1)
    await coordinator.prepare_save("checkpoint", "attempt", permit, cut)
    records = await coordinator.save()
    await coordinator.finish_save()
    publications = await coordinator.commit_save(
        {component_id: {0: result} for component_id, result in records.items()}
    )
    await coordinator.publish_save()
    await coordinator.restore("checkpoint", cut, publications, topology, topology)
    assert owner.restored["training"]["model_state"]["weight"].tolist() == [1.0, 2.0]
    assert owner.restored["local"] == {"reader_offset": 7}
    await coordinator.close()
