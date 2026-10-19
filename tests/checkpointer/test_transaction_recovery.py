import asyncio
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
import pytest_asyncio

from fluxway.common.types import DrainBoundary
from fluxway.common.types.checkpoint import (
    CheckpointCommitPermit,
    CheckpointComponentPublication,
    CheckpointCut,
    CheckpointManifest,
    CheckpointTransactionState,
    SpmdCheckpoint,
    StepReference,
    TopologySnapshot,
)
from fluxway.control_plane.checkpointer.component import CheckpointComponentRegistry
from fluxway.control_plane.checkpointer.inline import StateCheckpointComponent
from fluxway.control_plane.checkpointer.pickle_component import (
    WorkerLocalCheckpointComponent,
)
from fluxway.control_plane.checkpointer.repository import CheckpointRepository
from fluxway.control_plane.checkpointer.safetensors_component import (
    SafeTensorsCheckpointComponent,
)
from fluxway.spmd.worker.checkpoint import WorkerCheckpointCoordinator


class RecoveryOwner(StateCheckpointComponent):
    def __init__(self):
        super().__init__("state", lambda *_: {}, lambda *_: None)
        self.aborts = []
        self.acknowledgements = []
        self.fail_abort = False
        self.fail_acknowledge = False

    async def abort_save(self, permit):
        self.aborts.append(permit)
        if self.fail_abort:
            raise OSError("owner abort failed")
        await super().abort_save(permit)

    async def acknowledge_save(self, permit):
        assert isinstance(permit, CheckpointCommitPermit)
        self.acknowledgements.append(permit)
        if self.fail_acknowledge:
            raise OSError("owner acknowledgement failed")


def stage_attempt(repository, *, committed=False):
    permit = repository.issue_commit_permit(
        "attempt", repository.claim_coordinator_generation()
    )
    manifest = CheckpointManifest(
        checkpoint_id="snapshot",
        attempt_id=permit.attempt_id,
        coordinator_generation=permit.coordinator_generation,
        cut=CheckpointCut.for_step("cut", 10),
        checkpoint_sequence=1,
        references=(StepReference(10),),
        framework_state={},
        spmds={
            "train": SpmdCheckpoint(
                source_topology=TopologySnapshot(2, ("shard",), (2,)),
                runtime_state={
                    "drain_boundary": DrainBoundary(committed_cut=10).to_dict()
                },
                components={"state": CheckpointComponentPublication("state")},
            )
        },
    )
    repository.stage_transaction(manifest.checkpoint_id, manifest, permit)
    repository.publish(manifest, permit)
    repository.decide_transaction(manifest.checkpoint_id, permit)
    if committed:
        repository.mark_committed(manifest.checkpoint_id, permit)
    return manifest, permit


@pytest.mark.asyncio
async def test_inline_recovery_does_not_require_a_live_worker(
    checkpointer_factory, tmp_path
):
    repository = CheckpointRepository(tmp_path / "repository")
    manifest, _ = stage_attempt(repository)
    component = StateCheckpointComponent("state", lambda *_: {}, lambda *_: None)
    checkpointer = await checkpointer_factory(
        object(),
        str(repository.root),
        interval_steps=1,
        component_registries={
            "train": CheckpointComponentRegistry((component.registration(),))
        },
    )

    assert await checkpointer.restore_latest() is None
    assert repository.transaction(manifest.checkpoint_id).status is (
        CheckpointTransactionState.ABANDONED
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
async def test_failed_owner_recovery_preserves_the_transaction_for_retry(
    checkpointer_factory, tmp_path, committed
):
    repository = CheckpointRepository(tmp_path / "repository")
    manifest, permit = stage_attempt(repository, committed=committed)
    owner = RecoveryOwner()
    owner.fail_acknowledge = committed
    owner.fail_abort = not committed
    checkpointer = await checkpointer_factory(
        object(),
        str(repository.root),
        interval_steps=1,
        component_registries={
            "train": CheckpointComponentRegistry((owner.registration(),))
        },
    )

    with pytest.raises(OSError, match="owner .* failed"):
        await checkpointer._fence_incomplete_transactions()
    assert repository.transaction(manifest.checkpoint_id).status is (
        CheckpointTransactionState.COMMIT_DECIDED
    )
    assert repository.is_committed(manifest.checkpoint_id) is committed

    owner.fail_abort = owner.fail_acknowledge = False
    await checkpointer._fence_incomplete_transactions()
    assert repository.transaction(manifest.checkpoint_id).status is (
        CheckpointTransactionState.COMMITTED
        if committed
        else CheckpointTransactionState.ABANDONED
    )
    assert repository.is_committed(manifest.checkpoint_id) is committed
    assert owner.acknowledgements == ([permit, permit] if committed else [])
    assert owner.aborts == ([] if committed else [permit, permit])
    await checkpointer._fence_incomplete_transactions()
    assert owner.acknowledgements == ([permit, permit] if committed else [])
    assert owner.aborts == ([] if committed else [permit, permit])


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["registry", "component"])
async def test_missing_committed_owner_does_not_complete_recovery(
    checkpointer_factory, tmp_path, missing
):
    repository = CheckpointRepository(tmp_path / "repository")
    manifest, _ = stage_attempt(repository, committed=True)
    registries = (
        {} if missing == "registry" else {"train": CheckpointComponentRegistry(())}
    )
    checkpointer = await checkpointer_factory(
        object(),
        str(repository.root),
        interval_steps=1,
        component_registries=registries,
    )

    with pytest.raises(KeyError):
        await checkpointer._fence_incomplete_transactions()
    assert repository.is_committed(manifest.checkpoint_id)
    assert repository.transaction(manifest.checkpoint_id).status is (
        CheckpointTransactionState.COMMIT_DECIDED
    )


def test_repository_cannot_abandon_a_committed_marker(tmp_path):
    repository = CheckpointRepository(tmp_path / "repository")
    manifest, permit = stage_attempt(repository, committed=True)

    with pytest.raises(RuntimeError, match="globally committed"):
        repository.abandon_transaction(manifest.checkpoint_id, permit)
    assert repository.read(manifest.checkpoint_id) == manifest
    assert repository.transaction(manifest.checkpoint_id).status is (
        CheckpointTransactionState.COMMIT_DECIDED
    )


async def write_worker_files(root, topology, permit, component_type, workers):
    cohort = [
        WorkerCheckpointCoordinator(
            (
                component_type(
                    "state",
                    lambda instance, session: {"rank": session.rank},
                    lambda *_: None,
                ).registration(),
            ),
            object(),
            str(root),
            "train",
            rank,
            topology,
        )
        for rank in range(topology.world_size)
    ]
    workers.extend(cohort)
    for worker in cohort:
        await worker.prepare_save(
            f"snapshot-{permit.attempt_id}",
            permit.attempt_id,
            permit,
            CheckpointCut.for_step("cut", 10),
        )
    results = await asyncio.gather(*(worker.save() for worker in cohort))
    await asyncio.gather(*(worker.finish_save() for worker in cohort))
    gathered = {"state": {rank: result["state"] for rank, result in enumerate(results)}}
    publication = (await cohort[0].commit_save(gathered))["state"]
    for worker in cohort[1:]:
        await worker.commit_save(gathered)
    return publication


@pytest_asyncio.fixture
async def workers():
    active = []
    try:
        yield active
    finally:
        await asyncio.gather(*(worker.rollback_active() for worker in active))
        await asyncio.gather(*(worker.close() for worker in active))


def publication_paths(publication):
    return tuple(
        Path(unquote(urlparse(checkpoint_object.storage_locator.decode()).path))
        for partition in publication.partitions
        for checkpoint_object in partition.record.checkpoint_objects
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "component_type", [WorkerLocalCheckpointComponent, SafeTensorsCheckpointComponent]
)
@pytest.mark.parametrize("decided", [False, True])
async def test_filesystem_recovery_removes_all_ranks_of_only_the_failed_attempt(
    checkpointer_factory, tmp_path, component_type, decided, workers
):
    repository = CheckpointRepository(tmp_path / "repository")
    generation = repository.claim_coordinator_generation()
    topology = TopologySnapshot(2, ("shard",), (2,))
    retained_permit = repository.issue_commit_permit("retained", generation)
    retained = await write_worker_files(
        repository.root, topology, retained_permit, component_type, workers
    )
    retained_paths = publication_paths(retained)
    permit = repository.issue_commit_permit("attempt", generation)
    publication = await write_worker_files(
        repository.root, topology, permit, component_type, workers
    )
    failed_paths = publication_paths(publication)
    assert len(failed_paths) == len(retained_paths) == 2
    assert all(path.is_dir() for path in (*failed_paths, *retained_paths))
    manifest = CheckpointManifest(
        checkpoint_id="snapshot-attempt",
        attempt_id=permit.attempt_id,
        coordinator_generation=permit.coordinator_generation,
        cut=CheckpointCut.for_step("cut", 10),
        checkpoint_sequence=1,
        references=(StepReference(10),),
        framework_state={},
        spmds={
            "train": SpmdCheckpoint(
                source_topology=topology,
                runtime_state={
                    "drain_boundary": DrainBoundary(committed_cut=10).to_dict()
                },
                components={"state": publication},
            )
        },
    )
    repository.stage_transaction(manifest.checkpoint_id, manifest, permit)
    if decided:
        repository.publish(manifest, permit)
        repository.decide_transaction(manifest.checkpoint_id, permit)
    component = component_type("state", lambda *_: {}, lambda *_: None)
    checkpointer = await checkpointer_factory(
        object(),
        str(repository.root),
        interval_steps=1,
        component_registries={
            "train": CheckpointComponentRegistry((component.registration(),))
        },
    )

    await checkpointer._fence_incomplete_transactions()
    assert repository.transaction(manifest.checkpoint_id).status is (
        CheckpointTransactionState.ABANDONED
    )
    assert all(not path.exists() for path in failed_paths)
    assert all(path.is_dir() for path in retained_paths)
    await checkpointer._fence_incomplete_transactions()
    assert all(path.is_dir() for path in retained_paths)
