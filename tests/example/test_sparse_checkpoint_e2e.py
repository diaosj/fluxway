import asyncio
import hashlib
from dataclasses import replace

import lance
import numpy as np
import pytest
import sparse_embedding as se
from sparse_embedding.checkpoint.spi import CheckpointConflictError

from fluxway.common.types.checkpoint import (
    CheckpointComponentCapability,
    CheckpointCommitPermit,
    CheckpointCut,
    CheckpointAssembly,
    SaveCheckpointRequest,
    SpmdCheckpoint,
    StepReference,
    CheckpointCleanupBlocked,
    TopologySnapshot,
)
from fluxway.common.types import DrainBoundary
from fluxway.control_plane.checkpointer.component import CheckpointComponentRegistry
from fluxway.control_plane.checkpointer.checkpoint import FluxwayCheckpointer
from fluxway.control_plane.checkpointer.repository import CheckpointRepository
from fluxway.common.types.checkpoint import CheckpointRetentionPolicy
from fluxway.integrations.sparse_embedding.checkpoint import (
    LocalLanceStorageFactory,
    SparseEmbeddingCheckpointComponent,
)
from fluxway.example.recommendation.sparse_spmd import modulo_partition
from fluxway.spmd.worker.checkpoint import WorkerCheckpointCoordinator


def _config() -> se.EmbeddingConfig:
    return se.EmbeddingConfig(dims=[4], thread_budget=1, value_dtype="float32")


def _store(keys: np.ndarray) -> se.EmbeddingStore:
    store = se.EmbeddingStore(_config())
    values = np.repeat(keys[:, None].astype(np.float32), 4, axis=1)
    store.assign(keys, values)
    return store


def _lookup(store: se.EmbeddingStore, keys: np.ndarray) -> np.ndarray:
    values, _ = store.lookup(keys, dim=4)
    return np.from_dlpack(values).reshape(-1, 4)


class _Owner:
    def __init__(self, store: se.EmbeddingStore) -> None:
        self.store = store


class _LocalLanceController:
    def __init__(self, owner, topology) -> None:
        self.owner = owner
        self.topology = topology
        self.restore_owner = None
        self._active = {}

    async def request_snapshot(
        self,
        checkpoint_id,
        attempt_id,
        permit,
        cut,
        repository_dir,
        deadline=None,
    ):
        coordinator = _coordinator(repository_dir, 0, self.topology, self.owner)
        await coordinator.prepare_save(checkpoint_id, attempt_id, permit, cut)
        record = (await coordinator.save())["embedding"]
        await coordinator.finish_save()
        publications = await coordinator.commit_save({"embedding": {0: record}})
        self._active[checkpoint_id] = coordinator
        return CheckpointAssembly(
            cut=cut,
            framework_state={},
            spmds={
                "embedding": SpmdCheckpoint(
                    source_topology=self.topology,
                    runtime_state={
                        "drain_boundary": DrainBoundary(
                            committed_cut=cut.position or 0
                        ).to_dict()
                    },
                    components=publications,
                )
            },
        )

    async def publish_snapshot(self, checkpoint_id, permit, deadline=None):
        coordinator = self._active.pop(checkpoint_id)
        await coordinator.publish_save()
        await coordinator.close()

    async def rollback_snapshot(self, checkpoint_id, permit):
        coordinator = self._active.pop(checkpoint_id, None)
        if coordinator is not None:
            await coordinator.rollback_active()
            await coordinator.close()

    def prepare_checkpoint_transaction(self, checkpoint_id, permit, cut, control_state):
        return {}

    def apply_prepared_checkpoint_transaction(
        self, checkpoint_id, permit, cut, control_state, prepared
    ):
        return None

    def abort_checkpoint_transaction(self, checkpoint_id, permit):
        return None

    async def restore_snapshot(self, checkpoint_id, manifest, repository_dir):
        if self.restore_owner is None:
            raise RuntimeError("restore owner is not configured")
        coordinator = _coordinator(repository_dir, 0, self.topology, self.restore_owner)
        snapshot = manifest.spmds["embedding"]
        await coordinator.restore(
            checkpoint_id,
            manifest.cut,
            snapshot.components,
            snapshot.source_topology,
            self.topology,
        )
        await coordinator.close()


def _coordinator(root, rank, topology, owner):
    component = SparseEmbeddingCheckpointComponent(
        modulo_partition, LocalLanceStorageFactory()
    )
    return WorkerCheckpointCoordinator(
        (component.registration(),),
        owner,
        str(root),
        "embedding",
        rank,
        topology,
    )


@pytest.mark.asyncio
async def test_fluxway_restores_selected_step_from_exact_lance_version(tmp_path):
    topology = TopologySnapshot(1, (), (), "modulo-v1")
    owner = _Owner(_store(np.asarray([1, 2], dtype=np.uint64)))
    controller = _LocalLanceController(owner, topology)
    control_component = SparseEmbeddingCheckpointComponent(
        modulo_partition, LocalLanceStorageFactory()
    )
    checkpointer = await FluxwayCheckpointer.create(
        controller,
        str(tmp_path / "repository"),
        interval_steps=1,
        retention_policy=CheckpointRetentionPolicy(retain_last=3),
        component_registries={
            "embedding": CheckpointComponentRegistry(
                (control_component.registration(),)
            )
        },
    )

    first = await checkpointer.save(SaveCheckpointRequest.for_step(10))
    owner.store = _store(np.asarray([3, 4], dtype=np.uint64))
    second = await checkpointer.save(SaveCheckpointRequest.for_step(20))
    assert first.checkpoint_id != second.checkpoint_id

    controller.restore_owner = _Owner(se.EmbeddingStore(_config()))
    restored_step = await checkpointer.restore(StepReference(10))
    assert restored_step == 10
    restored = controller.restore_owner.store
    assert restored.total_size() == 2
    np.testing.assert_allclose(
        _lookup(restored, np.asarray([1, 2], dtype=np.uint64)),
        np.asarray([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32),
    )
    await checkpointer.close()


@pytest.mark.asyncio
async def test_fluxway_automatically_cleanups_sparse_versions(tmp_path):
    topology = TopologySnapshot(1, (), (), "modulo-v1")
    owner = _Owner(_store(np.asarray([1], dtype=np.uint64)))
    controller = _LocalLanceController(owner, topology)
    control_component = SparseEmbeddingCheckpointComponent(
        modulo_partition,
        LocalLanceStorageFactory(),
    )
    repository = tmp_path / "repository"
    checkpointer = await FluxwayCheckpointer.create(
        controller,
        str(repository),
        interval_steps=1,
        retention_policy=CheckpointRetentionPolicy(retain_last=1),
        component_registries={
            "embedding": CheckpointComponentRegistry(
                (control_component.registration(),)
            )
        },
    )

    first = await checkpointer.save(SaveCheckpointRequest.for_step(1))
    owner.store = _store(np.asarray([2], dtype=np.uint64))
    await checkpointer.save(SaveCheckpointRequest.for_step(2))
    await checkpointer.close()

    checkpoint_object = (
        first.spmds["embedding"]
        .components["embedding"]
        .shared_records[0]
        .checkpoint_objects[0]
    )
    assert CheckpointRepository(
        tmp_path / "repository"
    ).is_checkpoint_object_cleaned_up("embedding", checkpoint_object)
    next_checkpointer = await FluxwayCheckpointer.create(
        controller,
        str(repository),
        interval_steps=1,
        retention_policy=CheckpointRetentionPolicy(retain_last=1),
        component_registries={
            "embedding": CheckpointComponentRegistry(
                (control_component.registration(),)
            )
        },
    )
    with pytest.raises(FileNotFoundError, match="checkpoint reference not found"):
        await next_checkpointer.restore(StepReference(1))
    await next_checkpointer.close()

    storage_root = (
        repository / "application-checkpoint_objects" / "embedding" / "embedding"
    )
    rows = lance.dataset(str(storage_root / "snapshot_manifest")).to_table().to_pylist()
    assert all(row["checkpoint_name"] != first.checkpoint_id for row in rows)


@pytest.mark.asyncio
async def test_sparse_lease_blocks_automatic_cleanup_until_release(tmp_path):
    topology = TopologySnapshot(1, (), (), "modulo-v1")
    owner = _Owner(_store(np.asarray([1], dtype=np.uint64)))
    controller = _LocalLanceController(owner, topology)
    control_component = SparseEmbeddingCheckpointComponent(
        modulo_partition,
        LocalLanceStorageFactory(),
    )
    checkpointer = await FluxwayCheckpointer.create(
        controller,
        str(tmp_path / "repository"),
        interval_steps=1,
        retention_policy=CheckpointRetentionPolicy(retain_last=1),
        component_registries={
            "embedding": CheckpointComponentRegistry(
                (control_component.registration(),)
            )
        },
    )
    first = await checkpointer.save(SaveCheckpointRequest.for_step(1))
    checkpoint_object = (
        first.spmds["embedding"]
        .components["embedding"]
        .shared_records[0]
        .checkpoint_objects[0]
    )
    lease = await checkpointer.hold_checkpoint_object(
        "embedding",
        checkpoint_object,
        "evaluator",
        60,
        "evaluation-1",
    )
    owner.store = _store(np.asarray([2], dtype=np.uint64))

    with pytest.raises(CheckpointCleanupBlocked):
        await checkpointer.save(SaveCheckpointRequest.for_step(2))
    await checkpointer.release_checkpoint_hold("embedding", lease)
    await checkpointer.save(SaveCheckpointRequest.for_step(2))
    await checkpointer.close()
    assert CheckpointRepository(
        tmp_path / "repository"
    ).is_checkpoint_object_cleaned_up("embedding", checkpoint_object)


@pytest.mark.asyncio
async def test_local_lance_component_roundtrip_with_elastic_restore(tmp_path):
    source_topology = TopologySnapshot(2, ("shard",), (2,), "modulo-v1")
    permit = CheckpointCommitPermit("attempt-1", 7, b"coordinator-token")
    cut = CheckpointCut.for_step("cut-10", 10)
    source_keys = (
        np.asarray([0, 2, 4, 6], dtype=np.uint64),
        np.asarray([1, 3, 5, 7], dtype=np.uint64),
    )
    source = [
        _coordinator(tmp_path, rank, source_topology, _Owner(_store(keys)))
        for rank, keys in enumerate(source_keys)
    ]

    for coordinator in source:
        await coordinator.prepare_save(
            "snapshot-10",
            "attempt-1",
            permit,
            cut,
        )
    save_results = await asyncio.gather(*(coordinator.save() for coordinator in source))
    for coordinator in source:
        await coordinator.finish_save()
    gathered = {
        "embedding": {
            rank: records["embedding"] for rank, records in enumerate(save_results)
        }
    }
    publication = (await source[0].commit_save(gathered))["embedding"]
    await source[1].commit_save(gathered)
    for coordinator in source:
        await coordinator.publish_save()
        await coordinator.close()

    checkpoint_object = publication.shared_records[0].checkpoint_objects[0]
    assert checkpoint_object.storage_locator
    assert str(tmp_path).encode() not in checkpoint_object.storage_locator
    manifest_path = (
        tmp_path
        / "application-checkpoint_objects"
        / "embedding"
        / "embedding"
        / "snapshot_manifest"
    )
    manifest = lance.dataset(str(manifest_path)).to_table().to_pylist()[0]
    assert manifest["checkpoint_name"] == "snapshot-10"
    assert manifest["save_attempt"] == "attempt-1"
    assert manifest["authority_scope"] == "fluxway:embedding:embedding"
    assert manifest["authority_generation"] == permit.coordinator_generation
    assert manifest["commit_authority"] == hashlib.sha256(permit.token).hexdigest()

    with pytest.raises(ValueError, match="checkpoint_object component"):
        replace(
            publication,
            shared_records=(
                replace(
                    publication.shared_records[0],
                    checkpoint_objects=(
                        replace(checkpoint_object, component_id="other"),
                    ),
                ),
            ),
        )

    invalid_checkpoint_objects = (
        replace(checkpoint_object, checkpoint_object_id="other"),
        replace(checkpoint_object, format_version=checkpoint_object.format_version + 1),
    )
    for invalid in invalid_checkpoint_objects:
        invalid_publication = replace(
            publication,
            shared_records=(
                replace(publication.shared_records[0], checkpoint_objects=(invalid,)),
            ),
        )
        invalid_coordinator = _coordinator(
            tmp_path,
            0,
            source_topology,
            _Owner(se.EmbeddingStore(_config())),
        )
        with pytest.raises(ValueError):
            await invalid_coordinator.restore(
                "snapshot-10",
                cut,
                {"embedding": invalid_publication},
                source_topology,
                source_topology,
            )
        await invalid_coordinator.close()

    checkpoint_object_topology = TopologySnapshot(3, ("shard",), (3,), "modulo-v1")
    checkpoint_object_owners = [_Owner(se.EmbeddingStore(_config())) for _ in range(3)]
    checkpoint_object = [
        _coordinator(tmp_path, rank, checkpoint_object_topology, owner)
        for rank, owner in enumerate(checkpoint_object_owners)
    ]
    for coordinator in checkpoint_object:
        await coordinator.restore(
            "snapshot-10",
            cut,
            {"embedding": publication},
            source_topology,
            checkpoint_object_topology,
        )

    all_keys = np.arange(8, dtype=np.uint64)
    for rank, owner in enumerate(checkpoint_object_owners):
        expected = all_keys[all_keys % checkpoint_object_topology.world_size == rank]
        assert owner.store.total_size() == expected.size
        np.testing.assert_allclose(
            _lookup(owner.store, expected),
            np.repeat(expected[:, None].astype(np.float32), 4, axis=1),
        )
    for coordinator in checkpoint_object:
        await coordinator.close()


def test_sparse_component_declares_only_implemented_capabilities():
    registration = SparseEmbeddingCheckpointComponent(
        modulo_partition, LocalLanceStorageFactory()
    ).registration()
    assert registration.capabilities == frozenset(
        {
            CheckpointComponentCapability.COMPONENT_LIFECYCLE,
            CheckpointComponentCapability.SAVE,
            CheckpointComponentCapability.RESTORE,
            CheckpointComponentCapability.OBJECT_LIFECYCLE,
            CheckpointComponentCapability.OBJECT_CLEANUP,
        }
    )


@pytest.mark.asyncio
async def test_local_lance_component_rejects_stale_commit_authority(tmp_path):
    topology = TopologySnapshot(1, (), (), "modulo-v1")
    cut = CheckpointCut.for_step("cut", 1)
    owner = _Owner(_store(np.asarray([1], dtype=np.uint64)))
    stale = _coordinator(tmp_path, 0, topology, owner)
    current = _coordinator(tmp_path, 0, topology, owner)
    stale_permit = CheckpointCommitPermit("attempt-old", 1, b"old")
    current_permit = CheckpointCommitPermit("attempt-new", 2, b"new")

    await current.prepare_save(
        "shared-snapshot",
        "attempt-new",
        current_permit,
        cut,
    )
    current_record = (await current.save())["embedding"]
    await current.finish_save()

    with pytest.raises(CheckpointConflictError, match="authority is stale"):
        await stale.prepare_save(
            "shared-snapshot",
            "attempt-old",
            stale_permit,
            cut,
        )
    await stale.close()

    publication = await current.commit_save({"embedding": {0: current_record}})
    assert publication["embedding"].shared_records
    await current.publish_save()
    await current.close()
