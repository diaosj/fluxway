from dataclasses import replace

import pytest
import msgpack
from typing import cast

from fluxway.common.types.checkpoint import (
    CheckpointIdReference,
    CheckpointObjectRef,
    CheckpointComponentPublication,
    CheckpointObjectRecord,
    CheckpointCut,
    NameReference,
    CheckpointManifest,
    SpmdCheckpoint,
    StepReference,
    TopologySnapshot,
    CheckpointTransactionState,
)
from fluxway.common.types import DrainBoundary
from fluxway.control_plane.checkpointer.repository import CheckpointRepository


def _manifest(sequence: int) -> CheckpointManifest:
    checkpoint_object = CheckpointObjectRef(
        component_id="state",
        checkpoint_object_id="state",
        version_id=f"version-{sequence}",
        storage_locator=f"gravitino://catalog/version-{sequence}".encode(),
        format_version=1,
    )
    return CheckpointManifest(
        schema_version=1,
        checkpoint_id=f"snapshot-{sequence}",
        attempt_id=f"attempt-{sequence}",
        coordinator_generation=1,
        cut=CheckpointCut.for_step(f"cut-{sequence}", sequence),
        checkpoint_sequence=sequence,
        references=(StepReference(sequence), NameReference("best_model")),
        framework_state={},
        spmds={
            "train": SpmdCheckpoint(
                source_topology=TopologySnapshot(1, (), (), "v1"),
                runtime_state={
                    "drain_boundary": DrainBoundary(committed_cut=sequence).to_dict()
                },
                components={
                    "state": CheckpointComponentPublication(
                        component_id="state",
                        format_version=1,
                        shared_records=(
                            CheckpointObjectRecord(
                                checkpoint_objects=(checkpoint_object,)
                            ),
                        ),
                    )
                },
            )
        },
    )


def test_repository_round_trips_opaque_checkpoint_objects_and_references(tmp_path):
    repository = CheckpointRepository(tmp_path / "repository")
    manifest = _manifest(7)
    permit = repository.issue_commit_permit(
        "attempt-7", repository.claim_coordinator_generation()
    )
    repository.publish(manifest, permit)
    repository.mark_committed(manifest.checkpoint_id, permit)

    assert repository.read("snapshot-7") == manifest
    assert repository.resolve(StepReference(7)) == manifest
    assert repository.resolve(NameReference("best_model")) == manifest
    assert repository.list_steps() == (7,)


def test_repository_tracks_cleaned_up_logical_checkpoint_objects(tmp_path):
    repository = CheckpointRepository(tmp_path / "repository")
    manifest = _manifest(1)
    permit = repository.issue_commit_permit(
        "attempt-1", repository.claim_coordinator_generation()
    )
    repository.publish(manifest, permit)
    repository.mark_committed(manifest.checkpoint_id, permit)
    checkpoint_object = (
        manifest.spmds["train"]
        .components["state"]
        .shared_records[0]
        .checkpoint_objects[0]
    )

    assert repository.contains_checkpoint_object("train", checkpoint_object)
    repository.mark_checkpoint_object_cleaned_up("train", checkpoint_object)
    assert repository.is_checkpoint_object_cleaned_up("train", checkpoint_object)
    assert not repository.contains_checkpoint_object("train", checkpoint_object)


def test_committed_marker_consumes_commit_permit(tmp_path):
    repository = CheckpointRepository(tmp_path / "repository")
    generation = repository.claim_coordinator_generation()
    permit = repository.issue_commit_permit("attempt-1", generation)
    repository.publish(_manifest(1), permit)

    repository.require_commit_permit(permit)
    repository.mark_committed("snapshot-1", permit)

    with pytest.raises(PermissionError, match="stale or revoked"):
        repository.require_commit_permit(permit)
    second = replace(
        _manifest(2),
        attempt_id=permit.attempt_id,
        coordinator_generation=permit.coordinator_generation,
    )
    with pytest.raises(PermissionError, match="stale or revoked"):
        repository.publish(second, permit)


def test_uncommitted_manifest_is_not_restorable_or_retained(tmp_path):
    repository = CheckpointRepository(tmp_path / "repository")
    manifest = _manifest(1)
    permit = repository.issue_commit_permit(
        "attempt-1", repository.claim_coordinator_generation()
    )
    repository.publish(manifest, permit)

    assert repository.latest() is None
    assert repository.list() == ()
    assert repository.list_steps() == ()
    with pytest.raises(FileNotFoundError, match="committed checkpoint"):
        repository.read(manifest.checkpoint_id)
    with pytest.raises(FileNotFoundError, match="committed checkpoint"):
        repository.resolve(CheckpointIdReference(manifest.checkpoint_id))


def test_uncommitted_manifest_does_not_hide_prior_named_reference(tmp_path):
    repository = CheckpointRepository(tmp_path / "repository")
    first = _manifest(1)
    first_permit = repository.issue_commit_permit(
        "attempt-1", repository.claim_coordinator_generation()
    )
    repository.publish(first, first_permit)
    repository.mark_committed(first.checkpoint_id, first_permit)
    second = replace(_manifest(2), coordinator_generation=first.coordinator_generation)
    second_permit = repository.issue_commit_permit(
        "attempt-2", first.coordinator_generation
    )
    repository.publish(second, second_permit)

    assert repository.resolve(NameReference("best_model")) == first


def test_stale_coordinator_cannot_mark_manifest_committed(tmp_path):
    repository = CheckpointRepository(tmp_path / "repository")
    generation = repository.claim_coordinator_generation()
    permit = repository.issue_commit_permit("attempt-1", generation)
    manifest = _manifest(1)
    repository.publish(manifest, permit)
    repository.claim_coordinator_generation()

    with pytest.raises(PermissionError, match="stale or revoked"):
        repository.mark_committed(manifest.checkpoint_id, permit)
    assert repository.latest() is None


def test_committed_marker_fences_permit_when_index_update_fails(tmp_path, monkeypatch):
    repository = CheckpointRepository(tmp_path / "repository")
    generation = repository.claim_coordinator_generation()
    permit = repository.issue_commit_permit("attempt-1", generation)
    manifest = _manifest(1)
    repository.publish(manifest, permit)
    write_permit_state = repository._write_permit_state_unlocked

    def fail_write(_state):
        raise OSError("injected index failure")

    monkeypatch.setattr(
        repository,
        "_write_permit_state_unlocked",
        fail_write,
    )

    with pytest.raises(OSError, match="injected index failure"):
        repository.mark_committed(manifest.checkpoint_id, permit)
    assert repository.latest() == manifest
    with pytest.raises(PermissionError, match="stale or revoked"):
        repository.require_commit_permit(permit)

    monkeypatch.setattr(repository, "_write_permit_state_unlocked", write_permit_state)
    repository.mark_committed(manifest.checkpoint_id, permit)
    with pytest.raises(PermissionError, match="stale or revoked"):
        repository.require_commit_permit(permit)


def test_coordinator_takeover_revokes_unpublished_commit_permit(tmp_path):
    repository = CheckpointRepository(tmp_path / "repository")
    generation = repository.claim_coordinator_generation()
    permit = repository.issue_commit_permit("attempt-1", generation)
    repository.claim_coordinator_generation()

    with pytest.raises(PermissionError, match="stale or revoked"):
        repository.publish(_manifest(1), permit)


def test_spmd_snapshot_rejects_component_identity_aliases():
    with pytest.raises(ValueError, match="mapping key"):
        SpmdCheckpoint(
            source_topology=TopologySnapshot(1, (), (), "v1"),
            runtime_state={"drain_boundary": DrainBoundary(committed_cut=1).to_dict()},
            components={
                "declared": CheckpointComponentPublication(
                    component_id="actual",
                    format_version=1,
                )
            },
        )


def test_checkpoint_object_publication_requires_one_logical_version():
    checkpoint_objects = tuple(
        CheckpointObjectRef(
            component_id="state",
            checkpoint_object_id="state",
            version_id=version_id,
            storage_locator=f"opaque:{version_id}".encode(),
            format_version=1,
        )
        for version_id in ("version-1", "version-2")
    )

    with pytest.raises(ValueError, match="one logical checkpoint_object version"):
        CheckpointComponentPublication(
            component_id="state",
            format_version=1,
            shared_records=(
                CheckpointObjectRecord(checkpoint_objects=checkpoint_objects),
            ),
        )


def test_repository_permit_revocation_fencing(tmp_path):
    repo = CheckpointRepository(tmp_path / "repo")
    gen = repo.claim_coordinator_generation()
    permit = repo.issue_commit_permit("attempt-1", gen)
    manifest = _manifest(1)

    repo.stage_transaction(manifest.checkpoint_id, manifest, permit)
    checkpoint_id_transaction = repo.transaction(manifest.checkpoint_id)
    assert checkpoint_id_transaction is not None
    assert checkpoint_id_transaction.status == CheckpointTransactionState.STAGED

    repo.revoke_commit_permit(permit)

    with pytest.raises(PermissionError, match="stale or revoked"):
        repo.decide_transaction(manifest.checkpoint_id, permit)


def test_repository_abandon_transaction_from_staged(tmp_path):
    repo = CheckpointRepository(tmp_path / "repo")
    gen = repo.claim_coordinator_generation()
    permit = repo.issue_commit_permit("attempt-1", gen)
    manifest = _manifest(1)

    repo.stage_transaction(manifest.checkpoint_id, manifest, permit)

    record = repo.abandon_transaction(manifest.checkpoint_id, permit)
    assert record.status == CheckpointTransactionState.ABANDONED
    tx = repo.transaction(manifest.checkpoint_id)
    assert tx is not None
    assert tx.status == CheckpointTransactionState.ABANDONED
    assert repo.list() == ()
    assert repo.latest() is None


def test_repository_decided_transaction_can_be_abandoned_when_fenced(tmp_path):
    repo = CheckpointRepository(tmp_path / "repo")
    gen = repo.claim_coordinator_generation()
    permit = repo.issue_commit_permit("attempt-1", gen)
    manifest = _manifest(1)

    repo.stage_transaction(manifest.checkpoint_id, manifest, permit)
    repo.decide_transaction(manifest.checkpoint_id, permit)

    record = repo.abandon_transaction(manifest.checkpoint_id, permit)
    assert record.status == CheckpointTransactionState.ABANDONED
    assert repo.transaction(manifest.checkpoint_id) == record
    assert repo.latest() is None
    with pytest.raises(FileNotFoundError, match="committed checkpoint"):
        repo.resolve(CheckpointIdReference(manifest.checkpoint_id))


def test_corrupt_committed_checkpoint_fails_restore_lookup(tmp_path):
    repo = CheckpointRepository(tmp_path / "repo")
    generation = repo.claim_coordinator_generation()
    permit = repo.issue_commit_permit("attempt-1", generation)
    manifest = _manifest(1)
    repo.publish(manifest, permit)
    repo.mark_committed(manifest.checkpoint_id, permit)

    marker_path = repo.root / "manifests" / manifest.checkpoint_id / "committed.sha256"
    marker = msgpack.unpackb(marker_path.read_bytes(), raw=False, strict_map_key=False)
    marker["manifest_digest"] = "0" * 64
    marker_path.write_bytes(cast(bytes, msgpack.packb(marker, use_bin_type=True)))

    with pytest.raises(ValueError, match="committed marker conflicts"):
        repo.resolve(CheckpointIdReference(manifest.checkpoint_id))


def test_stage_transaction_precedes_manifest_publication(tmp_path):
    repo = CheckpointRepository(tmp_path / "repo")
    gen = repo.claim_coordinator_generation()
    permit = repo.issue_commit_permit("attempt-1", gen)
    manifest = _manifest(1)

    # 1. Stage transaction writes transaction record before manifest is published
    tx = repo.stage_transaction(manifest.checkpoint_id, manifest, permit)
    assert tx.status == CheckpointTransactionState.STAGED

    manifest_path = (
        repo.root / "manifests" / manifest.checkpoint_id / "manifest.msgpack"
    )
    assert not manifest_path.exists()

    # 2. Publish writes manifest
    repo.publish(manifest, permit)
    assert manifest_path.exists()
