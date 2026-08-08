from dataclasses import replace

import pytest

from fluxway.common.types.checkpoint import (
    CheckpointObjectRef,
    ComponentCheckpoint,
    CheckpointObjectRecord,
    CheckpointCut,
    NameReference,
    CheckpointManifest,
    SpmdCheckpoint,
    StepReference,
    TopologySnapshot,
)
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
        schema_version=4,
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
                runtime_state={},
                components={
                    "state": ComponentCheckpoint(
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


def test_manifest_publication_consumes_commit_permit(tmp_path):
    repository = CheckpointRepository(tmp_path / "repository")
    generation = repository.claim_coordinator_generation()
    permit = repository.issue_commit_permit("attempt-1", generation)
    repository.publish(_manifest(1), permit)

    with pytest.raises(PermissionError, match="stale or revoked"):
        repository.require_commit_permit(permit)
    second = replace(
        _manifest(2),
        attempt_id=permit.attempt_id,
        coordinator_generation=permit.coordinator_generation,
    )
    with pytest.raises(PermissionError, match="stale or revoked"):
        repository.publish(second, permit)


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
            runtime_state={},
            components={
                "declared": ComponentCheckpoint(
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
        ComponentCheckpoint(
            component_id="state",
            format_version=1,
            shared_records=(
                CheckpointObjectRecord(checkpoint_objects=checkpoint_objects),
            ),
        )
