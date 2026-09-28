"""Filesystem repository for immutable manifests and logical references."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import os
import secrets
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import msgpack

from fluxway.common.types.checkpoint import (
    CheckpointObjectRef,
    CheckpointComponentPublication,
    CheckpointObjectRecord,
    CheckpointCut,
    CheckpointCommitPermit,
    CheckpointReference,
    NameReference,
    CheckpointCleanupIntent,
    CleanupReservationBinding,
    PartitionRecord,
    CleanupReservation,
    CheckpointManifest,
    CheckpointIdReference,
    checkpoint_position,
    SpmdCheckpoint,
    StepReference,
    TopologySnapshot,
    CheckpointTransactionState,
)
from fluxway.control_plane.checkpointer.directory import CheckpointDirectory


@dataclass(frozen=True, slots=True)
class CheckpointTransactionRecord:
    checkpoint_id: str
    manifest_digest: str
    permit: CheckpointCommitPermit
    status: CheckpointTransactionState

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint transaction requires checkpoint_id")
        if len(self.manifest_digest) != 64:
            raise ValueError("checkpoint transaction requires manifest digest")
        if not isinstance(self.status, CheckpointTransactionState):
            raise TypeError("checkpoint transaction status is invalid")


@dataclass(frozen=True, slots=True)
class CheckpointParticipantLease:
    checkpoint_id: str
    participant_id: str
    permit: CheckpointCommitPermit
    cut_digest: str
    generation: int
    expires_at: float

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint participant lease requires checkpoint_id")
        if not self.participant_id:
            raise ValueError("checkpoint participant lease requires participant_id")
        if not isinstance(self.permit, CheckpointCommitPermit):
            raise TypeError(
                "checkpoint participant lease requires CheckpointCommitPermit"
            )
        if len(self.cut_digest) != 64:
            raise ValueError("checkpoint participant lease requires cut_digest")
        if (
            not isinstance(self.generation, int)
            or isinstance(self.generation, bool)
            or self.generation < 0
        ):
            raise ValueError(
                "checkpoint participant lease generation must be non-negative integer"
            )
        if (
            not isinstance(self.expires_at, (int, float))
            or isinstance(self.expires_at, bool)
            or self.expires_at <= 0
        ):
            raise ValueError(
                "checkpoint participant lease expires_at must be positive number"
            )


def validate_checkpoint_value(value: Any, path: str = "checkpoint") -> None:
    if value is None or isinstance(value, str | int | float | bool | bytes):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key: {key!r}")
            validate_checkpoint_value(item, f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            validate_checkpoint_value(item, f"{path}[{index}]")
        return
    raise TypeError(
        f"{path} contains unsupported checkpoint value {type(value).__name__}"
    )


def _encode_topology(topology: TopologySnapshot) -> dict[str, Any]:
    return {
        "world_size": topology.world_size,
        "mesh_axis_names": list(topology.mesh_axis_names),
        "mesh_axis_shapes": list(topology.mesh_axis_shapes),
        "partition_scheme_version": topology.partition_scheme_version,
    }


def _decode_topology(payload: Mapping[str, Any]) -> TopologySnapshot:
    return TopologySnapshot(
        world_size=payload["world_size"],
        mesh_axis_names=tuple(payload["mesh_axis_names"]),
        mesh_axis_shapes=tuple(payload["mesh_axis_shapes"]),
        partition_scheme_version=payload["partition_scheme_version"],
    )


def _encode_checkpoint_object(checkpoint_object: CheckpointObjectRef) -> dict[str, Any]:
    return {
        "component_id": checkpoint_object.component_id,
        "checkpoint_object_id": checkpoint_object.checkpoint_object_id,
        "version_id": checkpoint_object.version_id,
        "storage_locator": checkpoint_object.storage_locator,
        "format_version": checkpoint_object.format_version,
        "partition_key": checkpoint_object.partition_key,
        "cleanup_group": checkpoint_object.cleanup_group,
    }


def _decode_checkpoint_object(payload: Mapping[str, Any]) -> CheckpointObjectRef:
    return CheckpointObjectRef(
        component_id=payload["component_id"],
        checkpoint_object_id=payload["checkpoint_object_id"],
        version_id=payload["version_id"],
        storage_locator=payload["storage_locator"],
        format_version=payload["format_version"],
        partition_key=payload.get("partition_key"),
        cleanup_group=payload.get("cleanup_group"),
    )


def _encode_cleanup_intent(intent: CheckpointCleanupIntent) -> dict[str, Any]:
    return {
        "artifact_id": intent.artifact_id,
        "reservations": [
            {
                "artifact_id": binding.artifact_id,
                "spmd_id": binding.spmd_id,
                "component_id": binding.component_id,
                "reservation": {
                    "attempt_id": binding.reservation.attempt_id,
                    "checkpoint_object_id": binding.reservation.checkpoint_object_id,
                    "claimed": [
                        _encode_checkpoint_object(checkpoint_object)
                        for checkpoint_object in binding.reservation.claimed
                    ],
                    "generation": binding.reservation.generation,
                },
            }
            for binding in intent.reservations
        ],
    }


def _decode_cleanup_intent(payload: Any) -> CheckpointCleanupIntent:
    if not isinstance(payload, Mapping):
        raise TypeError("cleanup intent root must be a mapping")
    raw_reservations = payload.get("reservations")
    if not isinstance(raw_reservations, list):
        raise TypeError("cleanup intent reservations must be a list")
    reservations = []
    for item in raw_reservations:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("reservation"), Mapping
        ):
            raise TypeError("cleanup intent reservation binding is invalid")
        reservation = item["reservation"]
        reservations.append(
            CleanupReservationBinding(
                artifact_id=item["artifact_id"],
                spmd_id=item["spmd_id"],
                component_id=item["component_id"],
                reservation=CleanupReservation(
                    attempt_id=reservation["attempt_id"],
                    checkpoint_object_id=reservation["checkpoint_object_id"],
                    claimed=tuple(
                        _decode_checkpoint_object(checkpoint_object)
                        for checkpoint_object in reservation["claimed"]
                    ),
                    generation=reservation["generation"],
                ),
            )
        )
    return CheckpointCleanupIntent(
        artifact_id=payload["artifact_id"],
        reservations=tuple(reservations),
    )


def _encode_record(record: CheckpointObjectRecord) -> dict[str, Any]:
    return {
        "payload": record.payload,
        "checkpoint_objects": [
            _encode_checkpoint_object(checkpoint_object)
            for checkpoint_object in record.checkpoint_objects
        ],
    }


def _decode_record(payload: Mapping[str, Any]) -> CheckpointObjectRecord:
    value = payload.get("payload")
    if value is not None and not isinstance(value, bytes):
        raise TypeError("record payload must be bytes or None")
    return CheckpointObjectRecord(
        payload=value if value is not None else b"",
        checkpoint_objects=tuple(
            _decode_checkpoint_object(item) for item in payload["checkpoint_objects"]
        ),
    )


def _encode_publication(publication: CheckpointComponentPublication) -> dict[str, Any]:
    return {
        "component_id": publication.component_id,
        "format_version": publication.format_version,
        "shared_records": [
            _encode_record(record) for record in publication.shared_records
        ],
        "partitions": [
            {
                "partition_key": partition.partition_key,
                "source_rank": partition.source_rank,
                "record": _encode_record(partition.record),
            }
            for partition in publication.partitions
        ],
    }


def _decode_publication(payload: Mapping[str, Any]) -> CheckpointComponentPublication:
    return CheckpointComponentPublication(
        component_id=payload["component_id"],
        format_version=payload["format_version"],
        shared_records=tuple(
            _decode_record(record) for record in payload["shared_records"]
        ),
        partitions=tuple(
            PartitionRecord(
                partition_key=partition["partition_key"],
                source_rank=partition.get("source_rank"),
                record=_decode_record(partition["record"]),
            )
            for partition in payload["partitions"]
        ),
    )


def _encode_reference(reference: CheckpointReference) -> dict[str, Any]:
    if isinstance(reference, StepReference):
        return {"kind": "step", "step": reference.step}
    if isinstance(reference, NameReference):
        return {
            "kind": "name",
            "name": reference.name,
            "retain_forever": reference.retain_forever,
        }
    if isinstance(reference, CheckpointIdReference):
        return {"kind": "snapshot", "checkpoint_id": reference.checkpoint_id}
    raise TypeError(f"unsupported checkpoint reference: {type(reference).__name__}")


def _decode_reference(payload: Mapping[str, Any]) -> CheckpointReference:
    kind = payload["kind"]
    if kind == "step":
        return StepReference(payload["step"])
    if kind == "name":
        return NameReference(payload["name"], payload["retain_forever"])
    if kind == "snapshot":
        return CheckpointIdReference(payload["checkpoint_id"])
    raise ValueError(f"unsupported checkpoint reference kind: {kind!r}")


def manifest_to_payload(manifest: CheckpointManifest) -> dict[str, Any]:
    validate_checkpoint_value(manifest.cut.stream_cuts, "manifest.cut.stream_cuts")
    validate_checkpoint_value(manifest.cut.sealed_ends, "manifest.cut.sealed_ends")
    validate_checkpoint_value(manifest.framework_state, "manifest.framework_state")
    for spmd_id, snapshot in manifest.spmds.items():
        runtime_position = checkpoint_position(snapshot.runtime_state)
        if manifest.cut.position != runtime_position:
            raise ValueError(
                f"manifest cut position {manifest.cut.position!r} does not match "
                f"SPMD {spmd_id!r} runtime position {runtime_position!r}"
            )
        validate_checkpoint_value(
            snapshot.runtime_state, f"manifest.spmds.{spmd_id}.runtime_state"
        )
        validate_checkpoint_value(
            snapshot.driver_components,
            f"manifest.spmds.{spmd_id}.driver_components",
        )
    return {
        "schema_version": manifest.schema_version,
        "checkpoint_id": manifest.checkpoint_id,
        "attempt_id": manifest.attempt_id,
        "coordinator_generation": manifest.coordinator_generation,
        "cut": {
            "cut_id": manifest.cut.cut_id,
            "position": manifest.cut.position,
            "stream_cuts": dict(manifest.cut.stream_cuts),
            "sealed_ends": dict(manifest.cut.sealed_ends),
            "control_epoch": manifest.cut.control_epoch,
        },
        "checkpoint_sequence": manifest.checkpoint_sequence,
        "references": [
            _encode_reference(reference) for reference in manifest.references
        ],
        "framework_state": dict(manifest.framework_state),
        "spmds": {
            spmd_id: {
                "source_topology": _encode_topology(snapshot.source_topology),
                "runtime_state": dict(snapshot.runtime_state),
                "driver_components": dict(snapshot.driver_components),
                "components": {
                    component_id: _encode_publication(publication)
                    for component_id, publication in snapshot.components.items()
                },
            }
            for spmd_id, snapshot in manifest.spmds.items()
        },
    }


def manifest_from_payload(payload: Any) -> CheckpointManifest:
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint manifest root must be a mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("only checkpoint manifest schema version 1 is supported")
    raw_cut = payload["cut"]
    if not isinstance(raw_cut, Mapping):
        raise TypeError("checkpoint manifest cut must be a mapping")
    cut = CheckpointCut(
        cut_id=raw_cut["cut_id"],
        position=raw_cut.get("position"),
        stream_cuts=raw_cut["stream_cuts"],
        sealed_ends=raw_cut["sealed_ends"],
        control_epoch=raw_cut["control_epoch"],
    )
    spmds: dict[str, SpmdCheckpoint] = {}
    for spmd_id, snapshot in payload["spmds"].items():
        runtime_position = checkpoint_position(snapshot["runtime_state"])
        if cut.position != runtime_position:
            raise ValueError(
                f"manifest cut position {cut.position!r} does not match "
                f"SPMD {spmd_id!r} runtime position {runtime_position!r}"
            )
        validate_checkpoint_value(
            snapshot["driver_components"],
            f"manifest.spmds.{spmd_id}.driver_components",
        )
        spmds[spmd_id] = SpmdCheckpoint(
            source_topology=_decode_topology(snapshot["source_topology"]),
            runtime_state=snapshot["runtime_state"],
            components={
                component_id: _decode_publication(publication)
                for component_id, publication in snapshot["components"].items()
            },
            driver_components=snapshot["driver_components"],
        )
    return CheckpointManifest(
        schema_version=payload["schema_version"],
        checkpoint_id=payload["checkpoint_id"],
        attempt_id=payload["attempt_id"],
        coordinator_generation=payload["coordinator_generation"],
        cut=cut,
        checkpoint_sequence=payload["checkpoint_sequence"],
        references=tuple(
            _decode_reference(reference) for reference in payload["references"]
        ),
        framework_state=payload["framework_state"],
        spmds=spmds,
    )


class CheckpointRepository:
    def __init__(self, root_dir: str | Path) -> None:
        self._root = CheckpointDirectory.open(root_dir).path
        self._manifests = self._root / "manifests"
        self._references = self._root / "refs"
        self._cleaned_up = self._root / "cleaned_up"
        self._cleanup_intents = self._root / "cleanup-intents"
        self._transactions = self._root / "transactions"
        self._manifests.mkdir(exist_ok=True)
        (self._references / "steps").mkdir(parents=True, exist_ok=True)
        (self._references / "names").mkdir(parents=True, exist_ok=True)
        self._cleaned_up.mkdir(exist_ok=True)
        self._cleanup_intents.mkdir(exist_ok=True)
        self._transactions.mkdir(exist_ok=True)
        self._lock_path = self._root / ".repository.lock"
        self._sequence_path = self._root / ".publication-sequence"
        self._permit_path = self._root / ".checkpoint-commit_save-permit.msgpack"
        self._participant_lease_path = (
            self._root / ".checkpoint-participant-leases.msgpack"
        )

    @property
    def root(self) -> Path:
        return self._root

    @contextmanager
    def _locked(self) -> Generator[None]:
        with self._lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def next_checkpoint_sequence(self) -> int:
        with self._locked():
            current = 0
            if self._sequence_path.exists():
                current = int(self._sequence_path.read_text(encoding="ascii"))
            next_sequence = current + 1
            self._atomic_write(self._sequence_path, str(next_sequence).encode("ascii"))
            return next_sequence

    def claim_coordinator_generation(self) -> int:
        with self._locked():
            payload = self._read_permit_state_unlocked()
            payload["coordinator_generation"] += 1
            payload["active_permits"] = {}
            self._write_permit_state_unlocked(payload)
            return payload["coordinator_generation"]

    def issue_commit_permit(
        self, attempt_id: str, coordinator_generation: int
    ) -> CheckpointCommitPermit:
        if not attempt_id:
            raise ValueError("commit_save permit requires attempt_id")
        with self._locked():
            payload = self._read_permit_state_unlocked()
            if payload["coordinator_generation"] != coordinator_generation:
                raise PermissionError("coordinator generation no longer owns commits")
            token = secrets.token_bytes(32)
            payload["active_permits"][attempt_id] = token
            self._write_permit_state_unlocked(payload)
            return CheckpointCommitPermit(
                attempt_id=attempt_id,
                coordinator_generation=coordinator_generation,
                token=token,
            )

    def require_commit_permit(self, permit: CheckpointCommitPermit) -> None:
        with self._locked():
            self._require_commit_permit_unlocked(permit)

    def publish(
        self,
        manifest: CheckpointManifest,
        permit: CheckpointCommitPermit,
        cleanup_intents: tuple[CheckpointCleanupIntent, ...] = (),
    ) -> None:
        if manifest.attempt_id != permit.attempt_id:
            raise PermissionError("manifest attempt does not match commit_save permit")
        if manifest.coordinator_generation != permit.coordinator_generation:
            raise PermissionError(
                "manifest generation does not match commit_save permit"
            )
        for intent in cleanup_intents:
            for binding in intent.reservations:
                if (
                    binding.reservation.attempt_id != permit.attempt_id
                    or binding.reservation.generation != permit.coordinator_generation
                ):
                    raise PermissionError("cleanup intent does not match commit permit")
        payload = manifest_to_payload(manifest)
        encoded = cast(bytes, msgpack.packb(payload, use_bin_type=True))
        manifest_dir = self._manifests / manifest.checkpoint_id
        final_path = manifest_dir / "manifest.msgpack"
        with self._locked():
            self._require_commit_permit_unlocked(permit)
            if final_path.exists():
                raise FileExistsError(
                    f"checkpoint manifest already exists: {final_path}"
                )
            intent_paths = tuple(
                self._cleanup_intent_path(intent) for intent in cleanup_intents
            )
            if len(intent_paths) != len(set(intent_paths)):
                raise ValueError("snapshot contains duplicate cleanup intents")
            existing_intents = tuple(path for path in intent_paths if path.exists())
            if existing_intents:
                raise FileExistsError(
                    f"cleanup intent already exists: {existing_intents[0]}"
                )
            manifest_dir.mkdir(parents=True, exist_ok=False)
            written_intents: list[Path] = []
            try:
                self._atomic_write(final_path, encoded)
                for intent, path in zip(cleanup_intents, intent_paths, strict=True):
                    self._atomic_write(
                        path,
                        cast(
                            bytes,
                            msgpack.packb(
                                _encode_cleanup_intent(intent), use_bin_type=True
                            ),
                        ),
                    )
                    written_intents.append(path)
            except BaseException:
                final_path.unlink(missing_ok=True)
                for path in written_intents:
                    path.unlink(missing_ok=True)
                manifest_dir.rmdir()
                raise

    def stage_transaction(
        self,
        checkpoint_id: str,
        manifest: CheckpointManifest,
        permit: CheckpointCommitPermit,
    ) -> CheckpointTransactionRecord:
        manifest_path = self._manifests / checkpoint_id / "manifest.msgpack"
        if manifest_path.is_file():
            encoded = manifest_path.read_bytes()
            actual = hashlib.sha256(encoded).hexdigest()
        else:
            payload = manifest_to_payload(manifest)
            encoded = msgpack.packb(payload, use_bin_type=True)
            actual = hashlib.sha256(cast(bytes, encoded)).hexdigest()
        if manifest.checkpoint_id != checkpoint_id:
            raise ValueError("staged manifest does not match repository payload")
        if (
            manifest.attempt_id != permit.attempt_id
            or manifest.coordinator_generation != permit.coordinator_generation
        ):
            raise PermissionError("staged manifest does not match commit permit")
        record = CheckpointTransactionRecord(
            checkpoint_id, actual, permit, CheckpointTransactionState.STAGED
        )
        path = self._transaction_path(checkpoint_id)
        with self._locked():
            if path.exists():
                existing = self._read_transaction_unlocked(path)
                if existing != record:
                    raise ValueError("checkpoint transaction staging conflicts")
                return existing
            self._atomic_write(path, self._encode_transaction(record))
        return record

    def decide_transaction(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> CheckpointTransactionRecord:
        return self._transition_transaction(
            checkpoint_id, permit, CheckpointTransactionState.COMMIT_DECIDED
        )

    def complete_transaction(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> CheckpointTransactionRecord:
        return self._transition_transaction(
            checkpoint_id, permit, CheckpointTransactionState.COMMITTED
        )

    def abandon_transaction(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> CheckpointTransactionRecord:
        with self._locked():
            path = self._transaction_path(checkpoint_id)
            if not path.is_file():
                raise FileNotFoundError("checkpoint transaction record not found")
            current = self._read_transaction_unlocked(path)
            if current.permit != permit:
                raise PermissionError("checkpoint transaction permit is stale")
            if current.status == CheckpointTransactionState.ABANDONED:
                return current
            if current.status not in {
                CheckpointTransactionState.STAGED,
                CheckpointTransactionState.COMMIT_DECIDED,
            }:
                raise RuntimeError(
                    f"invalid checkpoint transaction transition {current.status} -> "
                    f"{CheckpointTransactionState.ABANDONED}"
                )
            manifest_dir = self._manifests / checkpoint_id
            if manifest_dir.exists():
                for child in manifest_dir.iterdir():
                    child.unlink()
                manifest_dir.rmdir()
            for intent_path in self._cleanup_intent_paths_for_artifact_unlocked(
                checkpoint_id
            ):
                intent_path.unlink()
            self._discard_active_permit_unlocked(permit)
            return self._transition_transaction_unlocked(
                checkpoint_id, permit, CheckpointTransactionState.ABANDONED
            )

    def revoke_commit_permit(self, permit: CheckpointCommitPermit) -> None:
        with self._locked():
            self._discard_active_permit_unlocked(permit)

    def transaction(self, checkpoint_id: str) -> CheckpointTransactionRecord | None:
        path = self._transaction_path(checkpoint_id)
        if not path.is_file():
            return None
        return self._read_transaction_unlocked(path)

    def incomplete_transactions(self) -> tuple[CheckpointTransactionRecord, ...]:
        records = []
        for path in self._transactions.iterdir():
            if path.is_file() and path.suffix == ".msgpack":
                record = self._read_transaction_unlocked(path)
                if record.status in {
                    CheckpointTransactionState.STAGED,
                    CheckpointTransactionState.COMMIT_DECIDED,
                }:
                    records.append(record)
        return tuple(sorted(records, key=lambda item: item.checkpoint_id))

    def require_no_incomplete_transactions(self) -> None:
        incomplete = self.incomplete_transactions()
        if incomplete:
            raise RuntimeError(
                "checkpoint repository contains incomplete transactions: "
                + ", ".join(record.checkpoint_id for record in incomplete)
            )

    def read(self, checkpoint_id: str) -> CheckpointManifest:
        if not self.is_committed(checkpoint_id):
            raise FileNotFoundError(
                f"committed checkpoint not found: {checkpoint_id!r}"
            )
        path = self._manifests / checkpoint_id / "manifest.msgpack"
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint manifest not found: {path}")
        payload = msgpack.unpackb(path.read_bytes(), raw=False, strict_map_key=False)
        return manifest_from_payload(payload)

    def mark_committed(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> None:
        manifest_path = self._manifests / checkpoint_id / "manifest.msgpack"
        marker_path = self._commit_marker_path(checkpoint_id)
        with self._locked():
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"checkpoint manifest not found: {manifest_path}"
                )
            encoded_manifest = manifest_path.read_bytes()
            manifest = manifest_from_payload(
                msgpack.unpackb(
                    encoded_manifest,
                    raw=False,
                    strict_map_key=False,
                )
            )
            if manifest.attempt_id != permit.attempt_id:
                raise PermissionError(
                    "manifest attempt does not match committed marker permit"
                )
            if manifest.coordinator_generation != permit.coordinator_generation:
                raise PermissionError(
                    "manifest generation does not match committed marker permit"
                )
            marker = {
                "manifest_digest": hashlib.sha256(encoded_manifest).hexdigest(),
                "attempt_id": permit.attempt_id,
                "coordinator_generation": permit.coordinator_generation,
                "token_digest": hashlib.sha256(permit.token).hexdigest(),
            }
            encoded_marker = msgpack.packb(marker, use_bin_type=True)
            if not isinstance(encoded_marker, bytes):
                raise TypeError(
                    f"expected bytes from msgpack.packb, got {type(encoded_marker)}"
                )
            if marker_path.exists():
                if marker_path.read_bytes() != encoded_marker:
                    raise ValueError(
                        "checkpoint committed marker conflicts with manifest"
                    )
                self._discard_active_permit_unlocked(permit)
                return
            self._require_commit_permit_unlocked(permit)
            reference_paths = self._reference_paths(manifest.references)
            previous = {
                path: path.read_bytes() if path.exists() else None
                for path in reference_paths
            }
            try:
                for path in reference_paths:
                    self._atomic_write(path, checkpoint_id.encode("utf-8"))
                self._atomic_write(marker_path, encoded_marker)
            except OSError:
                for path, content in previous.items():
                    if content is None:
                        path.unlink(missing_ok=True)
                    else:
                        self._atomic_write(path, content)
                raise
            self._discard_active_permit_unlocked(permit)

    def is_committed(self, checkpoint_id: str) -> bool:
        manifest_path = self._manifests / checkpoint_id / "manifest.msgpack"
        marker_path = self._commit_marker_path(checkpoint_id)
        if not marker_path.is_file():
            return False
        if not manifest_path.is_file():
            raise ValueError("checkpoint committed marker has no manifest")
        encoded_manifest = manifest_path.read_bytes()
        manifest = manifest_from_payload(
            msgpack.unpackb(encoded_manifest, raw=False, strict_map_key=False)
        )
        marker = msgpack.unpackb(
            marker_path.read_bytes(), raw=False, strict_map_key=False
        )
        if (
            not isinstance(marker, Mapping)
            or set(marker)
            != {
                "manifest_digest",
                "attempt_id",
                "coordinator_generation",
                "token_digest",
            }
            or marker["manifest_digest"] != hashlib.sha256(encoded_manifest).hexdigest()
            or marker["attempt_id"] != manifest.attempt_id
            or marker["coordinator_generation"] != manifest.coordinator_generation
            or not isinstance(marker["token_digest"], str)
            or len(marker["token_digest"]) != 64
        ):
            raise ValueError("checkpoint committed marker conflicts with manifest")
        return True

    def resolve(self, reference: CheckpointReference) -> CheckpointManifest:
        if isinstance(reference, CheckpointIdReference):
            if not self.is_committed(reference.checkpoint_id):
                raise FileNotFoundError(
                    f"committed checkpoint not found: {reference.checkpoint_id!r}"
                )
            return self.read(reference.checkpoint_id)
        matches = [
            manifest
            for manifest in self.list()
            if self._manifest_has_reference(manifest, reference)
        ]
        if not matches:
            raise FileNotFoundError(f"checkpoint reference not found: {reference!r}")
        return matches[-1]

    def list(self) -> tuple[CheckpointManifest, ...]:
        manifests = [
            self.read(directory.name)
            for directory in self._manifests.iterdir()
            if directory.is_dir()
            and (directory / "manifest.msgpack").is_file()
            and self.is_committed(directory.name)
        ]
        return tuple(sorted(manifests, key=lambda item: item.checkpoint_sequence))

    def list_steps(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    reference.step
                    for manifest in self.list()
                    for reference in manifest.references
                    if isinstance(reference, StepReference)
                }
            )
        )

    def latest(self) -> CheckpointManifest | None:
        manifests = self.list()
        return manifests[-1] if manifests else None

    def mark_checkpoint_object_cleaned_up(
        self, spmd_id: str, checkpoint_object: CheckpointObjectRef
    ) -> None:
        if not spmd_id:
            raise ValueError("cleaned_up checkpoint_object requires SPMD ID")
        payload = {
            "spmd_id": spmd_id,
            "component_id": checkpoint_object.component_id,
            "checkpoint_object_id": checkpoint_object.checkpoint_object_id,
            "version_id": checkpoint_object.version_id,
        }
        encoded = msgpack.packb(payload, use_bin_type=True)
        path = self._cleaned_up_path(spmd_id, checkpoint_object)
        with self._locked():
            if path.exists():
                existing = msgpack.unpackb(
                    path.read_bytes(), raw=False, strict_map_key=False
                )
                if existing != payload:
                    raise ValueError(
                        f"cleaned_up checkpoint_object index conflicts with {path}"
                    )
                return
            self._atomic_write(path, cast(bytes, encoded))

    def is_checkpoint_object_cleaned_up(
        self, spmd_id: str, checkpoint_object: CheckpointObjectRef
    ) -> bool:
        if not spmd_id:
            raise ValueError("checkpoint_object lookup requires SPMD ID")
        path = self._cleaned_up_path(spmd_id, checkpoint_object)
        if not path.exists():
            return False
        payload = msgpack.unpackb(path.read_bytes(), raw=False, strict_map_key=False)
        expected = {
            "spmd_id": spmd_id,
            "component_id": checkpoint_object.component_id,
            "checkpoint_object_id": checkpoint_object.checkpoint_object_id,
            "version_id": checkpoint_object.version_id,
        }
        if payload != expected:
            raise ValueError(f"cleaned_up checkpoint_object index is corrupt: {path}")
        return True

    def cleaned_checkpoint_objects(
        self, manifests: tuple[CheckpointManifest, ...]
    ) -> frozenset[tuple[str, CheckpointObjectRef]]:
        return frozenset(
            (spmd_id, checkpoint_object)
            for manifest in manifests
            for spmd_id, snapshot in manifest.spmds.items()
            for publication in snapshot.components.values()
            for record in (
                *publication.shared_records,
                *(partition.record for partition in publication.partitions),
            )
            for checkpoint_object in record.checkpoint_objects
            if self.is_checkpoint_object_cleaned_up(spmd_id, checkpoint_object)
        )

    def contains_checkpoint_object(
        self, spmd_id: str, checkpoint_object: CheckpointObjectRef
    ) -> bool:
        if not spmd_id:
            raise ValueError("checkpoint_object lookup requires SPMD ID")
        for manifest in self.list():
            snapshot = manifest.spmds.get(spmd_id)
            if snapshot is None:
                continue
            publication = snapshot.components.get(checkpoint_object.component_id)
            if publication is None:
                continue
            records = (
                *publication.shared_records,
                *(partition.record for partition in publication.partitions),
            )
            if any(
                checkpoint_object in record.checkpoint_objects for record in records
            ):
                return not self.is_checkpoint_object_cleaned_up(
                    spmd_id, checkpoint_object
                )
        return False

    def list_cleanup_intents(
        self, checkpoint_id: str | None = None
    ) -> tuple[CheckpointCleanupIntent, ...]:
        intents = []
        for path in self._cleanup_intents.iterdir():
            if not path.is_file() or path.suffix != ".msgpack":
                continue
            payload = msgpack.unpackb(
                path.read_bytes(), raw=False, strict_map_key=False
            )
            item = _decode_cleanup_intent(payload)
            if self._cleanup_intent_path(item) != path:
                raise ValueError(f"cleanup intent index is corrupt: {path}")
            if checkpoint_id is None or item.artifact_id == checkpoint_id:
                intents.append(item)
        return tuple(
            sorted(
                intents,
                key=lambda item: item.artifact_id,
            )
        )

    def complete_cleanup_intent(self, intent: CheckpointCleanupIntent) -> None:
        path = self._cleanup_intent_path(intent)
        with self._locked():
            if not path.is_file():
                return
            encoded = path.read_bytes()
            payload = msgpack.unpackb(encoded, raw=False, strict_map_key=False)
            if _decode_cleanup_intent(payload) != intent:
                raise ValueError(f"cleanup intent index conflicts with {path}")
            path.unlink()

    def evict_manifests_for_cleanup_intents(
        self, cleanup_intents: tuple[CheckpointCleanupIntent, ...]
    ) -> tuple[str, ...]:
        """Remove active manifests that reference objects about to be deleted.

        The operation is repository-owned and independent of the saved or
        current SPMD topology. It removes active references before physical
        component cleanup, so a failed or interrupted cleanup leaves only
        unreferenced physical data rather than a visible broken checkpoint.
        """
        targets = {
            (binding.spmd_id, binding.component_id, checkpoint_object)
            for intent in cleanup_intents
            for binding in intent.reservations
            for checkpoint_object in binding.reservation.claimed
        }
        if not targets:
            return ()
        with self._locked():
            evicted: list[str] = []
            target_artifacts = {intent.artifact_id for intent in cleanup_intents}
            for manifest_dir in tuple(self._manifests.iterdir()):
                manifest_path = manifest_dir / "manifest.msgpack"
                marker_path = manifest_dir / "committed.sha256"
                if not manifest_path.is_file() or not marker_path.is_file():
                    continue
                manifest = manifest_from_payload(
                    msgpack.unpackb(
                        manifest_path.read_bytes(),
                        raw=False,
                        strict_map_key=False,
                    )
                )
                matched = False
                for spmd_id, snapshot in manifest.spmds.items():
                    for component_id, publication in snapshot.components.items():
                        records = (
                            *publication.shared_records,
                            *(partition.record for partition in publication.partitions),
                        )
                        if any(
                            (spmd_id, component_id, checkpoint_object) in targets
                            for record in records
                            for checkpoint_object in record.checkpoint_objects
                        ):
                            matched = True
                            break
                    if matched:
                        break
                if not matched:
                    continue
                checkpoint_id = manifest.checkpoint_id
                if checkpoint_id not in target_artifacts:
                    continue
                for reference_path in self._reference_paths(manifest.references):
                    if (
                        reference_path.is_file()
                        and reference_path.read_bytes() == checkpoint_id.encode("utf-8")
                    ):
                        reference_path.unlink()
                for child in manifest_dir.iterdir():
                    child.unlink()
                manifest_dir.rmdir()
                evicted.append(checkpoint_id)
            return tuple(sorted(evicted))

    def _reference_paths(
        self, references: tuple[CheckpointReference, ...]
    ) -> tuple[Path, ...]:
        paths: list[Path] = []
        for reference in references:
            if isinstance(reference, StepReference):
                paths.append(self._references / "steps" / str(reference.step))
            elif isinstance(reference, NameReference):
                encoded = base64.urlsafe_b64encode(
                    reference.name.encode("utf-8")
                ).decode("ascii")
                paths.append(self._references / "names" / encoded)
            elif isinstance(reference, CheckpointIdReference):
                continue
            else:
                raise TypeError(
                    f"unsupported checkpoint reference: {type(reference).__name__}"
                )
        return tuple(paths)

    def _commit_marker_path(self, checkpoint_id: str) -> Path:
        return self._manifests / checkpoint_id / "committed.sha256"

    @staticmethod
    def _manifest_has_reference(
        manifest: CheckpointManifest, reference: CheckpointReference
    ) -> bool:
        if isinstance(reference, StepReference):
            return any(
                isinstance(candidate, StepReference)
                and candidate.step == reference.step
                for candidate in manifest.references
            )
        if isinstance(reference, NameReference):
            return any(
                isinstance(candidate, NameReference)
                and candidate.name == reference.name
                for candidate in manifest.references
            )
        raise TypeError(f"unsupported checkpoint reference: {type(reference).__name__}")

    def _cleaned_up_path(
        self, spmd_id: str, checkpoint_object: CheckpointObjectRef
    ) -> Path:
        identity = "\x00".join(
            (
                spmd_id,
                checkpoint_object.component_id,
                checkpoint_object.checkpoint_object_id,
                checkpoint_object.version_id,
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self._cleaned_up / f"{digest}.msgpack"

    def _cleanup_intent_path(self, intent: CheckpointCleanupIntent) -> Path:
        identity = "\x00".join(
            (
                intent.artifact_id,
                *(
                    "\x01".join(
                        (
                            binding.spmd_id,
                            binding.component_id,
                            binding.reservation.checkpoint_object_id,
                            binding.reservation.attempt_id,
                            str(binding.reservation.generation),
                        )
                    )
                    for binding in intent.reservations
                ),
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self._cleanup_intents / f"{digest}.msgpack"

    def _cleanup_intents_for_artifact_unlocked(
        self, artifact_id: str
    ) -> tuple[CheckpointCleanupIntent, ...]:
        matches = []
        for path in self._cleanup_intents.iterdir():
            if not path.is_file() or path.suffix != ".msgpack":
                continue
            payload = msgpack.unpackb(
                path.read_bytes(), raw=False, strict_map_key=False
            )
            intent = _decode_cleanup_intent(payload)
            if intent.artifact_id == artifact_id:
                matches.append(intent)
        return tuple(matches)

    def _cleanup_intent_paths_for_artifact_unlocked(
        self, artifact_id: str
    ) -> tuple[Path, ...]:
        paths = []
        for path in self._cleanup_intents.iterdir():
            if not path.is_file() or path.suffix != ".msgpack":
                continue
            payload = msgpack.unpackb(
                path.read_bytes(), raw=False, strict_map_key=False
            )
            intent = _decode_cleanup_intent(payload)
            if intent.artifact_id == artifact_id:
                paths.append(path)
        return tuple(paths)

    def _transaction_path(self, checkpoint_id: str) -> Path:
        return self._transactions / f"{checkpoint_id}.msgpack"

    @staticmethod
    def _encode_transaction(record: CheckpointTransactionRecord) -> bytes:
        return cast(
            bytes,
            msgpack.packb(
                {
                    "checkpoint_id": record.checkpoint_id,
                    "manifest_digest": record.manifest_digest,
                    "attempt_id": record.permit.attempt_id,
                    "coordinator_generation": record.permit.coordinator_generation,
                    "token": record.permit.token,
                    "status": record.status.value,
                },
                use_bin_type=True,
            ),
        )

    @staticmethod
    def _read_transaction_unlocked(path: Path) -> CheckpointTransactionRecord:
        payload = msgpack.unpackb(path.read_bytes(), raw=False, strict_map_key=False)
        if not isinstance(payload, Mapping) or set(payload) != {
            "checkpoint_id",
            "manifest_digest",
            "attempt_id",
            "coordinator_generation",
            "token",
            "status",
        }:
            raise ValueError("checkpoint transaction record fields do not match")
        return CheckpointTransactionRecord(
            payload["checkpoint_id"],
            payload["manifest_digest"],
            CheckpointCommitPermit(
                payload["attempt_id"],
                payload["coordinator_generation"],
                payload["token"],
            ),
            CheckpointTransactionState(payload["status"]),
        )

    def _transition_transaction_unlocked(
        self,
        checkpoint_id: str,
        permit: CheckpointCommitPermit,
        status: CheckpointTransactionState,
    ) -> CheckpointTransactionRecord:
        if status not in {
            CheckpointTransactionState.COMMIT_DECIDED,
            CheckpointTransactionState.COMMITTED,
            CheckpointTransactionState.ABANDONED,
        }:
            raise ValueError("invalid checkpoint transaction transition")
        path = self._transaction_path(checkpoint_id)
        if status == CheckpointTransactionState.COMMIT_DECIDED:
            self._require_commit_permit_unlocked(permit)
        if not path.is_file():
            raise FileNotFoundError("checkpoint transaction record not found")
        current = self._read_transaction_unlocked(path)
        if current.permit != permit:
            raise PermissionError("checkpoint transaction permit is stale")
        if current.status == status:
            return current
        allowed = {
            CheckpointTransactionState.STAGED: {
                CheckpointTransactionState.COMMIT_DECIDED,
                CheckpointTransactionState.ABANDONED,
            },
            CheckpointTransactionState.COMMIT_DECIDED: {
                CheckpointTransactionState.COMMITTED,
                CheckpointTransactionState.ABANDONED,
            },
            CheckpointTransactionState.COMMITTED: set(),
            CheckpointTransactionState.ABANDONED: set(),
        }
        if status not in allowed[current.status]:
            raise RuntimeError(
                f"invalid checkpoint transaction transition {current.status} -> {status}"
            )
        updated = CheckpointTransactionRecord(
            current.checkpoint_id, current.manifest_digest, current.permit, status
        )
        self._atomic_write(path, self._encode_transaction(updated))
        return updated

    def _transition_transaction(
        self,
        checkpoint_id: str,
        permit: CheckpointCommitPermit,
        status: CheckpointTransactionState,
    ) -> CheckpointTransactionRecord:
        with self._locked():
            return self._transition_transaction_unlocked(checkpoint_id, permit, status)

    def _read_permit_state_unlocked(self) -> dict[str, Any]:
        if not self._permit_path.exists():
            return {"coordinator_generation": 0, "active_permits": {}}
        payload = msgpack.unpackb(
            self._permit_path.read_bytes(), raw=False, strict_map_key=False
        )
        if (
            not isinstance(payload, dict)
            or set(payload) != {"coordinator_generation", "active_permits"}
            or not isinstance(payload["coordinator_generation"], int)
            or payload["coordinator_generation"] < 0
            or not isinstance(payload["active_permits"], Mapping)
            or any(
                not isinstance(attempt, str) or not isinstance(token, bytes)
                for attempt, token in payload["active_permits"].items()
            )
        ):
            raise ValueError("checkpoint commit_save permit state is corrupt")
        return payload

    def _read_participant_leases_unlocked(
        self,
    ) -> dict[str, CheckpointParticipantLease]:
        if not self._participant_lease_path.exists():
            return {}
        payload = msgpack.unpackb(
            self._participant_lease_path.read_bytes(),
            raw=False,
            strict_map_key=False,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("participant lease record must be a mapping")
        leases: dict[str, CheckpointParticipantLease] = {}
        for key, value in payload.items():
            if not isinstance(value, Mapping) or set(value) != {
                "checkpoint_id",
                "participant_id",
                "attempt_id",
                "coordinator_generation",
                "token",
                "cut_digest",
                "generation",
                "expires_at",
            }:
                raise ValueError("participant lease record fields do not match")
            leases[str(key)] = CheckpointParticipantLease(
                value["checkpoint_id"],
                value["participant_id"],
                CheckpointCommitPermit(
                    value["attempt_id"],
                    value["coordinator_generation"],
                    value["token"],
                ),
                value["cut_digest"],
                value["generation"],
                value["expires_at"],
            )
        return leases

    def _write_participant_leases_unlocked(
        self, leases: Mapping[str, CheckpointParticipantLease]
    ) -> None:
        payload = {
            key: {
                "checkpoint_id": lease.checkpoint_id,
                "participant_id": lease.participant_id,
                "attempt_id": lease.permit.attempt_id,
                "coordinator_generation": lease.permit.coordinator_generation,
                "token": lease.permit.token,
                "cut_digest": lease.cut_digest,
                "generation": lease.generation,
                "expires_at": lease.expires_at,
            }
            for key, lease in leases.items()
        }
        self._atomic_write(
            self._participant_lease_path,
            cast(bytes, msgpack.packb(payload, use_bin_type=True)),
        )

    def _require_commit_permit_unlocked(
        self, permit: CheckpointCommitPermit
    ) -> dict[str, Any]:
        if self._committed_marker_matches_permit_unlocked(permit):
            raise PermissionError("checkpoint commit_save permit is stale or revoked")
        payload = self._read_permit_state_unlocked()
        active = payload["active_permits"].get(permit.attempt_id)
        if (
            payload["coordinator_generation"] != permit.coordinator_generation
            or active != permit.token
        ):
            raise PermissionError("checkpoint commit_save permit is stale or revoked")
        return payload

    def _discard_active_permit_unlocked(self, permit: CheckpointCommitPermit) -> None:
        payload = self._read_permit_state_unlocked()
        active = payload["active_permits"].get(permit.attempt_id)
        if (
            payload["coordinator_generation"] != permit.coordinator_generation
            or active != permit.token
        ):
            return
        del payload["active_permits"][permit.attempt_id]
        self._write_permit_state_unlocked(payload)

    def _committed_marker_matches_permit_unlocked(
        self, permit: CheckpointCommitPermit
    ) -> bool:
        token_digest = hashlib.sha256(permit.token).hexdigest()
        for directory in self._manifests.iterdir():
            if not directory.is_dir():
                continue
            marker_path = self._commit_marker_path(directory.name)
            if not marker_path.is_file() or not self.is_committed(directory.name):
                continue
            marker = msgpack.unpackb(
                marker_path.read_bytes(), raw=False, strict_map_key=False
            )
            if (
                marker["attempt_id"] == permit.attempt_id
                and marker["coordinator_generation"] == permit.coordinator_generation
                and marker["token_digest"] == token_digest
            ):
                return True
        return False

    def _write_permit_state_unlocked(self, payload: Mapping[str, Any]) -> None:
        self._atomic_write(
            self._permit_path,
            cast(bytes, msgpack.packb(dict(payload), use_bin_type=True)),
        )

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
