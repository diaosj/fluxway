"""Filesystem repository for immutable manifests and logical references."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import os
import secrets
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import msgpack

from fluxway.common.types.checkpoint import (
    CheckpointObjectRef,
    CheckpointComponentPublication,
    CheckpointObjectRecord,
    CheckpointCut,
    CheckpointCommitPermit,
    CheckpointReference,
    NameReference,
    PendingCleanup,
    PartitionRecord,
    CleanupReservation,
    CheckpointManifest,
    CheckpointIdReference,
    SpmdCheckpoint,
    StepReference,
    TopologySnapshot,
)
from fluxway.control_plane.checkpointer.directory import CheckpointDirectory


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


def _encode_pending_cleanup(pending: PendingCleanup) -> dict[str, Any]:
    reservation = pending.reservation
    return {
        "checkpoint_id": pending.checkpoint_id,
        "spmd_id": pending.spmd_id,
        "component_id": pending.component_id,
        "reservation": {
            "attempt_id": reservation.attempt_id,
            "checkpoint_object_id": reservation.checkpoint_object_id,
            "claimed": [
                _encode_checkpoint_object(checkpoint_object)
                for checkpoint_object in reservation.claimed
            ],
            "generation": reservation.generation,
        },
    }


def _decode_pending_cleanup(payload: Any) -> PendingCleanup:
    if not isinstance(payload, Mapping):
        raise TypeError("pending cleanup root must be a mapping")
    reservation = payload.get("reservation")
    if not isinstance(reservation, Mapping):
        raise TypeError("pending cleanup reservation must be a mapping")
    return PendingCleanup(
        checkpoint_id=payload["checkpoint_id"],
        spmd_id=payload["spmd_id"],
        component_id=payload["component_id"],
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
    return CheckpointObjectRecord(
        payload=value,
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
    return {
        "schema_version": manifest.schema_version,
        "checkpoint_id": manifest.checkpoint_id,
        "attempt_id": manifest.attempt_id,
        "coordinator_generation": manifest.coordinator_generation,
        "cut": {
            "cut_id": manifest.cut.cut_id,
            "logical_step": manifest.cut.logical_step,
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
    if payload.get("schema_version") != 4:
        raise ValueError("only checkpoint manifest schema version 4 is supported")
    raw_cut = payload["cut"]
    if not isinstance(raw_cut, Mapping):
        raise TypeError("checkpoint manifest cut must be a mapping")
    cut = CheckpointCut(
        cut_id=raw_cut["cut_id"],
        logical_step=raw_cut.get("logical_step"),
        stream_cuts=raw_cut["stream_cuts"],
        sealed_ends=raw_cut["sealed_ends"],
        control_epoch=raw_cut["control_epoch"],
    )
    spmds = {
        spmd_id: SpmdCheckpoint(
            source_topology=_decode_topology(snapshot["source_topology"]),
            runtime_state=snapshot["runtime_state"],
            components={
                component_id: _decode_publication(publication)
                for component_id, publication in snapshot["components"].items()
            },
        )
        for spmd_id, snapshot in payload["spmds"].items()
    }
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
        self._pending_cleanups = self._root / "pending-cleanups"
        self._manifests.mkdir(exist_ok=True)
        (self._references / "steps").mkdir(parents=True, exist_ok=True)
        (self._references / "names").mkdir(parents=True, exist_ok=True)
        self._cleaned_up.mkdir(exist_ok=True)
        self._pending_cleanups.mkdir(exist_ok=True)
        self._lock_path = self._root / ".repository.lock"
        self._sequence_path = self._root / ".publication-sequence"
        self._permit_path = self._root / ".checkpoint-commit_save-permit.msgpack"

    @property
    def root(self) -> Path:
        return self._root

    @contextmanager
    def _locked(self) -> Iterator[None]:
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
        pending_cleanups: tuple[PendingCleanup, ...] = (),
    ) -> None:
        if manifest.attempt_id != permit.attempt_id:
            raise PermissionError("manifest attempt does not match commit_save permit")
        if manifest.coordinator_generation != permit.coordinator_generation:
            raise PermissionError(
                "manifest generation does not match commit_save permit"
            )
        for pending in pending_cleanups:
            if pending.checkpoint_id != manifest.checkpoint_id:
                raise ValueError("pending cleanup belongs to another snapshot")
            if (
                pending.reservation.attempt_id != permit.attempt_id
                or pending.reservation.generation != permit.coordinator_generation
            ):
                raise PermissionError(
                    "pending cleanup does not match commit_save permit"
                )
        payload = manifest_to_payload(manifest)
        encoded = msgpack.packb(payload, use_bin_type=True)
        manifest_dir = self._manifests / manifest.checkpoint_id
        final_path = manifest_dir / "manifest.msgpack"
        with self._locked():
            permit_state = self._require_commit_permit_unlocked(permit)
            if final_path.exists():
                raise FileExistsError(
                    f"checkpoint manifest already exists: {final_path}"
                )
            pending_paths = tuple(
                self._pending_cleanup_path(pending) for pending in pending_cleanups
            )
            if len(pending_paths) != len(set(pending_paths)):
                raise ValueError("snapshot contains duplicate pending cleanups")
            existing_pending = tuple(path for path in pending_paths if path.exists())
            if existing_pending:
                raise FileExistsError(
                    f"pending cleanup already exists: {existing_pending[0]}"
                )
            manifest_dir.mkdir(parents=True, exist_ok=False)
            reference_paths = self._reference_paths(manifest.references)
            previous = {
                path: path.read_bytes() if path.exists() else None
                for path in reference_paths
            }
            written_pending: list[Path] = []
            try:
                self._atomic_write(final_path, encoded)
                for path in reference_paths:
                    self._atomic_write(path, manifest.checkpoint_id.encode("utf-8"))
                for pending, path in zip(pending_cleanups, pending_paths, strict=True):
                    self._atomic_write(
                        path,
                        msgpack.packb(
                            _encode_pending_cleanup(pending), use_bin_type=True
                        ),
                    )
                    written_pending.append(path)
                del permit_state["active_permits"][permit.attempt_id]
                self._write_permit_state_unlocked(permit_state)
            except BaseException:
                final_path.unlink(missing_ok=True)
                for path, content in previous.items():
                    if content is None:
                        path.unlink(missing_ok=True)
                    else:
                        self._atomic_write(path, content)
                for path in written_pending:
                    path.unlink(missing_ok=True)
                manifest_dir.rmdir()
                raise

    def read(self, checkpoint_id: str) -> CheckpointManifest:
        path = self._manifests / checkpoint_id / "manifest.msgpack"
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint manifest not found: {path}")
        payload = msgpack.unpackb(path.read_bytes(), raw=False, strict_map_key=False)
        return manifest_from_payload(payload)

    def resolve(self, reference: CheckpointReference) -> CheckpointManifest:
        if isinstance(reference, CheckpointIdReference):
            return self.read(reference.checkpoint_id)
        path = self._reference_paths((reference,))[0]
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint reference not found: {reference!r}")
        return self.read(path.read_text(encoding="utf-8"))

    def list(self) -> tuple[CheckpointManifest, ...]:
        manifests = [
            self.read(directory.name)
            for directory in self._manifests.iterdir()
            if directory.is_dir() and (directory / "manifest.msgpack").is_file()
        ]
        return tuple(sorted(manifests, key=lambda item: item.checkpoint_sequence))

    def list_steps(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                int(path.name)
                for path in (self._references / "steps").iterdir()
                if path.is_file() and path.name.isdigit()
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
            self._atomic_write(path, encoded)

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

    def list_pending_cleanups(
        self, checkpoint_id: str | None = None
    ) -> tuple[PendingCleanup, ...]:
        pending = []
        for path in self._pending_cleanups.iterdir():
            if not path.is_file() or path.suffix != ".msgpack":
                continue
            payload = msgpack.unpackb(
                path.read_bytes(), raw=False, strict_map_key=False
            )
            item = _decode_pending_cleanup(payload)
            if self._pending_cleanup_path(item) != path:
                raise ValueError(f"pending cleanup index is corrupt: {path}")
            if checkpoint_id is None or item.checkpoint_id == checkpoint_id:
                pending.append(item)
        return tuple(
            sorted(
                pending,
                key=lambda item: (
                    item.checkpoint_id,
                    item.spmd_id,
                    item.component_id,
                    item.reservation.checkpoint_object_id,
                ),
            )
        )

    def complete_pending_cleanup(self, pending: PendingCleanup) -> None:
        path = self._pending_cleanup_path(pending)
        with self._locked():
            if not path.is_file():
                raise FileNotFoundError(f"pending cleanup not found: {path}")
            encoded = path.read_bytes()
            payload = msgpack.unpackb(encoded, raw=False, strict_map_key=False)
            if _decode_pending_cleanup(payload) != pending:
                raise ValueError(f"pending cleanup index conflicts with {path}")
            path.unlink()

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

    def _pending_cleanup_path(self, pending: PendingCleanup) -> Path:
        identity = "\x00".join(
            (
                pending.checkpoint_id,
                pending.spmd_id,
                pending.component_id,
                pending.reservation.checkpoint_object_id,
                pending.reservation.attempt_id,
                str(pending.reservation.generation),
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self._pending_cleanups / f"{digest}.msgpack"

    def _pending_cleanups_for_snapshot_unlocked(
        self, checkpoint_id: str
    ) -> tuple[PendingCleanup, ...]:
        matches = []
        for path in self._pending_cleanups.iterdir():
            if not path.is_file() or path.suffix != ".msgpack":
                continue
            payload = msgpack.unpackb(
                path.read_bytes(), raw=False, strict_map_key=False
            )
            pending = _decode_pending_cleanup(payload)
            if pending.checkpoint_id == checkpoint_id:
                matches.append(pending)
        return tuple(matches)

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

    def _require_commit_permit_unlocked(
        self, permit: CheckpointCommitPermit
    ) -> dict[str, Any]:
        payload = self._read_permit_state_unlocked()
        active = payload["active_permits"].get(permit.attempt_id)
        if (
            payload["coordinator_generation"] != permit.coordinator_generation
            or active != permit.token
        ):
            raise PermissionError("checkpoint commit_save permit is stale or revoked")
        return payload

    def _write_permit_state_unlocked(self, payload: Mapping[str, Any]) -> None:
        self._atomic_write(
            self._permit_path,
            msgpack.packb(dict(payload), use_bin_type=True),
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
