"""Checkpoint integration for the optional ``sparse_embedding`` package."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, cast, Protocol

import msgpack  # type: ignore[import-untyped]
import numpy as np
import sparse_embedding as se  # type: ignore
from sparse_embedding.checkpoint import (  # type: ignore
    ElasticRestore,
    GravitinoLanceStorage,
    LocalLanceStorage,
    SparseCheckpointLocator,
)
from sparse_embedding.checkpoint.gravitino_lance import SdkConfig  # type: ignore
from sparse_embedding.checkpoint.spi import (  # type: ignore
    CheckpointCommitAuthority,
    CheckpointFragment,
    CheckpointParticipant,
    CheckpointSaveIdentity,
    CheckpointConflictError,
    CheckpointLeaseRequest as SparseLeaseRequest,
    CleanupReservation as SparseCleanupReservation,
    CheckpointLease as SparseCheckpointLease,
    SparseCheckpointStorage,
)

from fluxway.common.types.checkpoint import (
    CheckpointObjectRef,
    CheckpointObjectSpec,
    CheckpointComponentPublication,
    CheckpointObjectRecord,
    CheckpointRuntime,
    CheckpointRuntimeRole,
    CheckpointComponentSpec,
    CheckpointCommitPermit,
    SaveSession,
    CommitSaveSession,
    RestoreSession,
    CheckpointRestoreSpec,
    CheckpointSaveScope,
    CheckpointSaveSpec,
    WorkerSaveResult,
    LeaseConflictError,
    CheckpointCleanupBlocked,
    CheckpointHold,
    CheckpointRetentionPolicy,
    CleanupReservation,
    CleanupReservationRequest,
    RestoreTopologyMode,
)
from fluxway.control_plane.checkpointer.component import (
    CheckpointComponentRegistration,
    CheckpointComponent,
    CheckpointComponentLifecycle,
    CheckpointObjectCleanup,
    CheckpointObjectLifecycle,
    CheckpointComponentRestore,
    CheckpointComponentSave,
)


_WIRE_VERSION = 1
PartitionFn = Callable[[Any, int], Any]


def modulo_partition(keys: object, partition_count: int) -> np.ndarray:
    key_array = np.asarray(keys)
    if not key_array.flags.c_contiguous or not key_array.flags.writeable:
        key_array = np.array(key_array, copy=True, order="C")
    return np.from_dlpack(se.ModuloPartitioner(partition_count).partition(key_array))


class SparseCheckpointStorageFactory(Protocol):
    def __call__(
        self,
        runtime: CheckpointRuntime,
        component_id: str,
    ) -> SparseCheckpointStorage: ...


@dataclass(frozen=True, slots=True)
class LocalLanceStorageFactory:
    max_storage_bytes: int | None = None

    def __call__(
        self,
        runtime: CheckpointRuntime,
        component_id: str,
    ) -> SparseCheckpointStorage:
        root = (
            Path(runtime.repository_root)
            / "application-checkpoint_objects"
            / runtime.spmd_id
            / component_id
        )
        return LocalLanceStorage(root, max_storage_bytes=self.max_storage_bytes)


@dataclass(frozen=True, slots=True)
class GravitinoLanceStorageFactory:
    config: SdkConfig
    namespace: tuple[str, str]
    location: str
    max_storage_bytes: int | None = None
    lock_timeout_seconds: float = 60.0
    lock_ttl_seconds: float = 300.0

    def __call__(
        self,
        runtime: CheckpointRuntime,
        component_id: str,
    ) -> SparseCheckpointStorage:
        storage_component = f"{runtime.spmd_id}.{component_id}"
        storage_location = (
            f"{self.location.rstrip('/')}/{runtime.spmd_id}/{component_id}"
        )
        return GravitinoLanceStorage(
            self.config,
            self.namespace,
            component_id=storage_component,
            location=storage_location,
            max_storage_bytes=self.max_storage_bytes,
            lock_timeout_seconds=self.lock_timeout_seconds,
            lock_ttl_seconds=self.lock_ttl_seconds,
            provision=runtime.role is CheckpointRuntimeRole.CONTROL,
        )


class SparseEmbeddingCheckpointComponent(
    CheckpointComponent,
    CheckpointComponentLifecycle,
    CheckpointComponentSave,
    CheckpointComponentRestore,
    CheckpointObjectLifecycle,
    CheckpointObjectCleanup,
):
    def __init__(
        self,
        partition_fn: PartitionFn,
        storage_factory: SparseCheckpointStorageFactory,
        *,
        component_id: str = "embedding",
        retain_last: int | None = 3,
    ) -> None:
        if not callable(partition_fn):
            raise TypeError("partition_fn must be callable")
        if not callable(storage_factory):
            raise TypeError("storage_factory must be callable")
        self._descriptor = CheckpointComponentSpec(
            component_id=component_id,
            format_version=_WIRE_VERSION,
            checkpoint_objects=(
                CheckpointObjectSpec(
                    component_id,
                    CheckpointRetentionPolicy(retain_last=retain_last),
                ),
            ),
            save=CheckpointSaveSpec(
                prepare_scope=CheckpointSaveScope.LEADER,
                commit_scope=CheckpointSaveScope.LEADER,
            ),
            restore=CheckpointRestoreSpec(
                topology=RestoreTopologyMode.LOGICAL_PARTITION_REMAP
            ),
        )
        self._partition_fn = partition_fn
        self._storage_factory = storage_factory
        self._runtime: CheckpointRuntime | None = None
        self._storage: SparseCheckpointStorage | None = None

    @property
    def descriptor(self) -> CheckpointComponentSpec:
        return self._descriptor

    def registration(self) -> CheckpointComponentRegistration:
        return CheckpointComponentRegistration(
            component=self,
            component_lifecycle=self,
            save=self,
            restore=self,
            object_lifecycle=self,
            object_cleanup=self,
        )

    async def open(self, runtime: CheckpointRuntime) -> None:
        if self._runtime is not None:
            raise RuntimeError(
                f"sparse checkpoint component {self.descriptor.component_id!r} is open"
            )
        self._storage = await asyncio.to_thread(
            self._storage_factory,
            runtime,
            self.descriptor.component_id,
        )
        self._runtime = runtime

    async def prepare_save(self, session: SaveSession) -> None:
        self._require_worker(session)
        identity = self._identity(session, session.commit_permit)
        await asyncio.to_thread(self._require_storage().prepare_checkpoint, identity)

    async def save(self, session: SaveSession) -> WorkerSaveResult:
        self._require_worker(session)
        identity = self._identity(session, session.commit_permit)
        participant = CheckpointParticipant(
            session.rank, session.source_topology.world_size
        )
        fragment = await asyncio.to_thread(
            self._require_storage().write_checkpoint_fragment,
            session.instance.store,
            identity,
            participant,
        )
        return WorkerSaveResult(commit_metadata=self._encode_fragment(fragment))

    async def finish_save(self, session: SaveSession) -> None:
        self._require_worker(session)
        identity = self._identity(session, session.commit_permit)
        await asyncio.to_thread(
            self._require_storage().finalize_checkpoint,
            identity,
        )

    async def commit_save(
        self, session: CommitSaveSession
    ) -> CheckpointComponentPublication:
        self._require_worker(session)
        identity = self._identity(session, session.commit_permit)
        fragments = tuple(
            self._decode_fragment(record.commit_metadata)
            for _, record in sorted(session.worker_results.items())
        )
        reference_id = self._reference_id(
            self.descriptor.component_id,
            session.checkpoint_id,
        )
        storage_locator = await asyncio.to_thread(
            self._require_storage().commit_checkpoint,
            identity,
            fragments,
            None,
            reference_id,
        )
        storage_locator_payload = self._encode_storage_locator(storage_locator)
        version_id = (
            f"{storage_locator.checkpoint_name}:{storage_locator.checkpoint_version}"
        )
        checkpoint_object = CheckpointObjectRef(
            component_id=self.descriptor.component_id,
            checkpoint_object_id=self.descriptor.component_id,
            version_id=version_id,
            storage_locator=storage_locator_payload,
            format_version=self.descriptor.format_version,
        )
        return CheckpointComponentPublication(
            component_id=self.descriptor.component_id,
            format_version=self.descriptor.format_version,
            shared_records=(
                CheckpointObjectRecord(checkpoint_objects=(checkpoint_object,)),
            ),
        )

    async def rollback_save(self, session: SaveSession) -> None:
        self._require_worker(session)
        identity = self._identity(session, session.commit_permit)
        if session.rank == 0:
            await asyncio.to_thread(self._require_storage().abort_checkpoint, identity)

    async def publish_save(self, session: SaveSession) -> None:
        self._require_worker(session)
        identity = self._identity(session, session.commit_permit)
        if session.rank == 0:
            await asyncio.to_thread(
                self._require_storage().acknowledge_checkpoint,
                identity,
            )

    async def restore(self, session: RestoreSession) -> None:
        self._require_worker(session)
        publication = session.component_checkpoint
        if publication.component_id != self.descriptor.component_id:
            raise ValueError("sparse publication belongs to another component")
        if publication.format_version != self.descriptor.format_version:
            raise ValueError("sparse publication format version is unsupported")
        if len(publication.shared_records) != 1:
            raise ValueError("sparse publication requires one shared record")
        record = publication.shared_records[0]
        if len(record.checkpoint_objects) != 1:
            raise ValueError("sparse publication requires one checkpoint_object")
        checkpoint_object = record.checkpoint_objects[0]
        if checkpoint_object.component_id != self.descriptor.component_id:
            raise ValueError("sparse checkpoint_object belongs to another component")
        if checkpoint_object.checkpoint_object_id != self.descriptor.component_id:
            raise ValueError(
                "sparse checkpoint_object belongs to another checkpoint_object"
            )
        if checkpoint_object.format_version != self.descriptor.format_version:
            raise ValueError("sparse checkpoint_object format version is unsupported")
        storage_locator = self._decode_storage_locator(
            checkpoint_object.storage_locator
        )
        topology = session.current_topology or session.source_topology
        mode = ElasticRestore(
            partition_fn=self._partition_fn,
            partition_count=topology.world_size,
            target_partition_id=session.rank,
        )
        await asyncio.to_thread(
            self._require_storage().restore_checkpoint,
            session.instance.store,
            storage_locator,
            mode,
        )

    async def hold_checkpoint_object(
        self,
        checkpoint_object: CheckpointObjectRef,
        lease_holder_id: str,
        ttl_seconds: float,
        request_id: str,
    ) -> CheckpointHold:
        self._require_control()
        storage_locator = self._storage_locator_from_checkpoint_object(
            checkpoint_object
        )
        try:
            lease = await asyncio.to_thread(
                self._require_storage().acquire_checkpoint_lease,
                storage_locator,
                SparseLeaseRequest(lease_holder_id, ttl_seconds, request_id),
            )
        except CheckpointConflictError as error:
            raise LeaseConflictError(str(error)) from error
        return self._from_sparse_lease(checkpoint_object, lease)

    async def renew_checkpoint_hold(
        self, lease: CheckpointHold, ttl_seconds: float
    ) -> CheckpointHold:
        self._require_control()
        try:
            renewed = await asyncio.to_thread(
                self._require_storage().renew_checkpoint_lease,
                self._to_sparse_lease(lease),
                ttl_seconds,
            )
        except CheckpointConflictError as error:
            raise LeaseConflictError(str(error)) from error
        return self._from_sparse_lease(lease.checkpoint_object, renewed)

    async def release_checkpoint_hold(self, lease: CheckpointHold) -> None:
        self._require_control()
        try:
            await asyncio.to_thread(
                self._require_storage().release_checkpoint_lease,
                self._to_sparse_lease(lease),
            )
        except CheckpointConflictError as error:
            raise LeaseConflictError(str(error)) from error

    async def reserve_cleanup(
        self, request: CleanupReservationRequest
    ) -> CleanupReservation:
        self._require_control()
        candidates = tuple(
            checkpoint_object
            for checkpoint_object in request.ordered_candidates
            if checkpoint_object.version_id not in request.protected_versions
        )
        storage_locators = tuple(
            self._storage_locator_from_checkpoint_object(item) for item in candidates
        )
        reservation_id = self._reservation_id(
            request.permit.attempt_id,
            request.permit.coordinator_generation,
            request.checkpoint_object_id,
        )
        try:
            sparse_reservation = await asyncio.to_thread(
                self._require_storage().reserve_cleanup,
                reservation_id,
                storage_locators,
                request.required_count,
            )
        except CheckpointConflictError as error:
            raise CheckpointCleanupBlocked(str(error)) from error
        by_storage_locator = dict(zip(storage_locators, candidates, strict=True))
        return CleanupReservation(
            attempt_id=request.permit.attempt_id,
            checkpoint_object_id=request.checkpoint_object_id,
            claimed=tuple(
                by_storage_locator[item] for item in sparse_reservation.locators
            ),
            generation=request.permit.coordinator_generation,
        )

    async def commit_cleanup(
        self,
        reservation: CleanupReservation,
        permit: CheckpointCommitPermit,
    ) -> None:
        self._require_control()
        sparse = self._sparse_reservation(reservation, permit)
        await asyncio.to_thread(self._require_storage().commit_cleanup, sparse)

    async def rollback_cleanup(
        self,
        reservation: CleanupReservation,
        permit: CheckpointCommitPermit,
    ) -> None:
        self._require_control()
        sparse = self._sparse_reservation(reservation, permit)
        await asyncio.to_thread(self._require_storage().rollback_cleanup, sparse)

    async def cleanup(
        self,
        reservation: CleanupReservation,
    ) -> None:
        self._require_control()
        sparse = self._sparse_reservation_from_reservation(reservation)
        for checkpoint_object, storage_locator in zip(
            reservation.claimed, sparse.locators, strict=True
        ):
            await asyncio.to_thread(
                self._require_storage().cleanup_checkpoint,
                sparse,
                storage_locator,
                self._reference_id(
                    checkpoint_object.component_id, storage_locator.checkpoint_name
                ),
            )

    async def close(self) -> None:
        if self._runtime is None or self._storage is None:
            raise RuntimeError(
                f"sparse checkpoint component {self.descriptor.component_id!r} is not open"
            )
        storage = self._storage
        try:
            await asyncio.to_thread(storage.close)
        finally:
            self._storage = None
            self._runtime = None

    def _require_worker(self, context: SaveSession | RestoreSession) -> None:
        runtime = self._runtime
        if runtime is None or self._storage is None:
            raise RuntimeError(
                f"sparse checkpoint component {self.descriptor.component_id!r} is not open"
            )
        if runtime.role is not CheckpointRuntimeRole.WORKER:
            raise RuntimeError(
                "sparse checkpoint save and restore require a worker runtime"
            )
        if context.component_id != self.descriptor.component_id:
            raise ValueError("checkpoint context belongs to another component")

    def _require_control(self) -> None:
        runtime = self._runtime
        if runtime is None or self._storage is None:
            raise RuntimeError(
                f"sparse checkpoint component {self.descriptor.component_id!r} is not open"
            )
        if runtime.role is not CheckpointRuntimeRole.CONTROL:
            raise RuntimeError("sparse cleanup requires a control runtime")

    def _require_storage(self) -> SparseCheckpointStorage:
        if self._storage is None:
            raise RuntimeError(
                f"sparse checkpoint component {self.descriptor.component_id!r} is not open"
            )
        return self._storage

    def _storage_locator_from_checkpoint_object(
        self, checkpoint_object: CheckpointObjectRef
    ) -> SparseCheckpointLocator:
        if checkpoint_object.component_id != self.descriptor.component_id:
            raise ValueError("sparse checkpoint_object belongs to another component")
        if checkpoint_object.checkpoint_object_id != self.descriptor.component_id:
            raise ValueError(
                "sparse checkpoint_object belongs to another checkpoint_object"
            )
        if checkpoint_object.format_version != self.descriptor.format_version:
            raise ValueError("sparse checkpoint_object format version is unsupported")
        return self._decode_storage_locator(checkpoint_object.storage_locator)

    @staticmethod
    def _from_sparse_lease(
        checkpoint_object: CheckpointObjectRef, lease: SparseCheckpointLease
    ) -> CheckpointHold:
        return CheckpointHold(
            lease_id=lease.lease_id,
            checkpoint_object=checkpoint_object,
            lease_holder_id=lease.lease_holder_id,
            expires_at=lease.expires_at.timestamp(),
            generation=lease.generation,
            request_id=lease.request_id,
        )

    def _to_sparse_lease(self, lease: CheckpointHold) -> SparseCheckpointLease:
        return SparseCheckpointLease(
            lease_id=lease.lease_id,
            locator=self._storage_locator_from_checkpoint_object(
                lease.checkpoint_object
            ),
            lease_holder_id=lease.lease_holder_id,
            expires_at=datetime.fromtimestamp(lease.expires_at, UTC),
            generation=lease.generation,
            request_id=lease.request_id,
        )

    def _sparse_reservation(
        self,
        reservation: CleanupReservation,
        permit: CheckpointCommitPermit,
    ) -> SparseCleanupReservation:
        if (
            reservation.attempt_id != permit.attempt_id
            or reservation.generation != permit.coordinator_generation
        ):
            raise PermissionError(
                "cleanup reservation does not match commit_save permit"
            )
        return self._sparse_reservation_from_reservation(reservation)

    def _sparse_reservation_from_reservation(
        self,
        reservation: CleanupReservation,
    ) -> SparseCleanupReservation:
        return SparseCleanupReservation(
            self._reservation_id(
                reservation.attempt_id,
                reservation.generation,
                reservation.checkpoint_object_id,
            ),
            tuple(
                self._storage_locator_from_checkpoint_object(checkpoint_object)
                for checkpoint_object in reservation.claimed
            ),
        )

    def _reservation_id(
        self, attempt_id: str, generation: int, checkpoint_object_id: str
    ) -> str:
        return ":".join(
            (
                "fluxway",
                self.descriptor.component_id,
                checkpoint_object_id,
                str(generation),
                attempt_id,
            )
        )

    @staticmethod
    def _reference_id(component_id: str, checkpoint_name: str) -> str:
        return f"fluxway:{component_id}:{checkpoint_name}"

    @staticmethod
    def _identity(
        context: SaveSession, permit: CheckpointCommitPermit
    ) -> CheckpointSaveIdentity:
        if context.commit_permit != permit:
            raise PermissionError(
                "checkpoint commit_save permit does not match context"
            )
        return CheckpointSaveIdentity(
            checkpoint_name=context.checkpoint_id,
            save_attempt=context.attempt_id,
            commit_authority=CheckpointCommitAuthority(
                scope=f"fluxway:{context.spmd_id}:{context.component_id}",
                generation=permit.coordinator_generation,
                authority_id=hashlib.sha256(permit.token).hexdigest(),
            ),
        )

    @staticmethod
    def _encode_fragment(fragment: CheckpointFragment) -> bytes:
        return cast(
            bytes,
            msgpack.packb(
                {
                    "wire_version": _WIRE_VERSION,
                    "fragment_id": fragment.fragment_id,
                    "checkpoint_name": fragment.checkpoint_name,
                    "save_attempt": fragment.save_attempt,
                    "participant_index": fragment.participant.participant_index,
                    "participant_count": fragment.participant.participant_count,
                    "entry_count": fragment.entry_count,
                    "page_count": fragment.page_count,
                    "schema_digest": fragment.schema_digest,
                    "content_digest": fragment.content_digest,
                },
                use_bin_type=True,
            ),
        )

    @staticmethod
    def _decode_fragment(payload: bytes) -> CheckpointFragment:
        decoded = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        if (
            not isinstance(decoded, dict)
            or decoded.get("wire_version") != _WIRE_VERSION
        ):
            raise ValueError("sparse fragment wire payload is invalid")
        return CheckpointFragment(
            fragment_id=decoded["fragment_id"],
            checkpoint_name=decoded["checkpoint_name"],
            save_attempt=decoded["save_attempt"],
            participant=CheckpointParticipant(
                decoded["participant_index"], decoded["participant_count"]
            ),
            entry_count=decoded["entry_count"],
            page_count=decoded["page_count"],
            schema_digest=decoded["schema_digest"],
            content_digest=decoded["content_digest"],
        )

    @staticmethod
    def _encode_storage_locator(storage_locator: SparseCheckpointLocator) -> bytes:
        return cast(
            bytes,
            msgpack.packb(
                {
                    "wire_version": _WIRE_VERSION,
                    "checkpoint_name": storage_locator.checkpoint_name,
                    "checkpoint_version": storage_locator.checkpoint_version,
                },
                use_bin_type=True,
            ),
        )

    @staticmethod
    def _decode_storage_locator(payload: bytes) -> SparseCheckpointLocator:
        decoded = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        if (
            not isinstance(decoded, dict)
            or decoded.get("wire_version") != _WIRE_VERSION
        ):
            raise ValueError("sparse storage_locator wire payload is invalid")
        return SparseCheckpointLocator(
            decoded["checkpoint_name"], decoded["checkpoint_version"]
        )
