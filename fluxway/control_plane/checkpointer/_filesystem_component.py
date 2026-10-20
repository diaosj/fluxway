"""Private filesystem implementation shared by built-in state components."""

from __future__ import annotations

import asyncio
import copy
import fcntl
import hashlib
import inspect
import os
import shutil
import time
import uuid
from abc import abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from fluxway.common import msgpack_codec

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
    LeaseConflictError,
    PartitionRecord,
    WorkerSaveResult,
    RestoreTopologyMode,
    CheckpointCleanupBlocked,
    CheckpointHold,
    CleanupReservation,
    CleanupReservationRequest,
    TopologyMismatchError,
)
from fluxway.control_plane.checkpointer.component import (
    CheckpointComponentRegistration,
    CheckpointComponent,
    CheckpointComponentLifecycle,
    CheckpointComponentFinalize,
    CheckpointObjectCleanup,
    CheckpointObjectLifecycle,
    CheckpointComponentRestore,
    CheckpointComponentSave,
)
from fluxway.control_plane.checkpointer.repository import validate_checkpoint_value


def _as_async(function: Callable[..., Any], name: str) -> Callable[..., Awaitable[Any]]:
    if not callable(function):
        raise TypeError(f"{name} must be callable")
    if inspect.iscoroutinefunction(function):
        return function

    async def invoke(*args: Any) -> Any:
        result = function(*args)
        if inspect.isawaitable(result):
            raise TypeError(f"{name} returned an awaitable but is not async def")
        return result

    return invoke


class FilesystemCheckpointComponent(
    CheckpointComponent,
    CheckpointComponentLifecycle,
    CheckpointComponentFinalize,
    CheckpointComponentSave,
    CheckpointComponentRestore,
    CheckpointObjectLifecycle,
    CheckpointObjectCleanup,
):
    """Internal transaction and lifecycle implementation for one state format."""

    def __init__(
        self,
        component_id: str,
        save_state: Callable[[Any, SaveSession], Any],
        restore_state: Callable[[Any, Any, RestoreSession], Any],
        *,
        finalize_state: Callable[[Any, SaveSession], Any] | None = None,
        retention_domain: str | None = None,
        restore_topology: RestoreTopologyMode,
        output_path: str | None = None,
    ) -> None:
        self._descriptor = CheckpointComponentSpec(
            component_id=component_id,
            retention_domain=retention_domain,
            checkpoint_objects=(CheckpointObjectSpec("state"),),
            save=CheckpointSaveSpec(
                prepare_scope=CheckpointSaveScope.LEADER,
                commit_scope=CheckpointSaveScope.LEADER,
            ),
            restore=CheckpointRestoreSpec(topology=restore_topology),
            output_path=output_path,
        )
        self._save_state = _as_async(save_state, "save_state")
        self._restore_state = _as_async(restore_state, "restore_state")
        self._finalize_state = (
            _as_async(finalize_state, "finalize_state")
            if finalize_state is not None
            else None
        )
        self._runtime: CheckpointRuntime | None = None
        self._root: Path | None = None

    @property
    def descriptor(self) -> CheckpointComponentSpec:
        return self._descriptor

    def registration(self) -> CheckpointComponentRegistration:
        return CheckpointComponentRegistration(
            component=self,
            component_lifecycle=self,
            finalize=self if self._finalize_state is not None else None,
            save=self,
            restore=self,
            object_lifecycle=self,
            object_cleanup=self,
        )

    async def finalize_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker(session)
        if self._finalize_state is not None:
            await self._finalize_state(session.instance, session)

    async def open(self, runtime: CheckpointRuntime) -> None:
        if self._runtime is not None:
            raise RuntimeError(
                f"checkpoint component {self.descriptor.component_id!r} is open"
            )
        root = Path(runtime.repository_root).resolve()
        await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
        self._runtime = runtime
        self._root = root
        await self._open_resources(runtime)

    async def prepare_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker(session)
        staging = self._staging_root(session.attempt_id)
        if staging.exists():
            raise FileExistsError(
                f"checkpoint staging directory already exists: {staging}"
            )
        await asyncio.to_thread(staging.mkdir, parents=True, exist_ok=False)

    async def save(self, session: SaveSession) -> WorkerSaveResult:
        self._require_permit(session, session.commit_permit)
        self._require_worker(session)
        partition_key = f"rank/{session.rank}"
        staging = self._staging_path(session, partition_key)
        final = self._final_path(session, partition_key)
        if staging.exists() or final.exists():
            raise FileExistsError(
                f"checkpoint checkpoint_object path already exists: staging={staging} final={final}"
            )
        await asyncio.to_thread(staging.mkdir, parents=True, exist_ok=False)
        state = await self._save_state(session.instance, session)
        metadata = await self._write_state(state, staging, session)
        validate_checkpoint_value(metadata, "filesystem_component.metadata")
        checkpoint_object = CheckpointObjectRef(
            component_id=self.descriptor.component_id,
            checkpoint_object_id="state",
            version_id=session.checkpoint_id,
            storage_locator=final.as_uri().encode("utf-8"),
            partition_key=partition_key,
        )
        return WorkerSaveResult(
            record=CheckpointObjectRecord(
                payload=msgpack_codec.packb(metadata, use_bin_type=True),
                checkpoint_objects=(checkpoint_object,),
            ),
            commit_metadata=msgpack_codec.packb(
                {"staging_uri": staging.as_uri()}, use_bin_type=True
            ),
        )

    async def finish_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker(session)
        await self._finish_save_io(session)

    async def commit_save(
        self, session: CommitSaveSession
    ) -> CheckpointComponentPublication:
        self._require_permit(session, session.commit_permit)
        self._require_worker(session)
        moves: list[tuple[Path, Path]] = []
        partitions: list[PartitionRecord] = []
        try:
            for rank, prepared in sorted(session.worker_results.items()):
                if (
                    prepared.record is None
                    or len(prepared.record.checkpoint_objects) != 1
                ):
                    raise ValueError(f"checkpoint rank {rank} has an invalid record")
                metadata = msgpack_codec.unpackb(
                    prepared.commit_metadata, raw=False, strict_map_key=False
                )
                if not isinstance(metadata, Mapping):
                    raise TypeError(
                        f"checkpoint rank {rank} has invalid commit_save metadata"
                    )
                staging = self._file_path(metadata["staging_uri"].encode("utf-8"))
                checkpoint_object = prepared.record.checkpoint_objects[0]
                final = self._file_path(checkpoint_object.storage_locator)
                if not staging.is_dir():
                    raise FileNotFoundError(
                        f"checkpoint staging checkpoint_object missing: {staging}"
                    )
                if final.exists():
                    raise FileExistsError(
                        f"checkpoint final checkpoint_object exists: {final}"
                    )
                await asyncio.to_thread(final.parent.mkdir, parents=True, exist_ok=True)
                await asyncio.to_thread(os.replace, staging, final)
                moves.append((staging, final))
                partition_key = checkpoint_object.partition_key
                if partition_key is None:
                    raise ValueError(
                        "filesystem checkpoint_object requires partition_key"
                    )
                partitions.append(PartitionRecord(partition_key, rank, prepared.record))
        except BaseException:
            for staging, final in reversed(moves):
                if final.exists():
                    await asyncio.to_thread(os.replace, final, staging)
            raise
        return CheckpointComponentPublication(
            component_id=self.descriptor.component_id,
            partitions=tuple(partitions),
        )

    async def rollback_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker(session)
        partition_key = f"rank/{session.rank}"
        for path in (
            self._staging_path(session, partition_key),
            self._final_path(session, partition_key),
        ):
            try:
                await asyncio.to_thread(shutil.rmtree, path)
            except FileNotFoundError:
                pass

    async def publish_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker(session)

    async def abort_save(self, permit: CheckpointCommitPermit) -> None:
        self._require_control()
        for path in (
            self._staging_root(permit.attempt_id),
            self._final_component_root(permit.attempt_id),
        ):
            try:
                await asyncio.to_thread(shutil.rmtree, path)
            except FileNotFoundError:
                pass

    async def acknowledge_save(self, permit: CheckpointCommitPermit) -> None:
        # Filesystem payloads have no owner intent beyond the global manifest.
        self._require_control()

    async def restore(self, session: RestoreSession) -> None:
        self._require_worker(session)
        publication = session.component_checkpoint
        if publication.component_id != self.descriptor.component_id:
            raise ValueError("checkpoint publication belongs to another component")

        restore_spec = self.descriptor.restore
        if restore_spec is None:
            raise RuntimeError(
                f"checkpoint component {self.descriptor.component_id!r} has no restore specification"
            )

        current = session.current_topology or session.source_topology
        if (
            restore_spec.topology is RestoreTopologyMode.EXACT_TOPOLOGY
            and current != session.source_topology
        ):
            raise TopologyMismatchError(
                f"component {self.descriptor.component_id!r} requires exact topology"
            )
        record = self._select_restore_record(publication, session)
        state = await self._read_state(record, session)
        await self._restore_state(session.instance, state, session)

    async def hold_checkpoint_object(
        self,
        checkpoint_object: CheckpointObjectRef,
        lease_holder_id: str,
        ttl_seconds: float,
        request_id: str,
    ) -> CheckpointHold:
        self._require_control()
        if checkpoint_object.component_id != self.descriptor.component_id:
            raise ValueError("checkpoint_object belongs to another component")
        if not lease_holder_id or not request_id or ttl_seconds <= 0:
            raise ValueError(
                "lease requires lease holder, request ID, and positive TTL"
            )
        with self._lifecycle_lock():
            state = self._read_lifecycle(checkpoint_object)
            self._drop_expired_leases(state)
            if state["status"] != "live":
                raise LeaseConflictError(
                    f"checkpoint_object is not leaseable in lifecycle state {state['status']!r}"
                )
            if state["claim"] is not None:
                raise LeaseConflictError("checkpoint_object is reserved for cleanup")
            for raw in state["leases"].values():
                if raw["request_id"] == request_id:
                    if raw["lease_holder_id"] != lease_holder_id:
                        raise LeaseConflictError(
                            "lease request ID belongs to another lease holder"
                        )
                    self._write_lifecycle(checkpoint_object, state)
                    return self._lease_from_payload(checkpoint_object, raw)
            lease_id = uuid.uuid4().hex
            raw = {
                "lease_id": lease_id,
                "lease_holder_id": lease_holder_id,
                "expires_at": time.time() + ttl_seconds,
                "generation": 0,
                "request_id": request_id,
            }
            state["leases"][lease_id] = raw
            self._write_lifecycle(checkpoint_object, state)
            return self._lease_from_payload(checkpoint_object, raw)

    async def renew_checkpoint_hold(
        self, lease: CheckpointHold, ttl_seconds: float
    ) -> CheckpointHold:
        self._require_control()
        if ttl_seconds <= 0:
            raise ValueError("lease renewal TTL must be positive")
        with self._lifecycle_lock():
            state = self._read_lifecycle(lease.checkpoint_object)
            self._drop_expired_leases(state)
            raw = state["leases"].get(lease.lease_id)
            if raw is None or raw["generation"] != lease.generation:
                raise LeaseConflictError("lease generation CAS conflict")
            raw["generation"] += 1
            raw["expires_at"] = time.time() + ttl_seconds
            self._write_lifecycle(lease.checkpoint_object, state)
            return self._lease_from_payload(lease.checkpoint_object, raw)

    async def release_checkpoint_hold(self, lease: CheckpointHold) -> None:
        self._require_control()
        with self._lifecycle_lock():
            state = self._read_lifecycle(lease.checkpoint_object)
            raw = state["leases"].get(lease.lease_id)
            if raw is None:
                if state["released"].get(lease.lease_id) == lease.generation:
                    return
                raise LeaseConflictError("lease does not exist")
            if raw["generation"] != lease.generation:
                raise LeaseConflictError("lease generation CAS conflict")
            del state["leases"][lease.lease_id]
            state["released"][lease.lease_id] = lease.generation
            self._write_lifecycle(lease.checkpoint_object, state)

    async def reserve_cleanup(
        self, request: CleanupReservationRequest
    ) -> CleanupReservation:
        self._require_control()
        with self._lifecycle_lock():
            generation: int | None = None
            grouped: dict[str, list[tuple[CheckpointObjectRef, dict[str, Any]]]] = {}
            for checkpoint_object in request.ordered_candidates:
                if checkpoint_object.component_id != self.descriptor.component_id:
                    raise ValueError("cleanup candidate belongs to another component")
                state = self._read_lifecycle(checkpoint_object)
                self._drop_expired_leases(state)
                grouped.setdefault(checkpoint_object.version_id, []).append(
                    (checkpoint_object, state)
                )

            claimed_versions: list[
                list[tuple[CheckpointObjectRef, dict[str, Any]]]
            ] = []
            available_versions: list[
                list[tuple[CheckpointObjectRef, dict[str, Any]]]
            ] = []
            for version_id, version in grouped.items():
                if version_id in request.protected_versions:
                    continue
                claims = tuple(state["claim"] for _, state in version)
                owned = tuple(
                    isinstance(claim, dict)
                    and claim.get("attempt_id") == request.permit.attempt_id
                    for claim in claims
                )
                if any(owned):
                    if not all(owned):
                        raise LeaseConflictError(
                            "cleanup transaction has a partial checkpoint_object-version claim"
                        )
                    claim_generations = {claim.get("generation") for claim in claims}
                    if len(claim_generations) != 1:
                        raise LeaseConflictError(
                            "cleanup transaction has conflicting generations"
                        )
                    claim_generation = next(iter(claim_generations))
                    if not isinstance(claim_generation, int):
                        raise ValueError("cleanup claim generation is invalid")
                    if generation is None:
                        generation = claim_generation
                    elif generation != claim_generation:
                        raise LeaseConflictError(
                            "cleanup transaction has conflicting generations"
                        )
                    claimed_versions.append(version)
                    continue
                if any(
                    state["status"] != "live"
                    or state["leases"]
                    or state["claim"] is not None
                    for _, state in version
                ):
                    continue
                available_versions.append(version)

            needed = request.required_count - len(claimed_versions)
            if needed < 0:
                raise LeaseConflictError(
                    "cleanup transaction already claimed more versions than requested"
                )
            if len(available_versions) < needed:
                raise CheckpointCleanupBlocked(
                    f"component {self.descriptor.component_id!r} checkpoint_object "
                    f"{request.checkpoint_object_id!r} has {len(available_versions)} "
                    "cleanable versions "
                    f"but requires {needed}"
                )
            if generation is None:
                generation = request.permit.coordinator_generation
            selected_versions = available_versions[:needed]
            written: list[tuple[CheckpointObjectRef, dict[str, Any]]] = []
            try:
                for version in selected_versions:
                    for checkpoint_object, state in version:
                        original = copy.deepcopy(state)
                        state["claim"] = {
                            "attempt_id": request.permit.attempt_id,
                            "generation": generation,
                            "status": "reserved",
                        }
                        written.append((checkpoint_object, original))
                        self._write_lifecycle(checkpoint_object, state)
            except BaseException as primary_error:
                rollback_errors: list[BaseException] = []
                for checkpoint_object, original in reversed(written):
                    try:
                        self._write_lifecycle(checkpoint_object, original)
                    except BaseException as rollback_error:
                        rollback_errors.append(rollback_error)
                if rollback_errors:
                    raise BaseExceptionGroup(
                        "cleanup reservation and rollback failed",
                        [primary_error, *rollback_errors],
                    ) from None
                raise
            claimed = tuple(
                checkpoint_object
                for version in (*claimed_versions, *selected_versions)
                for checkpoint_object, _ in version
            )
        return CleanupReservation(
            attempt_id=request.permit.attempt_id,
            checkpoint_object_id=request.checkpoint_object_id,
            claimed=claimed,
            generation=generation,
        )

    async def commit_cleanup(
        self,
        reservation: CleanupReservation,
        permit: CheckpointCommitPermit,
    ) -> None:
        self._require_control()
        self._require_reservation_permit(reservation, permit)
        with self._lifecycle_lock():
            for checkpoint_object in reservation.claimed:
                state = self._read_lifecycle(checkpoint_object)
                claim = state["claim"]
                completed = state["completed_claim"]
                if (
                    claim is None
                    and isinstance(completed, dict)
                    and completed.get("attempt_id") == reservation.attempt_id
                    and completed.get("generation") == reservation.generation
                ):
                    continue
                claim = self._require_claim(state, reservation)
                if claim["status"] == "pending":
                    continue
                claim["status"] = "pending"
                self._write_lifecycle(checkpoint_object, state)

    async def rollback_cleanup(
        self,
        reservation: CleanupReservation,
        permit: CheckpointCommitPermit,
    ) -> None:
        self._require_control()
        self._require_reservation_permit(reservation, permit)
        with self._lifecycle_lock():
            for checkpoint_object in reservation.claimed:
                state = self._read_lifecycle(checkpoint_object)
                self._require_claim(state, reservation)
                state["claim"] = None
                state["status"] = "live"
                self._write_lifecycle(checkpoint_object, state)

    async def cleanup(
        self,
        reservation: CleanupReservation,
    ) -> None:
        self._require_control()
        errors: list[BaseException] = []
        for checkpoint_object in reservation.claimed:
            with self._lifecycle_lock():
                state = self._read_lifecycle(checkpoint_object)
                claim = self._require_claim(state, reservation)
                if claim["status"] != "pending":
                    raise LeaseConflictError(
                        "checkpoint_object cleanup was not published"
                    )
                claim["status"] = "cleaning_up"
                self._write_lifecycle(checkpoint_object, state)
            path = self._file_path(checkpoint_object.storage_locator)
            try:
                if path.is_dir():
                    await asyncio.to_thread(shutil.rmtree, path)
                elif path.exists():
                    await asyncio.to_thread(path.unlink)
            except BaseException as error:
                errors.append(error)
                with self._lifecycle_lock():
                    state = self._read_lifecycle(checkpoint_object)
                    claim = self._require_claim(state, reservation)
                    claim["status"] = "pending"
                    self._write_lifecycle(checkpoint_object, state)
                continue
            with self._lifecycle_lock():
                state = self._read_lifecycle(checkpoint_object)
                self._require_claim(state, reservation)
                state["claim"] = None
                state["status"] = "cleaned_up"
                state["completed_claim"] = {
                    "attempt_id": reservation.attempt_id,
                    "generation": reservation.generation,
                }
                self._write_lifecycle(checkpoint_object, state)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("checkpoint_object cleanup failed", errors)

    async def close(self) -> None:
        if self._runtime is None:
            raise RuntimeError(
                f"checkpoint component {self.descriptor.component_id!r} is not open"
            )
        await self._close_resources()
        self._runtime = None
        self._root = None

    def _select_restore_record(
        self, publication: CheckpointComponentPublication, context: RestoreSession
    ) -> CheckpointObjectRecord:
        partition_key = f"rank/{context.rank}"
        matches = [
            partition.record
            for partition in publication.partitions
            if partition.partition_key == partition_key
        ]
        if len(matches) != 1:
            raise KeyError(f"checkpoint partition not found: {partition_key!r}")
        return matches[0]

    def _require_worker(
        self, context: SaveSession | RestoreSession
    ) -> CheckpointRuntime:
        runtime = self._require_runtime()
        if runtime.role is not CheckpointRuntimeRole.WORKER:
            raise RuntimeError("checkpoint state operation requires worker runtime")
        if context.component_id != self.descriptor.component_id:
            raise ValueError("checkpoint context belongs to another component")
        return runtime

    @staticmethod
    def _require_permit(context: SaveSession, permit: CheckpointCommitPermit) -> None:
        if permit is None or context.commit_permit != permit:
            raise PermissionError(
                "checkpoint commit_save permit does not match context"
            )

    def _require_control(self) -> CheckpointRuntime:
        runtime = self._require_runtime()
        if runtime.role is not CheckpointRuntimeRole.CONTROL:
            raise RuntimeError(
                "checkpoint_object lifecycle operation requires control runtime"
            )
        return runtime

    def _require_runtime(self) -> CheckpointRuntime:
        if self._runtime is None or self._root is None:
            raise RuntimeError(
                f"checkpoint component {self.descriptor.component_id!r} is not open"
            )
        return self._runtime

    def _staging_root(self, attempt_id: str) -> Path:
        runtime = self._require_runtime()
        root = self._require_root()
        return (
            root
            / ".checkpoint-transactions"
            / quote(attempt_id, safe="")
            / quote(runtime.spmd_id, safe="")
            / quote(self.descriptor.component_id, safe="")
        )

    def _staging_path(self, context: SaveSession, partition_key: str) -> Path:
        return self._staging_root(context.attempt_id) / quote(partition_key, safe="")

    def _final_component_root(self, attempt_id: str) -> Path:
        runtime = self._require_runtime()
        root = self._require_root()
        return (
            root
            / "checkpoint_objects"
            / quote(attempt_id, safe="")
            / quote(runtime.spmd_id, safe="")
            / quote(self.descriptor.component_id, safe="")
        )

    def _final_path(self, context: SaveSession, partition_key: str) -> Path:
        return (
            self._final_component_root(context.attempt_id)
            / "state"
            / quote(partition_key, safe="")
        )

    def _require_root(self) -> Path:
        if self._root is None:
            raise RuntimeError(
                f"checkpoint component {self.descriptor.component_id!r} is not open"
            )
        return self._root

    def _lifecycle_root(self) -> Path:
        runtime = self._require_runtime()
        root = (
            self._require_root()
            / "checkpoint_objects"
            / ".lifecycle"
            / quote(runtime.spmd_id, safe="")
            / quote(self.descriptor.component_id, safe="")
        )
        root.mkdir(parents=True, exist_ok=True)
        return root

    @contextmanager
    def _lifecycle_lock(self) -> Generator[None, None, None]:
        lock_path = self._lifecycle_root() / ".lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _lifecycle_path(self, checkpoint_object: CheckpointObjectRef) -> Path:
        identity = b"\x00".join(
            (
                (
                    checkpoint_object.version_id
                    if isinstance(checkpoint_object.version_id, bytes)
                    else checkpoint_object.version_id.encode("utf-8")
                ),
                (checkpoint_object.partition_key or "").encode("utf-8"),
                checkpoint_object.storage_locator,
            )
        )
        digest = hashlib.sha256(identity).hexdigest()
        return (
            self._lifecycle_root()
            / quote(checkpoint_object.checkpoint_object_id, safe="")
            / f"{digest}.msgpack"
        )

    def _read_lifecycle(self, checkpoint_object: CheckpointObjectRef) -> dict[str, Any]:
        path = self._lifecycle_path(checkpoint_object)
        if not path.exists():
            return {
                "status": "live",
                "leases": {},
                "released": {},
                "claim": None,
                "completed_claim": None,
            }
        payload = msgpack_codec.unpackb(
            path.read_bytes(), raw=False, strict_map_key=False
        )
        if not isinstance(payload, dict):
            raise TypeError(f"checkpoint_object lifecycle state is corrupt: {path}")
        expected = {"status", "leases", "released", "claim", "completed_claim"}
        if set(payload) != expected:
            raise ValueError(
                f"checkpoint_object lifecycle state fields are invalid: {path}"
            )
        return payload

    def _write_lifecycle(
        self, checkpoint_object: CheckpointObjectRef, state: dict[str, Any]
    ) -> None:
        path = self._lifecycle_path(checkpoint_object)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(msgpack_codec.packb(state, use_bin_type=True))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _drop_expired_leases(state: dict[str, Any]) -> None:
        now = time.time()
        expired = [
            lease_id
            for lease_id, raw in state["leases"].items()
            if raw["expires_at"] <= now
        ]
        for lease_id in expired:
            del state["leases"][lease_id]

    @staticmethod
    def _lease_from_payload(
        checkpoint_object: CheckpointObjectRef, payload: Mapping[str, Any]
    ) -> CheckpointHold:
        return CheckpointHold(
            lease_id=payload["lease_id"],
            checkpoint_object=checkpoint_object,
            lease_holder_id=payload["lease_holder_id"],
            expires_at=payload["expires_at"],
            generation=payload["generation"],
            request_id=payload["request_id"],
        )

    @staticmethod
    def _require_claim(
        state: dict[str, Any], reservation: CleanupReservation
    ) -> dict[str, Any]:
        claim = state["claim"]
        if (
            not isinstance(claim, dict)
            or claim.get("attempt_id") != reservation.attempt_id
            or claim.get("generation") != reservation.generation
        ):
            raise LeaseConflictError("cleanup reservation CAS conflict")
        return claim

    @staticmethod
    def _require_reservation_permit(
        reservation: CleanupReservation,
        permit: CheckpointCommitPermit,
    ) -> None:
        if reservation.attempt_id != permit.attempt_id:
            raise LeaseConflictError("cleanup reservation belongs to another attempt")

    @staticmethod
    def _file_path(storage_locator: bytes) -> Path:
        if not isinstance(storage_locator, bytes):
            raise TypeError(
                "filesystem checkpoint_object storage_locator must be bytes"
            )
        uri = storage_locator.decode("utf-8")
        parsed = urlparse(uri)
        if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
            raise ValueError(f"filesystem component cannot access URI: {uri!r}")
        return Path(unquote(parsed.path))

    async def _open_resources(self, runtime: CheckpointRuntime) -> None:
        return None

    @abstractmethod
    async def _write_state(
        self, state: Any, directory: Path, context: SaveSession
    ) -> Mapping[str, Any]: ...

    @abstractmethod
    async def _read_state(
        self, record: CheckpointObjectRecord, context: RestoreSession
    ) -> Any: ...

    @abstractmethod
    async def _finish_save_io(self, context: SaveSession) -> None: ...

    @abstractmethod
    async def _close_resources(self) -> None: ...
