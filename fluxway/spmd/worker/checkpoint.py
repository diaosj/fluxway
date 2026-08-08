"""Worker-local checkpoint transaction coordinator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fluxway.common.types.checkpoint import (
    ComponentCheckpoint,
    CheckpointRuntime,
    CheckpointRuntimeRole,
    CheckpointBusyError,
    CheckpointCommitPermit,
    CheckpointCaptureMode,
    SaveSession,
    CommitSaveSession,
    RestoreSession,
    CheckpointCut,
    CheckpointTransactionState,
    CommitScope,
    EagerCheckpointUnsupportedError,
    PrepareScope,
    WorkerSaveResult,
    TopologySnapshot,
)
from fluxway.control_plane.checkpointer.backend import (
    CheckpointComponentRegistration,
    CheckpointComponentRegistry,
)


class WorkerCheckpointCoordinator:
    def __init__(
        self,
        backends: tuple[CheckpointComponentRegistration, ...],
        operator_instance: Any,
        repository_root: str,
        spmd_id: str,
        rank: int,
        source_topology: TopologySnapshot,
    ) -> None:
        if not repository_root or not spmd_id:
            raise ValueError("worker checkpoint coordinator requires root and SPMD ID")
        if rank < 0 or rank >= source_topology.world_size:
            raise ValueError("worker checkpoint coordinator rank is outside topology")
        self._registry = CheckpointComponentRegistry(backends)
        self._owner = operator_instance
        self._repository_root = repository_root
        self._spmd_id = spmd_id
        self._rank = rank
        self._source_topology = source_topology
        self._opened = False
        self._closed = False
        self._checkpoint_id: str | None = None
        self._attempt_id: str | None = None
        self._permit: CheckpointCommitPermit | None = None
        self._cut: CheckpointCut | None = None
        self._state: CheckpointTransactionState | None = None
        self._captured: dict[str, WorkerSaveResult] = {}

    @property
    def state(self) -> CheckpointTransactionState | None:
        return self._state

    @property
    def active_attempt_id(self) -> str | None:
        return self._attempt_id

    async def start_save(
        self,
        checkpoint_id: str,
        attempt_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
        capture_mode: CheckpointCaptureMode,
    ) -> None:
        self._require_available()
        if self._state is not None:
            raise CheckpointBusyError(
                f"worker already owns checkpoint attempt {self.active_attempt_id!r}"
            )
        if capture_mode is CheckpointCaptureMode.EAGER:
            raise EagerCheckpointUnsupportedError(
                "worker runtime does not yet provide an immutable eager save view"
            )
        await self._ensure_open()
        self._checkpoint_id = checkpoint_id
        self._attempt_id = attempt_id
        self._permit = permit
        self._cut = cut
        self._state = CheckpointTransactionState.CUT_CREATED
        attempted: list[CheckpointComponentRegistration] = []
        try:
            for backend in self._registry:
                if (
                    backend.descriptor.prepare_scope is PrepareScope.LEADER
                    and self._rank != 0
                ):
                    continue
                attempted.append(backend)
                await backend.component.start_save(
                    self._session(backend.descriptor.component_id)
                )
        except BaseException as primary_error:
            cleanup_errors = await self._abort_backends(tuple(reversed(attempted)))
            self._clear_transaction()
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "checkpoint start_save and rollback failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise

    async def save(self) -> Mapping[str, WorkerSaveResult]:
        self._require_state(CheckpointTransactionState.CUT_CREATED)
        self._state = CheckpointTransactionState.CAPTURING
        captured: dict[str, WorkerSaveResult] = {}
        try:
            for backend in self._registry:
                captured[
                    backend.descriptor.component_id
                ] = await backend.component.save(
                    self._session(backend.descriptor.component_id)
                )
        except BaseException as primary_error:
            cleanup_errors = await self._abort_backends(
                tuple(reversed(tuple(self._registry)))
            )
            self._clear_transaction()
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "checkpoint save and rollback failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise
        self._captured = captured
        self._state = CheckpointTransactionState.CAPTURED
        return dict(captured)

    async def finish_save(self) -> None:
        self._require_state(CheckpointTransactionState.CAPTURED)
        try:
            for backend in self._registry:
                await backend.component.finish_save(
                    self._session(backend.descriptor.component_id)
                )
        except BaseException as primary_error:
            cleanup_errors = await self._abort_backends(
                tuple(reversed(tuple(self._registry)))
            )
            self._clear_transaction()
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "checkpoint finish_save and rollback failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise
        self._state = CheckpointTransactionState.FINALIZED

    async def commit_save(
        self,
        gathered_records: Mapping[str, Mapping[int, WorkerSaveResult]],
    ) -> Mapping[str, ComponentCheckpoint]:
        self._require_state(CheckpointTransactionState.FINALIZED)
        publications: dict[str, ComponentCheckpoint] = {}
        try:
            for backend in self._registry:
                if (
                    backend.descriptor.commit_scope is CommitScope.LEADER
                    and self._rank != 0
                ):
                    continue
                component_id = backend.descriptor.component_id
                gathered = gathered_records.get(component_id)
                if gathered is None:
                    raise KeyError(
                        f"gathered checkpoint records missing backend {component_id!r}"
                    )
                save_session = self._session(component_id)
                publications[component_id] = await backend.component.commit_save(
                    CommitSaveSession(
                        instance=save_session.instance,
                        attempt_id=save_session.attempt_id,
                        checkpoint_id=save_session.checkpoint_id,
                        commit_permit=save_session.commit_permit,
                        cut=save_session.cut,
                        spmd_id=save_session.spmd_id,
                        rank=save_session.rank,
                        source_topology=save_session.source_topology,
                        component_id=save_session.component_id,
                        worker_results=gathered,
                    )
                )
        except BaseException as primary_error:
            cleanup_errors = await self._abort_backends(
                tuple(reversed(tuple(self._registry)))
            )
            self._clear_transaction()
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "checkpoint commit_save and rollback failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise
        self._state = CheckpointTransactionState.COMMITTED
        return publications

    async def mark_available(self) -> None:
        self._require_state(CheckpointTransactionState.COMMITTED)
        for backend in self._registry:
            await backend.component.on_checkpoint_available(
                self._session(backend.descriptor.component_id)
            )
        self._clear_transaction()

    async def rollback_save(self) -> None:
        if self._state is None:
            raise RuntimeError("worker has no active checkpoint transaction")
        errors = await self._abort_backends(tuple(reversed(tuple(self._registry))))
        self._clear_transaction()
        self._raise_errors("checkpoint rollback_save failed", errors)

    async def rollback_active(self) -> None:
        if self._state is not None:
            await self.rollback_save()

    async def restore(
        self,
        checkpoint_id: str,
        cut: CheckpointCut,
        publications: Mapping[str, ComponentCheckpoint],
        source_topology: TopologySnapshot,
        current_topology: TopologySnapshot,
    ) -> None:
        self._require_available()
        if self._state is not None:
            raise CheckpointBusyError("cannot restore during an active checkpoint")
        await self._ensure_open()
        for backend in self._registry:
            component_id = backend.descriptor.component_id
            publication = publications.get(component_id)
            if publication is None:
                raise KeyError(f"restore manifest is missing backend {component_id!r}")
            session = RestoreSession(
                instance=self._owner,
                checkpoint_id=checkpoint_id,
                cut=cut,
                spmd_id=self._spmd_id,
                rank=self._rank,
                saved_topology=source_topology,
                restore_topology=current_topology,
                component_id=component_id,
                component_checkpoint=publication,
                format_version=backend.descriptor.format_version,
            )
            await backend.component.restore(session)

    async def close(self) -> None:
        self._require_available()
        if self._state is not None:
            raise CheckpointBusyError(
                "checkpoint coordinator must rollback_save or publish its active transaction"
            )
        if self._opened:
            await self._registry.close_all()
        self._closed = True

    async def _ensure_open(self) -> None:
        if self._opened:
            return
        await self._registry.open_all(
            CheckpointRuntime(
                role=CheckpointRuntimeRole.WORKER,
                repository_root=self._repository_root,
                spmd_id=self._spmd_id,
                rank=self._rank,
            )
        )
        self._opened = True

    async def _abort_backends(
        self, backends: tuple[CheckpointComponentRegistration, ...]
    ) -> list[BaseException]:
        errors: list[BaseException] = []
        for backend in backends:
            try:
                await backend.component.rollback_save(
                    self._session(backend.descriptor.component_id)
                )
            except BaseException as error:
                errors.append(error)
        return errors

    def _session(self, component_id: str) -> SaveSession:
        if (
            self._checkpoint_id is None
            or self._attempt_id is None
            or self._permit is None
            or self._cut is None
        ):
            raise RuntimeError("checkpoint transaction context is not initialized")
        return SaveSession(
            instance=self._owner,
            attempt_id=self._attempt_id,
            checkpoint_id=self._checkpoint_id,
            commit_permit=self._permit,
            cut=self._cut,
            spmd_id=self._spmd_id,
            rank=self._rank,
            source_topology=self._source_topology,
            component_id=component_id,
            format_version=self._registry.require(
                component_id
            ).descriptor.format_version,
        )

    def _require_permit(self) -> CheckpointCommitPermit:
        if self._permit is None:
            raise RuntimeError("checkpoint transaction permit is not initialized")
        return self._permit

    def _require_state(self, state: CheckpointTransactionState) -> None:
        if self._state is not state:
            raise RuntimeError(
                f"checkpoint transaction requires {state.value}, "
                f"got {self._state.value if self._state else None}"
            )

    def _require_available(self) -> None:
        if self._closed:
            raise RuntimeError("checkpoint coordinator is closed")

    def _clear_transaction(self) -> None:
        self._checkpoint_id = None
        self._attempt_id = None
        self._permit = None
        self._cut = None
        self._state = None
        self._captured = {}

    @staticmethod
    def _raise_errors(message: str, errors: list[BaseException]) -> None:
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup(message, errors)
