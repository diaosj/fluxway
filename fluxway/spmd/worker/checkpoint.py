"""Worker-local checkpoint transaction coordinator."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

from fluxway.common.types.checkpoint import (
    CheckpointComponentPublication,
    CheckpointRuntime,
    CheckpointRuntimeRole,
    CheckpointBusyError,
    CheckpointCommitPermit,
    SaveSession,
    CommitSaveSession,
    RestoreSession,
    CheckpointCut,
    CheckpointSaveState,
    CheckpointSaveScope,
    WorkerSaveResult,
    TopologySnapshot,
)
from fluxway.control_plane.checkpointer.component import (
    CheckpointComponentRegistration,
    CheckpointComponentRegistry,
)


_PhaseResult = TypeVar("_PhaseResult")


class WorkerCheckpointCoordinator:
    def __init__(
        self,
        components: tuple[CheckpointComponentRegistration, ...],
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
        self._registry = CheckpointComponentRegistry(components)
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
        self._state: CheckpointSaveState | None = None
        self._entered_save_components: list[CheckpointComponentRegistration] = []

    @property
    def state(self) -> CheckpointSaveState | None:
        return self._state

    @property
    def active_attempt_id(self) -> str | None:
        return self._attempt_id

    async def prepare_save(
        self,
        checkpoint_id: str,
        attempt_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
    ) -> None:
        self._require_available()
        if self._state is not None:
            raise CheckpointBusyError(
                f"worker already owns checkpoint attempt {self.active_attempt_id!r}"
            )
        await self._ensure_open()
        self._checkpoint_id = checkpoint_id
        self._attempt_id = attempt_id
        self._permit = permit
        self._cut = cut
        self._state = CheckpointSaveState.FINALIZING

        async def operation() -> None:
            for component in self._registry:
                finalizer = component.finalize
                if finalizer is None:
                    continue
                await finalizer.finalize_save(
                    self._session(component.descriptor.component_id)
                )
            for component in self._registry:
                save = component.save
                if save is None:
                    continue
                if (
                    component.descriptor.save.prepare_scope
                    is CheckpointSaveScope.LEADER
                    and self._rank != 0
                ):
                    continue
                self._entered_save_components.append(component)
                await save.prepare_save(
                    self._session(component.descriptor.component_id)
                )
            self._state = CheckpointSaveState.PREPARED

        await self._run_phase(operation)

    async def save(self) -> Mapping[str, WorkerSaveResult]:
        self._require_state(CheckpointSaveState.PREPARED)
        self._state = CheckpointSaveState.SAVING

        async def operation() -> Mapping[str, WorkerSaveResult]:
            saved_results: dict[str, WorkerSaveResult] = {}
            for component in self._registry:
                save = component.save
                if save is None:
                    continue
                saved_results[component.descriptor.component_id] = await save.save(
                    self._session(component.descriptor.component_id)
                )
            self._state = CheckpointSaveState.SAVED
            return dict(saved_results)

        return await self._run_phase(operation)

    async def finish_save(self) -> None:
        self._require_state(CheckpointSaveState.SAVED)

        async def operation() -> None:
            for component in self._registry:
                save = component.save
                if save is None:
                    continue
                await save.finish_save(self._session(component.descriptor.component_id))
            self._state = CheckpointSaveState.FINISHED

        await self._run_phase(operation)

    async def commit_save(
        self,
        gathered_records: Mapping[str, Mapping[int, WorkerSaveResult]],
    ) -> Mapping[str, CheckpointComponentPublication]:
        self._require_state(CheckpointSaveState.FINISHED)

        async def operation() -> Mapping[str, CheckpointComponentPublication]:
            publications: dict[str, CheckpointComponentPublication] = {}
            for component in self._registry:
                save = component.save
                if save is None:
                    continue
                if (
                    component.descriptor.save.commit_scope is CheckpointSaveScope.LEADER
                    and self._rank != 0
                ):
                    continue
                component_id = component.descriptor.component_id
                gathered = gathered_records.get(component_id)
                if gathered is None:
                    raise KeyError(
                        f"gathered checkpoint records missing component {component_id!r}"
                    )
                save_session = self._session(component_id)
                publications[component_id] = await save.commit_save(
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
            self._state = CheckpointSaveState.COMMITTED
            return publications

        return await self._run_phase(operation)

    async def publish_save(self) -> None:
        self._require_state(CheckpointSaveState.COMMITTED)
        for component in self._registry:
            save = component.save
            if save is None:
                continue
            await save.publish_save(self._session(component.descriptor.component_id))
        self._clear_transaction()

    async def rollback_save(self) -> None:
        if self._state is None:
            raise RuntimeError("worker has no active checkpoint transaction")
        errors = await self._rollback_components(
            tuple(reversed(self._entered_save_components))
        )
        self._clear_transaction()
        self._raise_errors("checkpoint rollback_save failed", errors)

    async def rollback_active(self) -> None:
        if self._state is not None:
            await self.rollback_save()

    async def restore(
        self,
        checkpoint_id: str,
        cut: CheckpointCut,
        publications: Mapping[str, CheckpointComponentPublication],
        source_topology: TopologySnapshot,
        current_topology: TopologySnapshot,
    ) -> None:
        self._require_available()
        if self._state is not None:
            raise CheckpointBusyError("cannot restore during an active checkpoint")
        await self._ensure_open()
        for component_id, publication in publications.items():
            component = self._registry.require(component_id)
            restore = component.restore
            if restore is None:
                raise RuntimeError(
                    f"checkpoint component does not expose restore capability: {component_id!r}"
                )
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
                format_version=component.descriptor.format_version,
            )
            await restore.restore(session)

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

    async def _rollback_components(
        self, components: tuple[CheckpointComponentRegistration, ...]
    ) -> list[BaseException]:
        errors: list[BaseException] = []
        for component in components:
            save = component.save
            if save is None:
                continue
            try:
                await save.rollback_save(
                    self._session(component.descriptor.component_id)
                )
            except BaseException as error:
                errors.append(error)
        return errors

    async def _run_phase(
        self, operation: Callable[[], Awaitable[_PhaseResult]]
    ) -> _PhaseResult:
        try:
            return await operation()
        except BaseException as primary_error:
            cleanup_errors = await self._rollback_components(
                tuple(reversed(self._entered_save_components))
            )
            self._clear_transaction()
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "checkpoint phase and rollback failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise

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

    def _require_state(self, state: CheckpointSaveState) -> None:
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
        self._entered_save_components.clear()

    @staticmethod
    def _raise_errors(message: str, errors: list[BaseException]) -> None:
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup(message, errors)
